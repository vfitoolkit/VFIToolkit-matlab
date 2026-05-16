function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_DC1_e_raw(V,n_d1,n_d2,n_a1,n_a2,n_z,n_e,N_j, d_gridvals,d2_gridvals,a1_gridvals,a2_grid, z_gridvals_J,e_gridvals_J, pi_z_J,pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z,e), rather than standard (a,z,e,j)
% V is (a,j)-by-z-by-e
% pi_z_J is (j,z',z) for fastOLG
% z_gridvals_J is (j,N_z,l_z) for fastOLG
% pi_e_J is (j,e') for fastOLG
% e_gridvals_J is (j,N_e,l_e) for fastOLG

N_d1=prod(n_d1);
N_d2=prod(n_d2);
% N_d=N_d1*N_d2;
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);
N_e=prod(n_e);

z_gridvals_J=shiftdim(z_gridvals_J,-4); % [1,1,1,1,N_j,N_z,l_z]
e_gridvals_J=shiftdim(e_gridvals_J,-5); % [1,1,1,1,1,N_j,N_e,l_e]

%%
% n-Monotonicity
% vfoptions.level1n=11;
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

d2ind=repmat(gpuArray(1:1:N_d2)',N_d1,1);
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2);
jind=shiftdim(gpuArray(0:1:N_j-1),-3);
zind=shiftdim(gpuArray(0:1:N_z-1),-4);
eind=shiftdim(gpuArray(0:1:N_e-1),-5);

%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_z (note: N_aprime is just equal to N_a)

DiscountFactorParamsVec=CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec,2);
DiscountFactorParamsVec=shiftdim(DiscountFactorParamsVec,-2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

if vfoptions.EVpre==0
    aprimeFnParamsVec=CreateAgeMatrixFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_J(aprimeFn, n_d2, n_a2, N_j, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_j], whereas aprimeProbs is [N_d2,N_a2,N_j]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,1,1)+N_a1*repmat((a2primeIndex-1),N_a1,1,1); % [N_d2*N_a1,N_a2,N_j], autofill the [1,N_a1,N_j] dimensions for the first part
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,1,1)+N_a1*repmat(a2primeIndex,N_a1,1,1); % [N_d2*N_a1,N_a2,N_j], autofill the [1,N_a1,N_j] dimensions for the first part
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1,N_z);  % [N_d2*N_a1,N_a2,N_j,N_z]

    EVpre=[sum(V(N_a+1:end,:).*replem(reshape(pi_e_J,[N_j,1,N_e]),N_a-1,1,1),3); zeros(N_a,N_z,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_z_J to create expectations

    % Need to add the indexes for j to the aprimeIndex, remember fastOLG so V is (a,j)-by-z
    Vlower=reshape(EVpre(aprimeIndex+shiftdim(N_a*gpuArray(0:1:N_j-1),-1),:),[N_d2*N_a1,N_a2,N_j,N_z]);
    Vupper=reshape(EVpre(aprimeplus1Index+shiftdim(N_a*gpuArray(0:1:N_j-1),-1),:),[N_d2*N_a1,N_a2,N_j,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,N_j,zprime)
    % Already applied the probabilities from interpolating onto grid

    EV=EV.*shiftdim(pi_z_J,-2);
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=reshape(sum(EV,4),[N_d2*N_a1,N_a2,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a
elseif vfoptions.EVpre==1
    % This is used for 'Matched Expecations Path'
    aprimeFnParamsVec=CreateAgeMatrixFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_J(aprimeFn, n_d2, n_a2, N_j, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_j], whereas aprimeProbs is [N_d2,N_a2,N_j]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,1,1)+N_a1*repmat((a2primeIndex-1),N_a1,1,1); % [N_d2*N_a1,N_a2,N_j], autofill the [1,N_a1,N_j] dimensions for the first part
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,1,1)+N_a1*repmat(a2primeIndex,N_a1,1,1); % [N_d2*N_a1,N_a2,N_j], autofill the [1,N_a1,N_j] dimensions for the first part
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1,N_z);  % [N_d2*N_a1,N_a2,N_j,N_z]

    EVpre=sum(V.*replem(reshape(pi_e_J,[N_j,1,N_e]),N_a,1,1),3);

    % Need to add the indexes for j to the aprimeIndex, remember fastOLG so V is (a,j)-by-z
    Vlower=reshape(EVpre(aprimeIndex+shiftdim(N_a*gpuArray(0:1:N_j-1),-1),:),[N_d2*N_a1,N_a2,N_j,N_z]);
    Vupper=reshape(EVpre(aprimeplus1Index+shiftdim(N_a*gpuArray(0:1:N_j-1),-1),:),[N_d2*N_a1,N_a2,N_j,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,N_j,zprime)
    % Already applied the probabilities from interpolating onto grid

    EV=EV.*shiftdim(pi_z_J,-2);
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=reshape(sum(EV,4),[N_d2*N_a1,N_a2,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a
end

DiscountedEV=DiscountFactorParamsVec.*reshape(EV,[N_d2,N_a1,1,N_a2,N_j,N_z]); % (d2,a1prime,1,a2,j,z)
% Note: does not contain d1

V=zeros(N_a,N_j,N_z,N_e,'gpuArray');
Policy=zeros(N_a,N_j,N_z,N_e,'gpuArray');

if vfoptions.lowmemory==0
    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_ExpAsset_Disc_e(ReturnFn, n_d1, n_d2, n_a1, vfoptions.level1n,n_a2, n_z,n_e,N_j, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_grid, z_gridvals_J,e_gridvals_J, ReturnFnParamsAgeMatrix,1,0); % Level=1, Refine=0

    entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

    % First, we want a1prime conditional on (d,1,a)
    [~,maxindex1]=max(entireRHS_ii,[],2);

    % Now, get and store the full (d,aprime)
    [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a1,vfoptions.level1n*N_a2,N_j,N_z,N_e]),[],1);

    % Store
    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
    V(curraindex,:,:,:)=shiftdim(Vtempii,1);
    Policy(curraindex,:,:,:)=shiftdim(maxindex2,1);

    % Attempt for improved version
    maxgap=squeeze(max(max(max(max(max(maxindex1(:,1,2:end,:,:,:,:)-maxindex1(:,1,1:end-1,:,:,:,:),[],7),[],6),[],5),[],4),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:,:,:,:),n_a1-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are N_d-by-maxgap(ii)+1-by-1-by-N_j-by-N_z-by-N_e
            ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_ExpAsset_Disc_e(ReturnFn, n_d1, n_d2, maxgap(ii)+1, level1iidiff(ii),n_a2, n_z,n_e,N_j, d_gridvals, a1_gridvals(aprimeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J,e_gridvals_J, ReturnFnParamsAgeMatrix,2,0); % Level=2, Refine=0
            d2aprimejz=d2ind+N_d2*(aprimeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*jind+N_d2*N_a1*N_a2*N_j*zind; % with the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+repelem(reshape(DiscountedEV(d2aprimejz),[N_d*(maxgap(ii)+1),N_a2,N_j,N_z,N_e]),1,level1iidiff(ii),1,1);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,:,:)=shiftdim(Vtempii,1);
            % maxindex does not need reworking, as with expasset there is no a2prime
            % the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            dind=(rem(maxindex-1,N_d)+1);
            allind=reshape(dind,[1,1,level1iidiff(ii),N_a2,N_j,N_z,N_e])+N_d*a2ind+N_d*N_a2*jind+N_d*N_a2*N_j*zind+N_d*N_a2*N_j*N_z*eind; % loweredge is n_d-by-1-by-1-by-n_a2-by-N_j-by-n_z-by-n_e
            allind=reshape(allind,[1,level1iidiff(ii)*N_a2,N_j,N_z,N_e]);
            Policy(curraindex,:,:)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
        else
            loweredge=maxindex1(:,1,ii,:,:,:,:);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_fastOLG_ExpAsset_Disc_e(ReturnFn, n_d1, n_d2, 1, level1iidiff(ii),n_a2, n_z,n_e,N_j, d_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J,e_gridvals_J, ReturnFnParamsAgeMatrix,2,0); % Level=2, Refine=0
            d2aprimejz=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*jind+N_d2*N_a1*N_a2*N_j*zind; % with the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+repelem(reshape(DiscountedEV(d2aprimejz),[N_d,N_a2,N_j,N_z,N_e]),1,level1iidiff(ii),1,1);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,:,:)=shiftdim(Vtempii,1);
            % maxindex does not need reworking, as with expasset there is no a2prime
            % the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            dind=(rem(maxindex-1,N_d)+1);
            allind=reshape(dind,[1,1,level1iidiff(ii),N_a2,N_j,N_z,N_e])+N_d*a2ind+N_d*N_a2*jind+N_d*N_a2*N_j*zind+N_d*N_a2*N_j*N_z*eind; % loweredge is n_d-by-1-by-1-by-n_a2-by-N_j-by-n_z-by-n_e
            allind=reshape(allind,[1,level1iidiff(ii)*N_a2,N_j,N_z,N_e]);
            Policy(curraindex,:,:)=shiftdim(maxindex+N_d*loweredge(allind)-1,1); % loweredge
        end
    end


elseif vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e),'gpuArray');

    for e_c=1:N_e
        e_val=e_gridvals_J(:,e_c,:);

        % n-Monotonicity
        ReturnMatrix_ii_z=CreateReturnFnMatrix_fastOLG_ExpAsset_Disc_e(ReturnFn, n_d1, n_d2, n_a1, vfoptions.level1n,n_a2, n_z,special_n_e,N_j, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_grid, z_gridvals_J,e_val, ReturnFnParamsAgeMatrix,1,0); % Level=1, Refine=0

        entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedEV;

        % First, we want a1prime conditional on (d,1,a)
        [~,maxindex1]=max(entireRHS_ii_z,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii_z,[N_d*N_a1,vfoptions.level1n*N_a2,N_j],N_z),[],1);

        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V(curraindex,:,:,e_c)=shiftdim(Vtempii,1);
        Policy(curraindex,:,:,e_c)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:,:),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are N_d-by-maxgap(ii)+1-by-1-by-N_j-by-N_z
                ReturnMatrix_ii_z=CreateReturnFnMatrix_fastOLG_ExpAsset_Disc_e(ReturnFn, n_d1, n_d2, maxgap(ii)+1, level1iidiff(ii),n_a2, n_z,special_n_e,N_j, d_gridvals, a1_gridvals(aprimeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J,e_val, ReturnFnParamsAgeMatrix,2,0); % Level=2, Refine=0
                d2aprimej=d2ind+N_d2*(aprimeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*jind+N_d2*N_a1*N_a2*N_j*zind; % with the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii_z=ReturnMatrix_ii_z+repelem(reshape(DiscountedEV(d2aprimej),[N_d*(maxgap(ii)+1),N_a2,N_j,N_z]),1,level1iidiff(ii),1);
                [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                V(curraindex,:,:,e_c)=shiftdim(Vtempii,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                % the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d)+1);
                allind=reshape(dind,[1,1,level1iidiff(ii),N_a2,N_j,N_z])+N_d*a2ind+N_d*N_a2*jind+N_d*N_a2*N_j*zind; % loweredge is n_d-by-1-by-1-by-n_a2-by-N_j-by-n_z
                allind=reshape(allind,[1,level1iidiff(ii)*N_a2,N_j,N_z]);
                Policy(curraindex,:,:,e_c)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii_z=CreateReturnFnMatrix_fastOLG_ExpAsset_Disc_e(ReturnFn, n_d1, n_d2, 1, level1iidiff(ii),n_a2, n_z,special_n_e,N_j, d_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J,e_val, ReturnFnParamsAgeMatrix,2,0); % Level=2, Refine=0
                d2aprimej=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*jind+N_d2*N_a1*N_a2*N_j*zind; % with the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii_z=ReturnMatrix_ii_z+repelem(reshape(DiscountedEV(d2aprimej),[N_d,N_a2,N_j,N_z]),1,level1iidiff(ii),1);
                [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                V(curraindex,:,:,e_c)=shiftdim(Vtempii,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                % the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d)+1);
                allind=reshape(dind,[1,1,level1iidiff(ii),N_a2,N_j,N_z])+N_d*a2ind+N_d*N_a2*jind+N_d*N_a2*N_j*zind; % loweredge is n_d-by-1-by-1-by-n_a2-by-N_j-by-n_z
                allind=reshape(allind,[1,level1iidiff(ii)*N_a2,N_j,N_z]);
                Policy(curraindex,:,:,e_c)=shiftdim(maxindex+N_d*loweredge(allind)-1,1); % loweredge
            end
        end
    end
elseif vfoptions.lowmemory==2
    special_n_z=ones(1,length(n_z),'gpuArray');
    special_n_e=ones(1,length(n_e),'gpuArray');

    for z_c=1:N_z
        z_val=z_gridvals_J(:,z_c,:);
        DiscountedEV_z=DiscountedEV(:,:,:,z_c);

        for e_c=1:N_e
            e_val=e_gridvals_J(:,e_c,:);

            % n-Monotonicity
            ReturnMatrix_ii_z=CreateReturnFnMatrix_fastOLG_ExpAsset_Disc_e(ReturnFn, n_d1, n_d2, n_a1, vfoptions.level1n,n_a2, special_n_z,special_n_e,N_j, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_grid, z_val,e_val, ReturnFnParamsAgeMatrix,1,0); % Level=1, Refine=0

            entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedEV_z;

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(entireRHS_ii_z,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_z,[N_d*N_a1,vfoptions.level1n*N_a2,N_j]),[],1);

            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V(curraindex,:,z_c,e_c)=shiftdim(Vtempii,1);
            Policy(curraindex,:,z_c,e_c)=shiftdim(maxindex2,1);

            % Attempt for improved version
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),n_a-maxgap(ii)); % maxindex1(:,ii), but avoid going off top of grid when we add maxgap(ii) points
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are N_d-by-maxgap(ii)+1-by-1-by-N_j
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_fastOLG_ExpAsset_Disc_e(ReturnFn, n_d1, n_d2, maxgap(ii)+1, level1iidiff(ii),n_a2, special_n_z,special_n_e,N_j, d_gridvals, a1_gridvals(aprimeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val,e_val, ReturnFnParamsAgeMatrix,2,0); % Level=2, Refine=0
                    d2aprimej=d2ind+N_d2*(aprimeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*jind; % with the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii_z=ReturnMatrix_ii_z+repelem(reshape(DiscountedEV_z(d2aprimej),[N_d*(maxgap(ii)+1),N_a2,N_j]),1,level1iidiff(ii),1);
                    [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                    V(curraindex,:,z_c,e_c)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    % the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d)+1);
                    allind=reshape(dind,[1,1,level1iidiff(ii),N_a2,N_j])+N_d*a2ind+N_d*N_a2*jind; % loweredge is n_d-by-1-by-1-by-n_a2-by-N_j
                    allind=reshape(allind,[1,level1iidiff(ii)*N_a2,N_j]);
                    Policy(curraindex,:,z_c,e_c)=shiftdim(maxindex+N_d*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_fastOLG_ExpAsset_Disc_e(ReturnFn, n_d1, n_d2, 1, level1iidiff(ii),n_a2, special_n_z,special_n_e,N_j, d_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_val,e_val, ReturnFnParamsAgeMatrix,2,0); % Level=2, Refine=0
                    d2aprimej=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*jind; % with the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii_z=ReturnMatrix_ii_z+repelem(reshape(DiscountedEV_z(d2aprimej),[N_d,N_a2,N_j]),1,level1iidiff(ii),1);
                    [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                    V(curraindex,:,z_c,e_c)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    % the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d)+1);
                    allind=reshape(dind,[1,1,level1iidiff(ii),N_a2,N_j])+N_d*a2ind+N_d*N_a2*jind; % loweredge is n_d-by-1-by-1-by-n_a2-by-N_j
                    allind=reshape(allind,[1,level1iidiff(ii)*N_a2,N_j]);
                    Policy(curraindex,:,z_c,e_c)=shiftdim(maxindex+N_d*loweredge(allind)-1,1); % loweredge
                end
            end
        end
    end
end


%% fastOLG with z, so need to output to take certain shapes
V=reshape(V,[N_a*N_j,N_z,N_e]);
% Policy=reshape(Policy,[N_a,N_j,N_z,N_e]);

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point



end
