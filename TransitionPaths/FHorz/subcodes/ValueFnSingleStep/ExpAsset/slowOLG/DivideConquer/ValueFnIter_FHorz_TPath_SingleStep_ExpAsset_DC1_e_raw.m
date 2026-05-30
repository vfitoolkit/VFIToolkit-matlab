function [V,Policy]=ValueFnIter_FHorz_TPath_SingleStep_ExpAsset_DC1_e_raw(V,n_d1,n_d2,n_a1,n_a2,n_z,n_e,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% The V input is next period value fn (across all ages), the V output is this period.

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=N_d1*N_d2;
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);
N_e=prod(n_e);

Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

if vfoptions.lowmemory==0
    % precompute
    eBind=shiftdim((0:1:N_e-1),-2); % already includes -1
    % precompute
    zind=shiftdim((0:1:N_z-1),-3); % already includes -1
    zBind=shiftdim((0:1:N_z-1),-1); % already includes -1
elseif vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
    % precompute
    zind=shiftdim((0:1:N_z-1),-3); % already includes -1
    zBind=shiftdim((0:1:N_z-1),-1); % already includes -1
elseif vfoptions.lowmemory==2
    special_n_e=ones(1,length(n_e));
    special_n_z=ones(1,length(n_z));
end

% n-Monotonicity
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

a2ind=shiftdim((0:1:N_a2-1),-2);
a2Bind=gpuArray(0:1:N_a2-1);
d2ind=repelem((1:1:N_d2)',N_d1,1); % d2 component of each d=(d1,d2), [N_d,1]

%% j=N_j

% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,:,:,N_j);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if vfoptions.lowmemory==0
    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,n_z,n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % Level=1, Refine=0

    % First, we want a1prime conditional on (d,1,a)
    [~,maxindex1]=max(ReturnMatrix_ii,[],2);

    % Now, get and store the full (d,aprime)
    [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_z,N_e]),[],1);

    % Store
    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
    V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
    Policy(curraindex,:,:,N_j)=shiftdim(maxindex2,1);

    % Attempt for improved version
    maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
    for ii=1:(vfoptions.level1n-1)
        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z-by-n_e
            a1primeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_z-by-n_e
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0); % Level=2, Refine=0
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
            % maxindex does not need reworking, as with expasset there is no a2prime
            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            dind=(rem(maxindex-1,N_d1*N_d2)+1);
            allind=dind+N_d1*N_d2*repelem(a2Bind,1,level1iidiff(ii))+N_d1*N_d2*N_a2*zBind+N_d1*N_d2*N_a2*N_z*eBind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_a2-by-n_z-by-n_e
            Policy(curraindex,:,:,N_j)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
        else
            loweredge=maxindex1(:,1,ii,:,:,:);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,1,level1iidiff(ii),n_a2,n_z,n_e, d_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0); % Level=2, Refine=0
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
            % maxindex does not need reworking, as with expasset there is no a2prime
            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            dind=(rem(maxindex-1,N_d1*N_d2)+1);
            allind=dind+N_d1*N_d2*repelem(a2Bind,1,level1iidiff(ii))+N_d1*N_d2*N_a2*zBind+N_d1*N_d2*N_a2*N_z*eBind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_z-by-n_e
            Policy(curraindex,:,:,N_j)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
        end
    end
elseif vfoptions.lowmemory==1
    for e_c=1:N_e
        e_val=e_gridvals_J(e_c,:,N_j);
        % n-Monotonicity
        ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

        % First, we want a1prime conditional on (d,1,a)
        [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_e,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_z]),[],1);

        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
        Policy(curraindex,:,e_c,N_j)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_z-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,special_n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0); % Level=2, Refine=0
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d1*N_d2)+1);
                allind=dind+N_d1*N_d2*repelem(a2Bind,1,level1iidiff(ii))+N_d1*N_d2*N_a2*zBind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_a2-by-n_z
                Policy(curraindex,:,e_c,N_j)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,1,level1iidiff(ii),n_a2,n_z,special_n_e, d_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0); % Level=2, Refine=0
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d1*N_d2)+1);
                allind=dind+N_d1*N_d2*repelem(a2Bind,1,level1iidiff(ii))+N_d1*N_d2*N_a2*zBind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_z
                Policy(curraindex,:,e_c,N_j)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
            end
        end
    end
elseif vfoptions.lowmemory==2
    for z_c=1:N_z
        z_val=z_gridvals_J(z_c,:,N_j);
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);

            % n-Monotonicity
            ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,special_n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(ReturnMatrix_ii_ze,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_ze,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(curraindex,z_c,e_c,N_j)=shiftdim(maxindex2,1);

            % Attempt for improved version
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_z,special_n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0); % Level=2, Refine=0
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d1*N_d2)+1);
                    allind=dind+N_d1*N_d2*repelem(a2Bind,1,level1iidiff(ii)); % loweredge is n_d-by-1-by-1-by-n_a2
                    Policy(curraindex,z_c,e_c,N_j)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,1,level1iidiff(ii),n_a2,special_n_z,special_n_e, d_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0); % Level=2, Refine=0
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d1*N_d2)+1);
                    allind=dind+N_d1*N_d2*repelem(a2Bind,1,level1iidiff(ii)); % loweredge is n_d-by-1-by-1-by-n_a2
                    Policy(curraindex,z_c,e_c,N_j)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                end
            end
        end
    end
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end


    VKronNext_j=Vtemp_j; % Has been presaved before it was replaced
    Vtemp_j=V(:,:,:,jj); % Grab this before it is replaced/updated

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex-1,N_a1,1,1); % [N_d2*N_a1,N_a2]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1,1); % [N_d2*N_a1,N_a2]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,N_z); % [N_d2*N_a1,N_a2,N_z]

    EV=sum(shiftdim(pi_e_J(:,jj),-2).*VKronNext_j,3);

    Vlower=reshape(EV(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_z]);
    Vupper=reshape(EV(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)
    % Already applied the probabilities from interpolating onto grid

    % EV is over (d2,a1prime,a2,zprime)
    EV=EV.*shiftdim(pi_z_J(:,:,jj)',-2);
    EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
    EV=squeeze(sum(EV,3));
    % EV is over (d2,a1prime,a2,z)

    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_d2,N_a1,1,N_a2,N_z]); % (d2,a1prime,1,a2,z); d1-dim is implicit singleton, broadcasts at use sites

    if vfoptions.lowmemory==0

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,n_z,n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0); % Level=1, Refine=0

        entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedEV,N_d1,1,1,1,1);  % autofill e for DiscountedentireEV

        % First, we want a1prime conditional on (d,1,a)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_z,N_e]),[],1);

        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V(curraindex,:,:,jj)=shiftdim(Vtempii,1);
        Policy(curraindex,:,:,jj)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z-by-n_e
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_z-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0); % Level=2, Refine=0
                d2aprimez=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a*zind; % [N_d,maxgap+1,1,N_a2,N_z,N_e]; linear index into DiscountedEV [N_d2,N_a1,1,N_a2,N_z]
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d2aprimez),[N_d*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_z,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,:,jj)=shiftdim(Vtempii,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d1*N_d2)+1);
                allind=dind+N_d1*N_d2*repelem(a2Bind,1,level1iidiff(ii))+N_d1*N_d2*N_a2*zBind+N_d1*N_d2*N_a2*N_z*eBind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_z-by-n_e
                Policy(curraindex,:,:,jj)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,1,level1iidiff(ii),n_a2,n_z,n_e, d_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0); % Level=2, Refine=0
                d2aprimez=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*a2ind+N_d2*N_a*zind; % [N_d,1,1,N_a2,N_z,N_e]; linear index into DiscountedEV [N_d2,N_a1,1,N_a2,N_z]
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d2aprimez),[N_d,level1iidiff(ii)*N_a2,N_z,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,:,jj)=shiftdim(Vtempii,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d1*N_d2)+1);
                allind=dind+N_d1*N_d2*repelem(a2Bind,1,level1iidiff(ii))+N_d1*N_d2*N_a2*zBind+N_d1*N_d2*N_a2*N_z*eBind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_z-by-n_e
                Policy(curraindex,:,:,jj)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
            end
        end

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);

            % n-Monotonicity
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

            entireRHS_ii_e=ReturnMatrix_ii_e+repelem(DiscountedEV,N_d1,1,1,1,1);

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(entireRHS_ii_e,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_e,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_z]),[],1);

            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V(curraindex,:,e_c,jj)=shiftdim(Vtempii,1);
            Policy(curraindex,:,e_c,jj)=shiftdim(maxindex2,1);

            % Attempt for improved version
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_z
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,special_n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0); % Level=2, Refine=0
                    d2aprimez=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a*zind; % [N_d,maxgap+1,1,N_a2,N_z]; linear index into DiscountedEV [N_d2,N_a1,1,N_a2,N_z]
                    entireRHS_ii_e=reshape(ReturnMatrix_ii_e+DiscountedEV(d2aprimez),[N_d*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_z]);
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    V(curraindex,:,e_c,jj)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d1*N_d2)+1);
                    allind=dind+N_d1*N_d2*repelem(a2Bind,1,level1iidiff(ii))+N_d1*N_d2*N_a2*zBind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_z
                    Policy(curraindex,:,e_c,jj)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,1,level1iidiff(ii),n_a2,n_z,special_n_e, d_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0); % Level=2, Refine=0
                    d2aprimez=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*a2ind+N_d2*N_a*zind; % [N_d,1,1,N_a2,N_z]; linear index into DiscountedEV [N_d2,N_a1,1,N_a2,N_z]
                    entireRHS_ii_e=reshape(ReturnMatrix_ii_e+DiscountedEV(d2aprimez),[N_d,level1iidiff(ii)*N_a2,N_z]);
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    V(curraindex,:,e_c,jj)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d1*N_d2)+1);
                    allind=dind+N_d1*N_d2*repelem(a2Bind,1,level1iidiff(ii))+N_d1*N_d2*N_a2*zBind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_z
                    Policy(curraindex,:,e_c,jj)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                end
            end
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_z=DiscountedEV(:,:,:,:,z_c);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                % n-Monotonicity
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,special_n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

                entireRHS_ii_ze=ReturnMatrix_ii_ze+repelem(DiscountedEV_z,N_d1,1,1,1);

                % First, we want a1prime conditional on (d,1,a)
                [~,maxindex1]=max(entireRHS_ii_ze,[],2);

                % Now, get and store the full (d,aprime)
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_ze,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

                % Store
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V(curraindex,z_c,e_c,jj)=shiftdim(Vtempii,1);
                Policy(curraindex,z_c,e_c,jj)=shiftdim(maxindex2,1);

                % Attempt for improved version
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_z,special_n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0); % Level=2, Refine=0
                        d2aprime=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind; % [N_d,maxgap+1,1,N_a2]; linear index into DiscountedEV_z [N_d2,N_a1,1,N_a2]
                        entireRHS_ii_ze=reshape(ReturnMatrix_ii_ze+DiscountedEV_z(d2aprime),[N_d*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                        [Vtempii,maxindex]=max(entireRHS_ii_ze,[],1);
                        V(curraindex,z_c,e_c,jj)=shiftdim(Vtempii,1);
                        % maxindex does not need reworking, as with expasset there is no a2prime
                        %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                        dind=(rem(maxindex-1,N_d1*N_d2)+1);
                        allind=dind+N_d1*N_d2*repelem(a2Bind,1,level1iidiff(ii)); % loweredge is n_d-by-1-by-1-by-n_a2
                        Policy(curraindex,z_c,e_c,jj)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,1,level1iidiff(ii),n_a2,special_n_z,special_n_e, d_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0); % Level=2, Refine=0
                        d2aprime=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*a2ind; % [N_d,1,1,N_a2]; linear index into DiscountedEV_z [N_d2,N_a1,1,N_a2]
                        entireRHS_ii_ze=reshape(ReturnMatrix_ii_ze+DiscountedEV_z(d2aprime),[N_d*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                        [Vtempii,maxindex]=max(entireRHS_ii_ze,[],1);
                        V(curraindex,z_c,e_c,jj)=shiftdim(Vtempii,1);
                        % maxindex does not need reworking, as with expasset there is no a2prime
                        %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                        dind=(rem(maxindex-1,N_d1*N_d2)+1);
                        allind=dind+N_d1*N_d2*repelem(a2Bind,1,level1iidiff(ii)); % loweredge is n_d-by-1-by-1-by-n_a2
                        Policy(curraindex,z_c,e_c,jj)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                    end
                end
            end
        end
    end
end

%%
Policy=shiftdim(Policy,-1);



end