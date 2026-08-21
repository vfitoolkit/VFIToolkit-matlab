function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetS_DC1_nod1_raw(n_d2,n_a1,n_a2,n_z,N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Sophisticated quasi-hyperbolic discounting + ExperienceAsset, Divide-and-Conquer (DC1 over a1prime).
% d2 determines the experience asset a2, a1 is the standard endogenous state.
% Sophisticated: a single max under beta0*beta,
%   Vhat/Policy come from the  F + beta0*beta*EV  argmax (QH-perceived).
%   Vunderbar is the  F + beta*EV  RHS GATHERED at that same DC argmax (NOT re-maximised).
% The a2 lottery is resolved inside EV before the max (EV is indexed by the choice (d2,a1prime,a2)),
% so the gather returns R(policy)+beta*E[V(policy)] exactly.
% beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj).
% The backward continuation value is Vunderbar.

N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);

Vhat=zeros(N_a,N_z,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

if vfoptions.lowmemory>0
    special_n_z=ones(1,length(n_z));
else
    % precompute
    zind=shiftdim((0:1:N_z-1),-1); % already includes -1
end

% n-Monotonicity
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, n_a1, vfoptions.level1n, n_a2, n_z, d2_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % Level=1, Refine=0

        % First, we want a1prime conditional on (d,1,a)
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d2*N_a1,vfoptions.level1n*N_a2,N_z]),[],1);

        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        Vhat(curraindex,:,N_j)=shiftdim(Vtempii,1);
        Policy(curraindex,:,N_j)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_z
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, maxgap(ii)+1, level1iidiff(ii), n_a2, n_z, d2_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0); % Level=2, Refine=0
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Vhat(curraindex,:,N_j)=shiftdim(Vtempii,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d2)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=dind+N_d2*a2ind+N_d2*N_a2*zind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_a2
                Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, 1,level1iidiff(ii), n_a2, n_z, d2_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0); % Level=2, Refine=0
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Vhat(curraindex,:,N_j)=shiftdim(Vtempii,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d2)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=dind+N_d2*a2ind+N_d2*N_a2*zind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_z
                Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
            end
        end
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            % n-Monotonicity
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, n_a1, vfoptions.level1n, n_a2, special_n_z, d2_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(ReturnMatrix_ii_z,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_z,[N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            Vhat(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
            Policy(curraindex,z_c,N_j)=shiftdim(maxindex2,1);

            % Attempt for improved version
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, maxgap(ii)+1, level1iidiff(ii), n_a2, special_n_z, d2_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,2,0); % Level=2, Refine=0
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    Vhat(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                    Policy(curraindex,z_c,N_j)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, 1, level1iidiff(ii), n_a2, special_n_z, d2_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,2,0); % Level=2, Refine=0
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    Vhat(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                    Policy(curraindex,z_c,N_j)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                end
            end
        end
    end
    % Terminal period: no continuation, so Vunderbar equals Vhat
    Vunderbar(:,:,N_j)=Vhat(:,:,N_j);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_z]);    % First, switch V_Jplus1 into Kron form

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,N_z);  % [N_d2*N_a1,N_a2,N_z]

    Vlower=reshape(EVpre(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_z]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,zprime)
    % Already applied the probabilities from interpolating onto grid

    EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-2);
    EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
    EV=squeeze(sum(EV,3));
    % EV is over (d2,a1prime,a2,z)

    DiscountedEV_hat=beta0beta*reshape(EV,[N_d2,N_a1,1,N_a2,N_z]); % (d2,a1prime,1,a2,zprime)   % QH-perceived
    DiscountedEV_under=beta*reshape(EV,[N_d2,N_a1,1,N_a2,N_z]);   % exponential

    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, n_a1, vfoptions.level1n, n_a2, n_z, d2_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % Level=1, Refine=0

        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_hat;
        entireRHS_ii_under=ReturnMatrix_ii+DiscountedEV_under;

        % First, we want a1prime conditional on (d,1,a)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        entireRHS_ii_flat=reshape(entireRHS_ii,[N_d2*N_a1,vfoptions.level1n*N_a2,N_z]);
        [Vtempii,maxindex2]=max(entireRHS_ii_flat,[],1);

        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        Vhat(curraindex,:,N_j)=shiftdim(Vtempii,1);
        entireRHS_ii_under_flat=reshape(entireRHS_ii_under,[N_d2*N_a1,vfoptions.level1n*N_a2,N_z]);
        maxindexfull=maxindex2+N_d2*N_a1*(0:vfoptions.level1n*N_a2-1)+N_d2*N_a1*vfoptions.level1n*N_a2*shiftdim((0:N_z-1),-1);
        Vtempii_under=entireRHS_ii_under_flat(maxindexfull);
        Vunderbar(curraindex,:,N_j)=shiftdim(Vtempii_under,1);
        Policy(curraindex,:,N_j)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_z
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, maxgap(ii)+1, level1iidiff(ii), n_a2, n_z, d2_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0); % Level=2, Refine=0
                d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_z-1),-3); % [N_d2,maxgap+1,1,N_a2,N_z]; linear index into DiscountedEV_hat [N_d2,N_a1,1,N_a2,N_z]
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV_hat(d2aprimez),[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_z]);
                entireRHS_ii_under=reshape(ReturnMatrix_ii+DiscountedEV_under(d2aprimez),[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_z]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat(curraindex,:,N_j)=shiftdim(Vtempii,1);
                maxindexfull=maxindex+N_d2*(maxgap(ii)+1)*(0:level1iidiff(ii)*N_a2-1)+N_d2*(maxgap(ii)+1)*level1iidiff(ii)*N_a2*shiftdim((0:N_z-1),-1);
                Vtempii_under=entireRHS_ii_under(maxindexfull);
                Vunderbar(curraindex,:,N_j)=shiftdim(Vtempii_under,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d2)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=dind+N_d2*a2ind+N_d2*N_a2*zind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_a2
                Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, 1, level1iidiff(ii), n_a2, n_z, d2_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0); % Level=2, Refine=0
                d2aprimez=(1:1:N_d2)'+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_z-1),-3); % [N_d2,1,1,N_a2,N_z]; linear index into DiscountedEV_hat [N_d2,N_a1,1,N_a2,N_z]
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV_hat(d2aprimez),[N_d2,level1iidiff(ii)*N_a2,N_z]);
                entireRHS_ii_under=reshape(ReturnMatrix_ii+DiscountedEV_under(d2aprimez),[N_d2,level1iidiff(ii)*N_a2,N_z]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat(curraindex,:,N_j)=shiftdim(Vtempii,1);
                maxindexfull=maxindex+N_d2*(0:level1iidiff(ii)*N_a2-1)+N_d2*level1iidiff(ii)*N_a2*shiftdim((0:N_z-1),-1);
                Vtempii_under=entireRHS_ii_under(maxindexfull);
                Vunderbar(curraindex,:,N_j)=shiftdim(Vtempii_under,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d2)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=dind+N_d2*a2ind+N_d2*N_a2*zind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_z
                Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
            end
        end

    elseif vfoptions.lowmemory==1

        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            DiscountedEV_hat_z=DiscountedEV_hat(:,:,:,:,z_c);
            DiscountedEV_under_z=DiscountedEV_under(:,:,:,:,z_c);

            % n-Monotonicity
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, n_a1, vfoptions.level1n, n_a2, special_n_z, d2_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

            entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedEV_hat_z;
            entireRHS_ii_z_under=ReturnMatrix_ii_z+DiscountedEV_under_z;

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(entireRHS_ii_z,[],2);

            % Now, get and store the full (d,aprime)
            entireRHS_ii_z_flat=reshape(entireRHS_ii_z,[N_d2*N_a1,vfoptions.level1n*N_a2]);
            [Vtempii,maxindex2]=max(entireRHS_ii_z_flat,[],1);

            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            Vhat(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
            entireRHS_ii_z_under_flat=reshape(entireRHS_ii_z_under,[N_d2*N_a1,vfoptions.level1n*N_a2]);
            maxindexfull=maxindex2+N_d2*N_a1*(0:vfoptions.level1n*N_a2-1);
            Vtempii_under=entireRHS_ii_z_under_flat(maxindexfull);
            Vunderbar(curraindex,z_c,N_j)=shiftdim(Vtempii_under,1);
            Policy(curraindex,z_c,N_j)=shiftdim(maxindex2,1);

            % Attempt for improved version
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, maxgap(ii)+1, level1iidiff(ii), n_a2, special_n_z, d2_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0); % Level=2, Refine=0
                    d2aprime=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2); % [N_d2,maxgap+1,1,N_a2]; linear index into DiscountedEV_hat_z [N_d2,N_a1,1,N_a2]
                    entireRHS_ii_z=reshape(ReturnMatrix_ii_z+DiscountedEV_hat_z(d2aprime),[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                    entireRHS_ii_z_under=reshape(ReturnMatrix_ii_z+DiscountedEV_under_z(d2aprime),[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                    [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                    Vhat(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    maxindexfull=maxindex+N_d2*(maxgap(ii)+1)*(0:level1iidiff(ii)*N_a2-1);
                    Vtempii_under=entireRHS_ii_z_under(maxindexfull);
                    Vunderbar(curraindex,z_c,N_j)=shiftdim(Vtempii_under,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                    Policy(curraindex,z_c,N_j)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, 1, level1iidiff(ii), n_a2, special_n_z, d2_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0); % Level=2, Refine=0
                    d2aprime=(1:1:N_d2)'+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2); % [N_d2,1,1,N_a2]; linear index into DiscountedEV_hat_z [N_d2,N_a1,1,N_a2]
                    entireRHS_ii_z=reshape(ReturnMatrix_ii_z+DiscountedEV_hat_z(d2aprime),[N_d2,level1iidiff(ii)*N_a2]);
                    entireRHS_ii_z_under=reshape(ReturnMatrix_ii_z+DiscountedEV_under_z(d2aprime),[N_d2,level1iidiff(ii)*N_a2]);
                    [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                    Vhat(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    maxindexfull=maxindex+N_d2*(0:level1iidiff(ii)*N_a2-1);
                    Vtempii_under=entireRHS_ii_z_under(maxindexfull);
                    Vunderbar(curraindex,z_c,N_j)=shiftdim(Vtempii_under,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                    Policy(curraindex,z_c,N_j)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
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


    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,N_z);  % [N_d2*N_a1,N_a2,N_z]

    Vlower=reshape(Vunderbar(aprimeIndex(:),:,jj+1),[N_d2*N_a1,N_a2,N_z]);
    Vupper=reshape(Vunderbar(aprimeplus1Index(:),:,jj+1),[N_d2*N_a1,N_a2,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,zprime)
    % Already applied the probabilities from interpolating onto grid

    % EV is over (d2,a1prime,a2,zprime)
    EV=EV.*shiftdim(pi_z_J(:,:,jj)',-2);
    EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
    EV=squeeze(sum(EV,3));
    % EV is over (d2,a1prime,a2,z)

    DiscountedEV_hat=beta0beta*reshape(EV,[N_d2,N_a1,1,N_a2,N_z]); % (d2,a1prime,1,a2,zprime)   % QH-perceived
    DiscountedEV_under=beta*reshape(EV,[N_d2,N_a1,1,N_a2,N_z]);   % exponential

    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, n_a1, vfoptions.level1n, n_a2, n_z, d2_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0); % Level=1, Refine=0

        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_hat;
        entireRHS_ii_under=ReturnMatrix_ii+DiscountedEV_under;

        % First, we want a1prime conditional on (d,1,a)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        entireRHS_ii_flat=reshape(entireRHS_ii,[N_d2*N_a1,vfoptions.level1n*N_a2,N_z]);
        [Vtempii,maxindex2]=max(entireRHS_ii_flat,[],1);

        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        Vhat(curraindex,:,jj)=shiftdim(Vtempii,1);
        entireRHS_ii_under_flat=reshape(entireRHS_ii_under,[N_d2*N_a1,vfoptions.level1n*N_a2,N_z]);
        maxindexfull=maxindex2+N_d2*N_a1*(0:vfoptions.level1n*N_a2-1)+N_d2*N_a1*vfoptions.level1n*N_a2*shiftdim((0:N_z-1),-1);
        Vtempii_under=entireRHS_ii_under_flat(maxindexfull);
        Vunderbar(curraindex,:,jj)=shiftdim(Vtempii_under,1);
        Policy(curraindex,:,jj)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_z
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_z
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, maxgap(ii)+1, level1iidiff(ii), n_a2, n_z, d2_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0); % Level=2, Refine=0
                d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_z-1),-3); % [N_d2,maxgap+1,1,N_a2,N_z]; linear index into DiscountedEV_hat [N_d2,N_a1,1,N_a2,N_z]
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV_hat(d2aprimez),[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_z]);
                entireRHS_ii_under=reshape(ReturnMatrix_ii+DiscountedEV_under(d2aprimez),[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_z]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat(curraindex,:,jj)=shiftdim(Vtempii,1);
                maxindexfull=maxindex+N_d2*(maxgap(ii)+1)*(0:level1iidiff(ii)*N_a2-1)+N_d2*(maxgap(ii)+1)*level1iidiff(ii)*N_a2*shiftdim((0:N_z-1),-1);
                Vtempii_under=entireRHS_ii_under(maxindexfull);
                Vunderbar(curraindex,:,jj)=shiftdim(Vtempii_under,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d2)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=dind+N_d2*a2ind+N_d2*N_a2*zind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_a2
                Policy(curraindex,:,jj)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, 1, level1iidiff(ii), n_a2, n_z, d2_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0); % Level=2, Refine=0
                d2aprimez=(1:1:N_d2)'+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_z-1),-3); % [N_d2,1,1,N_a2,N_z]; linear index into DiscountedEV_hat [N_d2,N_a1,1,N_a2,N_z]
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV_hat(d2aprimez),[N_d2,level1iidiff(ii)*N_a2,N_z]);
                entireRHS_ii_under=reshape(ReturnMatrix_ii+DiscountedEV_under(d2aprimez),[N_d2,level1iidiff(ii)*N_a2,N_z]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vhat(curraindex,:,jj)=shiftdim(Vtempii,1);
                maxindexfull=maxindex+N_d2*(0:level1iidiff(ii)*N_a2-1)+N_d2*level1iidiff(ii)*N_a2*shiftdim((0:N_z-1),-1);
                Vtempii_under=entireRHS_ii_under(maxindexfull);
                Vunderbar(curraindex,:,jj)=shiftdim(Vtempii_under,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d2)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=dind+N_d2*a2ind+N_d2*N_a2*zind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_z
                Policy(curraindex,:,jj)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
            end
        end

    elseif vfoptions.lowmemory==1

        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_hat_z=DiscountedEV_hat(:,:,:,:,z_c);
            DiscountedEV_under_z=DiscountedEV_under(:,:,:,:,z_c);

            % n-Monotonicity
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, n_a1, vfoptions.level1n, n_a2, special_n_z, d2_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

            entireRHS_ii_z=ReturnMatrix_ii_z+DiscountedEV_hat_z;
            entireRHS_ii_z_under=ReturnMatrix_ii_z+DiscountedEV_under_z;

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(entireRHS_ii_z,[],2);

            % Now, get and store the full (d,aprime)
            entireRHS_ii_z_flat=reshape(entireRHS_ii_z,[N_d2*N_a1,vfoptions.level1n*N_a2]);
            [Vtempii,maxindex2]=max(entireRHS_ii_z_flat,[],1);

            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            Vhat(curraindex,z_c,jj)=shiftdim(Vtempii,1);
            entireRHS_ii_z_under_flat=reshape(entireRHS_ii_z_under,[N_d2*N_a1,vfoptions.level1n*N_a2]);
            maxindexfull=maxindex2+N_d2*N_a1*(0:vfoptions.level1n*N_a2-1);
            Vtempii_under=entireRHS_ii_z_under_flat(maxindexfull);
            Vunderbar(curraindex,z_c,jj)=shiftdim(Vtempii_under,1);
            Policy(curraindex,z_c,jj)=shiftdim(maxindex2,1);

            % Attempt for improved version
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, maxgap(ii)+1, level1iidiff(ii), n_a2, special_n_z, d2_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0); % Level=2, Refine=0
                    d2aprime=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2); % [N_d2,maxgap+1,1,N_a2]; linear index into DiscountedEV_hat_z [N_d2,N_a1,1,N_a2]
                    entireRHS_ii_z=reshape(ReturnMatrix_ii_z+DiscountedEV_hat_z(d2aprime),[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                    entireRHS_ii_z_under=reshape(ReturnMatrix_ii_z+DiscountedEV_under_z(d2aprime),[N_d2*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                    [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                    Vhat(curraindex,z_c,jj)=shiftdim(Vtempii,1);
                    maxindexfull=maxindex+N_d2*(maxgap(ii)+1)*(0:level1iidiff(ii)*N_a2-1);
                    Vtempii_under=entireRHS_ii_z_under(maxindexfull);
                    Vunderbar(curraindex,z_c,jj)=shiftdim(Vtempii_under,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                    Policy(curraindex,z_c,jj)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0, n_d2, 1, level1iidiff(ii), n_a2, special_n_z, d2_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0); % Level=2, Refine=0
                    d2aprime=(1:1:N_d2)'+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2); % [N_d2,1,1,N_a2]; linear index into DiscountedEV_hat_z [N_d2,N_a1,1,N_a2]
                    entireRHS_ii_z=reshape(ReturnMatrix_ii_z+DiscountedEV_hat_z(d2aprime),[N_d2,level1iidiff(ii)*N_a2]);
                    entireRHS_ii_z_under=reshape(ReturnMatrix_ii_z+DiscountedEV_under_z(d2aprime),[N_d2,level1iidiff(ii)*N_a2]);
                    [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                    Vhat(curraindex,z_c,jj)=shiftdim(Vtempii,1);
                    maxindexfull=maxindex+N_d2*(0:level1iidiff(ii)*N_a2-1);
                    Vtempii_under=entireRHS_ii_z_under(maxindexfull);
                    Vunderbar(curraindex,z_c,jj)=shiftdim(Vtempii_under,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                    Policy(curraindex,z_c,jj)=shiftdim(maxindex+N_d2*(loweredge(allind)-1),1);
                end
            end
        end
    end
end


%%
Policy=shiftdim(Policy,-1);


end