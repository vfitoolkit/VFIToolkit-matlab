function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetN_DC1_noz_e_raw(n_d1, n_d2,n_a1,n_a2,n_e,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid,e_gridvals_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Naive quasi-hyperbolic discounting + ExperienceAsset, Divide-and-Conquer (DC1 over a1prime).
% d2 determines the experience asset a2, a1 is the standard endogenous state.
% Naive: two passes over the same candidate set,
%   Valt/Policyalt maximise  F + beta*EV        (the exponential value)
%   Vtilde/Policy  maximise  F + beta0*beta*EV  (the QH-perceived value)
% beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj).
% The two discount factors generally pick different DC midpoints, so the beta pass uses maxgap_V and
% the beta0*beta pass uses maxgap; the level-1 return matrix is shared, level-2 matrices are *_dc.
% The backward continuation value is Valt (the exponential continuation value).

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=N_d1*N_d2;
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_e=prod(n_e);
d2ind=repelem((1:1:N_d2)',N_d1,1); % [N_d,1]; maps full d-index to d2-component

Valt=zeros(N_a,N_e,N_j,'gpuArray');
Vtilde=zeros(N_a,N_e,N_j,'gpuArray');
Policyalt=zeros(N_a,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z
Policy=zeros(N_a,N_e,N_j,'gpuArray');

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

if vfoptions.lowmemory==0
    % precompute
    eind=shiftdim((0:1:N_e-1),-1); % already includes -1
else
    special_n_e=ones(1,length(n_e));
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
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % Level=1, Refine=0

        % First, we want a1prime conditional on (d,1,a)
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_e]),[],1);

        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        Valt(curraindex,:,N_j)=shiftdim(Vtempii,1);
        Policyalt(curraindex,:,N_j)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2-by-n_e
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0); % Level=2, Refine=0
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Valt(curraindex,:,N_j)=shiftdim(Vtempii,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d1*N_d2)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*eind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_e
                Policyalt(curraindex,:,N_j)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2,1,level1iidiff(ii),n_a2,n_e, d_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0); % Level=2, Refine=0
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Valt(curraindex,:,N_j)=shiftdim(Vtempii,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d1*N_d2)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*eind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_e
                Policyalt(curraindex,:,N_j)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
            end

        end

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);

            % n-Monotonicity
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,special_n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_e,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            Valt(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
            Policyalt(curraindex,e_c,N_j)=shiftdim(maxindex2,1);

            % Attempt for improved version
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_a2-by-1-by-n_a2
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_val, ReturnFnParamsVec,2,0); % Level=2, Refine=0
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    Valt(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d1*N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d1*N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                    Policyalt(curraindex,e_c,N_j)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2,1,level1iidiff(ii),n_a2,special_n_e, d_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_val, ReturnFnParamsVec,2,0); % Level=2, Refine=0
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    Valt(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d1*N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d1*N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                    Policyalt(curraindex,e_c,N_j)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                end

            end
        end
    end

    % Terminal period: no continuation, so the QH-perceived objects equal the exponential ones
    Vtilde(:,:,N_j)=Valt(:,:,N_j);
    Policy(:,:,N_j)=Policyalt(:,:,N_j);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1);  % [N_d2*N_a1,N_a2]

    EVpre=sum(pi_e_J(:,N_j+1)'.*reshape(vfoptions.V_Jplus1,[N_a,N_e]),2);    % Expectations over e

    Vlower=reshape(EVpre(aprimeIndex(:)),[N_d2*N_a1,N_a2]);
    Vupper=reshape(EVpre(aprimeplus1Index(:)),[N_d2*N_a1,N_a2]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)
    % Already applied the probabilities from interpolating onto grid

    DiscountedEV_alt=beta*reshape(EV,[N_d2,N_a1,1,N_a2]); % (d2,a1prime,1,a2); d1-dim is implicit singleton, broadcasts at use sites   % exponential
    DiscountedEV_tilde=beta0beta*reshape(EV,[N_d2,N_a1,1,N_a2]);   % QH-perceived

    if vfoptions.lowmemory==0
        % n-Monotonicity
        % Level1 return matrix (shared by both passes)
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0); % Level=1, Refine=0

        %% Valt (beta)

        entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedEV_alt,N_d1,1,1,1); % autofill e for DiscountedentireEV

        % First, we want a1prime conditional on (d,1,a)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2alt]=max(reshape(entireRHS_ii,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_e]),[],1);

        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        Valt(curraindex,:,N_j)=shiftdim(Vtempii,1);
        Policyalt(curraindex,:,N_j)=shiftdim(maxindex2alt,1);

        % Attempt for improved version
        maxgap_V=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap_V(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap_V(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap_V(ii) points
                % loweredge is n_d-by-1-by-1-by-n_a2
                a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                % aprime possibilities are n_d-by-maxgap_V(ii)+1-by-1-by-n_a2-by-n_e
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2,maxgap_V(ii)+1,level1iidiff(ii),n_a2,n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0); % Level=2, Refine=0
                d2aprime=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2); % [N_d,maxgap_V+1,1,N_a2,N_e]; linear index into DiscountedEV_alt [N_d2,N_a1,1,N_a2]
                entireRHS_ii=reshape(ReturnMatrix_ii_dc+DiscountedEV_alt(d2aprime),[N_d*(maxgap_V(ii)+1),level1iidiff(ii)*N_a2,N_e]);
                [Vtempii,maxindexalt]=max(entireRHS_ii,[],1);
                Valt(curraindex,:,N_j)=shiftdim(Vtempii,1);
                % maxindexalt does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allindalt), need to 'add' the loweredge
                dindalt=(rem(maxindexalt-1,N_d1*N_d2)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allindalt=dindalt+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*eind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_e
                Policyalt(curraindex,:,N_j)=shiftdim(maxindexalt+N_d1*N_d2*(loweredge(allindalt)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, n_d1,n_d2,1,level1iidiff(ii),n_a2,n_e, d_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0); % Level=2, Refine=0
                d2aprime=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2); % [N_d,1,1,N_a2,N_e]; linear index into DiscountedEV_alt [N_d2,N_a1,1,N_a2]
                entireRHS_ii=reshape(ReturnMatrix_ii_dc+DiscountedEV_alt(d2aprime),[N_d,level1iidiff(ii)*N_a2,N_e]);
                [Vtempii,maxindexalt]=max(entireRHS_ii,[],1);
                Valt(curraindex,:,N_j)=shiftdim(Vtempii,1);
                % maxindexalt does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allindalt), need to 'add' the loweredge
                dindalt=(rem(maxindexalt-1,N_d1*N_d2)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allindalt=dindalt+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*eind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_e
                Policyalt(curraindex,:,N_j)=shiftdim(maxindexalt+N_d1*N_d2*(loweredge(allindalt)-1),1);
            end
        end

        %% Vtilde (beta0*beta)

        entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedEV_tilde,N_d1,1,1,1); % autofill e for DiscountedentireEV

        % First, we want a1prime conditional on (d,1,a)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_e]),[],1);

        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        Vtilde(curraindex,:,N_j)=shiftdim(Vtempii,1);
        Policy(curraindex,:,N_j)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-1-by-n_a2
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_e
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0); % Level=2, Refine=0
                d2aprime=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2); % [N_d,maxgap+1,1,N_a2,N_e]; linear index into DiscountedEV_tilde [N_d2,N_a1,1,N_a2]
                entireRHS_ii=reshape(ReturnMatrix_ii_dc+DiscountedEV_tilde(d2aprime),[N_d*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vtilde(curraindex,:,N_j)=shiftdim(Vtempii,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d1*N_d2)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*eind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_e
                Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, n_d1,n_d2,1,level1iidiff(ii),n_a2,n_e, d_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0); % Level=2, Refine=0
                d2aprime=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2); % [N_d,1,1,N_a2,N_e]; linear index into DiscountedEV_tilde [N_d2,N_a1,1,N_a2]
                entireRHS_ii=reshape(ReturnMatrix_ii_dc+DiscountedEV_tilde(d2aprime),[N_d,level1iidiff(ii)*N_a2,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vtilde(curraindex,:,N_j)=shiftdim(Vtempii,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d1*N_d2)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*eind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_e
                Policy(curraindex,:,N_j)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
            end
        end
    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);

            % n-Monotonicity
            % Level1 return matrix (shared by both passes)
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,special_n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

            %% Valt (beta)

            entireRHS_ii_e=ReturnMatrix_ii_e+repelem(DiscountedEV_alt,N_d1,1,1,1);

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(entireRHS_ii_e,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2alt]=max(reshape(entireRHS_ii_e,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            Valt(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
            Policyalt(curraindex,e_c,N_j)=shiftdim(maxindex2alt,1);

            % Attempt for improved version
            maxgap_V=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap_V(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap_V(ii) points
                    % loweredge is n_d-by-1-by-1-by-n_a2
                    a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                    % aprime possibilities are n_d-by-maxgap_V(ii)+1-by-1-by-n_a2
                    ReturnMatrix_ii_dc_e=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2,maxgap_V(ii)+1,level1iidiff(ii),n_a2,special_n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_val, ReturnFnParamsVec,3,0); % Level=2, Refine=0
                    d2aprime=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2); % [N_d,maxgap_V+1,1,N_a2]; linear index into DiscountedEV_alt [N_d2,N_a1,1,N_a2]
                    entireRHS_ii_e=reshape(ReturnMatrix_ii_dc_e+DiscountedEV_alt(d2aprime),[N_d*(maxgap_V(ii)+1),level1iidiff(ii)*N_a2]);
                    [Vtempii,maxindexalt]=max(entireRHS_ii_e,[],1);
                    Valt(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    % maxindexalt does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allindalt), need to 'add' the loweredge
                    dindalt=(rem(maxindexalt-1,N_d1*N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allindalt=dindalt+N_d1*N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                    Policyalt(curraindex,e_c,N_j)=shiftdim(maxindexalt+N_d1*N_d2*(loweredge(allindalt)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii_dc_e=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, n_d1,n_d2,1,level1iidiff(ii),n_a2,special_n_e, d_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_val, ReturnFnParamsVec,3,0); % Level=2, Refine=0
                    d2aprime=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2); % [N_d,1,1,N_a2]; linear index into DiscountedEV_alt [N_d2,N_a1,1,N_a2]
                    entireRHS_ii_e=reshape(ReturnMatrix_ii_dc_e+DiscountedEV_alt(d2aprime),[N_d,level1iidiff(ii)*N_a2]);
                    [Vtempii,maxindexalt]=max(entireRHS_ii_e,[],1);
                    Valt(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    % maxindexalt does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allindalt), need to 'add' the loweredge
                    dindalt=(rem(maxindexalt-1,N_d1*N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allindalt=dindalt+N_d1*N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                    Policyalt(curraindex,e_c,N_j)=shiftdim(maxindexalt+N_d1*N_d2*(loweredge(allindalt)-1),1);
                end
            end

            %% Vtilde (beta0*beta)

            entireRHS_ii_e=ReturnMatrix_ii_e+repelem(DiscountedEV_tilde,N_d1,1,1,1);

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(entireRHS_ii_e,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_e,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            Vtilde(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(curraindex,e_c,N_j)=shiftdim(maxindex2,1);

            % Attempt for improved version
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-1-by-n_a2
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2
                    ReturnMatrix_ii_dc_e=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_val, ReturnFnParamsVec,3,0); % Level=2, Refine=0
                    d2aprime=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2); % [N_d,maxgap+1,1,N_a2]; linear index into DiscountedEV_tilde [N_d2,N_a1,1,N_a2]
                    entireRHS_ii_e=reshape(ReturnMatrix_ii_dc_e+DiscountedEV_tilde(d2aprime),[N_d*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    Vtilde(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d1*N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d1*N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                    Policy(curraindex,e_c,N_j)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii_dc_e=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, n_d1,n_d2,1,level1iidiff(ii),n_a2,special_n_e, d_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_val, ReturnFnParamsVec,3,0); % Level=2, Refine=0
                    d2aprime=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2); % [N_d,1,1,N_a2]; linear index into DiscountedEV_tilde [N_d2,N_a1,1,N_a2]
                    entireRHS_ii_e=reshape(ReturnMatrix_ii_dc_e+DiscountedEV_tilde(d2aprime),[N_d,level1iidiff(ii)*N_a2]);
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    Vtilde(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d1*N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d1*N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                    Policy(curraindex,e_c,N_j)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
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
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1);  % [N_d2*N_a1,N_a2]

    EVpre=sum(pi_e_J(:,jj+1)'.*Valt(:,:,jj+1),2);    % Expectations over e

    Vlower=reshape(EVpre(aprimeIndex(:)),[N_d2*N_a1,N_a2]);
    Vupper=reshape(EVpre(aprimeplus1Index(:)),[N_d2*N_a1,N_a2]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)
    % Already applied the probabilities from interpolating onto grid

    DiscountedEV_alt=beta*reshape(EV,[N_d2,N_a1,1,N_a2]); % (d2,a1prime,1,a2); d1-dim is implicit singleton, broadcasts at use sites   % exponential
    DiscountedEV_tilde=beta0beta*reshape(EV,[N_d2,N_a1,1,N_a2]);   % QH-perceived

    if vfoptions.lowmemory==0
        % n-Monotonicity
        % Level1 return matrix (shared by both passes)
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0); % Level=1, Refine=0

        %% Valt (beta)

        entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedEV_alt,N_d1,1,1,1); % autofill e for DiscountedentireEV

        % First, we want a1prime conditional on (d,1,a)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2alt]=max(reshape(entireRHS_ii,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_e]),[],1);

        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        Valt(curraindex,:,jj)=shiftdim(Vtempii,1);
        Policyalt(curraindex,:,jj)=shiftdim(maxindex2alt,1);

        % Attempt for improved version
        maxgap_V=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap_V(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap_V(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap_V(ii) points
                % loweredge is n_d-by-1-by-1-by-n_a2-by-n_e
                a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                % aprime possibilities are n_d-by-maxgap_V(ii)+1-by-1-by-n_a2-by-n_e
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2,maxgap_V(ii)+1,level1iidiff(ii),n_a2,n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0); % Level=2, Refine=0
                d2aprime=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2); % [N_d,maxgap_V+1,1,N_a2,N_e]; linear index into DiscountedEV_alt [N_d2,N_a1,1,N_a2]
                entireRHS_ii=reshape(ReturnMatrix_ii_dc+DiscountedEV_alt(d2aprime),[N_d*(maxgap_V(ii)+1),level1iidiff(ii)*N_a2,N_e]);
                [Vtempii,maxindexalt]=max(entireRHS_ii,[],1);
                Valt(curraindex,:,jj)=shiftdim(Vtempii,1);
                % maxindexalt does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allindalt), need to 'add' the loweredge
                dindalt=(rem(maxindexalt-1,N_d1*N_d2)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allindalt=dindalt+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*eind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_e
                Policyalt(curraindex,:,jj)=shiftdim(maxindexalt+N_d1*N_d2*(loweredge(allindalt)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2,1,level1iidiff(ii),n_a2,n_e, d_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0); % Level=2, Refine=0
                d2aprime=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2); % [N_d,1,1,N_a2,N_e]; linear index into DiscountedEV_alt [N_d2,N_a1,1,N_a2]
                entireRHS_ii=reshape(ReturnMatrix_ii_dc+DiscountedEV_alt(d2aprime),[N_d,level1iidiff(ii)*N_a2,N_e]);
                [Vtempii,maxindexalt]=max(entireRHS_ii,[],1);
                Valt(curraindex,:,jj)=shiftdim(Vtempii,1);
                % maxindexalt does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allindalt), need to 'add' the loweredge
                dindalt=(rem(maxindexalt-1,N_d1*N_d2)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allindalt=dindalt+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*eind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_e
                Policyalt(curraindex,:,jj)=shiftdim(maxindexalt+N_d1*N_d2*(loweredge(allindalt)-1),1);
            end
        end

        %% Vtilde (beta0*beta)

        entireRHS_ii=ReturnMatrix_ii+repelem(DiscountedEV_tilde,N_d1,1,1,1); % autofill e for DiscountedentireEV

        % First, we want a1prime conditional on (d,1,a)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_e]),[],1);

        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        Vtilde(curraindex,:,jj)=shiftdim(Vtempii,1);
        Policy(curraindex,:,jj)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-1-by-n_a2-by-n_e
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2-by-n_e
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0); % Level=2, Refine=0
                d2aprime=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2); % [N_d,maxgap+1,1,N_a2,N_e]; linear index into DiscountedEV_tilde [N_d2,N_a1,1,N_a2]
                entireRHS_ii=reshape(ReturnMatrix_ii_dc+DiscountedEV_tilde(d2aprime),[N_d*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vtilde(curraindex,:,jj)=shiftdim(Vtempii,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d1*N_d2)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*eind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_e
                Policy(curraindex,:,jj)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2,1,level1iidiff(ii),n_a2,n_e, d_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0); % Level=2, Refine=0
                d2aprime=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2); % [N_d,1,1,N_a2,N_e]; linear index into DiscountedEV_tilde [N_d2,N_a1,1,N_a2]
                entireRHS_ii=reshape(ReturnMatrix_ii_dc+DiscountedEV_tilde(d2aprime),[N_d,level1iidiff(ii)*N_a2,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vtilde(curraindex,:,jj)=shiftdim(Vtempii,1);
                % maxindex does not need reworking, as with expasset there is no a2prime
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                dind=(rem(maxindex-1,N_d1*N_d2)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*eind; % loweredge is n_d-by-1-by-1-by-n_a2-by-n_e
                Policy(curraindex,:,jj)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
            end
        end

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);

            % n-Monotonicity
            % Level1 return matrix (shared by both passes)
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2,n_a1,vfoptions.level1n,n_a2,special_n_e, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, e_val, ReturnFnParamsVec,1,0); % Level=1, Refine=0

            %% Valt (beta)

            entireRHS_ii_e=ReturnMatrix_ii_e+repelem(DiscountedEV_alt,N_d1,1,1,1);

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(entireRHS_ii_e,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2alt]=max(reshape(entireRHS_ii_e,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            Valt(curraindex,e_c,jj)=shiftdim(Vtempii,1);
            Policyalt(curraindex,e_c,jj)=shiftdim(maxindex2alt,1);

            % Attempt for improved version
            maxgap_V=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap_V(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap_V(ii) points
                    % loweredge is n_d-by-1-by-1-by-n_a2
                    a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                    % aprime possibilities are n_d-by-maxgap_V(ii)+1-by-1-by-n_a2
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2,maxgap_V(ii)+1,level1iidiff(ii),n_a2,special_n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_val, ReturnFnParamsVec,3,0); % Level=2, Refine=0
                    d2aprime=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2); % [N_d,maxgap_V+1,1,N_a2]; linear index into DiscountedEV_alt [N_d2,N_a1,1,N_a2]
                    entireRHS_ii=reshape(ReturnMatrix_ii_dc+DiscountedEV_alt(d2aprime),[N_d*(maxgap_V(ii)+1),level1iidiff(ii)*N_a2]);
                    [Vtempii,maxindexalt]=max(entireRHS_ii,[],1);
                    Valt(curraindex,e_c,jj)=shiftdim(Vtempii,1);
                    % maxindexalt does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allindalt), need to 'add' the loweredge
                    dindalt=(rem(maxindexalt-1,N_d1*N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allindalt=dindalt+N_d1*N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                    Policyalt(curraindex,e_c,jj)=shiftdim(maxindexalt+N_d1*N_d2*(loweredge(allindalt)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2,1,level1iidiff(ii),n_a2,special_n_e, d_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_val, ReturnFnParamsVec,3,0); % Level=2, Refine=0
                    d2aprime=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2); % [N_d,1,1,N_a2]; linear index into DiscountedEV_alt [N_d2,N_a1,1,N_a2]
                    entireRHS_ii=reshape(ReturnMatrix_ii_dc+DiscountedEV_alt(d2aprime),[N_d,level1iidiff(ii)*N_a2]);
                    [Vtempii,maxindexalt]=max(entireRHS_ii,[],1);
                    Valt(curraindex,e_c,jj)=shiftdim(Vtempii,1);
                    % maxindexalt does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allindalt), need to 'add' the loweredge
                    dindalt=(rem(maxindexalt-1,N_d1*N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allindalt=dindalt+N_d1*N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                    Policyalt(curraindex,e_c,jj)=shiftdim(maxindexalt+N_d1*N_d2*(loweredge(allindalt)-1),1);
                end
            end

            %% Vtilde (beta0*beta)

            entireRHS_ii_e=ReturnMatrix_ii_e+repelem(DiscountedEV_tilde,N_d1,1,1,1);

            % First, we want a1prime conditional on (d,1,a)
            [~,maxindex1]=max(entireRHS_ii_e,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_e,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            Vtilde(curraindex,e_c,jj)=shiftdim(Vtempii,1);
            Policy(curraindex,e_c,jj)=shiftdim(maxindex2,1);

            % Attempt for improved version
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-1-by-n_a2
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_a2
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_e, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_val, ReturnFnParamsVec,3,0); % Level=2, Refine=0
                    d2aprime=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2); % [N_d,maxgap+1,1,N_a2]; linear index into DiscountedEV_tilde [N_d2,N_a1,1,N_a2]
                    entireRHS_ii=reshape(ReturnMatrix_ii_dc+DiscountedEV_tilde(d2aprime),[N_d*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vtilde(curraindex,e_c,jj)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d1*N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d1*N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                    Policy(curraindex,e_c,jj)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2,1,level1iidiff(ii),n_a2,special_n_e, d_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, e_val, ReturnFnParamsVec,3,0); % Level=2, Refine=0
                    d2aprime=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2); % [N_d,1,1,N_a2]; linear index into DiscountedEV_tilde [N_d2,N_a1,1,N_a2]
                    entireRHS_ii=reshape(ReturnMatrix_ii_dc+DiscountedEV_tilde(d2aprime),[N_d,level1iidiff(ii)*N_a2]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    Vtilde(curraindex,e_c,jj)=shiftdim(Vtempii,1);
                    % maxindex does not need reworking, as with expasset there is no a2prime
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    dind=(rem(maxindex-1,N_d1*N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
                    allind=dind+N_d1*N_d2*a2ind; % loweredge is n_d-by-1-by-1-by-n_a2
                    Policy(curraindex,e_c,jj)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                end
            end
        end
    end

end


%%
Policy=shiftdim(Policy,-1);
Policyalt=shiftdim(Policyalt,-1);



end