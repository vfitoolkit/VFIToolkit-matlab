function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteN_DC2A_nod1_noz_e_raw(n_d2, n_a1, n_a2, n_a3, n_e, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, e_gridvals_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Naive quasi-hyperbolic discounting + ExperienceAssete, nod1 DC2A pattern.
% Naive: two passes over the same candidate set,
%   Valt/Policyalt maximise  F + beta*EV        (the exponential value)
%   Vtilde/Policy  maximise  F + beta0*beta*EV  (the QH-perceived value)
% beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj).
% The two discount factors generally pick different DC midpoints, so the beta pass uses maxgap_V and
% the beta0*beta pass uses maxgap; the level-1 return matrix is shared, level-2 matrices are *_dc.
% Backward EVpre uses Valt (the exponential continuation value).
% Internal naming: a1 is the DC'd standard state, a2 the folded standard states, a3 the experience asset.
% Reuses CreateReturnFnMatrix_ExpAsset_Disc_DC2A passing n_e in the n_z slot.
% lowmemory=0 full vectorization; lowmemory=1 loop over e.

N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_e=prod(n_e);

Valt=zeros(N_a,N_e,N_j,'gpuArray'); % exponential-discounter value fn (beta)
Vtilde=zeros(N_a,N_e,N_j,'gpuArray'); % QH-perceived value fn (beta0*beta)
Policy=zeros(3,N_a,N_e,N_j,'gpuArray');
Policyalt=zeros(3,N_a,N_e,N_j,'gpuArray'); % exponential-discounter policy (Valt is computed at this)


if vfoptions.lowmemory==0
    eind=shiftdim((0:1:N_e-1),-1);
else
    special_n_e=ones(1,length(n_e));
end

level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

a2ind=gpuArray(0:N_a2-1)';
a3ind=gpuArray(0:N_a3-1)';

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_e]),[],1);

        curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                 +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
        d2ind  =rem(maxindex2-1,N_d2)+1;
        a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
        a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
        Valt(curraindex,:,N_j)       =shiftdim(Vtempii,1);
        Policy(1,curraindex,:,N_j)=d2ind;
        Policy(2,curraindex,:,N_j)=a1pind;
        Policy(3,curraindex,:,N_j)=a2pind;

        maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Valt(curraindex,:,N_j)=shiftdim(Vtempii,1);
                d2ind  =rem(maxindex-1,N_d2)+1;
                a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*eind;
                a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                Policy(1,curraindex,:,N_j)=d2ind;
                Policy(2,curraindex,:,N_j)=a1prime_rec;
                Policy(3,curraindex,:,N_j)=a2pind;
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Valt(curraindex,:,N_j)=shiftdim(Vtempii,1);
                d2ind  =rem(maxindex-1,N_d2)+1;
                a2pind =floor((maxindex-1)/N_d2)+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*eind;
                Policy(1,curraindex,:,N_j)=d2ind;
                Policy(2,curraindex,:,N_j)=loweredge(loweredge_idx);
                Policy(3,curraindex,:,N_j)=a2pind;
            end
        end

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 1);
            [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_e,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            d2ind  =rem(maxindex2-1,N_d2)+1;
            a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
            a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
            Valt(curraindex,e_c,N_j)       =shiftdim(Vtempii,1);
            Policy(1,curraindex,e_c,N_j)=d2ind;
            Policy(2,curraindex,e_c,N_j)=a1pind;
            Policy(3,curraindex,e_c,N_j)=a2pind;

            maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
                    Valt(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    d2ind  =rem(maxindex-1,N_d2)+1;
                    a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                    a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policy(1,curraindex,e_c,N_j)=d2ind;
                    Policy(2,curraindex,e_c,N_j)=a1prime_rec;
                    Policy(3,curraindex,e_c,N_j)=a2pind;
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_e,[],1);
                    Valt(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    d2ind  =rem(maxindex-1,N_d2)+1;
                    a2pind =floor((maxindex-1)/N_d2)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                    Policy(1,curraindex,e_c,N_j)=d2ind;
                    Policy(2,curraindex,e_c,N_j)=loweredge(loweredge_idx);
                    Policy(3,curraindex,e_c,N_j)=a2pind;
                end
            end
        end
    end

    Vtilde=Valt;
    Policyalt(:,:,:,N_j)=Policy(:,:,:,N_j); % terminal: QH and exponential discounter coincide
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EVpre=sum(pi_e_J(:,N_j+1)'.*reshape(vfoptions.V_Jplus1,[N_a,N_e]),2); % integrate out eprime: [N_a,1]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a3, n_e, d2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);
    % a3primeIndex/a3primeProbs are [N_d2,N_a3,N_e] (N_e here is the current e)

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem(a2ind,N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1); % [N_d2*N_a1*N_a2,N_a3,N_e]
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1); % [N_d2*N_a1*N_a2,N_a3,N_e] (e dim already present)

    Vlower=reshape(EVpre(aprimeIndex(:)),    [N_d2*N_a1*N_a2,N_a3,N_e]);
    Vupper=reshape(EVpre(aprimeplus1Index(:)),[N_d2*N_a1*N_a2,N_a3,N_e]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;

    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2*a1prime*a2prime,a3,e_cur)

    EVbase=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_e]);
    DiscountedEV_alt  =beta*EVbase;      % exponential (long-run) continuation
    DiscountedEV_tilde=beta0beta*EVbase; % QH-perceived continuation

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        %% Valt (beta)
        entireRHS_ii_V=ReturnMatrix_ii+DiscountedEV_alt;
        [~,maxindex1_V]=max(entireRHS_ii_V,[],2);
        [Vtempii_V,maxindex2_V]=max(reshape(entireRHS_ii_V,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_e]),[],1);
        curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                 +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
        d2ind_V  =rem(maxindex2_V-1,N_d2)+1;
        a1pind_V =rem(floor((maxindex2_V-1)/N_d2),N_a1)+1;
        a2pind_V =floor((maxindex2_V-1)/(N_d2*N_a1))+1;
        Valt(curraindex,:,N_j)       =shiftdim(Vtempii_V,1);
        Policyalt(1,curraindex,:,N_j)=d2ind_V;
        Policyalt(2,curraindex,:,N_j)=a1pind_V;
        Policyalt(3,curraindex,:,N_j)=a2pind_V;

        maxgap_V=squeeze(max(max(max(max(max( maxindex1_V(:,1,:,2:end,:,:,:)-maxindex1_V(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
            if maxgap_V(ii)>0
                loweredge=min(maxindex1_V(:,1,:,ii,:,:,:),N_a1-maxgap_V(ii));
                a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                d2aprimee=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_e-1),-5);
                entireRHS_ii_V=reshape(ReturnMatrix_ii_dc+DiscountedEV_alt(d2aprimee),[N_d2*(maxgap_V(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_e]);
                [Vtempii_V,maxindex_V]=max(entireRHS_ii_V,[],1);
                Valt(curraindex,:,N_j)=shiftdim(Vtempii_V,1);
                d2ind_V  =rem(maxindex_V-1,N_d2)+1;
                a1localind_V=rem(floor((maxindex_V-1)/N_d2),maxgap_V(ii)+1)+1;
                a2pind_V =floor((maxindex_V-1)/(N_d2*(maxgap_V(ii)+1)))+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind_V + N_d2*(a2pind_V-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*eind;
                a1prime_rec=a1localind_V+loweredge(loweredge_idx)-1;
                Policyalt(1,curraindex,:,N_j)=d2ind_V;
                Policyalt(2,curraindex,:,N_j)=a1prime_rec;
                Policyalt(3,curraindex,:,N_j)=a2pind_V;
            else
                loweredge=maxindex1_V(:,1,:,ii,:,:,:);
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                d2aprimee=(1:1:N_d2)' + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_e-1),-5);
                entireRHS_ii_V=reshape(ReturnMatrix_ii_dc+DiscountedEV_alt(d2aprimee),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_e]);
                [Vtempii_V,maxindex_V]=max(entireRHS_ii_V,[],1);
                Valt(curraindex,:,N_j)=shiftdim(Vtempii_V,1);
                d2ind_V  =rem(maxindex_V-1,N_d2)+1;
                a2pind_V =floor((maxindex_V-1)/N_d2)+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind_V + N_d2*(a2pind_V-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*eind;
                Policyalt(1,curraindex,:,N_j)=d2ind_V;
                Policyalt(2,curraindex,:,N_j)=loweredge(loweredge_idx);
                Policyalt(3,curraindex,:,N_j)=a2pind_V;
            end
        end

        %% Vtilde (beta0*beta)
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_tilde;
        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_e]),[],1);
        curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                 +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
        d2ind  =rem(maxindex2-1,N_d2)+1;
        a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
        a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
        Vtilde(curraindex,:,N_j)       =shiftdim(Vtempii,1);
        Policy(1,curraindex,:,N_j)=d2ind;
        Policy(2,curraindex,:,N_j)=a1pind;
        Policy(3,curraindex,:,N_j)=a2pind;

        maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                d2aprimee=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_e-1),-5);
                entireRHS_ii=reshape(ReturnMatrix_ii_dc+DiscountedEV_tilde(d2aprimee),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vtilde(curraindex,:,N_j)=shiftdim(Vtempii,1);
                d2ind  =rem(maxindex-1,N_d2)+1;
                a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*eind;
                a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                Policy(1,curraindex,:,N_j)=d2ind;
                Policy(2,curraindex,:,N_j)=a1prime_rec;
                Policy(3,curraindex,:,N_j)=a2pind;
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                d2aprimee=(1:1:N_d2)' + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_e-1),-5);
                entireRHS_ii=reshape(ReturnMatrix_ii_dc+DiscountedEV_tilde(d2aprimee),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vtilde(curraindex,:,N_j)=shiftdim(Vtempii,1);
                d2ind  =rem(maxindex-1,N_d2)+1;
                a2pind =floor((maxindex-1)/N_d2)+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*eind;
                Policy(1,curraindex,:,N_j)=d2ind;
                Policy(2,curraindex,:,N_j)=loweredge(loweredge_idx);
                Policy(3,curraindex,:,N_j)=a2pind;
            end
        end

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            DiscountedEV_alt_e  =DiscountedEV_alt(:,:,:,:,:,:,e_c);
            DiscountedEV_tilde_e=DiscountedEV_tilde(:,:,:,:,:,:,e_c);
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 1);
            %% Valt (beta)
            entireRHS_ii_e_V=ReturnMatrix_ii_e+DiscountedEV_alt_e;
            [~,maxindex1_V]=max(entireRHS_ii_e_V,[],2);
            [Vtempii_V,maxindex2_V]=max(reshape(entireRHS_ii_e_V,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            d2ind_V  =rem(maxindex2_V-1,N_d2)+1;
            a1pind_V =rem(floor((maxindex2_V-1)/N_d2),N_a1)+1;
            a2pind_V =floor((maxindex2_V-1)/(N_d2*N_a1))+1;
            Valt(curraindex,e_c,N_j)       =shiftdim(Vtempii_V,1);
            Policyalt(1,curraindex,e_c,N_j)=d2ind_V;
            Policyalt(2,curraindex,e_c,N_j)=a1pind_V;
            Policyalt(3,curraindex,e_c,N_j)=a2pind_V;

            maxgap_V=squeeze(max(max(max(max( maxindex1_V(:,1,:,2:end,:,:)-maxindex1_V(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1_V(:,1,:,ii,:,:),N_a1-maxgap_V(ii));
                    a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                    ReturnMatrix_ii_e_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 3);
                    d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_e_V=reshape(ReturnMatrix_ii_e_dc+DiscountedEV_alt_e(d2aprime),[N_d2*(maxgap_V(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                    [Vtempii_V,maxindex_V]=max(entireRHS_ii_e_V,[],1);
                    Valt(curraindex,e_c,N_j)=shiftdim(Vtempii_V,1);
                    d2ind_V  =rem(maxindex_V-1,N_d2)+1;
                    a1localind_V=rem(floor((maxindex_V-1)/N_d2),maxgap_V(ii)+1)+1;
                    a2pind_V =floor((maxindex_V-1)/(N_d2*(maxgap_V(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=d2ind_V + N_d2*(a2pind_V-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                    a1prime_rec=a1localind_V+loweredge(loweredge_idx)-1;
                    Policyalt(1,curraindex,e_c,N_j)=d2ind_V;
                    Policyalt(2,curraindex,e_c,N_j)=a1prime_rec;
                    Policyalt(3,curraindex,e_c,N_j)=a2pind_V;
                else
                    loweredge=maxindex1_V(:,1,:,ii,:,:);
                    ReturnMatrix_ii_e_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 3);
                    d2aprime=(1:1:N_d2)' + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_e_V=reshape(ReturnMatrix_ii_e_dc+DiscountedEV_alt_e(d2aprime),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                    [Vtempii_V,maxindex_V]=max(entireRHS_ii_e_V,[],1);
                    Valt(curraindex,e_c,N_j)=shiftdim(Vtempii_V,1);
                    d2ind_V  =rem(maxindex_V-1,N_d2)+1;
                    a2pind_V =floor((maxindex_V-1)/N_d2)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=d2ind_V + N_d2*(a2pind_V-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                    Policyalt(1,curraindex,e_c,N_j)=d2ind_V;
                    Policyalt(2,curraindex,e_c,N_j)=loweredge(loweredge_idx);
                    Policyalt(3,curraindex,e_c,N_j)=a2pind_V;
                end
            end

            %% Vtilde (beta0*beta)
            entireRHS_ii_e=ReturnMatrix_ii_e+DiscountedEV_tilde_e;
            [~,maxindex1]=max(entireRHS_ii_e,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_e,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            d2ind  =rem(maxindex2-1,N_d2)+1;
            a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
            a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
            Vtilde(curraindex,e_c,N_j)       =shiftdim(Vtempii,1);
            Policy(1,curraindex,e_c,N_j)=d2ind;
            Policy(2,curraindex,e_c,N_j)=a1pind;
            Policy(3,curraindex,e_c,N_j)=a2pind;

            maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_e_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 3);
                    d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_e=reshape(ReturnMatrix_ii_e_dc+DiscountedEV_tilde_e(d2aprime),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    Vtilde(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    d2ind  =rem(maxindex-1,N_d2)+1;
                    a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                    a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policy(1,curraindex,e_c,N_j)=d2ind;
                    Policy(2,curraindex,e_c,N_j)=a1prime_rec;
                    Policy(3,curraindex,e_c,N_j)=a2pind;
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    ReturnMatrix_ii_e_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 3);
                    d2aprime=(1:1:N_d2)' + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_e=reshape(ReturnMatrix_ii_e_dc+DiscountedEV_tilde_e(d2aprime),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    Vtilde(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
                    d2ind  =rem(maxindex-1,N_d2)+1;
                    a2pind =floor((maxindex-1)/N_d2)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                    Policy(1,curraindex,e_c,N_j)=d2ind;
                    Policy(2,curraindex,e_c,N_j)=loweredge(loweredge_idx);
                    Policy(3,curraindex,e_c,N_j)=a2pind;
                end
            end
        end
    end
end


%% Iterate backwards through j
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    EVpre=sum(pi_e_J(:,jj+1)'.*reshape(Valt(:,:,jj+1),[N_a,N_e]),2); % integrate out eprime: [N_a,1]

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a3, n_e, d2_gridvals, a3_grid, e_gridvals_J(:,:,jj), aprimeFnParamsVec,2);
    % a3primeIndex/a3primeProbs are [N_d2,N_a3,N_e] (N_e here is the current e)

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem(a2ind,N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1); % [N_d2*N_a1*N_a2,N_a3,N_e]
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1); % [N_d2*N_a1*N_a2,N_a3,N_e] (e dim already present)

    Vlower=reshape(EVpre(aprimeIndex(:)),    [N_d2*N_a1*N_a2,N_a3,N_e]);
    Vupper=reshape(EVpre(aprimeplus1Index(:)),[N_d2*N_a1*N_a2,N_a3,N_e]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;

    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2*a1prime*a2prime,a3,e_cur)

    EVbase=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_e]);
    DiscountedEV_alt  =beta*EVbase;      % exponential (long-run) continuation
    DiscountedEV_tilde=beta0beta*EVbase; % QH-perceived continuation

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
        %% Valt (beta)
        entireRHS_ii_V=ReturnMatrix_ii+DiscountedEV_alt;
        [~,maxindex1_V]=max(entireRHS_ii_V,[],2);
        [Vtempii_V,maxindex2_V]=max(reshape(entireRHS_ii_V,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_e]),[],1);
        curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                 +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
        d2ind_V  =rem(maxindex2_V-1,N_d2)+1;
        a1pind_V =rem(floor((maxindex2_V-1)/N_d2),N_a1)+1;
        a2pind_V =floor((maxindex2_V-1)/(N_d2*N_a1))+1;
        Valt(curraindex,:,jj)       =shiftdim(Vtempii_V,1);
        Policyalt(1,curraindex,:,jj)=d2ind_V;
        Policyalt(2,curraindex,:,jj)=a1pind_V;
        Policyalt(3,curraindex,:,jj)=a2pind_V;

        maxgap_V=squeeze(max(max(max(max(max( maxindex1_V(:,1,:,2:end,:,:,:)-maxindex1_V(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
            if maxgap_V(ii)>0
                loweredge=min(maxindex1_V(:,1,:,ii,:,:,:),N_a1-maxgap_V(ii));
                a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                d2aprimee=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_e-1),-5);
                entireRHS_ii_V=reshape(ReturnMatrix_ii_dc+DiscountedEV_alt(d2aprimee),[N_d2*(maxgap_V(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_e]);
                [Vtempii_V,maxindex_V]=max(entireRHS_ii_V,[],1);
                Valt(curraindex,:,jj)=shiftdim(Vtempii_V,1);
                d2ind_V  =rem(maxindex_V-1,N_d2)+1;
                a1localind_V=rem(floor((maxindex_V-1)/N_d2),maxgap_V(ii)+1)+1;
                a2pind_V =floor((maxindex_V-1)/(N_d2*(maxgap_V(ii)+1)))+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind_V + N_d2*(a2pind_V-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*eind;
                a1prime_rec=a1localind_V+loweredge(loweredge_idx)-1;
                Policyalt(1,curraindex,:,jj)=d2ind_V;
                Policyalt(2,curraindex,:,jj)=a1prime_rec;
                Policyalt(3,curraindex,:,jj)=a2pind_V;
            else
                loweredge=maxindex1_V(:,1,:,ii,:,:,:);
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                d2aprimee=(1:1:N_d2)' + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_e-1),-5);
                entireRHS_ii_V=reshape(ReturnMatrix_ii_dc+DiscountedEV_alt(d2aprimee),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_e]);
                [Vtempii_V,maxindex_V]=max(entireRHS_ii_V,[],1);
                Valt(curraindex,:,jj)=shiftdim(Vtempii_V,1);
                d2ind_V  =rem(maxindex_V-1,N_d2)+1;
                a2pind_V =floor((maxindex_V-1)/N_d2)+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind_V + N_d2*(a2pind_V-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*eind;
                Policyalt(1,curraindex,:,jj)=d2ind_V;
                Policyalt(2,curraindex,:,jj)=loweredge(loweredge_idx);
                Policyalt(3,curraindex,:,jj)=a2pind_V;
            end
        end

        %% Vtilde (beta0*beta)
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV_tilde;
        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_e]),[],1);
        curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                 +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
        d2ind  =rem(maxindex2-1,N_d2)+1;
        a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
        a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
        Vtilde(curraindex,:,jj)       =shiftdim(Vtempii,1);
        Policy(1,curraindex,:,jj)=d2ind;
        Policy(2,curraindex,:,jj)=a1pind;
        Policy(3,curraindex,:,jj)=a2pind;

        maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                d2aprimee=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_e-1),-5);
                entireRHS_ii=reshape(ReturnMatrix_ii_dc+DiscountedEV_tilde(d2aprimee),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vtilde(curraindex,:,jj)=shiftdim(Vtempii,1);
                d2ind  =rem(maxindex-1,N_d2)+1;
                a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*eind;
                a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                Policy(1,curraindex,:,jj)=d2ind;
                Policy(2,curraindex,:,jj)=a1prime_rec;
                Policy(3,curraindex,:,jj)=a2pind;
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:);
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                d2aprimee=(1:1:N_d2)' + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_e-1),-5);
                entireRHS_ii=reshape(ReturnMatrix_ii_dc+DiscountedEV_tilde(d2aprimee),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                Vtilde(curraindex,:,jj)=shiftdim(Vtempii,1);
                d2ind  =rem(maxindex-1,N_d2)+1;
                a2pind =floor((maxindex-1)/N_d2)+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*eind;
                Policy(1,curraindex,:,jj)=d2ind;
                Policy(2,curraindex,:,jj)=loweredge(loweredge_idx);
                Policy(3,curraindex,:,jj)=a2pind;
            end
        end

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            DiscountedEV_alt_e  =DiscountedEV_alt(:,:,:,:,:,:,e_c);
            DiscountedEV_tilde_e=DiscountedEV_tilde(:,:,:,:,:,:,e_c);
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 1);
            %% Valt (beta)
            entireRHS_ii_e_V=ReturnMatrix_ii_e+DiscountedEV_alt_e;
            [~,maxindex1_V]=max(entireRHS_ii_e_V,[],2);
            [Vtempii_V,maxindex2_V]=max(reshape(entireRHS_ii_e_V,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            d2ind_V  =rem(maxindex2_V-1,N_d2)+1;
            a1pind_V =rem(floor((maxindex2_V-1)/N_d2),N_a1)+1;
            a2pind_V =floor((maxindex2_V-1)/(N_d2*N_a1))+1;
            Valt(curraindex,e_c,jj)       =shiftdim(Vtempii_V,1);
            Policyalt(1,curraindex,e_c,jj)=d2ind_V;
            Policyalt(2,curraindex,e_c,jj)=a1pind_V;
            Policyalt(3,curraindex,e_c,jj)=a2pind_V;

            maxgap_V=squeeze(max(max(max(max( maxindex1_V(:,1,:,2:end,:,:)-maxindex1_V(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1_V(:,1,:,ii,:,:),N_a1-maxgap_V(ii));
                    a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                    ReturnMatrix_ii_e_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 3);
                    d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_e_V=reshape(ReturnMatrix_ii_e_dc+DiscountedEV_alt_e(d2aprime),[N_d2*(maxgap_V(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                    [Vtempii_V,maxindex_V]=max(entireRHS_ii_e_V,[],1);
                    Valt(curraindex,e_c,jj)=shiftdim(Vtempii_V,1);
                    d2ind_V  =rem(maxindex_V-1,N_d2)+1;
                    a1localind_V=rem(floor((maxindex_V-1)/N_d2),maxgap_V(ii)+1)+1;
                    a2pind_V =floor((maxindex_V-1)/(N_d2*(maxgap_V(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=d2ind_V + N_d2*(a2pind_V-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                    a1prime_rec=a1localind_V+loweredge(loweredge_idx)-1;
                    Policyalt(1,curraindex,e_c,jj)=d2ind_V;
                    Policyalt(2,curraindex,e_c,jj)=a1prime_rec;
                    Policyalt(3,curraindex,e_c,jj)=a2pind_V;
                else
                    loweredge=maxindex1_V(:,1,:,ii,:,:);
                    ReturnMatrix_ii_e_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 3);
                    d2aprime=(1:1:N_d2)' + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_e_V=reshape(ReturnMatrix_ii_e_dc+DiscountedEV_alt_e(d2aprime),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                    [Vtempii_V,maxindex_V]=max(entireRHS_ii_e_V,[],1);
                    Valt(curraindex,e_c,jj)=shiftdim(Vtempii_V,1);
                    d2ind_V  =rem(maxindex_V-1,N_d2)+1;
                    a2pind_V =floor((maxindex_V-1)/N_d2)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=d2ind_V + N_d2*(a2pind_V-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                    Policyalt(1,curraindex,e_c,jj)=d2ind_V;
                    Policyalt(2,curraindex,e_c,jj)=loweredge(loweredge_idx);
                    Policyalt(3,curraindex,e_c,jj)=a2pind_V;
                end
            end

            %% Vtilde (beta0*beta)
            entireRHS_ii_e=ReturnMatrix_ii_e+DiscountedEV_tilde_e;
            [~,maxindex1]=max(entireRHS_ii_e,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_e,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            d2ind  =rem(maxindex2-1,N_d2)+1;
            a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
            a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
            Vtilde(curraindex,e_c,jj)       =shiftdim(Vtempii,1);
            Policy(1,curraindex,e_c,jj)=d2ind;
            Policy(2,curraindex,e_c,jj)=a1pind;
            Policy(3,curraindex,e_c,jj)=a2pind;

            maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         +N_a1*repmat(repelem(a2ind,level1iidiff(ii),1),N_a3,1) +N_a1*N_a2*repelem(a3ind,level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_e_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 3);
                    d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_e=reshape(ReturnMatrix_ii_e_dc+DiscountedEV_tilde_e(d2aprime),[N_d2*(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    Vtilde(curraindex,e_c,jj)=shiftdim(Vtempii,1);
                    d2ind  =rem(maxindex-1,N_d2)+1;
                    a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                    a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policy(1,curraindex,e_c,jj)=d2ind;
                    Policy(2,curraindex,e_c,jj)=a1prime_rec;
                    Policy(3,curraindex,e_c,jj)=a2pind;
                else
                    loweredge=maxindex1(:,1,:,ii,:,:);
                    ReturnMatrix_ii_e_dc=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 3);
                    d2aprime=(1:1:N_d2)' + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                    entireRHS_ii_e=reshape(ReturnMatrix_ii_e_dc+DiscountedEV_tilde_e(d2aprime),[N_d2*1*N_a2,level1iidiff(ii)*N_a2*N_a3]);
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    Vtilde(curraindex,e_c,jj)=shiftdim(Vtempii,1);
                    d2ind  =rem(maxindex-1,N_d2)+1;
                    a2pind =floor((maxindex-1)/N_d2)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                    Policy(1,curraindex,e_c,jj)=d2ind;
                    Policy(2,curraindex,e_c,jj)=loweredge(loweredge_idx);
                    Policy(3,curraindex,e_c,jj)=a2pind;
                end
            end
        end
    end
end


end
