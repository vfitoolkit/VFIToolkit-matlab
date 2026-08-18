function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteS_DC2A_nod1_noz_e_raw(n_d2, n_a1, n_a2, n_a3, n_e, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, e_gridvals_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Sophisticated quasi-hyperbolic discounting + ExperienceAssete, nod1 DC2A pattern.
% Sophisticated: a single max under beta0*beta,
%   Vhat is the max of  F + beta0*beta*EV  (the QH-perceived value),
%   Vunderbar is the  F + beta*EV  RHS GATHERED at that same DC argmax (NOT re-maximised).
% The a3 lottery is resolved inside EV (before the max), so the gathered RHS is R(policy)+beta*E[Vhat(policy)].
% beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj).
% Backward EVpre uses Vunderbar.
% Internal naming: a1 is the DC'd standard state, a2 the folded standard states, a3 the experience asset.
% Reuses CreateReturnFnMatrix_ExpAsset_Disc_DC2A passing n_e in the n_z slot.
% lowmemory=0 full vectorization; lowmemory=1 loop over e.

N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_e=prod(n_e);

Vhat=zeros(N_a,N_e,N_j,'gpuArray'); % QH-perceived value fn (beta0*beta)
Vunderbar=zeros(N_a,N_e,N_j,'gpuArray'); % exponential value at the QH policy
Policy=zeros(3,N_a,N_e,N_j,'gpuArray');


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
        Vhat(curraindex,:,N_j)       =shiftdim(Vtempii,1);
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
                Vhat(curraindex,:,N_j)=shiftdim(Vtempii,1);
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
                Vhat(curraindex,:,N_j)=shiftdim(Vtempii,1);
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
            Vhat(curraindex,e_c,N_j)       =shiftdim(Vtempii,1);
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
                    Vhat(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
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
                    Vhat(curraindex,e_c,N_j)=shiftdim(Vtempii,1);
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

    Vunderbar(:,:,N_j)=Vhat(:,:,N_j); % terminal: no continuation, so Vunderbar equals Vhat
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
    DiscountedEV_hat  =beta0beta*EVbase; % QH-perceived (the single max uses this)
    DiscountedEV_under=beta*EVbase;      % exponential (gathered at that same argmax)

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        entireRHS_hat=ReturnMatrix_ii+DiscountedEV_hat;
        [~,maxindex1]=max(entireRHS_hat,[],2);
        M=vfoptions.level1n*N_a2*N_a3;
        entireRHS_hat_flat=reshape(entireRHS_hat,[N_d2*N_a1*N_a2,M,N_e]);
        [Vtempii_hat,maxindex2]=max(entireRHS_hat_flat,[],1);
        entireRHS_under=ReturnMatrix_ii+DiscountedEV_under;
        entireRHS_under_flat=reshape(entireRHS_under,[N_d2*N_a1*N_a2,M,N_e]);
        maxindexfull=maxindex2 + (N_d2*N_a1*N_a2)*(0:M-1) + (N_d2*N_a1*N_a2)*M*shiftdim((0:N_e-1),-1);
        Vtempii_under=entireRHS_under_flat(maxindexfull);
        curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                 +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
        d2ind  =rem(maxindex2-1,N_d2)+1;
        a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
        a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
        Vhat(curraindex,:,N_j)       =shiftdim(Vtempii_hat,1);
        Vunderbar(curraindex,:,N_j)       =shiftdim(Vtempii_under,1);
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
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                d2aprimee=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_e-1),-5);
                firstdim=N_d2*(maxgap(ii)+1)*N_a2;
                Mblock=level1iidiff(ii)*N_a2*N_a3;
                entireRHS_hat=reshape(ReturnMatrix_ii+DiscountedEV_hat(d2aprimee),[firstdim,Mblock,N_e]);
                entireRHS_under=reshape(ReturnMatrix_ii+DiscountedEV_under(d2aprimee),[firstdim,Mblock,N_e]);
                [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_e-1),-1);
                Vtempii_under=entireRHS_under(maxindexfull);
                Vhat(curraindex,:,N_j)=shiftdim(Vtempii_hat,1);
                Vunderbar(curraindex,:,N_j)=shiftdim(Vtempii_under,1);
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
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                d2aprimee=(1:1:N_d2)' + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_e-1),-5);
                firstdim=N_d2*1*N_a2;
                Mblock=level1iidiff(ii)*N_a2*N_a3;
                entireRHS_hat=reshape(ReturnMatrix_ii+DiscountedEV_hat(d2aprimee),[firstdim,Mblock,N_e]);
                entireRHS_under=reshape(ReturnMatrix_ii+DiscountedEV_under(d2aprimee),[firstdim,Mblock,N_e]);
                [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_e-1),-1);
                Vtempii_under=entireRHS_under(maxindexfull);
                Vhat(curraindex,:,N_j)=shiftdim(Vtempii_hat,1);
                Vunderbar(curraindex,:,N_j)=shiftdim(Vtempii_under,1);
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
            DiscountedEV_hat_e  =DiscountedEV_hat(:,:,:,:,:,:,e_c);
            DiscountedEV_under_e=DiscountedEV_under(:,:,:,:,:,:,e_c);
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 1);
            entireRHS_hat_e=ReturnMatrix_ii_e+DiscountedEV_hat_e;
            [~,maxindex1]=max(entireRHS_hat_e,[],2);
            M=vfoptions.level1n*N_a2*N_a3;
            entireRHS_hat_flat=reshape(entireRHS_hat_e,[N_d2*N_a1*N_a2,M]);
            [Vtempii_hat,maxindex2]=max(entireRHS_hat_flat,[],1);
            entireRHS_under_e=ReturnMatrix_ii_e+DiscountedEV_under_e;
            entireRHS_under_flat=reshape(entireRHS_under_e,[N_d2*N_a1*N_a2,M]);
            maxindexfull=maxindex2 + (N_d2*N_a1*N_a2)*(0:M-1);
            Vtempii_under=entireRHS_under_flat(maxindexfull);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            d2ind  =rem(maxindex2-1,N_d2)+1;
            a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
            a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
            Vhat(curraindex,e_c,N_j)       =shiftdim(Vtempii_hat,1);
            Vunderbar(curraindex,e_c,N_j)       =shiftdim(Vtempii_under,1);
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
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 3);
                    d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                    firstdim=N_d2*(maxgap(ii)+1)*N_a2;
                    Mblock=level1iidiff(ii)*N_a2*N_a3;
                    entireRHS_hat_e=reshape(ReturnMatrix_ii_e+DiscountedEV_hat_e(d2aprime),[firstdim,Mblock]);
                    entireRHS_under_e=reshape(ReturnMatrix_ii_e+DiscountedEV_under_e(d2aprime),[firstdim,Mblock]);
                    [Vtempii_hat,maxindex]=max(entireRHS_hat_e,[],1);
                    maxindexfull=maxindex + firstdim*(0:Mblock-1);
                    Vtempii_under=entireRHS_under_e(maxindexfull);
                    Vhat(curraindex,e_c,N_j)=shiftdim(Vtempii_hat,1);
                    Vunderbar(curraindex,e_c,N_j)=shiftdim(Vtempii_under,1);
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
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 3);
                    d2aprime=(1:1:N_d2)' + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                    firstdim=N_d2*1*N_a2;
                    Mblock=level1iidiff(ii)*N_a2*N_a3;
                    entireRHS_hat_e=reshape(ReturnMatrix_ii_e+DiscountedEV_hat_e(d2aprime),[firstdim,Mblock]);
                    entireRHS_under_e=reshape(ReturnMatrix_ii_e+DiscountedEV_under_e(d2aprime),[firstdim,Mblock]);
                    [Vtempii_hat,maxindex]=max(entireRHS_hat_e,[],1);
                    maxindexfull=maxindex + firstdim*(0:Mblock-1);
                    Vtempii_under=entireRHS_under_e(maxindexfull);
                    Vhat(curraindex,e_c,N_j)=shiftdim(Vtempii_hat,1);
                    Vunderbar(curraindex,e_c,N_j)=shiftdim(Vtempii_under,1);
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

    EVpre=sum(pi_e_J(:,jj+1)'.*reshape(Vunderbar(:,:,jj+1),[N_a,N_e]),2); % integrate out eprime: [N_a,1]

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
    DiscountedEV_hat  =beta0beta*EVbase; % QH-perceived (the single max uses this)
    DiscountedEV_under=beta*EVbase;      % exponential (gathered at that same argmax)

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
        entireRHS_hat=ReturnMatrix_ii+DiscountedEV_hat;
        [~,maxindex1]=max(entireRHS_hat,[],2);
        M=vfoptions.level1n*N_a2*N_a3;
        entireRHS_hat_flat=reshape(entireRHS_hat,[N_d2*N_a1*N_a2,M,N_e]);
        [Vtempii_hat,maxindex2]=max(entireRHS_hat_flat,[],1);
        entireRHS_under=ReturnMatrix_ii+DiscountedEV_under;
        entireRHS_under_flat=reshape(entireRHS_under,[N_d2*N_a1*N_a2,M,N_e]);
        maxindexfull=maxindex2 + (N_d2*N_a1*N_a2)*(0:M-1) + (N_d2*N_a1*N_a2)*M*shiftdim((0:N_e-1),-1);
        Vtempii_under=entireRHS_under_flat(maxindexfull);
        curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                 +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
        d2ind  =rem(maxindex2-1,N_d2)+1;
        a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
        a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
        Vhat(curraindex,:,jj)       =shiftdim(Vtempii_hat,1);
        Vunderbar(curraindex,:,jj)       =shiftdim(Vtempii_under,1);
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
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                d2aprimee=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_e-1),-5);
                firstdim=N_d2*(maxgap(ii)+1)*N_a2;
                Mblock=level1iidiff(ii)*N_a2*N_a3;
                entireRHS_hat=reshape(ReturnMatrix_ii+DiscountedEV_hat(d2aprimee),[firstdim,Mblock,N_e]);
                entireRHS_under=reshape(ReturnMatrix_ii+DiscountedEV_under(d2aprimee),[firstdim,Mblock,N_e]);
                [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_e-1),-1);
                Vtempii_under=entireRHS_under(maxindexfull);
                Vhat(curraindex,:,jj)=shiftdim(Vtempii_hat,1);
                Vunderbar(curraindex,:,jj)=shiftdim(Vtempii_under,1);
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
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                d2aprimee=(1:1:N_d2)' + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_e-1),-5);
                firstdim=N_d2*1*N_a2;
                Mblock=level1iidiff(ii)*N_a2*N_a3;
                entireRHS_hat=reshape(ReturnMatrix_ii+DiscountedEV_hat(d2aprimee),[firstdim,Mblock,N_e]);
                entireRHS_under=reshape(ReturnMatrix_ii+DiscountedEV_under(d2aprimee),[firstdim,Mblock,N_e]);
                [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_e-1),-1);
                Vtempii_under=entireRHS_under(maxindexfull);
                Vhat(curraindex,:,jj)=shiftdim(Vtempii_hat,1);
                Vunderbar(curraindex,:,jj)=shiftdim(Vtempii_under,1);
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
            DiscountedEV_hat_e  =DiscountedEV_hat(:,:,:,:,:,:,e_c);
            DiscountedEV_under_e=DiscountedEV_under(:,:,:,:,:,:,e_c);
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 1);
            entireRHS_hat_e=ReturnMatrix_ii_e+DiscountedEV_hat_e;
            [~,maxindex1]=max(entireRHS_hat_e,[],2);
            M=vfoptions.level1n*N_a2*N_a3;
            entireRHS_hat_flat=reshape(entireRHS_hat_e,[N_d2*N_a1*N_a2,M]);
            [Vtempii_hat,maxindex2]=max(entireRHS_hat_flat,[],1);
            entireRHS_under_e=ReturnMatrix_ii_e+DiscountedEV_under_e;
            entireRHS_under_flat=reshape(entireRHS_under_e,[N_d2*N_a1*N_a2,M]);
            maxindexfull=maxindex2 + (N_d2*N_a1*N_a2)*(0:M-1);
            Vtempii_under=entireRHS_under_flat(maxindexfull);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     +N_a1*repmat(repelem(a2ind,vfoptions.level1n,1),N_a3,1) +N_a1*N_a2*repelem(a3ind,vfoptions.level1n*N_a2,1);
            d2ind  =rem(maxindex2-1,N_d2)+1;
            a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
            a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
            Vhat(curraindex,e_c,jj)       =shiftdim(Vtempii_hat,1);
            Vunderbar(curraindex,e_c,jj)       =shiftdim(Vtempii_under,1);
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
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 3);
                    d2aprime=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                    firstdim=N_d2*(maxgap(ii)+1)*N_a2;
                    Mblock=level1iidiff(ii)*N_a2*N_a3;
                    entireRHS_hat_e=reshape(ReturnMatrix_ii_e+DiscountedEV_hat_e(d2aprime),[firstdim,Mblock]);
                    entireRHS_under_e=reshape(ReturnMatrix_ii_e+DiscountedEV_under_e(d2aprime),[firstdim,Mblock]);
                    [Vtempii_hat,maxindex]=max(entireRHS_hat_e,[],1);
                    maxindexfull=maxindex + firstdim*(0:Mblock-1);
                    Vtempii_under=entireRHS_under_e(maxindexfull);
                    Vhat(curraindex,e_c,jj)=shiftdim(Vtempii_hat,1);
                    Vunderbar(curraindex,e_c,jj)=shiftdim(Vtempii_under,1);
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
                    ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, e_val, ReturnFnParamsVec, 3);
                    d2aprime=(1:1:N_d2)' + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                    firstdim=N_d2*1*N_a2;
                    Mblock=level1iidiff(ii)*N_a2*N_a3;
                    entireRHS_hat_e=reshape(ReturnMatrix_ii_e+DiscountedEV_hat_e(d2aprime),[firstdim,Mblock]);
                    entireRHS_under_e=reshape(ReturnMatrix_ii_e+DiscountedEV_under_e(d2aprime),[firstdim,Mblock]);
                    [Vtempii_hat,maxindex]=max(entireRHS_hat_e,[],1);
                    maxindexfull=maxindex + firstdim*(0:Mblock-1);
                    Vtempii_under=entireRHS_under_e(maxindexfull);
                    Vhat(curraindex,e_c,jj)=shiftdim(Vtempii_hat,1);
                    Vunderbar(curraindex,e_c,jj)=shiftdim(Vtempii_under,1);
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
