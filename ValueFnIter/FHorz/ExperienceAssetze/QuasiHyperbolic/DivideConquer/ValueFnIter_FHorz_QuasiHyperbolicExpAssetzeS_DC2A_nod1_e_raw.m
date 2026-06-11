function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeS_DC2A_nod1_e_raw(n_d2, n_a1, n_a2, n_a3, n_z, n_e, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Sophisticated QH + ExpAssetze (z+e dep aprimeFn), DC2A pattern (nod1).
% lowmemory=0 full vectorization; lowmemory=1 loops z (e vectorized); lowmemory=2 nested z+e.

N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_z=prod(n_z);
N_e=prod(n_e);

Vhat=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(3,N_a,N_z,N_e,N_j,'gpuArray');

zind=shiftdim((0:1:N_z-1),-1);
eind=shiftdim((0:1:N_e-1),-2);

if vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
elseif vfoptions.lowmemory==2
    special_n_z=ones(1,length(n_z));
    special_n_e=ones(1,length(n_e));
end

level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% j=N_j (terminal)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, n_z, n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_z,N_e]),[],1);
        curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                 + N_a1   *repmat(repelem((0:N_a2-1)',vfoptions.level1n,1),N_a3,1) ...
                 + N_a1*N_a2*repelem((0:N_a3-1)',vfoptions.level1n*N_a2,1);
        d2ind  =rem(maxindex2-1,N_d2)+1;
        a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
        a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
        Vhat(curraindex,:,:,N_j)       =shiftdim(Vtempii,1);
        Policy(1,curraindex,:,:,N_j)   =d2ind;
        Policy(2,curraindex,:,:,N_j)   =a1pind;
        Policy(3,curraindex,:,:,N_j)   =a2pind;
        maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                     + N_a1   *repmat(repelem((0:N_a2-1)',level1iidiff(ii),1),N_a3,1) ...
                     + N_a1*N_a2*repelem((0:N_a3-1)',level1iidiff(ii)*N_a2,1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, n_z, n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Vhat(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                d2ind  =rem(maxindex-1,N_d2)+1;
                a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*zind + N_d2*N_a2*N_a2*N_a3*N_z*eind;
                a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                Policy(1,curraindex,:,:,N_j)=d2ind;
                Policy(2,curraindex,:,:,N_j)=a1prime_rec;
                Policy(3,curraindex,:,:,N_j)=a2pind;
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, n_z, n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Vhat(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                d2ind  =rem(maxindex-1,N_d2)+1;
                a2pind =floor((maxindex-1)/N_d2)+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*zind + N_d2*N_a2*N_a2*N_a3*N_z*eind;
                Policy(1,curraindex,:,:,N_j)=d2ind;
                Policy(2,curraindex,:,:,N_j)=loweredge(loweredge_idx);
                Policy(3,curraindex,:,:,N_j)=a2pind;
            end
        end
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            [~,maxindex1_z]=max(ReturnMatrix_ii_z,[],2);
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_z,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,1,N_e]),[],1);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     + N_a1   *repmat(repelem((0:N_a2-1)',vfoptions.level1n,1),N_a3,1) ...
                     + N_a1*N_a2*repelem((0:N_a3-1)',vfoptions.level1n*N_a2,1);
            d2ind  =rem(maxindex2-1,N_d2)+1;
            a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
            a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
            Vhat(curraindex,z_c,:,N_j)       =shiftdim(Vtempii,1);
            Policy(1,curraindex,z_c,:,N_j)   =d2ind;
            Policy(2,curraindex,z_c,:,N_j)   =a1pind;
            Policy(3,curraindex,z_c,:,N_j)   =a2pind;
            maxgap=squeeze(max(max(max(max(max( maxindex1_z(:,1,:,2:end,:,:,1,:)-maxindex1_z(:,1,:,1:end-1,:,:,1,:), [],8),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         + N_a1   *repmat(repelem((0:N_a2-1)',level1iidiff(ii),1),N_a3,1) ...
                         + N_a1*N_a2*repelem((0:N_a3-1)',level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1_z(:,1,:,ii,:,:,1,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_z,[],1);
                    Vhat(curraindex,z_c,:,N_j)=shiftdim(Vtempii,1);
                    d2ind  =rem(maxindex-1,N_d2)+1;
                    a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                    a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*eind;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policy(1,curraindex,z_c,:,N_j)=d2ind;
                    Policy(2,curraindex,z_c,:,N_j)=a1prime_rec;
                    Policy(3,curraindex,z_c,:,N_j)=a2pind;
                else
                    loweredge=maxindex1_z(:,1,:,ii,:,:,1,:);
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_z,[],1);
                    Vhat(curraindex,z_c,:,N_j)=shiftdim(Vtempii,1);
                    d2ind  =rem(maxindex-1,N_d2)+1;
                    a2pind =floor((maxindex-1)/N_d2)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*eind;
                    Policy(1,curraindex,z_c,:,N_j)=d2ind;
                    Policy(2,curraindex,z_c,:,N_j)=loweredge(loweredge_idx);
                    Policy(3,curraindex,z_c,:,N_j)=a2pind;
                end
            end
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, special_n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                [~,maxindex1_ze]=max(ReturnMatrix_ii_ze,[],2);
                [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_ze,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]),[],1);
                curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                         + N_a1   *repmat(repelem((0:N_a2-1)',vfoptions.level1n,1),N_a3,1) ...
                         + N_a1*N_a2*repelem((0:N_a3-1)',vfoptions.level1n*N_a2,1);
                d2ind  =rem(maxindex2-1,N_d2)+1;
                a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
                a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
                Vhat(curraindex,z_c,e_c,N_j)       =Vtempii;
                Policy(1,curraindex,z_c,e_c,N_j)   =d2ind;
                Policy(2,curraindex,z_c,e_c,N_j)   =a1pind;
                Policy(3,curraindex,z_c,e_c,N_j)   =a2pind;
                maxgap=squeeze(max(max(max(max( maxindex1_ze(:,1,:,2:end,:,:)-maxindex1_ze(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                             + N_a1   *repmat(repelem((0:N_a2-1)',level1iidiff(ii),1),N_a3,1) ...
                             + N_a1*N_a2*repelem((0:N_a3-1)',level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1_ze(:,1,:,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, special_n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 2);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii_ze,[],1);
                        Vhat(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        d2ind  =rem(maxindex-1,N_d2)+1;
                        a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                        a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policy(1,curraindex,z_c,e_c,N_j)=d2ind;
                        Policy(2,curraindex,z_c,e_c,N_j)=a1prime_rec;
                        Policy(3,curraindex,z_c,e_c,N_j)=a2pind;
                    else
                        loweredge=maxindex1_ze(:,1,:,ii,:,:);
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, special_n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 2);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii_ze,[],1);
                        Vhat(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        d2ind  =rem(maxindex-1,N_d2)+1;
                        a2pind =floor((maxindex-1)/N_d2)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                        Policy(1,curraindex,z_c,e_c,N_j)=d2ind;
                        Policy(2,curraindex,z_c,e_c,N_j)=loweredge(loweredge_idx);
                        Policy(3,curraindex,z_c,e_c,N_j)=a2pind;
                    end
                end
            end
        end
    end
    Vunderbar(:,:,:,N_j)=Vhat(:,:,:,N_j);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EVpre=squeeze(sum(reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]).*shiftdim(pi_e_J(:,N_j),-2),3));

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetzeFnMatrix(aprimeFn, n_d2, n_a3, n_z, n_e, d2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,1,N_z);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    EV=EV.*reshape(pi_z_J(:,:,N_j),[1,1,N_z,1,N_z]);
    EV(isnan(EV))=0;
    EV=squeeze(sum(EV,5));

    EVbase=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_z,N_e]);
    DiscountedEV_under=beta*EVbase;
    DiscountedEV_hat  =beta0beta*EVbase;

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, n_z, n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        entireRHS_hat=ReturnMatrix_ii+DiscountedEV_hat;
        [~,maxindex1]=max(entireRHS_hat,[],2);
        entireRHS_hat_flat=reshape(entireRHS_hat,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_z,N_e]);
        [Vtempii_hat,maxindex2]=max(entireRHS_hat_flat,[],1);
        entireRHS_under=ReturnMatrix_ii+DiscountedEV_under;
        entireRHS_under_flat=reshape(entireRHS_under,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_z,N_e]);
        M=vfoptions.level1n*N_a2*N_a3;
        maxindexfull=maxindex2 + (N_d2*N_a1*N_a2)*(0:M-1) + (N_d2*N_a1*N_a2)*M*shiftdim((0:N_z-1),-1) + (N_d2*N_a1*N_a2)*M*N_z*shiftdim((0:N_e-1),-2);
        Vtempii_under=entireRHS_under_flat(maxindexfull);
        curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                 + N_a1   *repmat(repelem((0:N_a2-1)',vfoptions.level1n,1),N_a3,1) ...
                 + N_a1*N_a2*repelem((0:N_a3-1)',vfoptions.level1n*N_a2,1);
        d2ind  =rem(maxindex2-1,N_d2)+1;
        a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
        a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
        Vhat(curraindex,:,:,N_j)         =shiftdim(Vtempii_hat,1);
        Vunderbar(curraindex,:,:,N_j)    =shiftdim(Vtempii_under,1);
        Policy(1,curraindex,:,:,N_j)     =d2ind;
        Policy(2,curraindex,:,:,N_j)     =a1pind;
        Policy(3,curraindex,:,:,N_j)     =a2pind;
        maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                     + N_a1   *repmat(repelem((0:N_a2-1)',level1iidiff(ii),1),N_a3,1) ...
                     + N_a1*N_a2*repelem((0:N_a3-1)',level1iidiff(ii)*N_a2,1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, n_z, n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                d2aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_z-1),-5) + N_d2*N_a1*N_a2*N_a3*N_z*shiftdim((0:1:N_e-1),-6);
                firstdim=N_d2*(maxgap(ii)+1)*N_a2;
                Mblock=level1iidiff(ii)*N_a2*N_a3;
                entireRHS_hat=reshape(ReturnMatrix_ii+DiscountedEV_hat(d2aprimez),[firstdim,Mblock,N_z,N_e]);
                entireRHS_under=reshape(ReturnMatrix_ii+DiscountedEV_under(d2aprimez),[firstdim,Mblock,N_z,N_e]);
                [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_z-1),-1) + firstdim*Mblock*N_z*shiftdim((0:N_e-1),-2);
                Vtempii_under=entireRHS_under(maxindexfull);
                Vhat(curraindex,:,:,N_j)      =shiftdim(Vtempii_hat,1);
                Vunderbar(curraindex,:,:,N_j) =shiftdim(Vtempii_under,1);
                d2ind  =rem(maxindex-1,N_d2)+1;
                a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*zind + N_d2*N_a2*N_a2*N_a3*N_z*eind;
                a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                Policy(1,curraindex,:,:,N_j)=d2ind;
                Policy(2,curraindex,:,:,N_j)=a1prime_rec;
                Policy(3,curraindex,:,:,N_j)=a2pind;
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, n_z, n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
                d2aprimez=(1:1:N_d2)' + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_z-1),-5) + N_d2*N_a1*N_a2*N_a3*N_z*shiftdim((0:1:N_e-1),-6);
                firstdim=N_d2*1*N_a2;
                Mblock=level1iidiff(ii)*N_a2*N_a3;
                entireRHS_hat=reshape(ReturnMatrix_ii+DiscountedEV_hat(d2aprimez),[firstdim,Mblock,N_z,N_e]);
                entireRHS_under=reshape(ReturnMatrix_ii+DiscountedEV_under(d2aprimez),[firstdim,Mblock,N_z,N_e]);
                [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_z-1),-1) + firstdim*Mblock*N_z*shiftdim((0:N_e-1),-2);
                Vtempii_under=entireRHS_under(maxindexfull);
                Vhat(curraindex,:,:,N_j)      =shiftdim(Vtempii_hat,1);
                Vunderbar(curraindex,:,:,N_j) =shiftdim(Vtempii_under,1);
                d2ind  =rem(maxindex-1,N_d2)+1;
                a2pind =floor((maxindex-1)/N_d2)+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*zind + N_d2*N_a2*N_a2*N_a3*N_z*eind;
                Policy(1,curraindex,:,:,N_j)=d2ind;
                Policy(2,curraindex,:,:,N_j)=loweredge(loweredge_idx);
                Policy(3,curraindex,:,:,N_j)=a2pind;
            end
        end
    elseif vfoptions.lowmemory==1
        error('lowmem=1 for QH+ExpAssetze S DC2A V_Jplus1 init not yet implemented')
    elseif vfoptions.lowmemory==2
        error('lowmem=2 for QH+ExpAssetze S DC2A V_Jplus1 init not yet implemented')
    end
end


%% Backward iteration
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

    EVpre=squeeze(sum(Vunderbar(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj),-2),3));

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetzeFnMatrix(aprimeFn, n_d2, n_a3, n_z, n_e, d2_gridvals, a3_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), aprimeFnParamsVec,2);

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,1,N_z);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_e,N_z]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    EV=EV.*reshape(pi_z_J(:,:,jj),[1,1,N_z,1,N_z]);
    EV(isnan(EV))=0;
    EV=squeeze(sum(EV,5));

    EVbase=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_z,N_e]);
    DiscountedEV_under=beta*EVbase;
    DiscountedEV_hat  =beta0beta*EVbase;

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, n_z, n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
        entireRHS_hat=ReturnMatrix_ii+DiscountedEV_hat;
        [~,maxindex1]=max(entireRHS_hat,[],2);
        entireRHS_hat_flat=reshape(entireRHS_hat,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_z,N_e]);
        [Vtempii_hat,maxindex2]=max(entireRHS_hat_flat,[],1);
        entireRHS_under=ReturnMatrix_ii+DiscountedEV_under;
        entireRHS_under_flat=reshape(entireRHS_under,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_z,N_e]);
        M=vfoptions.level1n*N_a2*N_a3;
        maxindexfull=maxindex2 + (N_d2*N_a1*N_a2)*(0:M-1) + (N_d2*N_a1*N_a2)*M*shiftdim((0:N_z-1),-1) + (N_d2*N_a1*N_a2)*M*N_z*shiftdim((0:N_e-1),-2);
        Vtempii_under=entireRHS_under_flat(maxindexfull);
        curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                 + N_a1   *repmat(repelem((0:N_a2-1)',vfoptions.level1n,1),N_a3,1) ...
                 + N_a1*N_a2*repelem((0:N_a3-1)',vfoptions.level1n*N_a2,1);
        d2ind  =rem(maxindex2-1,N_d2)+1;
        a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
        a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
        Vhat(curraindex,:,:,jj)        =shiftdim(Vtempii_hat,1);
        Vunderbar(curraindex,:,:,jj)   =shiftdim(Vtempii_under,1);
        Policy(1,curraindex,:,:,jj)    =d2ind;
        Policy(2,curraindex,:,:,jj)    =a1pind;
        Policy(3,curraindex,:,:,jj)    =a2pind;
        maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                     + N_a1   *repmat(repelem((0:N_a2-1)',level1iidiff(ii),1),N_a3,1) ...
                     + N_a1*N_a2*repelem((0:N_a3-1)',level1iidiff(ii)*N_a2,1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, n_z, n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                d2aprimez=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_z-1),-5) + N_d2*N_a1*N_a2*N_a3*N_z*shiftdim((0:1:N_e-1),-6);
                firstdim=N_d2*(maxgap(ii)+1)*N_a2;
                Mblock=level1iidiff(ii)*N_a2*N_a3;
                entireRHS_hat=reshape(ReturnMatrix_ii+DiscountedEV_hat(d2aprimez),[firstdim,Mblock,N_z,N_e]);
                entireRHS_under=reshape(ReturnMatrix_ii+DiscountedEV_under(d2aprimez),[firstdim,Mblock,N_z,N_e]);
                [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_z-1),-1) + firstdim*Mblock*N_z*shiftdim((0:N_e-1),-2);
                Vtempii_under=entireRHS_under(maxindexfull);
                Vhat(curraindex,:,:,jj)      =shiftdim(Vtempii_hat,1);
                Vunderbar(curraindex,:,:,jj) =shiftdim(Vtempii_under,1);
                d2ind  =rem(maxindex-1,N_d2)+1;
                a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*zind + N_d2*N_a2*N_a2*N_a3*N_z*eind;
                a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                Policy(1,curraindex,:,:,jj)=d2ind;
                Policy(2,curraindex,:,:,jj)=a1prime_rec;
                Policy(3,curraindex,:,:,jj)=a2pind;
            else
                loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, n_z, n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                d2aprimez=(1:1:N_d2)' + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_z-1),-5) + N_d2*N_a1*N_a2*N_a3*N_z*shiftdim((0:1:N_e-1),-6);
                firstdim=N_d2*1*N_a2;
                Mblock=level1iidiff(ii)*N_a2*N_a3;
                entireRHS_hat=reshape(ReturnMatrix_ii+DiscountedEV_hat(d2aprimez),[firstdim,Mblock,N_z,N_e]);
                entireRHS_under=reshape(ReturnMatrix_ii+DiscountedEV_under(d2aprimez),[firstdim,Mblock,N_z,N_e]);
                [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
                maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_z-1),-1) + firstdim*Mblock*N_z*shiftdim((0:N_e-1),-2);
                Vtempii_under=entireRHS_under(maxindexfull);
                Vhat(curraindex,:,:,jj)      =shiftdim(Vtempii_hat,1);
                Vunderbar(curraindex,:,:,jj) =shiftdim(Vtempii_under,1);
                d2ind  =rem(maxindex-1,N_d2)+1;
                a2pind =floor((maxindex-1)/N_d2)+1;
                a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*zind + N_d2*N_a2*N_a2*N_a3*N_z*eind;
                Policy(1,curraindex,:,:,jj)=d2ind;
                Policy(2,curraindex,:,:,jj)=loweredge(loweredge_idx);
                Policy(3,curraindex,:,:,jj)=a2pind;
            end
        end
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_under_z=DiscountedEV_under(:,:,:,:,:,:,z_c,:);
            DiscountedEV_hat_z  =DiscountedEV_hat  (:,:,:,:,:,:,z_c,:);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);
            entireRHS_hat_z=ReturnMatrix_ii_z+DiscountedEV_hat_z;
            [~,maxindex1_z]=max(entireRHS_hat_z,[],2);
            entireRHS_hat_flat=reshape(entireRHS_hat_z,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,1,N_e]);
            [Vtempii_hat,maxindex2]=max(entireRHS_hat_flat,[],1);
            entireRHS_under_z=ReturnMatrix_ii_z+DiscountedEV_under_z;
            entireRHS_under_flat=reshape(entireRHS_under_z,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,1,N_e]);
            M=vfoptions.level1n*N_a2*N_a3;
            maxindexfull=maxindex2 + (N_d2*N_a1*N_a2)*(0:M-1) + (N_d2*N_a1*N_a2)*M*shiftdim((0:N_e-1),-2);
            Vtempii_under=entireRHS_under_flat(maxindexfull);
            curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                     + N_a1   *repmat(repelem((0:N_a2-1)',vfoptions.level1n,1),N_a3,1) ...
                     + N_a1*N_a2*repelem((0:N_a3-1)',vfoptions.level1n*N_a2,1);
            d2ind  =rem(maxindex2-1,N_d2)+1;
            a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
            a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
            Vhat(curraindex,z_c,:,jj)        =shiftdim(Vtempii_hat,1);
            Vunderbar(curraindex,z_c,:,jj)   =shiftdim(Vtempii_under,1);
            Policy(1,curraindex,z_c,:,jj)    =d2ind;
            Policy(2,curraindex,z_c,:,jj)    =a1pind;
            Policy(3,curraindex,z_c,:,jj)    =a2pind;
            maxgap=squeeze(max(max(max(max(max( maxindex1_z(:,1,:,2:end,:,:,1,:)-maxindex1_z(:,1,:,1:end-1,:,:,1,:), [],8),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                         + N_a1   *repmat(repelem((0:N_a2-1)',level1iidiff(ii),1),N_a3,1) ...
                         + N_a1*N_a2*repelem((0:N_a3-1)',level1iidiff(ii)*N_a2,1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1_z(:,1,:,ii,:,:,1,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprime_z=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_e-1),-6);
                    firstdim=N_d2*(maxgap(ii)+1)*N_a2;
                    Mblock=level1iidiff(ii)*N_a2*N_a3;
                    entireRHS_hat_z=reshape(ReturnMatrix_ii_z+DiscountedEV_hat_z(d2aprime_z),[firstdim,Mblock,1,N_e]);
                    entireRHS_under_z=reshape(ReturnMatrix_ii_z+DiscountedEV_under_z(d2aprime_z),[firstdim,Mblock,1,N_e]);
                    [Vtempii_hat,maxindex]=max(entireRHS_hat_z,[],1);
                    maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_e-1),-2);
                    Vtempii_under=entireRHS_under_z(maxindexfull);
                    Vhat(curraindex,z_c,:,jj)      =shiftdim(Vtempii_hat,1);
                    Vunderbar(curraindex,z_c,:,jj) =shiftdim(Vtempii_under,1);
                    d2ind  =rem(maxindex-1,N_d2)+1;
                    a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                    a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*eind;
                    a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                    Policy(1,curraindex,z_c,:,jj)=d2ind;
                    Policy(2,curraindex,z_c,:,jj)=a1prime_rec;
                    Policy(3,curraindex,z_c,:,jj)=a2pind;
                else
                    loweredge=maxindex1_z(:,1,:,ii,:,:,1,:);
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
                    d2aprime_z=(1:1:N_d2)' + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_e-1),-6);
                    firstdim=N_d2*1*N_a2;
                    Mblock=level1iidiff(ii)*N_a2*N_a3;
                    entireRHS_hat_z=reshape(ReturnMatrix_ii_z+DiscountedEV_hat_z(d2aprime_z),[firstdim,Mblock,1,N_e]);
                    entireRHS_under_z=reshape(ReturnMatrix_ii_z+DiscountedEV_under_z(d2aprime_z),[firstdim,Mblock,1,N_e]);
                    [Vtempii_hat,maxindex]=max(entireRHS_hat_z,[],1);
                    maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_e-1),-2);
                    Vtempii_under=entireRHS_under_z(maxindexfull);
                    Vhat(curraindex,z_c,:,jj)      =shiftdim(Vtempii_hat,1);
                    Vunderbar(curraindex,z_c,:,jj) =shiftdim(Vtempii_under,1);
                    d2ind  =rem(maxindex-1,N_d2)+1;
                    a2pind =floor((maxindex-1)/N_d2)+1;
                    a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                    a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                    loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat + N_d2*N_a2*N_a2*N_a3*eind;
                    Policy(1,curraindex,z_c,:,jj)=d2ind;
                    Policy(2,curraindex,z_c,:,jj)=loweredge(loweredge_idx);
                    Policy(3,curraindex,z_c,:,jj)=a2pind;
                end
            end
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                DiscountedEV_under_ze=DiscountedEV_under(:,:,:,:,:,:,z_c,e_c);
                DiscountedEV_hat_ze  =DiscountedEV_hat  (:,:,:,:,:,:,z_c,e_c);
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, special_n_e, d2_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                entireRHS_hat_ze=ReturnMatrix_ii_ze+DiscountedEV_hat_ze;
                [~,maxindex1_ze]=max(entireRHS_hat_ze,[],2);
                entireRHS_hat_flat=reshape(entireRHS_hat_ze,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]);
                [Vtempii_hat,maxindex2]=max(entireRHS_hat_flat,[],1);
                entireRHS_under_ze=ReturnMatrix_ii_ze+DiscountedEV_under_ze;
                entireRHS_under_flat=reshape(entireRHS_under_ze,[N_d2*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3]);
                M=vfoptions.level1n*N_a2*N_a3;
                maxindexfull=maxindex2 + (N_d2*N_a1*N_a2)*(0:M-1);
                Vtempii_under=entireRHS_under_flat(maxindexfull);
                curraindex=repmat(level1ii',N_a2*N_a3,1) ...
                         + N_a1   *repmat(repelem((0:N_a2-1)',vfoptions.level1n,1),N_a3,1) ...
                         + N_a1*N_a2*repelem((0:N_a3-1)',vfoptions.level1n*N_a2,1);
                d2ind  =rem(maxindex2-1,N_d2)+1;
                a1pind =rem(floor((maxindex2-1)/N_d2),N_a1)+1;
                a2pind =floor((maxindex2-1)/(N_d2*N_a1))+1;
                Vhat(curraindex,z_c,e_c,jj)        =Vtempii_hat;
                Vunderbar(curraindex,z_c,e_c,jj)   =Vtempii_under;
                Policy(1,curraindex,z_c,e_c,jj)    =d2ind;
                Policy(2,curraindex,z_c,e_c,jj)    =a1pind;
                Policy(3,curraindex,z_c,e_c,jj)    =a2pind;
                maxgap=squeeze(max(max(max(max( maxindex1_ze(:,1,:,2:end,:,:)-maxindex1_ze(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2*N_a3,1) ...
                             + N_a1   *repmat(repelem((0:N_a2-1)',level1iidiff(ii),1),N_a3,1) ...
                             + N_a1*N_a2*repelem((0:N_a3-1)',level1iidiff(ii)*N_a2,1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1_ze(:,1,:,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, special_n_e, d2_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                        d2aprime_ze=(1:1:N_d2)' + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        firstdim=N_d2*(maxgap(ii)+1)*N_a2;
                        Mblock=level1iidiff(ii)*N_a2*N_a3;
                        entireRHS_hat_ze=reshape(ReturnMatrix_ii_ze+DiscountedEV_hat_ze(d2aprime_ze),[firstdim,Mblock]);
                        entireRHS_under_ze=reshape(ReturnMatrix_ii_ze+DiscountedEV_under_ze(d2aprime_ze),[firstdim,Mblock]);
                        [Vtempii_hat,maxindex]=max(entireRHS_hat_ze,[],1);
                        maxindexfull=maxindex + firstdim*(0:Mblock-1);
                        Vtempii_under=entireRHS_under_ze(maxindexfull);
                        Vhat(curraindex,z_c,e_c,jj)      =shiftdim(Vtempii_hat,1);
                        Vunderbar(curraindex,z_c,e_c,jj) =shiftdim(Vtempii_under,1);
                        d2ind  =rem(maxindex-1,N_d2)+1;
                        a1localind=rem(floor((maxindex-1)/N_d2),maxgap(ii)+1)+1;
                        a2pind =floor((maxindex-1)/(N_d2*(maxgap(ii)+1)))+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                        a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
                        Policy(1,curraindex,z_c,e_c,jj)=d2ind;
                        Policy(2,curraindex,z_c,e_c,jj)=a1prime_rec;
                        Policy(3,curraindex,z_c,e_c,jj)=a2pind;
                    else
                        loweredge=maxindex1_ze(:,1,:,ii,:,:);
                        ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0, n_d2, n_a2, n_a3, special_n_z, special_n_e, d2_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 3);
                        d2aprime_ze=(1:1:N_d2)' + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4);
                        firstdim=N_d2*1*N_a2;
                        Mblock=level1iidiff(ii)*N_a2*N_a3;
                        entireRHS_hat_ze=reshape(ReturnMatrix_ii_ze+DiscountedEV_hat_ze(d2aprime_ze),[firstdim,Mblock]);
                        entireRHS_under_ze=reshape(ReturnMatrix_ii_ze+DiscountedEV_under_ze(d2aprime_ze),[firstdim,Mblock]);
                        [Vtempii_hat,maxindex]=max(entireRHS_hat_ze,[],1);
                        maxindexfull=maxindex + firstdim*(0:Mblock-1);
                        Vtempii_under=entireRHS_under_ze(maxindexfull);
                        Vhat(curraindex,z_c,e_c,jj)      =shiftdim(Vtempii_hat,1);
                        Vunderbar(curraindex,z_c,e_c,jj) =shiftdim(Vtempii_under,1);
                        d2ind  =rem(maxindex-1,N_d2)+1;
                        a2pind =floor((maxindex-1)/N_d2)+1;
                        a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
                        a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
                        loweredge_idx=d2ind + N_d2*(a2pind-1) + N_d2*N_a2*a2ind_flat + N_d2*N_a2*N_a2*a3ind_flat;
                        Policy(1,curraindex,z_c,e_c,jj)=d2ind;
                        Policy(2,curraindex,z_c,e_c,jj)=loweredge(loweredge_idx);
                        Policy(3,curraindex,z_c,e_c,jj)=a2pind;
                    end
                end
            end
        end
    end
end


end
