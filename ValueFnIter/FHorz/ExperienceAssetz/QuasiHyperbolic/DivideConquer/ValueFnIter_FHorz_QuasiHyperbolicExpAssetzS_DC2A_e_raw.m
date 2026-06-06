function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_DC2A_e_raw(n_d1, n_d2, n_a1, n_a2, n_a3, n_z, n_e, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Sophisticated QH + ExpAssetz, DC2A pattern (with d1).
% Policy channels: 1=d (joint), 2=a1prime, 3=a2prime.

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=N_d1*N_d2;
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
d2ind_vec=repelem((1:1:N_d2)',N_d1,1);

level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% j=N_j (terminal)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_z, n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
    [~,maxindex1]=max(ReturnMatrix_ii,[],2);
    [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_z,N_e]),[],1);

    curraindex=repmat(level1ii',N_a2*N_a3,1) ...
             + N_a1   *repmat(repelem((0:N_a2-1)',vfoptions.level1n,1),N_a3,1) ...
             + N_a1*N_a2*repelem((0:N_a3-1)',vfoptions.level1n*N_a2,1);
    dind   =rem(maxindex2-1,N_d)+1;
    a1pind =rem(floor((maxindex2-1)/N_d),N_a1)+1;
    a2pind =floor((maxindex2-1)/(N_d*N_a1))+1;
    Vhat(curraindex,:,:,N_j)       =shiftdim(Vtempii,1);
    Policy(1,curraindex,:,:,N_j)   =dind;
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
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_z, n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            Vhat(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
            dind      =rem(maxindex-1,N_d)+1;
            a1localind=rem(floor((maxindex-1)/N_d),maxgap(ii)+1)+1;
            a2pind    =floor((maxindex-1)/(N_d*(maxgap(ii)+1)))+1;
            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
            loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*zind + N_d*N_a2*N_a2*N_a3*N_z*eind;
            a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
            Policy(1,curraindex,:,:,N_j)=dind;
            Policy(2,curraindex,:,:,N_j)=a1prime_rec;
            Policy(3,curraindex,:,:,N_j)=a2pind;
        else
            loweredge=maxindex1(:,1,:,ii,:,:,:,:);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_z, n_e, d_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
            [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
            Vhat(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
            dind   =rem(maxindex-1,N_d)+1;
            a2pind =floor((maxindex-1)/N_d)+1;
            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
            loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*zind + N_d*N_a2*N_a2*N_a3*N_z*eind;
            Policy(1,curraindex,:,:,N_j)=dind;
            Policy(2,curraindex,:,:,N_j)=loweredge(loweredge_idx);
            Policy(3,curraindex,:,:,N_j)=a2pind;
        end
    end

    Vunderbar(:,:,:,N_j)=Vhat(:,:,:,N_j);

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a3, n_z, d2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,N_z);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_z,N_z]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_z]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;

    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    EV=EV.*shiftdim(pi_z_J(:,:,N_j),-2);
    EV(isnan(EV))=0;
    EV=squeeze(sum(EV,4));

    EVbase=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_z]);
    DiscountedEV_under=beta*EVbase;
    DiscountedEV_hat  =beta0beta*EVbase;

    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_z, n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);

    entireRHS_hat=ReturnMatrix_ii+repelem(DiscountedEV_hat,N_d1,1,1,1,1,1,1);
    [~,maxindex1]=max(entireRHS_hat,[],2);
    entireRHS_hat_flat=reshape(entireRHS_hat,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_z,N_e]);
    [Vtempii_hat,maxindex2]=max(entireRHS_hat_flat,[],1);

    entireRHS_under=ReturnMatrix_ii+repelem(DiscountedEV_under,N_d1,1,1,1,1,1,1);
    entireRHS_under_flat=reshape(entireRHS_under,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_z,N_e]);
    M=vfoptions.level1n*N_a2*N_a3;
    maxindexfull=maxindex2 + (N_d*N_a1*N_a2)*(0:M-1) + (N_d*N_a1*N_a2)*M*shiftdim((0:N_z-1),-1) + (N_d*N_a1*N_a2)*M*N_z*shiftdim((0:N_e-1),-2);
    Vtempii_under=entireRHS_under_flat(maxindexfull);

    curraindex=repmat(level1ii',N_a2*N_a3,1) ...
             + N_a1   *repmat(repelem((0:N_a2-1)',vfoptions.level1n,1),N_a3,1) ...
             + N_a1*N_a2*repelem((0:N_a3-1)',vfoptions.level1n*N_a2,1);
    dind   =rem(maxindex2-1,N_d)+1;
    a1pind =rem(floor((maxindex2-1)/N_d),N_a1)+1;
    a2pind =floor((maxindex2-1)/(N_d*N_a1))+1;
    Vhat(curraindex,:,:,N_j)         =shiftdim(Vtempii_hat,1);
    Vunderbar(curraindex,:,:,N_j)    =shiftdim(Vtempii_under,1);
    Policy(1,curraindex,:,:,N_j)     =dind;
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
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_z, n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_z-1),-5);
            firstdim=N_d*(maxgap(ii)+1)*N_a2;
            Mblock=level1iidiff(ii)*N_a2*N_a3;
            entireRHS_hat=reshape(ReturnMatrix_ii+DiscountedEV_hat(d2aprimez),[firstdim,Mblock,N_z,N_e]);
            entireRHS_under=reshape(ReturnMatrix_ii+DiscountedEV_under(d2aprimez),[firstdim,Mblock,N_z,N_e]);
            [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
            maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_z-1),-1) + firstdim*Mblock*N_z*shiftdim((0:N_e-1),-2);
            Vtempii_under=entireRHS_under(maxindexfull);
            Vhat(curraindex,:,:,N_j)      =shiftdim(Vtempii_hat,1);
            Vunderbar(curraindex,:,:,N_j) =shiftdim(Vtempii_under,1);
            dind      =rem(maxindex-1,N_d)+1;
            a1localind=rem(floor((maxindex-1)/N_d),maxgap(ii)+1)+1;
            a2pind    =floor((maxindex-1)/(N_d*(maxgap(ii)+1)))+1;
            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
            loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*zind + N_d*N_a2*N_a2*N_a3*N_z*eind;
            a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
            Policy(1,curraindex,:,:,N_j)=dind;
            Policy(2,curraindex,:,:,N_j)=a1prime_rec;
            Policy(3,curraindex,:,:,N_j)=a2pind;
        else
            loweredge=maxindex1(:,1,:,ii,:,:,:,:);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_z, n_e, d_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_z-1),-5);
            firstdim=N_d*1*N_a2;
            Mblock=level1iidiff(ii)*N_a2*N_a3;
            entireRHS_hat=reshape(ReturnMatrix_ii+DiscountedEV_hat(d2aprimez),[firstdim,Mblock,N_z,N_e]);
            entireRHS_under=reshape(ReturnMatrix_ii+DiscountedEV_under(d2aprimez),[firstdim,Mblock,N_z,N_e]);
            [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
            maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_z-1),-1) + firstdim*Mblock*N_z*shiftdim((0:N_e-1),-2);
            Vtempii_under=entireRHS_under(maxindexfull);
            Vhat(curraindex,:,:,N_j)      =shiftdim(Vtempii_hat,1);
            Vunderbar(curraindex,:,:,N_j) =shiftdim(Vtempii_under,1);
            dind   =rem(maxindex-1,N_d)+1;
            a2pind =floor((maxindex-1)/N_d)+1;
            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
            loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*zind + N_d*N_a2*N_a2*N_a3*N_z*eind;
            Policy(1,curraindex,:,:,N_j)=dind;
            Policy(2,curraindex,:,:,N_j)=loweredge(loweredge_idx);
            Policy(3,curraindex,:,:,N_j)=a2pind;
        end
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
    [a3primeIndex,a3primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a3, n_z, d2_gridvals, a3_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec,2);

    a1_col =repmat(repelem((1:N_a1)',N_d2,1),N_a2,1);
    a2_col =repelem((0:N_a2-1)',N_d2*N_a1,1);
    a3pIdx_repd=repmat(a3primeIndex,N_a1*N_a2,1,1);
    aprimeIndex     =a1_col + N_a1*a2_col + N_a1*N_a2*(a3pIdx_repd-1);
    aprimeplus1Index=a1_col + N_a1*a2_col + N_a1*N_a2*a3pIdx_repd;
    aprimeProbs=repmat(a3primeProbs,N_a1*N_a2,1,1,N_z);

    Vlower=reshape(EVpre(aprimeIndex(:),:),    [N_d2*N_a1*N_a2,N_a3,N_z,N_z]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1*N_a2,N_a3,N_z,N_z]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;

    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    EV=EV.*shiftdim(pi_z_J(:,:,jj),-2);
    EV(isnan(EV))=0;
    EV=squeeze(sum(EV,4));

    EVbase=reshape(EV,[N_d2,N_a1,N_a2,1,1,N_a3,N_z]);
    DiscountedEV_under=beta*EVbase;
    DiscountedEV_hat  =beta0beta*EVbase;

    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_z, n_e, d_gridvals, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);

    entireRHS_hat=ReturnMatrix_ii+repelem(DiscountedEV_hat,N_d1,1,1,1,1,1,1);
    [~,maxindex1]=max(entireRHS_hat,[],2);
    entireRHS_hat_flat=reshape(entireRHS_hat,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_z,N_e]);
    [Vtempii_hat,maxindex2]=max(entireRHS_hat_flat,[],1);

    entireRHS_under=ReturnMatrix_ii+repelem(DiscountedEV_under,N_d1,1,1,1,1,1,1);
    entireRHS_under_flat=reshape(entireRHS_under,[N_d*N_a1*N_a2,vfoptions.level1n*N_a2*N_a3,N_z,N_e]);
    M=vfoptions.level1n*N_a2*N_a3;
    maxindexfull=maxindex2 + (N_d*N_a1*N_a2)*(0:M-1) + (N_d*N_a1*N_a2)*M*shiftdim((0:N_z-1),-1) + (N_d*N_a1*N_a2)*M*N_z*shiftdim((0:N_e-1),-2);
    Vtempii_under=entireRHS_under_flat(maxindexfull);

    curraindex=repmat(level1ii',N_a2*N_a3,1) ...
             + N_a1   *repmat(repelem((0:N_a2-1)',vfoptions.level1n,1),N_a3,1) ...
             + N_a1*N_a2*repelem((0:N_a3-1)',vfoptions.level1n*N_a2,1);
    dind   =rem(maxindex2-1,N_d)+1;
    a1pind =rem(floor((maxindex2-1)/N_d),N_a1)+1;
    a2pind =floor((maxindex2-1)/(N_d*N_a1))+1;
    Vhat(curraindex,:,:,jj)        =shiftdim(Vtempii_hat,1);
    Vunderbar(curraindex,:,:,jj)   =shiftdim(Vtempii_under,1);
    Policy(1,curraindex,:,:,jj)    =dind;
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
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_z, n_e, d_gridvals, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            d2aprimez=d2ind_vec + N_d2*(a1primeindexes-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_z-1),-5);
            firstdim=N_d*(maxgap(ii)+1)*N_a2;
            Mblock=level1iidiff(ii)*N_a2*N_a3;
            entireRHS_hat=reshape(ReturnMatrix_ii+DiscountedEV_hat(d2aprimez),[firstdim,Mblock,N_z,N_e]);
            entireRHS_under=reshape(ReturnMatrix_ii+DiscountedEV_under(d2aprimez),[firstdim,Mblock,N_z,N_e]);
            [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
            maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_z-1),-1) + firstdim*Mblock*N_z*shiftdim((0:N_e-1),-2);
            Vtempii_under=entireRHS_under(maxindexfull);
            Vhat(curraindex,:,:,jj)      =shiftdim(Vtempii_hat,1);
            Vunderbar(curraindex,:,:,jj) =shiftdim(Vtempii_under,1);
            dind      =rem(maxindex-1,N_d)+1;
            a1localind=rem(floor((maxindex-1)/N_d),maxgap(ii)+1)+1;
            a2pind    =floor((maxindex-1)/(N_d*(maxgap(ii)+1)))+1;
            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
            loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*zind + N_d*N_a2*N_a2*N_a3*N_z*eind;
            a1prime_rec=a1localind+loweredge(loweredge_idx)-1;
            Policy(1,curraindex,:,:,jj)=dind;
            Policy(2,curraindex,:,:,jj)=a1prime_rec;
            Policy(3,curraindex,:,:,jj)=a2pind;
        else
            loweredge=maxindex1(:,1,:,ii,:,:,:,:);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_z, n_e, d_gridvals, a1_grid(loweredge), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            d2aprimez=d2ind_vec + N_d2*(loweredge-1) + N_d2*N_a1*shiftdim((0:1:N_a2-1),-1) + N_d2*N_a1*N_a2*shiftdim((0:1:N_a3-1),-4) + N_d2*N_a1*N_a2*N_a3*shiftdim((0:1:N_z-1),-5);
            firstdim=N_d*1*N_a2;
            Mblock=level1iidiff(ii)*N_a2*N_a3;
            entireRHS_hat=reshape(ReturnMatrix_ii+DiscountedEV_hat(d2aprimez),[firstdim,Mblock,N_z,N_e]);
            entireRHS_under=reshape(ReturnMatrix_ii+DiscountedEV_under(d2aprimez),[firstdim,Mblock,N_z,N_e]);
            [Vtempii_hat,maxindex]=max(entireRHS_hat,[],1);
            maxindexfull=maxindex + firstdim*(0:Mblock-1) + firstdim*Mblock*shiftdim((0:N_z-1),-1) + firstdim*Mblock*N_z*shiftdim((0:N_e-1),-2);
            Vtempii_under=entireRHS_under(maxindexfull);
            Vhat(curraindex,:,:,jj)      =shiftdim(Vtempii_hat,1);
            Vunderbar(curraindex,:,:,jj) =shiftdim(Vtempii_under,1);
            dind   =rem(maxindex-1,N_d)+1;
            a2pind =floor((maxindex-1)/N_d)+1;
            a2ind_flat=repmat(repelem((0:N_a2-1),1,level1iidiff(ii)),1,N_a3);
            a3ind_flat=repelem((0:N_a3-1),1,level1iidiff(ii)*N_a2);
            loweredge_idx=dind + N_d*(a2pind-1) + N_d*N_a2*a2ind_flat + N_d*N_a2*N_a2*a3ind_flat + N_d*N_a2*N_a2*N_a3*zind + N_d*N_a2*N_a2*N_a3*N_z*eind;
            Policy(1,curraindex,:,:,jj)=dind;
            Policy(2,curraindex,:,:,jj)=loweredge(loweredge_idx);
            Policy(3,curraindex,:,:,jj)=a2pind;
        end
    end
end


end
