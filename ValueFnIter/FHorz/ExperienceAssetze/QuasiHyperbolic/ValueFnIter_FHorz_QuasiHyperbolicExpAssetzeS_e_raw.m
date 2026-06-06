function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeS_e_raw(n_d1,n_d2,n_a1,n_a2,n_z,n_e,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Sophisticated QH + ExpAssetze (z+e dep aprimeFn), baseline (no DC, no GI), no d1.

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=N_d1*N_d2;
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);
N_e=prod(n_e);

Vhat=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray');

a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

d2ind_vec=repelem((1:1:N_d2)',N_d1,1);

if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory==2
    special_n_z=ones(1,length(n_z));
    special_n_e=ones(1,length(n_e));
end

%% j=N_j (terminal)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1, n_d2, n_a1, n_a1,n_a2, n_z, n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        Vhat(:,:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,:,N_j)=shiftdim(maxindex,1);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1, n_d2, n_a1, n_a1,n_a2, n_z, special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0,0);
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            Vhat(:,:,e_c,N_j)=shiftdim(Vtemp,1);
            Policy(:,:,e_c,N_j)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1, n_d2, n_a1, n_a1,n_a2, special_n_z, special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,0,0);
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                Vhat(:,z_c,e_c,N_j)=Vtemp;
                Policy(:,z_c,e_c,N_j)=maxindex;
            end
        end
    end
    Vunderbar(:,:,:,N_j)=Vhat(:,:,:,N_j);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EVpre=sum(shiftdim(pi_e_J(:,N_j),-2).*reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]),3);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzeFnMatrix(aprimeFn, n_d2, n_a2, n_z, n_e, d2_gridvals, a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2,N_z,N_e)+N_a1*repmat(a2primeIndex-1,N_a1,1,1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2,N_z,N_e)+N_a1*repmat(a2primeIndex,N_a1,1,1,1);
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1,1,N_z);

    Vlower=reshape(EVpre(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;

    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    EV=EV.*reshape(pi_z_J(:,:,N_j),[1,1,N_z,1,N_z]);
    EV(isnan(EV))=0;
    EV=reshape(sum(EV,5),[N_d2*N_a1,N_a2,N_z,N_e]);

    entireEV=repelem(EV,1,N_a1,1,1);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1, n_d2, n_a1, n_a1,n_a2, n_z, n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0);
        entireRHS_hat=ReturnMatrix+repelem(beta0beta*entireEV,N_d1,1,1,1);
        [Vtemp,maxindex]=max(entireRHS_hat,[],1);
        Vhat(:,:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,:,N_j)=shiftdim(maxindex,1);
        entireRHS_under=ReturnMatrix+repelem(beta*entireEV,N_d1,1,1,1);
        maxindexfull=maxindex+N_d*N_a1*(0:1:N_a-1)+shiftdim(N_d*N_a1*N_a*(0:1:N_z-1),-1)+shiftdim(N_d*N_a1*N_a*N_z*(0:1:N_e-1),-2);
        Vunderbar(:,:,:,N_j)=shiftdim(entireRHS_under(maxindexfull),1);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            entireEV_e=entireEV(:,:,:,e_c);
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1, n_d2, n_a1, n_a1,n_a2, n_z, special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0,0);
            entireRHS_hat_e=ReturnMatrix_e+repelem(beta0beta*entireEV_e,N_d1,1,1);
            [Vtemp,maxindex]=max(entireRHS_hat_e,[],1);
            Vhat(:,:,e_c,N_j)=shiftdim(Vtemp,1);
            Policy(:,:,e_c,N_j)=shiftdim(maxindex,1);
            entireRHS_under_e=ReturnMatrix_e+repelem(beta*entireEV_e,N_d1,1,1);
            maxindexfull=maxindex+N_d*N_a1*(0:1:N_a-1)+shiftdim(N_d*N_a1*N_a*(0:1:N_z-1),-1);
            Vunderbar(:,:,e_c,N_j)=shiftdim(entireRHS_under_e(maxindexfull),1);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                entireEV_ze=entireEV(:,:,z_c,e_c);
                ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1, n_d2, n_a1, n_a1,n_a2, special_n_z, special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,0,0);
                entireRHS_hat_ze=ReturnMatrix_ze+repelem(beta0beta*entireEV_ze,N_d1,1);
                [Vtemp,maxindex]=max(entireRHS_hat_ze,[],1);
                Vhat(:,z_c,e_c,N_j)=Vtemp;
                Policy(:,z_c,e_c,N_j)=maxindex;
                entireRHS_under_ze=ReturnMatrix_ze+repelem(beta*entireEV_ze,N_d1,1);
                maxindexfull=maxindex+N_d*N_a1*(0:1:N_a-1);
                Vunderbar(:,z_c,e_c,N_j)=entireRHS_under_ze(maxindexfull);
            end
        end
    end
end

%% Backward induction
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

    EVpre=sum(shiftdim(pi_e_J(:,jj),-2).*Vunderbar(:,:,:,jj+1),3);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzeFnMatrix(aprimeFn, n_d2, n_a2, n_z, n_e, d2_gridvals, a2_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), aprimeFnParamsVec,2);

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2,N_z,N_e)+N_a1*repmat(a2primeIndex-1,N_a1,1,1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2,N_z,N_e)+N_a1*repmat(a2primeIndex,N_a1,1,1,1);
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1,1,N_z);

    Vlower=reshape(EVpre(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;

    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    EV=EV.*reshape(pi_z_J(:,:,jj),[1,1,N_z,1,N_z]);
    EV(isnan(EV))=0;
    EV=reshape(sum(EV,5),[N_d2*N_a1,N_a2,N_z,N_e]);

    entireEV=repelem(EV,1,N_a1,1,1);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1, n_d2, n_a1, n_a1,n_a2, n_z, n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,0,0);
        entireRHS_hat=ReturnMatrix+repelem(beta0beta*entireEV,N_d1,1,1,1);
        [Vtemp,maxindex]=max(entireRHS_hat,[],1);
        Vhat(:,:,:,jj)=shiftdim(Vtemp,1);
        Policy(:,:,:,jj)=shiftdim(maxindex,1);
        entireRHS_under=ReturnMatrix+repelem(beta*entireEV,N_d1,1,1,1);
        maxindexfull=maxindex+N_d*N_a1*(0:1:N_a-1)+shiftdim(N_d*N_a1*N_a*(0:1:N_z-1),-1)+shiftdim(N_d*N_a1*N_a*N_z*(0:1:N_e-1),-2);
        Vunderbar(:,:,:,jj)=shiftdim(entireRHS_under(maxindexfull),1);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            entireEV_e=entireEV(:,:,:,e_c);
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1, n_d2, n_a1, n_a1,n_a2, n_z, special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,0,0);
            entireRHS_hat_e=ReturnMatrix_e+repelem(beta0beta*entireEV_e,N_d1,1,1);
            [Vtemp,maxindex]=max(entireRHS_hat_e,[],1);
            Vhat(:,:,e_c,jj)=shiftdim(Vtemp,1);
            Policy(:,:,e_c,jj)=shiftdim(maxindex,1);
            entireRHS_under_e=ReturnMatrix_e+repelem(beta*entireEV_e,N_d1,1,1);
            maxindexfull=maxindex+N_d*N_a1*(0:1:N_a-1)+shiftdim(N_d*N_a1*N_a*(0:1:N_z-1),-1);
            Vunderbar(:,:,e_c,jj)=shiftdim(entireRHS_under_e(maxindexfull),1);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                entireEV_ze=entireEV(:,:,z_c,e_c);
                ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1, n_d2, n_a1, n_a1,n_a2, special_n_z, special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,0,0);
                entireRHS_hat_ze=ReturnMatrix_ze+repelem(beta0beta*entireEV_ze,N_d1,1);
                [Vtemp,maxindex]=max(entireRHS_hat_ze,[],1);
                Vhat(:,z_c,e_c,jj)=Vtemp;
                Policy(:,z_c,e_c,jj)=maxindex;
                entireRHS_under_ze=ReturnMatrix_ze+repelem(beta*entireEV_ze,N_d1,1);
                maxindexfull=maxindex+N_d*N_a1*(0:1:N_a-1);
                Vunderbar(:,z_c,e_c,jj)=entireRHS_under_ze(maxindexfull);
            end
        end
    end
end

Policy=shiftdim(Policy,-1);

end
