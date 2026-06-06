function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_nod1_e_raw(n_d2,n_a1,n_a2,n_z,n_e,N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Sophisticated QH + experienceassetz, no-d1 variant.

N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);
N_e=prod(n_e);

Vhat=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray');

a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

if vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
end

%% j=N_j (terminal)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0, n_d2, n_a1, n_a1,n_a2, n_z, n_e, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        Vhat(:,:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,:,N_j)=shiftdim(maxindex,1);
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0, n_d2, n_a1, n_a1,n_a2, special_n_z, n_e, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            Vhat(:,z_c,N_j)=Vtemp;
            Policy(:,z_c,N_j)=maxindex;
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
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a2, n_z, d2_gridvals, a2_grid, z_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2,N_z)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2,N_z)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1,N_z);

    Vlower=reshape(EVpre(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_z,N_z]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_z,N_z]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;

    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    EV=EV.*shiftdim(pi_z_J(:,:,N_j),-2);
    EV(isnan(EV))=0;
    EV=squeeze(sum(EV,4));

    entireEV=repelem(EV,1,N_a1,1);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0, n_d2, n_a1, n_a1,n_a2, n_z, n_e, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0);

        entireRHS_hat=ReturnMatrix+beta0beta*entireEV;
        [Vtemp,maxindex]=max(entireRHS_hat,[],1);
        Vhat(:,:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,:,N_j)=shiftdim(maxindex,1);
        entireRHS_std=ReturnMatrix+beta*entireEV;
        maxindexfull=maxindex+N_d2*N_a1*(0:1:N_a-1)+shiftdim(N_d2*N_a1*N_a*(0:1:N_z-1),-1)+shiftdim(N_d2*N_a1*N_a*N_z*(0:1:N_e-1),-2);
        Vunderbar(:,:,:,N_j)=shiftdim(entireRHS_std(maxindexfull),1);
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            entireEV_z=entireEV(:,:,z_c);
            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0, n_d2, n_a1, n_a1,n_a2, special_n_z, n_e, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0);

            entireRHS_hat=ReturnMatrix_z+beta0beta*entireEV_z;
            [Vtemp,maxindex]=max(entireRHS_hat,[],1);
            Vhat(:,z_c,N_j)=Vtemp;
            Policy(:,z_c,N_j)=maxindex;
            entireRHS_std=ReturnMatrix_z+beta*entireEV_z;
            maxindexfull=maxindex+N_d2*N_a1*(0:1:N_a-1);
            Vunderbar(:,z_c,N_j)=entireRHS_std(maxindexfull);
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
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a2, n_z, d2_gridvals, a2_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec,2);

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2,N_z)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2,N_z)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1,N_z);

    Vlower=reshape(EVpre(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_z,N_z]);
    Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_z,N_z]);
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0;

    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    EV=EV.*shiftdim(pi_z_J(:,:,jj),-2);
    EV(isnan(EV))=0;
    EV=squeeze(sum(EV,4));

    entireEV=repelem(EV,1,N_a1,1);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0, n_d2, n_a1, n_a1,n_a2, n_z, n_e, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,0,0);

        entireRHS_hat=ReturnMatrix+beta0beta*entireEV;
        [Vtemp,maxindex]=max(entireRHS_hat,[],1);
        Vhat(:,:,:,jj)=shiftdim(Vtemp,1);
        Policy(:,:,:,jj)=shiftdim(maxindex,1);
        entireRHS_std=ReturnMatrix+beta*entireEV;
        maxindexfull=maxindex+N_d2*N_a1*(0:1:N_a-1)+shiftdim(N_d2*N_a1*N_a*(0:1:N_z-1),-1)+shiftdim(N_d2*N_a1*N_a*N_z*(0:1:N_e-1),-2);
        Vunderbar(:,:,:,jj)=shiftdim(entireRHS_std(maxindexfull),1);
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            entireEV_z=entireEV(:,:,z_c);
            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0, n_d2, n_a1, n_a1,n_a2, special_n_z, n_e, d2_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0);

            entireRHS_hat=ReturnMatrix_z+beta0beta*entireEV_z;
            [Vtemp,maxindex]=max(entireRHS_hat,[],1);
            Vhat(:,z_c,jj)=Vtemp;
            Policy(:,z_c,jj)=maxindex;
            entireRHS_std=ReturnMatrix_z+beta*entireEV_z;
            maxindexfull=maxindex+N_d2*N_a1*(0:1:N_a-1);
            Vunderbar(:,z_c,jj)=entireRHS_std(maxindexfull);
        end
    end
end

Policy=shiftdim(Policy,-1);

end
