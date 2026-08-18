function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_nod1_noa1_raw(n_d2,n_a2,n_z,N_j, d2_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Sophisticated QH + experienceassetz, no-d1 variant.

N_d2=prod(n_d2);
N_a2=prod(n_a2);
N_a=N_a2;
N_z=prod(n_z);

Vhat=zeros(N_a,N_z,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray');


if vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
end

%% j=N_j (terminal)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d2, n_a2, n_z, d2_gridvals, a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        Vhat(:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,N_j)=shiftdim(maxindex,1);
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d2, n_a2, special_n_z, d2_gridvals, a2_grid, z_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            Vhat(:,z_c,N_j)=Vtemp;
            Policy(:,z_c,N_j)=maxindex;
        end
    end
    Vunderbar(:,:,N_j)=Vhat(:,:,N_j);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_z]);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a2, n_z, d2_gridvals, a2_grid, z_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);


    Vlower=reshape(EVpre(a2primeIndex(:),:),[N_d2,N_a2,N_z,N_z]); % (d2,a2,z,zprime)
    Vupper=reshape(EVpre(a2primeIndex(:)+1,:),[N_d2,N_a2,N_z,N_z]);
    a2primeProbs=repmat(a2primeProbs,1,1,1,N_z); % [N_d2,N_a2,N_z,N_zprime]
    skipinterp=(Vlower==Vupper);
    a2primeProbs(skipinterp)=0;

    EV=a2primeProbs.*Vlower+(1-a2primeProbs).*Vupper;
    EV=EV.*shiftdim(pi_z_J(:,:,N_j),-2);
    EV(isnan(EV))=0;
    EV=squeeze(sum(EV,4));

    entireEV=EV;

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d2, n_a2, n_z, d2_gridvals, a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec);

        entireRHS_hat=ReturnMatrix+beta0beta*entireEV;
        [Vtemp,maxindex]=max(entireRHS_hat,[],1);
        Vhat(:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,N_j)=shiftdim(maxindex,1);
        entireRHS_std=ReturnMatrix+beta*entireEV;
        maxindexfull=maxindex+N_d2*(0:1:N_a-1)+shiftdim(N_d2*N_a*(0:1:N_z-1),-1);
        Vunderbar(:,:,N_j)=shiftdim(entireRHS_std(maxindexfull),1);
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            entireEV_z=entireEV(:,:,z_c);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d2, n_a2, special_n_z, d2_gridvals, a2_grid, z_val, ReturnFnParamsVec);

            entireRHS_hat=ReturnMatrix_z+beta0beta*entireEV_z;
            [Vtemp,maxindex]=max(entireRHS_hat,[],1);
            Vhat(:,z_c,N_j)=Vtemp;
            Policy(:,z_c,N_j)=maxindex;
            entireRHS_std=ReturnMatrix_z+beta*entireEV_z;
            maxindexfull=maxindex+N_d2*(0:1:N_a-1);
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

    EVpre=Vunderbar(:,:,jj+1);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a2, n_z, d2_gridvals, a2_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec,2);


    Vlower=reshape(EVpre(a2primeIndex(:),:),[N_d2,N_a2,N_z,N_z]); % (d2,a2,z,zprime)
    Vupper=reshape(EVpre(a2primeIndex(:)+1,:),[N_d2,N_a2,N_z,N_z]);
    a2primeProbs=repmat(a2primeProbs,1,1,1,N_z); % [N_d2,N_a2,N_z,N_zprime]
    skipinterp=(Vlower==Vupper);
    a2primeProbs(skipinterp)=0;

    EV=a2primeProbs.*Vlower+(1-a2primeProbs).*Vupper;
    EV=EV.*shiftdim(pi_z_J(:,:,jj),-2);
    EV(isnan(EV))=0;
    EV=squeeze(sum(EV,4));

    entireEV=EV;

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d2, n_a2, n_z, d2_gridvals, a2_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec);

        entireRHS_hat=ReturnMatrix+beta0beta*entireEV;
        [Vtemp,maxindex]=max(entireRHS_hat,[],1);
        Vhat(:,:,jj)=shiftdim(Vtemp,1);
        Policy(:,:,jj)=shiftdim(maxindex,1);
        entireRHS_std=ReturnMatrix+beta*entireEV;
        maxindexfull=maxindex+N_d2*(0:1:N_a-1)+shiftdim(N_d2*N_a*(0:1:N_z-1),-1);
        Vunderbar(:,:,jj)=shiftdim(entireRHS_std(maxindexfull),1);
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            entireEV_z=entireEV(:,:,z_c);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d2, n_a2, special_n_z, d2_gridvals, a2_grid, z_val, ReturnFnParamsVec);

            entireRHS_hat=ReturnMatrix_z+beta0beta*entireEV_z;
            [Vtemp,maxindex]=max(entireRHS_hat,[],1);
            Vhat(:,z_c,jj)=Vtemp;
            Policy(:,z_c,jj)=maxindex;
            entireRHS_std=ReturnMatrix_z+beta*entireEV_z;
            maxindexfull=maxindex+N_d2*(0:1:N_a-1);
            Vunderbar(:,z_c,jj)=entireRHS_std(maxindexfull);
        end
    end
end

Policy=shiftdim(Policy,-1);

end
