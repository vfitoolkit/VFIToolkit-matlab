function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_noa1_raw(n_d1,n_d2,n_a2,n_z,N_j, d_gridvals, d2_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Naive quasi-hyperbolic + experienceassetz (baseline, with z, no e).
% Hybridises ValueFnIter_FHorz_ExpAssetz_raw (EV via aprimeFn for a2) and
% ValueFnIter_FHorz_QuasiHyperbolicN_raw (dual V_std / V_tilde pass).
%
% Vtilde, Policy : QH-perceived value and QH-greedy policy
% Valt, Policyalt : standard exponential discounter's value and policy

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=N_d1*N_d2;
N_a2=prod(n_a2);
N_a=N_a2;
N_z=prod(n_z);

Valt=zeros(N_a,N_z,N_j,'gpuArray');
Vtilde=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray');
Policyalt=zeros(N_a,N_z,N_j,'gpuArray');


if vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
end

%% j=N_j (terminal)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d2], n_a2, n_z, d_gridvals, a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        Valt(:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,N_j)=shiftdim(maxindex,1);
        Policyalt(:,:,N_j)=shiftdim(maxindex,1);
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d2], n_a2, special_n_z, d_gridvals, a2_grid, z_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            Valt(:,z_c,N_j)=Vtemp;
            Policy(:,z_c,N_j)=maxindex;
            Policyalt(:,z_c,N_j)=maxindex;
        end
    end
    Vtilde(:,:,N_j)=Valt(:,:,N_j);
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

    entireEV=repelem(EV,N_d1,1);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d2], n_a2, n_z, d_gridvals, a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec);

        % Pass 1: std-discounter Valt and Policyalt
        entireRHS=ReturnMatrix+beta*entireEV;
        [Vtemp,maxindexalt]=max(entireRHS,[],1);
        Valt(:,:,N_j)=shiftdim(Vtemp,1);
        Policyalt(:,:,N_j)=shiftdim(maxindexalt,1);
        % Pass 2: QH-perceived Vtilde and Policy
        entireRHS=ReturnMatrix+beta0beta*entireEV;
        [Vtemp,maxindex]=max(entireRHS,[],1);
        Vtilde(:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,N_j)=shiftdim(maxindex,1);
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            entireEV_z=entireEV(:,:,z_c);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d2], n_a2, special_n_z, d_gridvals, a2_grid, z_val, ReturnFnParamsVec);

            entireRHS_z=ReturnMatrix_z+beta*entireEV_z;
            [Vtemp,maxindexalt]=max(entireRHS_z,[],1);
            Valt(:,z_c,N_j)=Vtemp;
            Policyalt(:,z_c,N_j)=maxindexalt;
            entireRHS_z=ReturnMatrix_z+beta0beta*entireEV_z;
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            Vtilde(:,z_c,N_j)=Vtemp;
            Policy(:,z_c,N_j)=maxindex;
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

    % Naive: continuation uses Valt (the std-discounter future value), since
    % naive expects future selves to follow exponential discounting.
    EVpre=Valt(:,:,jj+1);

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

    entireEV=repelem(EV,N_d1,1);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d2], n_a2, n_z, d_gridvals, a2_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec);

        entireRHS=ReturnMatrix+beta*entireEV;
        [Vtemp,maxindexalt]=max(entireRHS,[],1);
        Valt(:,:,jj)=shiftdim(Vtemp,1);
        Policyalt(:,:,jj)=shiftdim(maxindexalt,1);
        entireRHS=ReturnMatrix+beta0beta*entireEV;
        [Vtemp,maxindex]=max(entireRHS,[],1);
        Vtilde(:,:,jj)=shiftdim(Vtemp,1);
        Policy(:,:,jj)=shiftdim(maxindex,1);
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            entireEV_z=entireEV(:,:,z_c);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d2], n_a2, special_n_z, d_gridvals, a2_grid, z_val, ReturnFnParamsVec);

            entireRHS_z=ReturnMatrix_z+beta*entireEV_z;
            [Vtemp,maxindexalt]=max(entireRHS_z,[],1);
            Valt(:,z_c,jj)=Vtemp;
            Policyalt(:,z_c,jj)=maxindexalt;
            entireRHS_z=ReturnMatrix_z+beta0beta*entireEV_z;
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            Vtilde(:,z_c,jj)=Vtemp;
            Policy(:,z_c,jj)=maxindex;
        end
    end
end

Policy=shiftdim(Policy,-1);
Policyalt=shiftdim(Policyalt,-1);

end
