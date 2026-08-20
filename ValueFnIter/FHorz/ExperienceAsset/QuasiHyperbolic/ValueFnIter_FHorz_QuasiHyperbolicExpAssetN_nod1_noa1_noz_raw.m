function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetN_nod1_noa1_noz_raw(n_d2,n_a2,N_j, d2_gridvals, a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d2=prod(n_d2);
N_a2=prod(n_a2);
N_a=N_a2;

Valt=zeros(N_a,N_j,'gpuArray');
Vtilde=zeros(N_a,N_j,'gpuArray');
Policyalt=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z
Policy=zeros(N_a,N_j,'gpuArray');

%%
d2_gridvals=gpuArray(d2_gridvals);
a2_grid=gpuArray(a2_grid);

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')

    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_noz(ReturnFn,n_d2, n_a2, d2_gridvals, a2_grid, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    Valt(:,N_j)=Vtemp;
    Policyalt(:,N_j)=maxindex;

    % Terminal period: no continuation, so the QH-perceived objects equal the exponential ones
    Vtilde(:,N_j)=Valt(:,N_j);
    Policy(:,N_j)=Policyalt(:,N_j);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,1]); % First, switch V_Jplus1 into Kron form

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2*N_a2,1], whereas aprimeProbs is [N_d2,N_a2]

    Vlower=reshape(EVpre(a2primeIndex),[N_d2,N_a2]);
    Vupper=reshape(EVpre(a2primeIndex+1),[N_d2,N_a2]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    a2primeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=a2primeProbs.*Vlower+(1-a2primeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)
    EV(isnan(EV))=0; % NaN from 0*(-Inf) at skipinterp positions; treat as zero contribution

    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_noz(ReturnFn,n_d2, n_a2, d2_gridvals, a2_grid, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

    entireRHS_alt=ReturnMatrix+beta*EV;
    [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
    Valt(:,N_j)=shiftdim(Vtemp_alt,1);
    Policyalt(:,N_j)=shiftdim(maxindex_alt,1);
    entireRHS_tilde=ReturnMatrix+beta0beta*EV;
    [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
    Vtilde(:,N_j)=shiftdim(Vtemp_tilde,1);
    Policy(:,N_j)=shiftdim(maxindex_tilde,1);

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
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2*N_a2,1], whereas aprimeProbs is [N_d2,N_a2]

    Vlower=reshape(Valt(a2primeIndex,jj+1),[N_d2,N_a2]);
    Vupper=reshape(Valt(a2primeIndex+1,jj+1),[N_d2,N_a2]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    a2primeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=a2primeProbs.*Vlower+(1-a2primeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)
    EV(isnan(EV))=0; % NaN from 0*(-Inf) at skipinterp positions; treat as zero contribution

    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_noz(ReturnFn,n_d2, n_a2, d2_gridvals, a2_grid, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command

    entireRHS_alt=ReturnMatrix+beta*EV;
    [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
    Valt(:,jj)=shiftdim(Vtemp_alt,1);
    Policyalt(:,jj)=shiftdim(maxindex_alt,1);
    entireRHS_tilde=ReturnMatrix+beta0beta*EV;
    [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
    Vtilde(:,jj)=shiftdim(Vtemp_tilde,1);
    Policy(:,jj)=shiftdim(maxindex_tilde,1);

end

%%
Policy=shiftdim(Policy,-1);
Policyalt=shiftdim(Policyalt,-1);


end
