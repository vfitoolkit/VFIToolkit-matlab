function [V,Policy2]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_noz_raw(V,n_d1,n_d2,n_a1,n_a2,N_j, d_gridvals,d2_gridvals,a1_gridvals,a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;


%% First, create the big 'next period (of transition path) expected value fn.
% fastOLG will be N_d*N_aprime by N_a*N_j*N_z (note: N_aprime is just equal to N_a)

DiscountFactorParamsVec=CreateAgeMatrixFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec,2);
DiscountFactorParamsVec=shiftdim(DiscountFactorParamsVec,-2);

% Create a matrix containing all the return function parameters (in order).
% Each column will be a specific parameter with the values at every age.
ReturnFnParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, ReturnFnParamNames,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

if vfoptions.EVpre==0
    aprimeFnParamsVec=CreateAgeMatrixFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_J(aprimeFn, n_d2, n_a2, N_j, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_j], whereas aprimeProbs is [N_d2,N_a2,N_j]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,1,1)+N_a1*repmat((a2primeIndex-1),N_a1,1,1); % [N_d2*N_a1,N_a2,N_j], autofill the [1,N_a1,N_j] dimensions for the first part
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,1,1)+N_a1*repmat(a2primeIndex,N_a1,1,1); % [N_d2*N_a1,N_a2,N_j], autofill the [1,N_a1,N_j] dimensions for the first part
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1);  % [N_d2*N_a1,N_a2,N_j]

    EVpre=[V(N_a+1:end); zeros(N_a,1,'gpuArray')]; % I use zeros in j=N_j so that can just use pi_z_J to create expectations

    % Need to add the indexes for j to the aprimeIndex, remember fastOLG so V is (a,j)-by-1
    Vlower=reshape(EVpre(aprimeIndex+shiftdim(N_a*gpuArray(0:1:N_j-1),-1)),[N_d2*N_a1,N_a2,N_j]);
    Vupper=reshape(EVpre(aprimeplus1Index+shiftdim(N_a*gpuArray(0:1:N_j-1),-1)),[N_d2*N_a1,N_a2,N_j]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terps of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,N_j)
    % Already applied the probabilities from interpolating onto grid
    
    EV=reshape(sum(EV,4),[N_d2*N_a1,N_a2,N_j]); % (aprime,1,j), 2nd dim will be autofilled with a
elseif vfoptions.EVpre==1
    % This is used for 'Matched Expecations Path'
    aprimeFnParamsVec=CreateAgeMatrixFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_J(aprimeFn, n_d2, n_a2, N_j, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_j], whereas aprimeProbs is [N_d2,N_a2,N_j]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,1,1)+N_a1*repmat((a2primeIndex-1),N_a1,1,1); % [N_d2*N_a1,N_a2,N_j], autofill the [1,N_a1,N_j] dimensions for the first part
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,1,1)+N_a1*repmat(a2primeIndex,N_a1,1,1); % [N_d2*N_a1,N_a2,N_j], autofill the [1,N_a1,N_j] dimensions for the first part
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1);  % [N_d2*N_a1,N_a2,N_j]

    % Need to add the indexes for j to the aprimeIndex, remember fastOLG so V is (a,j)-by-1
    Vlower=reshape(V(aprimeIndex+shiftdim(N_a*gpuArray(0:1:N_j-1),-1)),[N_d2*N_a1,N_a2,N_j]);
    Vupper=reshape(V(aprimeplus1Index+shiftdim(N_a*gpuArray(0:1:N_j-1),-1)),[N_d2*N_a1,N_a2,N_j]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terps of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,N_j)
    % Already applied the probabilities from interpolating onto grid
    
    EV=reshape(sum(EV,4),[N_d2*N_a1,N_a2,N_j]); % (aprime,1,j), 2nd dim will be autofilled with a
end

DiscountedEV=DiscountFactorParamsVec.*repelem(EV,N_d1,N_a1,1,1);

if vfoptions.lowmemory==0
    ReturnMatrix=CreateReturnFnMatrix_Case1_fastOLG_ExpAsset_Disc_Par2_noz(ReturnFn, n_d1, n_d2, n_a1, n_a1,n_a2,N_j, d_gridvals, a1_gridvals, a1_gridvals, a2_grid, ReturnFnParamsAgeMatrix,0,0);

    entireRHS=ReturnMatrix+DiscountedEV;

    % Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);

    V=reshape(Vtemp,[N_a*N_j,1]);
    Policy=shiftdim(maxindex,1);
end


%% fastOLG with z, so need to output to take certain shapes
% V=reshape(V,[N_a*N_j,1]);
% Policy=reshape(Policy,[N_a,N_j,1]);

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point



end
