function [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAssetz_nod1_e_raw(V,n_d2,n_a1,n_a2,n_z,n_e,N_j, d2_gridvals,a1_gridvals,a2_grid, z_gridvals_J,e_gridvals_J, pi_z_J,pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% fastOLG just means parallelize over "age" (j)
% fastOLG is done as (a,j,z,e), rather than standard (a,z,e,j)
% V is (a,j)-by-z-by-e
% pi_z_J is (j,z',z) for fastOLG
% z_gridvals_J is (j,N_z,l_z) for fastOLG
% pi_e_J is (j,e') for fastOLG
% e_gridvals_J is (j,N_e,l_e) for fastOLG

N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);
N_e=prod(n_e);

z_gridvals_J_zlj=permute(z_gridvals_J,[2,3,1]); % [N_z,l_z,N_j] for CreateExperienceAssetzFnMatrix_J
z_gridvals_J=shiftdim(z_gridvals_J,-4); % [1,1,1,1,N_j,N_z,l_z]
e_gridvals_J=reshape(e_gridvals_J,[1,1,1,1,1,N_j,N_e,length(n_e)]); % [1,1,1,1,1,N_j,N_e,l_e]; ExogShockSetup leaves it as (N_j,1,N_e,l_e), so reshape drops the middle singleton so N_e/l_e land on dims 7/8 (slicing assumes this)

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
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzFnMatrix_J(aprimeFn, n_d2, n_a2, n_z, N_j, d2_gridvals, a2_grid, z_gridvals_J_zlj, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_z,N_j], whereas aprimeProbs is [N_d2,N_a2,N_z,N_j]   (N_z here is the current z)
    a2primeIndex=permute(a2primeIndex,[1,2,4,3]); % [N_d2,N_a2,N_j,N_z]
    a2primeProbs=permute(a2primeProbs,[1,2,4,3]); % [N_d2,N_a2,N_j,N_z]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,1,1)+N_a1*repmat((a2primeIndex-1),N_a1,1,1,1); % [N_d2*N_a1,N_a2,N_j,N_z], autofill the [1,N_a1,N_j,N_z] dimensions for the first part
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,1,1)+N_a1*repmat(a2primeIndex,N_a1,1,1,1); % [N_d2*N_a1,N_a2,N_j,N_z], autofill the [1,N_a1,N_j,N_z] dimensions for the first part
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1,1,N_z);  % [N_d2*N_a1,N_a2,N_j,N_z,N_zprime]    (current z dim already present, just add zprime)

    EVpre=[sum(V(N_a+1:end,:,:).*pi_e_J(1:end-N_a,:,:),3); zeros(N_a,N_z,'gpuArray')]; % pi_e_J already kron-expanded to (a-j,1,e) upstream; j=N_j zero-padded

    % Need to add the indexes for j to the aprimeIndex, remember fastOLG so V is (a,j)-by-z
    Vlower=reshape(EVpre(aprimeIndex+shiftdim(N_a*gpuArray(0:1:N_j-1),-1),:),[N_d2*N_a1,N_a2,N_j,N_z,N_z]); % (d2*a1prime,a2,j,z,zprime)
    Vupper=reshape(EVpre(aprimeplus1Index+shiftdim(N_a*gpuArray(0:1:N_j-1),-1),:),[N_d2*N_a1,N_a2,N_j,N_z,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2*a1prime,a2,N_j,z,zprime)
    % Already applied the probabilities from interpolating onto grid

    EV=EV.*shiftdim(permute(pi_z_J,[1,3,2]),-2); % pi_z_J is (j,zprime,z); permute&shift to [1,1,j,z,zprime] -- current z is dim 4
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=reshape(sum(EV,5),[N_d2*N_a1,N_a2,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a
elseif vfoptions.EVpre==1
    % This is used for 'Matched Expecations Path'
    aprimeFnParamsVec=CreateAgeMatrixFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzFnMatrix_J(aprimeFn, n_d2, n_a2, n_z, N_j, d2_gridvals, a2_grid, z_gridvals_J_zlj, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_z,N_j], whereas aprimeProbs is [N_d2,N_a2,N_z,N_j]   (N_z here is the current z)
    a2primeIndex=permute(a2primeIndex,[1,2,4,3]); % [N_d2,N_a2,N_j,N_z]
    a2primeProbs=permute(a2primeProbs,[1,2,4,3]); % [N_d2,N_a2,N_j,N_z]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,1,1)+N_a1*repmat((a2primeIndex-1),N_a1,1,1,1); % [N_d2*N_a1,N_a2,N_j,N_z], autofill the [1,N_a1,N_j,N_z] dimensions for the first part
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,1,1)+N_a1*repmat(a2primeIndex,N_a1,1,1,1); % [N_d2*N_a1,N_a2,N_j,N_z], autofill the [1,N_a1,N_j,N_z] dimensions for the first part
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1,1,N_z);  % [N_d2*N_a1,N_a2,N_j,N_z,N_zprime]    (current z dim already present, just add zprime)

    EVpre=sum(V.*pi_e_J,3);

    % Need to add the indexes for j to the aprimeIndex, remember fastOLG so V is (a,j)-by-z
    Vlower=reshape(EVpre(aprimeIndex+shiftdim(N_a*gpuArray(0:1:N_j-1),-1),:),[N_d2*N_a1,N_a2,N_j,N_z,N_z]); % (d2*a1prime,a2,j,z,zprime)
    Vupper=reshape(EVpre(aprimeplus1Index+shiftdim(N_a*gpuArray(0:1:N_j-1),-1),:),[N_d2*N_a1,N_a2,N_j,N_z,N_z]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2*a1prime,a2,N_j,z,zprime)
    % Already applied the probabilities from interpolating onto grid

    EV=EV.*shiftdim(permute(pi_z_J,[1,3,2]),-2); % pi_z_J is (j,zprime,z); permute&shift to [1,1,j,z,zprime] -- current z is dim 4
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=reshape(sum(EV,5),[N_d2*N_a1,N_a2,N_j,N_z]); % (aprime,1,j,z), 2nd dim will be autofilled with a
end

DiscountedEV=DiscountFactorParamsVec.*repelem(EV,1,N_a1,1,1);

if vfoptions.lowmemory==0
    ReturnMatrix=CreateReturnFnMatrix_fastOLG_ExpAsset_Disc_e(ReturnFn, 0, n_d2, n_a1, n_a1,n_a2, n_z,n_e,N_j, d2_gridvals, a1_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix,0,0); % Level=0, Refine=0

    entireRHS=ReturnMatrix+DiscountedEV;

    % Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);

    V=reshape(Vtemp,[N_a*N_j,N_z,N_e]);
    Policy=reshape(maxindex,[N_a*N_j,N_z,N_e]);

elseif vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e),'gpuArray');
    V=zeros(N_a*N_j,N_z,N_e,'gpuArray');
    Policy=zeros(N_a*N_j,N_z,N_e,'gpuArray');

    for e_c=1:N_e
        e_val=e_gridvals_J(:,:,:,:,:,:,e_c,:); % e_gridvals_J is [1,1,1,1,1,N_j,N_e,l_e] — e on dim 7

        ReturnMatrix_e=CreateReturnFnMatrix_fastOLG_ExpAsset_Disc_e(ReturnFn, 0, n_d2, n_a1, n_a1,n_a2, n_z,special_n_e,N_j, d2_gridvals, a1_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_val, ReturnFnParamsAgeMatrix,0,0); % Level=0, Refine=0

        entireRHS_e=ReturnMatrix_e+DiscountedEV;

        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS_e,[],1);
        V(:,:,e_c)=reshape(Vtemp,[N_a*N_j,N_z]);
        Policy(:,:,e_c)=reshape(maxindex,[N_a*N_j,N_z]);
    end
elseif vfoptions.lowmemory==2
    special_n_z=ones(1,length(n_z),'gpuArray');
    special_n_e=ones(1,length(n_e),'gpuArray');
    V=zeros(N_a*N_j,N_z,N_e,'gpuArray');
    Policy=zeros(N_a*N_j,N_z,N_e,'gpuArray');

    for z_c=1:N_z
        z_val=z_gridvals_J(:,:,:,:,:,z_c,:); % z_gridvals_J is [1,1,1,1,N_j,N_z,l_z] — z on dim 6
        DiscountedEV_z=DiscountedEV(:,:,:,z_c);

        for e_c=1:N_e
            e_val=e_gridvals_J(:,:,:,:,:,:,e_c,:); % e_gridvals_J is [1,1,1,1,1,N_j,N_e,l_e] — e on dim 7

            ReturnMatrix_ze=CreateReturnFnMatrix_fastOLG_ExpAsset_Disc_e(ReturnFn, 0, n_d2, n_a1, n_a1,n_a2, special_n_z,special_n_e,N_j, d2_gridvals, a1_gridvals, a1_gridvals, a2_grid, z_val, e_val, ReturnFnParamsAgeMatrix,0,0); % Level=0, Refine=0

            entireRHS_ze=ReturnMatrix_ze+DiscountedEV_z;

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_ze,[],1);
            V(:,z_c,e_c)=reshape(Vtemp,[N_a*N_j,1]);
            Policy(:,z_c,e_c)=reshape(maxindex,[N_a*N_j,1]);
        end
    end
end


%% fastOLG with z, so need to output to take certain shapes
% V=reshape(V,[N_a*N_j,N_z]);
% Policy=reshape(Policy,[N_a,N_j,N_z]);

%% Output shape for policy
Policy=shiftdim(Policy,-1); % so first dim is just one point


end
