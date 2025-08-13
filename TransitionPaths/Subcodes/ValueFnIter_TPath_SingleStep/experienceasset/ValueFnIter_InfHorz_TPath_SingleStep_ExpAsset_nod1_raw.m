function [V,Policy]=ValueFnIter_InfHorz_TPath_SingleStep_ExpAsset_nod1_raw(Vnext,n_d2,n_a1, n_a2,n_z, d2_grid, a1_grid, a2_grid, z_gridvals, pi_z, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames,aprimeFnParamNames, vfoptions)

N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);

V=zeros(N_a,N_z,'gpuArray');
Policy=zeros(2,N_a,N_z,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
if vfoptions.lowmemory>0
    special_n_z=ones(1,length(n_z));
end

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames);
[a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix_Case1(aprimeFn, n_d2, n_a2, d2_grid, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
% Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2]
aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2]
aprimeProbs=repmat(a2primeProbs,N_a1,1,N_z);  % [N_d2*N_a1,N_a2,N_z]

Vlower=reshape(Vnext(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_z]);
Vupper=reshape(Vnext(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_z]);
% Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
skipinterp=(Vlower==Vupper);
aprimeProbs(skipinterp)=0; % effectively skips interpolation

% Switch EV from being in terps of a2prime to being in terms of d2 and a2
EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,zprime)
% Already applied the probabilities from interpolating onto grid

if vfoptions.lowmemory==0

    EV=EV.*shiftdim(pi_z',-2);
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=squeeze(sum(EV,3)); % sum over z', leaving a singular third dimension

    ReturnMatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, n_d2, n_a1,n_a2, n_z, d2_grid, a1_grid, a2_grid, z_gridvals, ReturnFnParamsVec);

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*repelem(EV,1,N_a1,1);

    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);

    V=shiftdim(Vtemp,1);
    Policy(1,:,:)=rem(maxindex-1,N_d2)+1;
    Policy(2,:,:)=ceil(maxindex/N_d2);

elseif vfoptions.lowmemory==1

    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        % Calc the condl expectation term (except beta), which depends on z but not on control variables
        EV_z=EV.*shiftdim(pi_z(z_c,:)',-2);
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,3);

        ReturnMatrix_z=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, n_d2, n_a1,n_a2, special_n_z, d2_grid, a1_grid, a2_grid, z_val, ReturnFnParamsVec);

        entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*repelem(EV_z,1,N_a1,1);

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS_z,[],1);
        V(:,z_c)=Vtemp;
        Policy(1,:,z_c)=rem(maxindex-1,N_d2)+1;
        Policy(2,:,z_c)=ceil(maxindex/N_d2);

    end
end



end
