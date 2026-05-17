function [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_nod_raw(V,n_a,n_z,N_j, a_grid,z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_a=prod(n_a);
N_z=prod(n_z);

Policy=zeros(N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z

%% j=N_j

% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,:,N_j);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, n_z, 0, a_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
%Calc the max and it's index
[Vtemp,maxindex]=max(ReturnMatrix,[],1);
V(:,:,N_j)=Vtemp;
Policy(:,:,N_j)=maxindex;


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    VKronNext_j=Vtemp_j; % Has been presaved before it was
    Vtemp_j=V(:,:,jj); % Grab this before it is replaced/updated


    ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, n_z, 0, a_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec,0);

    EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV.*ones(1,N_a,1);

    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);

    V(:,:,jj)=shiftdim(Vtemp,1);
    Policy(:,:,jj)=shiftdim(maxindex,1);
end

%% Output shape for policy
Policy=shiftdim(Policy,-1);

end
