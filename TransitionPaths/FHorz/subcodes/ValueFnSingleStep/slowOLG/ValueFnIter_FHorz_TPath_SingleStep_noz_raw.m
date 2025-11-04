function [V,Policy2]=ValueFnIter_FHorz_TPath_SingleStep_noz_raw(V,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);

Policy=zeros(N_a,N_j,'gpuArray'); % first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%% j=N_j

% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,N_j);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, n_d, n_a, d_gridvals, a_grid, ReturnFnParamsVec,0);
% Calc the max and it's index
[Vtemp,maxindex]=max(ReturnMatrix,[],1);
V(:,N_j)=Vtemp;
Policy(:,N_j)=maxindex;


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    j=N_j-reverse_j;
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,j);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    VKronNext_j=Vtemp_j; % Has been presaved before it was
    Vtemp_j=V(:,j); % Grab this before it is replaced/updated
    
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, n_d, n_a, d_gridvals, a_grid, ReturnFnParamsVec,0);

    entireRHS=ReturnMatrix+DiscountFactorParamsVec*kron(VKronNext_j,ones(N_d,1))*ones(1,N_a,1);

    %Calc the max and it's index
    [Vtemp,maxindex]=max(entireRHS,[],1);
    V(:,j)=Vtemp;
    Policy(:,j)=maxindex;
end

%% Separate d and aprime
Policy2=zeros(2,N_a,N_j,'gpuArray'); % first dim indexes the optimal choice for d and aprime rest of dimensions a,z
Policy2(1,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:)=shiftdim(ceil(Policy/N_d),-1);


end
