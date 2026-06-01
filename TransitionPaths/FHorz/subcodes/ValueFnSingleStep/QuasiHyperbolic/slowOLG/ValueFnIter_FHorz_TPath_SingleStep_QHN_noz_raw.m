function [V,Policy,Policyalt,Vtilde]=ValueFnIter_FHorz_TPath_SingleStep_QHN_noz_raw(V,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);

Policy=zeros(N_a,N_j,'gpuArray'); % first dim indexes the optimal choice for d and aprime rest of dimensions a,z
Policyalt=zeros(N_a,N_j,'gpuArray'); % exponential discounter optimal choice (Valt is computed at this)
Vtilde=zeros(N_a,N_j,'gpuArray'); % agent's-perspective value at QH-optimal policy under beta0beta

%% j=N_j

% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,N_j);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, n_d, n_a, d_gridvals, a_grid, ReturnFnParamsVec,0);
% Calc the max and it's index
[Vtemp,maxindex]=max(ReturnMatrix,[],1);
V(:,N_j)=Vtemp;
Policy(:,N_j)=maxindex;
Policyalt(:,N_j)=maxindex; % terminal period: QH and exponential discounter coincide
Vtilde(:,N_j)=V(:,N_j); % terminal: no continuation, Vtilde=Valt


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    j=N_j-reverse_j;

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,j);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,j);
    beta=prod(DiscountFactorParamsVec); % Discount factor between any two future periods
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,j);
    beta0beta=beta0*beta; % Discount factor between today and tomorrow.

    VKronNext_j=Vtemp_j; % Has been presaved before it was
    Vtemp_j=V(:,j); % Grab this before it is replaced/updated

    ReturnMatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, n_d, n_a, d_gridvals, a_grid, ReturnFnParamsVec,0);

    entireEV=kron(VKronNext_j,ones(N_d,1));

    % First Valt
    entireRHS_alt=ReturnMatrix+beta*entireEV*ones(1,N_a,1);
    [Vtemp,maxindex_alt]=max(entireRHS_alt,[],1);
    V(:,j)=Vtemp;
    Policyalt(:,j)=maxindex_alt;
    % Now Policy
    entireRHS=ReturnMatrix+beta0beta*entireEV*ones(1,N_a,1);
    [Vtilde_tmp,maxindex]=max(entireRHS,[],1);
    Vtilde(:,j)=Vtilde_tmp;
    Policy(:,j)=maxindex;
end

%%
Policy=shiftdim(Policy,-1);
Policyalt=shiftdim(Policyalt,-1);

end
