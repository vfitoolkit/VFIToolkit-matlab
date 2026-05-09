function [Vtilde,Policy,V]=ValueFnIter_FHorz_QuasiHyperbolicN_nod_noz_raw(n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Naive quasi-hyperbolic discounting
%
% DiscountFactorParamNames is the standard discount factor beta
% vfoptions.QHadditionaldiscount gives the name of beta_0, the additional discount factor parameter
%
% Let V_j be the standard (exponential discounting) solution to the value fn problem
% The 'Naive' quasi-hyperbolic solution takes current actions as if the future agent take actions as if having time-consistent (exponential discounting) preferences.
% V_naive_j= u_t+ beta_0 *E[V_{j+1}]
% See documentation for a fuller explanation of this.

N_a=prod(n_a);

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(N_a,N_j,'gpuArray'); % indexes the optimal choice for aprime rest of dimensions a,z


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);
% Nothing extra to do for final period with quasi-hyperbolic preferences

if ~isfield(vfoptions,'V_Jplus1')

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, 0, n_a, 0, a_grid, ReturnFnParamsVec);
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,N_j)=Vtemp;
    Policy(:,N_j)=maxindex;

    Vtilde=V;
else
    % Using V_Jplus1
    % Note: The V_Jplus1 input should be V for naive
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,1]);    % First, switch V_Jplus1 into Kron form

    Vtilde=V;

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec); % Discount factor between any two future periods
    beta0beta=Parameters.(vfoptions.QHadditionaldiscount)*beta; % Discount factor between today and tomorrow.

    VKronNext_j=V_Jplus1; % Note: The V_Jplus1 input should be V for naive

    EV=VKronNext_j;

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, 0, n_a, 0, a_grid, ReturnFnParamsVec);

    % For naive, we compute V which is the exponential discounter case, and then from this we get Vtilde and
    % Policy (which is Policytilde) that correspond to the naive quasihyperbolic discounter
    % First V
    entireRHS=ReturnMatrix+beta*EV; %*EV.*ones(1,N_a,1); % Use the two-future-periods discount factor
    [Vtemp,~]=max(entireRHS,[],1);
    V(:,N_j)=Vtemp;
    % Now Vtilde and Policy
    entireRHS=ReturnMatrix+beta0beta*EV; %*EV.*ones(1,N_a,1); % Use today-to-tomorrow discount factor
    [Vtemp,maxindex]=max(entireRHS,[],1);
    Vtilde(:,N_j)=Vtemp; % Evaluate what would have done under exponential discounting
    Policy(:,N_j)=maxindex; % Use the policy from solving the problem of Vtilde
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
    end


    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec); % Discount factor between any two future periods
    beta0beta=Parameters.(vfoptions.QHadditionaldiscount)*beta; % Discount factor between today and tomorrow.

    VKronNext_j=V(:,jj+1); % Use V (goes into the equation to determine V)

    EV=VKronNext_j;

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, 0, n_a, 0, a_grid, ReturnFnParamsVec);

    % For naive, we compute V which is the exponential discounter case, and then from this we get Vtilde and
    % Policy (which is Policytilde) that correspond to the naive quasihyperbolic discounter
    % First V
    entireRHS=ReturnMatrix+beta*EV; %*EV.*ones(1,N_a,1); % Use the two-future-periods discount factor
    [Vtemp,~]=max(entireRHS,[],1);
    V(:,jj)=Vtemp;
    % Now Vtilde and Policy
    entireRHS=ReturnMatrix+beta0beta*EV; %*EV.*ones(1,N_a,1); % Use today-to-tomorrow discount factor
    [Vtemp,maxindex]=max(entireRHS,[],1);
    Vtilde(:,jj)=Vtemp; % Evaluate what would have done under exponential discounting
    Policy(:,jj)=maxindex; % Use the policy from solving the problem of Vtilde
end

end
