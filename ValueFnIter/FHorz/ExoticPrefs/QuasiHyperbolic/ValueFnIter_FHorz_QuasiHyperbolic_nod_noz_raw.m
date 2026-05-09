function [V1,Policy,Valt]=ValueFnIter_FHorz_QuasiHyperbolic_nod_noz_raw(n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Interpretation of output differs by Naive/Sophisticated
% Naive:         {Vtilde, Policy, V}
% Sophisticated: {Vhat,  Policy, Vunderbar}
%
% DiscountFactorParamNames is the standard discount factor beta
% vfoptions.QHadditionaldiscount.gives the name of the beta_0 is the additional discount factor parameter
%
% Let V_j be the standard (exponential discounting) solution to the value fn problem
% The 'Naive' quasi-hyperbolic solution takes current actions as if the future agent take actions as if having time-consistent (exponential discounting) preferences.
% V_naive_j= u_t+ beta_0 *E[V_{j+1}]
% The 'Sophisticated' quasi-hyperbolic solution takes into account the time-inconsistent behaviour of their future self.
% Let Vunderbar_j be the exponential discounting value fn of the time-inconsistent policy function (aka. the policy-greedy exponential discounting value function of the time-inconsistent policy function)
% V_sophisticated_j=u_t+beta_0*E[Vunderbar_{j+1}]
% See documentation for a fuller explanation of this.

N_a=prod(n_a);

% Policy_extra=zeros(N_a,N_j,'gpuArray'); % indexes the optimal choice for aprime rest of dimensions a,z

V=zeros(N_a,N_j,'gpuArray'); % If Naive, then this is V, if Sophisticated then this is Vhat.
Policy=zeros(N_a,N_j,'gpuArray'); % indexes the optimal choice for aprime rest of dimensions a,z

%%
if vfoptions.lowmemory>0
    special_n_a=ones(1,length(n_a));
    a_gridvals=CreateGridvals(n_a,a_grid,1); % The 1 at end indicates want output in form of matrix.
end

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

    if strcmp(vfoptions.quasi_hyperbolic,'Naive')
        Vtilde=V;
    else % strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
        Vunderbar=V;
    end
else
    % Using V_Jplus1
    % Note: The V_Jplus1 input should be V if naive, Vunderbar if sophisticated
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,1]);    % First, switch V_Jplus1 into Kron form

    % Preallocate Vtilde and Vunderbar
    if strcmp(vfoptions.quasi_hyperbolic,'Naive')
        Vtilde=V;
    else % strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
        Vunderbar=V;
    end

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec); % Discount factor between any two future periods
    beta0beta=Parameters.(vfoptions.QHadditionaldiscount)*beta; % Discount factor between today and tomorrow.

    VKronNext_j=V_Jplus1; % Note: The V_Jplus1 input should be V if naive, Vunderbar if sophisticated
        
    EV=VKronNext_j;
        
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, 0, n_a, 0, a_grid, ReturnFnParamsVec);

    if strcmp(vfoptions.quasi_hyperbolic,'Naive')
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
    elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
        % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat)
        % and then we compute Vunderbar.
        % First Vhat
        entireRHS=ReturnMatrix+beta0beta*EV; %*EV.*ones(1,N_a,1);  % Use the today-to-tomorrow discount factor
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,N_j)=Vtemp; % Note that this is Vhat when sophisticated
        Policy(:,N_j)=maxindex; % This is the policy from solving the problem of Vhat
        % Now Vstar
        entireRHS=ReturnMatrix+beta*EV; %*EV.*ones(1,N_a,1); % Use the two-future-periods discount factor
        maxindexfull=maxindex+N_a*(0:1:N_a-1);
        Vunderbar(:,N_j)=entireRHS(maxindexfull); % Evaluate time-inconsistent policy using two-future-periods discount rate
    end
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
    
    if strcmp(vfoptions.quasi_hyperbolic,'Naive')
        VKronNext_j=V(:,jj+1); % Use V (goes into the equation to determine V)
    else % strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
        VKronNext_j=Vunderbar(:,jj+1); % Use Vunderbar (goes into the equation to determine Vhat)
    end

    EV=VKronNext_j;
        
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, 0, n_a, 0, a_grid, ReturnFnParamsVec);

    if strcmp(vfoptions.quasi_hyperbolic,'Naive')
        % For naive, we compute V which is the exponential discounter case, and then from this we get Vtilde and
        % Policy (which is Policytilde) that correspond to the naive quasihyperbolic discounter
        % First V
        entireRHS=ReturnMatrix+beta**EV; %EV.*ones(1,N_a,1); % Use the two-future-periods discount factor
        [Vtemp,~]=max(entireRHS,[],1);
        V(:,jj)=Vtemp;
        % Now Vtilde and Policy
        entireRHS=ReturnMatrix+beta0beta*EV; %*EV.*ones(1,N_a,1); % Use today-to-tomorrow discount factor
        [Vtemp,maxindex]=max(entireRHS,[],1);
        Vtilde(:,jj)=Vtemp; % Evaluate what would have done under exponential discounting
        Policy(:,jj)=maxindex; % Use the policy from solving the problem of Vtilde
    elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
        % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat)
        % and then we compute Vunderbar.
        % First Vhat
        entireRHS=ReturnMatrix+beta0beta*EV; %*EV.*ones(1,N_a,1);  % Use the today-to-tomorrow discount factor
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,jj)=Vtemp; % Note that this is Vhat when sophisticated
        Policy(:,jj)=maxindex; % This is the policy from solving the problem of Vhat
        % Now Vstar
        entireRHS=ReturnMatrix+beta*EV; %*EV.*ones(1,N_a,1); % Use the two-future-periods discount factor
        maxindexfull=maxindex+N_a*(0:1:N_a-1);
        Vunderbar(:,jj)=entireRHS(maxindexfull); % Evaluate time-inconsistent policy using two-future-periods discount rate
    end
end

if strcmp(vfoptions.quasi_hyperbolic,'Naive')
    V1=V;
    Valt=Vtilde;
elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
    V1=Vhat;
    Valt=Vunderbar;
end


end
