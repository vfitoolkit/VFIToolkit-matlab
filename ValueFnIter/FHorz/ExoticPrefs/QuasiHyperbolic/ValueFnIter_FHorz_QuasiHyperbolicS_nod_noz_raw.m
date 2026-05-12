function [Vunderbar,Policy,Vhat]=ValueFnIter_FHorz_QuasiHyperbolicS_nod_noz_raw(n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Sophisticated quasi-hyperbolic discounting
%
% DiscountFactorParamNames is the standard discount factor beta
% vfoptions.QHadditionaldiscount gives the name of beta_0, the additional discount factor parameter
%
% The 'Sophisticated' quasi-hyperbolic solution takes into account the time-inconsistent behaviour of their future self.
% Let Vunderbar_j be the exponential discounting value fn of the time-inconsistent policy function (aka. the policy-greedy exponential discounting value function of the time-inconsistent policy function)
% V_sophisticated_j=u_t+beta_0*E[Vunderbar_{j+1}]
% See documentation for a fuller explanation of this.

N_a=prod(n_a);

Vhat=zeros(N_a,N_j,'gpuArray');
Policy=zeros(N_a,N_j,'gpuArray'); % indexes the optimal choice for aprime rest of dimensions a,z


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);
% Nothing extra to do for final period with quasi-hyperbolic preferences

if ~isfield(vfoptions,'V_Jplus1')

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, 0, n_a, 0, a_grid, ReturnFnParamsVec,0);
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    Vhat(:,N_j)=Vtemp;
    Policy(:,N_j)=maxindex;

    Vunderbar=Vhat;
else
    % Using V_Jplus1
    % Note: The V_Jplus1 input should be Vunderbar for sophisticated
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,1]);    % First, switch V_Jplus1 into Kron form

    Vunderbar=Vhat;

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec); % Discount factor between any two future periods
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta; % Discount factor between today and tomorrow.

    VKronNext_j=V_Jplus1; % Note: The V_Jplus1 input should be Vunderbar for sophisticated

    EV=VKronNext_j;

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, 0, n_a, 0, a_grid, ReturnFnParamsVec,0);

    % For sophisticated we compute Vhat, and the Policy (which is Policyhat)
    % and then we compute Vunderbar.
    % First Vhat
    entireRHS=ReturnMatrix+beta0beta*EV; %*EV.*ones(1,N_a,1);  % Use the today-to-tomorrow discount factor
    [Vtemp,maxindex]=max(entireRHS,[],1);
    Vhat(:,N_j)=Vtemp; % Note that this is Vhat when sophisticated
    Policy(:,N_j)=maxindex; % This is the policy from solving the problem of Vhat
    % Now Vstar
    entireRHS=ReturnMatrix+beta*EV; %*EV.*ones(1,N_a,1); % Use the two-future-periods discount factor
    maxindexfull=maxindex+N_a*(0:1:N_a-1);
    Vunderbar(:,N_j)=entireRHS(maxindexfull); % Evaluate time-inconsistent policy using two-future-periods discount rate
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
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta; % Discount factor between today and tomorrow.

    VKronNext_j=Vunderbar(:,jj+1); % Use Vunderbar (goes into the equation to determine Vhat)

    EV=VKronNext_j;

    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, 0, n_a, 0, a_grid, ReturnFnParamsVec,0);

    % For sophisticated we compute Vhat, and the Policy (which is Policyhat)
    % and then we compute Vunderbar.
    % First Vhat
    entireRHS=ReturnMatrix+beta0beta*EV; %*EV.*ones(1,N_a,1);  % Use the today-to-tomorrow discount factor
    [Vtemp,maxindex]=max(entireRHS,[],1);
    Vhat(:,jj)=Vtemp; % Note that this is Vhat when sophisticated
    Policy(:,jj)=maxindex; % This is the policy from solving the problem of Vhat
    % Now Vstar
    entireRHS=ReturnMatrix+beta*EV; %*EV.*ones(1,N_a,1); % Use the two-future-periods discount factor
    maxindexfull=maxindex+N_a*(0:1:N_a-1);
    Vunderbar(:,jj)=entireRHS(maxindexfull); % Evaluate time-inconsistent policy using two-future-periods discount rate
end

end
