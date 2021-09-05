function [V, Policy]=ValueFnIter_Case1_FHorz_QuasiHyperbolic(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% (last two entries of) DiscountFactorParamNames contains the names for the two parameters relating to
% Quasi-hyperbolic preferences.
% Let V_j be the standard (exponential discounting) solution to the value fn problem
% The 'Naive' quasi-hyperbolic solution takes current actions as if the
% future agent take actions as if having time-consistent (exponential discounting) preferences.
% V_naive_j= u_t+ beta_0 *E[V_{j+1}]
% The 'Sophisticated' quasi-hyperbolic solution takes into account the time-inconsistent behaviour of their future self.
% Let Vhat_j be the exponential discounting value fn of the
% time-inconsistent policy function (aka. the policy-greedy exponential discounting value function of the time-inconsistent policy function)
% V_sophisticated_j=u_t+beta_0*E[Vhat_{j+1}]
% See documentation for a fuller explanation of this.

V=nan; % The value function of the quasi-hyperbolic agent
Policy=nan; % The value function of the quasi-hyperbolic agent

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if ~isfield(vfoptions,'quasi_hyperbolic')
    vfoptions.quasi_hyperbolic='Naive'; % This is the default, alternative is 'Sophisticated'.
elseif ~strcmp(vfoptions.quasi_hyperbolic,'Naive') && ~strcmp(vfoptions.quasi_hyperbolic,'Sophisticated') 
    % Check that one of the possible options have been used. If not then error.
    fprintf('ERROR: vfoptions.quasi_hyperbolic must be either Naive or Sophisticated (check spelling and capital letter) \n')
    dbstack
    return
end

% CANNOT YET DEAL WITH DYNASTY WHEN USING QUASI-HYPERBOLIC

%% Just do the standard case
if vfoptions.parallel==2
    if N_d==0
        [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_QuasiHyperbolic_no_d_raw(n_a, n_z, N_j, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    else
        [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_QuasiHyperbolic_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    end
elseif vfoptions.parallel==0 || vfoptions.parallel==1
    disp('ERROR: Quasi-Hyperbolic currently only implemented for Parallel=2: email robertdkirkby@gmail.com')
    dbstack
    return
%     if N_d==0
%         % Following command is somewhat misnamed, as actually does Par0 and Par1
%         [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_QuasiHyperbolic_no_d_Par0_raw(n_a, n_z, N_j, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
%     else
%         % Following command is somewhat misnamed, as actually does Par0 and Par1
%         [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_QuasiHyperbolic_Par0_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
%     end
end

%Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
V=reshape(VKron,[n_a,n_z,N_j]);
Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, n_z, N_j,vfoptions);

% Sometimes numerical rounding errors (of the order of 10^(-16) can mean
% that Policy is not integer valued. The following corrects this by converting to int64 and then
% makes the output back into double as Matlab otherwise cannot use it in
% any arithmetical expressions.
if vfoptions.policy_forceintegertype==1
    fprintf('USING vfoptions to force integer... \n')
    % First, give some output on the size of any changes in Policy as a
    % result of turning the values into integers
    temp=max(max(max(abs(round(Policy)-Policy))));
    while ndims(temp)>1
        temp=max(temp);
    end
    fprintf('  CHECK: Maximum change when rounding values of Policy is %8.6f (if these are not of numerical rounding error size then something is going wrong) \n', temp)
    % Do the actual rounding to integers
    Policy=round(Policy);
    % Somewhat unrelated, but also do a double-check that Policy is now all positive integers
    temp=min(min(min(Policy)));
    while ndims(temp)>1
        temp=min(temp);
    end
    fprintf('  CHECK: Minimum value of Policy is %8.6f (if this is <=0 then something is wrong) \n', temp)
%     Policy=uint64(Policy);
%     Policy=double(Policy);
elseif vfoptions.policy_forceintegertype==2
    % Do the actual rounding to integers
    Policy=round(Policy);
end

end