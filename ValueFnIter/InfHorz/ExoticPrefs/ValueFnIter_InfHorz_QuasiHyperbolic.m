function [V, Policy]=ValueFnIter_InfHorz_QuasiHyperbolic(V0, n_d,n_a,n_z,d_gridvals,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, vfoptions,Parameters,ReturnFnParamNames)
% (last two entries of) DiscountFactorParamNames contains the names for the two parameters relating to
% Quasi-hyperbolic preferences.
% Let V_j be the standard (exponential discounting) solution to the value fn problem
% The 'Naive' quasi-hyperbolic solution takes current actions as if the
% future agent take actions as if having time-consistent (exponential discounting) preferences.
% V_naive_j: Vtilde= u_t+ beta0beta *E[V_{j+1}]
% The 'Sophisticated' quasi-hyperbolic solution takes into account the time-inconsistent behaviour of their future self.
% Let Vunderbar_j be the exponential discounting value fn of the
% time-inconsistent policy function (aka. the policy-greedy exponential discounting value function of the time-inconsistent policy function)
% V_sophisticated_j: Vhat=u_t+beta0beta*E[Vundberbar_{j+1}]
% See documentation for a fuller explanation of this.

V=nan; % The value function of the quasi-hyperbolic agent
Policy=nan; % The value function of the quasi-hyperbolic agent

N_d=prod(n_d);
% N_a=prod(n_a);
% N_z=prod(n_z);

if ~isfield(vfoptions,'quasi_hyperbolic')
    vfoptions.quasi_hyperbolic='Naive'; % This is the default, alternative is 'Sophisticated'.
elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
    vfoptions.maxiter=1000;
    % Sophisticated quasihyperbolic in infinite horizon seems to struggle to converge, so end after fixed number of grid points
    % My impression is that accurate convergence would require insane number of grid points.
    vfoptions.verbose=1;
    % I set verbose so you can see if it appears to have gotten as close to
    % convergence as it can before reaching maxiter (if currdist is cycling
    % through numbers of the same magnitude for a while before stopping)
elseif ~strcmp(vfoptions.quasi_hyperbolic,'Naive') && ~strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
    % Check that one of the possible options have been used. If not then error.
    error('vfoptions.quasi_hyperbolic must be either Naive or Sophisticated (check spelling and capital letter) \n')
end

if ~isfield(vfoptions,'QHadditionaldiscount')
    error('You must declare vfoptions.QHadditionaldiscount when using quasi-hyperbolic discouting (you have vfoptions.exoticpreferences set to QuasiHyperbolic)')
end

%%

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
beta0=Parameters.(vfoptions.QHadditionaldiscount{1});

%%
% Use the same ReturnMatrix for both the continuation value, and the value fn
ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_grid,ReturnFnParamsVec);

%%
if strcmp(vfoptions.quasi_hyperbolic,'Naive') % For Naive, just solve the standard value function problem, and then just one step following that.
    if N_d==0
        % First calculate the exponential discounting solution
        [V,~]=ValueFnIter_InfHorz_nod_raw(V0, n_a, n_z, pi_z, prod(DiscountFactorParamsVec), ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
        % Then the Naive quasi-hyperbolic from this
        [V,Policy]=ValueFnIter_InfHorz_QuasiHyperbolicN_nod_raw(V, n_a, n_z, pi_z, beta0, ReturnMatrix);
        Policy=shiftdim(Policy,-1);
    else
        % First calculate the exponential discounting solution
        [V, ~]=ValueFnIter_InfHorz_raw(V0, n_d,n_a,n_z, pi_z, prod(DiscountFactorParamsVec), ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, vfoptions.maxiter);
        % Then the Naive quasi-hyperbolic from this
        [V, Policy]=ValueFnIter_InfHorz_QuasiHyperbolicN_raw(V, n_d,n_a,n_z, pi_z, beta0, ReturnMatrix);
    end
elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated') % For Naive, just solve the standard value function problem, and then just one step following that.
    if N_d==0
        [V,Policy]=ValueFnIter_InfHorz_QuasiHyperbolicS_nod_raw(V0, n_a, n_z, pi_z, DiscountFactorParamsVec, beta0, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
        Policy=shiftdim(Policy,-1);
    else
        [V, Policy]=ValueFnIter_InfHorz_QuasiHyperbolicS_raw(V0, n_d,n_a,n_z, pi_z, DiscountFactorParamsVec, beta0, ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, vfoptions.maxiter);
    end
end


%%
V=reshape(V,[n_a,n_z]);
Policy=UnKronPolicyIndexes_Case1(Policy, n_d, n_a, n_z,vfoptions);

end
