function varargout=ValueFnIter_InfHorz_QuasiHyperbolic(V0, n_d,n_a,n_z,d_gridvals,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, vfoptions,Parameters,ReturnFnParamNames)
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
%
% Outputs are returned via varargout. Caller may request:
%   Naive:         [V, Policy, Valt, Policyalt]
%                  V         = Vtilde     (QH-discounted, used for the QH-optimal choice)
%                  Policy    = QH-optimal choice (argmax of Vtilde)
%                  Valt      = V_std      (std-discounted continuation, computed at Policyalt)
%                  Policyalt = std-optimal choice (argmax of V_std). Naive needs Policy AND
%                  Policyalt to reconstruct V/Valt from policy alone.
%   Sophisticated: [V, Policy, Valt]
%                  V       = Vhat       (QH-discounted from current self's perspective)
%                  Policy  = equilibrium choice
%                  Valt    = Vunderbar  (realised continuation under future selves' own QH choices)
%                  Policy alone suffices to reconstruct V and Valt.

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

isNaive=strcmp(vfoptions.quasi_hyperbolic,'Naive');

%%

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
beta0=Parameters.(vfoptions.QHadditionaldiscount{1});

%%
if vfoptions.gridinterplayer==1
    % Default GI path (postGI, howardsgreedy=0, howardssparse=0). d-refinement is
    % preference-independent, so the refined return-matrix construction is identical
    % to the standard postGI raws; only the VFI/extra step uses QH discounting.
    if ~isscalar(n_a)
        error('QuasiHyperbolic with vfoptions.gridinterplayer=1 is only implemented for scalar n_a (single endogenous state). Multi-dim n_a (2A) variants are not yet supported.')
    end
    if N_d==0
        if isNaive
            [V, Policy, Valt, Policyalt]=ValueFnIter_InfHorz_QuasiHyperbolicN_postGI_nod_raw(V0, n_a, n_z, a_grid, z_grid, pi_z, DiscountFactorParamsVec, beta0, ReturnFn, ReturnFnParamsVec, vfoptions);
        else
            [V, Policy, Valt, Policyalt]=ValueFnIter_InfHorz_QuasiHyperbolicS_postGI_nod_raw(V0, n_a, n_z, a_grid, z_grid, pi_z, DiscountFactorParamsVec, beta0, ReturnFn, ReturnFnParamsVec, vfoptions);
        end
    else
        if isNaive
            [V, Policy, Valt, Policyalt]=ValueFnIter_InfHorz_QuasiHyperbolicN_Refine_postGI_raw(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_grid, pi_z, DiscountFactorParamsVec, beta0, ReturnFn, ReturnFnParamsVec, vfoptions);
        else
            [V, Policy, Valt, Policyalt]=ValueFnIter_InfHorz_QuasiHyperbolicS_Refine_postGI_raw(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_grid, pi_z, DiscountFactorParamsVec, beta0, ReturnFn, ReturnFnParamsVec, vfoptions);
        end
    end
elseif N_d>0 && strcmp(vfoptions.solnmethod,'purediscretization_refinement')
    % Default GPU path for d-case: refine d out of the return function, then solve
    % the QH problem on the refined (nod-shaped) matrix. See subfn for the argument
    % that d-refinement is preference-independent.
    [V, Policy, Valt, Policyalt]=ValueFnIter_InfHorz_QuasiHyperbolic_Refine(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_grid, pi_z, ReturnFn, ReturnFnParamsVec, DiscountFactorParamsVec, beta0, vfoptions, isNaive);
else
    % Use the same ReturnMatrix for both the continuation value, and the value fn
    ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_grid,ReturnFnParamsVec,0);

    if isNaive % For Naive, just solve the standard value function problem, and then just one step following that.
        if N_d==0
            % First calculate the exponential discounting solution
            [Valt,Policyalt]=ValueFnIter_InfHorz_nod_raw(V0, n_a, n_z, pi_z, prod(DiscountFactorParamsVec), ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            % Then the Naive quasi-hyperbolic from this
            [V,Policy]=ValueFnIter_InfHorz_QuasiHyperbolicN_nod_raw(Valt, n_a, n_z, pi_z, beta0, ReturnMatrix);
            Policy=shiftdim(Policy,-1);
        else
            % First calculate the exponential discounting solution
            [Valt, Policyalt]=ValueFnIter_InfHorz_raw(V0, n_d,n_a,n_z, pi_z, prod(DiscountFactorParamsVec), ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, vfoptions.maxiter);
            % Then the Naive quasi-hyperbolic from this
            [V, Policy]=ValueFnIter_InfHorz_QuasiHyperbolicN_raw(Valt, n_d,n_a,n_z, pi_z, beta0, ReturnMatrix);
        end
    else % Sophisticated
        if N_d==0
            [V,Policy,Valt]=ValueFnIter_InfHorz_QuasiHyperbolicS_nod_raw(V0, n_a, n_z, pi_z, DiscountFactorParamsVec, beta0, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            Policy=shiftdim(Policy,-1);
        else
            [V, Policy,Valt]=ValueFnIter_InfHorz_QuasiHyperbolicS_raw(V0, n_d,n_a,n_z, pi_z, DiscountFactorParamsVec, beta0, ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, vfoptions.maxiter);
        end
    end
end


%%
V=reshape(V,[n_a,n_z]);
Valt=reshape(Valt,[n_a,n_z]);
if N_d==0
    Policy=UnKronPolicyIndexes1_z(Policy, n_a, n_a, n_z, vfoptions);
    if isNaive
        Policyalt=UnKronPolicyIndexes1_z(Policyalt, n_a, n_a, n_z, vfoptions);
    end
else
    Policy=UnKronPolicyIndexes1_z(Policy, [n_d,n_a], n_a, n_z, vfoptions);
    if isNaive
        Policyalt=UnKronPolicyIndexes1_z(Policyalt, [n_d,n_a], n_a, n_z, vfoptions);
    end
end

if isNaive
    varargout={V, Policy, Valt, Policyalt};
else
    varargout={V, Policy, Valt, []};
end

end
