function varargout=ValueFnIter_FHorz_QuasiHyperbolic(n_d,n_a,n_z,N_j,d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
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

% V=nan; % The value function of the quasi-hyperbolic agent
% Policy=nan; % The value function of the quasi-hyperbolic agent

N_d=prod(n_d);
% N_a=prod(n_a);
N_z=prod(n_z);

if ~isfield(vfoptions,'quasi_hyperbolic')
    vfoptions.quasi_hyperbolic='Naive'; % This is the default, alternative is 'Sophisticated'.
    fprintf('NOTE: using Naive quasi-hyperbolic as type of quasi-hyperbolic was not specified \n')
elseif ~strcmp(vfoptions.quasi_hyperbolic,'Naive') && ~strcmp(vfoptions.quasi_hyperbolic,'Sophisticated') 
    % Check that one of the possible options have been used. If not then error.
    error('vfoptions.quasi_hyperbolic must be either Naive or Sophisticated (check spelling and capital letter) \n')
end

if ~isfield(vfoptions,'QHadditionaldiscount')
    error('You must declare vfoptions.QHadditionaldiscount when using quasi-hyperbolic discouting (you have vfoptions.exoticpreferences set to QuasiHyperbolic)')
end

%%
if vfoptions.divideandconquer==1
    [VKron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolic_DC(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
end
if vfoptions.gridinterplayer==1
    [VKron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolic_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
end


%% Just do the standard case
if vfoptions.parallel==2
    if N_d==0
        if isfield(vfoptions,'n_e')
            if N_z==0
                [VKron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolic_nod_noz_e_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolic_nod_e_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else % UP TO HERE
            if N_z==0
                [VKron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolic_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolic_nod_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
        % Policy without d
        PolicyKron=shiftdim(PolicyKron,-1);
    else
        if isfield(vfoptions,'n_e')
            if N_z==0
                [VKron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolic_noz_e_raw(n_d,n_a, vfoptions.n_e, N_j, d_gridvals, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolic_e_raw(n_d,n_a, n_z, vfoptions.n_e, N_j, d_gridvals, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolic_noz_raw(n_d,n_a, N_j, d_gridvals, a_grid, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolic_raw(n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    end
end

%Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if vfoptions.outputkron==0
    if isfield(vfoptions,'n_e')
        if N_z==0
            V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
        else
            V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz_e(PolicyKron, n_d, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
        end
    else
        if N_z==0
            V=reshape(VKron,[n_a,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz_noz(PolicyKron, n_d, n_a, N_j, vfoptions);
        else
            V=reshape(VKron,[n_a,n_z,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, n_z, N_j, vfoptions);
        end
    end
    nOutputs = nargout;
    if nOutputs==3 % Valt will be output, so also need to unkron it
        if isfield(vfoptions,'n_e')
            if N_z==0
                Valt=reshape(ValtKron,[n_a,vfoptions.n_e,N_j]);
            else
                Valt=reshape(ValtKron,[n_a,n_z,vfoptions.n_e,N_j]);
            end
        else
            if N_z==0
                Valt=reshape(ValtKron,[n_a,N_j]);
            else
                Valt=reshape(ValtKron,[n_a,n_z,N_j]);
            end
        end
    end
else
    V=VKron;
    Policy=PolicyKron;
end



% The basic version just returns two outputs, but it is possible to request
% three as might want to see the 'other' value fn which is used in the expectations.
nOutputs = nargout;
if nOutputs==2
    varargout={V,Policy}; % Policy will be Policytilde, value fn is Vtilde if naive, Vhat if sophisticated
elseif nOutputs==3
    varargout={V,Policy,Valt}; % Valt with be V if naive, Vunderbar if sophisticated
end

end
