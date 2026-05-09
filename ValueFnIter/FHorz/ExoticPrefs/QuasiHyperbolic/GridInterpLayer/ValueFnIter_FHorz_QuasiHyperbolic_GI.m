function [V1, Policy,Valt]=ValueFnIter_FHorz_QuasiHyperbolic_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Quasi-hyperbolic discounting wrapper for GI finite-horizon value function iteration.
% Dispatches to ValueFnIter_FHorz_GI_QuasiHyperbolic_raw or _nod_raw.
%
% Naive:        varargout = {Vtilde, Policy}          or {Vtilde, Policy, V}
% Sophisticated: varargout = {Vhat,  Policy}          or {Vhat,  Policy, Vunderbar}

N_d=prod(n_d);
N_z=prod(n_z);

if ~isfield(vfoptions,'ngridinterp')
    error('You must declare vfoptions.ngridinterp when using the grid interpolation layer')
end

%%
if isscalar(n_a)
    if N_e==0
        if N_z==0
            if N_d==0

            else

            end
        else
            if N_d==0

            else

            end
        end
    else
        if N_z==0
            if N_d==0

            else

            end
        else
            if N_d==0
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_GI_QuasiHyperbolic_nod_raw(n_a,n_z,N_j, a_grid, z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_GI_QuasiHyperbolic_raw(n_d,n_a,n_z,N_j, d_gridvalsvals, a_grid, z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    end
else
    error('ValueFnIter_FHorz_GI_QuasiHyperbolic currently only supports scalar n_a (one endogenous state)')
end

%% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if N_d==0
    Case2policies=[n_a,vfoptions.ngridinterp];
else
    Case2policies=[n_d,n_a,vfoptions.ngridinterp];
end


if vfoptions.outputkron==0
    if N_z==0
        V1=reshape(V1Kron,[n_a,N_j]);
        Policy=UnKronPolicyIndexes_Case2_FHorz_noz(PolicyKron, Case2policies, n_a, N_j, vfoptions);
        Valt=reshape(ValtKron,[n_a,N_j]);
    else
        V1=reshape(V1Kron,[n_a,n_z,N_j]);
        Policy=UnKronPolicyIndexes_Case2_FHorz(PolicyKron, Case2policies, n_a, n_z, N_j, vfoptions);
        Valt=reshape(ValtKron,[n_a,n_z,N_j]);
    end
else
    V1=VKron;
    Policy=PolicyKron;
    Valt=ValtKron;
end


end
