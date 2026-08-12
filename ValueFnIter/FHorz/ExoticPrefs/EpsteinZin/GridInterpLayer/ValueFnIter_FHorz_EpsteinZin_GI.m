function [VKron,PolicyKron]=ValueFnIter_FHorz_EpsteinZin_GI(n_d,n_a,n_z,N_j,d_gridvals,a_grid,z_gridvals_J,pi_z_J,ReturnFn,Parameters,DiscountFactorParamNames,ReturnFnParamNames,vfoptions,sj,warmglow,ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Epstein-Zin with grid interpolation layer: routes to the GI1 raws (mirrors
% ValueFnIter_FHorz_QuasiHyperbolic_GI). Returns V and Policy in Kron form; the
% caller (ValueFnIter_FHorz_EpsteinZin) does the UnKron.

if ~isscalar(n_a)
    error('Epstein-Zin with the grid interpolation layer only supports a single endogenous state (scalar n_a) for now')
end
if ~isfield(vfoptions,'ngridinterp')
    error('You must declare vfoptions.ngridinterp when using the grid interpolation layer')
end

N_d=prod(n_d);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

if N_e==0
    if N_z==0
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_EpsteinZin_GI1_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_EpsteinZin_GI1_noz_raw(n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        end
    else
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_EpsteinZin_GI1_nod_raw(n_a,n_z,N_j, a_grid, z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_EpsteinZin_GI1_raw(n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        end
    end
else % N_e
    if N_z==0
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_EpsteinZin_GI1_nod_noz_e_raw(n_a,vfoptions.n_e,N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_EpsteinZin_GI1_noz_e_raw(n_d,n_a,vfoptions.n_e,N_j, d_gridvals, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        end
    else
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_EpsteinZin_GI1_nod_e_raw(n_a,n_z,vfoptions.n_e,N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_EpsteinZin_GI1_e_raw(n_d,n_a,n_z,vfoptions.n_e,N_j, d_gridvals, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        end
    end
end


end
