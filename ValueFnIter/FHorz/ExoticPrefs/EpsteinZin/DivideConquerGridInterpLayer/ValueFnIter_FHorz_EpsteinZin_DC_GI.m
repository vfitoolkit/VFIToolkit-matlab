function [VKron,PolicyKron]=ValueFnIter_FHorz_EpsteinZin_DC_GI(n_d,n_a,n_z,N_j,d_gridvals,a_grid,z_gridvals_J,pi_z_J,ReturnFn,Parameters,DiscountFactorParamNames,ReturnFnParamNames,vfoptions,sj,warmglow,ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Epstein-Zin with divide-and-conquer plus grid interpolation layer: sets the
% level1n default and routes to the DC1_GI1 raws (mirrors
% ValueFnIter_FHorz_QuasiHyperbolic_DC_GI). Returns V and Policy in Kron form;
% the caller (ValueFnIter_FHorz_EpsteinZin) does the UnKron.

if ~isscalar(n_a)
    error('Epstein-Zin with divide-and-conquer plus the grid interpolation layer only supports a single endogenous state (scalar n_a) for now')
end
if ~isfield(vfoptions,'ngridinterp')
    error('You must declare vfoptions.ngridinterp when using the grid interpolation layer')
end

if ~isfield(vfoptions,'level1n')
    vfoptions.level1n=floor(sqrt(n_a(1)));
    if vfoptions.verbose==1
        fprintf('Suggestion: When using vfoptions.divideandconquer it will be faster or slower if you set different values of vfoptions.level1n (for smaller models 7 or 9 is good, but for larger models something 15 or 21 can be better) \n')
    end
end
vfoptions.level1n=min(vfoptions.level1n,n_a);

N_d=prod(n_d);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

if N_e==0
    if N_z==0
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_EpsteinZin_DC1_GI1_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_EpsteinZin_DC1_GI1_noz_raw(n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        end
    else
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_EpsteinZin_DC1_GI1_nod_raw(n_a,n_z,N_j, a_grid, z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_EpsteinZin_DC1_GI1_raw(n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        end
    end
else % N_e
    if N_z==0
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_EpsteinZin_DC1_GI1_nod_noz_e_raw(n_a,vfoptions.n_e,N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_EpsteinZin_DC1_GI1_noz_e_raw(n_d,n_a,vfoptions.n_e,N_j, d_gridvals, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        end
    else
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_EpsteinZin_DC1_GI1_nod_e_raw(n_a,n_z,vfoptions.n_e,N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_EpsteinZin_DC1_GI1_e_raw(n_d,n_a,n_z,vfoptions.n_e,N_j, d_gridvals, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        end
    end
end


end
