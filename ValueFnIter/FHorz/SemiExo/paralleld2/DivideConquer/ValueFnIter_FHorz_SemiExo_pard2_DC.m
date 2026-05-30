function [V,Policy]=ValueFnIter_FHorz_SemiExo_pard2_DC(n_d1,n_d2,n_a,n_semiz,n_z,N_j,d1_gridvals,d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% vfoptions are already set by ValueFnIter_FHorz()
% Handles vfoptions.divideandconquer==1, vfoptions.gridinterplayer==0, vfoptions.pard2==1

N_d1=prod(n_d1);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

%% Dispatch
% NOTE: I TRIED PARALLEL OVER d2, BUT IT SEEMED TO BE SLOWER RATHER THAN FASTER. NOT SURE WHY.
if N_d1==0
    error('vfoptions.pard2 not yet implemented for this combo')
else
    if N_e==0
        error('vfoptions.pard2 not yet implemented for this combo')
    else
        if N_z==0
            error('vfoptions.pard2 not yet implemented for this combo')
        else
            warning(' I TRIED PARALLEL OVER d2, BUT IT SEEMED TO BE SLOWER RATHER THAN FASTER. NOT SURE WHY.')
            [VKron, PolicyKron]=ValueFnIter_FHorz_SemiExo_pard2_DC1_e_raw(n_d1,n_d2,n_a,n_z,n_semiz,  vfoptions.n_e, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
end




%% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if vfoptions.outputkron==1
    V=VKron;
    Policy=PolicyKron;
    return
end

% Because of how we have N_semiz*N_z together, use the _z commands to UnKron
if N_z==0
    n_bothz=n_semiz;
else
    n_bothz=[n_semiz,n_z];
end


if N_d1==0
    error('vfoptions.pard2 not yet implemented for this combo')
else
    if N_e==0
        error('vfoptions.pard2 not yet implemented for this combo')
    else
        Policy=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron,[n_d1,n_d2,n_a],n_a,n_bothz,n_e,N_j,vfoptions);
    end
end


end
