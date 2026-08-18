function [V,Policy]=ValueFnIter_FHorz_EpsteinZin_SemiExo(n_d,n_a,n_z,n_semiz,N_j,d_grid,a_grid,z_gridvals_J,semiz_gridvals_J,pi_z_J,pi_semiz_J,ReturnFn,Parameters,DiscountFactorParamNames,ReturnFnParamNames,vfoptions,sj,warmglow,ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Epstein-Zin with semi-exogenous shocks: splits d into d1/d2, routes to the
% EpsteinZinSemiExo raws, and does its own UnKron (mirrors
% ValueFnIter_FHorz_QuasiHyperbolicSemiExo / ValueFnIter_FHorz_SemiExo).
% The ezc1-ezc8/sj/warmglow preamble is done by the caller
% (ValueFnIter_FHorz_EpsteinZin) and just passed through to the raws.
% vfoptions.l_dsemiz gives the number of decision variables that control the
% semi-exogenous transitions (they must be the last decision variables).

%% Split n_d into n_d1 (other decisions) and n_d2 (semiz controller)
l_dsemiz=vfoptions.l_dsemiz;
if length(n_d)==l_dsemiz
    n_d1=0;
    n_d2=n_d;
else
    n_d1=n_d(1:end-l_dsemiz);
    n_d2=n_d(end-l_dsemiz+1:end);
end

N_d1=prod(n_d1);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

% Split d_grid into d1_grid and d2_grid, and create the gridvals
if N_d1==0
    d1_gridvals=[];
    d2_grid=d_grid;
    d2_gridvals=CreateGridvals(n_d2,d2_grid,1);
else
    d1_grid=d_grid(1:sum(n_d1));
    d2_grid=d_grid(sum(n_d1)+1:end);
    d1_gridvals=CreateGridvals(n_d1,d1_grid,1);
    d2_gridvals=CreateGridvals(n_d2,d2_grid,1);
end

%% Divide-and-conquer and/or grid interpolation route to their own subfns (steps 2 and 3 of the EZ-SemiExo build)
if vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==1
    error('Epstein-Zin with semi-exogenous shocks plus divide-and-conquer plus the grid interpolation layer is not yet implemented')
elseif vfoptions.divideandconquer==1
    error('Epstein-Zin with semi-exogenous shocks plus divide-and-conquer is not yet implemented')
elseif vfoptions.gridinterplayer==1
    error('Epstein-Zin with semi-exogenous shocks plus the grid interpolation layer is not yet implemented')
end

%%
if N_d1==0
    if N_e==0
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_EpsteinZin_SemiExo_nod1_noz_raw(n_d2,n_a,n_semiz, N_j, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_EpsteinZin_SemiExo_nod1_raw(n_d2,n_a,n_z,n_semiz, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        end
    else
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_EpsteinZin_SemiExo_nod1_noz_e_raw(n_d2,n_a,n_semiz, vfoptions.n_e, N_j, d2_gridvals, a_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_EpsteinZin_SemiExo_nod1_e_raw(n_d2,n_a,n_z,n_semiz,  vfoptions.n_e, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        end
    end
else
    if N_e==0
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_EpsteinZin_SemiExo_noz_raw(n_d1,n_d2,n_a,n_semiz, N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_EpsteinZin_SemiExo_raw(n_d1,n_d2,n_a,n_z,n_semiz, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        end
    else
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_EpsteinZin_SemiExo_noz_e_raw(n_d1,n_d2,n_a,vfoptions.n_semiz, vfoptions.n_e, N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_EpsteinZin_SemiExo_e_raw(n_d1,n_d2,n_a,n_z,vfoptions.n_semiz,  vfoptions.n_e, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
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
    n_bothz=vfoptions.n_semiz;
else
    n_bothz=[vfoptions.n_semiz,n_z];
end

% First dimension of PolicyKron is (d1,d2,aprime), or if no d1, then (d2,aprime)
if N_d1==0
    if N_e==0
        V=reshape(VKron,[n_a,n_bothz,N_j]);
        Policy=UnKronPolicyIndexes2_FHorz_z(PolicyKron,n_d2,n_a,n_a,n_bothz,N_j,vfoptions);
    else
        V=reshape(VKron,[n_a,n_bothz, vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes2_FHorz_z_e(PolicyKron,n_d2,n_a,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
    end
else
    if N_e==0
        V=reshape(VKron,[n_a,n_bothz,N_j]);
        Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron,n_d1,n_d2,n_a,n_a,n_bothz,N_j,vfoptions);
    else
        V=reshape(VKron,[n_a,n_bothz, vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes3_FHorz_z_e(PolicyKron,n_d1,n_d2,n_a,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
    end
end



end
