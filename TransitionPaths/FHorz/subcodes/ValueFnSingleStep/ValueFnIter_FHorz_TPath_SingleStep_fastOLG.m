function [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG(VKron,n_d,n_a,n_z,N_j,d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The VKron input is next period value fn, the VKron output is this period.
% 'fastOLG' just means parallelizing across all of the "ages" (j) at once.

% V is done as (a,j)-by-z [this form makes the expectations easier]
% Policy is done as a-by-j-by-z [this form is easier later, and easier for handling DC1]
% (fastOLG requires swapping order of j and z)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
% N_z=0 is handled elsewhere
% N_e is handled elsewhere

if strcmp(vfoptions.exoticpreferences,'None')
    if vfoptions.divideandconquer==0
        if vfoptions.gridinterplayer==0
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else % vfoptions.gridinterplayer==1
            if isscalar(n_a)
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI1_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI1_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else % 2 endogenous states
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI2A_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI2A_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
        end
    else % vfoptions.divideandconquer==1
        if ~isscalar(n_a) && length(vfoptions.level1n)>1
            if vfoptions.level1n(2)>=n_a(2)
                vfoptions.level1n=vfoptions.level1n(1); % DC2A: only first endo dim is divide-and-conquered
            else
                error('With two endogenous states, can only do divide-and-conquer in the first endogenous state (not in both)')
            end
        end
        if vfoptions.gridinterplayer==0
            if isscalar(n_a)
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else % 2 endogenous states
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC2A_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC2A_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
        else % vfoptions.gridinterplayer==1
            if isscalar(n_a)
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_GI1_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_GI1_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else % 2 endogenous states
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC2A_GI2A_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC2A_GI2A_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
        end

    end
elseif strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
    % V slot = Valt (Naive) / Vunderbar (Sophisticated); Policy = QH choice; drop QH-only extras
    [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_QH_fastOLG(VKron,n_d,n_a,n_z,N_j,d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return % QH variant already un-Krons Policy
else
    error('Not yet implemented exoticpreferences for transtion paths (email me :)')
end


%% Policy in transition paths
% Note: The actual ordering of N_z,N_j is not relevant to how this command works, so can just mix them up. [as long as N_z not n_z]
if isscalar(n_a)
    if N_d==0
        PolicyKron=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_a,N_a,N_j,N_z,vfoptions);
    else
        if vfoptions.gridinterplayer==0
            PolicyKron=UnKronPolicyIndexes1_FHorz_z(PolicyKron,[n_d,n_a],N_a,N_j,N_z,vfoptions);
        else
            PolicyKron=UnKronPolicyIndexes2_FHorz_z(PolicyKron,n_d,n_a,N_a,N_j,N_z,vfoptions);
        end
    end
else
    n_a1=n_a(1);
    n_a2=n_a(2:end);
    if N_d==0
        if vfoptions.gridinterplayer==0
            PolicyKron=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_a,N_a,N_j,N_z,vfoptions);
        else
            PolicyKron=UnKronPolicyIndexes2_FHorz_z(PolicyKron,n_a1,n_a2,N_a,N_j,N_z,vfoptions);
        end
    else
        if vfoptions.gridinterplayer==0
            PolicyKron=UnKronPolicyIndexes1_FHorz_z(PolicyKron,[n_d,n_a],N_a,N_j,N_z,vfoptions);
        else
            PolicyKron=UnKronPolicyIndexes3_FHorz_z(PolicyKron,n_d,n_a1,n_a2,N_a,N_j,N_z,vfoptions);
        end
    end
end

end
