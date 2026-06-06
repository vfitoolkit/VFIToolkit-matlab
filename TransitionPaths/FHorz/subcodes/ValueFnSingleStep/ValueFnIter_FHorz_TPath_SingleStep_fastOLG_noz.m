function [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_noz(VKron,n_d,n_a,N_j,d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The VKron input is next period value fn, the VKron output is this period.
% 'fastOLG' just means parallelizing across all of the "ages" (j) at once.

% V is done as a-by-j
% Policy is done as a-by-j
% (fastOLG is easy without z)

N_d=prod(n_d);
N_a=prod(n_a);
% z and e are handled elsewhere


if strcmp(vfoptions.exoticpreferences,'None')
    if vfoptions.divideandconquer==0
        if vfoptions.gridinterplayer==0
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_nod_noz_raw(VKron, n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_noz_raw(VKron, n_d, n_a, N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames);
            end
        else % vfoptions.gridinterplayer==1
            if isscalar(n_a)
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI1_nod_noz_raw(VKron, n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames,vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI1_noz_raw(VKron, n_d, n_a, N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames,vfoptions);
                end
            else % 2 endogenous states
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI2A_nod_noz_raw(VKron, n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames,vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI2A_noz_raw(VKron, n_d, n_a, N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames,vfoptions);
                end
            end
        end
    elseif vfoptions.divideandconquer==1
        if ~isscalar(n_a) && length(vfoptions.level1n)>1
            if vfoptions.level1n(2)>=n_a(2)
                vfoptions.level1n=vfoptions.level1n(1);
            else
                error('With two endogenous states, can only do divide-and-conquer in the first endogenous state (not in both)')
            end
        end
        if vfoptions.gridinterplayer==0
            if isscalar(n_a)
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_nod_noz_raw(VKron, n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames,vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_noz_raw(VKron, n_d, n_a, N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames,vfoptions);
                end
            else % 2 endogenous states
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC2A_nod_noz_raw(VKron, n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames,vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC2A_noz_raw(VKron, n_d, n_a, N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames,vfoptions);
                end
            end
        else % vfoptions.gridinterplayer==1
            if isscalar(n_a)
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_GI1_nod_noz_raw(VKron, n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames,vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_GI1_noz_raw(VKron, n_d, n_a, N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames,vfoptions);
                end
            else % 2 endogenous states
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC2A_GI2A_nod_noz_raw(VKron, n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames,vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC2A_GI2A_noz_raw(VKron, n_d, n_a, N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames,vfoptions);
                end
            end
        end
    end
elseif strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
    % V slot = Valt (Naive) / Vunderbar (Sophisticated); Policy = QH choice; drop QH-only extras
    [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_QH_fastOLG_noz(VKron,n_d,n_a,N_j,d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return % QH variant already un-Krons Policy
else
    error('Not yet implemented exoticpreferences for transtion paths (email me :)')
end


%% Policy in transition paths
if isscalar(n_a)
    if N_d==0
        PolicyKron=UnKronPolicyIndexes1_FHorz_noz(PolicyKron,n_a,N_a,N_j,vfoptions);
    else
        if vfoptions.gridinterplayer==0
            PolicyKron=UnKronPolicyIndexes1_FHorz_noz(PolicyKron,[n_d,n_a],N_a,N_j,vfoptions);
        else
            PolicyKron=UnKronPolicyIndexes2_FHorz_noz(PolicyKron,n_d,n_a,N_a,N_j,vfoptions);
        end
    end
else
    n_a1=n_a(1);
    n_a2=n_a(2:end);
    if N_d==0
        if vfoptions.gridinterplayer==0
            PolicyKron=UnKronPolicyIndexes1_FHorz_noz(PolicyKron,n_a,N_a,N_j,vfoptions);
        else
            PolicyKron=UnKronPolicyIndexes2_FHorz_noz(PolicyKron,n_a1,n_a2,N_a,N_j,vfoptions);
        end
    else
        if vfoptions.gridinterplayer==0
            PolicyKron=UnKronPolicyIndexes1_FHorz_noz(PolicyKron,[n_d,n_a],N_a,N_j,vfoptions);
        else
            PolicyKron=UnKronPolicyIndexes3_FHorz_noz(PolicyKron,n_d,n_a1,n_a2,N_a,N_j,vfoptions);
        end
    end
end

end
