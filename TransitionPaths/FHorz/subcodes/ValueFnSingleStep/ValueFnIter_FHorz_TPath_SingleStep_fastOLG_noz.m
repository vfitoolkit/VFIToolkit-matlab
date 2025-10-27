function [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_noz(VKron,n_d,n_a,N_j,d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The VKron input is next period value fn, the VKron output is this period.
% 'fastOLG' just means parallelizing across all of the "ages" (j) at once.

% V is done as a-by-j
% Policy is done as a-by-j
% (fastOLG is easy without z)

N_d=prod(n_d);
% z and e are handled elsewhere

if strcmp(vfoptions.exoticpreferences,'None')
    if vfoptions.divideandconquer==0
        if vfoptions.gridinterplayer==0
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_nod_noz_raw(VKron, n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_noz_raw(VKron, n_d, n_a, N_j, d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames);
            end
        else % vfoptions.gridinterplayer==1
            if N_d==0

            else

            end
        end
    elseif vfoptions.divideandconquer==1
        if vfoptions.gridinterplayer==0
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_nod_noz_raw(VKron, n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames,vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_noz_raw(VKron, n_d, n_a, N_j, d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames,vfoptions);
            end
        else % vfoptions.gridinterplayer==1
            if N_d==0

            else

            end
        end
    end
end

% Sometimes numerical rounding errors (of the order of 10^(-16) can mean
% that Policy is not integer valued. The following corrects this by converting to int64 and then
% makes the output back into double as Matlab otherwise cannot use it in
% any arithmetical expressions.
if vfoptions.policy_forceintegertype==1 || vfoptions.policy_forceintegertype==2
    PolicyKron=uint64(PolicyKron);
    PolicyKron=double(PolicyKron);
end

end
