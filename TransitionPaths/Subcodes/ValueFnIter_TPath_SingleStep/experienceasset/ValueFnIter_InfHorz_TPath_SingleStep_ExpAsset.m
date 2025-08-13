function [VKron, PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_ExpAsset(VKron,n_d1,n_d2,n_a1,n_a2,n_z,d1_grid,d2_grid, a1_grid, a2_grid, z_gridvals, pi_z, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% The VKron input is next period value fn, the VKron output is this period.

% vfoptions must be already fully set up (this command is for internal use only so it should be)

N_d1=prod(n_d1);
% N_d2=prod(n_d2);
N_a1=prod(n_a1);
% N_z=prod(n_z);

%%
if vfoptions.divideandconquer==1
    vfoptions.level1n=vfoptions.level1n(1);
end

if strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
    dbstack
    error('QuasiHyperbolic Preferences Not yet supported')
elseif strcmp(vfoptions.exoticpreferences,'EpsteinZin')
    dbstack
    error('EpsteinZin Preferences Not yet supported')
end

%% Solve the standard problem
% Note: being infinite horizon, I don't imagine anyone will come here without z variable
if vfoptions.divideandconquer==0
    if N_a1==0
        error('Not yet implemented InfHorz TPath with experienceasset without a second standard endogenous state')
    else % N_a1
        if N_d1==0
            [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_ExpAsset_nod1_raw(VKron, n_d2, n_a1, n_a2, n_z, d2_grid, a1_grid, a2_grid, z_gridvals, pi_z, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_ExpAsset_raw(VKron, n_d1, n_d2, n_a1, n_a2, n_z, d1_grid, d2_grid, a1_grid, a2_grid, z_gridvals, pi_z, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    end
elseif vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==0
    if N_a1==0
        error('Not yet implemented InfHorz TPath with experienceasset and divide-and-conquer without a second standard endogenous state')
    else % N_a1
        if N_d1==0
            [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_ExpAsset_DC1_nod1_raw(VKron,n_d2, n_a1, n_a2, n_z, d2_grid, a1_grid, a2_grid, z_gridvals, pi_z, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            % [VKron, PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_ExpAsset_DC1_raw(VKron,n_d,n_a,n_z, d_grid, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
elseif vfoptions.divideandconquer==0 && vfoptions.gridinterplayer==1
    error('Have not yet implemented combo of vfoptions.gridinterplayer=1 with vfoptions.divideandconquer=0')
elseif vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==1
    if N_a1==0
        error('Not yet implemented InfHorz TPath with experienceasset and divide-and-conquer and grid interpolation layer without a second standard endogenous state')
    else % N_a1
        if N_d1==0
            % [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_ExpAsset_DC1_GI_nod1_raw(VKron,n_d2, n_a1, n_a2, n_z, d2_grid, a1_grid, a2_grid, z_gridvals, pi_z, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            % [VKron, PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_ExpAsset_DC1_GI_raw(VKron,n_d,n_a,n_z, d_grid, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
end


%%
% Sometimes numerical rounding errors (of the order of 10^(-16) can mean
% that Policy is not integer valued. The following corrects this by converting to int64 and then
% makes the output back into double as Matlab otherwise cannot use it in
% any arithmetical expressions.
if vfoptions.policy_forceintegertype==1 || vfoptions.policy_forceintegertype==2
    PolicyKron=uint64(PolicyKron);
    PolicyKron=double(PolicyKron);
end

end
