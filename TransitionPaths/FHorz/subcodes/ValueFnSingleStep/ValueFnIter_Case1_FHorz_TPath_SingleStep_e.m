function [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_TPath_SingleStep_e(VKron,n_d,n_a,n_z,n_e,N_j,d_grid, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The VKron input is next period value fn, the VKron output is this period.

N_d=prod(n_d);

%% If get to here then not using exoticpreferences nor StateDependentVariables_z
% N_z==0 is handled by a different command
if N_d==0
    [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_TPath_SingleStep_nod_e_raw(VKron,n_a, n_z,n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
else
    [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_TPath_SingleStep_e_raw(VKron,n_d,n_a,n_z,n_e, N_j, d_grid, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
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