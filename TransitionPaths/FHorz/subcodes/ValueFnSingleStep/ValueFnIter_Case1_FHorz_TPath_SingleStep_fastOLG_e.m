function [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_TPath_SingleStep_fastOLG_e(VKron,n_d,n_a,n_z,n_e,N_j,d_grid, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The VKron input is next period value fn, the VKron output is this period.
% 'fastOLG' just means parallelizing across all of the "ages" (j) at once.

% VKron=reshape(VKron,[prod(n_a),prod(n_z),N_j]);
PolicyKron=nan;

N_d=prod(n_d);

if strcmp(vfoptions.exoticpreferences,'None')
    if vfoptions.divideandconquer==0
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_TPath_SingleStep_fastOLG_nod_e_raw(VKron,n_a, n_z, n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_TPath_SingleStep_fastOLG_e_raw(VKron,n_d,n_a,n_z, n_e, N_j, d_grid, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else % vfoptions.divideandconquer==1

    end
else 
    error('Not yet implemented exoticpreferences for transtion paths (email me :)')
end


%%
% %Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
% V=reshape(VKron,[n_a,n_z,N_j]);
% Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, n_z, N_j,vfoptions);

% Sometimes numerical rounding errors (of the order of 10^(-16) can mean
% that Policy is not integer valued. The following corrects this by converting to int64 and then
% makes the output back into double as Matlab otherwise cannot use it in
% any arithmetical expressions.
if vfoptions.policy_forceintegertype==1 || vfoptions.policy_forceintegertype==2
    PolicyKron=uint64(PolicyKron);
    PolicyKron=double(PolicyKron);
end

end