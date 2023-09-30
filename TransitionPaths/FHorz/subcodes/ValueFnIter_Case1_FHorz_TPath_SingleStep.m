function [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_TPath_SingleStep(VKron,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The VKron input is next period value fn, the VKron output is this period.

% VKron=reshape(VKron,[prod(n_a),prod(n_z),N_j]);
PolicyKron=nan;

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=0;
if isfield(vfoptions,'n_e')
    N_e=prod(vfoptions.n_e);
end


% if strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
%     if strcmp(vfoptions.quasi_hyperbolic,'Naive')
%         if N_d==0
%             [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_NQHyperbolic_SingleStep_no_d_raw(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid_J, pi_z_J, DiscountFactorParamNames, ReturnFn, vfoptions,Parameters,ReturnFnParamNames);
%         else
%             [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_NQHyperbolic_SingleStep_raw(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid_J, pi_z_J, DiscountFactorParamNames, ReturnFn, vfoptions,Parameters,ReturnFnParamNames);
%         end
%     elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
%         if N_d==0
%             [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_SQHyperbolic_SingleStep_no_d_raw(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid_J, pi_z_J, DiscountFactorParamNames, ReturnFn, vfoptions,Parameters,ReturnFnParamNames);
%         else
%             [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_SQHyperbolic_SingleStep_raw(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid_J, pi_z_J, DiscountFactorParamNames, ReturnFn, vfoptions,Parameters,ReturnFnParamNames);
%         end
%     end
% elseif strcmp(vfoptions.exoticpreferences,'EpsteinZin')
%     if N_d==0
%         [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_EpZin_TPath_SingleStep_no_d_raw(VKron,n_a, n_z, N_j, a_grid, z_grid_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
%     else
%         [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_EpZin_TPath_SingleStep_raw(VKron,n_d,n_a,n_z, N_j, d_grid, a_grid, z_grid_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
%     end
% end


%% If get to here then not using exoticpreferences nor StateDependentVariables_z
% N_z==0 is handled by a different command
if N_d==0
    if N_e==0
        [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_TPath_SingleStep_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_grid_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    else
        [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_TPath_SingleStep_nod_e_raw(VKron,n_a, n_z,vfoptions.n_e, N_j, a_grid, z_grid_J, vfoptions.e_grid_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    end
else
    if N_e==0
        [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_TPath_SingleStep_raw(VKron,n_d,n_a,n_z, N_j, d_grid, a_grid, z_grid_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    else
        [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_TPath_SingleStep_e_raw(VKron,n_d,n_a,n_z,vfoptions.n_e, N_j, d_grid, a_grid, z_grid_J, vfoptions.e_grid_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    end
end


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