function [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_TPath_SingleStep_Par1(VKron,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The VKron input is next period value fn, the VKron output is this period.

% VKron=reshape(VKron,[prod(n_a),prod(n_z),N_j]);
PolicyKron=nan;

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

% %% Check the sizes of some of the inputs
% if size(d_grid)~=[N_d, 1]
%     disp('ERROR: d_grid is not the correct shape (should be  of size N_d-by-1)')
%     dbstack
%     return
% elseif size(a_grid)~=[N_a, 1]
%     disp('ERROR: a_grid is not the correct shape (should be  of size N_a-by-1)')
%     dbstack
%     return
% elseif size(z_grid)~=[N_z, 1]
%     disp('ERROR: z_grid is not the correct shape (should be  of size N_z-by-1)')
%     dbstack
%     return
% elseif size(pi_z)~=[N_z, N_z]
%     disp('ERROR: pi is not of size N_z-by-N_z')
%     dbstack
%     return
% end

if vfoptions.verbose==1
    vfoptions
end

% if vfoptions.exoticpreferences==0
%     if length(DiscountFactorParamNames)~=1
%         disp('WARNING: There should only be a single Discount Factor (in DiscountFactorParamNames) when using standard VFI')
%         dbstack
%     end
% elseif vfoptions.exoticpreferences==1 % Multiple discount factors. It is assumed that the product
%     %NOT YET IMPLEMENTED
% %    [V, Policy]=ValueFnIter_Case1_QuasiGeometric(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, vfoptions,Parameters,ReturnFnParamNames);
% %    return
% elseif vfoptions.exoticpreferences==2 % Epstein-Zin preferences
%     %NOT YET IMPLEMENTED
% %     [V, Policy]=ValueFnIter_Case1_EpsteinZin(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, vfoptions,Parameters,ReturnFnParamNames);
% %     return
% end

% %%
% if isfield(vfoptions,'StateDependentVariables_z')==1
%     if vfoptions.verbose==1
%         fprintf('StateDependentVariables_z option is being used \n')
%     end
%     
%     if N_d==0
%         [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_TPath_SingleStep_no_d_SDVz_raw(VKron,n_a, n_z, N_j, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
%     else
%         [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_TPath_SingleStep_SDVz_raw(VKron,n_d,n_a,n_z, N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
%     end
%     
% %     %Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
% %     V=reshape(VKron,[n_a,n_z,N_j]);
% %     Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, n_z, N_j,vfoptions);
%     
%     % Sometimes numerical rounding errors (of the order of 10^(-16) can mean
%     % that Policy is not integer valued. The following corrects this by converting to int64 and then
%     % makes the output back into double as Matlab otherwise cannot use it in
%     % any arithmetical expressions.
%     if vfoptions.policy_forceintegertype==1
%         PolicyKron=uint64(PolicyKron);
%         PolicyKron=double(PolicyKron);
%     end
%     
%     return
% end

%% 
if N_d==0
    [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_TPath_SingleStep_no_d_Par1_raw(VKron,n_a, n_z, N_j, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
else
    [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_TPath_SingleStep_Par1_raw(VKron,n_d,n_a,n_z, N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
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