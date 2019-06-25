function [V, Policy]=ValueFnIter_Case1_FHorz_EpsteinZin(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% DiscountFactorParamNames contains the names for the three parameters relating to
% Epstein-Zin preferences. Calling them beta, gamma, and psi,
% respectively the Epstein-Zin preferences are given by
% U_t= [ (1-beta)*u_t^(1-1/psi) + beta (E[(U_{t+1}^(1-gamma)])^((1-1/psi)/(1-gamma))]^(1/(1-1/psi))
% where
%  u_t is per-period utility function. c_t if just consuption, or ((c_t)^v(1-l_t)^(1-v)) if consumption and leisure (1-l_t)

V=nan;
Policy=nan;

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

% CANNOT YET DEAL WITH DYNASTY WHEN USING EPSTEIN-ZIN
% %% Deal with Dynasty_CareAboutDecendents if need to do that.
% if isfield(vfoptions,'Dynasty_CareAboutDecendents')==1
%     if vfoptions.verbose==1
%         fprintf('Dynasty_CareAboutDecendents option is being used \n')
%     end
%     if isfield(vfoptions,'tolerance')==0
%         vfoptions.tolerance=10^(-9);
%     end
%     
%     if vfoptions.parallel==2
%         if N_d==0
%             [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_no_d_Dynasty_raw(n_a, n_z, N_j, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
%         else
%             [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_Dynasty_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
%         end
%     elseif vfoptions.parallel==0 || vfoptions.parallel==1
%         if N_d==0
%             % Following command is somewhat misnamed, as actually does Par0 and Par1
%             [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_no_d_Par0_Dynasty_raw(n_a, n_z, N_j, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
%         else
%             % Following command is somewhat misnamed, as actually does Par0 and Par1
%             [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_Par0_Dynasty_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
%         end
%     end
%     
%     %Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
%     V=reshape(VKron,[n_a,n_z,N_j]);
%     Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, n_z, N_j,vfoptions);
%     
%     % Sometimes numerical rounding errors (of the order of 10^(-16) can mean
%     % that Policy is not integer valued. The following corrects this by converting to int64 and then
%     % makes the output back into double as Matlab otherwise cannot use it in
%     % any arithmetical expressions.
%     if vfoptions.policy_forceintegertype==1
%         fprintf('USING vfoptions to force integer... \n')
%         % First, give some output on the size of any changes in Policy as a
%         % result of turning the values into integers
%         temp=max(max(max(abs(round(Policy)-Policy))));
%         while ndims(temp)>1
%             temp=max(temp);
%         end
%         fprintf('  CHECK: Maximum change when rounding values of Policy is %8.6f (if these are not of numerical rounding error size then something is going wrong) \n', temp)
%         % Do the actual rounding to integers
%         Policy=round(Policy);
%         % Somewhat unrelated, but also do a double-check that Policy is now all positive integers
%         temp=min(min(min(Policy)));
%         while ndims(temp)>1
%             temp=min(temp);
%         end
%         fprintf('  CHECK: Minimum value of Policy is %8.6f (if this is <=0 then something is wrong) \n', temp)
%         %     Policy=uint64(Policy);
%         %     Policy=double(Policy);
%     end
%     
%     return
% end


%% Just do the standard case
if vfoptions.parallel==2
    if N_d==0
        [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_EpsteinZin_no_d_raw(n_a, n_z, N_j, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    else
        [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_EpsteinZin_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    end
elseif vfoptions.parallel==0 || vfoptions.parallel==1
    disp('ERROR: Epstein-Zin currently only implemented for Parallel=2: email robertdkirkby@gmail.com')
    dbstack
    return
%     if N_d==0
%         % Following command is somewhat misnamed, as actually does Par0 and Par1
%         [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_EpsteinZin_no_d_Par0_raw(n_a, n_z, N_j, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
%     else
%         % Following command is somewhat misnamed, as actually does Par0 and Par1
%         [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_EpsteinZin_Par0_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
%     end
end

%Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
V=reshape(VKron,[n_a,n_z,N_j]);
Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, n_z, N_j,vfoptions);

% Sometimes numerical rounding errors (of the order of 10^(-16) can mean
% that Policy is not integer valued. The following corrects this by converting to int64 and then
% makes the output back into double as Matlab otherwise cannot use it in
% any arithmetical expressions.
if vfoptions.policy_forceintegertype==1
    fprintf('USING vfoptions to force integer... \n')
    % First, give some output on the size of any changes in Policy as a
    % result of turning the values into integers
    temp=max(max(max(abs(round(Policy)-Policy))));
    while ndims(temp)>1
        temp=max(temp);
    end
    fprintf('  CHECK: Maximum change when rounding values of Policy is %8.6f (if these are not of numerical rounding error size then something is going wrong) \n', temp)
    % Do the actual rounding to integers
    Policy=round(Policy);
    % Somewhat unrelated, but also do a double-check that Policy is now all positive integers
    temp=min(min(min(Policy)));
    while ndims(temp)>1
        temp=min(temp);
    end
    fprintf('  CHECK: Minimum value of Policy is %8.6f (if this is <=0 then something is wrong) \n', temp)
%     Policy=uint64(Policy);
%     Policy=double(Policy);
elseif vfoptions.policy_forceintegertype==2
    % Do the actual rounding to integers
    Policy=round(Policy);
end

end