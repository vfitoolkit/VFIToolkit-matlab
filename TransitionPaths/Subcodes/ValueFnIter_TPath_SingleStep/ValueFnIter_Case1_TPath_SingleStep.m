function [VKron, PolicyKron]=ValueFnIter_Case1_TPath_SingleStep(VKron,n_d,n_a,n_z,d_grid, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The VKron input is next period value fn, the VKron output is this period.

% vfoptions must be already fully set up (this command is for internal use only so it should be)

N_d=prod(n_d);
% N_a=prod(n_a);
% N_z=prod(n_z);

%% 
% If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
pi_z=gpuArray(pi_z);
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
z_gridvals=gpuArray(z_gridvals);

if vfoptions.verbose==1
    vfoptions
end

if strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
    dbstack
    error('Not yet supported')
elseif strcmp(vfoptions.exoticpreferences,'EpsteinZin')
    dbstack
    error('Not yet supported')
end

%% Solve the standard problem
% Note: being infinite horizon, I don't imagine anyone will come here without z variable
if vfoptions.divideandconquer==0
    if N_d==0
        [VKron,PolicyKron]=ValueFnIter_Case1_TPath_SingleStep_nod_raw(VKron,n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    else
        [VKron, PolicyKron]=ValueFnIter_Case1_TPath_SingleStep_raw(VKron,n_d,n_a,n_z, d_grid, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    end
elseif vfoptions.divideandconquer==1
    if length(n_a)==1
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_Case1_TPath_SingleStep_DC1_nod_raw(VKron,n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_Case1_TPath_SingleStep_DC1_raw(VKron,n_d,n_a,n_z, d_grid, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else
        error('vfoptions.divideandconquer=1 only implemented for length(n_a)=1; only when there is just one endogenous state')
    end
end

% if strcmp(vfoptions.solnmethod,'purediscretization_refinement')
%     % COMMENT: testing a transition in model of Pijoan-Mas (2006) it
%     % seems refirement is slower for transtions, so this is never
%     % really used for anything.
%     [VKron, PolicyKron]=ValueFnIter_Case1_TPath_SingleStep_Refine_raw(VKron,n_d,n_a,n_z, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
% end

%%
% %Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
% V=reshape(VKron,[n_a,n_z]);
% Policy=UnKronPolicyIndexes_Case1(PolicyKron, n_d, n_a, n_z,vfoptions);

% Sometimes numerical rounding errors (of the order of 10^(-16) can mean
% that Policy is not integer valued. The following corrects this by converting to int64 and then
% makes the output back into double as Matlab otherwise cannot use it in
% any arithmetical expressions.
if vfoptions.policy_forceintegertype==1 || vfoptions.policy_forceintegertype==2
    PolicyKron=uint64(PolicyKron);
    PolicyKron=double(PolicyKron);
end

end