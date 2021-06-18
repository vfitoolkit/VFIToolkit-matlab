function [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_TPath_SingleStep_fastOLG(VKron,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The VKron input is next period value fn, the VKron output is this period.
% 'fastOLG' just means parallelizing across all of the "ages" (j) at once.

% DEBUGGING
vfoptions

% VKron=reshape(VKron,[prod(n_a),prod(n_z),N_j]);
PolicyKron=nan;

%% Check which vfoptions have been used, set all others to defaults 
if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
%     vfoptions.exoticpreferences=0;
    vfoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    vfoptions.returnmatrix=2;
    vfoptions.verbose=0;
    vfoptions.lowmemory=0;
    vfoptions.exoticpreferences=0;
    vfoptions.polindorval=1;
    vfoptions.policy_forceintegertype=0;
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if isfield(vfoptions,'parallel')==0
        vfoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    end
    if vfoptions.parallel==2
        vfoptions.returnmatrix=2; % On GPU, must use this option
    end
    if isfield(vfoptions,'lowmemory')==0
        vfoptions.lowmemory=0;
    end
    if isfield(vfoptions,'verbose')==0
        vfoptions.verbose=0;
    end
    if isfield(vfoptions,'returnmatrix')==0
        if isa(ReturnFn,'function_handle')==1
            vfoptions.returnmatrix=0;
        else
            vfoptions.returnmatrix=1;
        end
    end
    if isfield(vfoptions,'exoticpreferences')==0
        vfoptions.exoticpreferences=0;
    end
    if isfield(vfoptions,'polindorval')==0
        vfoptions.polindorval=1;
    end
    if isfield(vfoptions,'policy_forceintegertype')==0
        vfoptions.policy_forceintegertype=0;
    end
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

%% Check the sizes of some of the inputs
if size(d_grid)~=[N_d, 1]
    disp('ERROR: d_grid is not the correct shape (should be  of size N_d-by-1)')
    dbstack
    return
elseif size(a_grid)~=[N_a, 1]
    disp('ERROR: a_grid is not the correct shape (should be  of size N_a-by-1)')
    dbstack
    return
elseif size(z_grid)~=[N_z, 1]
    disp('ERROR: z_grid is not the correct shape (should be  of size N_z-by-1)')
    dbstack
    return
elseif size(pi_z)~=[N_z, N_z]
    disp('ERROR: pi is not of size N_z-by-N_z')
    dbstack
    return
end


%% 
if vfoptions.parallel==2 
   % If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
   pi_z=gpuArray(pi_z);
   d_grid=gpuArray(d_grid);
   a_grid=gpuArray(a_grid);
   z_grid=gpuArray(z_grid);
else
   % If using CPU make sure all the relevant inputs are CPU arrays (not standard arrays)
   % This may be completely unnecessary.
   pi_z=gather(pi_z);
   d_grid=gather(d_grid);
   a_grid=gather(a_grid);
   z_grid=gather(z_grid);
end

if vfoptions.verbose==1
    vfoptions
end

if vfoptions.exoticpreferences==0
    if vfoptions.lowmemory==0
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_TPath_SingleStep_fastOLG_no_d_raw(VKron,n_a, n_z, N_j, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_TPath_SingleStep_fastOLG_raw(VKron,n_d,n_a,n_z, N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    elseif vfoptions.lowmemory==1
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_TPath_SingleStep_fastOLGlowmem_no_d_raw(VKron,n_a, n_z, N_j, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_TPath_SingleStep_fastOLGlowmem_raw(VKron,n_d,n_a,n_z, N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
elseif vfoptions.exoticpreferences==1 % Multiple discount factors. It is assumed that the product
    %NOT YET IMPLEMENTED
%    [V, Policy]=ValueFnIter_Case1_QuasiGeometric(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, vfoptions,Parameters,ReturnFnParamNames);
%    return
elseif vfoptions.exoticpreferences==2 % Epstein-Zin preferences
    if N_d==0
        [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_EpZin_TPath_SingleStep_fastOLG_no_d_raw(VKron,n_a, n_z, N_j, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    else
        [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_EpZin_TPath_SingleStep_fastOLG_raw(VKron,n_d,n_a,n_z, N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    end
    return
end

%% % NOT YET IMPLEMENTED
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