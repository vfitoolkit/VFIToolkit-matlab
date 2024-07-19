function [VKron, PolicyKron]=ValueFnIter_Case1_TPath_SingleStep(VKron,n_d,n_a,n_z,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The VKron input is next period value fn, the VKron output is this period.

% VKron=reshape(VKron,[prod(n_a),prod(n_z),N_j]);
PolicyKron=nan;

%% Check which vfoptions have been used, set all others to defaults 
if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
    vfoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    vfoptions.returnmatrix=2;
    vfoptions.verbose=0;
    vfoptions.lowmemory=0;
    vfoptions.exoticpreferences='None';
    vfoptions.polindorval=1;
    vfoptions.policy_forceintegertype=0;
    vfoptions.solnmethod='purediscretization';
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if isfield(vfoptions,'parallel')==0
        vfoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    end
    if isfield(vfoptions,'verbose')==0
        vfoptions.verbose=0;
    end
    if isfield(vfoptions,'lowmemory')==0
        vfoptions.lowmemory=0;
    end
    if isfield(vfoptions,'exoticpreferences')==0
        vfoptions.exoticpreferences='None';
    end
    if isfield(vfoptions,'polindorval')==0
        vfoptions.polindorval=1;
    end
    if isfield(vfoptions,'policy_forceintegertype')==0
        vfoptions.policy_forceintegertype=0;
    end
    if isfield(vfoptions,'solnmethod')==0
        vfoptions.solnmethod='purediscretization';
    end
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

%% 
% if vfoptions.parallel==2 
% If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
pi_z=gpuArray(pi_z);
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
z_grid=gpuArray(z_grid);

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

%%
if isfield(vfoptions,'StateDependentVariables_z')==1
    dbstack
    error('Not yet supported')
end

%% If get to here then not using exoticpreferences nor StateDependentVariables_z
if N_d==0
    [VKron,PolicyKron]=ValueFnIter_Case1_TPath_SingleStep_no_d_raw(VKron,n_a, n_z, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
else
    if strcmp(vfoptions.solnmethod,'purediscretization')
        [VKron, PolicyKron]=ValueFnIter_Case1_TPath_SingleStep_raw(VKron,n_d,n_a,n_z, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    elseif strcmp(vfoptions.solnmethod,'purediscretization_refinement')
        % COMMENT: testing a transition in model of Pijoan-Mas (2006) it
        % seems refirement is slower for transtions, so this is never
        % really used for anything.
        [VKron, PolicyKron]=ValueFnIter_Case1_TPath_SingleStep_Refine_raw(VKron,n_d,n_a,n_z, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    end
end

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