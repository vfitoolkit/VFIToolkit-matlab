function [V, Policy]=ValueFnIter_FieldExp_FHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TreatmentParams, TreatmentAgeRange, TreatmentDuration, vfoptions)

V=struct();
Policy=struct();

%% Check which vfoptions have been used, set all others to defaults 
if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
    vfoptions.parallel=1+(gpuDeviceCount>0);
    vfoptions.verbose=0;
    vfoptions.lowmemory=0;
    % Model setup
    % vfoptions.exoticpreferences='None';
    % vfoptions.dynasty=0; % Dynasty not supported for Field experiments
    vfoptions.incrementaltype=0; % (vector indicating endogenous state is an incremental endogenous state variable)
    % Exogenous shocks
    vfoptions.n_e=0;
    vfoptions.n_semiz=0;
    % Internal use only
    vfoptions.outputkron=0; % If 1, leave output in Kron form
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(vfoptions,'parallel')
        vfoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(vfoptions,'lowmemory')
        vfoptions.lowmemory=0;
    end
    if ~isfield(vfoptions,'verbose')
        vfoptions.verbose=0;
    end
    % Model setup
    if ~isfield(vfoptions,'incrementaltype')
        vfoptions.incrementaltype=0; % (vector indicating endogenous state is an incremental endogenous state variable)
    end
    % vfoptions.exoticpreferences='None';
    % vfoptions.dynasty=0; % Dynasty not supported for Field experiments
    % Exogenous shocks
    vfoptions.n_e=0;
    vfoptions.n_semiz=0;
    if isfield(vfoptions,'ExogShockFn')
        vfoptions.ExogShockFnParamNames=getAnonymousFnInputNames(vfoptions.ExogShockFn);
    end
    if isfield(vfoptions,'EiidShockFn')
        vfoptions.EiidShockFnParamNames=getAnonymousFnInputNames(vfoptions.EiidShockFn);
    end
    % Internal use only
    if ~isfield(vfoptions,'outputkron')
        vfoptions.outputkron=0; % If 1, leave output in Kron form
    end
end

% Can skip checking input sizes as this will be done when solving for the control group anyway.

%%
if isempty(n_d)
    n_d=0;
end
if isempty(n_z)
    n_z=0;
end
N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

%% If quasi-hyperbolic discounting is used, send it off to be treated specially
if isfield(vfoptions,'exoticpreferences')
    if strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
        [V, Policy]=ValueFnIter_FieldExp_FHorz_QuasiHyperbolic(n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        return
    end
else
    vfoptions.exoticpreferences='None';
end



%% First, just solve the control group
if vfoptions.verbose==1
    fprintf('Field-experiment: Solving control-group \n')
end
[V.control, Policy.control]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

% I want the Kron versions of V.control as it makes combining the treatment and control easier (which is done below)
if N_e==0
    if N_z==0
        VcontrolKron=reshape(V.control,[N_a,N_j]);
        PolicycontrolKron=KronPolicyIndexes_FHorz_Case1_noz(Policy.control, n_d, n_a, N_j);
    else
        VcontrolKron=reshape(V.control,[N_a,N_z,N_j]);
        PolicycontrolKron=KronPolicyIndexes_FHorz_Case1(Policy.control, n_d, n_a, n_z, N_j);
    end
else
    if N_z==0
        VcontrolKron=reshape(V.control,[N_a,N_e,N_j]);
        PolicycontrolKron=KronPolicyIndexes_FHorz_Case1(Policy.control, n_d, n_a, vfoptions.n_e, N_j); % Treat e as z (because no z)
    else
        VcontrolKron=reshape(V.control,[N_a,N_z,N_e,N_j]);
        PolicycontrolKron=KronPolicyIndexes_FHorz_Case1(Policy.control, n_d, n_a, n_z, N_j, vfoptions.n_e);
    end
end


%% Implement new way of handling ReturnFn inputs
if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);
if N_z==0
    l_z=0;
end
if N_e==0
    l_e=0;
else
    l_e=length(vfoptions.n_e);
end
% If no ReturnFnParamNames inputted, then figure it out from ReturnFn
if isempty(ReturnFnParamNames)
    temp=getAnonymousFnInputNames(ReturnFn);
    if length(temp)>(l_d+l_a+l_a+l_z+l_e) % This is largely pointless, the ReturnFn is always going to have some parameters
        ReturnFnParamNames={temp{l_d+l_a+l_a+l_z+l_e+1:end}}; % the first inputs will always be (d,aprime,a,z)
    else
        ReturnFnParamNames={};
    end
end


%% Exogenous shock grids
[z_gridvals_J,pi_z_J,vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3);

%% 
% If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);

if vfoptions.verbose==1
    vfoptions
end


%% Change the parameters to treatment parameters
treatparamnames=fieldnames(TreatmentParams);
for nn=1:length(treatparamnames)
    Parameters.(treatparamnames{nn})=TreatmentParams.(treatparamnames{nn});
end

% Identify the age-dependent parameters so that we can easily use just the
% parts relevant to the treatment ages
AgeDepParams=struct();
paramnames=fieldnames(Parameters);
for nn=1:length(paramnames)
    if length(Parameters.(paramnames{nn}))==N_j
        AgeDepParams.(paramnames{nn})=Parameters.(paramnames{nn});
    end
end
agedepparamnames=fieldnames(AgeDepParams);

vfoptions.outputkron=1;
%% Now, loop over the ages in 
for j_p=TreatmentAgeRange(1):TreatmentAgeRange(2)
    if vfoptions.verbose==1
        fprintf('Field-experiment: Solving treatment-group initial treatment age %i (of %i to %i) \n',j_p,TreatmentAgeRange(1),TreatmentAgeRange(2))
    end

    % First, get the 'V in period j_p+j_d' from V.control
    if N_z==0 && N_e==0
        vfoptions.V_Jplus1=VcontrolKron(:,j_p+TreatmentDuration); % (a,j)
    elseif N_z>0 && N_e>0
        vfoptions.V_Jplus1=VcontrolKron(:,:,:,j_p+TreatmentDuration); % (a,z,e,j)
    else % One of z and e is used
        vfoptions.V_Jplus1=VcontrolKron(:,:,j_p+TreatmentDuration); % (a,z/e,j)
    end
    % Instead of N_j we are going to use TreatmentDuration

    % We need to replace the age-dependent parameters with just the relevant periods.
    for nn=1:length(agedepparamnames)
        temp=AgeDepParams.(agedepparamnames{nn});
        Parameters.(agedepparamnames{nn})=temp(j_p:j_p+TreatmentDuration-1);
    end

    %% If using exotic preferences, do those
    if strcmp(vfoptions.exoticpreferences,'EpsteinZin')
        % Note: TreatmentDuration periods
        % Note: vfoptions.outputkron=1;
        [VKron_jp, PolicyKron_jp]=ValueFnIter_Case1_FHorz_EpsteinZin(n_d,n_a,n_z,TreatmentDuration,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    else
        %% Otherwise just do the standard case
        if N_d==0
            if N_e==0
                if N_z==0
                    [VKron_jp,PolicyKron_jp]=ValueFnIter_Case1_FHorz_nod_noz_raw(n_a, TreatmentDuration, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron_jp,PolicyKron_jp]=ValueFnIter_Case1_FHorz_nod_raw(n_a, n_z, TreatmentDuration, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else
                if N_z==0
                    [VKron_jp,PolicyKron_jp]=ValueFnIter_Case1_FHorz_nod_noz_e_raw(n_a, vfoptions.n_e, TreatmentDuration, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron_jp,PolicyKron_jp]=ValueFnIter_Case1_FHorz_nod_e_raw(n_a, n_z, vfoptions.n_e, TreatmentDuration, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
        else
            if N_e==0
                if N_z==0
                    [VKron_jp, PolicyKron_jp]=ValueFnIter_Case1_FHorz_noz_raw(n_d,n_a, TreatmentDuration, d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron_jp, PolicyKron_jp]=ValueFnIter_Case1_FHorz_raw(n_d,n_a,n_z, TreatmentDuration, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else
                [VKron_jp, PolicyKron_jp]=ValueFnIter_Case1_FHorz_e_raw(n_d,n_a,n_z,  vfoptions.n_e, TreatmentDuration, d_grid, a_grid, z_gridvals_J, e_gridvals_J,pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    end

    % Combine VKron_jp and PolicyKron_jp with the control versions
    VKron=VcontrolKron;
    PolicyKron=PolicycontrolKron;
    if N_e==0
        if N_z==0
            VKron(:,j_p:j_p+TreatmentDuration-1)=VKron_jp;
            if N_d==0
                PolicyKron(:,j_p:j_p+TreatmentDuration-1)=PolicyKron_jp;
            else
                PolicyKron(:,:,j_p:j_p+TreatmentDuration-1)=PolicyKron_jp;
            end
        else
            VKron(:,:,j_p:j_p+TreatmentDuration-1)=VKron_jp;
            if N_d==0
                PolicyKron(:,:,j_p:j_p+TreatmentDuration-1)=PolicyKron_jp;
            else
                PolicyKron(:,:,:,j_p:j_p+TreatmentDuration-1)=PolicyKron_jp;
            end
        end
    else
        if N_z==0
            VKron(:,:,j_p:j_p+TreatmentDuration-1)=VKron_jp;
            if N_d==0
                PolicyKron(:,:,j_p:j_p+TreatmentDuration-1)=PolicyKron_jp;
            else
                PolicyKron(:,:,:,j_p:j_p+TreatmentDuration-1)=PolicyKron_jp;
            end
        else
            VKron(:,:,:,j_p:j_p+TreatmentDuration-1)=VKron_jp;
            if N_d==0
                PolicyKron(:,:,:,j_p:j_p+TreatmentDuration-1)=PolicyKron_jp;
            else
                PolicyKron(:,:,:,:,j_p:j_p+TreatmentDuration-1)=PolicyKron_jp;
            end
        end
    end
    
    %Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
    if N_e==0
        if N_z==0
            V_jp=reshape(VKron,[n_a,N_j]);
            Policy_jp=UnKronPolicyIndexes_Case1_FHorz_noz(PolicyKron, n_d, n_a, N_j, vfoptions);
        else
            V_jp=reshape(VKron,[n_a,n_z,N_j]);
            Policy_jp=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, n_z, N_j, vfoptions);
        end
    else
        if N_z==0
            V_jp=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
            Policy_jp=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
        else
            V_jp=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
            Policy_jp=UnKronPolicyIndexes_Case1_FHorz_e(PolicyKron, n_d, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
        end
    end

    V.(['treatmentage',num2str(j_p)])=V_jp;
    Policy.(['treatmentage',num2str(j_p)])=Policy_jp;


end


end
