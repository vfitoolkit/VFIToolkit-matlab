function [V, Policy]=ValueFnIter_Case1_FieldExp_FHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TreatmentParams, TreatmentAgeRange, TreatmentDuration, vfoptions)

V=struct();
Policy=struct();

if isempty(n_d)
    n_d=0;
end
if isempty(n_z)
    n_z=0;
end
N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
if isfield(vfoptions,'n_e')
    N_e=prod(vfoptions.n_e);
end

% If quasi-hyperbolic discounting is used, send it off to be treated specially
if isfield(vfoptions,'exoticpreferences')
    if strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
        [V, Policy]=ValueFnIter_Case1_FieldExp_FHorz_QuasiHyperbolic(n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
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

%% Check which vfoptions have been used, set all others to defaults 
if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
%     vfoptions.exoticpreferences='None';
%     vfoptions.dynasty=0; % Dynasty not supported for Field experiments
    vfoptions.parallel=1+(gpuDeviceCount>0);
    if vfoptions.parallel==2
        vfoptions.returnmatrix=2; % On GPU, must use this option
    end
    if isfield(vfoptions,'returnmatrix')==0
        if isa(ReturnFn,'function_handle')==1
            vfoptions.returnmatrix=0;
        else
            vfoptions.returnmatrix=1;
        end
    end
    vfoptions.verbose=0;
    vfoptions.lowmemory=0;
    if prod(n_z)>50
        vfoptions.paroverz=1; % This is just a refinement of lowmemory=0
    else
        vfoptions.paroverz=0;
    end
    vfoptions.incrementaltype=0; % (vector indicating endogenous state is an incremental endogenous state variable)
    vfoptions.polindorval=1;
    vfoptions.policy_forceintegertype=0;
    vfoptions.outputkron=0; % If 1, leave output in Kron form
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
%     if ~isfield(vfoptions,'exoticpreferences')
%         vfoptions.exoticpreferences='None';
%     end
%     vfoptions.dynasty=0; % Dynasty not supported for Field experiments
    if ~isfield(vfoptions,'parallel')
        vfoptions.parallel=1+(gpuDeviceCount>0);
    end
    if vfoptions.parallel==2
        vfoptions.returnmatrix=2; % On GPU, must use this option
    end
    if ~isfield(vfoptions,'returnmatrix')
        if isa(ReturnFn,'function_handle')==1
            vfoptions.returnmatrix=0;
        else
            vfoptions.returnmatrix=1;
        end
    end
    if ~isfield(vfoptions,'lowmemory')
        vfoptions.lowmemory=0;
    end
    if ~isfield(vfoptions,'paroverz') % Only used when vfoptions.lowmemory=0
        if prod(n_z)>50
            vfoptions.paroverz=1;
        else
            vfoptions.paroverz=0;
        end
    end
    if ~isfield(vfoptions,'verbose')
        vfoptions.verbose=0;
    end
    if isfield(vfoptions,'incrementaltype')==0
        vfoptions.incrementaltype=0; % (vector indicating endogenous state is an incremental endogenous state variable)
    end
    if ~isfield(vfoptions,'polindorval')
        vfoptions.polindorval=1;
    end
    if ~isfield(vfoptions,'policy_forceintegertype')
        vfoptions.policy_forceintegertype=0;
    end
    if isfield(vfoptions,'ExogShockFn')
        vfoptions.ExogShockFnParamNames=getAnonymousFnInputNames(vfoptions.ExogShockFn);
    end
    if isfield(vfoptions,'EiidShockFn')
        vfoptions.EiidShockFnParamNames=getAnonymousFnInputNames(vfoptions.EiidShockFn);
    end
    if ~isfield(vfoptions,'outputkron')
        vfoptions.outputkron=0; % If 1, leave output in Kron form
    end
end

% Can skip checking input sizes as this will be done when solving for the control group anyway.

%% Implement new way of handling ReturnFn inputs
if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);
if n_z(1)==0
    l_z=0;
end
if isfield(vfoptions,'n_e')
    l_e=length(vfoptions.n_e);
else
    l_e=0;
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

% Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
% Gradually rolling these out so that all the commands build off of these
z_gridvals_J=zeros(prod(n_z),length(n_z),'gpuArray');
pi_z_J=zeros(prod(n_z),prod(n_z),'gpuArray');
if isfield(vfoptions,'ExogShockFn')
    if isfield(vfoptions,'ExogShockFnParamNames')
        for jj=1:N_j
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
            ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
            for ii=1:length(ExogShockFnParamsVec)
                ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
            end
            [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
            pi_z_J(:,:,jj)=gpuArray(pi_z);
            if all(size(z_grid)==[sum(n_z),1])
                z_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(n_z,z_grid,1));
            else % already joint-grid
                z_gridvals_J(:,:,jj)=gpuArray(z_grid,1);
            end
        end
    else
        for jj=1:N_j
            [z_grid,pi_z]=vfoptions.ExogShockFn(N_j);
            pi_z_J(:,:,jj)=gpuArray(pi_z);
            if all(size(z_grid)==[sum(n_z),1])
                z_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(n_z,z_grid,1));
            else % already joint-grid
                z_gridvals_J(:,:,jj)=gpuArray(z_grid,1);
            end
        end
    end
elseif prod(n_z)==0 % no z
    z_gridvals_J=[];
elseif ndims(z_grid)==3 % already an age-dependent joint-grid
    if all(size(z_grid)==[prod(n_z),length(n_z),N_j])
        z_gridvals_J=z_grid;
    end
    pi_z_J=pi_z;
elseif all(size(z_grid)==[sum(n_z),N_j]) % age-dependent grid
    for jj=1:N_j
        z_gridvals_J(:,:,jj)=CreateGridvals(n_z,z_grid(:,jj),1);
    end
    pi_z_J=pi_z;
elseif all(size(z_grid)==[prod(n_z),length(n_z)]) % joint grid
    z_gridvals_J=z_grid.*ones(1,1,N_j,'gpuArray');
    pi_z_J=pi_z.*ones(1,1,N_j,'gpuArray');
elseif all(size(z_grid)==[sum(n_z),1]) % basic grid
    z_gridvals_J=CreateGridvals(n_z,z_grid,1).*ones(1,1,N_j,'gpuArray');
    pi_z_J=pi_z.*ones(1,1,N_j,'gpuArray');
end

% If using e variable, do same for this
if isfield(vfoptions,'n_e')
    if prod(vfoptions.n_e)==0
        vfoptions=rmfield(vfoptions,'n_e');
    else
        if isfield(vfoptions,'e_grid_J')
            error('No longer use vfoptions.e_grid_J, instead just put the age-dependent grid in vfoptions.e_grid (functionality of VFI Toolkit has changed to make it easier to use)')
        end
        if ~isfield(vfoptions,'e_grid') % && ~isfield(vfoptions,'e_grid_J')
            error('You are using an e (iid) variable, and so need to declare vfoptions.e_grid')
        elseif ~isfield(vfoptions,'pi_e')
            error('You are using an e (iid) variable, and so need to declare vfoptions.pi_e')
        end

        e_gridvals_J=zeros(prod(vfoptions.n_e),length(vfoptions.n_e),'gpuArray');
        pi_e_J=zeros(prod(vfoptions.n_e),prod(vfoptions.n_e),'gpuArray');

        if isfield(vfoptions,'EiidShockFn')
            if isfield(vfoptions,'EiidShockFnParamNames')
                for jj=1:N_j
                    EiidShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.EiidShockFnParamNames,jj);
                    EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                    for ii=1:length(EiidShockFnParamsVec)
                        EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                    end
                    [vfoptions.e_grid,vfoptions.pi_e]=vfoptions.EiidShockFn(EiidShockFnParamsCell{:});
                    pi_e_J(:,jj)=gpuArray(vfoptions.pi_e);
                    if all(size(vfoptions.e_grid)==[sum(vfoptions.n_e),1])
                        e_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(vfoptions.n_e,vfoptions.e_grid,1));
                    else % already joint-grid
                        e_gridvals_J(:,:,jj)=gpuArray(vfoptions.e_grid,1);
                    end
                end
            else
                for jj=1:N_j
                    [vfoptions.e_grid,vfoptions.pi_e]=vfoptions.EiidShockFn(N_j);
                    pi_e_J(:,jj)=gpuArray(vfoptions.pi_e);
                    if all(size(vfoptions.e_grid)==[sum(vfoptions.n_e),1])
                        e_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(vfoptions.n_e,vfoptions.e_grid,1));
                    else % already joint-grid
                        e_gridvals_J(:,:,jj)=gpuArray(vfoptions.e_grid,1);
                    end
                end
            end
        elseif ndims(vfoptions.e_grid)==3 % already an age-dependent joint-grid
            if all(size(vfoptions.e_grid)==[prod(vfoptions.n_e),length(vfoptions.n_e),N_j])
                e_gridvals_J=vfoptions.e_grid;
            end
            pi_e_J=vfoptions.pi_e;
        elseif all(size(vfoptions.e_grid)==[sum(vfoptions.n_e),N_j]) % age-dependent stacked-grid
            for jj=1:N_j
                e_gridvals_J(:,:,jj)=CreateGridvals(vfoptions.n_e,vfoptions.e_grid(:,jj),1);
            end
            pi_e_J=vfoptions.pi_e;
        elseif all(size(vfoptions.e_grid)==[prod(vfoptions.n_e),length(vfoptions.n_e)]) % joint grid
            e_gridvals_J=vfoptions.e_grid.*ones(1,1,N_j,'gpuArray');
            pi_e_J=vfoptions.pi_e.*ones(1,N_j,'gpuArray');
        elseif all(size(vfoptions.e_grid)==[sum(vfoptions.n_e),1]) % basic grid
            e_gridvals_J=CreateGridvals(vfoptions.n_e,vfoptions.e_grid,1).*ones(1,1,N_j,'gpuArray');
            pi_e_J=vfoptions.pi_e.*ones(1,N_j,'gpuArray');
        end
    end
end


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
            if isfield(vfoptions,'n_e')
                if N_z==0
                    [VKron_jp,PolicyKron_jp]=ValueFnIter_Case1_FHorz_nod_noz_e_raw(n_a, vfoptions.n_e, TreatmentDuration, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron_jp,PolicyKron_jp]=ValueFnIter_Case1_FHorz_nod_e_raw(n_a, n_z, vfoptions.n_e, TreatmentDuration, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else
                if N_z==0
                    [VKron_jp,PolicyKron_jp]=ValueFnIter_Case1_FHorz_nod_noz_raw(n_a, TreatmentDuration, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron_jp,PolicyKron_jp]=ValueFnIter_Case1_FHorz_nod_raw(n_a, n_z, TreatmentDuration, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
        else
            if isfield(vfoptions,'n_e')
                [VKron_jp, PolicyKron_jp]=ValueFnIter_Case1_FHorz_e_raw(n_d,n_a,n_z,  vfoptions.n_e, TreatmentDuration, d_grid, a_grid, z_gridvals_J, e_gridvals_J,pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                if N_z==0
                    [VKron_jp, PolicyKron_jp]=ValueFnIter_Case1_FHorz_noz_raw(n_d,n_a, TreatmentDuration, d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron_jp, PolicyKron_jp]=ValueFnIter_Case1_FHorz_raw(n_d,n_a,n_z, TreatmentDuration, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
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
    if isfield(vfoptions,'n_e')
        if N_z==0
            V_jp=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
            Policy_jp=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
        else
            V_jp=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
            Policy_jp=UnKronPolicyIndexes_Case1_FHorz_e(PolicyKron, n_d, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
        end
    else
        if N_z==0
            V_jp=reshape(VKron,[n_a,N_j]);
            Policy_jp=UnKronPolicyIndexes_Case1_FHorz_noz(PolicyKron, n_d, n_a, N_j, vfoptions);
        else
            V_jp=reshape(VKron,[n_a,n_z,N_j]);
            Policy_jp=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, n_z, N_j, vfoptions);
        end
    end

    % Sometimes numerical rounding errors (of the order of 10^(-16) can mean
    % that Policy is not integer valued. The following corrects this by converting to int64 and then
    % makes the output back into double as Matlab otherwise cannot use it in
    % any arithmetical expressions.
    if vfoptions.policy_forceintegertype==1
        fprintf('USING vfoptions to force integer... \n')
        % First, give some output on the size of any changes in Policy as a
        % result of turning the values into integers
        temp=max(max(max(abs(round(Policy_jp)-Policy_jp))));
        while ndims(temp)>1
            temp=max(temp);
        end
        fprintf('  CHECK: Maximum change when rounding values of Policy is %8.6f (if these are not of numerical rounding error size then something is going wrong) \n', temp)
        % Do the actual rounding to integers
        Policy_jp=round(Policy_jp);
        % Somewhat unrelated, but also do a double-check that Policy is now all positive integers
        temp=min(min(min(Policy_jp)));
        while ndims(temp)>1
            temp=min(temp);
        end
        fprintf('  CHECK: Minimum value of Policy is %8.6f (if this is <=0 then something is wrong) \n', temp)
        %     Policy=uint64(Policy);
        %     Policy=double(Policy);
    elseif vfoptions.policy_forceintegertype==2
        % Do the actual rounding to integers
        Policy_jp=round(Policy_jp);
    end

    V.(['treatmentage',num2str(j_p)])=V_jp;
    Policy.(['treatmentage',num2str(j_p)])=Policy_jp;


end


end