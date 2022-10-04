function [V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

V=nan;
Policy=nan;

%% Check which vfoptions have been used, set all others to defaults 
if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
%     vfoptions.exoticpreferences=0;
    vfoptions.dynasty=0;
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
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
%     if ~isfield(vfoptions,'exoticpreferences')
%         vfoptions.exoticpreferences='None';
%     end
    if ~isfield(vfoptions,'dynasty')
        vfoptions.dynasty=0;
    end
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
end

if isempty(n_d)
    n_d=0;
end
N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if ~all(size(d_grid)==[sum(n_d), 1])
    if ~isempty(n_d) % Make sure d is being used before complaining about size of d_grid
        if n_d~=0
            error('d_grid is not the correct shape (should be of size sum(n_d)-by-1)')
        end
    end
elseif ~all(size(a_grid)==[sum(n_a), 1])
    error('a_grid is not the correct shape (should be of size sum(n_a)-by-1)')
elseif ~all(size(z_grid)==[sum(n_z), 1]) && ~all(size(z_grid)==[prod(n_z),length(n_z)])
    error('z_grid is not the correct shape (should be of size sum(n_z)-by-1)')
elseif ~isequal(size(pi_z), [N_z, N_z])
    error('pi is not of size N_z-by-N_z')
elseif isfield(vfoptions,'n_e')
    if ~isfield(vfoptions,'e_grid') && ~isfield(vfoptions,'e_grid_J')
        error('When using vfoptions.n_e you must declare vfoptions.e_grid (or vfoptions.e_grid_J)')
    elseif ~isfield(vfoptions,'pi_e') && ~isfield(vfoptions,'pi_e_J')
        error('When using vfoptions.n_e you must declare vfoptions.pi_e (or vfoptions.pi_e_J)')
    else
        if  ~all(size(vfoptions.e_grid)==[sum(vfoptions.n_e), 1]) && ~all(size(vfoptions.e_grid)==[prod(vfoptions.n_e),length(vfoptions.n_e)])
            error('vfoptions.e_grid is not the correct shape (should be of size sum(n_e)-by-1)')
        elseif ~all(size(vfoptions.pi_e)==[prod(vfoptions.n_e),1])
            error('vfoptions.pi_e is not the correct shape (should be of size N_e-by-1)')
        end
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

if isfield(vfoptions,'exoticpreferences')
    if strcmp(vfoptions.exoticpreferences,'None')
        % Just ignore and will then continue on.
    elseif strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
       [V, Policy]=ValueFnIter_Case1_FHorz_QuasiHyperbolic(n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
       return
    elseif strcmp(vfoptions.exoticpreferences,'EpsteinZin')
        if vfoptions.dynasty==0
            [V, Policy]=ValueFnIter_Case1_FHorz_EpsteinZin(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            return
        else
            error('CANNOT USE EPSTEIN-ZIN PREFERENCES TOGETHER WITH DYNASTY (email robertdkirkby@gmail.com if you need this option)')
        end
    end
end


%% Deal with StateDependentVariables_z if need to do that.
if isfield(vfoptions,'StateDependentVariables_z')==1
    if vfoptions.verbose==1
        fprintf('StateDependentVariables_z option is being used \n')
    end
    
    if N_d==0
        [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_no_d_SDVz_raw(n_a, n_z, N_j, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    else
        [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_SDVz_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
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
    end
    
    return
end

%% Deal with dynasty if need to do that.
if vfoptions.dynasty==1
    if vfoptions.verbose==1
        fprintf('dynasty option is being used \n')
    end
    if isfield(vfoptions,'tolerance')==0
        vfoptions.tolerance=10^(-9);
    end
    
    if vfoptions.parallel==2
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_no_d_Dynasty_raw(n_a, n_z, N_j, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_Dynasty_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    elseif vfoptions.parallel==0 || vfoptions.parallel==1
        if N_d==0
            % Following command is somewhat misnamed, as actually does Par0 and Par1
            [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_no_d_Par0_Dynasty_raw(n_a, n_z, N_j, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            % Following command is somewhat misnamed, as actually does Par0 and Par1
            [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_Par0_Dynasty_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
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
    end
    
    return
end

%% Detect if using incremental endogenous states and solve this using purediscretization, prior to the main purediscretization routines
if any(vfoptions.incrementaltype)
    % Incremental Endogenous States: aprime either equals a, or one grid point higher (unchanged on incremental increase)
    [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_Increment(n_d,n_a,n_z,d_grid,a_grid,z_grid,N_j,pi_z,ReturnFn,Parameters,ReturnFnParamNames,DiscountFactorParamNames,vfoptions);
    
    %Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
    if isfield(vfoptions,'n_e')
        if N_z==0
            V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
        else
            V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz_e(PolicyKron, n_d, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
        end
    else
        V=reshape(VKron,[n_a,n_z,N_j]);
        Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, n_z, N_j, vfoptions);
    end
    
    return
end


%% Just do the standard case
if N_d==0
    if vfoptions.parallel==2
        if isfield(vfoptions,'n_e')
            if isfield(vfoptions,'e_grid_J')
                e_grid=vfoptions.e_grid_J(:,1); % Just a placeholder
            else
                e_grid=vfoptions.e_grid;
            end
            if isfield(vfoptions,'pi_e_J')
                pi_e=vfoptions.pi_e_J(:,1); % Just a placeholder
            else
                pi_e=vfoptions.pi_e;
            end
            if N_z==0
                [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_nod_noz_e_raw(n_a, vfoptions.n_e, N_j, a_grid, e_grid, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_nod_e_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_grid, e_grid, pi_z, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_nod_raw(n_a, n_z, N_j, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    elseif vfoptions.parallel==1
        if N_z==0
            [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_nod_noz_Par1_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_nod_Par1_raw(n_a, n_z, N_j, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    elseif vfoptions.parallel==0
        [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_nod_Par0_raw(n_a, n_z, N_j, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    end
else
    if vfoptions.parallel==2
        if isfield(vfoptions,'n_e')
            if isfield(vfoptions,'e_grid_J')
                e_grid=vfoptions.e_grid_J(:,1); % Just a placeholder
            else
                e_grid=vfoptions.e_grid;
            end
            if isfield(vfoptions,'pi_e_J')
                pi_e=vfoptions.pi_e_J(:,1); % Just a placeholder
            else
                pi_e=vfoptions.pi_e;
            end
            [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_e_raw(n_d,n_a,n_z,  vfoptions.n_e, N_j, d_grid, a_grid, z_grid, e_grid, pi_z, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    elseif vfoptions.parallel==1
        if N_z==0 || N_z==1 % Would normally just parallel cpu over z, but if there is not z then treat this as 'special case'
            [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_noz_Par1_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);            
        else % Normal...
            [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_Par1_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    elseif vfoptions.parallel==0
        [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_Par0_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    end
end

%Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if isfield(vfoptions,'n_e')
    if N_z==0
        V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
    else
        V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes_Case1_FHorz_e(PolicyKron, n_d, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
    end
else
    V=reshape(VKron,[n_a,n_z,N_j]);
    Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, n_z, N_j, vfoptions);
end

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