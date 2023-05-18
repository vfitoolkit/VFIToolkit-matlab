function [V, Policy]=ValueFnIter_Case3_FHorz(n_d,n_a,n_z,n_u,N_j,d_grid, a_grid, z_grid, u_grid, pi_z, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

V=nan;
Policy=nan;

%% Check which vfoptions have been used, set all others to defaults 
if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
%     vfoptions.exoticpreferences=0;
    vfoptions.parallel=1+(gpuDeviceCount>0);
    vfoptions.verbose=0;
    vfoptions.lowmemory=0;
    if prod(n_z)>50
        vfoptions.paroverz=1; % This is just a refinement of lowmemory=0
    else
        vfoptions.paroverz=0;
    end
    vfoptions.polindorval=1;
    vfoptions.policy_forceintegertype=0;
    vfoptions.outputkron=0; % If 1 then leave output in Kron form
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
%     if ~isfield(vfoptions,'exoticpreferences')
%         vfoptions.exoticpreferences='None';
%     end
    if ~isfield(vfoptions,'parallel')
        vfoptions.parallel=1+(gpuDeviceCount>0);
    end
    if vfoptions.parallel==2
        vfoptions.returnmatrix=2; % On GPU, must use this option
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
        vfoptions.outputkron=0; % If 1 then leave output in Kron form
    end
end

if isempty(n_d)
    n_d=0;
end
if isempty(n_z)
    n_z=0;
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
    if N_z>0
        error('z_grid is not the correct shape (should be of size sum(n_z)-by-1)')
    end
elseif ~isequal(size(pi_z), [N_z, N_z])
    if N_z>0
        error('pi is not of size N_z-by-N_z')
    end
elseif isfield(vfoptions,'n_e')
    if ~isfield(vfoptions,'e_grid') && ~isfield(vfoptions,'e_grid_J')
        error('When using vfoptions.n_e you must declare vfoptions.e_grid (or vfoptions.e_grid_J)')
    elseif ~isfield(vfoptions,'pi_e') && ~isfield(vfoptions,'pi_e_J')
        error('When using vfoptions.n_e you must declare vfoptions.pi_e (or vfoptions.pi_e_J)')
    else
        % check size of e_grid and pi_e
        if isfield(vfoptions,'e_grid')
            if  ~all(size(vfoptions.e_grid)==[sum(vfoptions.n_e), 1]) && ~all(size(vfoptions.e_grid)==[prod(vfoptions.n_e),length(vfoptions.n_e)])
                error('vfoptions.e_grid is not the correct shape (should be of size sum(n_e)-by-1)')
            end
        else % using e_grid_J
            % HAVE NOT YET IMPLEMENTED A CHECK OF THE SIZE OF e_grid_J
        end
        if isfield(vfoptions,'pi_e')
            if ~all(size(vfoptions.pi_e)==[prod(vfoptions.n_e),1])
                error('vfoptions.pi_e is not the correct shape (should be of size N_e-by-1)')
            end
        else % using pi_e_J
            if ~all(size(vfoptions.pi_e_J)==[prod(vfoptions.n_e),N_j])
                error('vfoptions.pi_e_J is not the correct shape (should be of size N_e-by-N_j)')
            end
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
    if length(temp)>(l_d+l_a+l_z+l_e) % This is largely pointless, the ReturnFn is always going to have some parameters
        ReturnFnParamNames={temp{l_d+l_a+l_z+l_e+1:end}}; % the first inputs will always be (d,aprime,a,z)
    else
        ReturnFnParamNames={};
    end
end

% aprimeFnParamNames in same fashion
l_u=length(n_u);
temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d+l_u)
    aprimeFnParamNames={temp{l_d+l_u+1:end}}; % the first inputs will always be (d,u)
else
    aprimeFnParamNames={};
end

%% 
if vfoptions.parallel==2 
   % If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
   pi_z=gpuArray(pi_z);
   pi_u=gpuArray(pi_u);
   d_grid=gpuArray(d_grid);
   a_grid=gpuArray(a_grid);
   z_grid=gpuArray(z_grid);
   u_grid=gpuArray(u_grid);
else
   % If using CPU make sure all the relevant inputs are CPU arrays (not standard arrays)
   % This may be completely unnecessary.
   pi_z=gather(pi_z);
   pi_u=gather(pi_u);
   d_grid=gather(d_grid);
   a_grid=gather(a_grid);
   z_grid=gather(z_grid);
   u_grid=gather(u_grid);
end

if vfoptions.verbose==1
    vfoptions
end

if isfield(vfoptions,'exoticpreferences')
    if strcmp(vfoptions.exoticpreferences,'None')
        % Just ignore and will then continue on.
    elseif strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
%        [V, Policy]=ValueFnIter_Case1_FHorz_QuasiHyperbolic(n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
%        return
    elseif strcmp(vfoptions.exoticpreferences,'EpsteinZin')
        [V, Policy]=ValueFnIter_Case3_FHorz_EpsteinZin(n_d,n_a,n_z,n_u,N_j,d_grid, a_grid, z_grid, u_grid, pi_z, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        return
    end
end

%% Just do the standard case
if l_a>1
    error('Have not yet implemented Case3 for more than one a variable, please contact me if you want/need this')
end

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
            [VKron, PolicyKron]=ValueFnIter_Case3_FHorz_noz_e_raw(n_d,n_a,vfoptions.n_e,n_u, N_j, d_grid, a_grid, e_grid, u_grid, pi_e, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_Case3_FHorz_e_raw(n_d,n_a,n_z,vfoptions.n_e,n_u, N_j, d_grid, a_grid, z_grid, e_grid, u_grid, pi_z, pi_e, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    else
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_Case3_FHorz_noz_raw(n_d,n_a,n_u, N_j, d_grid, a_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_Case3_FHorz_raw(n_d,n_a,n_z,n_u, N_j, d_grid, a_grid, z_grid, u_grid, pi_z, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    end
else
    error('Only gpu parallelization supported for Case3')
end

%Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
% Note: Policy in Case3 has same form as in Case2, so just reuse those commands for UnKron
if vfoptions.outputkron==0
    if isfield(vfoptions,'n_e')
        if N_z==0
            V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz(PolicyKron, n_d, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
        else
            V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz_e(PolicyKron, n_d, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
        end
    else
        if N_z==0
            V=reshape(VKron,[n_a,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz_noz(PolicyKron, n_d, n_a, N_j, vfoptions);
        else
            V=reshape(VKron,[n_a,n_z,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz(PolicyKron, n_d, n_a, n_z, N_j, vfoptions);
        end
    end
else
    V=VKron;
    Policy=PolicyKron;
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