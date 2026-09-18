function varargout=ValueFnIter_Case1_VFHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

%% Check which vfoptions have been used, set all others to defaults
if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    % If vfoptions is not given, just use all the defaults
    vfoptions.verbose=0; % =1 print out feedback on what is happening internally
    vfoptions.divideandconquer=0; % =1 Use divide-and-conquer to exploit monotonicity
    vfoptions.gridinterplayer=0; % Interpolate between grid points (not yet implemented for alternative preferences)
    vfoptions.lowmemory=0; % use more loops and less parallelization, reduce memory use but at the cost of slower runtimes
    % Alternative model setups
    vfoptions.incrementaltype=0; % (vector indicating endogenous state is an incremental endogenous state variable)
    vfoptions.exoticpreferences='None';
    vfoptions.dynasty=0;
    vfoptions.experienceasset=0;
    vfoptions.experienceassetu=0;
    vfoptions.experienceassete=0;
    vfoptions.experienceassetz=0;
    vfoptions.experienceassetze=0;
    vfoptions.experienceassetsemiz=0;
    vfoptions.riskyasset=0;
    vfoptions.residualasset=0;
    vfoptions.n_ambiguity=0;
    vfoptions.n_e=0;
    vfoptions.n_semiz=0;
    % Largely just for internal use only
    vfoptions.parallel=1+(gpuDeviceCount>0);
    % When calling as a subcommand, the following are used internally
    vfoptions.outputkron=0; % If 1 then leave output in Kron form
    vfoptions.alreadygridvals=0; % =1 when calling as a subcommand
    vfoptions.alreadygridvals_semiexo=0; % =1 when calling as a subcommand
    vfoptions.precision = underlyingType(a_grid);
else
    % Check vfoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(vfoptions,'verbose')
        vfoptions.verbose=0;
    end
    if ~isfield(vfoptions,'divideandconquer')
        vfoptions.divideandconquer=0; % =1 Use divide-and-conquer to exploit monotonicity
    end
    if ~isfield(vfoptions,'gridinterplayer')
        vfoptions.gridinterplayer=0; % =1 Interpolate between grid points (not yet implemented for most cases)
    elseif vfoptions.gridinterplayer(1)==1
        if ~isfield(vfoptions,'ngridinterp')
            error('When using vfoptions.gridinterplayer=1 you must set vfoptions.ngridinterp (number of points to interpolate for aprime between each consecutive pair of points in a_grid)')
        end
    end
    if ~isfield(vfoptions,'lowmemory')
        vfoptions.lowmemory=0;
    end
    % Alternative model setups
    if ~isfield(vfoptions,'incrementaltype')
        vfoptions.incrementaltype=0; % (vector indicating endogenous state is an incremental endogenous state variable)
    end
    if ~isfield(vfoptions,'exoticpreferences')
        vfoptions.exoticpreferences='None';
    end
    if ~isfield(vfoptions,'dynasty')
        vfoptions.dynasty=0;
    end
    if ~isfield(vfoptions,'experienceasset')
        vfoptions.experienceasset=0;
    end
    if ~isfield(vfoptions,'experienceassetu')
        vfoptions.experienceassetu=0;
    end
    if ~isfield(vfoptions,'experienceassete')
        vfoptions.experienceassete=0;
    end
    if ~isfield(vfoptions,'experienceassetz')
        vfoptions.experienceassetz=0;
    end
    if ~isfield(vfoptions,'experienceassetze')
        vfoptions.experienceassetze=0;
    end
    if ~isfield(vfoptions,'experienceassetsemiz')
        vfoptions.experienceassetsemiz=0;
    end
    if ~isfield(vfoptions,'riskyasset')
        vfoptions.riskyasset=0;
    end
    if ~isfield(vfoptions,'residualasset')
        vfoptions.residualasset=0;
    end
    if ~isfield(vfoptions,'n_ambiguity')
        vfoptions.n_ambiguity=0;
    end
    if ~isfield(vfoptions,'n_e')
        vfoptions.n_e=0;
    end
    if ~isfield(vfoptions,'n_semiz')
        vfoptions.n_semiz=0;
    end
    % Largely just for internal use only
    if ~isfield(vfoptions,'parallel')
        vfoptions.parallel=1+(gpuDeviceCount>0);
    end
    % When calling as a subcommand, the following are used internally
    if ~isfield(vfoptions,'outputkron')
        vfoptions.outputkron=0; % If 1 then leave output in Kron form
    end
    if ~isfield(vfoptions,'alreadygridvals')
        vfoptions.alreadygridvals=0; % =1 when calling as a subcommand
    end
    if ~isfield(vfoptions,'alreadygridvals_semiexo')
        vfoptions.alreadygridvals_semiexo=0; % =1 when calling as a subcommand
    end
    if ~isfield(vfoptions,'precision')
        vfoptions.precision = underlyingType(a_grid);
    end
end

% --- SMART nargin PARSER ---
if isempty(ReturnFnParamNames)
    if isfield(vfoptions, 'ReturnFnParamNames')
        ReturnFnParamNames = vfoptions.ReturnFnParamNames;
    else
        temp = getAnonymousFnInputNames(ReturnFn);

        num_d_vars = length(n_d);
        if num_d_vars == 1 && n_d(1) == 0; num_d_vars = 0; end
        num_a_vars = length(n_a);
        num_z_vars = length(n_z);
        if num_z_vars == 1 && n_z(1) == 0; num_z_vars = 0; end

        is_exp  = vfoptions.experienceasset > 0;
        is_expz = vfoptions.experienceassetz > 0;
        has_semiz = prod(vfoptions.n_semiz) > 0;
        has_e = prod(vfoptions.n_e) > 0;

        if is_exp || is_expz
            if is_exp
                l_a2 = vfoptions.experienceasset;
            else
                l_a2 = vfoptions.experienceassetz;
            end
            num_a1 = num_a_vars - l_a2;
            num_a2 = l_a2;

            % ExpAsset structure: D, A1prime, A1, A2, Z
            num_prefix_args = num_d_vars + 2*num_a1 + num_a2 + num_z_vars;
            if has_semiz
                num_prefix_args = num_prefix_args + length(vfoptions.n_semiz);
            end
        elseif vfoptions.riskyasset == 1
            num_u_vars = length(vfoptions.n_u);
            % RiskyAsset structure: D, A1prime, A2prime, A1, A2, Z, U
            num_prefix_args = num_d_vars + 4 + num_z_vars + num_u_vars;
            if has_semiz
                num_prefix_args = num_prefix_args + length(vfoptions.n_semiz);
            end
        else
            % Standard Case: D, Aprime, A, Z, E
            num_prefix_args = num_d_vars + 2*num_a_vars + num_z_vars;
            if has_e
                num_prefix_args = num_prefix_args + length(vfoptions.n_e);
            end
        end

        if length(temp) > num_prefix_args
            ReturnFnParamNames = {temp{num_prefix_args + 1 : end}};
        else
            ReturnFnParamNames = {};
        end
    end
end

is_EZ = strcmp(vfoptions.exoticpreferences, 'EpsteinZin') || strcmp(vfoptions.exoticpreferences, 'QHEpsteinZin');
if is_EZ
    % Reject asset types this dispatcher does not handle: every asset type it does handle is
    % dispatched below and returns, so an unsupported flag would otherwise be silently ignored.
    if vfoptions.experienceasset>=1 || vfoptions.experienceassetu>=1 || vfoptions.experienceassetz>=1 || vfoptions.experienceassete>=1 || vfoptions.experienceassetze>=1 || vfoptions.experienceassetsemiz>=1
        % Bypass this legacy restriction for our vectorized QHEpsteinZin tensor
        if strcmp(vfoptions.exoticpreferences, 'EpsteinZin')
            error('Epstein-Zin preferences are not implemented for the experience assets (only for riskyasset, or for the standard endogenous states)')
        end
    end
    if vfoptions.residualasset==1
        error('Epstein-Zin preferences are not implemented for residualasset')
    end
    if vfoptions.dynasty==1
        error('Epstein-Zin preferences are not implemented for dynasty')
    end

    %% Some Epstein-Zin specific options need to be set if they are not already declared
    if ~isfield(vfoptions,'EZriskaversion')
        error('When using Epstein-Zin preferences you must declare vfoptions.EZriskaversion (coefficient controlling risk aversion)')
    end
    if ~isfield(vfoptions,'EZutils')
        vfoptions.EZutils=1; % Use EZ preferences with general utility function (0 gives traditional EZ with exogenous labor, 2 gives traditional EZ with endogenous labor)
    end
    if vfoptions.EZutils==1
        % Have to do EZ preferences differently depending on whether the utility function is >=0 or <=0.
        % vfoptions.EZpositiveutility=1 if utility is positive; Note, in this case when EZriskaversion is higher, the risk aversion is larger (EZriskaversion>0 is risk averse)
        % vfoptions.EZpositiveutility=0 if utility is negative; Note, in this case when EZriskaversion is lower, the risk aversion is larger  (EZriskaversion<0 is risk averse)
        if ~isfield(vfoptions,'EZpositiveutility')
            warning('Using Epstein-Zin preferences it is assumed the utility/return function is negative valued, if not you need to set vfoptions.EZpositiveutility=1')
            vfoptions.EZpositiveutility=0; % User did not specify. Guess that it is negative as most common things (like CES) are negative valued.
        end
    else
        % Traditional EZ preferences requires you to specify the EIS parameter
        if ~isfield(vfoptions,'EZeis')
            error('When using Epstein-Zin preferences you must declare vfoptions.EZeis (elasticity of intertemporal substitution)')
        end
    end
    if ~isfield(vfoptions,'EZoneminusbeta')
        vfoptions.EZoneminusbeta=0; % default essentially does nothing
        %=1 Put a (1-beta)* term on the this period return
        %=2 Put a (1-sj*beta)* term on the this period return
    end
    % Set up sj
    if isfield(vfoptions,'survivalprobability')
        sj=Parameters.(vfoptions.survivalprobability);
        if length(sj)~=N_j
            error('Survival probabilities must be of the same length as N_j')
        end
    elseif isfield(vfoptions,'WarmGlowBequestsFn')
        % If you have warm-glow but do not specify survival probabilities it is assumed you only get it at end of final period
        sj=ones(N_j,1); % conditional survival probabilities
        sj(end)=0;
        warning('You have used vfoptions.WarmGlowBequestsFn, but have not set vfoptions.survivalprobability, it is assumed you only want to have the warm-glow at the end of the final period')
    else
        sj=ones(N_j,1); % conditional survival probabilities
    end
    % Declare warmglow indicator
    if isfield(vfoptions,'WarmGlowBequestsFn')
        warmglow=1;
        temp=getAnonymousFnInputNames(vfoptions.WarmGlowBequestsFn);
        vfoptions.WarmGlowBequestsFnParamsNames={temp{2:end}};
    else
        warmglow=0;
    end
    vfoptions = EpsteinZinSetup_VFHorz(N_j, Parameters, ReturnFnParamNames, DiscountFactorParamNames, vfoptions);
end

if vfoptions.divideandconquer==1
    if ~isfield(vfoptions,'level1n')
        if isscalar(n_a)
            vfoptions.level1n=floor(sqrt(n_a(1)));
            if n_a(1)<5
                error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
            end
        elseif length(n_a)==2
            vfoptions.level1n=[floor(sqrt(n_a(1))),n_a(2)]; % default DC2A: level1n(2)==n_a(2) triggers DC2A branch
            if n_a(1)<5
                error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
            end
        end
        if vfoptions.verbose==1
            fprintf('Suggestion: When using vfoptions.divideandconquer it will be faster or slower if you set different values of vfoptions.level1n (for smaller models 7 or 9 is good, but for larger models something 15 or 21 can be better) \n')
        end
    else
        if ~isscalar(n_a) && isscalar(vfoptions.level1n)
            vfoptions.level1n=[vfoptions.level1n,n_a(2:end)]; % user only needs to declare level1n for first dimension. Fill out the rest with n_a(2:end).
        end
    end
end

if vfoptions.parallel == 2
    if ~isempty(d_grid), d_grid = gpuArray(d_grid); end
    if ~isempty(a_grid), a_grid = gpuArray(a_grid); end
    if ~isempty(z_grid), z_grid = gpuArray(z_grid); end
    if ~isempty(pi_z),   pi_z   = gpuArray(pi_z);   end
end

%% Semi-exogenous shock gridvals and pi
%% Exogenous shock gridvals and pi
if isfield(vfoptions, 'semiz_gridvals_J') && ~isempty(vfoptions.semiz_gridvals_J)
    % --- Complex Semi-Exogenous Expansion Path ---
    sz_J = vfoptions.semiz_gridvals_J;
    N_semiz = size(sz_J, 1);
    num_semiz_vars = size(sz_J, 2);
    num_periods = size(sz_J, 3);

    if N_z > 0
        z_J = repmat(z_grid, [1, 1, num_periods]);
        num_z_vars = size(z_grid, 2);
    else
        z_J = [];
        num_z_vars = 0;
    end

    z_gridvals_J = zeros(N_semiz * max(1, N_z), num_semiz_vars + num_z_vars, num_periods, 'like', sz_J);
    for t = 1:num_periods
        if N_z > 0
            semiz_expanded = kron(sz_J(:,:,t), ones(N_z, 1));
            z_expanded = kron(ones(N_semiz, 1), z_J(:,:,t));
            z_gridvals_J(:,:,t) = [semiz_expanded, z_expanded];
        else
            z_gridvals_J(:,:,t) = sz_J(:,:,t);
        end
    end
    pi_z_J = pi_z;
    n_combined_z = [vfoptions.n_semiz, n_z];
else
    % --- Simple Standard Z Path (Zero Overhead) ---
    if N_j > 1 && size(z_grid, ndims(z_grid)) ~= N_j
        % Replicate across periods if static
        z_gridvals_J = repmat(z_grid, [1, 1, N_j]);
    else
        z_gridvals_J = z_grid;
    end
    pi_z_J = pi_z;
    n_combined_z = n_z;
end

N_d = prod(n_d);
N_a = prod(n_a);
N_z = prod(n_z);
N_z_safe = max(1, N_z);

%% Exogenous shock gridvals and pi
if isfield(vfoptions, 'semiz_gridvals_J') && ~isempty(vfoptions.semiz_gridvals_J)
    % 1. Extract the pre-computed static semiz tensor
    sz_J = vfoptions.semiz_gridvals_J;
    N_semiz = size(sz_J, 1);
    num_semiz_vars = size(sz_J, 2);
    num_periods = size(sz_J, 3);

    % 2. Get the z grid
    if N_z > 0
        z_J = repmat(z_grid, [1, 1, num_periods]);
        num_z_vars = size(z_grid, 2);
    else
        z_J = [];
        num_z_vars = 0;
    end

    % 3. Combine them via Kronecker expansion for each period
    z_gridvals_J = zeros(N_semiz * max(1, N_z), num_semiz_vars + num_z_vars, num_periods, 'like', sz_J);
    for t = 1:num_periods
        if N_z > 0
            semiz_expanded = kron(sz_J(:,:,t), ones(N_z, 1));
            z_expanded = kron(ones(N_semiz, 1), z_J(:,:,t));
            z_gridvals_J(:,:,t) = [semiz_expanded, z_expanded];
        else
            z_gridvals_J(:,:,t) = sz_J(:,:,t);
        end
    end

    % 4. Pass raw transition matrix; decoupled sequential evaluation handles it
    pi_z_J = pi_z;
    n_combined_z = [vfoptions.n_semiz, n_z];
else
    % Fallback to standard z
    z_gridvals_J = z_grid;
    pi_z_J = pi_z;
    n_combined_z = n_z;
end

%% Quasi-Hyperbolic dispatcher (no divide-and-conquer)
if strcmp(vfoptions.exoticpreferences, 'QuasiHyperbolic')
    if nargout == 4
        [V, Policy, Valt, Policyalt] = ValueFnIter_VFHorz_QuasiHyperbolic(n_d, n_a, n_combined_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        varargout = {V, Policy, Valt, Policyalt};
    else
        [V, Policy, Valt] = ValueFnIter_VFHorz_QuasiHyperbolic(n_d, n_a, n_combined_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        varargout = {V, Policy, Valt, []};
    end
    return;
elseif strcmp(vfoptions.exoticpreferences, 'QHEpsteinZin')
    if nargout == 4
        [V, Policy, Valt, Policyalt] = ValueFnIter_VFHorz_QHEpsteinZin(n_d, n_a, n_combined_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        varargout = {V, Policy, Valt, Policyalt};
    else
        [V, Policy, Valt] = ValueFnIter_VFHorz_QHEpsteinZin(n_d, n_a, n_combined_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        varargout = {V, Policy, Valt, []};
    end
    return;
end

% ---------------------------------------------------------------------
% UNIVERSAL PACKER: Unstack Endogenous, Decision, and Exogenous Grids
% ---------------------------------------------------------------------
has_e = isfield(vfoptions, 'n_e') && prod(vfoptions.n_e) > 0;
n_e_pass = 0; e_grid_pass = [];
if has_e
    n_e_pass = vfoptions.n_e;
    e_grid_pass = vfoptions.e_grid;
    e_work = vfoptions.e_grid;
else
    e_work = ones(1, 1, 'like', a_grid);
end

z_pass = [];
if N_z > 0
    % Pass period 1 for initial sizing; dynamic time-varying Z is handled in the reverse_j loop
    z_pass = z_gridvals_J(:,:,1);
end

% --- 2. Dimension and ExpAsset Slicing ---
l_a2 = 0;
if vfoptions.experienceasset > 0; l_a2 = vfoptions.experienceasset; end
if vfoptions.experienceassetz > 0; l_a2 = vfoptions.experienceassetz; end

if l_a2 > 0
    n_a1 = n_a(1:end-l_a2);
    n_a2 = n_a(end-l_a2+1:end);
    N_a2 = prod(n_a2);
else
    n_a1 = n_a;
    n_a2 = [];
    N_a2 = 0;
end
N_a1 = prod(n_a1);

% --- 2b. Universal Grid Packing ---
a1_grid_len = sum(n_a1);
a1_grid_vals = a_grid(1:a1_grid_len);
a2_grid_vals = a_grid(a1_grid_len+1:end);

% Pack D and A1 (Endogenous)
[TensorReturnFn, D_cells_block, A1_cells, Z_cells, E_cells] = CreateTensorFnAndCells(ReturnFn, n_d, n_a1, n_combined_z, n_e_pass, d_grid, a1_grid_vals, [], []);

% Pack A2 (Experience)
if l_a2 > 0
    [TensoraprimeFn, ~, A2_cells, ~, ~] = CreateTensorFnAndCells(vfoptions.aprimeFn, 0, n_a2, 0, 0, [], a2_grid_vals, [], []);
else
    TensoraprimeFn = [];
    A2_cells = {};
end

% Re-construct the legacy A1_mat and A2_mat formats expected by the lower TensorBlock
A1_mat = zeros(N_a1, length(n_a1), 'like', a_grid);
for i_a = 1:length(n_a1)
    A1_mat(:, i_a) = A1_cells{i_a}(:);
end

A2_mat = zeros(N_a2, length(n_a2), 'like', a_grid);
a2_grids_1d = cell(1, length(n_a2));
offset = 0;
for i_a = 1:length(n_a2)
    A2_mat(:, i_a) = A2_cells{i_a}(:);
    a2_grids_1d{i_a} = a2_grid_vals((offset + 1):(offset + n_a2(i_a)));
    offset = offset + n_a2(i_a);
end

% Ensure D_cells_block is formatted for the QHEZ 5D Tensor [N_d, 1, 1, 1, 1]
for i_d = 1:length(D_cells_block)
    D_cells_block{i_d} = reshape(D_cells_block{i_d}, [N_d, 1, 1, 1, 1]);
end

% Extract aprimeFn Params for ExpAsset
if l_a2 > 0
    aprimeFn = vfoptions.aprimeFn;
    if isfield(vfoptions, 'aprimeFnParamNames')
        aprimeFnParamNames = vfoptions.aprimeFnParamNames;
    else
        temp = getAnonymousFnInputNames(aprimeFn);
        num_prefix = length(n_d) + length(n_a2) + length(n_z);
        if length(temp) > num_prefix
            aprimeFnParamNames = {temp{num_prefix+1:end}};
        else
            aprimeFnParamNames = {};
        end
    end
    % --- TENSOR BRIDGE FIX: Filter out state variables misidentified as parameters ---
    aprimeFnParamNames = aprimeFnParamNames(isfield(Parameters, aprimeFnParamNames));
else
    aprimeFn = [];
    aprimeFnParamNames = {};
end

% --- Standardize Dimensions for the Slicer & Allocator ---
N_d_safe = max(1, prod(n_d));
has_d = sum(n_d) > 0;
n_a_work = prod(n_a);
a1_work = A1_cells{1}(:); % Extract primary asset grid for interpolation

has_semiz = prod(vfoptions.n_semiz) > 0;
if has_semiz
    % Detect if the CPU wrapper already expanded n_z
    if length(n_z) >= length(vfoptions.n_semiz) && isequal(n_z(1:length(vfoptions.n_semiz)), vfoptions.n_semiz)
        N_semiz = prod(vfoptions.n_semiz);
        n_all_z = n_z;
        N_z_exog = max(1, prod(n_z) / N_semiz);
    else
        N_semiz = prod(vfoptions.n_semiz);
        n_all_z = [vfoptions.n_semiz, n_z];
        N_z_exog = max(1, prod(n_z));
    end
else
    N_semiz = 1;
    n_all_z = n_z;
    N_z_exog = max(1, prod(n_z));
end
has_z = prod(n_z) > 0;
n_z_work = N_semiz * N_z_exog;

n_e_work = max(1, prod(n_e_pass));

N_ze = n_z_work * n_e_work;

if vfoptions.gridinterplayer(1) == 1
    PolicyKron = zeros(3, n_a_work, n_z_work, n_e_work, N_j, 'like', a_grid);
else
    PolicyKron = zeros(n_a_work, n_z_work, n_e_work, N_j, 'like', a_grid);
end
V_next = zeros(n_a_work, n_z_work, n_e_work, 'like', a_grid);

% --- Grid Interpolation Setup ---
if vfoptions.gridinterplayer(1) == 1
    n2short = vfoptions.ngridinterp;
    n2long  = n2short * 2 + 3;
    % Use a_work instead of a_gridvals(:,1)
    a1prime_grid = interp1(1:1:N_a1, a1_work, linspace(1, N_a1, N_a1 + (N_a1 - 1) * n2short))';
else
    n2short = 0;
    n2long  = 0;
    a1prime_grid = [];
end

% =========================================================
% UNIVERSAL MIX-IN: EPSTEIN-ZIN VS CRRA (Base Orchestrator)
% =========================================================
if is_EZ
    ezc2 = vfoptions.ezc2; ezc3 = vfoptions.ezc3; ezc4 = vfoptions.ezc4;
    ezc5 = vfoptions.ezc5; ezc6 = vfoptions.ezc6; ezc7 = vfoptions.ezc7;
    ezc8 = vfoptions.ezc8; sj = vfoptions.sj; warmglow = vfoptions.warmglow;
else
    % Neutral CRRA fallbacks (collapses EZ math to standard)
    ezc2 = ones(N_j,1); ezc3 = 1; ezc4 = 1;
    ezc5 = ones(N_j,1); ezc6 = ones(N_j,1); ezc7 = ones(N_j,1);
    ezc8 = ones(N_j,1); sj = ones(N_j,1); warmglow = 0;
end

% --- Slicer Setup (Multi-Axis) ---
if ismember(vfoptions.lowmemory, [0, 5])
    ze_chunks = {1:N_ze};
elseif vfoptions.lowmemory == 1
    chunk_size = 300; % Safe to crank back up to 300 with FP32!
    num_chunks = ceil(N_ze / chunk_size);
    ze_chunks = cell(1, num_chunks);
    for c = 1:num_chunks
        ze_chunks{c} = (c-1)*chunk_size + 1 : min(c*chunk_size, N_ze);
    end
else
    ze_chunks = num2cell(1:N_ze);
end

% --- Determine N_a1 and N_a2 for Slicing ---
is_exp  = vfoptions.experienceasset > 0;
is_expz = vfoptions.experienceassetz > 0;
if is_exp || is_expz
    if is_exp; l_a2 = vfoptions.experienceasset; else; l_a2 = vfoptions.experienceassetz; end
    N_a1 = max(1, prod(n_a(1:end-l_a2)));
    N_a2 = prod(n_a(end-l_a2+1:end));
    num_a1_pass = length(n_a) - l_a2;
else
    N_a1 = max(1, prod(n_a));
    N_a2 = 1;
    num_a1_pass = length(n_a);
end

if ismember(vfoptions.lowmemory, [4, 5]) && (is_exp || is_expz)
    a2_chunks = num2cell(1:N_a2);
else
    a2_chunks = {1:N_a2};
end

for reverse_j = 0:N_j-1
    jj = N_j - reverse_j;
    if jj == N_j && (~isfield(vfoptions, 'V_Jplus1') || isempty(vfoptions.V_Jplus1))
        if warmglow == 1
            % Evaluate WarmGlowBequestsFn across terminal asset choices
            % (Assuming a_grid serves as the terminal asset choice grid for bequests)
            wg_params = CreateCellFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames, jj);
            % Evaluate terminal warm glow across the asset space
            V_warmglow = vfoptions.WarmGlowBequestsFn(a_grid, wg_params{:});
            V_next = repmat(V_warmglow, [1, N_z_safe, n_e_work]);
        else
            V_next = zeros(n_a_work, n_z_work, n_e_work, 'like', a_grid);
        end
    end

    ReturnFnParamsCell = CreateCellFromParams(Parameters, ReturnFnParamNames, jj, vfoptions.precision);
    DiscountFactorParamsVec = CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj, vfoptions.precision);
    beta_j = prod(DiscountFactorParamsVec);

    if l_a2 > 0
        aprimeFnParamsCell = CreateCellFromParams(Parameters, aprimeFnParamNames, jj);
    else
        aprimeFnParamsCell = {};
    end

    % --- EZ V_next Transformation ---
    valid_V = isfinite(V_next) & (V_next ~= 0);
    V_transformed = V_next;
    if ezc5(jj) == 1
        V_transformed(valid_V) = ezc4 * V_next(valid_V);
    else
        V_transformed(valid_V) = max(ezc4 * V_next(valid_V), 0).^ezc5(jj);
    end
    V_transformed(V_next == 0) = 0;

    % --- Sequential EV Computation (Applying Z and SemiZ transitions) ---
    N_semiz_local = 1;
    N_dsemiz = 1;
    if has_semiz && length(n_d) > 0
        N_semiz_local = max(1, prod(vfoptions.n_semiz));
        if isfield(vfoptions, 'l_dsemiz')
            N_dsemiz = prod(n_d(end-vfoptions.l_dsemiz+1:end));
        else
            N_dsemiz = n_d(end); % Default to the last decision variable
        end
    end
    N_z_exog = max(1, n_z_work / N_semiz_local);

    EV = zeros(N_a, N_semiz_local * N_z_exog, n_e_work, N_dsemiz, 'like', V_next);

    for ie = 1:n_e_work
        V_curr = V_transformed(:,:,ie);

        % 1. Apply Exogenous Z Transition (if it exists)
        if N_z_exog > 1 && has_z
            pi_z_j = pi_z_J(:, :, min(jj, size(pi_z_J, 3)));
            V_slice = reshape(V_curr, [N_a * N_semiz_local, N_z_exog]);
            V_z_eval = V_slice * pi_z_j';
            V_z_eval = reshape(V_z_eval, [N_a, N_semiz_local, N_z_exog]);
        else
            V_z_eval = reshape(V_curr, [N_a, N_semiz_local, N_z_exog]);
        end

        % 2. Apply Semi-Exogenous Transition (if it exists)
        if has_semiz
            pi_semiz_j = vfoptions.pi_semiz_J(:, :, :, min(jj, size(vfoptions.pi_semiz_J, 4)));

            % Permute to [N_semiz_local, N_a * N_z_exog] for matrix multiplication
            V_perm = reshape(permute(V_z_eval, [2, 1, 3]), [N_semiz_local, N_a * N_z_exog]);
            for idsemiz = 1:N_dsemiz
                pi_semiz_d = pi_semiz_j(:, :, idsemiz);
                EV_perm = pi_semiz_d * V_perm;
                EV_d = permute(reshape(EV_perm, [N_semiz_local, N_a, N_z_exog]), [2, 1, 3]);
                EV(:,:,ie,idsemiz) = reshape(EV_d, [N_a, N_semiz_local * N_z_exog]);
            end
        else
            EV(:,:,ie,1) = reshape(V_z_eval, [N_a, N_semiz_local * N_z_exog]);
        end
    end

    % sj is ones by default, but vfoptions and Parameters can change that
    EV = EV * sj(jj);

    % --- EZ Certainty Equivalent Reverse Transformation ---
    valid_EV = isfinite(EV) & (EV ~= 0);
    if ezc6(jj) ~= 1
        EV(valid_EV) = max(EV(valid_EV), 0).^ezc6(jj);
    end
    if ezc8(jj) ~= 1
        EV(valid_EV) = max(EV(valid_EV), 0).^ezc8(jj);
    end

    % Flatten to match the block tensor evaluator structure
    EV_flat_ze = reshape(EV, [N_a, N_ze, N_dsemiz]);

    [Z_mesh, E_mesh] = ndgrid(1:n_z_work, 1:n_e_work);
    ZE_z_idx = Z_mesh(:);
    ZE_e_idx = E_mesh(:);

    % --- The Master Orchestrator Pre-Computation ---
    N_d_safe = max(1, N_d);

    % Allocate GPU tensors for THIS period's slices
    V_j_max        = zeros(N_a, N_ze, 'like', V_next);
    Pol_apr_max    = zeros(N_a, N_ze, 'like', V_next);
    Pol_d_max      = zeros(N_a, N_ze, 'like', V_next);
    Pol_L2idx_max  = zeros(N_a, N_ze, 'like', V_next);
    Pol_L2flag_max = zeros(N_a, N_ze, 'like', V_next);

    % --- Pre-build dsemiz index tensor ---
    if N_dsemiz > 1
        if isfield(vfoptions, 'l_dsemiz')
            N_d_prefix = max(1, prod(n_d(1:end-vfoptions.l_dsemiz)));
        else
            N_d_prefix = max(1, prod(n_d(1:end-1)));
        end
        dsemiz_idx = ceil((1:N_d_safe)' / N_d_prefix);
        dsemiz_idx_tensor = reshape(dsemiz_idx, [N_d_safe, 1, 1, 1]);
    else
        dsemiz_idx_tensor = ones(N_d_safe, 1, 1, 1);
    end

    % --- The Master Orchestrator Loop ---
    if vfoptions.divideandconquer == 1
        for i_ze = 1:length(ze_chunks)
            curr_ze = ze_chunks{i_ze};
            N_ze_local = length(curr_ze);

            % 1. Slice EV and setup shock cells for this chunk
            start_idx = (min(curr_ze) - 1) * N_a + 1;
            end_idx   = max(curr_ze) * N_a;
            EV_local  = EV_flat_ze(:, curr_ze, :);

            if has_semiz || has_z
                num_z_vars = size(z_gridvals_J, 2);
                Z_cells_local = cell(1, num_z_vars);
                for iz = 1:num_z_vars
                    Z_cells_local{iz} = reshape(z_gridvals_J(ZE_z_idx(curr_ze), iz), [1, 1, 1, 1, N_ze_local]);
                end
            else
                Z_cells_local = {};
            end

            if has_e
                num_e_vars = size(e_work, 2);
                E_cells_local = cell(1, num_e_vars);
                for ie = 1:num_e_vars
                    E_cells_local{ie} = reshape(e_work(ZE_e_idx(curr_ze), ie), [1, 1, 1, 1, N_ze_local]);
                end
            else
                E_cells_local = {};
            end

            z_offset_local = reshape((0:N_ze_local-1) * N_a, [1, 1, 1, N_ze_local]);

            if vfoptions.gridinterplayer
                EV_interp_local = interp1(a1_work, reshape(EV_local, [N_a, N_ze_local * N_dsemiz]), a1prime_grid);
                EV_interp_local = reshape(EV_interp_local, [length(a1prime_grid), N_ze_local, N_dsemiz]);
                z_offset_fine_local = reshape((0:N_ze_local-1) * length(a1prime_grid), [1, 1, 1, N_ze_local]);
            else
                EV_interp_local = [];
                z_offset_fine_local = [];
            end

            % 2. Bind LocalBlockFn for the full asset space (N_a1 * N_a2) using N_ze_local
            LocalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar) Evaluate_Case1_TensorBlock(...
                state_idx, loweredge_matrix, maxgap_scalar, N_a1, N_a2, N_d_safe, N_ze_local, ...
                Z_cells_local, E_cells_local, D_cells_block, A1_mat, A2_mat, a2_grids_1d, l_a2, ...
                vfoptions.gridinterplayer, n2short, n2long, beta_j, EV_local, EV_interp_local, z_offset_local, z_offset_fine_local, a1prime_grid, ...
                TensorReturnFn, ReturnFnParamsCell, ezc2(jj), ezc3, ezc4, ezc7(jj), ...
                TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor);

            vfoptions.level1n = vfoptions.level1n(1);
            [v, p_apr, p_d, p_l2idx, p_l2flag] = ValueFnIter_DC1_Slicer(N_a1 * N_a2, N_a, 1, N_ze_local, vfoptions, LocalBlockFn);

            V_j_max(:, curr_ze)     = reshape(v,     [N_a1 * N_a2, N_ze_local]);
            Pol_apr_max(:, curr_ze) = reshape(p_apr, [N_a1 * N_a2, N_ze_local]);
            Pol_d_max(:, curr_ze)   = reshape(p_d,   [N_a1 * N_a2, N_ze_local]);
            if vfoptions.gridinterplayer(1) == 1
                Pol_L2idx_max(:, curr_ze)  = reshape(p_l2idx,  [N_a1 * N_a2, N_ze_local]);
                Pol_L2flag_max(:, curr_ze) = reshape(p_l2flag, [N_a1 * N_a2, N_ze_local]);
            end
        end
    else
        for i_a2 = 1:length(a2_chunks)
            curr_a2 = a2_chunks{i_a2};
            N_a2_local = length(curr_a2);
            start_a_idx = (min(curr_a2) - 1) * N_a1 + 1;
            end_a_idx   = max(curr_a2) * N_a1;

            for i_ze = 1:length(ze_chunks)
                curr_ze = ze_chunks{i_ze};
                N_ze_local = length(curr_ze);

                if l_a2 > 0
                    % Slice A2 locally for the current chunk
                    A2_local = A2_mat(curr_a2, :);
                else
                    A2_local = [];
                end
                N_a2_local = size(A2_local, 1);

                start_idx = (min(curr_ze) - 1) * N_a + 1;
                end_idx   = max(curr_ze) * N_a;
                EV_local  = EV_flat_ze(start_idx : end_idx);

                if has_semiz || has_z
                    num_z_vars = size(z_gridvals_J, 2);
                    Z_cells_local = cell(1, num_z_vars);
                    for iz = 1:num_z_vars
                        Z_cells_local{iz} = reshape(z_gridvals_J(ZE_z_idx(curr_ze), iz), [1, 1, 1, 1, N_ze_local]);
                    end
                else
                    Z_cells_local = {};
                end

                if has_e
                    num_e_vars = size(e_work, 2);
                    E_cells_local = cell(1, num_e_vars);
                    for ie = 1:num_e_vars
                        E_cells_local{ie} = reshape(e_work(ZE_e_idx(curr_ze), ie), [1, 1, 1, 1, N_ze_local]);
                    end
                else
                    E_cells_local = {};
                end

                z_offset_local = reshape((0:N_ze_local-1) * N_a, [1, 1, 1, N_ze_local]);

                if vfoptions.gridinterplayer
                    EV_interp_local = interp1(a1_work, reshape(EV_local, [N_a, N_ze_local * N_dsemiz]), a1prime_grid);
                    EV_interp_local = reshape(EV_interp_local, [length(a1prime_grid), N_ze_local, N_dsemiz]);
                    z_offset_fine_local = reshape((0:N_ze_local-1) * length(a1prime_grid), [1, 1, 1, N_ze_local]);
                else
                    EV_interp_local = [];
                    z_offset_fine_local = [];
                end

                LocalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar) Evaluate_Case1_TensorBlock(...
                    state_idx, loweredge_matrix, maxgap_scalar, N_a1, N_a2_local, N_d_safe, N_ze_local, ...
                    Z_cells_local, E_cells_local, D_cells_block, A1_mat, A2_local, a2_grids_1d, l_a2, ...
                    vfoptions.gridinterplayer, n2short, n2long, beta_j, EV_local, EV_interp_local, z_offset_local, z_offset_fine_local, a1prime_grid, ...
                    TensorReturnFn, ReturnFnParamsCell, ezc2(jj), ezc3, ezc4, ezc7(jj), ...
                    TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor);

                if vfoptions.divideandconquer == 1
                    vfoptions.level1n = vfoptions.level1n(1);
                    [v, p_apr, p_d, p_l2idx, p_l2flag] = ValueFnIter_DC1_Slicer(start_a_idx:end_a_idx, N_a, 1, N_ze_local, vfoptions, LocalBlockFn);
                else
                    [v, p_apr, p_d, p_l2idx, p_l2flag] = LocalBlockFn(start_a_idx:end_a_idx, [], 0);
                end

                if l_a2 > 0
                    N_a_local = N_a1 * N_a2_local;
                else
                    N_a_local = N_a1;
                end

                V_j_max(start_a_idx:end_a_idx, curr_ze)     = reshape(v,     [N_a_local, N_ze_local]);
                Pol_apr_max(start_a_idx:end_a_idx, curr_ze) = reshape(p_apr, [N_a_local, N_ze_local]);
                Pol_d_max(start_a_idx:end_a_idx, curr_ze)   = reshape(p_d,   [N_a_local, N_ze_local]);
                if vfoptions.gridinterplayer(1) == 1
                    Pol_L2idx_max(start_a_idx:end_a_idx, curr_ze)  = reshape(p_l2idx,  [N_a_local, N_ze_local]);
                    Pol_L2flag_max(start_a_idx:end_a_idx, curr_ze) = reshape(p_l2flag, [N_a_local, N_ze_local]);
                end
            end
        end
    end

    V_j_max     = reshape(V_j_max,     [N_a, n_z_work, n_e_work]);
    Pol_apr_max = reshape(Pol_apr_max, [N_a, n_z_work, n_e_work]);
    Pol_d_max   = reshape(Pol_d_max,   [N_a, n_z_work, n_e_work]);
    if vfoptions.gridinterplayer(1) == 1
        Pol_L2idx_max  = reshape(Pol_L2idx_max,  [N_a, n_z_work, n_e_work]);
        Pol_L2flag_max = reshape(Pol_L2flag_max, [N_a, n_z_work, n_e_work]);
    end

    if vfoptions.gridinterplayer(1) == 1
        adjust = (Pol_L2idx_max < 1 + n2short + 1);
        lower_grid_pt = Pol_apr_max - adjust;
        subgrid_step  = adjust .* Pol_L2idx_max + (1 - adjust) .* (Pol_L2idx_max - n2short - 1);
        if N_d > 0
            PolicyKron(1, :, :, :, jj) = (lower_grid_pt - 1) * N_d + Pol_d_max;
        else
            PolicyKron(1, :, :, :, jj) = lower_grid_pt;
        end
        PolicyKron(2, :, :, :, jj) = subgrid_step;
        PolicyKron(3, :, :, :, jj) = Pol_L2flag_max;
    else
        if N_d > 0
            PolicyKron(:, :, :, jj) = (Pol_apr_max - 1) * N_d + Pol_d_max;
        else
            PolicyKron(:, :, :, jj) = Pol_apr_max;
        end
    end
    V(:, :, :, jj) = V_j_max;
    V_next = V_j_max;
end

if N_z == 0
    V = squeeze(V);
end

if N_d == 0
    n_daprime = n_a(1:num_a1_pass);
else
    n_daprime = [n_d, n_a(1:num_a1_pass)];
end

if vfoptions.gridinterplayer(1) ~= 1
    PolicyKron = shiftdim(PolicyKron, -1);
end

if isfield(vfoptions, 'outputkron') && vfoptions.outputkron == 1
    varargout{1} = V;
    varargout{2} = PolicyKron;
    return
end

%% Iterative Policy Unpacking (System RAM Handoff)
% MATLAB GPUs enforce a strict 32-bit signed integer limit for array indexing
% (max 2,147,483,647 elements per array). For high-resolution models exceeding
% this limit, the Policy tensor is unpacked iteratively by period and assembled
% safely in 64-bit System RAM.

disp('Unpacking Policy tensor to System RAM...');

num_pol_vars = length(n_daprime);
n_daprime_col = n_daprime(:);
divisors = cumprod([1; n_daprime_col(1:end-1)]);

% Allocate in System RAM ('single', NOT 'like' PolicyKron)
Policy_flat = zeros([num_pol_vars, n_a_work, n_z_work, n_e_work, N_j], vfoptions.precision);

for jj = 1:N_j
    PK_j = PolicyKron(:,:,:,:,jj);
    % Compute the 148MB chunk on GPU, then gather immediately to CPU RAM
    P_j_gpu = mod(floor((PK_j - 1) ./ divisors), n_daprime_col) + 1;
    Policy_flat(:,:,:,:,jj) = gather(P_j_gpu);
end

% Gather V to CPU to keep memory domains aligned for StationaryDist
V_cpu = gather(V);

if has_z && has_e
    Policy = reshape(Policy_flat, [num_pol_vars, n_a, n_all_z, n_e_pass, N_j]);
    V = reshape(V_cpu, [n_a, n_all_z, n_e_pass, N_j]);
elseif has_z && ~has_e
    Policy = reshape(Policy_flat, [num_pol_vars, n_a, n_all_z, N_j]);
    V = reshape(V_cpu, [n_a, n_all_z, N_j]);
elseif ~has_z && has_e
    Policy = reshape(Policy_flat, [num_pol_vars, n_a, n_e_pass, N_j]);
    V = reshape(V_cpu, [n_a, n_e_pass, N_j]);
else
    Policy = reshape(Policy_flat, [num_pol_vars, n_a, N_j]);
    V = reshape(V_cpu, [n_a, N_j]);
end

varargout{1} = V;
varargout{2} = Policy;
end


function [V_j_max, Pol_apr_max, Pol_d_max, Pol_L2idx_max, Pol_L2flag_max] = Evaluate_Case1_TensorBlock(...
    state_idx, loweredge_matrix, maxgap_scalar, N_a1, N_a2, N_d_safe, N_ze_local, ...
    Z_cells_block, E_cells_block, D_cells_block, A1_mat, A2_mat, a2_grids_1d, l_a2, ...
    gridinterplayer, n2short, n2long, beta_j, EV_local, EV_interp_local, z_offset_local, z_offset_fine_local, a1prime_grid, ...
    TensorReturnFn, ReturnFnParamsCell, ezc2_j, ezc3, ezc4, ezc7_j, ...
    TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor)

% Inside Evaluate_Case1_TensorBlock, when building A2_mat or slicing:
% Instead of assuming state_idx starts at 1, map state_idx relative to the current chunk:
local_state_idx = state_idx - min(state_idx) + 1;
N_block = length(local_state_idx);

% 1. Build A1 and A2 Cells dynamically
num_a1 = size(A1_mat, 2);
Apr_cells = cell(1, num_a1);
A1_cells  = cell(1, num_a1);
for ia = 1:num_a1
    Apr_cells{ia} = reshape(A1_mat(:,ia), [1, N_a1, 1, 1, 1]);
    A1_cells{ia}  = reshape(A1_mat(:,ia), [1, 1, N_a1, 1, 1]);
end

% 2. Evaluate ReturnFn with raw numeric arrays for A2
if l_a2 > 0
    N_a2 = size(A2_mat, 1);
    num_a2 = size(A2_mat, 2);
    A2_cells = cell(1, num_a2);
    for ia = 1:num_a2
        A2_cells{ia} = reshape(A2_mat(:,ia), [1, 1, 1, N_a2, 1]);
    end
    F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, A2_cells{:}, Z_cells_block{:}, E_cells_block{:}, ReturnFnParamsCell{:});
else
    F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, Z_cells_block{:}, E_cells_block{:}, ReturnFnParamsCell{:});
end

% 3. Format Expected Values (EV_bounded)
if l_a2 > 0
    % ExpAsset Transition Interpolation
    A2_prime = TensoraprimeFn(D_cells_block{:}, A2_cells{:}, Z_cells_block{:}, aprimeFnParamsCell{:});
    a2_grid_1d_vec = a2_grids_1d{1};
    a2_min = a2_grid_1d_vec(1);
    a2_max = a2_grid_1d_vec(end);
    a2_prime_clipped = max(a2_min, min(A2_prime, a2_max));
    idx = discretize(a2_prime_clipped, a2_grid_1d_vec);
    idx(isnan(idx)) = N_a2 - 1;
    idx = max(1, min(idx, N_a2 - 1));
    a2_left = reshape(a2_grid_1d_vec(idx), size(idx));
    a2_right = reshape(a2_grid_1d_vec(idx+1), size(idx));
    weight = (a2_prime_clipped - a2_left) ./ (a2_right - a2_left);
    weight(a2_right == a2_left) = 0;

    A1pr_idx = reshape(1:N_a1, [1, N_a1, 1, 1, 1]);
    ZE_idx   = reshape(1:N_ze_local, [1, 1, 1, 1, N_ze_local]);

    % --- CORRECTED MULTI-SHOCK INDEXING OFFSET ---
    idx_left  = A1pr_idx + (idx - 1) * N_a1 + (ZE_idx - 1) * (N_a1 * N_a2);
    idx_right = A1pr_idx + (idx) * N_a1 + (ZE_idx - 1) * (N_a1 * N_a2);

    max_idx_row = size(EV_local, 1); % Dynamically match EV_local dimensions
    linear_idx_left  = min(max_idx_row, max(1, idx_left  + (dsemiz_idx_tensor - 1) * max_idx_row));
    linear_idx_right = min(max_idx_row, max(1, idx_right + (dsemiz_idx_tensor - 1) * max_idx_row));

    EV_left  = EV_local(linear_idx_left);
    EV_right = EV_local(linear_idx_right);
    EV_bounded = EV_left + weight .* (EV_right - EV_left);
else
    % Standard Endogenous
    apr_idx_tensor = reshape(1:N_a1, [1, N_a1, 1, 1, 1]);
    ZE_idx = reshape(0:N_ze_local-1, [1, 1, 1, 1, N_ze_local]);
    idx_base = apr_idx_tensor + ZE_idx * N_a1;

    max_idx_row = N_a1 * N_ze_local;
    linear_idx = idx_base + (dsemiz_idx_tensor - 1) * max_idx_row;

    EV_bounded = reshape(EV_local(linear_idx(:)), size(linear_idx));
end

% --- 4. RHS Evaluation, Choice Optimization, and State Slicing ---
FLAT_CHOICES = max(1, N_d_safe) * N_a1;
if l_a2 > 0
    N_a = N_a1 * N_a2;
else
    N_a = N_a1;
end

FLAT_STATES  = N_a * N_ze_local;

RHS = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bounded, beta_j, 1, ezc2_j, ezc3, ezc4, ezc7_j);
RHS_flat = reshape(RHS, [FLAT_CHOICES, FLAT_STATES]);

[V_sub_coarse, Pol_sub_idx] = max(RHS_flat, [], 1);

d_idx_local   = mod(Pol_sub_idx - 1, max(1, N_d_safe)) + 1;
apr_idx_local = ceil(Pol_sub_idx / max(1, N_d_safe));

% Reshape to full grid size first
V_full        = reshape(V_sub_coarse,   [N_a, N_ze_local]);
Pol_apr_full  = reshape(apr_idx_local, [N_a, N_ze_local]);
Pol_d_full    = reshape(d_idx_local,   [N_a, N_ze_local]);

% Sub-select ONLY the requested state_idx rows (crucial for D&C compatibility)
V_j_max        = V_full(state_idx, :);
Pol_apr_max    = Pol_apr_full(state_idx, :);
Pol_d_max      = Pol_d_full(state_idx, :);
Pol_L2idx_max  = [];
Pol_L2flag_max = [];
end