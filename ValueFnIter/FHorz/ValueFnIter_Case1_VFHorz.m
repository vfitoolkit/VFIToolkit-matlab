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

        % 1. Count Decision Variables (D)
        % Safely handles scalar 0, empty arrays, or multi-dimensional flags
        if isequal(n_d, 0) || isempty(n_d)
            num_d_vars = 0;
        else
            num_d_vars = length(n_d);
        end

        % 2. Count Exogenous Variables (Z)
        if isequal(n_z, 0) || isempty(n_z)
            num_z_vars = 0;
        else
            num_z_vars = length(n_z);
        end

        % 3. Extract and Split Asset Variables (A1 and A2) early
        l_a2 = 0;
        if vfoptions.experienceasset > 0
            l_a2 = vfoptions.experienceasset;
        elseif vfoptions.experienceassetz > 0
            l_a2 = vfoptions.experienceassetz;
        end
        num_a2 = l_a2;
        num_a1 = length(n_a) - num_a2;

        % 4. Count Semi-Exogenous (SemiZ), Transitory (E), and Ambiguity/Risky (U) shocks
        num_semiz_vars = 0;
        if isfield(vfoptions, 'n_semiz') && prod(vfoptions.n_semiz) > 0
            num_semiz_vars = length(vfoptions.n_semiz);
        end

        num_e_vars = 0;
        if isfield(vfoptions, 'n_e') && prod(vfoptions.n_e) > 0
            num_e_vars = length(vfoptions.n_e);
        end

        num_u_vars = 0;
        if vfoptions.riskyasset == 1 && isfield(vfoptions, 'n_u')
            num_u_vars = length(vfoptions.n_u);
        end

        % 5. Unified Prefix Argument Count
        % Fundamentally covers all toolkit variants (Standard, ExpAsset, RiskyAsset)
        % Structure: D + A1prime (num_a1) + A1 (num_a1) + A2 (num_a2) + SemiZ + Z + E + U
        num_prefix_args = num_d_vars + (2 * num_a1) + num_a2 + num_semiz_vars + num_z_vars + num_e_vars + num_u_vars;

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

%% Exogenous shock gridvals and pi
if vfoptions.alreadygridvals==0
    [z_gridvals_J, pi_z_J, vfoptions] = ExogShockSetup_FHorz(n_z, z_grid, pi_z, N_j, Parameters, vfoptions, 3, 0);
else
    z_gridvals_J = z_grid;
    pi_z_J = pi_z;
end

%% Semi-exogenous shock gridvals and pi
if isfield(vfoptions, 'n_semiz') && prod(vfoptions.n_semiz) > 0
    N_semiz = prod(vfoptions.n_semiz);
else
    N_semiz = 0;
end

if vfoptions.alreadygridvals_semiexo==0
    if N_semiz > 0
        vfoptions = SemiExogShockSetup_FHorz(n_d, N_j, d_grid, Parameters, vfoptions, 3);
    end
end

% --- Tensor Bridge: Combine Z and SemiZ into a single Cartesian state space ---
N_d = prod(n_d);
N_a = prod(n_a);
N_z = prod(n_z);
N_z_safe = max(1, N_z);

if N_semiz > 0 && isfield(vfoptions, 'semiz_gridvals_J')
    sz_J = vfoptions.semiz_gridvals_J;
    num_semiz_vars = size(sz_J, 2);
    num_periods = size(sz_J, 3);
    if N_z > 0
        num_z_vars = size(z_gridvals_J, 2);
    else
        num_z_vars = 0;
    end

    z_gridvals_J_combined = zeros(N_semiz * max(1, N_z), num_semiz_vars + num_z_vars, num_periods, 'like', sz_J);
    for t = 1:num_periods
        if N_z > 0
            semiz_expanded = kron(sz_J(:,:,t), ones(N_z, 1));
            z_expanded = kron(ones(N_semiz, 1), z_gridvals_J(:,:,t));
            z_gridvals_J_combined(:,:,t) = [semiz_expanded, z_expanded];
        else
            z_gridvals_J_combined(:,:,t) = sz_J(:,:,t);
        end
    end
    z_gridvals_J = z_gridvals_J_combined;
    n_combined_z = [vfoptions.n_semiz, n_z];
else
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
[TensorReturnFn, D_cells_block, A1_cells, ~, ~] = CreateTensorFnAndCells(ReturnFn, n_d, n_a1, n_combined_z, n_e_pass, d_grid, a1_grid_vals, [], []);

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

    % Compute interpolation indices and weights ONCE
    idx = discretize(a1prime_grid, a1_work);
    idx(isnan(idx) | idx == length(a1_work)) = length(a1_work) - 1;

    interp_left_idx = idx(:);
    interp_right_idx = idx(:) + 1;

    a1_left = a1_work(interp_left_idx);
    a1_right = a1_work(interp_right_idx);
    interp_weights = (a1prime_grid(:) - a1_left) ./ (a1_right - a1_left);
    interp_weights(a1_right == a1_left) = 0;

    % Move to GPU if necessary
    if vfoptions.parallel == 2
        interp_left_idx = gpuArray(interp_left_idx);
        interp_right_idx = gpuArray(interp_right_idx);
        interp_weights = gpuArray(interp_weights);
    end
else
    n2short = 0;
    n2long  = 0;
    a1prime_grid = [];
    interp_left_idx = [];
    interp_right_idx = [];
    interp_weights = [];
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
    % Chunk perfectly along the E dimension to maintain Cartesian orthogonality
    e_chunk_size = max(1, floor(300 / n_z_work));
    num_chunks = ceil(n_e_work / e_chunk_size);
    ze_chunks = cell(1, num_chunks);
    for c = 1:num_chunks
        e_start = (c-1)*e_chunk_size + 1;
        e_end   = min(c*e_chunk_size, n_e_work);
        
        % Build exactly the linear indices for this Cartesian block
        [Z_sub, E_sub] = ndgrid(1:n_z_work, e_start:e_end);
        ze_chunks{c} = sub2ind([n_z_work, n_e_work], Z_sub(:), E_sub(:))';
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

% --- PRE-COMPUTE CHUNK METADATA ONCE ---
chunk_meta = cell(1, length(ze_chunks));
for i_ze = 1:length(ze_chunks)
    c_ze = ze_chunks{i_ze};

    % Perform index math on CPU to avoid GPU sorting overhead
    if isa(c_ze, 'gpuArray'), c_ze_cpu = gather(c_ze); else, c_ze_cpu = c_ze; end
    [z_ind, e_ind] = ind2sub([n_z_work, n_e_work], c_ze_cpu);

    meta.z_vals = unique(z_ind);
    meta.e_vals = unique(e_ind);
    meta.n_z_loc = length(meta.z_vals);
    meta.n_e_loc = length(meta.e_vals);
    meta.N_ze_local = length(c_ze);
    meta.z_offset_local = reshape((0:meta.N_ze_local-1) * N_a, [1, 1, 1, meta.N_ze_local]);

    % If using gridinterplayer, pre-compute fine offset
    if vfoptions.gridinterplayer(1) == 1
        meta.z_offset_fine_local = reshape((0:meta.N_ze_local-1) * length(a1prime_grid), [1, 1, 1, meta.N_ze_local]);
    else
        meta.z_offset_fine_local = [];
    end

    chunk_meta{i_ze} = meta;
end

% Initialize base parameters once
base_ReturnFnParamsCell = CreateCellFromParams(Parameters, ReturnFnParamNames, 1, vfoptions.precision);

% Identify which parameters are age-dependent (length == N_j)
is_age_dependent = false(1, length(ReturnFnParamNames));
for ip = 1:length(ReturnFnParamNames)
    if numel(Parameters.(ReturnFnParamNames{ip})) == N_j
        is_age_dependent(ip) = true;
    end

    % Force GPU typing on static parameters once
    if vfoptions.parallel == 2 && isnumeric(base_ReturnFnParamsCell{ip}) && ~isa(base_ReturnFnParamsCell{ip}, 'gpuArray')
        base_ReturnFnParamsCell{ip} = gpuArray(base_ReturnFnParamsCell{ip});
    end
end

for reverse_j = 0:N_j-1
    jj = N_j - reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

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

    ReturnFnParamsCell = base_ReturnFnParamsCell;

    % Update ONLY the age-dependent parameters
    for ip = find(is_age_dependent)
        val = cast(Parameters.(ReturnFnParamNames{ip})(jj), vfoptions.precision);
        if vfoptions.parallel == 2
            ReturnFnParamsCell{ip} = gpuArray(val);
        else
            ReturnFnParamsCell{ip} = val;
        end
    end
    DiscountFactorParamsVec = CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj, vfoptions.precision);
    beta_j = prod(DiscountFactorParamsVec);

    if l_a2 > 0
        aprimeFnParamsCell = CreateCellFromParams(Parameters, aprimeFnParamNames, jj);
    else
        aprimeFnParamsCell = {};
    end

    % Pre-cell-ify the full z and e grids once per period rather than calling cellfun/reshape repeatedly
    if has_semiz || has_z
        z_current_slice = z_gridvals_J(:,:,min(jj, size(z_gridvals_J,3)));
        num_z_vars = size(z_current_slice, 2);
        Z_cells = cell(1, num_z_vars);
        for iz = 1:num_z_vars
            Z_cells{iz} = z_current_slice(:, iz);
        end
    else
        Z_cells = {};
    end

    % Do we need to handle e_gridvals_J?
    if has_e
        num_e_vars = size(e_work, 2);
        E_cells = cell(1, num_e_vars);
        for ie = 1:num_e_vars
            E_cells{ie} = e_work(:, ie);
        end
    else
        E_cells = {};
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

    % =================================================================
    % --- i.i.d. Shock (e) Integration ---
    % =================================================================
    if has_e
        % Ensure the probability vector is on the GPU to prevent mtimes crashes
        if vfoptions.parallel == 2 && ~isa(vfoptions.pi_e, 'gpuArray')
            vfoptions.pi_e = gpuArray(vfoptions.pi_e);
        end

        % The agent does not know next period's i.i.d. shock. 
        % We must integrate out the future e dimension before applying Markov transitions.
        V_trans_flat = reshape(V_transformed, [N_a * n_z_work, n_e_work]);
        V_expected_e = V_trans_flat * vfoptions.pi_e(:);

        % Expand back out to [N_a, n_z_work, n_e_work] so the tensor slicing 
        % implicitly maps the identical expectation across all current e states.
        V_transformed = repmat(reshape(V_expected_e, [N_a, n_z_work, 1]), [1, 1, n_e_work]);
    end

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
            % Look up pre-computed bounds instantly
            meta = chunk_meta{i_ze};
            n_z_loc = meta.n_z_loc;
            n_e_loc = meta.n_e_loc;
            z_offset_local = meta.z_offset_local;
            z_offset_fine_local = meta.z_offset_fine_local;

            % 1. Extract the actual grid indices for this chunk
            curr_ze = ze_chunks{i_ze};
            N_ze_local = length(curr_ze);

            % 2. Slice EV and setup shock cells for this chunk
            EV_local = EV_flat_ze(:, curr_ze, :);

            if has_semiz || has_z
                % STRICT FIX: Lock to the exact passed dimensions, ignore grid shape
                num_z_vars = length(n_combined_z);
                Z_cells_local = cell(1, num_z_vars);

                % If z_gridvals_J remained a 2D stacked matrix [19x76] due to PType bypass,
                % we MUST slice it manually into its Cartesian Z-components.
                if size(z_gridvals_J, 2) ~= num_z_vars
                    % Recover the 3D tensor shape [34, 2, 76] dynamically
                    z_inflated = reshape(z_gridvals_J, [N_z, num_z_vars, size(z_gridvals_J, ndims(z_gridvals_J))]);
                    for iz = 1:num_z_vars
                        Z_cells_local{iz} = reshape(z_inflated(meta.z_vals, iz, min(jj, size(z_inflated,3))), [1, 1, 1, n_z_loc, 1]);
                    end
                else
                    for iz = 1:num_z_vars
                        Z_cells_local{iz} = reshape(z_gridvals_J(meta.z_vals, iz, min(jj, size(z_gridvals_J,3))), [1, 1, 1, n_z_loc, 1]);
                    end
                end
            else
                Z_cells_local = {};
            end

            if has_e
                num_e_vars = size(e_work, 2);
                E_cells_local = cell(1, num_e_vars);
                for ie_var = 1:num_e_vars
                    E_cells_local{ie_var} = reshape(e_work(meta.e_vals, ie_var), [1, 1, 1, 1, n_e_loc]);
                end
            else
                E_cells_local = {};
            end

            % 3. Flatten EV_local to [N_a, N_cols] for 2D indexing
            if vfoptions.gridinterplayer(1) == 1
                N_cols = N_ze_local * N_dsemiz;
                EV_2d = reshape(EV_local, [N_a, N_cols]);

                % Fast manual linear interpolation
                EV_left_val = EV_2d(interp_left_idx, :);
                EV_right_val = EV_2d(interp_right_idx, :);
                EV_interp_flat = EV_left_val + interp_weights .* (EV_right_val - EV_left_val);

                EV_interp_local = reshape(EV_interp_flat, [length(a1prime_grid), N_ze_local, N_dsemiz]);
            else
                EV_interp_local = [];
            end

            % 4. --- HOIST EV_BOUNDED: Compute once per chunk, not per slice! ---
            if l_a2 == 0
                % Because chunking is perfectly Cartesian, N_ze_local == n_z_loc * n_e_loc
                EV_reshaped = reshape(EV_local, [N_a1, n_z_loc, n_e_loc, N_dsemiz]);

                % Extract the exact D slices using dsemiz_idx_tensor [N_d_safe, 1, 1, 1]
                EV_d_sliced = EV_reshaped(:, :, :, dsemiz_idx_tensor(:));

                % Permute to broadcast shape: [N_d_safe, N_a1, 1, n_z_loc, n_e_loc]
                EV_bounded_pre = beta_j .* permute(EV_d_sliced, [4, 1, 5, 2, 3]);
                % By casting this 'like' EV_bounded_pre, it lives permanently on the GPU 
                % and completely prevents PCIe bus transfers during the DC zoom loop.
                d_vec = reshape(0:N_d_safe-1, [N_d_safe, 1, 1, 1, 1]);
                z_vec = reshape((0:n_z_loc-1) * (N_d_safe * N_a1), [1, 1, 1, n_z_loc, 1]);
                e_vec = reshape((0:n_e_loc-1) * (N_d_safe * N_a1 * n_z_loc), [1, 1, 1, 1, n_e_loc]);
                static_EV_offset = cast(d_vec + 1 + z_vec + e_vec, 'like', EV_bounded_pre);
            else
                EV_bounded_pre = [];
                static_EV_offset = [];
            end

            % 5. Bind LocalBlockFn passing unmixed global dimensions for broadcasting
            LocalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar) Evaluate_Case1_TensorBlock(...
                state_idx, loweredge_matrix, maxgap_scalar, N_a1, N_a2, N_d_safe, N_ze_local, ...
                Z_cells_local, E_cells_local, D_cells_block, A1_mat, A2_mat, a2_grids_1d, l_a2, ...
                vfoptions.gridinterplayer, n2short, n2long, beta_j, EV_local, EV_bounded_pre, EV_interp_local, a1prime_grid, ...
                TensorReturnFn, ReturnFnParamsCell, ezc2(jj), ezc3, ezc4, ezc7(jj), ...
                TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor, n_z_loc, n_e_loc, static_EV_offset);

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

                % We need to pull these for the static offset hoist and the TensorBlock!
                meta = chunk_meta{i_ze};
                n_z_loc = meta.n_z_loc;
                n_e_loc = meta.n_e_loc;

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
                    num_z_vars = length(n_combined_z);
                    Z_cells_local = cell(1, num_z_vars);
                    if size(z_gridvals_J, 2) ~= num_z_vars
                        z_inflated = reshape(z_gridvals_J, [N_z, num_z_vars, size(z_gridvals_J, ndims(z_gridvals_J))]);
                        for iz = 1:num_z_vars
                            Z_cells_local{iz} = reshape(z_inflated(ZE_z_idx(curr_ze), iz, min(jj, size(z_inflated,3))), [1, 1, 1, 1, N_ze_local]);
                        end
                    else
                        for iz = 1:num_z_vars
                            Z_cells_local{iz} = reshape(z_gridvals_J(ZE_z_idx(curr_ze), iz, min(jj, size(z_gridvals_J,3))), [1, 1, 1, 1, N_ze_local]);
                        end
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

                % --- HOIST EV_BOUNDED for non-DC loop ---
                if l_a2 == 0
                    apr_idx_tensor = reshape(1:N_a1, [1, N_a1, 1, 1, 1]);
                    z_offset_broadcast = reshape((0:N_ze_local-1) * N_a, [1, 1, 1, 1, N_ze_local]);
                    idx_base = apr_idx_tensor + z_offset_broadcast;
                    max_idx_row = N_a1 * n_z_work * n_e_work;
                    linear_idx_pre = idx_base + (dsemiz_idx_tensor - 1) * max_idx_row;

                    EV_bounded_pre = beta_j .* reshape(EV_local(linear_idx_pre(:)), [N_d_safe, N_a1, 1, 1, N_ze_local]);
                    % --- NEW: Hoist static offset math for DC Zoom Scenario A ---
                    % By casting this 'like' EV_bounded_pre, it lives permanently on the GPU 
                    % and completely prevents PCIe bus transfers during the DC zoom loop.
                    d_vec = reshape(0:N_d_safe-1, [N_d_safe, 1, 1, 1, 1]);
                    z_vec = reshape((0:n_z_loc-1) * (N_d_safe * N_a1), [1, 1, 1, n_z_loc, 1]);
                    e_vec = reshape((0:n_e_loc-1) * (N_d_safe * N_a1 * n_z_loc), [1, 1, 1, 1, n_e_loc]);
                    static_EV_offset = cast(d_vec + 1 + z_vec + e_vec, 'like', EV_bounded_pre);
                else
                    EV_bounded_pre = [];
                    static_EV_offset = [];
                end

                LocalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar) Evaluate_Case1_TensorBlock(...
                    state_idx, loweredge_matrix, maxgap_scalar, N_a1, N_a2_local, N_d_safe, N_ze_local, ...
                    Z_cells_local, E_cells_local, D_cells_block, A1_mat, A2_local, a2_grids_1d, l_a2, ...
                    vfoptions.gridinterplayer, n2short, n2long, beta_j, EV_local, EV_bounded_pre, EV_interp_local, a1prime_grid, ...
                    TensorReturnFn, ReturnFnParamsCell, ezc2(jj), ezc3, ezc4, ezc7(jj), ...
                    TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor, n_z_loc, n_e_loc, static_EV_offset);

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

%% Smart Policy Unpacking (System RAM Handoff)
% MATLAB GPUs enforce a strict 32-bit signed integer limit for array indexing (~2.14B).
% We check the total size: if safe, we do a high-speed bulk unpack.
% If massive, we fall back to the iterative memory-safe loop.

disp('Unpacking Policy tensor to System RAM...');
num_pol_vars = length(n_daprime);
n_daprime_col = n_daprime(:);
divisors = cumprod([1; n_daprime_col(1:end-1)]);

total_elements = num_pol_vars * n_a_work * n_z_work * n_e_work * N_j;
MAX_INT32 = 2147483647;

if total_elements < (MAX_INT32 * 0.9) % 90% safety margin threshold
    % --- FAST PATH: Single Bulk Operation ---
    % Implicit expansion applies the divisors across the entire 5D tensor at once
    P_gpu = mod(floor((PolicyKron - 1) ./ divisors), n_daprime_col) + 1;
    Policy_flat = gather(P_gpu);
else
    % --- SLOW PATH: Iterative Unpacking ---
    disp('Using memory-safe iterative unpacking due to massive array size...');
    Policy_flat = zeros([num_pol_vars, n_a_work, n_z_work, n_e_work, N_j], vfoptions.precision);
    for jj = 1:N_j
        PK_j = PolicyKron(:,:,:,:,jj);
        P_j_gpu = mod(floor((PK_j - 1) ./ divisors), n_daprime_col) + 1;
        Policy_flat(:,:,:,:,jj) = gather(P_j_gpu);
    end
end

% Gather V to CPU to keep memory domains aligned for StationaryDist
% V_cpu = gather(V);
V_cpu = V;

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
    gridinterplayer, n2short, n2long, beta_j, EV_local, EV_bounded_pre, EV_interp_local, a1prime_grid, ...
    TensorReturnFn, ReturnFnParamsCell, ezc2_j, ezc3, ezc4, ezc7_j, ...
    TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor, n_z_loc, n_e_loc, static_EV_offset)

N_states = length(state_idx);

if l_a2 > 0
    N_a2_dims = size(A2_mat, 1);
    [a1_sub, a2_sub] = ind2sub([N_a1, N_a2_dims], state_idx);
else
    a1_sub = state_idx;
end

% 1. Build A1 Cells dynamically for EXACTLY the states requested
num_a1 = size(A1_mat, 2);
A1_cells  = cell(1, num_a1);
for ia = 1:num_a1
    A1_cells{ia}  = reshape(A1_mat(a1_sub, ia), [1, 1, N_states, 1, 1]);
end

if l_a2 > 0
    num_a2 = size(A2_mat, 2);
    A2_cells = cell(1, num_a2);
    for ia = 1:num_a2
        A2_cells{ia} = reshape(A2_mat(a2_sub, ia), [1, 1, N_states, 1, 1]);
    end
else
    A2_cells = {};
end


if maxgap_scalar == 0
    % =================================================================
    % BRANCH 1: COARSE GRID EVALUATION (maxgap_scalar == 0)
    % =================================================================
    Apr_cells = cell(1, num_a1);
    for ia = 1:num_a1
        Apr_cells{ia} = reshape(A1_mat(:,ia), [1, N_a1, 1, 1, 1]);
    end

    if l_a2 > 0
        F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, A2_cells{:}, Z_cells_block{:}, E_cells_block{:}, ReturnFnParamsCell{:});
    else
        F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, Z_cells_block{:}, E_cells_block{:}, ReturnFnParamsCell{:});
    end

    if l_a2 > 0
        % ExpAsset Transition Interpolation
        A2_prime = TensoraprimeFn(D_cells_block{:}, A2_cells{:}, Z_cells_block{:}, E_cells_block{:}, aprimeFnParamsCell{:});
        a2_grid_1d_vec = a2_grids_1d{1};
        a2_min = a2_grid_1d_vec(1);
        a2_max = a2_grid_1d_vec(end);
        a2_prime_clipped = max(a2_min, min(A2_prime, a2_max));
        idx = discretize(a2_prime_clipped, a2_grid_1d_vec);
        idx(isnan(idx)) = N_a2_dims - 1;
        idx = max(1, min(idx, N_a2_dims - 1));
        a2_left = reshape(a2_grid_1d_vec(idx), size(idx));
        a2_right = reshape(a2_grid_1d_vec(idx+1), size(idx));
        weight = (a2_prime_clipped - a2_left) ./ (a2_right - a2_left);
        weight(a2_right == a2_left) = 0;

        A1pr_idx = reshape(1:N_a1, [1, N_a1, 1, 1, 1]);
        ZE_idx   = reshape(1:N_ze_local, [1, 1, 1, 1, N_ze_local]);

        idx_left  = A1pr_idx + (idx - 1) * N_a1 + (ZE_idx - 1) * (N_a1 * N_a2_dims);
        idx_right = A1pr_idx + (idx) * N_a1 + (ZE_idx - 1) * (N_a1 * N_a2_dims);

        max_idx_row = size(EV_local, 1);
        linear_idx_left  = min(max_idx_row, max(1, idx_left  + (dsemiz_idx_tensor - 1) * max_idx_row));
        linear_idx_right = min(max_idx_row, max(1, idx_right + (dsemiz_idx_tensor - 1) * max_idx_row));

        EV_left  = EV_local(linear_idx_left);
        EV_right = EV_local(linear_idx_right);
        EV_bounded = EV_left + weight .* (EV_right - EV_left);
        EV_bounded = beta_j .* EV_bounded;
    else
        EV_bounded = EV_bounded_pre;
    end

    FLAT_CHOICES = max(1, N_d_safe) * N_a1;
    FLAT_STATES  = N_states * N_ze_local;

    RHS = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bounded, 1, 1, ezc2_j, ezc3, ezc4, ezc7_j);
    RHS_flat = reshape(RHS, [FLAT_CHOICES, FLAT_STATES]);
    [V_sub_coarse, Pol_sub_idx] = max(RHS_flat, [], 1);

    d_idx_local   = mod(Pol_sub_idx - 1, max(1, N_d_safe)) + 1;
    apr_idx_local = ceil(Pol_sub_idx / max(1, N_d_safe));

    V_j_max        = reshape(V_sub_coarse,  [N_states, N_ze_local]);
    Pol_apr_max    = reshape(apr_idx_local, [N_states, N_ze_local]);
    Pol_d_max      = reshape(d_idx_local,   [N_states, N_ze_local]);
    Pol_L2idx_max  = [];
    Pol_L2flag_max = [];

else
    % =================================================================
    % BRANCH 2: DC ZOOM PHASE (maxgap_scalar > 0)
    % =================================================================

    num_states_lower = size(loweredge_matrix, 1);
    if num_states_lower == 1 && N_states > 1
        loweredge_matrix = repmat(loweredge_matrix, N_states, 1);
    end

    if gridinterplayer(1) == 0
        % -------------------------------------------------------------
        % SCENARIO A: Standard DC Segment Zoom (No Interpolation)
        % -------------------------------------------------------------
        num_choices = maxgap_scalar + 1;

        % Base index strictly maps to the coarse grid
        base_idx = reshape(loweredge_matrix, [1, 1, N_states, n_z_loc, n_e_loc]);
        offsets = reshape(0:maxgap_scalar, [1, num_choices, 1, 1, 1]);
        choice_idx = base_idx + offsets;
        choice_idx = max(1, min(choice_idx, N_a1)); % Safety bound

        Apr_cells = cell(1, num_a1);
        for ia = 1:num_a1
            % FAST EXTRACT: Indexing a column vector natively returns an array
            % of the exact same ND-shape. This completely bypasses reshape().
            grid_col = A1_mat(:, ia);
            Apr_cells{ia} = grid_col(choice_idx);
        end

        if l_a2 > 0
            F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, A2_cells{:}, Z_cells_block{:}, E_cells_block{:}, ReturnFnParamsCell{:});

            % Standard ExpAsset transition for local choices
            A2_prime = TensoraprimeFn(D_cells_block{:}, A2_cells{:}, Z_cells_block{:}, E_cells_block{:}, aprimeFnParamsCell{:});
            a2_grid_1d_vec = a2_grids_1d{1};
            a2_prime_clipped = max(a2_grid_1d_vec(1), min(A2_prime, a2_grid_1d_vec(end)));

            idx = discretize(a2_prime_clipped, a2_grid_1d_vec);
            idx(isnan(idx)) = N_a2_dims - 1;
            idx = max(1, min(idx, N_a2_dims - 1));

            a2_left = reshape(a2_grid_1d_vec(idx), size(idx));
            a2_right = reshape(a2_grid_1d_vec(idx+1), size(idx));
            weight = (a2_prime_clipped - a2_left) ./ (a2_right - a2_left);
            weight(a2_right == a2_left) = 0;

            ZE_idx = reshape(1:N_ze_local, [1, 1, 1, 1, N_ze_local]);
            idx_left  = choice_idx + (idx - 1) * N_a1 + (ZE_idx - 1) * (N_a1 * N_a2_dims);
            idx_right = choice_idx + (idx) * N_a1 + (ZE_idx - 1) * (N_a1 * N_a2_dims);

            max_idx_row = size(EV_local, 1);
            linear_idx_left  = min(max_idx_row, max(1, idx_left  + (dsemiz_idx_tensor - 1) * max_idx_row));
            linear_idx_right = min(max_idx_row, max(1, idx_right + (dsemiz_idx_tensor - 1) * max_idx_row));

            EV_bounded = EV_local(linear_idx_left) + weight .* (EV_local(linear_idx_right) - EV_local(linear_idx_left));
            EV_bounded = beta_j .* EV_bounded;
        else
            F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, Z_cells_block{:}, E_cells_block{:}, ReturnFnParamsCell{:});

            % 100% Static Offset Extraction
            a_offset = (choice_idx - 1) * N_d_safe;
            linear_idx = static_EV_offset + a_offset;
            EV_bounded = EV_bounded_pre(linear_idx);
        end

    else
        % -------------------------------------------------------------
        % SCENARIO B: Grid Interpolation Zoom (a1prime_grid)
        % -------------------------------------------------------------
        num_choices = n2long;

        % Scale coarse grid indices to fine grid bounds
        L2_base = (loweredge_matrix - 1) * (n2short + 1) + 1;
        base_idx = reshape(L2_base, [1, 1, N_states, n_z_loc, n_e_loc]);

        % CRITICAL FIX 1: Offsets must be perfectly symmetric around the L2 base index
        start_offset = -(n2short + 1);
        end_offset   = (n2short + 1);
        offsets = reshape(start_offset:end_offset, [1, num_choices, 1, 1, 1]);

        raw_choice_idx = base_idx + offsets;

        % Identify out-of-bounds indices so we can penalize them later
        out_of_bounds = (raw_choice_idx < 1) | (raw_choice_idx > length(a1prime_grid));

        % Safely clip to prevent indexing errors in the grid extraction
        choice_idx = max(1, min(raw_choice_idx, length(a1prime_grid)));

        Apr_cells = cell(1, num_a1);
        for ia = 1:num_a1
            Apr_cells{ia} = a1prime_grid(choice_idx);
        end

        if l_a2 > 0
            error('Experience asset with gridinterplayer=1 is not yet supported in the tensor bridge L2 phase.');
        else
            F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, Z_cells_block{:}, E_cells_block{:}, ReturnFnParamsCell{:});

            % PENALIZE OUT-OF-BOUNDS CHOICES WITH NaN
            % Uses native GPU implicit expansion to completely bypass the massive
            % memory allocation and deallocation overhead of repmat()
            penalty = zeros(size(out_of_bounds), 'like', F_tensor);
            penalty(out_of_bounds) = NaN;
            F_tensor = F_tensor + penalty;

            EV_interp_reshaped = reshape(EV_interp_local, [length(a1prime_grid), n_z_loc, n_e_loc, N_dsemiz]);
            z_offset = reshape((0:n_z_loc-1) * length(a1prime_grid), [1, 1, 1, n_z_loc, 1]);
            e_offset = reshape((0:n_e_loc-1) * (length(a1prime_grid) * n_z_loc), [1, 1, 1, 1, n_e_loc]);

            L2_linear_idx = choice_idx + z_offset + e_offset;

            if N_dsemiz > 1
                L2_linear_idx = L2_linear_idx + (dsemiz_idx_tensor - 1) * (length(a1prime_grid) * n_z_loc * n_e_loc);
            end

            EV_bounded = EV_interp_reshaped(L2_linear_idx);
            EV_bounded = beta_j .* EV_bounded;
        end
    end

    % --- RHS Evaluation (Universal to both Zoom Scenarios) ---
    FLAT_STATES = N_states * n_z_loc * n_e_loc;

    RHS = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bounded, 1, 1, ezc2_j, ezc3, ezc4, ezc7_j);
    RHS_flat = reshape(RHS, [max(1, N_d_safe) * num_choices, FLAT_STATES]);

    [V_sub_fine, Pol_sub_idx] = max(RHS_flat, [], 1);

    d_idx_local = mod(Pol_sub_idx - 1, max(1, N_d_safe)) + 1;
    apr_offset  = ceil(Pol_sub_idx / max(1, N_d_safe));

    V_j_max   = reshape(V_sub_fine,  [N_states, n_z_loc * n_e_loc]);
    Pol_d_max = reshape(d_idx_local, [N_states, n_z_loc * n_e_loc]);

    if gridinterplayer(1) == 0
        % Standard DC: exact choice maps directly to the coarse grid
        base_idx_flat = reshape(base_idx, [1, FLAT_STATES]);
        absolute_idx_flat = base_idx_flat + apr_offset - 1;

        Pol_apr_max    = reshape(absolute_idx_flat, [N_states, n_z_loc * n_e_loc]);
        Pol_L2idx_max  = [];
        Pol_L2flag_max = [];
    else
        % CRITICAL FIX 2: Pol_L2idx_max must be the RELATIVE offset (1 to n2long).
        % Pol_apr_max must be preserved as the coarse grid base index.
        Pol_apr_max    = reshape(loweredge_matrix, [N_states, n_z_loc * n_e_loc]);
        Pol_L2idx_max  = reshape(apr_offset, [N_states, n_z_loc * n_e_loc]);
        Pol_L2flag_max = ones(N_states, n_z_loc * n_e_loc, 'like', V_j_max);
    end
end


end
