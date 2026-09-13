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
    elseif vfoptions.gridinterplayer==1
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
end

is_EZ = strcmp(vfoptions.exoticpreferences, 'EpsteinZin');
if is_EZ
    % Reject asset types this dispatcher does not handle: every asset type it does handle is
    % dispatched below and returns, so an unsupported flag would otherwise be silently ignored.
    if vfoptions.experienceasset>=1 || vfoptions.experienceassetu>=1 || vfoptions.experienceassetz>=1 || vfoptions.experienceassete>=1 || vfoptions.experienceassetze>=1 || vfoptions.experienceassetsemiz>=1
        error('Epstein-Zin preferences are not implemented for the experience assets (only for riskyasset, or for the standard endogenous states)')
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
    [ezc2, ezc3, ezc4, ezc5, ezc6, ezc7, ezc8, sj, warmglow] = ...
        EpsteinZinSetup_FHorz(N_j, Parameters, ReturnFnParamNames, DiscountFactorParamNames, vfoptions);
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

if isempty(ReturnFnParamNames)
    ReturnFnParamNames = ReturnFnParamNamesFn(ReturnFn, n_d, n_a, n_z, N_j, vfoptions, Parameters);
end

if vfoptions.parallel == 2
    if ~isempty(d_grid), d_grid = gpuArray(d_grid); end
    if ~isempty(a_grid), a_grid = gpuArray(a_grid); end
    if ~isempty(z_grid), z_grid = gpuArray(z_grid); end
    if ~isempty(pi_z),   pi_z   = gpuArray(pi_z);   end
end

%% Semi-exogenous shock gridvals and pi
if vfoptions.alreadygridvals_semiexo==0
    if isfield(vfoptions, 'n_semiz') && prod(vfoptions.n_semiz)>0
        % Internally, only ever use age-dependent joint-grids
        vfoptions = SemiExogShockSetup_FHorz(n_d, N_j, d_grid, Parameters, vfoptions, 3);
    end
end

N_d = prod(n_d);
N_a = prod(n_a);
N_z = prod(n_z);
N_z_safe = max(1, N_z);

%% Exogenous shock gridvals and pi
if N_z > 0
    if vfoptions.alreadygridvals == 0
        % ExogShockSetup_FHorz is called with KeepOriginalGrid==0 here
        [z_gridvals_J, pi_z_J, vfoptions] = ExogShockSetup_FHorz(n_z, z_grid, pi_z, N_j, Parameters, vfoptions, 3, 0);
    else
        z_gridvals_J = z_grid;
        pi_z_J = pi_z;
    end
else
    z_gridvals_J = [];
    pi_z_J = [];
end

%% Experience Asset Dispatch
if isfield(vfoptions, 'experienceasset') && vfoptions.experienceasset > 0
    l_a2 = vfoptions.experienceasset; % Supports l_a2 >= 1
    l_d2 = 1; % Toolkit default: last decision variable drives experience asset

    % Split Decision Grids
    if length(n_d) > l_d2
        n_d1 = n_d(1:end-l_d2);
        d1_grid = d_grid(1:sum(n_d1));
    else
        n_d1 = 0;
        d1_grid = [];
    end
    n_d2 = n_d(end-l_d2+1:end);
    d2_grid = d_grid(sum(n_d1)+1:end);

    d1_gridvals = CreateGridvals(n_d1, d1_grid, 1);
    d2_gridvals = CreateGridvals(n_d2, d2_grid, 1);

    % Split Asset Grids
    if length(n_a) > l_a2
        n_a1 = n_a(1:end-l_a2);
        a1_grid = a_grid(1:sum(n_a1));
        a1_gridvals = CreateGridvals(n_a1, a1_grid, 1);
    else
        n_a1 = 0;
        a1_grid = [];
        a1_gridvals = [];
    end
    n_a2 = n_a(end-l_a2+1:end);
    a2_grid = a_grid(sum(n_a1)+1:end);

    % Dispatch to the vectorized ExpAsset handler and bail out of Case1
    [V, Policy] = ValueFnIter_VFHorz_ExpAsset(n_d1, n_d2, n_a1, n_a2, n_z, N_j, ...
        d1_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, ...
        pi_z_J, ReturnFn, Parameters, ...
        DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

    varargout = {V, Policy};
    return
end

%% Semi-exogenous state Dispatch
% The transition matrix of the exogenous shocks depends on the value of the 'last' decision variable(s).
if isfield(vfoptions, 'n_semiz') && prod(vfoptions.n_semiz)>0
    if length(n_d) > vfoptions.l_dsemiz
        n_d1 = n_d(1:end-vfoptions.l_dsemiz);
        d1_grid = d_grid(1:sum(n_d1));
    else
        n_d1 = 0; 
        d1_grid = [];
    end
    n_d2 = n_d(end-vfoptions.l_dsemiz+1:end); % n_d2 influences transition probs
    d2_grid = d_grid(sum(n_d1)+1:end);

    d1_gridvals = CreateGridvals(n_d1, d1_grid, 1);
    d2_gridvals = CreateGridvals(n_d2, d2_grid, 1);

    % Dispatch to the vectorized SemiExo handler and bail out of Case1
    [V, Policy] = ValueFnIter_VFHorz_SemiExo(n_d1, n_d2, n_a, vfoptions.n_semiz, n_z, N_j, ...
        d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, vfoptions.semiz_gridvals_J, ...
        pi_z_J, vfoptions.pi_semiz_J, ReturnFn, Parameters, ...
        DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

    varargout = {V, Policy};
    return
end

% Standardize missing dimensions to length-1 singletons
if isempty(d_grid) || N_d == 0
    n_d_vars = 0;
    d_work = zeros(1, 1, 'like', a_grid);
    n_d_work = 1;
else
    n_d_vars = length(n_d);
    d_work = d_grid;
    n_d_work = N_d;
end
has_d = (n_d_work > 0 && n_d(1) > 0);

% Set up D_cells once outside the reverse_j loop
if has_d
    num_d = length(n_d);
    if num_d > 1
        % 1. Extract 1D grid vectors from the stacked d_grid
        d_grids_1d = cell(1, num_d);
        offset = 0;
        for i_d = 1:num_d
            d_grids_1d{i_d} = d_grid((offset + 1):(offset + n_d(i_d)));
            offset = offset + n_d(i_d);
        end
        
        % 2. Form Cartesian coordinates matching the Kron order: [N_d x num_d]
        [D_mesh{1:num_d}] = ndgrid(d_grids_1d{:});
        
        % 3. Pack into cell array, each variable spanning Dim 4: [1, 1, 1, N_d]
        D_cells = cell(1, num_d);
        for i_d = 1:num_d
            D_cells{i_d} = shiftdim(D_mesh{i_d}(:), -3);
        end
    else
        D_cells = { shiftdim(d_work(:), -3) };
    end
else
    D_cells = {};
end

% ---------------------------------------------------------------------
% Unstack Endogenous States (a, n1, n2, ...)
% ---------------------------------------------------------------------
num_a = length(n_a);
if num_a > 1
    a_grids_1d = cell(1, num_a);
    offset = 0;
    for i_a = 1:num_a
        a_grids_1d{i_a} = a_grid((offset + 1):(offset + n_a(i_a)));
        offset = offset + n_a(i_a);
    end
    [A_mesh_raw{1:num_a}] = ndgrid(a_grids_1d{:});
    A_mat = zeros(N_a, num_a, 'like', a_grid);
    for i_a = 1:num_a
        A_mat(:, i_a) = A_mesh_raw{i_a}(:);
    end
else
    A_mat = a_grid(:);
end
a_work = A_mat(:, 1); % Primary asset grid for interpolation
n_a_work = N_a;

if isempty(z_gridvals_J) || N_z == 0
    N_z_exog = 0;
    z_work_1 = zeros(1, 1, 'like', a_grid);
else
    N_z_exog = N_z;
    z_work_1 = squeeze(z_gridvals_J(:, :, 1));
end

has_semiz = isfield(vfoptions, 'n_semiz') && ~isempty(vfoptions.n_semiz) && prod(vfoptions.n_semiz) > 0;
if has_semiz
    N_semiz = prod(vfoptions.n_semiz);
    n_all_z = [vfoptions.n_semiz, n_z];
else
    N_semiz = 1;
    n_all_z = n_z;
end
n_z_work = N_semiz * max(1, N_z_exog);
has_z = (N_z_exog > 0);

has_e = isfield(vfoptions, 'n_e') && ~isempty(vfoptions.n_e) && prod(vfoptions.n_e) > 0;
if has_e
    n_e_vars = length(vfoptions.n_e);
    n_e_work = prod(vfoptions.n_e);
    if n_e_vars > 1
        e_grids_1d = cell(1, n_e_vars);
        offset = 0;
        for i_e = 1:n_e_vars
            e_grids_1d{i_e} = vfoptions.e_grid((offset + 1):(offset + vfoptions.n_e(i_e)));
            offset = offset + vfoptions.n_e(i_e);
        end
        [E_mesh_raw{1:n_e_vars}] = ndgrid(e_grids_1d{:});
        e_work = zeros(n_e_work, n_e_vars, 'like', a_grid);
        for i_e = 1:n_e_vars
            e_work(:, i_e) = E_mesh_raw{i_e}(:);
        end
    else
        e_work = vfoptions.e_grid(:);
    end
else
    n_e_vars = 0;
    n_e_work = 1;
    e_work   = gpuArray(0); % dummy scalar keeping rank/signatures consistent
end

V = zeros(n_a_work, n_z_work, n_e_work, N_j, 'like', a_grid);
if vfoptions.gridinterplayer == 1
    PolicyKron = zeros(3, n_a_work, n_z_work, n_e_work, N_j, 'like', a_grid);
else
    PolicyKron = zeros(n_a_work, n_z_work, n_e_work, N_j, 'like', a_grid);
end
V_next = zeros(n_a_work, n_z_work, n_e_work, 'like', a_grid);

% --- Grid Interpolation Setup ---
if vfoptions.gridinterplayer == 1
    n2short = vfoptions.ngridinterp;
    n2long  = n2short * 2 + 3;
    % Fix: Use a_work instead of a_gridvals(:,1)
    a1prime_grid = interp1(1:1:N_a, a_work, linspace(1, N_a, N_a + (N_a - 1) * n2short))';
else
    n2short = 0;
    n2long  = 0;
    a1prime_grid = [];
end

for reverse_j = 0:N_j-1
    jj = N_j - reverse_j;
    
    if jj == N_j && isfield(vfoptions, 'V_Jplus1') && ~isempty(vfoptions.V_Jplus1)
        V_next = reshape(gpuArray(vfoptions.V_Jplus1), [N_a, N_z_safe]);
    end
    
    DiscountFactorParamsVec = CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj);
    beta_j = prod(DiscountFactorParamsVec);
    ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, jj);
    if ~iscell(ReturnFnParamsVec); ReturnFnParamsVec = num2cell(ReturnFnParamsVec); end
    
    if N_z > 0
        pi_z_j = pi_z_J(:, :, min(jj, size(pi_z_J, 3)));
        if n_e_work > 1
            % Handle 3D matrix multiplication for multi-e
            EV = zeros(N_a, N_z_safe, n_e_work, 'like', V_next);
            for ie = 1:n_e_work
                EV(:,:,ie) = V_next(:,:,ie) * pi_z_j';
            end
        else
            EV = V_next * pi_z_j';
        end
    else
        EV = V_next;
    end

    % --- The ZE Flattening Trick ---
    N_ze = N_z_safe * n_e_work;
    EV_flat_ze = reshape(EV, [N_a, N_ze]);

    [Z_mesh, E_mesh] = ndgrid(1:N_z_safe, 1:n_e_work);
    ZE_z_idx = Z_mesh(:);
    ZE_e_idx = E_mesh(:);

% --- Determine Exogenous Memory Chunking (lowmemory) ---
    lowmem_level = 0;
    if isfield(vfoptions, 'lowmemory') && ~isempty(vfoptions.lowmemory)
        lowmem_level = vfoptions.lowmemory;
    end

    if lowmem_level == 0
        % Vectorize everything
        ze_chunks = {1:N_ze};
    elseif lowmem_level == 1
        if N_z_safe > 1 && n_e_work > 1
            % z & e present: Parallel over z, loop over e
            ze_chunks = cell(1, n_e_work);
            for ie = 1:n_e_work
                ze_chunks{ie} = (ie - 1) * N_z_safe + 1 : ie * N_z_safe;
            end
        else
            % Only z or only e present: loop over that active shock
            ze_chunks = num2cell(1:N_ze);
        end
    elseif lowmem_level == 2
        % z & e present: loop both (evaluate one ZE combination at a time)
        ze_chunks = num2cell(1:N_ze);
    else
        error('Invalid lowmemory level requested for the current shock combination.');
    end

    % Define the unified block engine (Accepts ze_idx)
    EvalBlockFn = @(state_idx, ze_idx, loweredge_matrix, maxgap_scalar) Evaluate_Case1_TensorBlock(...
        state_idx, ze_idx, loweredge_matrix, maxgap_scalar, N_a, N_d, ...
        has_z, has_e, ZE_z_idx, ZE_e_idx, e_work, ...
        vfoptions.gridinterplayer, n2short, n2long, beta_j, EV_flat_ze, a_gridvals, a1prime_grid, ...
        z_gridvals_J(:,:,min(jj, size(z_gridvals_J,3))), D_cells, ReturnFn, ReturnFnParamsVec);

    % Preallocate output tensors
    V_j_max        = zeros(N_a, N_ze, 'like', EV_flat_ze);
    Pol_apr_max    = zeros(N_a, N_ze, 'like', EV_flat_ze);
    Pol_d_max      = zeros(N_a, N_ze, 'like', EV_flat_ze);
    Pol_L2idx_max  = zeros(N_a, N_ze, 'like', EV_flat_ze);
    Pol_L2flag_max = zeros(N_a, N_ze, 'like', EV_flat_ze);

    % --- The Master Orchestrator Loop ---
    for i_ze = 1:length(ze_chunks)
        curr_ze = ze_chunks{i_ze};
        
        % Create a localized closure for the Slicer so it only sees the current ZE slice
        LocalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar) EvalBlockFn(state_idx, curr_ze, loweredge_matrix, maxgap_scalar);
        
        if isfield(vfoptions, 'divideandconquer') && vfoptions.divideandconquer == 1
            vfoptions.level1n = vfoptions.level1n(1); 
            [v, p_apr, p_d, p_l2idx, p_l2flag] = ValueFnIter_DC1_Slicer(N_a, N_a, 1, length(curr_ze), vfoptions, LocalBlockFn);
        else
            % Brute Force over 'a' (no fake endogenous looping here!)
            [v, p_apr, p_d, p_l2idx, p_l2flag] = LocalBlockFn(1:N_a, [], 0);
        end
        
        % Slot results directly into the preallocated flat tensors
        V_j_max(:, curr_ze)     = reshape(v,     [N_a, length(curr_ze)]);
        Pol_apr_max(:, curr_ze) = reshape(p_apr, [N_a, length(curr_ze)]);
        Pol_d_max(:, curr_ze)   = reshape(p_d,   [N_a, length(curr_ze)]);
        if vfoptions.gridinterplayer == 1
            Pol_L2idx_max(:, curr_ze)  = reshape(p_l2idx,  [N_a, length(curr_ze)]);
            Pol_L2flag_max(:, curr_ze) = reshape(p_l2flag, [N_a, length(curr_ze)]);
        end
    end
    
    % Squeeze Outputs back to full 3D [N_a, N_z, N_e] structure
    V_j_max     = reshape(V_j_max,     [N_a, N_z_safe, n_e_work]);
    Pol_apr_max = reshape(Pol_apr_max, [N_a, N_z_safe, n_e_work]);
    Pol_d_max   = reshape(Pol_d_max,   [N_a, N_z_safe, n_e_work]);
    if vfoptions.gridinterplayer == 1
        Pol_L2idx_max  = reshape(Pol_L2idx_max,  [N_a, N_z_safe, n_e_work]);
        Pol_L2flag_max = reshape(Pol_L2flag_max, [N_a, N_z_safe, n_e_work]);
    end
    
    V(:, :, :, jj) = V_j_max;
    V_next = V_j_max;

end

if N_z == 0
    V = squeeze(V);
end

if N_d == 0
    n_daprime = n_a;
else
    n_daprime = [n_d, n_a];
end

if vfoptions.gridinterplayer == 0
    PolicyKron = shiftdim(PolicyKron, -1);
end

if isfield(vfoptions, 'outputkron') && vfoptions.outputkron == 1
    varargout{1} = V;
    varargout{2} = PolicyKron;
    return
end

if has_z && has_e
    Policy = UnKronPolicyIndexes1_FHorz_z_e(PolicyKron, n_daprime, n_a, n_all_z, n_e_work, N_j, vfoptions);
    % Flatten compound Z and E dimensions for downstream toolkit compatibility
    Policy = reshape(Policy, [size(Policy, 1), n_a_work, n_z_work, n_e_work, N_j]);
    V = reshape(V, [n_a_work, n_z_work, n_e_work, N_j]);
elseif has_z && ~has_e
    Policy = UnKronPolicyIndexes1_FHorz_z(PolicyKron, n_daprime, n_a, n_all_z, N_j, vfoptions);
    Policy = reshape(Policy, [size(Policy, 1), n_a_work, n_z_work, N_j]);
    V = reshape(V, [n_a_work, n_z_work, N_j]);
elseif ~has_z && has_e
    Policy = UnKronPolicyIndexes1_FHorz_e(PolicyKron, n_daprime, n_a, n_e_work, N_j, vfoptions);
    Policy = reshape(Policy, [size(Policy, 1), n_a_work, n_e_work, N_j]);
    V = reshape(V, [n_a_work, n_e_work, N_j]);
else
    Policy = UnKronPolicyIndexes1_FHorz_noz(PolicyKron, n_daprime, n_a, N_j, vfoptions);
    Policy = reshape(Policy, [size(Policy, 1), n_a_work, N_j]);
    V = reshape(V, [n_a_work, N_j]);
end

varargout{1} = V;
varargout{2} = Policy;


end

function [V_j_max, Pol_apr_max, Pol_d_max, Pol_L2idx_max, Pol_L2flag_max] = Evaluate_Case1_TensorBlock(...
    state_idx, loweredge_matrix, maxgap_scalar, N_a, N_ze, N_d, ...
    has_z, has_e, ZE_z_idx, ZE_e_idx, e_work, ...
    gridinterplayer, n2short, n2long, beta_j, EV, a_gridvals, a1prime_grid, ...
    z_gridvals_j, D_cells, ReturnFn, ReturnFnParamsVec)

N_block = length(state_idx);
N_d_safe = max(1, N_d);

% --- 1. Choice Grid Setup (Implicit Dimensions) ---
if isempty(loweredge_matrix)
    N_choice = N_a;
    apr_idx_tensor = reshape(1:N_a, [1, N_choice, 1, 1]);
else
    N_choice = maxgap_scalar + 1;
    offset = reshape(0:maxgap_scalar, [1, N_choice, 1, 1]);
    base_edge = reshape(loweredge_matrix, [1, 1, 1, N_ze]);
    apr_idx_tensor = base_edge + offset; % Size: [1, N_choice, 1, N_ze]
end

% --- 2. State & Choice Tensor Construction (Zero Repmats) ---
a_work_local = a_gridvals(:, 1);

% Force Dim 1 to be singleton [1, N_choice, 1, ...]
if isempty(loweredge_matrix)
    apr_in = reshape(a_work_local(apr_idx_tensor(:)), [1, N_choice, 1, 1]);
else
    apr_in = reshape(a_work_local(apr_idx_tensor(:)), [1, N_choice, 1, N_ze]);
end

a_in = reshape(a_work_local(state_idx), [1, 1, N_block, 1]);

if N_d > 0
    D_cells_block = cell(size(D_cells));
    for id = 1:length(D_cells)
        D_cells_block{id} = reshape(D_cells{id}, [N_d_safe, 1, 1, 1]);
    end
else
    D_cells_block = {};
end

if has_z
    num_z_vars = size(z_gridvals_j, 2);
    Z_cells_block = cell(1, num_z_vars);
    for iz = 1:num_z_vars
        Z_cells_block{iz} = reshape(z_gridvals_j(ZE_z_idx, iz), [1, 1, 1, N_ze]);
    end
else
    Z_cells_block = {};
end

if has_e
    num_e_vars = size(e_work, 2);
    E_cells_block = cell(1, num_e_vars);
    for ie = 1:num_e_vars
        E_cells_block{ie} = reshape(e_work(ZE_e_idx, ie), [1, 1, 1, N_ze]);
    end
else
    E_cells_block = {};
end

% --- 3. Evaluate Return Function & Coarse RHS ---
F_tensor = ReturnFn(D_cells_block{:}, apr_in, a_in, Z_cells_block{:}, E_cells_block{:}, ReturnFnParamsVec{:});

EV_flat = reshape(EV, [N_a * N_ze, 1]);
z_offset = reshape((0:N_ze-1) * N_a, [1, 1, 1, N_ze]);
linear_idx = apr_idx_tensor + z_offset; % Size: [1, N_choice, 1, N_ze]
EV_bounded = reshape(EV_flat(linear_idx(:)), [1, N_choice, 1, N_ze]);

RHS = F_tensor + beta_j .* EV_bounded;

% Zero-overhead guard to ensure implicit expansion reached full 4D shape
expected_sz = [N_d_safe, N_choice, N_block, N_ze];
if ~isequal(size(RHS), expected_sz)
    RHS = RHS + zeros(expected_sz, 'like', EV);
end

RHS_flat = reshape(RHS, [N_d_safe * N_choice, N_block * N_ze]);
[V_sub_coarse, Pol_sub_idx] = max(RHS_flat, [], 1);

d_idx_local   = mod(Pol_sub_idx - 1, N_d_safe) + 1;
apr_idx_local = ceil(Pol_sub_idx / N_d_safe);

if isempty(loweredge_matrix)
    apr_idx_coarse = apr_idx_local;
else
    loweredge_2d = repmat(reshape(loweredge_matrix, [1, N_ze]), [N_block, 1]);
    apr_idx_local_2d = reshape(apr_idx_local, [N_block, N_ze]);
    apr_idx_coarse = loweredge_2d + apr_idx_local_2d - 1;
end

apr_idx_coarse = reshape(apr_idx_coarse, [N_block, N_ze]);
d_idx_coarse   = reshape(d_idx_local, [N_block, N_ze]);

% --- 4. The Continuous Sub-Grid Refinement (GI1) ---
if gridinterplayer
    midpoint = max(min(apr_idx_coarse, N_a - 1), 2);
    base_idx = midpoint + (midpoint - 1) * n2short;
    offset   = (-n2short-1 : 1 : n2short+1)';
    fine_idx = base_idx(:)' + offset; 

    % Explicitly guarantee [1, n2long, N_block, N_ze]
    fine_idx_4d = reshape(fine_idx, [1, n2long, N_block, N_ze]);
    apr_in_fine = reshape(a1prime_grid(fine_idx(:)), [1, n2long, N_block, N_ze]);
    
    % We reuse the implicitly sized a_in, D_cells, Z_cells, and E_cells directly!
    F_tensor_fine = ReturnFn(D_cells_block{:}, apr_in_fine, a_in, Z_cells_block{:}, E_cells_block{:}, ReturnFnParamsVec{:});

    EV_interp = interp1(a_work_local, EV, a1prime_grid);
    z_offset_fine = reshape((0:N_ze-1) * length(a1prime_grid), [1, 1, 1, N_ze]);
    linear_fine_idx = fine_idx_4d + z_offset_fine; % Size: [1, n2long, N_block, N_ze]
    EV_fine = reshape(EV_interp(linear_fine_idx(:)), [1, n2long, N_block, N_ze]);

    RHS_fine = F_tensor_fine + beta_j .* EV_fine;
    
    expected_sz_fine = [N_d_safe, n2long, N_block, N_ze];
    if ~isequal(size(RHS_fine), expected_sz_fine)
        RHS_fine = RHS_fine + zeros(expected_sz_fine, 'like', EV);
    end
    
    RHS_fine_flat = reshape(RHS_fine, [N_d_safe * n2long, N_block * N_ze]);
    [V_sub_fine, maxindexL2] = max(RHS_fine_flat, [], 1);

    d_idx_fine    = mod(maxindexL2 - 1, N_d_safe) + 1;
    apr_step_fine = ceil(maxindexL2 / N_d_safe);

    isInfLower    = (RHS_fine_flat(1:N_d_safe, :) == -Inf);
    isInfUpper    = (RHS_fine_flat(end-N_d_safe+1:end, :) == -Inf);
    
    inLowerStrict = (apr_step_fine >= 2) & (apr_step_fine <= n2short + 1);
    inUpperStrict = (apr_step_fine >= n2short + 3) & (apr_step_fine <= n2long - 1);
    
    linear_win_d = d_idx_fine + (0:N_block*N_ze-1)*N_d_safe;
    L2flag_fine = 2 + (inLowerStrict & isInfLower(linear_win_d)) - (inUpperStrict & isInfUpper(linear_win_d));

    V_j_max        = reshape(V_sub_fine,    [N_block, N_ze]);
    Pol_apr_max    = reshape(midpoint,      [N_block, N_ze]);
    Pol_d_max      = reshape(d_idx_fine,    [N_block, N_ze]);
    Pol_L2idx_max  = reshape(apr_step_fine, [N_block, N_ze]);
    Pol_L2flag_max = reshape(L2flag_fine,   [N_block, N_ze]);
else
    V_j_max        = reshape(V_sub_coarse,   [N_block, N_ze]);
    Pol_apr_max    = reshape(apr_idx_coarse, [N_block, N_ze]);
    Pol_d_max      = d_idx_coarse;
    Pol_L2idx_max  = []; 
    Pol_L2flag_max = [];
end


end