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

%% Risky Asset state Dispatch
if isfield(vfoptions, 'riskyasset') && vfoptions.riskyasset == 1

    % 1. Split standard and risky endogenous states (NEW)
    vfoptions = SetupNonStandardEndoStates_FHorz(n_d, n_a, d_grid, a_grid, vfoptions);
    n_a1 = vfoptions.n_a1;
    n_a2 = vfoptions.n_a2;
    a1_grid = vfoptions.a1_grid;
    a2_grid = vfoptions.a2_grid;

    % 2. Extract risky asset variables from vfoptions
    n_u = vfoptions.n_u;
    u_grid = vfoptions.u_grid;
    pi_u = vfoptions.pi_u;
    aprimeFn = vfoptions.aprimeFn;

    % 3. Dynamically extract aprimeFnParamNames
    l_d = length(n_d);
    if isfield(vfoptions, 'refine_d')
        l_d = l_d - vfoptions.refine_d(1);
        % If semiz is active, d4 is part of the decision vector but not in aprimeFn
        if isfield(vfoptions, 'n_semiz') && prod(vfoptions.n_semiz) > 0 && length(vfoptions.refine_d) >= 4
            l_d = l_d - vfoptions.refine_d(4);
        end
    end
    l_u = length(n_u); 
    temp = getAnonymousFnInputNames(aprimeFn);
    if length(temp) > (l_d + l_u)
        aprimeFnParamNames = {temp{l_d + l_u + 1 : end}};
    else
        aprimeFnParamNames = {};
    end

    % 4. Route to the Universal Tensor Architecture (EZ and CRRA unified!)
    if isfield(vfoptions, 'n_semiz') && prod(vfoptions.n_semiz) > 0
        [V, Policy] = ValueFnIter_VFHorz_RiskyAssetSemiExo(n_d, n_a1, n_a2, vfoptions.n_semiz, n_z, n_u, N_j, ...
            d_grid, a1_grid, a2_grid, vfoptions.semiz_gridvals_J, z_gridvals_J, u_grid, ...
            vfoptions.pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, ...
            DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    else
        [V, Policy] = ValueFnIter_VFHorz_RiskyAsset(n_d, n_a1, n_a2, n_z, n_u, N_j, ...
            d_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ...
            ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ...
            ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    end

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

for reverse_j = 0:N_j-1
    jj = N_j - reverse_j;

    if jj == N_j && isfield(vfoptions, 'V_Jplus1') && ~isempty(vfoptions.V_Jplus1)
        V_next = reshape(gpuArray(vfoptions.V_Jplus1), [N_a, N_z_safe]);
    end

    DiscountFactorParamsVec = CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj);
    beta_j = prod(DiscountFactorParamsVec);
    ReturnFnParamsCell = CreateCellFromParams(Parameters, ReturnFnParamNames, jj);

    % --- EZ V_next Transformation ---
    valid_V = isfinite(V_next) & (V_next ~= 0);
    V_transformed = V_next;
    if ezc5(jj) == 1
        V_transformed(valid_V) = ezc4 * V_next(valid_V);
    else
        V_transformed(valid_V) = max(ezc4 * V_next(valid_V), 0).^ezc5(jj);
    end
    V_transformed(V_next == 0) = 0;

    if N_z > 0
        pi_z_j = pi_z_J(:, :, min(jj, size(pi_z_J, 3)));
        if n_e_work > 1 
            EV = zeros(N_a, N_z_safe, n_e_work, 'like', V_next);
            for ie = 1:n_e_work
                EV(:,:,ie) = V_transformed(:,:,ie) * pi_z_j'; % Use V_transformed
            end
        else
            EV = V_transformed * pi_z_j'; % Use V_transformed
        end
    else
        EV = V_transformed; % Use V_transformed
    end

    % --- EZ Certainty Equivalent Reverse Transformation (ezc6 & ezc8) ---
    valid_EV = isfinite(EV) & (EV ~= 0);
    if ezc6(jj) ~= 1
        EV(valid_EV) = max(EV(valid_EV), 0).^ezc6(jj);
    end
    if ezc8(jj) ~= 1
        EV(valid_EV) = max(EV(valid_EV), 0).^ezc8(jj);
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

    % --- The Master Orchestrator Pre-Computation ---
    a_work_local = A_mat(:, 1);
    z_gridvals_j_local = [];
    if has_z
        z_gridvals_j_local = z_gridvals_J(:,:,min(jj, size(z_gridvals_J,3)));
    end

    % Pre-build D_cells_block (Loop Invariant for both Slicer and ZE chunks)
    N_d_safe = max(1, N_d);
    if N_d > 0
        D_cells_block = cell(size(D_cells));
        for id = 1:length(D_cells)
            D_cells_block{id} = reshape(D_cells{id}, [N_d_safe, 1, 1, 1]);
        end
    else
        D_cells_block = {};
    end

    % Preallocate output tensors
    V_j_max        = zeros(N_a, N_ze, 'like', EV_flat_ze);
    Pol_apr_max    = zeros(N_a, N_ze, 'like', EV_flat_ze);
    Pol_d_max      = zeros(N_a, N_ze, 'like', EV_flat_ze);
    Pol_L2idx_max  = zeros(N_a, N_ze, 'like', EV_flat_ze);
    Pol_L2flag_max = zeros(N_a, N_ze, 'like', EV_flat_ze);

    % --- The Master Orchestrator Loop ---
    for i_ze = 1:length(ze_chunks)
        curr_ze = ze_chunks{i_ze};
        N_ze_local = length(curr_ze);

        % 1. Pre-build Exogenous Cells (Loop Invariant for Slicer!)
        if has_z
            num_z_vars = size(z_gridvals_j_local, 2);
            Z_cells_local = cell(1, num_z_vars);
            for iz = 1:num_z_vars
                Z_cells_local{iz} = reshape(z_gridvals_j_local(ZE_z_idx(curr_ze), iz), [1, 1, 1, N_ze_local]);
            end
        else
            Z_cells_local = {};
        end

        if has_e
            num_e_vars = size(e_work, 2);
            E_cells_local = cell(1, num_e_vars);
            for ie = 1:num_e_vars
                E_cells_local{ie} = reshape(e_work(ZE_e_idx(curr_ze), ie), [1, 1, 1, N_ze_local]);
            end
        else
            E_cells_local = {};
        end

        % 2. Pre-build EV dependencies and Interpolations (Loop Invariant for Slicer!)
        EV_local = EV_flat_ze(:, curr_ze);
        z_offset_local = reshape((0:N_ze_local-1) * N_a, [1, 1, 1, N_ze_local]);

        if vfoptions.gridinterplayer
            EV_interp_local = interp1(a_work_local, EV_local, a1prime_grid);
            z_offset_fine_local = reshape((0:N_ze_local-1) * length(a1prime_grid), [1, 1, 1, N_ze_local]);
        else
            EV_interp_local = [];
            z_offset_fine_local = [];
        end

        % Create a localized closure for the Slicer so it only executes pure math
        LocalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar) Evaluate_Case1_TensorBlock(...
            state_idx, loweredge_matrix, maxgap_scalar, N_a, N_d_safe, N_ze_local, ...
            Z_cells_local, E_cells_local, D_cells_block, ...
            vfoptions.gridinterplayer, n2short, n2long, beta_j, EV_local, EV_interp_local, z_offset_local, z_offset_fine_local, a_work_local, a1prime_grid, ...
            ReturnFn, ReturnFnParamsCell, ezc2(jj), ezc3, ezc4, ezc7(jj)); % <--- Added the 4 EZ constants

        if isfield(vfoptions, 'divideandconquer') && vfoptions.divideandconquer == 1
            vfoptions.level1n = vfoptions.level1n(1);
            [v, p_apr, p_d, p_l2idx, p_l2flag] = ValueFnIter_DC1_Slicer(N_a, N_a, 1, N_ze_local, vfoptions, LocalBlockFn);
        else
            [v, p_apr, p_d, p_l2idx, p_l2flag] = LocalBlockFn(1:N_a, [], 0);
        end

        % Slot results directly into the preallocated flat tensors
        V_j_max(:, curr_ze)     = reshape(v,     [N_a, N_ze_local]);
        Pol_apr_max(:, curr_ze) = reshape(p_apr, [N_a, N_ze_local]);
        Pol_d_max(:, curr_ze)   = reshape(p_d,   [N_a, N_ze_local]);
        if vfoptions.gridinterplayer == 1
            Pol_L2idx_max(:, curr_ze)  = reshape(p_l2idx,  [N_a, N_ze_local]);
            Pol_L2flag_max(:, curr_ze) = reshape(p_l2flag, [N_a, N_ze_local]);
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

    % --- Pack PolicyKron ---
    if vfoptions.gridinterplayer == 1
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
    state_idx, loweredge_matrix, maxgap_scalar, N_a, N_d_safe, N_ze_local, ...
    Z_cells_block, E_cells_block, D_cells_block, ...
    gridinterplayer, n2short, n2long, beta_j, EV_local, EV_interp_local, z_offset_local, z_offset_fine_local, a_work_local, a1prime_grid, ...
    ReturnFn, ReturnFnParamsCell, ezc2_j, ezc3, ezc4, ezc7_j)

N_block = length(state_idx);

% --- 1. Choice Grid Setup (Implicit Dimensions) ---
if isempty(loweredge_matrix)
    N_choice = N_a;
    apr_idx_tensor = reshape(1:N_a, [1, N_choice, 1, 1]);
else
    offset_vec = 0:gather(maxgap_scalar);
    N_choice = length(offset_vec);
    offset = reshape(gpuArray(offset_vec), [1, N_choice, 1, 1]);
    base_edge = reshape(loweredge_matrix, [1, 1, 1, N_ze_local]);
    apr_idx_tensor = base_edge + offset;
end

% --- 2. State & Choice Tensor Construction ---
apr_in = reshape(a_work_local(apr_idx_tensor(:)), size(apr_idx_tensor));
a_in = reshape(a_work_local(state_idx), [1, 1, N_block, 1]);

% --- 3. Evaluate Return Function & Coarse RHS ---
F_tensor = ReturnFn(D_cells_block{:}, apr_in, a_in, Z_cells_block{:}, E_cells_block{:}, ReturnFnParamsCell{:});

EV_flat = reshape(EV_local, [N_a * N_ze_local, 1]);
linear_idx = apr_idx_tensor + z_offset_local;
EV_bounded = reshape(EV_flat(linear_idx(:)), size(linear_idx));
    
% (Note: ezc1_j is 1 here, since we are doing standard RHS)
RHS = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bounded, beta_j, 1, ezc2_j, ezc3, ezc4, ezc7_j);

expected_sz = [N_d_safe, N_choice, N_block, N_ze_local];
if ~isequal(size(RHS), expected_sz)
    RHS = RHS + zeros(expected_sz, 'like', EV_local);
end

RHS_flat = reshape(RHS, [N_d_safe * N_choice, N_block * N_ze_local]);
[V_sub_coarse, Pol_sub_idx] = max(RHS_flat, [], 1);

d_idx_local   = mod(Pol_sub_idx - 1, N_d_safe) + 1;
apr_idx_local = ceil(Pol_sub_idx / N_d_safe);

if isempty(loweredge_matrix)
    apr_idx_coarse = apr_idx_local;
else
    loweredge_2d = repmat(reshape(loweredge_matrix, [1, N_ze_local]), [N_block, 1]);
    apr_idx_local_2d = reshape(apr_idx_local, [N_block, N_ze_local]);
    apr_idx_coarse = loweredge_2d + apr_idx_local_2d - 1;
end

apr_idx_coarse = reshape(apr_idx_coarse, [N_block, N_ze_local]);
d_idx_coarse   = reshape(d_idx_local, [N_block, N_ze_local]);

% --- 4. The Continuous Sub-Grid Refinement (GI1) ---
if gridinterplayer
    midpoint = max(min(apr_idx_coarse, N_a - 1), 2);
    base_idx = midpoint + (midpoint - 1) * n2short;
    offset   = (-n2short-1 : 1 : n2short+1)';
    fine_idx = base_idx(:)' + offset;

    fine_idx_4d = reshape(fine_idx, [1, n2long, N_block, N_ze_local]);
    apr_in_fine = reshape(a1prime_grid(fine_idx(:)), [1, n2long, N_block, N_ze_local]);
    a_in_fine   = reshape(a_work_local(state_idx), [1, 1, N_block, 1]);

    F_tensor_fine = ReturnFn(D_cells_block{:}, apr_in_fine, a_in_fine, Z_cells_block{:}, E_cells_block{:}, ReturnFnParamsCell{:});

    EV_flat = reshape(EV_local, [N_a * N_ze_local, 1]);
    linear_idx = apr_idx_tensor + z_offset_local;
    EV_bounded = reshape(EV_flat(linear_idx(:)), size(linear_idx));

    % (Note: ezc1_j is 1 here, since we are doing standard RHS)
    RHS = Evaluate_Universal_RHS_VFHorz(F_tensor_fine, EV_bounded, beta_j, 1, ezc2_j, ezc3, ezc4, ezc7_j);

    expected_sz_fine = [N_d_safe, n2long, N_block, N_ze_local];
    if ~isequal(size(RHS_fine), expected_sz_fine)
        RHS_fine = RHS_fine + zeros(expected_sz_fine, 'like', EV_local);
    end

    RHS_fine_flat = reshape(RHS_fine, [N_d_safe * n2long, N_block * N_ze_local]);
    [V_sub_fine, maxindexL2] = max(RHS_fine_flat, [], 1);

    d_idx_fine    = mod(maxindexL2 - 1, N_d_safe) + 1;
    apr_step_fine = ceil(maxindexL2 / N_d_safe);

    isInfLower    = (RHS_fine_flat(1:N_d_safe, :) == -Inf);
    isInfUpper    = (RHS_fine_flat(end-N_d_safe+1:end, :) == -Inf);

    inLowerStrict = (apr_step_fine >= 2) & (apr_step_fine <= n2short + 1);
    inUpperStrict = (apr_step_fine >= n2short + 3) & (apr_step_fine <= n2long - 1);

    linear_win_d = d_idx_fine + (0:N_block*N_ze_local-1)*N_d_safe;
    L2flag_fine = 2 + (inLowerStrict & isInfLower(linear_win_d)) - (inUpperStrict & isInfUpper(linear_win_d));

    V_j_max        = reshape(V_sub_fine,    [N_block, N_ze_local]);
    Pol_apr_max    = reshape(midpoint,      [N_block, N_ze_local]);
    Pol_d_max      = reshape(d_idx_fine,    [N_block, N_ze_local]);
    Pol_L2idx_max  = reshape(apr_step_fine, [N_block, N_ze_local]);
    Pol_L2flag_max = reshape(L2flag_fine,   [N_block, N_ze_local]);
else
    V_j_max        = reshape(V_sub_coarse,   [N_block, N_ze_local]);
    Pol_apr_max    = reshape(apr_idx_coarse, [N_block, N_ze_local]);
    Pol_d_max      = d_idx_coarse;
    Pol_L2idx_max  = [];
    Pol_L2flag_max = [];
end


end