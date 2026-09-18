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
if vfoptions.alreadygridvals_semiexo==0
    if prod(vfoptions.n_semiz)>0
        % Default AgeDependence=1 (Bypassed if master script passes alreadygridvals_semiexo=1)
        vfoptions = SemiExogShockSetup_FHorz(n_d, N_j, d_grid, Parameters, vfoptions, 1);
    end
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
        z_J = repmat(z_grid, [1, 1, num_periods]); % <-- Flattening removed
        num_z_vars = size(z_grid, 2);
    else
        z_J = [];
        num_z_vars = 0;
    end
    
    % 3. Combine them via Kronecker expansion for each period
    z_gridvals_J = zeros(N_semiz * max(1, N_z), num_semiz_vars + num_z_vars, num_periods, 'like', sz_J);
    for t = 1:num_periods
        if N_z > 0
            % Repeat semiz for every z
            semiz_expanded = kron(sz_J(:,:,t), ones(N_z, 1));
            % Repeat z for every semiz
            z_expanded = kron(ones(N_semiz, 1), z_J(:,:,t));
            z_gridvals_J(:,:,t) = [semiz_expanded, z_expanded];
        else
            z_gridvals_J(:,:,t) = sz_J(:,:,t);
        end
    end

    % 4. Combine (but not densely) the transition matrices
    % Pass raw pi_z, QHEZ will decouple and apply it sequentially.
    pi_z_J = pi_z;
    n_combined_z = [vfoptions.n_semiz, n_z];
else
    % Fallback to standard z
    z_gridvals_J = z_grid;
    pi_z_J = pi_z;
    n_combined_z = n_z;
end

%% Quasi-Hyperbolic dispatcher (no divide-and-conquer)
if isfield(vfoptions, 'exoticpreferences')
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
end

% ---------------------------------------------------------------------
% UNIVERSAL PACKER: Unstack Endogenous, Decision, and Exogenous Grids
% ---------------------------------------------------------------------
has_e = isfield(vfoptions, 'n_e') && prod(vfoptions.n_e) > 0;
n_e_pass = 0; e_grid_pass = [];
if has_e
    n_e_pass = vfoptions.n_e;
    e_grid_pass = vfoptions.e_grid;
end

z_pass = [];
if N_z > 0
    % Pass period 1 for initial sizing; dynamic time-varying Z is handled in the reverse_j loop
    z_pass = z_gridvals_J(:,:,1);
end

% ONE CALL TO RULE THEM ALL
[TensorReturnFn, D_cells, A_cells, Z_cells, E_cells] = CreateTensorFnAndCells(ReturnFn, n_d, n_a, n_z, n_e_pass, d_grid, a_grid, z_pass, e_grid_pass);
TensoraprimeFn = CreateTensorBridge(vfoptions.aprimeFn);

% --- Standardize Dimensions for the Slicer & Allocator ---
N_d_safe = max(1, prod(n_d));
has_d = (N_d_safe > 1) || (length(n_d) > 0 && n_d(1) > 0);
n_a_work = prod(n_a);
a_work = A_cells{1}(:); % Extract primary asset grid for interpolation

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
has_z = (N_z_exog > 0);
n_z_work = N_semiz * N_z_exog;

n_e_work = max(1, prod(n_e_pass));

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

    DiscountFactorParamsVec = CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj, vfoptions.precision);
    beta_j = prod(DiscountFactorParamsVec);
    ReturnFnParamsCell = CreateCellFromParams(Parameters, ReturnFnParamNames, jj, vfoptions.precision);

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
    N_dsemiz = 1;
    if has_semiz && length(n_d) > 0
        if isfield(vfoptions, 'l_dsemiz')
            N_dsemiz = prod(n_d(end-vfoptions.l_dsemiz+1:end));
        else
            N_dsemiz = n_d(end); % Default to the last decision variable
        end
    end

    EV = zeros(N_a, n_z_work, n_e_work, N_dsemiz, 'like', V_next);

    for ie = 1:n_e_work
        V_curr = V_transformed(:,:,ie);

        % 1. Apply Exogenous Z Transition (if it exists)
        if has_z
            pi_z_j = pi_z_J(:, :, min(jj, size(pi_z_J, 3)));
            V_slice = reshape(V_curr, [N_a * N_semiz, N_z_exog]);
            V_z_eval = V_slice * pi_z_j';
            V_z_eval = reshape(V_z_eval, [N_a, N_semiz, N_z_exog]);
        else
            V_z_eval = reshape(V_curr, [N_a, N_semiz, N_z_exog]);
        end

        % 2. Apply Semi-Exogenous Transition (if it exists)
        if has_semiz
            pi_semiz_j = vfoptions.pi_semiz_J(:, :, :, min(jj, size(vfoptions.pi_semiz_J, 4)));

            % Permute to [N_semiz, N_a * N_z_exog] for matrix multiplication
            V_perm = reshape(permute(V_z_eval, [2, 1, 3]), [N_semiz, N_a * N_z_exog]);

            for idsemiz = 1:N_dsemiz
                pi_semiz_d = pi_semiz_j(:, :, idsemiz);
                EV_perm = pi_semiz_d * V_perm;
                EV_d = permute(reshape(EV_perm, [N_semiz, N_a, N_z_exog]), [2, 1, 3]);
                EV(:,:,ie,idsemiz) = reshape(EV_d, [N_a, n_z_work]);
            end
        else
            EV(:,:,ie,1) = reshape(V_z_eval, [N_a, n_z_work]);
        end
    end

    % --- EZ Certainty Equivalent Reverse Transformation ---
    valid_EV = isfinite(EV) & (EV ~= 0);
    if ezc6(jj) ~= 1
        EV(valid_EV) = max(EV(valid_EV), 0).^ezc6(jj);
    end
    if ezc8(jj) ~= 1
        EV(valid_EV) = max(EV(valid_EV), 0).^ezc8(jj);
    end

    % --- The ZE Flattening Trick ---
    N_ze = n_z_work * n_e_work;
    EV_flat_ze = reshape(EV, [N_a, N_ze, N_dsemiz]);

    [Z_mesh, E_mesh] = ndgrid(1:n_z_work, 1:n_e_work);
    ZE_z_idx = Z_mesh(:);
    ZE_e_idx = E_mesh(:);

    % --- Slicer Setup (Multi-Axis) ---
    if ismember(vfoptions.lowmemory, [0, 4])
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

    % --- The Master Orchestrator Pre-Computation ---
    a_work_local = a_work;
    z_gridvals_j_local = [];
    if n_z_work > 1 && ~isempty(z_gridvals_J)
        z_gridvals_j_local = z_gridvals_J(:,:,min(jj, size(z_gridvals_J, 3)));
    end
    if has_semiz || has_z
        semiz_j = [];
        if has_semiz
            semiz_j = vfoptions.semiz_gridvals_J(:,:,min(jj, size(vfoptions.semiz_gridvals_J, 3)));
        end
        z_j = [];
        if has_z
            z_j = z_gridvals_J(:,:,min(jj, size(z_gridvals_J, 3)));
        end
        if has_semiz && has_z
            [sz_idx, z_idx] = ndgrid(1:size(semiz_j, 1), 1:size(z_j, 1));
            z_gridvals_j_local = [semiz_j(sz_idx(:), :), z_j(z_idx(:), :)];
        elseif has_semiz
            z_gridvals_j_local = semiz_j;
        else
            z_gridvals_j_local = z_j;
        end
    end

    N_d_safe = max(1, N_d);
    if N_d > 0
        D_cells_block = cell(size(D_cells));
        for id = 1:length(D_cells)
            D_cells_block{id} = reshape(D_cells{id}, [N_d_safe, 1, 1, 1]);
        end
    else
        D_cells_block = {};
    end

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

    V_j_max        = zeros(N_a, N_ze, 'like', V_next);
    Pol_apr_max    = zeros(N_a, N_ze, 'like', V_next);
    Pol_d_max      = zeros(N_a, N_ze, 'like', V_next);
    Pol_L2idx_max  = zeros(N_a, N_ze, 'like', V_next);
    Pol_L2flag_max = zeros(N_a, N_ze, 'like', V_next);

    % --- The Master Orchestrator Loop ---
    for i_a2 = 1:length(a2_chunks)
        curr_a2 = a2_chunks{i_a2};
        N_a2_local = length(curr_a2);
        start_a_idx = (min(curr_a2) - 1) * N_a1 + 1;
        end_a_idx   = max(curr_a2) * N_a1;

        for i_ze = 1:length(ze_chunks)
            curr_ze = ze_chunks{i_ze};
            N_ze_local = length(curr_ze);

            if has_semiz || has_z
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

            EV_local = EV_flat_ze(:, curr_ze, :);
            z_offset_local = reshape((0:N_ze_local-1) * N_a, [1, 1, 1, N_ze_local]);

            if vfoptions.gridinterplayer
                EV_interp_local = interp1(a_work_local, reshape(EV_local, [N_a, N_ze_local * N_dsemiz]), a1prime_grid);
                EV_interp_local = reshape(EV_interp_local, [length(a1prime_grid), N_ze_local, N_dsemiz]);
                z_offset_fine_local = reshape((0:N_ze_local-1) * length(a1prime_grid), [1, 1, 1, N_ze_local]);
            else
                EV_interp_local = [];
                z_offset_fine_local = [];
            end

            LocalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar) Evaluate_Case1_TensorBlock(...
                state_idx, loweredge_matrix, maxgap_scalar, N_a, N_d_safe, N_ze_local, ...
                Z_cells_local, E_cells_local, D_cells_block, A_cells, num_a1_pass, ...
                vfoptions.gridinterplayer, n2short, n2long, beta_j, EV_local, EV_interp_local, z_offset_local, z_offset_fine_local, a1prime_grid, ...
                TensorReturnFn, ReturnFnParamsCell, ezc2(jj), ezc3, ezc4, ezc7(jj), ...
                TensoraprimeFn, A_cells{end}(:), N_dsemiz, dsemiz_idx_tensor);

            if vfoptions.divideandconquer == 1
                vfoptions.level1n = vfoptions.level1n(1);
                [v, p_apr, p_d, p_l2idx, p_l2flag] = ValueFnIter_DC1_Slicer(N_a1 * N_a2_local, N_a, 1, N_ze_local, vfoptions, LocalBlockFn);
            else
                [v, p_apr, p_d, p_l2idx, p_l2flag] = LocalBlockFn(start_a_idx:end_a_idx, [], 0);
            end

            V_j_max(start_a_idx:end_a_idx, curr_ze)     = reshape(v,     [N_a1 * N_a2_local, N_ze_local]);
            Pol_apr_max(start_a_idx:end_a_idx, curr_ze) = reshape(p_apr, [N_a1 * N_a2_local, N_ze_local]);
            Pol_d_max(start_a_idx:end_a_idx, curr_ze)   = reshape(p_d,   [N_a1 * N_a2_local, N_ze_local]);
            if vfoptions.gridinterplayer(1) == 1
                Pol_L2idx_max(start_a_idx:end_a_idx, curr_ze)  = reshape(p_l2idx,  [N_a1 * N_a2_local, N_ze_local]);
                Pol_L2flag_max(start_a_idx:end_a_idx, curr_ze) = reshape(p_l2flag, [N_a1 * N_a2_local, N_ze_local]);
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
Policy_flat = zeros([num_pol_vars, n_a_work, n_z_work, n_e_work, N_j], 'single');

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
    state_idx, loweredge_matrix, maxgap_scalar, N_a, N_d_safe, N_ze_local, ...
    Z_cells_block, E_cells_block, D_cells_block, A_cells, num_a1, ...
    gridinterplayer, n2short, n2long, beta_j, EV_local, EV_interp_local, z_offset_local, z_offset_fine_local, a1prime_grid, ...
    TensorReturnFn, ReturnFnParamsCell, ezc2_j, ezc3, ezc4, ezc7_j, Tensora2primeFn, a2_grid_full, N_dsemiz, dsemiz_idx_tensor)

N_block = length(state_idx);

% --- 1. SAFELY Extract 1D Asset Grids & Rebuild Sequences ---
num_assets = length(A_cells);
grid_1D = cell(1, num_assets);

for ia = 1:num_assets
    % unique() safely flattens any shape (implicitly expanded or full)
    % and extracts the pure 1D monotonic grid for this asset
    grid_1D{ia} = unique(A_cells{ia}(:));
end

% Rebuild the exact N_a Cartesian state sequence natively
state_grids = cell(1, num_assets);
if num_assets > 1
    [state_grids{:}] = ndgrid(grid_1D{:});
else
    state_grids{1} = grid_1D{1};
end

asset_seq = cell(1, num_assets);
for ia = 1:num_assets
    asset_seq{ia} = state_grids{ia}(:).'; % 1 x N_a sequence
end

N_choice = 1;
for ia = 1:num_a1
    N_choice = N_choice * length(grid_1D{ia});
end

% --- 2. Build Choice Tensor ---
if isempty(loweredge_matrix)
    apr_idx_tensor = reshape(1:N_choice, [1, N_choice, 1, 1]);
else
    offset_vec = 0:gather(maxgap_scalar);
    N_choice = length(offset_vec);
    offset = reshape(gpuArray(offset_vec), [1, N_choice, 1, 1]);
    base_edge = reshape(loweredge_matrix, [1, 1, 1, N_ze_local]);
    apr_idx_tensor = base_edge + offset;
end

apr_in_coarse = cell(1, num_assets);
a_in_fine     = cell(1, num_assets);

% Natively rebuild the choice grid to guarantee correct Cartesian combinations
if num_a1 > 0
    choice_grids = cell(1, num_a1);
    if num_a1 > 1
        [choice_grids{:}] = ndgrid(grid_1D{1:num_a1});
    else
        choice_grids{1} = grid_1D{1};
    end

    for ia = 1:num_a1
        flat_choice = choice_grids{ia}(:);
        sub_idx = min(max(apr_idx_tensor, 1), numel(flat_choice));
        apr_in_coarse{ia} = reshape(flat_choice(sub_idx), size(apr_idx_tensor));
    end
end

% Safely map the state grid values using the reconstructed sequence
for ia = 1:num_assets
    state_sub = min(max(state_idx, 1), N_a);
    a_in_fine{ia} = reshape(asset_seq{ia}(state_sub), [1, 1, N_block, 1]);
end

F_tensor = TensorReturnFn(D_cells_block{:}, apr_in_coarse{1:num_a1}, a_in_fine{:}, Z_cells_block{:}, E_cells_block{:}, ReturnFnParamsCell{:});

% --- 3. Evaluate Experience Asset Transition natively ---
a2_grid = grid_1D{end}; % Use the clean 1D grid we just extracted
installpv_tensor = D_cells_block{1}; % D1 is installpv
solarpv_tensor   = a_in_fine{end};

a2_prime_vals = Tensora2primeFn(installpv_tensor, solarpv_tensor, 0, 0, 0, 0);
a2_prime_vals = max(a2_grid(1), min(a2_grid(end), a2_prime_vals));

[~, a2_idx] = histc(a2_prime_vals(:), a2_grid);
a2_idx = max(1, min(a2_idx, length(a2_grid) - 1));
a2_prob = (a2_grid(a2_idx + 1) - a2_prime_vals(:)) ./ (a2_grid(a2_idx + 1) - a2_grid(a2_idx));
a2_prob(isnan(a2_prob)) = 1;

a2_idx_tensor  = reshape(a2_idx, [N_d_safe, 1, N_block, 1]);
a2_prob_tensor = reshape(a2_prob, [N_d_safe, 1, N_block, 1]);

% --- 4. EV Lookup with 4D SemiZ Bounds Clamping ---
EV_flat = reshape(EV_local, [N_a * N_ze_local, N_dsemiz]);
max_idx_row = N_a * N_ze_local;

row_idx_lower = max(1, min(max_idx_row, apr_idx_tensor + (a2_idx_tensor - 1) * N_choice + z_offset_local));
row_idx_upper = max(1, min(max_idx_row, apr_idx_tensor + (a2_idx_tensor) * N_choice + z_offset_local));

linear_idx_lower = row_idx_lower + (dsemiz_idx_tensor - 1) * max_idx_row;
linear_idx_upper = row_idx_upper + (dsemiz_idx_tensor - 1) * max_idx_row;

EV_lower = reshape(EV_flat(linear_idx_lower(:)), size(linear_idx_lower));
EV_upper = reshape(EV_flat(linear_idx_upper(:)), size(linear_idx_upper));
EV_bounded = a2_prob_tensor .* EV_lower + (1 - a2_prob_tensor) .* EV_upper;

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

V_j_max        = reshape(V_sub_coarse,   [N_block, N_ze_local]);
Pol_apr_max    = reshape(apr_idx_coarse, [N_block, N_ze_local]);
Pol_d_max      = d_idx_coarse;
Pol_L2idx_max  = [];
Pol_L2flag_max = [];


end
