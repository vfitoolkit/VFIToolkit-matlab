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
    e_work   = shiftdim(gpuArray(vfoptions.e_grid),-2);
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
        EV = V_next * pi_z_j';
    else
        EV = V_next;
    end
    
    % Define the Unified GPU Tensor Engine for Case1
    EvalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar) Evaluate_Case1_TensorBlock(...
        state_idx, loweredge_matrix, maxgap_scalar, N_a, N_z_safe, ...
        gridinterplayer, n2short, n2long, beta_j, EV, a_gridvals, a1prime_grid, ...
        z_gridvals_J(:,:,min(jj, size(z_gridvals_J,3))), ReturnFn, ReturnFnParamsVec);

    % The Time-Loop Router
    if isfield(vfoptions, 'divideandconquer') && vfoptions.divideandconquer == 1
        % level1n is handled by the toolkit's default logic (e.g., floor(sqrt(N_a)))
        vfoptions.level1n = vfoptions.level1n(1); % Ensure scalar for 1D DC1
        
        [V_j_max, Pol_apr_max, Pol_d1_dummy, Pol_L2idx_max, Pol_L2flag_max] = ...
            ValueFnIter_DC1_Slicer(N_a, N_a, 1, N_z_safe, vfoptions, EvalBlockFn);
    else
        % Brute Force
        [V_j_max, Pol_apr_max, Pol_L2idx_max, Pol_L2flag_max] = EvalBlockFn(1:N_a, [], 0);
    end
    
    % ... (Proceed to gridinterplayer PolicyKron packing as normal, using Pol_apr_max) ...
    V(:, :, jj) = V_j_max;
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

function [V_j_max, Pol_apr_max, Pol_L2idx_max, Pol_L2flag_max] = Evaluate_Case1_TensorBlock(...
    state_idx, loweredge_matrix, maxgap_scalar, N_a, N_z_safe, ...
    gridinterplayer, n2short, n2long, beta_j, EV, a_gridvals, a1prime_grid, ...
    z_gridvals_j, ReturnFn, ReturnFnParamsVec)

N_block = length(state_idx);

% --- 1. Choice Grid Setup (The Ragged Edge Handler) ---
if isempty(loweredge_matrix)
    % BRUTE FORCE: Evaluate all choices
    N_choice = N_a;
    apr_idx_tensor = repmat((1:N_a)', [1, N_block, N_z_safe]); 
else
    % DC1 SLICER: Evaluate only the ragged bounds passed by the CPU
    N_choice = maxgap_scalar + 1;
    % loweredge_matrix is [N_block, 1, N_z_safe]. Expand it into the bounded window:
    offset = reshape(0:maxgap_scalar, [N_choice, 1, 1]);
    apr_idx_tensor = repmat(shiftdim(loweredge_matrix, -1), [N_choice, 1, 1]) + repmat(offset, [1, N_block, N_z_safe]);
end

% --- 2. State & Choice Tensor Construction ---
% States
a_in = repmat(a_gridvals(state_idx, 1)', [N_choice, 1, N_z_safe]);
if N_z_safe > 1
    z_in = repmat(reshape(z_gridvals_j(:,1), [1, 1, N_z_safe]), [N_choice, N_block, 1]);
    Z_cells = {z_in};
else
    Z_cells = {};
end

% Choices (Map indices to exact grid values)
apr_in = a_gridvals(apr_idx_tensor, 1);

% --- 3. Evaluate Return Function & Coarse RHS ---
F_tensor = ReturnFn(apr_in, a_in, Z_cells{:}, ReturnFnParamsVec{:});

% Map the ragged choice indices to extract specific EV bounds
EV_flat = reshape(EV, [N_a * N_z_safe, 1]);
z_offset = repmat(reshape((0:N_z_safe-1) * N_a, [1, 1, N_z_safe]), [N_choice, N_block, 1]);
EV_bounded = reshape(EV_flat(apr_idx_tensor + z_offset), [N_choice, N_block, N_z_safe]);

RHS = F_tensor + beta_j .* EV_bounded;

[V_sub_coarse, Pol_sub_idx] = max(RHS, [], 1);

% Map the local chunk index back to the global coarse a' index
if isempty(loweredge_matrix)
    apr_idx_coarse = Pol_sub_idx;
else
    % Extract the exact global index that won from our ragged tracker
    linear_win_idx = Pol_sub_idx + (0:N_block*N_z_safe-1)*N_choice;
    apr_idx_coarse = reshape(apr_idx_tensor(linear_win_idx), [1, N_block, N_z_safe]);
end

% --- 4. The Continuous Sub-Grid Refinement (GI1) ---
if gridinterplayer
    apr_idx_coarse_flat = reshape(apr_idx_coarse, [N_block, N_z_safe]);
    midpoint = max(min(apr_idx_coarse_flat, N_a - 1), 2);

    base_idx = midpoint + (midpoint - 1) * n2short;
    offset   = (-n2short-1 : 1 : n2short+1)';
    fine_idx = base_idx(:)' + offset; % [n2long, N_block * N_z_safe]

    apr_in_fine = a1prime_grid(fine_idx);
    a_in_fine   = repmat(a_gridvals(state_idx, 1)', [n2long, N_z_safe]);
    if N_z_safe > 1
        z_in_fine = repmat(reshape(z_gridvals_j(:,1), [1, N_z_safe]), [n2long, N_block]);
        Z_fine = {z_in_fine};
    else
        Z_fine = {};
    end

    F_tensor_fine = ReturnFn(apr_in_fine, a_in_fine, Z_fine{:}, ReturnFnParamsVec{:});

    % Interpolate EV globally, then slice the micro-grid
    EV_interp = interp1((1:N_a)', EV, a1prime_grid);
    z_offset_fine = repmat(reshape((0:N_z_safe-1) * length(a1prime_grid), [1, N_z_safe]), [n2long, N_block]);
    EV_fine = EV_interp(fine_idx + z_offset_fine);

    RHS_fine = F_tensor_fine + beta_j .* EV_fine;
    [V_sub_fine, maxindexL2] = max(RHS_fine, [], 1);

    isInfLower    = (RHS_fine(1, :) == -Inf);
    isInfUpper    = (RHS_fine(end, :) == -Inf);
    inLowerStrict = (maxindexL2 >= 2) & (maxindexL2 <= n2short + 1);
    inUpperStrict = (maxindexL2 >= n2short + 3) & (maxindexL2 <= n2long - 1);
    L2flag_fine   = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    V_j_max        = reshape(V_sub_fine,  [N_block, N_z_safe]);
    Pol_apr_max    = reshape(midpoint,    [N_block, N_z_safe]);
    Pol_L2idx_max  = reshape(maxindexL2,  [N_block, N_z_safe]);
    Pol_L2flag_max = reshape(L2flag_fine, [N_block, N_z_safe]);
else
    V_j_max        = reshape(V_sub_coarse,   [N_block, N_z_safe]);
    Pol_apr_max    = reshape(apr_idx_coarse, [N_block, N_z_safe]);
    Pol_L2idx_max  = []; 
    Pol_L2flag_max = [];
end


end