function varargout=ValueFnIter_Case1_VFHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
%% Check which vfoptions have been used, set all others to defaults
if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    vfoptions.verbose=0;
    vfoptions.divideandconquer=0;
    vfoptions.gridinterplayer=0;
    vfoptions.lowmemory=0;
    vfoptions.incrementaltype=0;
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
    vfoptions.parallel=1+(gpuDeviceCount>0);
    vfoptions.outputkron=0;
    vfoptions.alreadygridvals=0;
    vfoptions.alreadygridvals_semiexo=0;
    vfoptions.precision = underlyingType(a_grid);
else
    if ~isfield(vfoptions,'verbose'); vfoptions.verbose=0; end
    if ~isfield(vfoptions,'divideandconquer'); vfoptions.divideandconquer=0; end
    if ~isfield(vfoptions,'gridinterplayer')
        vfoptions.gridinterplayer=0;
    elseif vfoptions.gridinterplayer(1)==1
        if ~isfield(vfoptions,'ngridinterp')
            error('When using vfoptions.gridinterplayer=1 you must set vfoptions.ngridinterp')
        end
    end
    if ~isfield(vfoptions,'lowmemory'); vfoptions.lowmemory=0; end
    if ~isfield(vfoptions,'incrementaltype'); vfoptions.incrementaltype=0; end
    if ~isfield(vfoptions,'exoticpreferences'); vfoptions.exoticpreferences='None'; end
    if ~isfield(vfoptions,'dynasty'); vfoptions.dynasty=0; end
    if ~isfield(vfoptions,'experienceasset'); vfoptions.experienceasset=0; end
    if ~isfield(vfoptions,'experienceassetu'); vfoptions.experienceassetu=0; end
    if ~isfield(vfoptions,'experienceassete'); vfoptions.experienceassete=0; end
    if ~isfield(vfoptions,'experienceassetz'); vfoptions.experienceassetz=0; end
    if ~isfield(vfoptions,'experienceassetze'); vfoptions.experienceassetze=0; end
    if ~isfield(vfoptions,'experienceassetsemiz'); vfoptions.experienceassetsemiz=0; end
    if ~isfield(vfoptions,'riskyasset'); vfoptions.riskyasset=0; end
    if ~isfield(vfoptions,'residualasset'); vfoptions.residualasset=0; end
    if ~isfield(vfoptions,'n_ambiguity'); vfoptions.n_ambiguity=0; end
    if ~isfield(vfoptions,'n_e'); vfoptions.n_e=0; end
    if ~isfield(vfoptions,'n_semiz'); vfoptions.n_semiz=0; end
    if ~isfield(vfoptions,'parallel'); vfoptions.parallel=1+(gpuDeviceCount>0); end
    if ~isfield(vfoptions,'outputkron'); vfoptions.outputkron=0; end
    if ~isfield(vfoptions,'alreadygridvals'); vfoptions.alreadygridvals=0; end
    if ~isfield(vfoptions,'alreadygridvals_semiexo'); vfoptions.alreadygridvals_semiexo=0; end
    if ~isfield(vfoptions,'precision'); vfoptions.precision = underlyingType(a_grid); end
end

% --- SMART nargin PARSER ---
if isempty(ReturnFnParamNames)
    if isfield(vfoptions, 'ReturnFnParamNames')
        ReturnFnParamNames = vfoptions.ReturnFnParamNames;
    else
        temp = getAnonymousFnInputNames(ReturnFn);
        if isequal(n_d, 0) || isempty(n_d); num_d_vars = 0; else; num_d_vars = length(n_d); end
        if isequal(n_z, 0) || isempty(n_z); num_z_vars = 0; else; num_z_vars = length(n_z); end

        l_a_exp = 0;
        if vfoptions.experienceasset > 0; l_a_exp = vfoptions.experienceasset; end
        if vfoptions.experienceassetz > 0; l_a_exp = vfoptions.experienceassetz; end
        num_a_exp = l_a_exp;
        num_a_endo = length(n_a) - num_a_exp;

        num_semiz_vars = 0; if isfield(vfoptions, 'n_semiz') && prod(vfoptions.n_semiz) > 0; num_semiz_vars = length(vfoptions.n_semiz); end
        num_e_vars = 0; if isfield(vfoptions, 'n_e') && prod(vfoptions.n_e) > 0; num_e_vars = length(vfoptions.n_e); end
        num_u_vars = 0; if vfoptions.riskyasset == 1 && isfield(vfoptions, 'n_u'); num_u_vars = length(vfoptions.n_u); end

        if vfoptions.riskyasset == 1
            num_d1 = 0; if length(vfoptions.refine_d) >= 1; num_d1 = vfoptions.refine_d(1); end
            num_d3 = 0; if length(vfoptions.refine_d) >= 3; num_d3 = vfoptions.refine_d(3); end
            num_prefix_args = num_d1 + num_d3 + 1 + num_semiz_vars + num_z_vars;
        else
            num_prefix_args = num_d_vars + (2 * num_a_endo) + num_a_exp + num_semiz_vars + num_z_vars + num_e_vars + num_u_vars;
        end
        if length(temp) > num_prefix_args; ReturnFnParamNames = {temp{num_prefix_args + 1 : end}}; else; ReturnFnParamNames = {}; end
        ReturnFnParamNames = ReturnFnParamNames(isfield(Parameters, ReturnFnParamNames));
    end
end

is_EZ = strcmp(vfoptions.exoticpreferences, 'EpsteinZin') || strcmp(vfoptions.exoticpreferences, 'QHEpsteinZin');

if isfield(vfoptions,'survivalprobability')
    sj=Parameters.(vfoptions.survivalprobability);
elseif isfield(vfoptions,'WarmGlowBequestsFn')
    sj=ones(N_j,1); sj(end)=0;
else
    sj=ones(N_j,1);
end

if isfield(vfoptions,'WarmGlowBequestsFn')
    warmglow=1;
    temp=getAnonymousFnInputNames(vfoptions.WarmGlowBequestsFn);
    vfoptions.WarmGlowBequestsFnParamsNames={temp{2:end}};
else
    warmglow=0;
end

if is_EZ
    vfoptions = EpsteinZinSetup_VFHorz(N_j, Parameters, ReturnFnParamNames, DiscountFactorParamNames, vfoptions);
end

if vfoptions.divideandconquer==1
    if ~isfield(vfoptions,'level1n')
        if isscalar(n_a)
            vfoptions.level1n=floor(sqrt(n_a(1)));
        elseif length(n_a)>=2
            vfoptions.level1n=[floor(sqrt(n_a(1))),n_a(2:end)];
        end
    else
        if ~isscalar(n_a) && isscalar(vfoptions.level1n)
            vfoptions.level1n=[vfoptions.level1n,n_a(2:end)];
        end
    end
end

if vfoptions.parallel == 2
    if ~isempty(d_grid), d_grid = gpuArray(d_grid); end
    if ~isempty(a_grid), a_grid = gpuArray(a_grid); end
    if ~isempty(z_grid), z_grid = gpuArray(z_grid); end
    if ~isempty(pi_z),   pi_z   = gpuArray(pi_z);   end
end

if vfoptions.alreadygridvals==0
    [z_gridvals_J, pi_z_J, vfoptions] = ExogShockSetup_FHorz(n_z, z_grid, pi_z, N_j, Parameters, vfoptions, 3, 0);
else
    z_gridvals_J = z_grid; pi_z_J = pi_z;
end

if isfield(vfoptions, 'n_semiz') && prod(vfoptions.n_semiz) > 0; N_semiz = prod(vfoptions.n_semiz); else; N_semiz = 0; end
if vfoptions.alreadygridvals_semiexo==0
    if N_semiz > 0; vfoptions = SemiExogShockSetup_FHorz(n_d, N_j, d_grid, Parameters, vfoptions, 3); end
end

N_d = prod(n_d); N_a = prod(n_a); N_z = prod(n_z); N_z_safe = max(1, N_z);
if N_semiz > 0 && isfield(vfoptions, 'semiz_gridvals_J')
    sz_J = vfoptions.semiz_gridvals_J; num_semiz_vars = size(sz_J, 2); num_periods = size(sz_J, 3);
    if N_z > 0; num_z_vars = size(z_gridvals_J, 2); else; num_z_vars = 0; end
    z_gridvals_J_combined = zeros(N_semiz * max(1, N_z), num_semiz_vars + num_z_vars, num_periods, 'like', sz_J);
    for t = 1:num_periods
        if N_z > 0
            semiz_expanded = kron(ones(N_z, 1), sz_J(:,:,t));
            z_expanded = kron(z_gridvals_J(:,:,t), ones(N_semiz, 1));
            z_gridvals_J_combined(:,:,t) = [semiz_expanded, z_expanded];
        else
            z_gridvals_J_combined(:,:,t) = sz_J(:,:,t);
        end
    end
    z_gridvals_J = z_gridvals_J_combined; n_combined_z = [vfoptions.n_semiz, n_z];
else
    n_combined_z = n_z;
end

if strcmp(vfoptions.exoticpreferences, 'QuasiHyperbolic') || strcmp(vfoptions.exoticpreferences, 'QHEpsteinZin')
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
% MULTI-AXIS STATE PARSER: Unstack Endogenous, Experience & Exogenous Grids
% ---------------------------------------------------------------------
has_e = isfield(vfoptions, 'n_e') && prod(vfoptions.n_e) > 0;
n_e_pass = 0; e_grid_pass = [];
if has_e; n_e_pass = vfoptions.n_e; e_grid_pass = vfoptions.e_grid; e_work = vfoptions.e_grid; else; e_work = ones(1, 1, 'like', a_grid); end

l_a_exp = 0;
if vfoptions.experienceasset > 0; l_a_exp = vfoptions.experienceasset; end
if vfoptions.experienceassetz > 0; l_a_exp = vfoptions.experienceassetz; end

num_a_exp = l_a_exp;
num_a_endo = length(n_a) - num_a_exp;

% Dynamic Routing for Slicing and Interpolation
n_a1_dc = n_a(1);
if num_a_endo > 1; n_a2_endo = n_a(2:num_a_endo); else; n_a2_endo = []; end
if num_a_exp > 0; n_a2_exp = n_a(end-num_a_exp+1:end); else; n_a2_exp = []; end

N_a1_dc = n_a1_dc;
N_a2_endo = max(1, prod(n_a2_endo));
N_a2_exp = max(1, prod(n_a2_exp));

% Extract full grids
a1_endo_grid_len = sum(n_a(1:num_a_endo));
a1_endo_grid_vals = a_grid(1:a1_endo_grid_len);
a2_exp_grid_vals = a_grid(a1_endo_grid_len+1:end);

A1_grids_1d = cell(1, num_a_endo);
offset = 0;
for i = 1:num_a_endo
    A1_grids_1d{i} = a1_endo_grid_vals((offset + 1):(offset + n_a(i)));
    offset = offset + n_a(i);
end

% Universal Packing for full states (creates fully meshed arrays)
[TensorReturnFn, D_cells_block, A1_cells, ~, ~] = CreateTensorFnAndCells(ReturnFn, n_d, n_a(1:num_a_endo), n_combined_z, n_e_pass, d_grid, a1_endo_grid_vals, [], []);

if l_a_exp > 0
    [TensoraprimeFn, ~, A2_cells, ~, ~] = CreateTensorFnAndCells(vfoptions.aprimeFn, 0, n_a2_exp, 0, 0, [], a2_exp_grid_vals, [], []);
else
    TensoraprimeFn = []; A2_cells = {};
end

A1_mat = zeros(N_a1_dc * N_a2_endo, num_a_endo, 'like', a_grid);
for i_a = 1:num_a_endo; A1_mat(:, i_a) = A1_cells{i_a}(:); end

A2_mat = zeros(N_a2_exp, num_a_exp, 'like', a_grid); a2_grids_1d = cell(1, num_a_exp); offset = 0;
for i_a = 1:num_a_exp
    A2_mat(:, i_a) = A2_cells{i_a}(:);
    a2_grids_1d{i_a} = a2_exp_grid_vals((offset + 1):(offset + n_a2_exp(i_a)));
    offset = offset + n_a2_exp(i_a);
end

for i_d = 1:length(D_cells_block); D_cells_block{i_d} = reshape(D_cells_block{i_d}, [max(1,prod(n_d)), 1, 1, 1, 1]); end

if l_a_exp > 0 || vfoptions.riskyasset == 1
    aprimeFn = vfoptions.aprimeFn;
    if isfield(vfoptions, 'aprimeFnParamNames'); aprimeFnParamNames = vfoptions.aprimeFnParamNames;
    else
        temp = getAnonymousFnInputNames(aprimeFn);
        num_prefix = length(n_d) + num_a_exp + length(n_z);
        if length(temp) > num_prefix; aprimeFnParamNames = {temp{num_prefix+1:end}}; else; aprimeFnParamNames = {}; end
    end
    aprimeFnParamNames = aprimeFnParamNames(isfield(Parameters, aprimeFnParamNames));
else
    aprimeFn = []; aprimeFnParamNames = {};
end

N_d_safe = max(1, prod(n_d)); n_a_work = prod(n_a);

has_semiz = prod(vfoptions.n_semiz) > 0;
if has_semiz
    if length(n_z) >= length(vfoptions.n_semiz) && isequal(n_z(1:length(vfoptions.n_semiz)), vfoptions.n_semiz)
        N_semiz = prod(vfoptions.n_semiz); n_all_z = n_z; N_z_exog = max(1, prod(n_z) / N_semiz);
    else
        N_semiz = prod(vfoptions.n_semiz); n_all_z = [vfoptions.n_semiz, n_z]; N_z_exog = max(1, prod(n_z));
    end
else
    N_semiz = 1; n_all_z = n_z; N_z_exog = max(1, prod(n_z));
end
has_z = prod(n_z) > 0; n_z_work = N_semiz * N_z_exog; n_e_work = max(1, prod(n_e_pass)); N_ze = n_z_work * n_e_work;

if vfoptions.gridinterplayer(1) == 1
    PolicyKron = zeros(3, n_a_work, n_z_work, n_e_work, N_j, 'like', a_grid);
else
    PolicyKron = zeros(n_a_work, n_z_work, n_e_work, N_j, 'like', a_grid);
end
V = zeros(n_a_work, n_z_work, n_e_work, N_j, 'like', a_grid); V_next = zeros(n_a_work, n_z_work, n_e_work, 'like', a_grid);

% --- Grid Interpolation Setup (Strictly bounds A1_DC) ---
if vfoptions.gridinterplayer(1) == 1
    n2short = vfoptions.ngridinterp; n2long  = n2short * 2 + 3;
    a1_dc_grid = A1_grids_1d{1};
    a1prime_grid = interp1(1:1:N_a1_dc, a1_dc_grid, linspace(1, N_a1_dc, N_a1_dc + (N_a1_dc - 1) * n2short))';
    idx = discretize(a1prime_grid, a1_dc_grid); idx(isnan(idx) | idx == length(a1_dc_grid)) = length(a1_dc_grid) - 1;
    interp_left_idx = idx(:); interp_right_idx = idx(:) + 1;
    a1_left = a1_dc_grid(interp_left_idx); a1_right = a1_dc_grid(interp_right_idx);
    interp_weights = (a1prime_grid(:) - a1_left) ./ (a1_right - a1_left); interp_weights(a1_right == a1_left) = 0;
    if vfoptions.parallel == 2
        interp_left_idx = gpuArray(interp_left_idx); interp_right_idx = gpuArray(interp_right_idx); interp_weights = gpuArray(interp_weights);
    end
else
    n2short = 0; n2long  = 0; a1prime_grid = []; interp_left_idx = []; interp_right_idx = []; interp_weights = [];
end

if is_EZ
    ezc2 = vfoptions.ezc2; ezc3 = vfoptions.ezc3; ezc4 = vfoptions.ezc4; ezc5 = vfoptions.ezc5; ezc6 = vfoptions.ezc6; ezc7 = vfoptions.ezc7; ezc8 = vfoptions.ezc8;
else
    ezc2 = ones(N_j,1); ezc3 = 1; ezc4 = 1; ezc5 = ones(N_j,1); ezc6 = ones(N_j,1); ezc7 = ones(N_j,1); ezc8 = ones(N_j,1);
end

if vfoptions.riskyasset == 1
    disp('V-World: Dispatching Risky Asset model to Tensor Bridge...');
    if length(n_a) > 1
        pass_n_a1 = n_a(1:end-1); pass_n_a2 = n_a(end);
        a1_grid_len = sum(pass_n_a1); pass_a1_grid = a_grid(1:a1_grid_len); pass_a2_grid = a_grid(a1_grid_len+1:end);
    else
        pass_n_a1 = []; pass_n_a2 = n_a; pass_a1_grid = []; pass_a2_grid = a_grid;
    end
    [V, Policy] = ValueFnIter_VFHorz_RiskyAsset_EpsteinZin(...
        n_d, pass_n_a1, pass_n_a2, n_combined_z, vfoptions.n_u, N_j, ...
        d_grid, pass_a1_grid, pass_a2_grid, z_gridvals_J, vfoptions.u_grid, pi_z_J, vfoptions.pi_u, ...
        ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ...
        ReturnFnParamNames, aprimeFnParamNames, vfoptions, ...
        sj, warmglow, ezc2, ezc3, ezc4, ezc5, ezc6, ezc7, ezc8);
    varargout{1} = V; varargout{2} = Policy; if nargout > 2, varargout{3} = []; end; if nargout > 3, varargout{4} = []; end
    return;
end

if ismember(vfoptions.lowmemory, [0, 5]); ze_chunks = {1:N_ze};
elseif vfoptions.lowmemory == 1
    ze_chunks = num2cell(1:N_ze); % Legacy strict Z-loop matching for L2 Cache protection
else; ze_chunks = num2cell(1:N_ze); end

if ismember(vfoptions.lowmemory, [4, 5]) && num_a_exp > 0; a2_chunks = num2cell(1:N_a2_exp); else; a2_chunks = {1:N_a2_exp}; end

chunk_meta = cell(1, length(ze_chunks));
for i_ze = 1:length(ze_chunks)
    c_ze = ze_chunks{i_ze}; if isa(c_ze, 'gpuArray'), c_ze_cpu = gather(c_ze); else, c_ze_cpu = c_ze; end
    [z_ind, e_ind] = ind2sub([n_z_work, n_e_work], c_ze_cpu);
    meta.z_vals = unique(z_ind); meta.e_vals = unique(e_ind);
    meta.n_z_loc = length(meta.z_vals); meta.n_e_loc = length(meta.e_vals); meta.N_ze_local = length(c_ze);
    meta.z_offset_local = reshape((0:meta.N_ze_local-1) * N_a, [1, 1, 1, meta.N_ze_local]);
    if vfoptions.gridinterplayer(1) == 1; meta.z_offset_fine_local = reshape((0:meta.N_ze_local-1) * length(a1prime_grid), [1, 1, 1, meta.N_ze_local]); else; meta.z_offset_fine_local = []; end
    chunk_meta{i_ze} = meta;
end

base_ReturnFnParamsCell = CreateCellFromParams(Parameters, ReturnFnParamNames, 1, vfoptions.precision);
is_age_dependent = false(1, length(ReturnFnParamNames));
for ip = 1:length(ReturnFnParamNames)
    if numel(Parameters.(ReturnFnParamNames{ip})) == N_j; is_age_dependent(ip) = true; end
    if vfoptions.parallel == 2 && isnumeric(base_ReturnFnParamsCell{ip}) && ~isa(base_ReturnFnParamsCell{ip}, 'gpuArray'); base_ReturnFnParamsCell{ip} = gpuArray(base_ReturnFnParamsCell{ip}); end
end

N_semiz_local = 1; N_dsemiz = 1;
if has_semiz && length(n_d) > 0
    N_semiz_local = max(1, prod(vfoptions.n_semiz));
    if isfield(vfoptions, 'l_dsemiz'); N_dsemiz = prod(n_d(end-vfoptions.l_dsemiz+1:end)); else; N_dsemiz = n_d(end); end
end
N_z_exog = max(1, n_z_work / N_semiz_local);

for reverse_j = 0:N_j-1
    jj = N_j - reverse_j;
    if vfoptions.verbose==1; fprintf('Finite horizon: %i of %i \n',jj, N_j); end
    ReturnFnParamsCell = base_ReturnFnParamsCell; pi_z_j = pi_z_J(:, :, min(jj, size(pi_z_J, 3)));
    for ip = find(is_age_dependent)
        val = cast(Parameters.(ReturnFnParamNames{ip})(jj), vfoptions.precision);
        if vfoptions.parallel == 2; ReturnFnParamsCell{ip} = gpuArray(val); else; ReturnFnParamsCell{ip} = val; end
    end
    DiscountFactorParamsVec = CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj, vfoptions.precision); beta_j = prod(DiscountFactorParamsVec);
    if l_a_exp > 0; aprimeFnParamsCell = CreateCellFromParams(Parameters, aprimeFnParamNames, jj); else; aprimeFnParamsCell = {}; end

    if jj == N_j && isfield(vfoptions, 'V_Jplus1') && ~isempty(vfoptions.V_Jplus1)
        V_next = reshape(vfoptions.V_Jplus1, [n_a_work, n_z_work, n_e_work]);
        if vfoptions.parallel == 2 && ~isa(V_next, 'gpuArray'); V_next = gpuArray(V_next); end
    end

    if jj == N_j && (~isfield(vfoptions, 'V_Jplus1') || isempty(vfoptions.V_Jplus1))
        if warmglow == 1
            wg_params = CreateCellFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames, jj);
            WG_eval = vfoptions.WarmGlowBequestsFn(a_grid, wg_params{:});
            if isscalar(WG_eval); WG_eval = WG_eval * ones(size(a_grid), 'like', a_grid); end
            if is_EZ
                valid_wg = isfinite(WG_eval) & (WG_eval ~= 0); WG_transformed = WG_eval;
                if ezc5(jj) == 1; WG_transformed(valid_wg) = ezc4 * WG_eval(valid_wg); else; WG_transformed(valid_wg) = max(ezc4 * WG_eval(valid_wg), 0).^ezc5(jj); end
                WG_transformed(WG_eval == 0) = 0; WG_eval = WG_transformed;
            end
            EV = repmat(reshape(WG_eval, [N_a, 1, 1, 1]), [1, N_semiz_local * N_z_exog, n_e_work, N_dsemiz]);
        else
            EV = zeros(N_a, N_semiz_local * N_z_exog, n_e_work, N_dsemiz, 'like', a_grid);
        end
        V_next = zeros(n_a_work, n_z_work, n_e_work, 'like', a_grid);
    else
        if has_e && isfield(vfoptions, 'e_gridvals_J'); e_work = vfoptions.e_gridvals_J(:, :, min(jj, size(vfoptions.e_gridvals_J, 3))); end
        valid_V = isfinite(V_next) & (V_next ~= 0); V_transformed = V_next;
        if ezc5(jj) == 1; V_transformed(valid_V) = ezc4 * V_next(valid_V); else; V_transformed(valid_V) = max(ezc4 * V_next(valid_V), 0).^ezc5(jj); end
        V_transformed(V_next == 0) = 0;

        if has_e
            if isfield(vfoptions, 'pi_e_J'); pi_e_j = vfoptions.pi_e_J(:, min(jj + 1, size(vfoptions.pi_e_J, 2))); else; pi_e_j = vfoptions.pi_e; end
            if vfoptions.parallel == 2 && ~isa(pi_e_j, 'gpuArray'); pi_e_j = gpuArray(pi_e_j); end
            V_trans_flat = reshape(V_transformed, [N_a * n_z_work, n_e_work]);
            V_inf_mask = (V_trans_flat == -Inf); V_safe = V_trans_flat; V_safe(V_inf_mask) = -1e250;
            V_expected_e = V_safe * pi_e_j(:);
            inf_restore = (V_inf_mask * (pi_e_j(:) > 0)) > 0; V_expected_e(inf_restore) = -Inf;
            V_transformed = repmat(reshape(V_expected_e, [N_a, n_z_work, 1]), [1, 1, n_e_work]);
        end

        EV = zeros(N_a, N_semiz_local * N_z_exog, n_e_work, N_dsemiz, 'like', V_next);
        for ie = 1:n_e_work
            V_curr = V_transformed(:,:,ie);
            if N_z_exog > 1 && has_z
                V_slice = reshape(V_curr, [N_a * N_semiz_local, N_z_exog]);
                V_inf_mask = (V_slice == -Inf); V_safe = V_slice; V_safe(V_inf_mask) = -1e250;
                V_z_eval = V_safe * pi_z_j';
                inf_restore = (V_inf_mask * (pi_z_j' > 0)) > 0; V_z_eval(inf_restore) = -Inf;
                V_z_eval = reshape(V_z_eval, [N_a, N_semiz_local, N_z_exog]);
            else
                V_z_eval = reshape(V_curr, [N_a, N_semiz_local, N_z_exog]);
            end
            if has_semiz
                pi_semiz_j = vfoptions.pi_semiz_J(:, :, :, min(jj, size(vfoptions.pi_semiz_J, 4)));
                V_perm = reshape(permute(V_z_eval, [2, 1, 3]), [N_semiz_local, N_a * N_z_exog]);
                V_inf_mask = (V_perm == -Inf); V_safe = V_perm; V_safe(V_inf_mask) = -1e250;
                for idsemiz = 1:N_dsemiz
                    pi_semiz_d = pi_semiz_j(:, :, idsemiz); EV_perm = pi_semiz_d * V_safe;
                    inf_restore = (pi_semiz_d > 0) * V_inf_mask > 0; EV_perm(inf_restore) = -Inf;
                    EV_d = permute(reshape(EV_perm, [N_semiz_local, N_a, N_z_exog]), [2, 1, 3]);
                    EV(:,:,ie,idsemiz) = reshape(EV_d, [N_a, N_semiz_local * N_z_exog]);
                end
            else; EV(:,:,ie,1) = reshape(V_z_eval, [N_a, N_semiz_local * N_z_exog]); end
        end

        if warmglow == 1
            wg_params = CreateCellFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames, jj);
            WG_eval = vfoptions.WarmGlowBequestsFn(a_grid, wg_params{:});
            if isscalar(WG_eval); WG_eval = WG_eval * ones(size(a_grid), 'like', a_grid); end
            if is_EZ
                valid_wg = isfinite(WG_eval) & (WG_eval ~= 0); WG_transformed = WG_eval;
                if ezc5(jj) == 1; WG_transformed(valid_wg) = ezc4 * WG_eval(valid_wg); else; WG_transformed(valid_wg) = max(ezc4 * WG_eval(valid_wg), 0).^ezc5(jj); end
                WG_transformed(WG_eval == 0) = 0; WG_eval = WG_transformed;
            end
            WG_eval = reshape(WG_eval, [N_a, 1, 1, 1]); EV = EV * sj(jj) + (1 - sj(jj)) * WG_eval;
        end
    end

    valid_EV = isfinite(EV) & (EV ~= 0);
    if ezc6(jj) ~= 1; EV(valid_EV) = max(EV(valid_EV), 0).^ezc6(jj); end
    if ezc8(jj) ~= 1; EV(valid_EV) = max(EV(valid_EV), 0).^ezc8(jj); end
    EV_flat_ze = reshape(EV, [N_a, N_ze, N_dsemiz]);

    V_j_max        = zeros(N_a, N_ze, 'like', V_next);
    Pol_apr_max    = zeros(N_a, N_ze, 'like', V_next);
    Pol_d_max      = zeros(N_a, N_ze, 'like', V_next);
    Pol_L2idx_max  = zeros(N_a, N_ze, 'like', V_next);
    Pol_L2flag_max = zeros(N_a, N_ze, 'like', V_next);

    if N_dsemiz > 1
        if isfield(vfoptions, 'l_dsemiz'); N_d_prefix = max(1, prod(n_d(1:end-vfoptions.l_dsemiz))); else; N_d_prefix = max(1, prod(n_d(1:end-1))); end
        dsemiz_idx = ceil((1:N_d_safe)' / N_d_prefix); dsemiz_idx_tensor = reshape(dsemiz_idx, [N_d_safe, 1, 1, 1]);
    else
        dsemiz_idx_tensor = ones(N_d_safe, 1, 1, 1);
    end

    if vfoptions.divideandconquer == 1
        for i_ze = 1:length(ze_chunks)
            meta = chunk_meta{i_ze}; n_z_loc = meta.n_z_loc; n_e_loc = meta.n_e_loc;
            curr_ze = ze_chunks{i_ze}; N_ze_local = length(curr_ze);
            EV_local = EV_flat_ze(:, curr_ze, :);
            if has_semiz || has_z
                num_z_vars = length(n_combined_z); Z_cells_local = cell(1, num_z_vars);
                if size(z_gridvals_J, 2) ~= num_z_vars
                    z_inflated = reshape(z_gridvals_J, [N_z, num_z_vars, size(z_gridvals_J, ndims(z_gridvals_J))]);
                    for iz = 1:num_z_vars; Z_cells_local{iz} = reshape(z_inflated(meta.z_vals, iz, min(jj, size(z_inflated,3))), [1, 1, 1, n_z_loc, 1]); end
                else
                    for iz = 1:num_z_vars; Z_cells_local{iz} = reshape(z_gridvals_J(meta.z_vals, iz, min(jj, size(z_gridvals_J,3))), [1, 1, 1, n_z_loc, 1]); end
                end
            else; Z_cells_local = {}; end
            if has_e
                num_e_vars = size(e_work, 2); E_cells_local = cell(1, num_e_vars);
                for ie_var = 1:num_e_vars; E_cells_local{ie_var} = reshape(e_work(meta.e_vals, ie_var), [1, 1, 1, 1, n_e_loc]); end
            else; E_cells_local = {}; end

            if vfoptions.gridinterplayer(1) == 1
                N_cols = N_ze_local * N_dsemiz; zero_weights = (interp_weights == 0); one_weights = (interp_weights == 1);
                N_a2_rem = N_a2_endo * N_a2_exp;
                EV_2d = reshape(EV_local, [N_a1_dc, N_a2_rem * N_cols]);
                EV_left_val = EV_2d(interp_left_idx, :); EV_right_val = EV_2d(interp_right_idx, :);
                EV_interp_flat = EV_left_val + interp_weights .* (EV_right_val - EV_left_val);
                EV_interp_flat(zero_weights, :) = EV_left_val(zero_weights, :); EV_interp_flat(one_weights, :) = EV_right_val(one_weights, :);
                EV_interp_flat(isnan(EV_interp_flat)) = -Inf;
                EV_interp_local = reshape(EV_interp_flat, [length(a1prime_grid), N_a2_rem, N_ze_local, N_dsemiz]);
            else; EV_interp_local = []; end

            if l_a_exp == 0
                EV_reshaped = reshape(EV_local, [N_a1_dc * N_a2_endo, n_z_loc, n_e_loc, N_dsemiz]);
                EV_d_sliced = EV_reshaped(:, :, :, dsemiz_idx_tensor(:));
                EV_bounded_pre = beta_j .* permute(EV_d_sliced, [4, 1, 5, 2, 3]);
                d_vec = reshape(0:N_d_safe-1, [N_d_safe, 1, 1, 1, 1, 1]);
                z_vec = reshape((0:n_z_loc-1) * (N_d_safe * N_a1_dc * N_a2_endo), [1, 1, 1, 1, n_z_loc, 1]);
                e_vec = reshape((0:n_e_loc-1) * (N_d_safe * N_a1_dc * N_a2_endo * n_z_loc), [1, 1, 1, 1, 1, n_e_loc]);
                static_EV_offset = cast(d_vec + 1 + z_vec + e_vec, 'like', EV_bounded_pre);
            else; EV_bounded_pre = []; static_EV_offset = []; end

            vfoptions.level1n = vfoptions.level1n(1);
            LocalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar, dc_mode_override) Evaluate_Case1_TensorBlock(...
                state_idx, loweredge_matrix, maxgap_scalar, N_a1_dc, N_a2_endo, max(1, N_a2_exp), N_d_safe, N_ze_local, ...
                Z_cells_local, E_cells_local, D_cells_block, A1_mat, A2_mat, A1_grids_1d, a2_grids_1d, ...
                vfoptions.gridinterplayer, n2short, n2long, beta_j, EV_local, EV_bounded_pre, EV_interp_local, a1prime_grid, ...
                TensorReturnFn, ReturnFnParamsCell, ezc2(jj), ezc3, ezc4, ezc7(jj), ...
                TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor, n_z_loc, n_e_loc, static_EV_offset, dc_mode_override);

            if vfoptions.gridinterplayer(1) == 1
                temp_vfoptions = vfoptions; temp_vfoptions.gridinterplayer = 0;
                LocalBlockFn_Coarse = @(state_idx, loweredge_matrix, maxgap_scalar) LocalBlockFn(state_idx, loweredge_matrix, maxgap_scalar, 2);
                if num_a_endo == 1
                    [~, p_apr_coarse, ~, ~, ~] = ValueFnIter_DC1_Slicer(N_a1_dc * N_a2_exp, N_a, 1, N_ze_local, temp_vfoptions, LocalBlockFn_Coarse);
                    loweredge_pass = p_apr_coarse;
                else
                    [~, ~, ~, ~, ~, p_a1_per_a2] = ValueFnIter_DC2A_Slicer(N_a1_dc, N_a2_endo, N_a2_endo * N_a2_exp, N_a1_dc, N_ze_local, temp_vfoptions, LocalBlockFn_Coarse);
                    loweredge_pass = p_a1_per_a2;
                end

                % --- VRAM Protection: Chunk the Grid Interp Fine Pass ---
                flat_choices = max(1, N_d_safe) * n2long * max(1, N_a2_endo);
                max_states_per_chunk = max(1, floor(40000000 / (flat_choices * N_ze_local)));

                v = zeros(N_a, N_ze_local, 'like', EV_local);
                p_apr = zeros(N_a, N_ze_local, 'like', EV_local);
                p_d = zeros(N_a, N_ze_local, 'like', EV_local);
                p_l2idx = zeros(N_a, N_ze_local, 'like', EV_local);
                p_l2flag = zeros(N_a, N_ze_local, 'like', EV_local);

                for chunk_start = 1:max_states_per_chunk:N_a
                    chunk_end = min(N_a, chunk_start + max_states_per_chunk - 1);
                    state_chunk = chunk_start:chunk_end;
                    c_idx = chunk_start:chunk_end;

                    if num_a_endo == 1
                        loweredge_chunk = loweredge_pass(state_chunk, :);
                    else
                        loweredge_chunk = loweredge_pass(:, state_chunk, :);
                    end
                    [v_c, p_apr_c, p_d_c, p_l2idx_c, p_l2flag_c] = LocalBlockFn(state_chunk, loweredge_chunk, n2long - 1, 0);

                    v(c_idx, :) = v_c;
                    p_apr(c_idx, :) = p_apr_c;
                    p_d(c_idx, :) = p_d_c;
                    p_l2idx(c_idx, :) = p_l2idx_c;
                    p_l2flag(c_idx, :) = p_l2flag_c;
                end
            else
                LocalBlockFn_Standard = @(state_idx, loweredge_matrix, maxgap_scalar) LocalBlockFn(state_idx, loweredge_matrix, maxgap_scalar, 0);
                if num_a_endo == 1
                    [v, p_apr, p_d, p_l2idx, p_l2flag] = ValueFnIter_DC1_Slicer(N_a1_dc * N_a2_exp, N_a, 1, N_ze_local, vfoptions, LocalBlockFn_Standard);
                else
                    [v, p_apr, p_d, p_l2idx, p_l2flag] = ValueFnIter_DC2A_Slicer(N_a1_dc, N_a2_endo, N_a2_endo * N_a2_exp, N_a1_dc, N_ze_local, vfoptions, LocalBlockFn_Standard);
                end
            end
            V_j_max(:, curr_ze)     = reshape(v,     [N_a, N_ze_local]);
            Pol_apr_max(:, curr_ze) = reshape(p_apr, [N_a, N_ze_local]);
            Pol_d_max(:, curr_ze)   = reshape(p_d,   [N_a, N_ze_local]);
            if vfoptions.gridinterplayer(1) == 1
                Pol_L2idx_max(:, curr_ze)  = reshape(p_l2idx,  [N_a, N_ze_local]);
                Pol_L2flag_max(:, curr_ze) = reshape(p_l2flag, [N_a, N_ze_local]);
            end
        end
    else
        % Non-DC block
        for i_a2 = 1:length(a2_chunks)
            curr_a2 = a2_chunks{i_a2}; N_a2_local = length(curr_a2);
            start_a_idx = (min(curr_a2) - 1) * (N_a1_dc * N_a2_endo) + 1; end_a_idx   = max(curr_a2) * (N_a1_dc * N_a2_endo);
            for i_ze = 1:length(ze_chunks)
                curr_ze = ze_chunks{i_ze}; N_ze_local = length(curr_ze);
                meta = chunk_meta{i_ze}; n_z_loc = meta.n_z_loc; n_e_loc = meta.n_e_loc;
                if l_a_exp > 0; A2_local = A2_mat(curr_a2, :); else; A2_local = []; end
                EV_local = EV_flat_ze(:, curr_ze, :);
                if has_semiz || has_z
                    num_z_vars = length(n_combined_z); Z_cells_local = cell(1, num_z_vars);
                    if size(z_gridvals_J, 2) ~= num_z_vars
                        z_inflated = reshape(z_gridvals_J, [N_z, num_z_vars, size(z_gridvals_J, ndims(z_gridvals_J))]);
                        for iz = 1:num_z_vars; Z_cells_local{iz} = reshape(z_inflated(meta.z_vals, iz, min(jj, size(z_inflated,3))), [1, 1, 1, n_z_loc, 1]); end
                    else
                        for iz = 1:num_z_vars; Z_cells_local{iz} = reshape(z_gridvals_J(meta.z_vals, iz, min(jj, size(z_gridvals_J,3))), [1, 1, 1, n_z_loc, 1]); end
                    end
                else; Z_cells_local = {}; end
                if has_e
                    num_e_vars = size(e_work, 2); E_cells_local = cell(1, num_e_vars);
                    for ie = 1:num_e_vars; E_cells_local{ie} = reshape(e_work(meta.e_vals, ie), [1, 1, 1, 1, n_e_loc]); end
                else; E_cells_local = {}; end

                if vfoptions.gridinterplayer(1) == 1
                    N_cols = N_ze_local * N_dsemiz; zero_weights = (interp_weights == 0); one_weights = (interp_weights == 1);
                    if l_a_exp > 0
                        EV_2d = reshape(EV_local, [N_a1_dc, N_a2_endo * N_a2_local * N_cols]);
                        EV_left_val = EV_2d(interp_left_idx, :); EV_right_val = EV_2d(interp_right_idx, :);
                        EV_interp_flat = EV_left_val + interp_weights .* (EV_right_val - EV_left_val);
                        EV_interp_flat(zero_weights, :) = EV_left_val(zero_weights, :); EV_interp_flat(one_weights, :) = EV_right_val(one_weights, :);
                        EV_interp_flat(isnan(EV_interp_flat)) = -Inf;
                        EV_interp_local = reshape(EV_interp_flat, [length(a1prime_grid), N_a2_endo, N_a2_local, N_ze_local, N_dsemiz]);
                    else
                        EV_2d = reshape(EV_local, [N_a1_dc, N_a2_endo * N_cols]);
                        EV_left_val = EV_2d(interp_left_idx, :); EV_right_val = EV_2d(interp_right_idx, :);
                        EV_interp_flat = EV_left_val + interp_weights .* (EV_right_val - EV_left_val);
                        EV_interp_flat(zero_weights, :) = EV_left_val(zero_weights, :); EV_interp_flat(one_weights, :) = EV_right_val(one_weights, :);
                        EV_interp_flat(isnan(EV_interp_flat)) = -Inf;
                        EV_interp_local = reshape(EV_interp_flat, [length(a1prime_grid), N_a2_endo, N_ze_local, N_dsemiz]);
                    end
                else; EV_interp_local = []; end

                if l_a_exp == 0
                    EV_bounded_pre = beta_j .* reshape(EV_local(:), [N_d_safe, N_a1_dc * N_a2_endo, 1, n_z_loc, n_e_loc]);
                    d_vec = reshape(0:N_d_safe-1, [N_d_safe, 1, 1, 1, 1, 1]);
                    z_vec = reshape((0:n_z_loc-1) * (N_d_safe * N_a1_dc * N_a2_endo), [1, 1, 1, 1, n_z_loc, 1]);
                    e_vec = reshape((0:n_e_loc-1) * (N_d_safe * N_a1_dc * N_a2_endo * n_z_loc), [1, 1, 1, 1, 1, n_e_loc]);
                    static_EV_offset = cast(d_vec + 1 + z_vec + e_vec, 'like', EV_bounded_pre);
                else; EV_bounded_pre = []; static_EV_offset = []; end

                LocalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar, dc_mode_override) Evaluate_Case1_TensorBlock(...
                    state_idx, loweredge_matrix, maxgap_scalar, N_a1_dc, N_a2_endo, max(1, N_a2_local), N_d_safe, N_ze_local, ...
                    Z_cells_local, E_cells_local, D_cells_block, A1_mat, A2_local, A1_grids_1d, a2_grids_1d, ...
                    vfoptions.gridinterplayer, n2short, n2long, beta_j, EV_local, EV_bounded_pre, EV_interp_local, a1prime_grid, ...
                    TensorReturnFn, ReturnFnParamsCell, ezc2(jj), ezc3, ezc4, ezc7(jj), ...
                    TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor, n_z_loc, n_e_loc, static_EV_offset, dc_mode_override);

                state_list = start_a_idx:end_a_idx; total_states = length(state_list);
                flat_choices = max(1, N_d_safe) * N_a1_dc * N_a2_endo;
                max_states_per_chunk = max(1, floor(40000000 / (flat_choices * n_z_loc * n_e_loc)));

                v_concat = zeros(total_states, N_ze_local, 'like', EV_local);
                p_apr_concat = zeros(total_states, N_ze_local, 'like', EV_local);
                p_d_concat = zeros(total_states, N_ze_local, 'like', EV_local);
                p_l2idx_concat = zeros(total_states, N_ze_local, 'like', EV_local);
                p_l2flag_concat = zeros(total_states, N_ze_local, 'like', EV_local);

                for chunk_start = 1:max_states_per_chunk:total_states
                    chunk_end = min(total_states, chunk_start + max_states_per_chunk - 1);
                    state_chunk = state_list(chunk_start:chunk_end);
                    c_idx = chunk_start:chunk_end;

                    if vfoptions.gridinterplayer(1) == 1
                        [~, p_apr_coarse, ~, ~, ~, p_a1_per_a2] = LocalBlockFn(state_chunk, [], 0, 2);
                        if num_a_endo == 1; loweredge_pass = p_apr_coarse; else; loweredge_pass = p_a1_per_a2; end
                        [v_c, p_apr_c, p_d_c, p_l2idx_c, p_l2flag_c] = LocalBlockFn(state_chunk, loweredge_pass, n2long - 1, 0);
                    else
                        [v_c, p_apr_c, p_d_c, p_l2idx_c, p_l2flag_c] = LocalBlockFn(state_chunk, [], 0, 0);
                    end

                    v_concat(c_idx, :) = v_c;
                    p_apr_concat(c_idx, :) = p_apr_c;
                    p_d_concat(c_idx, :) = p_d_c;
                    if vfoptions.gridinterplayer(1) == 1
                        p_l2idx_concat(c_idx, :) = p_l2idx_c;
                        p_l2flag_concat(c_idx, :) = p_l2flag_c;
                    end
                end
                V_j_max(start_a_idx:end_a_idx, curr_ze)     = reshape(v_concat,     [length(state_list), N_ze_local]);
                Pol_apr_max(start_a_idx:end_a_idx, curr_ze) = reshape(p_apr_concat, [length(state_list), N_ze_local]);
                Pol_d_max(start_a_idx:end_a_idx, curr_ze)   = reshape(p_d_concat,   [length(state_list), N_ze_local]);
                if vfoptions.gridinterplayer(1) == 1
                    Pol_L2idx_max(start_a_idx:end_a_idx, curr_ze)  = reshape(p_l2idx_concat,  [length(state_list), N_ze_local]);
                    Pol_L2flag_max(start_a_idx:end_a_idx, curr_ze) = reshape(p_l2flag_concat, [length(state_list), N_ze_local]);
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
        if N_d > 0; PolicyKron(1, :, :, :, jj) = (Pol_apr_max - 1) * N_d + Pol_d_max; else; PolicyKron(1, :, :, :, jj) = Pol_apr_max; end
        PolicyKron(2, :, :, :, jj) = Pol_L2idx_max; PolicyKron(3, :, :, :, jj) = Pol_L2flag_max;
    else
        if N_d > 0; PolicyKron(:, :, :, jj) = (Pol_apr_max - 1) * N_d + Pol_d_max; else; PolicyKron(:, :, :, jj) = Pol_apr_max; end
    end
    V(:, :, :, jj) = V_j_max; V_next = V_j_max;
end

if N_z == 0; V = squeeze(V); end
if N_d == 0; n_daprime = n_a(1:num_a_endo); else; n_daprime = [n_d, n_a(1:num_a_endo)]; end
if vfoptions.gridinterplayer(1) ~= 1; PolicyKron = shiftdim(PolicyKron, -1); end

if isfield(vfoptions, 'outputkron') && vfoptions.outputkron == 1
    varargout{1} = V; varargout{2} = PolicyKron; return
end

disp('Unpacking Policy tensor to System RAM...');
num_pol_vars = length(n_daprime); n_daprime_col = n_daprime(:); divisors = cumprod([1; n_daprime_col(1:end-1)]);
MAX_INT32 = 2147483647;

if vfoptions.gridinterplayer(1) == 1
    total_elements = (num_pol_vars + 2) * n_a_work * n_z_work * n_e_work * N_j;
    if total_elements < (MAX_INT32 * 0.9)
        BaseIndexKron = PolicyKron(1, :, :, :, :);
        P_base_gpu = mod(floor((BaseIndexKron - 1) ./ divisors), n_daprime_col) + 1;
        P_gpu = [P_base_gpu; PolicyKron(2:3, :, :, :, :)]; Policy_flat = gather(P_gpu);
    else
        disp('Using memory-safe iterative unpacking due to massive array size...');
        Policy_flat = zeros([num_pol_vars + 2, n_a_work, n_z_work, n_e_work, N_j], vfoptions.precision);
        for jj = 1:N_j
            PK_j = PolicyKron(1, :, :, :, jj); P_base = mod(floor((PK_j - 1) ./ divisors), n_daprime_col) + 1;
            Policy_flat(:, :, :, :, jj) = gather([P_base; PolicyKron(2:3, :, :, :, jj)]);
        end
    end
else
    total_elements = num_pol_vars * n_a_work * n_z_work * n_e_work * N_j;
    if total_elements < (MAX_INT32 * 0.9)
        P_gpu = mod(floor((PolicyKron - 1) ./ divisors), n_daprime_col) + 1; Policy_flat = gather(P_gpu);
    else
        disp('Using memory-safe iterative unpacking due to massive array size...');
        Policy_flat = zeros([num_pol_vars, n_a_work, n_z_work, n_e_work, N_j], vfoptions.precision);
        for jj = 1:N_j
            PK_j = PolicyKron(:, :, :, :, jj); P_j_gpu = mod(floor((PK_j - 1) ./ divisors), n_daprime_col) + 1;
            Policy_flat(:, :, :, :, jj) = gather(P_j_gpu);
        end
    end
end

V_cpu = gather(V); out_pol_vars = size(Policy_flat, 1);
out_n_a = n_a(n_a > 0); if isempty(out_n_a); out_n_a = 1; end
out_n_all_z = n_all_z(n_all_z > 0); if isempty(out_n_all_z); out_n_all_z = 1; end
state_shape = out_n_a;
if has_z || has_semiz; state_shape = [state_shape, out_n_all_z]; end
if has_e; state_shape = [state_shape, n_e_pass]; end
state_shape = [state_shape, N_j];
Policy = reshape(Policy_flat, [out_pol_vars, state_shape]);
V = reshape(V_cpu, state_shape);
varargout{1} = V; varargout{2} = Policy;
end

function [V_j_max, Pol_apr_max, Pol_d_max, Pol_L2idx_max, Pol_L2flag_max, Pol_a1_per_a2] = Evaluate_Case1_TensorBlock(...
    state_idx, loweredge_matrix, maxgap_scalar, N_a1_dc, N_a2_endo, N_a_exp, N_d_safe, N_ze_local, ...
    Z_cells_block, E_cells_block, D_cells_block, A1_mat, A2_mat, A1_grids_1d, a2_grids_1d, ...
    gridinterplayer, n2short, n2long, beta_j, EV_local, EV_bounded_pre, EV_interp_local, a1prime_grid, ...
    TensorReturnFn, ReturnFnParamsCell, ezc2_j, ezc3, ezc4, ezc7_j, ...
    TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor, n_z_loc, n_e_loc, static_EV_offset, is_dc_mode)

N_states = length(state_idx); num_a1_vars = length(A1_grids_1d);
FLAT_STATES = N_states * N_ze_local;

if N_a_exp > 1; [a1_sub, a2_sub] = ind2sub([N_a1_dc * N_a2_endo, N_a_exp], state_idx); else; a1_sub = state_idx; end

% --- 2D Flat-Pack: States ---
A1_flat = cell(1, num_a1_vars);
for ia = 1:num_a1_vars
    val = reshape(A1_mat(a1_sub, ia), [N_states, 1]);
    A1_flat{ia} = reshape(repmat(val, [1, N_ze_local]), [1, FLAT_STATES]);
end

if N_a_exp > 1
    num_a_exp_vars = size(A2_mat, 2); A2_flat = cell(1, num_a_exp_vars);
    for ia = 1:num_a_exp_vars
        val = reshape(A2_mat(a2_sub, ia), [N_states, 1]);
        A2_flat{ia} = reshape(repmat(val, [1, N_ze_local]), [1, FLAT_STATES]);
    end
else
    A2_flat = {};
end

Z_flat = cell(1, length(Z_cells_block));
for iz = 1:length(Z_cells_block)
    val = reshape(Z_cells_block{iz}, [1, n_z_loc]);
    val = repmat(val, [N_states, 1]);
    Z_flat{iz} = repmat(val(:).', [1, n_e_loc]);
end

E_flat = cell(1, length(E_cells_block));
for ie = 1:length(E_cells_block)
    val = reshape(E_cells_block{ie}, [1, n_e_loc]);
    E_flat{ie} = repelem(val, 1, N_states * n_z_loc);
end

ZE_idx = repmat(1:N_ze_local, [N_states, 1]);
ZE_idx_flat = reshape(ZE_idx, [1, FLAT_STATES]);

if isempty(loweredge_matrix)
    if gridinterplayer(1) == 0 || is_dc_mode == 2
        % =================================================================
        % BRANCH 1A: COARSE EVALUATION (2D)
        % =================================================================
        grids_for_choices = A1_grids_1d;
        if num_a1_vars > 1; [mesh_out{1:num_a1_vars}] = ndgrid(grids_for_choices{:}); else; mesh_out{1} = grids_for_choices{1}; end
        num_choices_total = numel(mesh_out{1});
        FLAT_CHOICES = N_d_safe * num_choices_total;

        Apr_flat = cell(1, num_a1_vars);
        for ia = 1:num_a1_vars
            val = reshape(mesh_out{ia}, [num_choices_total, 1]);
            Apr_flat{ia} = repelem(val, N_d_safe, 1);
        end

        D_flat = cell(1, length(D_cells_block));
        for id = 1:length(D_cells_block)
            val = reshape(D_cells_block{id}, [N_d_safe, 1]);
            D_flat{id} = repmat(val, [num_choices_total, 1]);
        end

        choice_idx_linear = reshape(1:num_choices_total, [num_choices_total, 1]);

        if N_a_exp > 1
            F_tensor = TensorReturnFn(D_flat{:}, Apr_flat{:}, A1_flat{:}, A2_flat{:}, Z_flat{:}, E_flat{:}, ReturnFnParamsCell{:});
            A2_prime = TensoraprimeFn(D_flat{:}, A2_flat{:}, Z_flat{:}, E_flat{:}, aprimeFnParamsCell{:});
            a2_grid_1d_vec = a2_grids_1d{1}; a2_prime_clipped = max(a2_grid_1d_vec(1), min(A2_prime, a2_grid_1d_vec(end)));
            idx = discretize(a2_prime_clipped, a2_grid_1d_vec); idx(isnan(idx)) = N_a_exp - 1; idx = max(1, min(idx, N_a_exp - 1));
            a2_left = reshape(a2_grid_1d_vec(idx), size(idx)); a2_right = reshape(a2_grid_1d_vec(idx+1), size(idx));
            weight = (a2_prime_clipped - a2_left) ./ (a2_right - a2_left); weight(a2_right == a2_left) = 0;

            N_a2_global = max(1, prod(cellfun(@length, a2_grids_1d)));
            choice_idx_exp = repelem(choice_idx_linear, N_d_safe, 1);
            idx_left  = choice_idx_exp + (idx - 1) * (N_a1_dc * N_a2_endo) + (ZE_idx_flat - 1) * (N_a1_dc * N_a2_endo * N_a2_global);
            idx_right = choice_idx_exp + (idx) * (N_a1_dc * N_a2_endo) + (ZE_idx_flat - 1) * (N_a1_dc * N_a2_endo * N_a2_global);

            max_idx_row = size(EV_local, 1);
            dsemiz_flat = reshape(dsemiz_idx_tensor, [N_d_safe, 1]);
            dsemiz_expanded = repmat(dsemiz_flat, [num_choices_total, 1]);
            linear_idx_left  = min(max_idx_row, max(1, idx_left  + (dsemiz_expanded - 1) * max_idx_row));
            linear_idx_right = min(max_idx_row, max(1, idx_right + (dsemiz_expanded - 1) * max_idx_row));

            EV_bounded = EV_local(linear_idx_left) + weight .* (EV_local(linear_idx_right) - EV_local(linear_idx_left));
            EV_bounded(weight == 0) = EV_local(linear_idx_left(weight == 0)); EV_bounded(weight == 1) = EV_local(linear_idx_right(weight == 1));
            EV_bounded(isnan(EV_bounded)) = -Inf; EV_bounded = beta_j .* EV_bounded;
        else
            F_tensor = TensorReturnFn(D_flat{:}, Apr_flat{:}, A1_flat{:}, Z_flat{:}, E_flat{:}, ReturnFnParamsCell{:});
            choice_idx_exp = repelem(choice_idx_linear, N_d_safe, 1);

            stride_z = N_d_safe * N_a1_dc * N_a2_endo;
            ze_base = reshape((0:n_z_loc-1)' * stride_z + (0:n_e_loc-1) * stride_z * n_z_loc, [1, N_ze_local]);
            ze_expanded = reshape(repmat(ze_base, [N_states, 1]), [1, FLAT_STATES]);

            d_expanded = repmat((1:N_d_safe)', [num_choices_total, 1]);
            linear_idx_EV = d_expanded + (choice_idx_exp - 1) * N_d_safe + ze_expanded;
            EV_bounded = EV_bounded_pre(linear_idx_EV);
        end

        RHS = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bounded, 1, 1, ezc2_j, ezc3, ezc4, ezc7_j);
        clear F_tensor EV_bounded; % Memory Hoist

        [V_sub_coarse, Pol_sub_idx] = max(RHS, [], 1);

        if nargout > 5
            RHS_for_d = reshape(RHS, [N_d_safe, num_choices_total, FLAT_STATES]);
            RHS_max_d = max(RHS_for_d, [], 1);
            clear RHS_for_d;
            RHS_4D = reshape(RHS_max_d, [N_a1_dc, N_a2_endo, N_states, N_ze_local]);
            clear RHS_max_d;
            [~, max_a1_idx] = max(RHS_4D, [], 1);
            clear RHS_4D;
            Pol_a1_per_a2 = reshape(max_a1_idx, [N_a2_endo, N_states, N_ze_local]);
        else
            Pol_a1_per_a2 = [];
        end
        clear RHS;

        d_idx_local   = mod(Pol_sub_idx - 1, max(1, N_d_safe)) + 1; apr_idx_local = ceil(Pol_sub_idx / max(1, N_d_safe));
        V_j_max        = reshape(V_sub_coarse,  [N_states, N_ze_local]);
        Pol_apr_max    = reshape(apr_idx_local, [N_states, N_ze_local]);
        Pol_d_max      = reshape(d_idx_local,   [N_states, N_ze_local]);
        Pol_L2idx_max  = []; Pol_L2flag_max = [];

    else
        % =================================================================
        % BRANCH 1B: FULL FINE GRID EVALUATION (2D)
        % =================================================================
        grids_for_choices = A1_grids_1d; grids_for_choices{1} = a1prime_grid;
        if num_a1_vars > 1; [mesh_out{1:num_a1_vars}] = ndgrid(grids_for_choices{:}); else; mesh_out{1} = grids_for_choices{1}; end
        num_choices_total = numel(mesh_out{1});
        FLAT_CHOICES = N_d_safe * num_choices_total;

        Apr_flat = cell(1, num_a1_vars);
        for ia = 1:num_a1_vars
            val = reshape(mesh_out{ia}, [num_choices_total, 1]);
            Apr_flat{ia} = repelem(val, N_d_safe, 1);
        end

        D_flat = cell(1, length(D_cells_block));
        for id = 1:length(D_cells_block)
            val = reshape(D_cells_block{id}, [N_d_safe, 1]);
            D_flat{id} = repmat(val, [num_choices_total, 1]);
        end

        choice_idx_linear = reshape(1:num_choices_total, [num_choices_total, 1]);

        if N_a_exp > 1
            F_tensor = TensorReturnFn(D_flat{:}, Apr_flat{:}, A1_flat{:}, A2_flat{:}, Z_flat{:}, E_flat{:}, ReturnFnParamsCell{:});
            A2_prime = TensoraprimeFn(D_flat{:}, A2_flat{:}, Z_flat{:}, E_flat{:}, aprimeFnParamsCell{:});
            a2_grid_1d_vec = a2_grids_1d{1}; a2_prime_clipped = max(a2_grid_1d_vec(1), min(A2_prime, a2_grid_1d_vec(end)));
            idx = discretize(a2_prime_clipped, a2_grid_1d_vec); idx(isnan(idx)) = N_a_exp - 1; idx = max(1, min(idx, N_a_exp - 1));
            a2_left = reshape(a2_grid_1d_vec(idx), size(idx)); a2_right = reshape(a2_grid_1d_vec(idx+1), size(idx));
            weight = (a2_prime_clipped - a2_left) ./ (a2_right - a2_left); weight(a2_right == a2_left) = 0;

            choice_idx_exp = repelem(choice_idx_linear, N_d_safe, 1);
            idx_left  = choice_idx_exp + (idx - 1) * (length(a1prime_grid) * N_a2_endo) + (ZE_idx_flat - 1) * (length(a1prime_grid) * N_a2_endo * max(1, N_a_exp));
            idx_right = choice_idx_exp + (idx) * (length(a1prime_grid) * N_a2_endo) + (ZE_idx_flat - 1) * (length(a1prime_grid) * N_a2_endo * max(1, N_a_exp));

            if N_dsemiz > 1
                dsemiz_flat = reshape(dsemiz_idx_tensor, [N_d_safe, 1]);
                dsemiz_expanded = repmat(dsemiz_flat, [num_choices_total, 1]);
                idx_left = idx_left + (dsemiz_expanded - 1) * (length(a1prime_grid) * N_a2_endo * max(1, N_a_exp) * N_ze_local);
                idx_right = idx_right + (dsemiz_expanded - 1) * (length(a1prime_grid) * N_a2_endo * max(1, N_a_exp) * N_ze_local);
            end

            EV_left = EV_interp_local(idx_left); EV_right = EV_interp_local(idx_right);
            EV_bounded = EV_left + weight .* (EV_right - EV_left);
            clear EV_left EV_right;

            EV_bounded(weight == 0) = EV_local(idx_left(weight == 0)); EV_bounded(weight == 1) = EV_local(idx_right(weight == 1));
            EV_bounded(isnan(EV_bounded)) = -Inf; EV_bounded = beta_j .* EV_bounded;
        else
            F_tensor = TensorReturnFn(D_flat{:}, Apr_flat{:}, A1_flat{:}, Z_flat{:}, E_flat{:}, ReturnFnParamsCell{:});
            stride_z = length(a1prime_grid) * N_a2_endo;
            ze_base = reshape((0:n_z_loc-1)' * stride_z + (0:n_e_loc-1) * stride_z * n_z_loc, [1, N_ze_local]);
            ze_expanded = reshape(repmat(ze_base, [N_states, 1]), [1, FLAT_STATES]);

            choice_idx_exp = repelem(choice_idx_linear, N_d_safe, 1);
            L2_linear_idx = choice_idx_exp + ze_expanded;

            if N_dsemiz > 1
                dsemiz_flat = reshape(dsemiz_idx_tensor, [N_d_safe, 1]);
                dsemiz_expanded = repmat(dsemiz_flat, [num_choices_total, 1]);
                L2_linear_idx = L2_linear_idx + (dsemiz_expanded - 1) * (stride_z * N_ze_local);
            end

            EV_bounded = beta_j .* EV_interp_local(L2_linear_idx);
        end

        RHS = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bounded, 1, 1, ezc2_j, ezc3, ezc4, ezc7_j);
        clear F_tensor EV_bounded; % Memory Hoist

        [V_sub_fine, Pol_sub_idx] = max(RHS, [], 1);
        Pol_a1_per_a2 = [];
        d_idx_local = mod(Pol_sub_idx - 1, max(1, N_d_safe)) + 1; apr_offset  = ceil(Pol_sub_idx / max(1, N_d_safe));
        V_j_max   = reshape(V_sub_fine,  [N_states, N_ze_local]); Pol_d_max = reshape(d_idx_local, [N_states, N_ze_local]);

        a1_apr_offset = mod(apr_offset - 1, length(a1prime_grid)) + 1;
        a2_offset_factor = ceil(apr_offset / length(a1prime_grid));

        Pol_apr_max = floor((a1_apr_offset - 1) / (n2short + 1)) + 1; Pol_apr_max = min(Pol_apr_max, N_a1_dc - 1);
        Pol_L2idx_max = a1_apr_offset - (Pol_apr_max - 1) * (n2short + 1);
        Pol_apr_max = Pol_apr_max + (a2_offset_factor - 1) * N_a1_dc;

        Pol_apr_max    = reshape(Pol_apr_max, [N_states, N_ze_local]); Pol_L2idx_max  = reshape(Pol_L2idx_max, [N_states, N_ze_local]);
        Pol_L2flag_max = 2 * ones(1, FLAT_STATES, 'like', V_j_max);

        idx_lower_coarse = (a1_apr_offset(:)' - 1) * (n2short + 1) + 1;
        idx_upper_coarse = min(length(a1prime_grid), idx_lower_coarse + (n2short + 1));

        row_lower = d_idx_local + (idx_lower_coarse - 1) * N_d_safe + (a2_offset_factor - 1) * (length(a1prime_grid) * N_d_safe);
        row_upper = d_idx_local + (idx_upper_coarse - 1) * N_d_safe + (a2_offset_factor - 1) * (length(a1prime_grid) * N_d_safe);
        lin_lower = row_lower + (0:FLAT_STATES-1) * FLAT_CHOICES;
        lin_upper = row_upper + (0:FLAT_STATES-1) * FLAT_CHOICES;

        isInfLower = (RHS(lin_lower) == -Inf); isInfUpper = (RHS(lin_upper) == -Inf);
        clear RHS; % Memory Hoist

        isInnerOrUpper = (Pol_L2idx_max(:)' > 1); isInnerOrLower = (Pol_L2idx_max(:)' < n2short + 2);
        Pol_L2flag_max(isInnerOrUpper & isInfLower) = 3; Pol_L2flag_max(isInnerOrLower & isInfUpper) = 1;
        Pol_L2flag_max = reshape(Pol_L2flag_max, [N_states, N_ze_local]);
    end
else
    % =================================================================
    % BRANCH 2: ZOOM PHASE (2D)
    % =================================================================
    Pol_a1_per_a2 = [];

    a1_idx = mod(loweredge_matrix - 1, N_a1_dc) + 1;
    if numel(a1_idx) == N_states * N_ze_local
        loweredge_matrix_2d = repmat(reshape(a1_idx, [1, FLAT_STATES]), [N_a2_endo, 1]);
    elseif numel(a1_idx) == N_a2_endo * N_states * N_ze_local
        loweredge_matrix_2d = reshape(a1_idx, [N_a2_endo, FLAT_STATES]);
    else
        a1_idx = reshape(a1_idx, [1, N_states / max(1, N_a_exp), n_z_loc, n_e_loc]);
        loweredge_matrix_2d = reshape(repmat(a1_idx, [N_a2_endo, max(1, N_a_exp), 1, 1]), [N_a2_endo, FLAT_STATES]);
    end

    if gridinterplayer(1) == 0 || is_dc_mode == 2
        % -------------------------------------------------------------
        % SCENARIO 2A: Standard DC Segment Zoom
        % -------------------------------------------------------------
        num_choices_total = (maxgap_scalar + 1) * N_a2_endo;
        FLAT_CHOICES = N_d_safe * num_choices_total;

        offsets_a1 = reshape(0:maxgap_scalar, [maxgap_scalar + 1, 1]);
        loweredge_expanded = repelem(loweredge_matrix_2d, maxgap_scalar + 1, 1);
        offsets_expanded = repmat(offsets_a1, [N_a2_endo, FLAT_STATES]);

        choice_idx_a1 = max(1, min(loweredge_expanded + offsets_expanded, length(A1_grids_1d{1})));
        a2_base = repelem((1:N_a2_endo)', maxgap_scalar + 1, 1);
        choice_idx_a2 = repmat(a2_base, [1, FLAT_STATES]);

        Apr_flat = cell(1, num_a1_vars);
        Apr_flat{1} = repelem(A1_grids_1d{1}(choice_idx_a1), N_d_safe, 1);

        if num_a1_vars > 2; [mesh_a2{1:num_a1_vars-1}] = ndgrid(A1_grids_1d{2:end}); else; mesh_a2{1} = A1_grids_1d{2}; end
        for ia = 2:num_a1_vars
            flat_grid = mesh_a2{ia-1}(:);
            Apr_flat{ia} = repelem(flat_grid(choice_idx_a2), N_d_safe, 1);
        end

        choice_idx_linear = choice_idx_a1 + (choice_idx_a2 - 1) * length(A1_grids_1d{1});

        D_flat = cell(1, length(D_cells_block));
        for id = 1:length(D_cells_block)
            val = reshape(D_cells_block{id}, [N_d_safe, 1]);
            D_flat{id} = repmat(val, [num_choices_total, FLAT_STATES]);
        end

        if N_a_exp > 1
            F_tensor = TensorReturnFn(D_flat{:}, Apr_flat{:}, A1_flat{:}, A2_flat{:}, Z_flat{:}, E_flat{:}, ReturnFnParamsCell{:});
            A2_prime = TensoraprimeFn(D_flat{:}, A2_flat{:}, Z_flat{:}, E_flat{:}, aprimeFnParamsCell{:});
            a2_grid_1d_vec = a2_grids_1d{1}; a2_prime_clipped = max(a2_grid_1d_vec(1), min(A2_prime, a2_grid_1d_vec(end)));
            idx = discretize(a2_prime_clipped, a2_grid_1d_vec); idx(isnan(idx)) = N_a_exp - 1; idx = max(1, min(idx, N_a_exp - 1));
            a2_left = reshape(a2_grid_1d_vec(idx), size(idx)); a2_right = reshape(a2_grid_1d_vec(idx+1), size(idx));
            weight = (a2_prime_clipped - a2_left) ./ (a2_right - a2_left); weight(a2_right == a2_left) = 0;

            N_a2_global = max(1, prod(cellfun(@length, a2_grids_1d)));
            choice_idx_exp = repelem(choice_idx_linear, N_d_safe, 1);
            idx_left  = choice_idx_exp + (idx - 1) * (N_a1_dc * N_a2_endo) + (ZE_idx_flat - 1) * (N_a1_dc * N_a2_endo * N_a2_global);
            idx_right = choice_idx_exp + (idx) * (N_a1_dc * N_a2_endo) + (ZE_idx_flat - 1) * (N_a1_dc * N_a2_endo * N_a2_global);

            max_idx_row = size(EV_local, 1);
            dsemiz_flat = reshape(dsemiz_idx_tensor, [N_d_safe, 1]);
            dsemiz_expanded = repmat(dsemiz_flat, [num_choices_total, FLAT_STATES]);
            linear_idx_left  = min(max_idx_row, max(1, idx_left  + (dsemiz_expanded - 1) * max_idx_row));
            linear_idx_right = min(max_idx_row, max(1, idx_right + (dsemiz_expanded - 1) * max_idx_row));

            EV_bounded = EV_local(linear_idx_left) + weight .* (EV_local(linear_idx_right) - EV_local(linear_idx_left));
            EV_bounded(weight == 0) = EV_local(linear_idx_left(weight == 0)); EV_bounded(weight == 1) = EV_local(linear_idx_right(weight == 1));
            EV_bounded(isnan(EV_bounded)) = -Inf; EV_bounded = beta_j .* EV_bounded;
        else
            F_tensor = TensorReturnFn(D_flat{:}, Apr_flat{:}, A1_flat{:}, Z_flat{:}, E_flat{:}, ReturnFnParamsCell{:});

            stride_z = N_d_safe * N_a1_dc * N_a2_endo;
            ze_base = reshape((0:n_z_loc-1)' * stride_z + (0:n_e_loc-1) * stride_z * n_z_loc, [1, N_ze_local]);
            ze_expanded = reshape(repmat(ze_base, [N_states, 1]), [1, FLAT_STATES]);

            d_expanded = repmat((1:N_d_safe)', [num_choices_total, FLAT_STATES]);
            choice_idx_exp = repelem(choice_idx_linear, N_d_safe, 1);
            linear_idx_EV = d_expanded + (choice_idx_exp - 1) * N_d_safe + ze_expanded;
            EV_bounded = EV_bounded_pre(linear_idx_EV);
        end
    else
        % -------------------------------------------------------------
        % SCENARIO 2B: Grid Interpolation Zoom
        % -------------------------------------------------------------
        loweredge_matrix_bounds = max(2, min(loweredge_matrix_2d, length(A1_grids_1d{1}) - 1));
        L2_base = (loweredge_matrix_bounds - 1) * (n2short + 1) + 1;

        num_choices_total = n2long * N_a2_endo;
        FLAT_CHOICES = N_d_safe * num_choices_total;

        start_offset = -(n2short + 1); end_offset = (n2short + 1);
        offsets_a1 = reshape(start_offset:end_offset, [n2long, 1]);

        L2_base_expanded = repelem(L2_base, n2long, 1);
        offsets_expanded = repmat(offsets_a1, [N_a2_endo, FLAT_STATES]);

        raw_choice_idx_a1 = L2_base_expanded + offsets_expanded;
        out_of_bounds = (raw_choice_idx_a1 < 1) | (raw_choice_idx_a1 > length(a1prime_grid));
        choice_idx_a1 = max(1, min(raw_choice_idx_a1, length(a1prime_grid)));

        a2_base = repelem((1:N_a2_endo)', n2long, 1);
        choice_idx_a2 = repmat(a2_base, [1, FLAT_STATES]);

        Apr_flat = cell(1, num_a1_vars);
        Apr_flat{1} = repelem(a1prime_grid(choice_idx_a1), N_d_safe, 1);

        if num_a1_vars > 2; [mesh_a2{1:num_a1_vars-1}] = ndgrid(A1_grids_1d{2:end}); else; mesh_a2{1} = A1_grids_1d{2}; end
        for ia = 2:num_a1_vars
            flat_grid = mesh_a2{ia-1}(:);
            Apr_flat{ia} = repelem(flat_grid(choice_idx_a2), N_d_safe, 1);
        end

        choice_idx_linear = choice_idx_a1 + (choice_idx_a2 - 1) * length(a1prime_grid);

        D_flat = cell(1, length(D_cells_block));
        for id = 1:length(D_cells_block)
            val = reshape(D_cells_block{id}, [N_d_safe, 1]);
            D_flat{id} = repmat(val, [num_choices_total, FLAT_STATES]);
        end

        if N_a_exp > 1
            F_tensor = TensorReturnFn(D_flat{:}, Apr_flat{:}, A1_flat{:}, A2_flat{:}, Z_flat{:}, E_flat{:}, ReturnFnParamsCell{:});
            A2_prime = TensoraprimeFn(D_flat{:}, A2_flat{:}, Z_flat{:}, E_flat{:}, aprimeFnParamsCell{:});
            a2_grid_1d_vec = a2_grids_1d{1}; a2_prime_clipped = max(a2_grid_1d_vec(1), min(A2_prime, a2_grid_1d_vec(end)));
            idx = discretize(a2_prime_clipped, a2_grid_1d_vec); idx(isnan(idx)) = N_a_exp - 1; idx = max(1, min(idx, N_a_exp - 1));
            a2_left = reshape(a2_grid_1d_vec(idx), size(idx)); a2_right = reshape(a2_grid_1d_vec(idx+1), size(idx));
            weight = (a2_prime_clipped - a2_left) ./ (a2_right - a2_left); weight(a2_right == a2_left) = 0;

            choice_idx_exp = repelem(choice_idx_linear, N_d_safe, 1);
            idx_left  = choice_idx_exp + (idx - 1) * (length(a1prime_grid) * N_a2_endo) + (ZE_idx_flat - 1) * (length(a1prime_grid) * N_a2_endo * max(1, N_a_exp));
            idx_right = choice_idx_exp + (idx) * (length(a1prime_grid) * N_a2_endo) + (ZE_idx_flat - 1) * (length(a1prime_grid) * N_a2_endo * max(1, N_a_exp));

            if N_dsemiz > 1
                dsemiz_flat = reshape(dsemiz_idx_tensor, [N_d_safe, 1]);
                dsemiz_expanded = repmat(dsemiz_flat, [num_choices_total, FLAT_STATES]);
                idx_left = idx_left + (dsemiz_expanded - 1) * (length(a1prime_grid) * N_a2_endo * max(1, N_a_exp) * N_ze_local);
                idx_right = idx_right + (dsemiz_expanded - 1) * (length(a1prime_grid) * N_a2_endo * max(1, N_a_exp) * N_ze_local);
            end

            EV_left = EV_interp_local(idx_left); EV_right = EV_interp_local(idx_right);
            EV_bounded = EV_left + weight .* (EV_right - EV_left);
            clear EV_left EV_right;

            EV_bounded(weight == 0) = EV_local(idx_left(weight == 0)); EV_bounded(weight == 1) = EV_local(idx_right(weight == 1));
            out_of_bounds_flat = repelem(out_of_bounds, N_d_safe, 1);
            EV_bounded(out_of_bounds_flat) = -Inf; EV_bounded(isnan(EV_bounded)) = -Inf; EV_bounded = beta_j .* EV_bounded;
        else
            F_tensor = TensorReturnFn(D_flat{:}, Apr_flat{:}, A1_flat{:}, Z_flat{:}, E_flat{:}, ReturnFnParamsCell{:});

            stride_z = length(a1prime_grid) * N_a2_endo;
            ze_base = reshape((0:n_z_loc-1)' * stride_z + (0:n_e_loc-1) * stride_z * n_z_loc, [1, N_ze_local]);
            ze_expanded = reshape(repmat(ze_base, [N_states, 1]), [1, FLAT_STATES]);

            choice_idx_exp = repelem(choice_idx_linear, N_d_safe, 1);
            L2_linear_idx = choice_idx_exp + ze_expanded;

            if N_dsemiz > 1
                dsemiz_flat = reshape(dsemiz_idx_tensor, [N_d_safe, 1]);
                dsemiz_expanded = repmat(dsemiz_flat, [num_choices_total, FLAT_STATES]);
                L2_linear_idx = L2_linear_idx + (dsemiz_expanded - 1) * (stride_z * N_ze_local);
            end

            EV_bounded = beta_j .* EV_interp_local(L2_linear_idx);
            out_of_bounds_flat = repelem(out_of_bounds, N_d_safe, 1);
            EV_bounded(out_of_bounds_flat) = -Inf;
        end
    end

    RHS = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bounded, 1, 1, ezc2_j, ezc3, ezc4, ezc7_j);
    clear F_tensor EV_bounded; % Memory Hoist

    [V_sub_fine, Pol_sub_idx] = max(RHS, [], 1);

    if nargout > 5
        RHS_for_d = reshape(RHS, [N_d_safe, num_choices_total, FLAT_STATES]);
        RHS_max_d = max(RHS_for_d, [], 1);
        clear RHS_for_d;

        if gridinterplayer(1) == 0 || is_dc_mode == 2
            RHS_4D = reshape(RHS_max_d, [maxgap_scalar + 1, N_a2_endo, FLAT_STATES]);
        else
            RHS_4D = reshape(RHS_max_d, [n2long, N_a2_endo, FLAT_STATES]);
        end
        clear RHS_max_d;

        [~, max_a1_idx_rel] = max(RHS_4D, [], 1);
        clear RHS_4D;

        max_a1_idx_rel = reshape(max_a1_idx_rel, [N_a2_endo, FLAT_STATES]);
        Pol_a1_per_a2 = min(loweredge_matrix_2d + max_a1_idx_rel - 1, N_a1_dc);
        Pol_a1_per_a2 = reshape(Pol_a1_per_a2, [N_a2_endo, N_states, N_ze_local]);
    end

    d_idx_local = mod(Pol_sub_idx - 1, max(1, N_d_safe)) + 1;
    apr_offset  = ceil(Pol_sub_idx / max(1, N_d_safe));
    V_j_max   = reshape(V_sub_fine,  [N_states, N_ze_local]);
    Pol_d_max = reshape(d_idx_local, [N_states, N_ze_local]);

    if gridinterplayer(1) == 0 || is_dc_mode == 2
        a1_apr_offset = mod(apr_offset - 1, maxgap_scalar + 1) + 1;
        a2_offset_factor = ceil(apr_offset / (maxgap_scalar + 1));

        lin_idx_loweredge = a2_offset_factor + (0:FLAT_STATES-1) * N_a2_endo;
        chosen_loweredge = loweredge_matrix_2d(lin_idx_loweredge);

        a1_Pol_apr = chosen_loweredge + a1_apr_offset - 1;
        Pol_apr_max = a1_Pol_apr + (a2_offset_factor - 1) * N_a1_dc;
        Pol_apr_max = reshape(Pol_apr_max, [N_states, N_ze_local]);
        Pol_L2idx_max = []; Pol_L2flag_max = [];
    else
        a1_apr_offset = mod(apr_offset - 1, n2long) + 1;
        a2_offset_factor = ceil(apr_offset / n2long);
        chosen_offset = start_offset + a1_apr_offset - 1;

        lin_idx_loweredge = a2_offset_factor + (0:FLAT_STATES-1) * N_a2_endo;
        chosen_loweredge = loweredge_matrix_2d(lin_idx_loweredge);

        abs_fine_idx_flat = (chosen_loweredge - 1) * (n2short + 1) + 1 + chosen_offset;
        a1_Pol_apr = floor((abs_fine_idx_flat - 1) / (n2short + 1)) + 1;
        a1_Pol_apr = min(a1_Pol_apr, N_a1_dc - 1);
        Pol_L2idx_max = abs_fine_idx_flat - (a1_Pol_apr - 1) * (n2short + 1);

        Pol_apr_max = a1_Pol_apr + (a2_offset_factor - 1) * N_a1_dc;
        Pol_apr_max = reshape(Pol_apr_max, [N_states, N_ze_local]);
        Pol_L2idx_max = reshape(Pol_L2idx_max, [N_states, N_ze_local]);

        idx_lower_coarse = (a1_apr_offset(:)' - 1) * (n2short + 1) + 1;
        idx_upper_coarse = min(length(a1prime_grid), idx_lower_coarse + (n2short + 1));
        % The RHS matrix in Zoom Phase only has n2long asset choices.
        % The lower bound is local index 1, the upper bound is local index n2long.
        row_lower = d_idx_local + (1 - 1) * N_d_safe + (a2_offset_factor - 1) * (n2long * N_d_safe);
        row_upper = d_idx_local + (n2long - 1) * N_d_safe + (a2_offset_factor - 1) * (n2long * N_d_safe);

        lin_lower = row_lower + (0:FLAT_STATES-1) * FLAT_CHOICES;
        lin_upper = row_upper + (0:FLAT_STATES-1) * FLAT_CHOICES;

        isInfLower = (RHS(lin_lower) == -Inf);
        isInfUpper = (RHS(lin_upper) == -Inf);
        clear RHS; % Memory Hoist

        inLowerStrict = (a1_apr_offset >= 2) & (a1_apr_offset <= n2short + 1);
        inUpperStrict = (a1_apr_offset >= n2short + 3) & (a1_apr_offset <= n2long - 1);

        Pol_L2flag_max = 2 * ones(1, FLAT_STATES, 'like', V_j_max);
        Pol_L2flag_max(inLowerStrict & isInfLower) = 3;
        Pol_L2flag_max(inUpperStrict & isInfUpper) = 1;
        Pol_L2flag_max = reshape(Pol_L2flag_max, [N_states, N_ze_local]);
    end
end


end