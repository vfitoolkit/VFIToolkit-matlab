function [V_max, Pol_apr, Pol_d1, Pol_L2idx, Pol_L2flag] = ValueFnIter_DC2A_Slicer(N_a1_dc, N_other_states, N_choice_a1_dc, N_ze, vfoptions, EvalBlockFn)
% Multi-Axis Divide-and-Conquer Slicer (DC2A Architecture)
% N_a1_dc: Number of states in the primary DC dimension
% N_other_states: Total size of all secondary discrete endogenous + experience states
% N_choice_a1_dc: Number of primary choices evaluated (n_a1_dc for coarse, n2long for zoom)

gridinterplayer = vfoptions.gridinterplayer(1) == 1;

level1ii = round(linspace(1, N_a1_dc, vfoptions.level1n(1)));
num_anchors = length(level1ii);

V_max   = -inf(N_a1_dc, N_other_states, N_ze, 'gpuArray');
Pol_apr = ones(N_a1_dc, N_other_states, N_ze, 'gpuArray');
Pol_d1  = ones(N_a1_dc, N_other_states, N_ze, 'gpuArray');

if gridinterplayer
    Pol_L2idx  = ones(N_a1_dc, N_other_states, N_ze, 'gpuArray');
    Pol_L2flag = 2 * ones(N_a1_dc, N_other_states, N_ze, 'gpuArray');
else
    Pol_L2idx = [];
    Pol_L2flag = [];
end

% ---------------------------------------------------------
% PHASE 1: The Anchor Pass
% ---------------------------------------------------------
% Generate the absolute Cartesian state coordinates for the anchors across all secondary states
anch_state_chunk = level1ii(:) + (0:N_other_states-1) * N_a1_dc;
anch_state_chunk = anch_state_chunk(:)';

[V_anch, Pol_apr_anch, Pol_d1_anch, L2idx_anch, L2flag_anch] = EvalBlockFn(anch_state_chunk, [], 0);

V_max(level1ii, :, :)   = reshape(V_anch, [num_anchors, N_other_states, N_ze]);
Pol_apr(level1ii, :, :) = reshape(Pol_apr_anch, [num_anchors, N_other_states, N_ze]);
Pol_d1(level1ii, :, :)  = reshape(Pol_d1_anch, [num_anchors, N_other_states, N_ze]);
if gridinterplayer
    Pol_L2idx(level1ii, :, :)  = reshape(L2idx_anch, [num_anchors, N_other_states, N_ze]);
    Pol_L2flag(level1ii, :, :) = reshape(L2flag_anch, [num_anchors, N_other_states, N_ze]);
end

% ---------------------------------------------------------
% PHASE 2: Multi-Axis Bounding Logic
% ---------------------------------------------------------
% Extract ONLY the primary asset (a1) choice index from the absolute fused choice tensor
Pol_a1_idx_anch = mod(Pol_apr(level1ii, :, :) - 1, N_choice_a1_dc) + 1;

% Evaluate maxgap strictly along the a1 state dimension, holding a2 constant
maxgap = squeeze(max(max(Pol_a1_idx_anch(2:end, :, :) - Pol_a1_idx_anch(1:end-1, :, :), [], 3), [], 2));
if iscolumn(maxgap); maxgap = maxgap'; end

% ---------------------------------------------------------
% PHASE 3: Micro-Batch Dispatch
% ---------------------------------------------------------
for ii = 1:(num_anchors - 1)
    segment_a1_states = (level1ii(ii) + 1) : (level1ii(ii+1) - 1);
    if isempty(segment_a1_states); continue; end

    % Generate absolute Cartesian states for this bounded segment
    seg_state_chunk = segment_a1_states(:) + (0:N_other_states-1) * N_a1_dc;
    seg_state_chunk = seg_state_chunk(:)';

    if maxgap(ii) > 0
        loweredge_a1 = min(Pol_a1_idx_anch(ii, :, :), N_choice_a1_dc - maxgap(ii));
        [V_seg, Pol_apr_seg, Pol_d1_seg, L2idx_seg, L2flag_seg] = EvalBlockFn(seg_state_chunk, loweredge_a1, maxgap(ii));
    else
        loweredge_a1 = Pol_a1_idx_anch(ii, :, :);
        [V_seg, Pol_apr_seg, Pol_d1_seg, L2idx_seg, L2flag_seg] = EvalBlockFn(seg_state_chunk, loweredge_a1, 0);
    end

    num_seg = length(segment_a1_states);
    V_max(segment_a1_states, :, :)   = reshape(V_seg, [num_seg, N_other_states, N_ze]);
    Pol_apr(segment_a1_states, :, :) = reshape(Pol_apr_seg, [num_seg, N_other_states, N_ze]);
    Pol_d1(segment_a1_states, :, :)  = reshape(Pol_d1_seg, [num_seg, N_other_states, N_ze]);
    if gridinterplayer
        Pol_L2idx(segment_a1_states, :, :)  = reshape(L2idx_seg, [num_seg, N_other_states, N_ze]);
        Pol_L2flag(segment_a1_states, :, :) = reshape(L2flag_seg, [num_seg, N_other_states, N_ze]);
    end
end

% Flatten back to standard format for the master orchestrator
N_total_states = N_a1_dc * N_other_states;
V_max = reshape(V_max, [N_total_states, N_ze]);
Pol_apr = reshape(Pol_apr, [N_total_states, N_ze]);
Pol_d1 = reshape(Pol_d1, [N_total_states, N_ze]);
if gridinterplayer
    Pol_L2idx = reshape(Pol_L2idx, [N_total_states, N_ze]);
    Pol_L2flag = reshape(Pol_L2flag, [N_total_states, N_ze]);
end


end