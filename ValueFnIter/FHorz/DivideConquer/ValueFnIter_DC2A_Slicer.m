function [V_max, Pol_apr, Pol_d1, Pol_L2idx, Pol_L2flag, Pol_a1_per_a2_out] = ValueFnIter_DC2A_Slicer(N_a1_dc, N_a2_endo, N_other_states, N_choice_a1_dc, N_ze, vfoptions, EvalBlockFn)
% Multi-Axis Divide-and-Conquer Slicer (DC2A Architecture)

gridinterplayer = vfoptions.gridinterplayer(1) == 1;
level1ii = round(linspace(1, N_a1_dc, vfoptions.level1n(1)));
num_anchors = length(level1ii);

V_max   = -inf(N_a1_dc, N_other_states, N_ze, 'gpuArray');
Pol_apr = ones(N_a1_dc, N_other_states, N_ze, 'gpuArray');
Pol_d1  = ones(N_a1_dc, N_other_states, N_ze, 'gpuArray');
Pol_a1_per_a2_out = ones(N_a2_endo, N_a1_dc, N_other_states, N_ze, 'gpuArray');

if gridinterplayer
    Pol_L2idx  = ones(N_a1_dc, N_other_states, N_ze, 'gpuArray');
    Pol_L2flag = 2 * ones(N_a1_dc, N_other_states, N_ze, 'gpuArray');
else
    Pol_L2idx = []; Pol_L2flag = [];
end

% --- PHASE 1: The Anchor Pass ---
anch_state_chunk = level1ii(:) + (0:N_other_states-1) * N_a1_dc;
anch_state_chunk = anch_state_chunk(:)';

[V_anch, Pol_apr_anch, Pol_d1_anch, L2idx_anch, L2flag_anch, Pol_a1_per_a2] = EvalBlockFn(anch_state_chunk, [], 0);

V_max(level1ii, :, :)   = reshape(V_anch, [num_anchors, N_other_states, N_ze]);
Pol_apr(level1ii, :, :) = reshape(Pol_apr_anch, [num_anchors, N_other_states, N_ze]);
Pol_d1(level1ii, :, :)  = reshape(Pol_d1_anch, [num_anchors, N_other_states, N_ze]);
Pol_a1_per_a2_out(:, level1ii, :, :) = reshape(Pol_a1_per_a2, [N_a2_endo, num_anchors, N_other_states, N_ze]);

if gridinterplayer
    Pol_L2idx(level1ii, :, :)  = reshape(L2idx_anch, [num_anchors, N_other_states, N_ze]);
    Pol_L2flag(level1ii, :, :) = reshape(L2flag_anch, [num_anchors, N_other_states, N_ze]);
end

% --- PHASE 2: Conditional Multi-Axis Bounding ---
Pol_a1_per_a2_reshaped = reshape(Pol_a1_per_a2, [N_a2_endo, num_anchors, N_other_states, N_ze]);
Pol_a1_per_a2_reshaped = permute(Pol_a1_per_a2_reshaped, [2, 3, 4, 1]);
Pol_a1_anch_for_gap = permute(Pol_a1_per_a2_reshaped, [1, 4, 2, 3]);

maxgap = max(max(max(Pol_a1_anch_for_gap(2:end,:,:,:) - Pol_a1_anch_for_gap(1:end-1,:,:,:), [], 4), [], 3), [], 2);
maxgap = squeeze(maxgap);
if iscolumn(maxgap); maxgap = maxgap'; end
if isempty(maxgap) && num_anchors == 1; maxgap = 0; end

% --- PHASE 3: Massive Segment Batching ---
anchor_map = zeros(1, N_a1_dc);
for ii = 1:(num_anchors - 1)
    anchor_map((level1ii(ii) + 1) : (level1ii(ii+1) - 1)) = ii;
end

segment_a1_states = find(anchor_map > 0);
num_seg = length(segment_a1_states);

if num_seg > 0
    % Extract bounds for all states simultaneously based on their anchor mapping
    mapped_anchors = anchor_map(segment_a1_states);
    loweredge_a1 = Pol_a1_anch_for_gap(mapped_anchors, :, :, :); % [num_seg, N_a2_endo, N_other_states, N_ze]
    loweredge_a1 = permute(loweredge_a1, [2, 1, 3, 4]); % [N_a2_endo, num_seg, N_other_states, N_ze]
    loweredge_a1 = reshape(loweredge_a1, [N_a2_endo, num_seg * N_other_states, N_ze]);

    global_maxgap = max(maxgap);
    if isempty(global_maxgap); global_maxgap = 0; end

    % Shift bounds downward safely so the maxgap window doesn't exceed grid size
    loweredge_a1 = min(loweredge_a1, N_choice_a1_dc - global_maxgap);

    seg_state_chunk = segment_a1_states(:) + (0:N_other_states-1) * N_a1_dc;
    seg_state_chunk = seg_state_chunk(:)';

    % One massive batched call for every segment state
    if global_maxgap > 0
        [V_seg, Pol_apr_seg, Pol_d1_seg, L2idx_seg, L2flag_seg, Pol_a1_per_a2_seg] = EvalBlockFn(seg_state_chunk, loweredge_a1, global_maxgap);
    else
        [V_seg, Pol_apr_seg, Pol_d1_seg, L2idx_seg, L2flag_seg, Pol_a1_per_a2_seg] = EvalBlockFn(seg_state_chunk, loweredge_a1, 0);
    end

    V_max(segment_a1_states, :, :)   = reshape(V_seg, [num_seg, N_other_states, N_ze]);
    Pol_apr(segment_a1_states, :, :) = reshape(Pol_apr_seg, [num_seg, N_other_states, N_ze]);
    Pol_d1(segment_a1_states, :, :)  = reshape(Pol_d1_seg, [num_seg, N_other_states, N_ze]);
    Pol_a1_per_a2_out(:, segment_a1_states, :, :) = reshape(Pol_a1_per_a2_seg, [N_a2_endo, num_seg, N_other_states, N_ze]);

    if gridinterplayer
        Pol_L2idx(segment_a1_states, :, :)  = reshape(L2idx_seg, [num_seg, N_other_states, N_ze]);
        Pol_L2flag(segment_a1_states, :, :) = reshape(L2flag_seg, [num_seg, N_other_states, N_ze]);
    end
end

N_total_states = N_a1_dc * N_other_states;
V_max = reshape(V_max, [N_total_states, N_ze]);
Pol_apr = reshape(Pol_apr, [N_total_states, N_ze]);
Pol_d1 = reshape(Pol_d1, [N_total_states, N_ze]);
Pol_a1_per_a2_out = reshape(Pol_a1_per_a2_out, [N_a2_endo, N_total_states, N_ze]);

if gridinterplayer
    Pol_L2idx = reshape(Pol_L2idx, [N_total_states, N_ze]);
    Pol_L2flag = reshape(Pol_L2flag, [N_total_states, N_ze]);
end


end