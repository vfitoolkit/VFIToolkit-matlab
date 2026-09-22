function [V_max, Pol_apr, Pol_d1, Pol_L2idx, Pol_L2flag] = ValueFnIter_DC1_Slicer(N_a1_dc, N_other_states, N_choice_a1_dc, N_ze, vfoptions, EvalBlockFn)
% Batched 1D Divide-and-Conquer Slicer (DC1 Architecture)

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
    Pol_L2idx = []; Pol_L2flag = [];
end

% --- PHASE 1: The Anchor Pass ---
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

% --- PHASE 2: 1D Bounding ---
% Isolate the primary asset choice index from the absolute pointer
Pol_a1_idx_anch = mod(Pol_apr(level1ii, :, :) - 1, N_choice_a1_dc) + 1;

maxgap = max(max(Pol_a1_idx_anch(2:end,:,:) - Pol_a1_idx_anch(1:end-1,:,:), [], 3), [], 2);
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
    mapped_anchors = anchor_map(segment_a1_states);

    % Extract the exact lower edge bound for every individual state simultaneously
    loweredge_a1 = Pol_a1_idx_anch(mapped_anchors, :, :); % [num_seg, N_other_states, N_ze]
    loweredge_a1 = reshape(loweredge_a1, [1, num_seg * N_other_states, N_ze]);

    global_maxgap = max(maxgap);
    if isempty(global_maxgap); global_maxgap = 0; end

    % Shift bounds safely downward so the maxgap window doesn't exceed grid size
    loweredge_a1 = min(loweredge_a1, N_choice_a1_dc - global_maxgap);

    seg_state_chunk = segment_a1_states(:) + (0:N_other_states-1) * N_a1_dc;
    seg_state_chunk = seg_state_chunk(:)';

    % --- VRAM Protection: Chunk the massive batched segment pass ---
    flat_choices = global_maxgap + 1;
    max_states_per_chunk = max(1, floor(15000000 / (flat_choices * N_ze)));
    total_seg = length(seg_state_chunk);

    V_seg = []; Pol_apr_seg = []; Pol_d1_seg = []; L2idx_seg = []; L2flag_seg = [];

    for chunk_start = 1:max_states_per_chunk:total_seg
        chunk_end = min(total_seg, chunk_start + max_states_per_chunk - 1);
        c_idx = chunk_start:chunk_end;

        st_chunk = seg_state_chunk(c_idx);
        low_chunk = loweredge_a1(1, c_idx, :); % Extract chunk from 1D bounds

        if global_maxgap > 0
            [v_c, pa_c, pd_c, l2i_c, l2f_c] = EvalBlockFn(st_chunk, low_chunk, global_maxgap);
        else
            [v_c, pa_c, pd_c, l2i_c, l2f_c] = EvalBlockFn(st_chunk, low_chunk, 0);
        end

        V_seg = [V_seg; v_c]; Pol_apr_seg = [Pol_apr_seg; pa_c]; Pol_d1_seg = [Pol_d1_seg; pd_c];
        if gridinterplayer
            L2idx_seg = [L2idx_seg; l2i_c]; L2flag_seg = [L2flag_seg; l2f_c];
        end
    end

    V_max(segment_a1_states, :, :)   = reshape(V_seg, [num_seg, N_other_states, N_ze]);
    Pol_apr(segment_a1_states, :, :) = reshape(Pol_apr_seg, [num_seg, N_other_states, N_ze]);
    Pol_d1(segment_a1_states, :, :)  = reshape(Pol_d1_seg, [num_seg, N_other_states, N_ze]);
    if gridinterplayer
        Pol_L2idx(segment_a1_states, :, :)  = reshape(L2idx_seg, [num_seg, N_other_states, N_ze]);
        Pol_L2flag(segment_a1_states, :, :) = reshape(L2flag_seg, [num_seg, N_other_states, N_ze]);
    end
end

% Flatten back to Slicer output requirements
N_total_states = N_a1_dc * N_other_states;
V_max = reshape(V_max, [N_total_states, N_ze]);
Pol_apr = reshape(Pol_apr, [N_total_states, N_ze]);
Pol_d1 = reshape(Pol_d1, [N_total_states, N_ze]);
if gridinterplayer
    Pol_L2idx = reshape(Pol_L2idx, [N_total_states, N_ze]);
    Pol_L2flag = reshape(Pol_L2flag, [N_total_states, N_ze]);
end


end