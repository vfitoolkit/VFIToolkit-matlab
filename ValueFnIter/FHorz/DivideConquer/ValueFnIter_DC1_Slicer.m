function [V_max, Pol_apr, Pol_d1, Pol_L2idx, Pol_L2flag] = ValueFnIter_DC1_Slicer(N_a1, N_choice, N_a2, N_z, vfoptions, EvalBlockFn)
% Universal CPU Divide-and-Conquer (n-Monotonicity) Slicer
% Accepts a function handle (EvalBlockFn) to evaluate dense tensor blocks

gridinterplayer = isfield(vfoptions, 'gridinterplayer') && vfoptions.gridinterplayer == 1;

% 1. Setup Anchors
level1ii = round(linspace(1, N_a1, vfoptions.level1n));
num_anchors = length(level1ii);

% Preallocate global outputs
V_max   = -inf(N_a1, N_a2, N_z, 'gpuArray');
Pol_apr = ones(N_a1, N_a2, N_z, 'gpuArray');
Pol_d1  = ones(N_a1, N_a2, N_z, 'gpuArray');
if gridinterplayer
    Pol_L2idx  = ones(N_a1, N_a2, N_z, 'gpuArray');
    Pol_L2flag = 2 * ones(N_a1, N_a2, N_z, 'gpuArray');
else
    Pol_L2idx = []; Pol_L2flag = [];
end

% ---------------------------------------------------------
% PHASE 1: The Anchor Pass (Global Survey)
% ---------------------------------------------------------
% Evaluate the full choice grid (1:N_choice) for the anchor states
[V_anch, Pol_apr_anch, Pol_d1_anch, L2idx_anch, L2flag_anch] = EvalBlockFn(level1ii, [], 0); 

V_max(level1ii, :, :)   = V_anch;
Pol_apr(level1ii, :, :) = Pol_apr_anch;
Pol_d1(level1ii, :, :)  = Pol_d1_anch;
if gridinterplayer
    Pol_L2idx(level1ii, :, :)  = L2idx_anch;
    Pol_L2flag(level1ii, :, :) = L2flag_anch;
end

% ---------------------------------------------------------
% PHASE 2: Bounding Logic (maxgap)
% ---------------------------------------------------------
% Find the maximum choice gap across all a2 and z dimensions
maxgap = squeeze(max(max(Pol_apr_anch(2:end, :, :) - Pol_apr_anch(1:end-1, :, :), [], 3), [], 2));

% ---------------------------------------------------------
% PHASE 3: Micro-Batch Dispatch
% ---------------------------------------------------------
for ii = 1:(num_anchors - 1)
    segment_states = (level1ii(ii) + 1) : (level1ii(ii+1) - 1);
    if isempty(segment_states)
        continue; 
    end

    if maxgap(ii) > 0
        % Calculate lower edge strictly avoiding going off the top of the grid
        loweredge = min(Pol_apr_anch(ii, :, :), N_choice - maxgap(ii)); 

        % Dispatch bounded chunk (Engine handles the offset math)
        [V_seg, Pol_apr_seg, Pol_d1_seg, L2idx_seg, L2flag_seg] = EvalBlockFn(segment_states, loweredge, maxgap(ii));
    else
        % Exact choice is known
        loweredge = Pol_apr_anch(ii, :, :);
        [V_seg, Pol_apr_seg, Pol_d1_seg, L2idx_seg, L2flag_seg] = EvalBlockFn(segment_states, loweredge, 0);
    end

    V_max(segment_states, :, :)   = V_seg;
    Pol_apr(segment_states, :, :) = Pol_apr_seg;
    Pol_d1(segment_states, :, :)  = Pol_d1_seg;
    if gridinterplayer
        Pol_L2idx(segment_states, :, :)  = L2idx_seg;
        Pol_L2flag(segment_states, :, :) = L2flag_seg;
    end
end


end