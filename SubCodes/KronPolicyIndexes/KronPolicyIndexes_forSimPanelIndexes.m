function PolicyKron = KronPolicyIndexes_forSimPanelIndexes(Policy, n_d, n_a, N_z, N_j, simoptions, keep_d)
% Specialized Kron-packer for SimulateTimeSeries/FHorz/SimPanelIndexes/SimPanelIndexes_FHorz.m,
% SimPanelIndexes_FHorz_semiz.m, and SimPanelValues_AggShocks_FHorz_Case1.m.
%
% Input:  Policy  (l_d+l_a [+1 under GI], n_a..., N_z?, N_j)
%         n_d, n_a: standard grid-dim vectors; n_d=0 indicates no d.
%         N_z: scalar — 0 for noz, N_z>0 for with-(z and/or e). Folded inside.
%              For separate z and e, caller passes N_z*N_e and reshapes the
%              output afterwards to split them back into two trailing dims.
%         Under simoptions.gridinterplayer==1, Policy includes both an L2 index
%         row and a trailing PolicyL2flag row; both are kept in the output.
%         keep_d: optional, default 0.
%                 0 → drop d   (output rows: [a_kron]      or [a_kron, L2, L2flag])
%                 1 → keep d   (output rows: [d_kron, a_kron] or [d_kron, a_kron, L2, L2flag])
%
% Output: PolicyKron — row count depends on keep_d and gridinterplayer:
%         non-GI: 1 row (keep_d=0) or 2 rows (keep_d=1; d_kron at row 1)
%         GI:     3 rows (keep_d=0) or 4 rows (keep_d=1); trailing rows are L2_index, L2flag
%         Trailing dims: (..., N_a, N_z, N_j) [or (..., N_a, N_j) if N_z==0]
%
% L2flag is preserved so downstream simulation can apply the force-to-lower/upper
% override at -Inf-neighbour cases.

if ~exist('keep_d','var')
    keep_d = 0;
end

N_a=prod(n_a);
l_a=length(n_a);

if n_d(1)==0
    l_d=0;
    keep_d = 0; % nothing to keep
else
    l_d=length(n_d);
end

% Canonical reshape, separate d and aprime rows
if N_z==0
    Policy=reshape(Policy,[size(Policy,1),N_a,N_j]);
    Policy_d=Policy(1:l_d,:,:);              % empty if l_d==0
    Policy_aprime=Policy(l_d+1:end,:,:);
else
    Policy=reshape(Policy,[size(Policy,1),N_a,N_z,N_j]);
    Policy_d=Policy(1:l_d,:,:,:);
    Policy_aprime=Policy(l_d+1:end,:,:,:);
end

% Trailing-dim shape for the final output
if N_z==0
    tail=[N_a,N_j];
else
    tail=[N_a,N_z,N_j];
end
n_cols=prod(tail);

% Flatten to (rows, n_cols)
Policy_aprime=reshape(Policy_aprime,[size(Policy_aprime,1), n_cols]);
if keep_d
    Policy_d=reshape(Policy_d,[l_d, n_cols]);
end

% Pack a1..al_a into a single Kron index (column-major)
if l_a==1
    a_kron_flat = Policy_aprime(1,:);
else
    a_offset = [0; ones(l_a-1,1,'gpuArray')];
    a_stride = [1; gpuArray(cumprod(n_a(1:end-1)'))];
    a_rows = Policy_aprime(1:l_a,:);
    a_kron_flat = sum((a_rows - a_offset) .* a_stride, 1);
end

% Pack d1..dl_d into a single Kron index (column-major) if keep_d=1
if keep_d
    if l_d==1
        d_kron_flat = Policy_d(1,:);
    else
        d_offset = [0; ones(l_d-1,1,'gpuArray')];
        d_stride = [1; gpuArray(cumprod(n_d(1:end-1)'))];
        d_kron_flat = sum((Policy_d - d_offset) .* d_stride, 1);
    end
end

if simoptions.gridinterplayer==0
    if keep_d
        PolicyKron = reshape([d_kron_flat; a_kron_flat], [2, tail]);
    else
        PolicyKron = reshape(a_kron_flat, [1, tail]);
    end
else
    % Under GI rows of Policy_aprime are [a1, ..., al_a, L2, L2flag]
    L2_flat    = Policy_aprime(l_a+1,:);
    flag_flat  = Policy_aprime(l_a+2,:);
    if keep_d
        PolicyKron = reshape([d_kron_flat; a_kron_flat; L2_flat; flag_flat], [4, tail]);
    else
        PolicyKron = reshape([a_kron_flat; L2_flat; flag_flat], [3, tail]);
    end
end

end
