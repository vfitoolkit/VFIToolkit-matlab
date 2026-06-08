function [a2primeIndexes,a2primeProbs]=CreateExperienceAssetzeFnMatrix_J(aprimeFn, n_d, n_a2, n_z, n_e, N_j, d_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, aprimeFnParams, aprimeIndexAsColumn) % since a2 is one-dimensional, can be a2_grid or a2_gridvals
% Age-dependent (_J) version of CreateExperienceAssetzeFnMatrix: enumerate
% a2prime=aprimeFn(d, a2, z, e) over ALL d (used during value-function
% iteration).
%
% Companion file CreateaprimePolicyExperienceAssetze_J.m does the same but
% only for the Policy-chosen d, used in simulation / agent-distribution.
%
% Dimension layout: d in dim 1, a2 in dim 2, z in dim 3, e in dim 4, j in dim 5.
% aprimeFnParams is a [N_j, n_params] matrix; each column is shifted via
% shiftdim(...,-4) so j lives in dim 5 during arrayfun.
%
% Output sizes:
%   l_a2==1 (legacy):
%     a2primeIndexes - shape depends on aprimeIndexAsColumn:
%                        1 => matrix [N_d*N_a2, N_z, N_e, N_j]
%                        2 => matrix [N_d, N_a2, N_z, N_e, N_j]
%     a2primeProbs   - [N_d, N_a2, N_z, N_e, N_j]
%   l_a2==2 (multi-dim, per-dim factored):
%     a2primeIndexes - col=1 => [l_a2, N_d*N_a2, N_z, N_e, N_j]
%                      col=2 => [l_a2, N_d, N_a2, N_z, N_e, N_j]
%     a2primeProbs   - matches a2primeIndexes (lower-grid index + prob of lower per dim)
%     Caller does nested 2-corner interp with skipinterp at each level.

ParamCell=cell(size(aprimeFnParams,2),1);
for ii=1:size(aprimeFnParams,2)
    ParamCell(ii,1)={shiftdim(aprimeFnParams(:,ii),-4)}; % j is fifth dimension
end

N_d=prod(n_d);
N_a2=prod(n_a2);
N_z=prod(n_z);
N_e=prod(n_e);

l_d=length(n_d);
if N_d==0
    l_d=0;
end
l_a2=length(n_a2);
if l_d>4
    error('experienceassetze does not allow for more than four of d variable (you have length(n_d)>4)')
end
if l_a2>2
    error('experienceassetze currently supports length(n_a2) in {1,2}')
end
l_z=length(n_z);
l_e=length(n_e);

if nargin(aprimeFn)~=l_d+l_a2+l_z+l_e+(l_a2>=2)+size(aprimeFnParams,2)
    % When l_a2>=2, aprimeFn takes an extra 'whicha' integer selector slot
    % between the z,e inputs and the parameter inputs.
    error('Number of inputs to aprimeFn does not fit with size of aprimeFnParams')
end

if l_d>=1
    d1vals=d_gridvals(:,1);
    if l_d>=2
        d2vals=d_gridvals(:,2);
        if l_d>=3
            d3vals=d_gridvals(:,3);
            if l_d>=4
                d4vals=d_gridvals(:,4);
            end
        end
    end
end
% permute once: [N_z, l_z, N_j] -> [1, 1, N_z, 1, N_j, l_z]; then index final dim per z column
z_gridvals_Jp=permute(z_gridvals_J,[4,5,1,6,3,2]);
if l_z>=1
    z1vals=z_gridvals_Jp(:,:,:,:,:,1);   % [1, 1, N_z, 1, N_j]
    if l_z>=2
        z2vals=z_gridvals_Jp(:,:,:,:,:,2);
        if l_z>=3
            z3vals=z_gridvals_Jp(:,:,:,:,:,3);
            if l_z>=4
                z4vals=z_gridvals_Jp(:,:,:,:,:,4);
            end
        end
    end
end
% permute once: [N_e, l_e, N_j] -> [1, 1, 1, N_e, N_j, l_e]; then index final dim per e column
e_gridvals_Jp=permute(e_gridvals_J,[4,5,6,1,3,2]);
if l_e>=1
    e1vals=e_gridvals_Jp(:,:,:,:,:,1);   % [1, 1, 1, N_e, N_j]
    if l_e>=2
        e2vals=e_gridvals_Jp(:,:,:,:,:,2);
        if l_e>=3
            e3vals=e_gridvals_Jp(:,:,:,:,:,3);
            if l_e>=4
                e4vals=e_gridvals_Jp(:,:,:,:,:,4);
            end
        end
    end
end

zecell={};
if l_z>=1, zecell{end+1}=z1vals; end
if l_z>=2, zecell{end+1}=z2vals; end
if l_z>=3, zecell{end+1}=z3vals; end
if l_z>=4, zecell{end+1}=z4vals; end
if l_e>=1, zecell{end+1}=e1vals; end
if l_e>=2, zecell{end+1}=e2vals; end
if l_e>=3, zecell{end+1}=e3vals; end
if l_e>=4, zecell{end+1}=e4vals; end

if l_a2==1
    a2vals=shiftdim(a2_grid,-1);

    if l_d==1
        a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, zecell{:}, ParamCell{:});
    elseif l_d==2
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, zecell{:}, ParamCell{:});
    elseif l_d==3
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, zecell{:}, ParamCell{:});
    elseif l_d==4
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, zecell{:}, ParamCell{:});
    end

    %% Calcuate grid indexes and probs from the values
    a2primeVals=reshape(a2primeVals,[1,N_d*N_a2*N_z*N_e,N_j]);

    a2_griddiff=a2_grid(2:end)-a2_grid(1:end-1);

    if N_d*N_a2*N_z*N_e*N_j*N_a2<1000000
        [~,a2primeIndexes]=max((a2_grid>a2primeVals),[],1);
        a2primeIndexes=a2primeIndexes-1;
        a2primeIndexes(a2primeIndexes==0)=1;
        a2primeIndexes=reshape(a2primeIndexes,[N_d*N_a2*N_z*N_e,N_j]);

        aprime_residual=reshape(a2primeVals,[N_d*N_a2*N_z*N_e,N_j])-a2_grid(a2primeIndexes);
        a2primeProbs=1-aprime_residual./a2_griddiff(a2primeIndexes);

        offTopOfGrid=(a2primeVals>=a2_grid(end));
        a2primeIndexes(offTopOfGrid)=n_a2-1;
        a2primeProbs(offTopOfGrid)=0;
        offBottomOfGrid=(a2primeVals<=a2_grid(1));
        a2primeProbs(offBottomOfGrid)=1;
    else
        a2primeIndexes=discretize(a2primeVals,a2_grid);
        a2primeIndexes=reshape(a2primeIndexes,[N_d*N_a2*N_z*N_e,N_j]);

        offBottomOfGrid=(a2primeVals<=a2_grid(1));
        a2primeIndexes(offBottomOfGrid)=1;
        offTopOfGrid=(a2primeVals>=a2_grid(end));
        a2primeIndexes(offTopOfGrid)=n_a2-1;

        aprime_residual=reshape(a2primeVals,[N_d*N_a2*N_z*N_e,N_j])-a2_grid(a2primeIndexes);
        a2primeProbs=1-aprime_residual./a2_griddiff(a2primeIndexes);
        a2primeProbs(offBottomOfGrid)=1;
        a2primeProbs(offTopOfGrid)=0;
    end

    if aprimeIndexAsColumn==1
        a2primeIndexes=reshape(a2primeIndexes,[N_d*N_a2,N_z,N_e,N_j]);
    else
        a2primeIndexes=reshape(a2primeIndexes,[N_d,N_a2,N_z,N_e,N_j]);
    end
    a2primeProbs=reshape(a2primeProbs,[N_d,N_a2,N_z,N_e,N_j]);

elseif l_a2==2
    %% Multi-dim a2 (l_a2=2): bilinear interp, per-dim factored output
    % Re-layout: d in dim 1, a2_1 in dim 2, a2_2 in dim 3, z in dim 4, e in dim 5, j in dim 6.
    % Existing zecell entries have z in dim 3, e in dim 4, j in dim 5 (l_a2==1 layout).
    % Existing ParamCell entries have j in dim 5. Shift all down by one dim.
    n_a2_1=n_a2(1); n_a2_2=n_a2(2);
    a2_grid_1=a2_grid(1:n_a2_1);
    a2_grid_2=a2_grid(n_a2_1+1:n_a2_1+n_a2_2);
    a2_1_vals=shiftdim(a2_grid_1,-1); % dim 2
    a2_2_vals=shiftdim(a2_grid_2,-2); % dim 3

    for ii=1:numel(zecell)
        zecell{ii}=shiftdim(zecell{ii},-1); % insert leading singleton
    end
    for ii=1:numel(ParamCell)
        ParamCell{ii}=shiftdim(ParamCell{ii},-1); % j moves dim 5 -> dim 6
    end

    % GPU arrayfun requires scalar output; call once per a2 dim with whicha
    % selector. whicha slot is after z,e and before parameters.
    if l_d==1
        a2primeVals_1=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, zecell{:}, 1, ParamCell{:});
        a2primeVals_2=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, zecell{:}, 2, ParamCell{:});
    elseif l_d==2
        a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, zecell{:}, 1, ParamCell{:});
        a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, zecell{:}, 2, ParamCell{:});
    elseif l_d==3
        a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, zecell{:}, 1, ParamCell{:});
        a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, zecell{:}, 2, ParamCell{:});
    elseif l_d==4
        a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, zecell{:}, 1, ParamCell{:});
        a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, zecell{:}, 2, ParamCell{:});
    end
    % a2primeVals_*: shape [N_d, n_a2_1, n_a2_2, N_z, N_e, N_j]; reshape to [N_d*N_a2*N_z*N_e, N_j]
    a2primeVals_1=reshape(a2primeVals_1,[N_d*N_a2*N_z*N_e,N_j]);
    a2primeVals_2=reshape(a2primeVals_2,[N_d*N_a2*N_z*N_e,N_j]);

    [loIdx_1, prob_1]=local_interp1d(a2primeVals_1, a2_grid_1, n_a2_1);
    [loIdx_2, prob_2]=local_interp1d(a2primeVals_2, a2_grid_2, n_a2_2);

    N=N_d*N_a2*N_z*N_e*N_j;
    a2primeIndexes=zeros(l_a2,N,'gpuArray');
    a2primeProbs=zeros(l_a2,N,'gpuArray');
    a2primeIndexes(1,:)=loIdx_1(:);
    a2primeIndexes(2,:)=loIdx_2(:);
    a2primeProbs(1,:)=prob_1(:);
    a2primeProbs(2,:)=prob_2(:);

    if aprimeIndexAsColumn==1
        a2primeIndexes=reshape(a2primeIndexes,[l_a2,N_d*N_a2,N_z,N_e,N_j]);
    else
        a2primeIndexes=reshape(a2primeIndexes,[l_a2,N_d,N_a2,N_z,N_e,N_j]);
    end
    a2primeProbs=reshape(a2primeProbs,[l_a2,N_d,N_a2,N_z,N_e,N_j]);
end


end


function [loIdx, prob]=local_interp1d(aprimeVals, grid, n_grid)
% 1D linear-interp: lower-grid index in 1..n_grid and prob of lower point.
% Inputs are flattened internally; outputs are column vectors of length numel(aprimeVals).
apvals=aprimeVals(:);
N=numel(apvals);
griddiff=grid(2:end)-grid(1:end-1);

if N*n_grid<1000000
    [~,upIdx]=max((grid>apvals'),[],1);
    loIdx=upIdx-1;
    loIdx(loIdx==0)=1;
    loIdx=loIdx(:);
    residual=apvals-grid(loIdx);
    prob=1-residual./griddiff(loIdx);
    offTop=(apvals>=grid(end));
    loIdx(offTop)=n_grid-1;
    prob(offTop)=0;
    offBottom=(apvals<=grid(1));
    prob(offBottom)=1;
else
    loIdx=discretize(apvals,grid);
    loIdx=loIdx(:);
    offBottom=(apvals<=grid(1));
    loIdx(offBottom)=1;
    offTop=(apvals>=grid(end));
    loIdx(offTop)=n_grid-1;
    residual=apvals-grid(loIdx);
    prob=1-residual./griddiff(loIdx);
    prob(offBottom)=1;
    prob(offTop)=0;
end
end
