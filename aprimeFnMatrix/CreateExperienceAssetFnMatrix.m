function [a2primeIndexes,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d, n_a2, d_gridvals, a2_grid, aprimeFnParams, aprimeIndexAsColumn)
% For experienceasset: enumerate a2prime=aprimeFn(d, a2) over ALL d (used
% during value-function iteration). Because the true value of a2prime will
% (almost always) lie between two consecutive points in a2_grid, it is
% linearly interpolated back on to a2_grid. Thus the continuous a2prime is
% represented by (index of lower grid point in a2primeIndexes, probability
% of lower grid point in a2primeProbs) on a2_grid; the upper index is
% implicitly lower+1 with prob 1-minus-prob-of-lower.
%
% Output sizes:
%   l_a2==1 (legacy):
%     a2primeIndexes - col=1 => [N_d*N_a2, 1]; col=2 => [N_d, N_a2]
%     a2primeProbs   - [N_d, N_a2]; upper idx = lower+1, prob upper = 1-prob lower
%   l_a2==2 (multi-dim, Kaprimepts=4 corners):
%     a2primeIndexes - col=1 => [Kaprimepts, N_d*N_a2]; col=2 => [Kaprimepts, N_d, N_a2]
%     a2primeProbs   - same shape as a2primeIndexes
%     Each row c=1..Kaprimepts is one corner of the bilinear interpolation lattice;
%     the index is the Kron'd linear index in N_a2=prod(n_a2) space.

ParamCell=cell(length(aprimeFnParams),1);
for ii=1:length(aprimeFnParams)
    if size(aprimeFnParams(ii))~=[1,1]
        error('Using experienceasset does not allow for any of aprimeFn parameters to be anything but a scalar')
    end
    ParamCell(ii,1)={aprimeFnParams(ii)};
end

N_d=prod(n_d);
N_a2=prod(n_a2);

l_d=length(n_d);
if N_d==0
    l_d=0;
end
l_a2=length(n_a2);
if l_d>4
    error('experienceasset does not allow for more than four of d variable (you have length(n_d)>4)')
end
if l_a2>2
    error('experienceasset currently supports length(n_a2) in {1,2}')
end

if nargin(aprimeFn)~=l_d+l_a2+(l_a2>=2)+length(aprimeFnParams)
    % When l_a2>=2, aprimeFn takes an extra 'whicha' integer selector slot after the a2 inputs.
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

if l_a2==1
    a2vals=shiftdim(a2_grid(1:n_a2(1)),-1);

    if l_d==1
        a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, ParamCell{:});
    elseif l_d==2
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, ParamCell{:});
    elseif l_d==3
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, ParamCell{:});
    elseif l_d==4
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, ParamCell{:});
    end


    %% Calcuate grid indexes and probs from the values
    a2primeVals=reshape(a2primeVals,[1,N_d*N_a2]);

    a2_griddiff=a2_grid(2:end)-a2_grid(1:end-1); % Distance between point and the next point

    % For small aprimeVals and a_grid, max() is faster than discretize()
    if N_d*N_a2*N_a2<1000000
        [~,a2primeIndexes]=max((a2_grid>a2primeVals),[],1);
        a2primeIndexes=a2primeIndexes-1;
        a2primeIndexes(a2primeIndexes==0)=1;
        a2primeIndexes=reshape(a2primeIndexes,[N_d*N_a2,1]);

        aprime_residual=a2primeVals'-a2_grid(a2primeIndexes);
        a2primeProbs=1-aprime_residual./a2_griddiff(a2primeIndexes);

        offTopOfGrid=(a2primeVals>=a2_grid(end));
        a2primeIndexes(offTopOfGrid)=n_a2-1;
        a2primeProbs(offTopOfGrid)=0;
        offBottomOfGrid=(a2primeVals<=a2_grid(1));
        a2primeProbs(offBottomOfGrid)=1;
    else
        a2primeIndexes=discretize(a2primeVals,a2_grid);
        offBottomOfGrid=(a2primeVals<=a2_grid(1));
        a2primeIndexes(offBottomOfGrid)=1;
        offTopOfGrid=(a2primeVals>=a2_grid(end));
        a2primeIndexes(offTopOfGrid)=n_a2-1;
        aprime_residual=a2primeVals'-a2_grid(a2primeIndexes);
        a2primeProbs=1-aprime_residual./a2_griddiff(a2primeIndexes);
        a2primeProbs(offBottomOfGrid)=1;
        a2primeProbs(offTopOfGrid)=0;
    end

    if aprimeIndexAsColumn==1 % value fn codes want column when no z
        a2primeIndexes=a2primeIndexes';
    else % aprimeIndexAsColumn==2 % value fn with z, or simulation, want matrix
        a2primeIndexes=reshape(a2primeIndexes,[N_d,N_a2]);
    end
    a2primeProbs=reshape(a2primeProbs,[N_d,N_a2]);

elseif l_a2==2
    %% Multi-dim a2 (l_a2=2): bilinear interp, Kaprimepts=4 corners
    n_a2_1=n_a2(1); n_a2_2=n_a2(2);
    a2_grid_1=a2_grid(1:n_a2_1);
    a2_grid_2=a2_grid(n_a2_1+1:n_a2_1+n_a2_2);
    a2_1_vals=shiftdim(a2_grid_1,-1); % dim 2
    a2_2_vals=shiftdim(a2_grid_2,-2); % dim 3

    % GPU arrayfun requires scalar output; call once per a2 dim with whicha selector
    if l_d==1
        a2primeVals_1=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, 1, ParamCell{:});
        a2primeVals_2=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, 2, ParamCell{:});
    elseif l_d==2
        a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, 1, ParamCell{:});
        a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, 2, ParamCell{:});
    elseif l_d==3
        a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, 1, ParamCell{:});
        a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, 2, ParamCell{:});
    elseif l_d==4
        a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, 1, ParamCell{:});
        a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, 2, ParamCell{:});
    end

    [loIdx_1, prob_1]=local_interp1d(a2primeVals_1, a2_grid_1, n_a2_1);
    [loIdx_2, prob_2]=local_interp1d(a2primeVals_2, a2_grid_2, n_a2_2);

    % Per-dim factored output (NOT Kron-folded):
    %   a2primeIndexes(k,:) = lower-grid index in a2_k dim (1..n_a2(k))
    %   a2primeProbs(k,:)   = probability of lower grid point in a2_k dim
    % Caller does nested 2-corner interp with skipinterp at each level (bit-exact when V is flat).
    a2primeIndexes=zeros(l_a2,N_d*N_a2,'gpuArray');
    a2primeProbs=zeros(l_a2,N_d*N_a2,'gpuArray');
    a2primeIndexes(1,:)=loIdx_1(:);
    a2primeIndexes(2,:)=loIdx_2(:);
    a2primeProbs(1,:)=prob_1(:);
    a2primeProbs(2,:)=prob_2(:);

    if aprimeIndexAsColumn==1 % column-flat layout
        % already [l_a2, N_d*N_a2]
    else % aprimeIndexAsColumn==2 % matrix layout
        a2primeIndexes=reshape(a2primeIndexes,[l_a2,N_d,N_a2]);
    end
    a2primeProbs=reshape(a2primeProbs,[l_a2,N_d,N_a2]);
end


end


function [loIdx, prob]=local_interp1d(aprimeVals, grid, n_grid)
% 1D linear-interp: lower-grid index in 1..n_grid and prob of lower point.
% Inputs are flattened internally; outputs are column vectors of length numel(aprimeVals).
apvals=aprimeVals(:);
N=numel(apvals);
griddiff=grid(2:end)-grid(1:end-1);

if N*n_grid<1000000
    [~,upIdx]=max((grid>apvals'),[],1); % [1,N]
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
