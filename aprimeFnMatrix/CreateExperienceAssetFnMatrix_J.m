function [a2primeIndexes,a2primeProbs]=CreateExperienceAssetFnMatrix_J(aprimeFn, n_d, n_a2, N_j, d_gridvals, a2_grid, aprimeFnParams, aprimeIndexAsColumn)
% Age-dependent (_J) version of CreateExperienceAssetFnMatrix.
%
% Output sizes:
%   l_a2==1 (legacy):
%     a2primeIndexes - col=1 => [N_d*N_a2, N_j]; col=2 => [N_d, N_a2, N_j]
%     a2primeProbs   - [N_d, N_a2, N_j]
%   l_a2==2 (multi-dim, per-dim factored):
%     a2primeIndexes - col=1 => [l_a2, N_d*N_a2, N_j]; col=2 => [l_a2, N_d, N_a2, N_j]
%       a2primeIndexes(k,...) = lower-grid index in a2_k dim
%     a2primeProbs   - same shape; a2primeProbs(k,...) = prob of lower in a2_k dim
%     Caller does nested 2-corner interp with skipinterp at each level.

ParamCell=cell(size(aprimeFnParams,2),1);
for ii=1:size(aprimeFnParams,2)
    ParamCell(ii,1)={shiftdim(aprimeFnParams(:,ii),-2)}; % j is third dimension
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

if nargin(aprimeFn)~=l_d+l_a2+(l_a2>=2)+size(aprimeFnParams,2)
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
    a2vals=shiftdim(a2_grid,-1);

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
    a2primeVals=reshape(a2primeVals,[1,N_d*N_a2,N_j]);

    a2_griddiff=a2_grid(2:end)-a2_grid(1:end-1);

    if N_d*N_a2*N_j*N_a2<1000000
        [~,a2primeIndexes]=max((a2_grid>a2primeVals),[],1);
        a2primeIndexes=a2primeIndexes-1;
        a2primeIndexes(a2primeIndexes==0)=1;
        a2primeIndexes=reshape(a2primeIndexes,[N_d*N_a2,N_j]);

        aprime_residual=reshape(a2primeVals,[N_d*N_a2,N_j])-a2_grid(a2primeIndexes);
        a2primeProbs=1-aprime_residual./a2_griddiff(a2primeIndexes);

        offTopOfGrid=(a2primeVals>=a2_grid(end));
        a2primeIndexes(offTopOfGrid)=n_a2-1;
        a2primeProbs(offTopOfGrid)=0;
        offBottomOfGrid=(a2primeVals<=a2_grid(1));
        a2primeProbs(offBottomOfGrid)=1;
    else
        a2primeIndexes=discretize(a2primeVals,a2_grid);
        a2primeIndexes=reshape(a2primeIndexes,[N_d*N_a2,N_j]);
        offBottomOfGrid=(a2primeVals<=a2_grid(1));
        a2primeIndexes(offBottomOfGrid)=1;
        offTopOfGrid=(a2primeVals>=a2_grid(end));
        a2primeIndexes(offTopOfGrid)=n_a2-1;
        aprime_residual=reshape(a2primeVals,[N_d*N_a2,N_j])-a2_grid(a2primeIndexes);
        a2primeProbs=1-aprime_residual./a2_griddiff(a2primeIndexes);
        a2primeProbs(offBottomOfGrid)=1;
        a2primeProbs(offTopOfGrid)=0;
    end

    if aprimeIndexAsColumn==1
        a2primeIndexes=reshape(a2primeIndexes,[N_d*N_a2,N_j]);
    else
        a2primeIndexes=reshape(a2primeIndexes,[N_d,N_a2,N_j]);
    end
    a2primeProbs=reshape(a2primeProbs,[N_d,N_a2,N_j]);

elseif l_a2==2
    %% Multi-dim a2 (l_a2=2): bilinear interp, Kaprimepts=4 corners
    n_a2_1=n_a2(1); n_a2_2=n_a2(2);
    a2_grid_1=a2_grid(1:n_a2_1);
    a2_grid_2=a2_grid(n_a2_1+1:n_a2_1+n_a2_2);
    % Lay out: d in dim 1, j in dim 2 (set by shiftdim(...,-2) below), a2_1 in dim 3, a2_2 in dim 4
    a2_1_vals=shiftdim(a2_grid_1,-2); % dim 3
    a2_2_vals=shiftdim(a2_grid_2,-3); % dim 4
    % Note: ParamCell entries are already shiftdim(...,-2) so j sits in dim 3; that's fine because
    % the a2_1 dim 3 is broadcast against j separately by arrayfun position semantics.
    % To avoid a dim collision we instead put j in dim 5 here:
    for ii=1:numel(ParamCell)
        ParamCell{ii}=shiftdim(ParamCell{ii},2); % move j from dim 3 to dim 5
    end

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
    % a2primeVals_*: shape [N_d, 1, n_a2_1, n_a2_2, N_j]; reshape to [N_d*n_a2_1*n_a2_2, N_j] = [N_d*N_a2, N_j]
    a2primeVals_1=reshape(a2primeVals_1,[N_d*N_a2,N_j]);
    a2primeVals_2=reshape(a2primeVals_2,[N_d*N_a2,N_j]);

    %% Per-dim grid indexes and probs (inlined 1D linear-interp; mirrors l_a2==1 above)
    N_total=N_d*N_a2*N_j;
    a2primeVals_1=reshape(a2primeVals_1,[1,N_total]);
    a2primeVals_2=reshape(a2primeVals_2,[1,N_total]);
    a2_griddiff_1=a2_grid_1(2:end)-a2_grid_1(1:end-1);
    a2_griddiff_2=a2_grid_2(2:end)-a2_grid_2(1:end-1);

    % --- a2 dim 1 ---
    if N_total*n_a2_1<1000000
        [~,loIdx_1]=max((a2_grid_1>a2primeVals_1),[],1);
        loIdx_1=loIdx_1-1;
        loIdx_1(loIdx_1==0)=1;
        aprime_residual_1=a2primeVals_1'-a2_grid_1(loIdx_1);
        prob_1=1-aprime_residual_1./a2_griddiff_1(loIdx_1);
        offTopOfGrid_1=(a2primeVals_1>=a2_grid_1(end));
        loIdx_1(offTopOfGrid_1)=n_a2_1-1;
        prob_1(offTopOfGrid_1)=0;
        offBottomOfGrid_1=(a2primeVals_1<=a2_grid_1(1));
        prob_1(offBottomOfGrid_1)=1;
    else
        loIdx_1=discretize(a2primeVals_1,a2_grid_1);
        offBottomOfGrid_1=(a2primeVals_1<=a2_grid_1(1));
        loIdx_1(offBottomOfGrid_1)=1;
        offTopOfGrid_1=(a2primeVals_1>=a2_grid_1(end));
        loIdx_1(offTopOfGrid_1)=n_a2_1-1;
        aprime_residual_1=a2primeVals_1'-a2_grid_1(loIdx_1);
        prob_1=1-aprime_residual_1./a2_griddiff_1(loIdx_1);
        prob_1(offBottomOfGrid_1)=1;
        prob_1(offTopOfGrid_1)=0;
    end

    % --- a2 dim 2 ---
    if N_total*n_a2_2<1000000
        [~,loIdx_2]=max((a2_grid_2>a2primeVals_2),[],1);
        loIdx_2=loIdx_2-1;
        loIdx_2(loIdx_2==0)=1;
        aprime_residual_2=a2primeVals_2'-a2_grid_2(loIdx_2);
        prob_2=1-aprime_residual_2./a2_griddiff_2(loIdx_2);
        offTopOfGrid_2=(a2primeVals_2>=a2_grid_2(end));
        loIdx_2(offTopOfGrid_2)=n_a2_2-1;
        prob_2(offTopOfGrid_2)=0;
        offBottomOfGrid_2=(a2primeVals_2<=a2_grid_2(1));
        prob_2(offBottomOfGrid_2)=1;
    else
        loIdx_2=discretize(a2primeVals_2,a2_grid_2);
        offBottomOfGrid_2=(a2primeVals_2<=a2_grid_2(1));
        loIdx_2(offBottomOfGrid_2)=1;
        offTopOfGrid_2=(a2primeVals_2>=a2_grid_2(end));
        loIdx_2(offTopOfGrid_2)=n_a2_2-1;
        aprime_residual_2=a2primeVals_2'-a2_grid_2(loIdx_2);
        prob_2=1-aprime_residual_2./a2_griddiff_2(loIdx_2);
        prob_2(offBottomOfGrid_2)=1;
        prob_2(offTopOfGrid_2)=0;
    end

    % Per-dim factored output (NOT Kron-folded):
    %   a2primeIndexes(k,:) = lower-grid index in a2_k dim (1..n_a2(k))
    %   a2primeProbs(k,:)   = probability of lower grid point in a2_k dim
    % Caller does nested 2-corner interp with skipinterp at each level (bit-exact when V is flat).
    N=N_d*N_a2*N_j;
    a2primeIndexes=zeros(l_a2,N,'gpuArray');
    a2primeProbs=zeros(l_a2,N,'gpuArray');
    a2primeIndexes(1,:)=loIdx_1(:);
    a2primeIndexes(2,:)=loIdx_2(:);
    a2primeProbs(1,:)=prob_1(:);
    a2primeProbs(2,:)=prob_2(:);

    if aprimeIndexAsColumn==1
        a2primeIndexes=reshape(a2primeIndexes,[l_a2,N_d*N_a2,N_j]);
    else
        a2primeIndexes=reshape(a2primeIndexes,[l_a2,N_d,N_a2,N_j]);
    end
    a2primeProbs=reshape(a2primeProbs,[l_a2,N_d,N_a2,N_j]);
end


end
