function [a2primeIndexes,a2primeProbs]=CreateExperienceAssetsemizFnMatrix(aprimeFn, n_d, n_a2, n_semiz, d_gridvals, a2_grid, semiz_gridvals, aprimeFnParams, aprimeIndexAsColumn)  % since a2 is one-dimensional, can be a2_grid or a2_gridvals
% For experienceassetsemiz: enumerate a2prime=aprimeFn(d, a2, semiz) over ALL d
% (used during value-function iteration). Because the true value of a2prime
% will (almost always) lie between two consecutive points in a2_grid, it is
% linearly interpolated back on to a2_grid. Thus the continuous a2prime is
% represented by (index of lower grid point in a2primeIndexes, probability
% of lower grid point in a2primeProbs) on a2_grid; the upper index is
% implicitly lower+1 with prob 1-minus-prob-of-lower.
%
% Companion file CreateaprimePolicyExperienceAssetsemiz.m does the same but only
% for the Policy-chosen d (one d per state), used in simulation /
% agent-distribution. This file is used during value-function iteration
% where every d must be evaluated.
%
% Output sizes:
%   l_a2==1 (legacy):
%     a2primeIndexes - shape depends on aprimeIndexAsColumn:
%                        1 => column vector [N_d*N_a2*N_semiz, 1]
%                        2 => matrix [N_d, N_a2, N_semiz]
%                        3 => matrix [N_d*N_a2, N_semiz]
%     a2primeProbs   - [N_d, N_a2, N_semiz]
%   l_a2==2 (multi-dim, per-dim factored):
%     a2primeIndexes - col=1 => [l_a2, N_d*N_a2*N_semiz]
%                      col=3 => [l_a2, N_d*N_a2, N_semiz]
%                      col=2 => [l_a2, N_d, N_a2, N_semiz]
%     a2primeProbs   - lower-grid index + prob of lower, per a2 dimension.
%     Caller folds the four corners and does nested 2-corner interp with skipinterp
%     at each level (plus per-contribution isnan cleanup for 0*(-Inf)).

ParamCell=cell(length(aprimeFnParams),1);
for ii=1:length(aprimeFnParams)
    if size(aprimeFnParams(ii))~=[1,1]
        error('Using experienceassetsemiz does not allow for any of aprimeFn parameters to be anything but a scalar')
    end
    ParamCell(ii,1)={aprimeFnParams(ii)};
end

N_d=prod(n_d);
N_a2=prod(n_a2);
N_semiz=prod(n_semiz);

l_d=length(n_d);
if N_d==0
    l_d=0;
end
l_a2=length(n_a2);
if l_d>4
    error('experienceassetsemiz does not allow for more than four of d variable (you have length(n_d)>4)')
end
l_semiz=length(n_semiz);

if l_a2>2
    error('experienceassetsemiz does not allow for more than two experience assets (you have length(n_a2)>2)')
end
if nargin(aprimeFn)~=l_d+l_a2+l_semiz+(l_a2>=2)+length(aprimeFnParams)
    % When l_a2>=2, aprimeFn takes an extra 'whicha' integer selector slot, which sits
    % after semiz and before the parameters: aprimeFn(d,a2_1,a2_2,semiz,whicha,params...)
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
else % l_a2==2: a2_grid is the STACKED grid, split it per dimension
    n_a2_1=n_a2(1); n_a2_2=n_a2(2);
    a2_grid_1=a2_grid(1:n_a2_1);
    a2_grid_2=a2_grid(n_a2_1+1:n_a2_1+n_a2_2);
    a2_1_vals=shiftdim(a2_grid_1,-1); % dim 2
    a2_2_vals=shiftdim(a2_grid_2,-2); % dim 3
end
if l_semiz>=1
    semiz1vals=shiftdim(semiz_gridvals(:,1),-1-l_a2);
    if l_semiz>=2
        semiz2vals=shiftdim(semiz_gridvals(:,2),-1-l_a2);
        if l_semiz>=3
            semiz3vals=shiftdim(semiz_gridvals(:,3),-1-l_a2);
            if l_semiz>=4
                semiz4vals=shiftdim(semiz_gridvals(:,4),-1-l_a2);
                if l_semiz>=5
                    error('Max of four semiz variables supported (contact if you need more)')
                end
            end
        end
    end
end

if l_a2==1
    if l_semiz==1
        if l_d==1
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, semiz1vals, ParamCell{:});
        elseif l_d==2
            a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, semiz1vals, ParamCell{:});
        elseif l_d==3
            a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, semiz1vals, ParamCell{:});
        elseif l_d==4
            a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, semiz1vals, ParamCell{:});
        end
    elseif l_semiz==2
        if l_d==1
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, semiz1vals,semiz2vals, ParamCell{:});
        elseif l_d==2
            a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, semiz1vals,semiz2vals, ParamCell{:});
        elseif l_d==3
            a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, semiz1vals,semiz2vals, ParamCell{:});
        elseif l_d==4
            a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, semiz1vals,semiz2vals, ParamCell{:});
        end
    elseif l_semiz==3
        if l_d==1
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, semiz1vals,semiz2vals,semiz3vals, ParamCell{:});
        elseif l_d==2
            a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, semiz1vals,semiz2vals,semiz3vals, ParamCell{:});
        elseif l_d==3
            a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, semiz1vals,semiz2vals,semiz3vals, ParamCell{:});
        elseif l_d==4
            a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, semiz1vals,semiz2vals,semiz3vals, ParamCell{:});
        end
    elseif l_semiz==4
        if l_d==1
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, semiz1vals,semiz2vals,semiz3vals,semiz4vals, ParamCell{:});
        elseif l_d==2
            a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, semiz1vals,semiz2vals,semiz3vals,semiz4vals, ParamCell{:});
        elseif l_d==3
            a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, semiz1vals,semiz2vals,semiz3vals,semiz4vals, ParamCell{:});
        elseif l_d==4
            a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, semiz1vals,semiz2vals,semiz3vals,semiz4vals, ParamCell{:});
        end
    end
else
    % l_a2==2: GPU arrayfun is scalar-output only, so call aprimeFn once per a2 dimension
    % with the 'whicha' selector (which sits after semiz and before the parameters).
    if l_semiz==1
        if l_d==1
            a2primeVals_1=arrayfun(aprimeFn, d1vals, a2_1_vals,a2_2_vals, semiz1vals, 1, ParamCell{:});
            a2primeVals_2=arrayfun(aprimeFn, d1vals, a2_1_vals,a2_2_vals, semiz1vals, 2, ParamCell{:});
        elseif l_d==2
            a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals,a2_2_vals, semiz1vals, 1, ParamCell{:});
            a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals,a2_2_vals, semiz1vals, 2, ParamCell{:});
        elseif l_d==3
            a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals,a2_2_vals, semiz1vals, 1, ParamCell{:});
            a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals,a2_2_vals, semiz1vals, 2, ParamCell{:});
        elseif l_d==4
            a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals,a2_2_vals, semiz1vals, 1, ParamCell{:});
            a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals,a2_2_vals, semiz1vals, 2, ParamCell{:});
        end
    elseif l_semiz==2
        if l_d==1
            a2primeVals_1=arrayfun(aprimeFn, d1vals, a2_1_vals,a2_2_vals, semiz1vals,semiz2vals, 1, ParamCell{:});
            a2primeVals_2=arrayfun(aprimeFn, d1vals, a2_1_vals,a2_2_vals, semiz1vals,semiz2vals, 2, ParamCell{:});
        elseif l_d==2
            a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals,a2_2_vals, semiz1vals,semiz2vals, 1, ParamCell{:});
            a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals,a2_2_vals, semiz1vals,semiz2vals, 2, ParamCell{:});
        elseif l_d==3
            a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals,a2_2_vals, semiz1vals,semiz2vals, 1, ParamCell{:});
            a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals,a2_2_vals, semiz1vals,semiz2vals, 2, ParamCell{:});
        elseif l_d==4
            a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals,a2_2_vals, semiz1vals,semiz2vals, 1, ParamCell{:});
            a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals,a2_2_vals, semiz1vals,semiz2vals, 2, ParamCell{:});
        end
    elseif l_semiz==3
        if l_d==1
            a2primeVals_1=arrayfun(aprimeFn, d1vals, a2_1_vals,a2_2_vals, semiz1vals,semiz2vals,semiz3vals, 1, ParamCell{:});
            a2primeVals_2=arrayfun(aprimeFn, d1vals, a2_1_vals,a2_2_vals, semiz1vals,semiz2vals,semiz3vals, 2, ParamCell{:});
        elseif l_d==2
            a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals,a2_2_vals, semiz1vals,semiz2vals,semiz3vals, 1, ParamCell{:});
            a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals,a2_2_vals, semiz1vals,semiz2vals,semiz3vals, 2, ParamCell{:});
        elseif l_d==3
            a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals,a2_2_vals, semiz1vals,semiz2vals,semiz3vals, 1, ParamCell{:});
            a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals,a2_2_vals, semiz1vals,semiz2vals,semiz3vals, 2, ParamCell{:});
        elseif l_d==4
            a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals,a2_2_vals, semiz1vals,semiz2vals,semiz3vals, 1, ParamCell{:});
            a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals,a2_2_vals, semiz1vals,semiz2vals,semiz3vals, 2, ParamCell{:});
        end
    elseif l_semiz==4
        if l_d==1
            a2primeVals_1=arrayfun(aprimeFn, d1vals, a2_1_vals,a2_2_vals, semiz1vals,semiz2vals,semiz3vals,semiz4vals, 1, ParamCell{:});
            a2primeVals_2=arrayfun(aprimeFn, d1vals, a2_1_vals,a2_2_vals, semiz1vals,semiz2vals,semiz3vals,semiz4vals, 2, ParamCell{:});
        elseif l_d==2
            a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals,a2_2_vals, semiz1vals,semiz2vals,semiz3vals,semiz4vals, 1, ParamCell{:});
            a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals,a2_2_vals, semiz1vals,semiz2vals,semiz3vals,semiz4vals, 2, ParamCell{:});
        elseif l_d==3
            a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals,a2_2_vals, semiz1vals,semiz2vals,semiz3vals,semiz4vals, 1, ParamCell{:});
            a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals,a2_2_vals, semiz1vals,semiz2vals,semiz3vals,semiz4vals, 2, ParamCell{:});
        elseif l_d==4
            a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals,a2_2_vals, semiz1vals,semiz2vals,semiz3vals,semiz4vals, 1, ParamCell{:});
            a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals,a2_2_vals, semiz1vals,semiz2vals,semiz3vals,semiz4vals, 2, ParamCell{:});
        end
    end
end

%% Calcuate grid indexes and probs from the values
if l_a2==1
    a2primeVals=reshape(a2primeVals,[1,N_d*N_a2*N_semiz]);

    a2_griddiff=a2_grid(2:end)-a2_grid(1:end-1); % Distance between point and the next point

    % For small aprimeVals and a_grid, max() is faster than discretize()
    % http://discourse.vfitoolkit.com/t/example-attanasio-low-sanchez-marcos-2008/257/25
    if N_d*N_a2*N_semiz*N_a2<1000000
        [~,a2primeIndexes]=max((a2_grid>a2primeVals),[],1); % Keep the dimension corresponding to aprimeVals, minimize over the a_grid dimension
        % Note, this is going to find the 'first' grid point which is bigger than a2primeVals
        % This is the 'upper' grid point
        % Have to have special treatment for trying to leave the ends of the grid (I fix these below)

        % Switch to lower grid point index
        a2primeIndexes=a2primeIndexes-1;
        a2primeIndexes(a2primeIndexes==0)=1;

        % Now, find the probabilities
        aprime_residual=a2primeVals'-a2_grid(a2primeIndexes);
        % Probability of the 'lower' points
        a2primeProbs=1-aprime_residual./a2_griddiff(a2primeIndexes);

        % Those points which tried to leave the top of the grid have probability 1 of the 'upper' point (0 of lower point)
        offTopOfGrid=(a2primeVals>=a2_grid(end));
        a2primeIndexes(offTopOfGrid)=n_a2-1; % lower grid point is the one before the end point
        a2primeProbs(offTopOfGrid)=0;
        % Those points which tried to leave the bottom of the grid have probability 0 of the 'upper' point (1 of lower point)
        offBottomOfGrid=(a2primeVals<=a2_grid(1));
        % aprimeIndexes(offBottomOfGrid)=1; % Has already been handled
        a2primeProbs(offBottomOfGrid)=1;
    else
        a2primeIndexes=discretize(a2primeVals,a2_grid); % Finds the lower grid point
        % Have to have special treatment for trying to leave the ends of the grid

        % Those points which tried to leave the bottom of the grid have probability 0 of the 'upper' point (1 of lower point)
        offBottomOfGrid=(a2primeVals<=a2_grid(1));
        a2primeIndexes(offBottomOfGrid)=1; % Has already been handled
        % Those points which tried to leave the top of the grid have probability 1 of the 'upper' point (0 of lower point)
        offTopOfGrid=(a2primeVals>=a2_grid(end));
        a2primeIndexes(offTopOfGrid)=n_a2-1; % lower grid point is the one before the end point

        % Now, find the probabilities
        aprime_residual=a2primeVals'-a2_grid(a2primeIndexes);
        % Probability of the 'lower' points
        a2primeProbs=1-aprime_residual./a2_griddiff(a2primeIndexes);
        % And clean up the ends of the grid
        a2primeProbs(offBottomOfGrid)=1;
        a2primeProbs(offTopOfGrid)=0;
    end

    if aprimeIndexAsColumn==1 % value fn codes want column
%     aprimeIndexes=reshape(aprimeIndexes,[N_d*N_a2*N_semiz,1]);
        a2primeIndexes=a2primeIndexes'; % This is just doing the commented out reshape above
    elseif aprimeIndexAsColumn==3 % value fn with another asset uses 3
        a2primeIndexes=reshape(a2primeIndexes,[N_d*N_a2,N_semiz]);
    else % aprimeIndexAsColumn==2 % value fn codes and simulation codes want matrix
        a2primeIndexes=reshape(a2primeIndexes,[N_d,N_a2,N_semiz]);
    end
    a2primeProbs=reshape(a2primeProbs,[N_d,N_a2,N_semiz]);

elseif l_a2==2
    %% Multi-dim a2 (l_a2=2): bilinear interp, per-dim factored output.
    % All shape contracts match the l_a2==1 path above, with a leading l_a2 dimension.
    a2primeVals_1=reshape(a2primeVals_1,[1,N_d*N_a2*N_semiz]);
    a2primeVals_2=reshape(a2primeVals_2,[1,N_d*N_a2*N_semiz]);
    a2_griddiff_1=a2_grid_1(2:end)-a2_grid_1(1:end-1);
    a2_griddiff_2=a2_grid_2(2:end)-a2_grid_2(1:end-1);

    % --- a2 dim 1 ---
    if N_d*N_a2*N_semiz*n_a2_1<1000000
        [~,loIdx_1]=max((a2_grid_1>a2primeVals_1),[],1);
        loIdx_1=loIdx_1-1;
        loIdx_1(loIdx_1==0)=1;
        loIdx_1=loIdx_1(:); % force column: when n_a2_1==2, a2_griddiff_1 is scalar and would otherwise return the shape of loIdx_1
        aprime_residual_1=a2primeVals_1(:)-a2_grid_1(loIdx_1);
        prob_1=1-aprime_residual_1./a2_griddiff_1(loIdx_1);
        offTopOfGrid_1=(a2primeVals_1(:)>=a2_grid_1(end));
        loIdx_1(offTopOfGrid_1)=n_a2_1-1;
        prob_1(offTopOfGrid_1)=0;
        offBottomOfGrid_1=(a2primeVals_1(:)<=a2_grid_1(1));
        prob_1(offBottomOfGrid_1)=1;
    else
        loIdx_1=discretize(a2primeVals_1,a2_grid_1);
        offBottomOfGrid_1=(a2primeVals_1(:)<=a2_grid_1(1));
        loIdx_1(offBottomOfGrid_1)=1;
        offTopOfGrid_1=(a2primeVals_1(:)>=a2_grid_1(end));
        loIdx_1(offTopOfGrid_1)=n_a2_1-1;
        loIdx_1=loIdx_1(:); % force column (see above)
        aprime_residual_1=a2primeVals_1(:)-a2_grid_1(loIdx_1);
        prob_1=1-aprime_residual_1./a2_griddiff_1(loIdx_1);
        prob_1(offBottomOfGrid_1)=1;
        prob_1(offTopOfGrid_1)=0;
    end

    % --- a2 dim 2 ---
    if N_d*N_a2*N_semiz*n_a2_2<1000000
        [~,loIdx_2]=max((a2_grid_2>a2primeVals_2),[],1);
        loIdx_2=loIdx_2-1;
        loIdx_2(loIdx_2==0)=1;
        loIdx_2=loIdx_2(:); % force column: when n_a2_2==2, a2_griddiff_2 is scalar (see dim 1 note above)
        aprime_residual_2=a2primeVals_2(:)-a2_grid_2(loIdx_2);
        prob_2=1-aprime_residual_2./a2_griddiff_2(loIdx_2);
        offTopOfGrid_2=(a2primeVals_2(:)>=a2_grid_2(end));
        loIdx_2(offTopOfGrid_2)=n_a2_2-1;
        prob_2(offTopOfGrid_2)=0;
        offBottomOfGrid_2=(a2primeVals_2(:)<=a2_grid_2(1));
        prob_2(offBottomOfGrid_2)=1;
    else
        loIdx_2=discretize(a2primeVals_2,a2_grid_2);
        offBottomOfGrid_2=(a2primeVals_2(:)<=a2_grid_2(1));
        loIdx_2(offBottomOfGrid_2)=1;
        offTopOfGrid_2=(a2primeVals_2(:)>=a2_grid_2(end));
        loIdx_2(offTopOfGrid_2)=n_a2_2-1;
        loIdx_2=loIdx_2(:); % force column (see above)
        aprime_residual_2=a2primeVals_2(:)-a2_grid_2(loIdx_2);
        prob_2=1-aprime_residual_2./a2_griddiff_2(loIdx_2);
        prob_2(offBottomOfGrid_2)=1;
        prob_2(offTopOfGrid_2)=0;
    end

    % Per-dim factored output (NOT Kron-folded):
    %   a2primeIndexes(k,:) = lower-grid index in the a2_k dim (1..n_a2(k))
    %   a2primeProbs(k,:)   = probability of that lower grid point
    % Caller does nested 2-corner interp with skipinterp at each level.
    a2primeIndexes=zeros(l_a2,N_d*N_a2*N_semiz,'gpuArray');
    a2primeProbs=zeros(l_a2,N_d*N_a2*N_semiz,'gpuArray');
    a2primeIndexes(1,:)=loIdx_1(:);
    a2primeIndexes(2,:)=loIdx_2(:);
    a2primeProbs(1,:)=prob_1(:);
    a2primeProbs(2,:)=prob_2(:);

    if aprimeIndexAsColumn==1 % column-flat layout
        % already [l_a2, N_d*N_a2*N_semiz]
    elseif aprimeIndexAsColumn==3 % value fn with another asset uses 3
        a2primeIndexes=reshape(a2primeIndexes,[l_a2,N_d*N_a2,N_semiz]);
    else % aprimeIndexAsColumn==2 % matrix layout
        a2primeIndexes=reshape(a2primeIndexes,[l_a2,N_d,N_a2,N_semiz]);
    end
    % Note: every consumer reads this as a2primeProbs(k,:), and A(k,:) ravels identically
    % whatever the trailing shape, so one reshape serves all three layouts (matches the ze builder).
    a2primeProbs=reshape(a2primeProbs,[l_a2,N_d,N_a2,N_semiz]);
end


end
