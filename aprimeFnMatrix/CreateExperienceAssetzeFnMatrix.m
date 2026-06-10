function [a2primeIndexes,a2primeProbs]=CreateExperienceAssetzeFnMatrix(aprimeFn, n_d, n_a2, n_z, n_e, d_gridvals, a2_grid, z_gridvals, e_gridvals, aprimeFnParams, aprimeIndexAsColumn)  % since a2 is one-dimensional, can be a2_grid or a2_gridvals
% For experienceassetze: enumerate a2prime=aprimeFn(d, a2, z, e) over ALL d
% (used during value-function iteration). Because the true value of a2prime
% will (almost always) lie between two consecutive points in a2_grid, it is
% linearly interpolated back on to a2_grid. Thus the continuous a2prime is
% represented by (index of lower grid point in a2primeIndexes, probability
% of lower grid point in a2primeProbs) on a2_grid; the upper index is
% implicitly lower+1 with prob 1-minus-prob-of-lower.
%
% Companion file CreateaprimePolicyExperienceAssetze.m does the same but
% only for the Policy-chosen d (one d per state), used in simulation /
% agent-distribution. This file is used during value-function iteration
% where every d must be evaluated.
%
% Output sizes:
%   l_a2==1 (legacy):
%     a2primeIndexes - shape depends on aprimeIndexAsColumn:
%                        1 => column vector [N_d*N_a2*N_z*N_e, 1]
%                        2 => matrix [N_d, N_a2, N_z, N_e]
%                        3 => matrix [N_d*N_a2, N_z, N_e]
%     a2primeProbs   - [N_d, N_a2, N_z, N_e]
%   l_a2==2 (multi-dim, per-dim factored):
%     a2primeIndexes - col=1 => [l_a2, N_d*N_a2*N_z*N_e]
%                      col=3 => [l_a2, N_d*N_a2, N_z, N_e]
%                      col=2 => [l_a2, N_d, N_a2, N_z, N_e]
%     a2primeProbs   - matches a2primeIndexes (lower-grid index + prob of lower per dim)
%     Caller does nested 2-corner interp with skipinterp at each level.

ParamCell=cell(length(aprimeFnParams),1);
for ii=1:length(aprimeFnParams)
    if size(aprimeFnParams(ii))~=[1,1]
        error('Using experienceassetze does not allow for any of aprimeFn parameters to be anything but a scalar')
    end
    ParamCell(ii,1)={aprimeFnParams(ii)};
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

if nargin(aprimeFn)~=l_d+l_a2+l_z+l_e+(l_a2>=2)+length(aprimeFnParams)
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
if l_z>=1
    z1vals=shiftdim(z_gridvals(:,1),-1-l_a2);
    if l_z>=2
        z2vals=shiftdim(z_gridvals(:,2),-1-l_a2);
        if l_z>=3
            z3vals=shiftdim(z_gridvals(:,3),-1-l_a2);
            if l_z>=4
                z4vals=shiftdim(z_gridvals(:,4),-1-l_a2);
            end
        end
    end
end
if l_e>=1
    e1vals=shiftdim(e_gridvals(:,1),-1-l_a2-l_z);
    if l_e>=2
        e2vals=shiftdim(e_gridvals(:,2),-1-l_a2-l_z);
        if l_e>=3
            e3vals=shiftdim(e_gridvals(:,3),-1-l_a2-l_z);
            if l_e>=4
                e4vals=shiftdim(e_gridvals(:,4),-1-l_a2-l_z);
            end
        end
    end
end

if l_a2==1
    a2vals=shiftdim(a2_grid,-1);

    if l_z==1
        if l_e==1
            if l_d==1
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,e1vals, ParamCell{:});
            elseif l_d==2
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, z1vals,e1vals, ParamCell{:});
            elseif l_d==3
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, z1vals,e1vals, ParamCell{:});
            elseif l_d==4
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, z1vals,e1vals, ParamCell{:});
            end
        elseif l_e==2
            if l_d==1
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,e1vals,e2vals, ParamCell{:});
            elseif l_d==2
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, z1vals,e1vals,e2vals, ParamCell{:});
            elseif l_d==3
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, z1vals,e1vals,e2vals, ParamCell{:});
            elseif l_d==4
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, z1vals,e1vals,e2vals, ParamCell{:});
            end
        elseif l_e==3
            if l_d==1
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_d==2
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, z1vals,e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_d==3
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, z1vals,e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_d==4
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, z1vals,e1vals,e2vals,e3vals, ParamCell{:});
            end
        elseif l_e==4
            if l_d==1
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            elseif l_d==2
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, z1vals,e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            elseif l_d==3
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, z1vals,e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            elseif l_d==4
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, z1vals,e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        end
    elseif l_z==2
        if l_e==1
            if l_d==1
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,e1vals, ParamCell{:});
            elseif l_d==2
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, z1vals,z2vals,e1vals, ParamCell{:});
            elseif l_d==3
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, z1vals,z2vals,e1vals, ParamCell{:});
            elseif l_d==4
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, z1vals,z2vals,e1vals, ParamCell{:});
            end
        elseif l_e==2
            if l_d==1
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,e1vals,e2vals, ParamCell{:});
            elseif l_d==2
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, z1vals,z2vals,e1vals,e2vals, ParamCell{:});
            elseif l_d==3
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, z1vals,z2vals,e1vals,e2vals, ParamCell{:});
            elseif l_d==4
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, z1vals,z2vals,e1vals,e2vals, ParamCell{:});
            end
        elseif l_e==3
            if l_d==1
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_d==2
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, z1vals,z2vals,e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_d==3
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, z1vals,z2vals,e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_d==4
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, z1vals,z2vals,e1vals,e2vals,e3vals, ParamCell{:});
            end
        elseif l_e==4
            if l_d==1
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            elseif l_d==2
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, z1vals,z2vals,e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            elseif l_d==3
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, z1vals,z2vals,e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            elseif l_d==4
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, z1vals,z2vals,e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        end
    elseif l_z==3
        if l_e==1
            if l_d==1
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,z3vals,e1vals, ParamCell{:});
            elseif l_d==2
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, z1vals,z2vals,z3vals,e1vals, ParamCell{:});
            elseif l_d==3
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, z1vals,z2vals,z3vals,e1vals, ParamCell{:});
            elseif l_d==4
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, z1vals,z2vals,z3vals,e1vals, ParamCell{:});
            end
        elseif l_e==2
            if l_d==1
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,z3vals,e1vals,e2vals, ParamCell{:});
            elseif l_d==2
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, z1vals,z2vals,z3vals,e1vals,e2vals, ParamCell{:});
            elseif l_d==3
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, z1vals,z2vals,z3vals,e1vals,e2vals, ParamCell{:});
            elseif l_d==4
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, z1vals,z2vals,z3vals,e1vals,e2vals, ParamCell{:});
            end
        elseif l_e==3
            if l_d==1
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,z3vals,e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_d==2
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, z1vals,z2vals,z3vals,e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_d==3
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, z1vals,z2vals,z3vals,e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_d==4
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, z1vals,z2vals,z3vals,e1vals,e2vals,e3vals, ParamCell{:});
            end
        elseif l_e==4
            if l_d==1
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,z3vals,e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            elseif l_d==2
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, z1vals,z2vals,z3vals,e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            elseif l_d==3
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, z1vals,z2vals,z3vals,e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            elseif l_d==4
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, z1vals,z2vals,z3vals,e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        end
    elseif l_z==4
        if l_e==1
            if l_d==1
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,z3vals,z4vals,e1vals, ParamCell{:});
            elseif l_d==2
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, z1vals,z2vals,z3vals,z4vals,e1vals, ParamCell{:});
            elseif l_d==3
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, z1vals,z2vals,z3vals,z4vals,e1vals, ParamCell{:});
            elseif l_d==4
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, z1vals,z2vals,z3vals,z4vals,e1vals, ParamCell{:});
            end
        elseif l_e==2
            if l_d==1
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals, ParamCell{:});
            elseif l_d==2
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals, ParamCell{:});
            elseif l_d==3
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals, ParamCell{:});
            elseif l_d==4
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals, ParamCell{:});
            end
        elseif l_e==3
            if l_d==1
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_d==2
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_d==3
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_d==4
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals,e3vals, ParamCell{:});
            end
        elseif l_e==4
            if l_d==1
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            elseif l_d==2
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            elseif l_d==3
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            elseif l_d==4
                a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        end
    end

    %% Calcuate grid indexes and probs from the values
    a2primeVals=reshape(a2primeVals,[1,N_d*N_a2*N_z*N_e]);

    a2_griddiff=a2_grid(2:end)-a2_grid(1:end-1); % Distance between point and the next point

    % For small aprimeVals and a_grid, max() is faster than discretize()
    % http://discourse.vfitoolkit.com/t/example-attanasio-low-sanchez-marcos-2008/257/25
    if N_d*N_a2*N_z*N_e*N_a2<1000000
        [~,a2primeIndexes]=max((a2_grid>a2primeVals),[],1); % Keep the dimension corresponding to aprimeVals, minimize over the a_grid dimension
        % Note, this is going to find the 'first' grid point which is bigger than a2primeVals
        % This is the 'upper' grid point
        % Have to have special treatment for trying to leave the ends of the grid (I fix these below)

        % Switch to lower grid point index
        a2primeIndexes=a2primeIndexes-1;
        a2primeIndexes(a2primeIndexes==0)=1;
        a2primeIndexes=a2primeIndexes(:); % force column: when n_a2==2, a2_griddiff is scalar and would otherwise return shape of a2primeIndexes (triggering NxN broadcast)

        % Now, find the probabilities
        aprime_residual=a2primeVals(:)-a2_grid(a2primeIndexes);
        % Probability of the 'lower' points
        a2primeProbs=1-aprime_residual./a2_griddiff(a2primeIndexes);

        % Those points which tried to leave the top of the grid have probability 1 of the 'upper' point (0 of lower point)
        offTopOfGrid=(a2primeVals(:)>=a2_grid(end));
        a2primeIndexes(offTopOfGrid)=n_a2-1; % lower grid point is the one before the end point
        a2primeProbs(offTopOfGrid)=0;
        % Those points which tried to leave the bottom of the grid have probability 0 of the 'upper' point (1 of lower point)
        offBottomOfGrid=(a2primeVals(:)<=a2_grid(1));
        % aprimeIndexes(offBottomOfGrid)=1; % Has already been handled
        a2primeProbs(offBottomOfGrid)=1;
    else
        a2primeIndexes=discretize(a2primeVals,a2_grid); % Finds the lower grid point
        % Have to have special treatment for trying to leave the ends of the grid

        % Those points which tried to leave the bottom of the grid have probability 0 of the 'upper' point (1 of lower point)
        offBottomOfGrid=(a2primeVals(:)<=a2_grid(1));
        a2primeIndexes(offBottomOfGrid)=1; % Has already been handled
        % Those points which tried to leave the top of the grid have probability 1 of the 'upper' point (0 of lower point)
        offTopOfGrid=(a2primeVals(:)>=a2_grid(end));
        a2primeIndexes(offTopOfGrid)=n_a2-1; % lower grid point is the one before the end point
        a2primeIndexes=a2primeIndexes(:); % force column (see above)

        % Now, find the probabilities
        aprime_residual=a2primeVals(:)-a2_grid(a2primeIndexes);
        % Probability of the 'lower' points
        a2primeProbs=1-aprime_residual./a2_griddiff(a2primeIndexes);
        % And clean up the ends of the grid
        a2primeProbs(offBottomOfGrid)=1;
        a2primeProbs(offTopOfGrid)=0;
    end

    if aprimeIndexAsColumn==1 % value fn codes want column
%     aprimeIndexes=reshape(aprimeIndexes,[N_d*N_a2*N_z*N_e,1]);
        a2primeIndexes=a2primeIndexes'; % This is just doing the commented out reshape above
    elseif aprimeIndexAsColumn==3 % value fn with another asset uses 3
        a2primeIndexes=reshape(a2primeIndexes,[N_d*N_a2,N_z,N_e]);
    else % aprimeIndexAsColumn==2 % value fn codes and simulation codes want matrix
        a2primeIndexes=reshape(a2primeIndexes,[N_d,N_a2,N_z,N_e]);
    end
    a2primeProbs=reshape(a2primeProbs,[N_d,N_a2,N_z,N_e]);

elseif l_a2==2
    %% Multi-dim a2 (l_a2=2): bilinear interp, per-dim factored output
    n_a2_1=n_a2(1); n_a2_2=n_a2(2);
    a2_grid_1=a2_grid(1:n_a2_1);
    a2_grid_2=a2_grid(n_a2_1+1:n_a2_1+n_a2_2);
    a2_1_vals=shiftdim(a2_grid_1,-1); % dim 2
    a2_2_vals=shiftdim(a2_grid_2,-2); % dim 3

    % GPU arrayfun requires scalar output; call once per a2 dim with whicha
    % selector. whicha slot is after z,e and before parameters.
    if l_z==1
        if l_e==1
            if l_d==1
                a2primeVals_1=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,e1vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,e1vals, 2, ParamCell{:});
            elseif l_d==2
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,e1vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,e1vals, 2, ParamCell{:});
            elseif l_d==3
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,e1vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,e1vals, 2, ParamCell{:});
            elseif l_d==4
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,e1vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,e1vals, 2, ParamCell{:});
            end
        elseif l_e==2
            if l_d==1
                a2primeVals_1=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,e1vals,e2vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,e1vals,e2vals, 2, ParamCell{:});
            elseif l_d==2
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,e1vals,e2vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,e1vals,e2vals, 2, ParamCell{:});
            elseif l_d==3
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,e1vals,e2vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,e1vals,e2vals, 2, ParamCell{:});
            elseif l_d==4
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,e1vals,e2vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,e1vals,e2vals, 2, ParamCell{:});
            end
        elseif l_e==3
            if l_d==1
                a2primeVals_1=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,e1vals,e2vals,e3vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,e1vals,e2vals,e3vals, 2, ParamCell{:});
            elseif l_d==2
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,e1vals,e2vals,e3vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,e1vals,e2vals,e3vals, 2, ParamCell{:});
            elseif l_d==3
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,e1vals,e2vals,e3vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,e1vals,e2vals,e3vals, 2, ParamCell{:});
            elseif l_d==4
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,e1vals,e2vals,e3vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,e1vals,e2vals,e3vals, 2, ParamCell{:});
            end
        elseif l_e==4
            if l_d==1
                a2primeVals_1=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,e1vals,e2vals,e3vals,e4vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,e1vals,e2vals,e3vals,e4vals, 2, ParamCell{:});
            elseif l_d==2
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,e1vals,e2vals,e3vals,e4vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,e1vals,e2vals,e3vals,e4vals, 2, ParamCell{:});
            elseif l_d==3
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,e1vals,e2vals,e3vals,e4vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,e1vals,e2vals,e3vals,e4vals, 2, ParamCell{:});
            elseif l_d==4
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,e1vals,e2vals,e3vals,e4vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,e1vals,e2vals,e3vals,e4vals, 2, ParamCell{:});
            end
        end
    elseif l_z==2
        if l_e==1
            if l_d==1
                a2primeVals_1=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals, 2, ParamCell{:});
            elseif l_d==2
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals, 2, ParamCell{:});
            elseif l_d==3
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals, 2, ParamCell{:});
            elseif l_d==4
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals, 2, ParamCell{:});
            end
        elseif l_e==2
            if l_d==1
                a2primeVals_1=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals,e2vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals,e2vals, 2, ParamCell{:});
            elseif l_d==2
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals,e2vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals,e2vals, 2, ParamCell{:});
            elseif l_d==3
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals,e2vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals,e2vals, 2, ParamCell{:});
            elseif l_d==4
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals,e2vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals,e2vals, 2, ParamCell{:});
            end
        elseif l_e==3
            if l_d==1
                a2primeVals_1=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals,e2vals,e3vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals,e2vals,e3vals, 2, ParamCell{:});
            elseif l_d==2
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals,e2vals,e3vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals,e2vals,e3vals, 2, ParamCell{:});
            elseif l_d==3
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals,e2vals,e3vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals,e2vals,e3vals, 2, ParamCell{:});
            elseif l_d==4
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals,e2vals,e3vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals,e2vals,e3vals, 2, ParamCell{:});
            end
        elseif l_e==4
            if l_d==1
                a2primeVals_1=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals,e2vals,e3vals,e4vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals,e2vals,e3vals,e4vals, 2, ParamCell{:});
            elseif l_d==2
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals,e2vals,e3vals,e4vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals,e2vals,e3vals,e4vals, 2, ParamCell{:});
            elseif l_d==3
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals,e2vals,e3vals,e4vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals,e2vals,e3vals,e4vals, 2, ParamCell{:});
            elseif l_d==4
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals,e2vals,e3vals,e4vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,z2vals,e1vals,e2vals,e3vals,e4vals, 2, ParamCell{:});
            end
        end
    elseif l_z==3
        if l_e==1
            if l_d==1
                a2primeVals_1=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals, 2, ParamCell{:});
            elseif l_d==2
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals, 2, ParamCell{:});
            elseif l_d==3
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals, 2, ParamCell{:});
            elseif l_d==4
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals, 2, ParamCell{:});
            end
        elseif l_e==2
            if l_d==1
                a2primeVals_1=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals,e2vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals,e2vals, 2, ParamCell{:});
            elseif l_d==2
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals,e2vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals,e2vals, 2, ParamCell{:});
            elseif l_d==3
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals,e2vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals,e2vals, 2, ParamCell{:});
            elseif l_d==4
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals,e2vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals,e2vals, 2, ParamCell{:});
            end
        elseif l_e==3
            if l_d==1
                a2primeVals_1=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals,e2vals,e3vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals,e2vals,e3vals, 2, ParamCell{:});
            elseif l_d==2
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals,e2vals,e3vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals,e2vals,e3vals, 2, ParamCell{:});
            elseif l_d==3
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals,e2vals,e3vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals,e2vals,e3vals, 2, ParamCell{:});
            elseif l_d==4
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals,e2vals,e3vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals,e2vals,e3vals, 2, ParamCell{:});
            end
        elseif l_e==4
            if l_d==1
                a2primeVals_1=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals,e2vals,e3vals,e4vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals,e2vals,e3vals,e4vals, 2, ParamCell{:});
            elseif l_d==2
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals,e2vals,e3vals,e4vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals,e2vals,e3vals,e4vals, 2, ParamCell{:});
            elseif l_d==3
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals,e2vals,e3vals,e4vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals,e2vals,e3vals,e4vals, 2, ParamCell{:});
            elseif l_d==4
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals,e2vals,e3vals,e4vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,e1vals,e2vals,e3vals,e4vals, 2, ParamCell{:});
            end
        end
    elseif l_z==4
        if l_e==1
            if l_d==1
                a2primeVals_1=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals, 2, ParamCell{:});
            elseif l_d==2
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals, 2, ParamCell{:});
            elseif l_d==3
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals, 2, ParamCell{:});
            elseif l_d==4
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals, 2, ParamCell{:});
            end
        elseif l_e==2
            if l_d==1
                a2primeVals_1=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals, 2, ParamCell{:});
            elseif l_d==2
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals, 2, ParamCell{:});
            elseif l_d==3
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals, 2, ParamCell{:});
            elseif l_d==4
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals, 2, ParamCell{:});
            end
        elseif l_e==3
            if l_d==1
                a2primeVals_1=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals,e3vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals,e3vals, 2, ParamCell{:});
            elseif l_d==2
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals,e3vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals,e3vals, 2, ParamCell{:});
            elseif l_d==3
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals,e3vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals,e3vals, 2, ParamCell{:});
            elseif l_d==4
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals,e3vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals,e3vals, 2, ParamCell{:});
            end
        elseif l_e==4
            if l_d==1
                a2primeVals_1=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals,e3vals,e4vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals,e3vals,e4vals, 2, ParamCell{:});
            elseif l_d==2
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals,e3vals,e4vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals,e3vals,e4vals, 2, ParamCell{:});
            elseif l_d==3
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals,e3vals,e4vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals,e3vals,e4vals, 2, ParamCell{:});
            elseif l_d==4
                a2primeVals_1=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals,e3vals,e4vals, 1, ParamCell{:});
                a2primeVals_2=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2_1_vals, a2_2_vals, z1vals,z2vals,z3vals,z4vals,e1vals,e2vals,e3vals,e4vals, 2, ParamCell{:});
            end
        end
    end

    %% Per-dim grid indexes and probs (inlined 1D linear-interp; mirrors l_a2==1 above)
    a2primeVals_1=reshape(a2primeVals_1,[1,N_d*N_a2*N_z*N_e]);
    a2primeVals_2=reshape(a2primeVals_2,[1,N_d*N_a2*N_z*N_e]);
    a2_griddiff_1=a2_grid_1(2:end)-a2_grid_1(1:end-1);
    a2_griddiff_2=a2_grid_2(2:end)-a2_grid_2(1:end-1);

    % --- a2 dim 1 ---
    if N_d*N_a2*N_z*N_e*n_a2_1<1000000
        [~,loIdx_1]=max((a2_grid_1>a2primeVals_1),[],1);
        loIdx_1=loIdx_1-1;
        loIdx_1(loIdx_1==0)=1;
        loIdx_1=loIdx_1(:); % force column: when n_a2_1==2, a2_griddiff_1 is scalar and would otherwise return shape of loIdx_1
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
    if N_d*N_a2*N_z*N_e*n_a2_2<1000000
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
    %   a2primeIndexes(k,:) = lower-grid index in a2_k dim (1..n_a2(k))
    %   a2primeProbs(k,:)   = probability of lower grid point in a2_k dim
    % Caller does nested 2-corner interp with skipinterp at each level.
    a2primeIndexes=zeros(l_a2,N_d*N_a2*N_z*N_e,'gpuArray');
    a2primeProbs=zeros(l_a2,N_d*N_a2*N_z*N_e,'gpuArray');
    a2primeIndexes(1,:)=loIdx_1(:);
    a2primeIndexes(2,:)=loIdx_2(:);
    a2primeProbs(1,:)=prob_1(:);
    a2primeProbs(2,:)=prob_2(:);

    if aprimeIndexAsColumn==1 % column-flat layout
        % already [l_a2, N_d*N_a2*N_z*N_e]
    elseif aprimeIndexAsColumn==3 % value fn with another asset uses 3
        a2primeIndexes=reshape(a2primeIndexes,[l_a2,N_d*N_a2,N_z,N_e]);
    else % aprimeIndexAsColumn==2 % matrix layout
        a2primeIndexes=reshape(a2primeIndexes,[l_a2,N_d,N_a2,N_z,N_e]);
    end
    a2primeProbs=reshape(a2primeProbs,[l_a2,N_d,N_a2,N_z,N_e]);
end


end
