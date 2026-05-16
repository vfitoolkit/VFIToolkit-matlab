function [a2primeIndexes,a2primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d, n_a2, n_z, d_gridvals, a2_grid, z_gridvals, aprimeFnParams, aprimeIndexAsColumn)  % since a2 is one-dimensional, can be a2_grid or a2_gridvals
% For experienceassetz: enumerate a2prime=aprimeFn(d, a2, z) over ALL d
% (used during value-function iteration). Because the true value of a2prime
% will (almost always) lie between two consecutive points in a2_grid, it is
% linearly interpolated back on to a2_grid. Thus the continuous a2prime is
% represented by (index of lower grid point in a2primeIndexes, probability
% of lower grid point in a2primeProbs) on a2_grid; the upper index is
% implicitly lower+1 with prob 1-minus-prob-of-lower.
%
% Companion file CreateaprimePolicyExperienceAssetz.m does the same but only
% for the Policy-chosen d (one d per state), used in simulation /
% agent-distribution. This file is used during value-function iteration
% where every d must be evaluated.
%
% Output sizes:
%   a2primeIndexes - shape depends on aprimeIndexAsColumn:
%                      1 => column vector [N_d*N_a2*N_z, 1]
%                      2 => matrix [N_d, N_a2, N_z]
%                      3 => matrix [N_d*N_a2, N_z]
%   a2primeProbs   - [N_d, N_a2, N_z]

ParamCell=cell(length(aprimeFnParams),1);
for ii=1:length(aprimeFnParams)
    if size(aprimeFnParams(ii))~=[1,1]
        error('Using experienceassetz does not allow for any of aprimeFn parameters to be anything but a scalar')
    end
    ParamCell(ii,1)={aprimeFnParams(ii)};
end

N_d=prod(n_d);
N_a2=prod(n_a2);
N_z=prod(n_z);

l_d=length(n_d);
if N_d==0
    l_d=0;
end
l_a2=length(n_a2);
if l_d>4
    error('experienceassetz does not allow for more than four of d variable (you have length(n_d)>4)')
end
l_z=length(n_z);

if nargin(aprimeFn)~=l_d+l_a2+l_z+length(aprimeFnParams)
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
a2vals=shiftdim(a2_grid,-1);
if l_z>=1
    z1vals=shiftdim(z_gridvals(:,1),-1-l_a2);
    if l_z>=2
        z2vals=shiftdim(z_gridvals(:,2),-1-l_a2);
        if l_z>=3
            z3vals=shiftdim(z_gridvals(:,3),-1-l_a2);
            if l_z>=4
                z4vals=shiftdim(z_gridvals(:,4),-1-l_a2);
                if l_z>=5
                    error('Max of four z variables supported (contact if you need more)')
                end
            end
        end
    end
end

if l_z==1
    if l_d==1
        a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals, ParamCell{:});
    elseif l_d==2
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, z1vals, ParamCell{:});
    elseif l_d==3
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, z1vals, ParamCell{:});
    elseif l_d==4
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, z1vals, ParamCell{:});
    end
elseif l_z==2
    if l_d==1
        a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals, ParamCell{:});
    elseif l_d==2
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, z1vals,z2vals, ParamCell{:});
    elseif l_d==3
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, z1vals,z2vals, ParamCell{:});
    elseif l_d==4
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, z1vals,z2vals, ParamCell{:});
    end
elseif l_z==3
    if l_d==1
        a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,z3vals, ParamCell{:});
    elseif l_d==2
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, z1vals,z2vals,z3vals, ParamCell{:});
    elseif l_d==3
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, z1vals,z2vals,z3vals, ParamCell{:});
    elseif l_d==4
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, z1vals,z2vals,z3vals, ParamCell{:});
    end
elseif l_z==4
    if l_d==1
        a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
    elseif l_d==2
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
    elseif l_d==3
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
    elseif l_d==4
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
    end
end

%% Calcuate grid indexes and probs from the values
if l_a2==1
    a2primeVals=reshape(a2primeVals,[1,N_d*N_a2*N_z]);

    a2_griddiff=a2_grid(2:end)-a2_grid(1:end-1); % Distance between point and the next point

    % For small aprimeVals and a_grid, max() is faster than discretize()
    % http://discourse.vfitoolkit.com/t/example-attanasio-low-sanchez-marcos-2008/257/25
    if N_d*N_a2*N_z*N_a2<1000000
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
%     aprimeIndexes=reshape(aprimeIndexes,[N_d*N_a2*N_z,1]);
        a2primeIndexes=a2primeIndexes'; % This is just doing the commented out reshape above
    elseif aprimeIndexAsColumn==3 % value fn with another asset uses 3
        a2primeIndexes=reshape(a2primeIndexes,[N_d*N_a2,N_z]);
    else % aprimeIndexAsColumn==2 % value fn codes and simulation codes want matrix
        a2primeIndexes=reshape(a2primeIndexes,[N_d,N_a2,N_z]);
    end
    a2primeProbs=reshape(a2primeProbs,[N_d,N_a2,N_z]);
end


end
