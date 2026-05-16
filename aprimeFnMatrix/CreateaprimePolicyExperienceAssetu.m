function [a2primeIndexes, a2primeProbs]=CreateaprimePolicyExperienceAssetu(Policy,aprimeFn, whichisdforexpasset, n_d, n_a1,n_a2, N_z, n_u, d_grid, a2_grid, u_grid, aprimeFnParams)
% For experienceassetu: compute a2prime=aprimeFn(d, a2, u) using the
% Policy-chosen d for each state (one d per state), used in simulation /
% agent-distribution. Note: u is i.i.d. drawn BETWEEN periods, so Policy
% does NOT depend on u; u only enters when computing a2prime from the
% already-chosen d. Because the true value of a2prime will (almost always)
% lie between two consecutive points in a2_grid, it is linearly
% interpolated back on to a2_grid. Thus the continuous a2prime is
% represented by (index of lower grid point in a2primeIndexes, probability
% of lower grid point in a2primeProbs) on a2_grid; the upper index is
% implicitly lower+1 with prob 1-minus-prob-of-lower.
%
% The input Policy will contain aprime (except for the experience asset)
% and the decision variables (d2, and where applicable d1). The output is
% just the Policy for a2prime (the experience asset). As well as the
% related probabilities.
%
% Companion file CreateExperienceAssetuFnMatrix.m does the same but for ALL
% d (not just the Policy-chosen one), used during value-function iteration
% to find Policy. This file is used afterwards, once Policy has been
% chosen, for simulation / agent-distribution.
%
% Output sizes (when N_z==0):
%   a2primeIndexes - [N_a, N_u]
%   a2primeProbs   - [N_a, N_u]
% Output sizes (when N_z>0):
%   a2primeIndexes - [N_a, N_z, N_u]
%   a2primeProbs   - [N_a, N_z, N_u]

ParamCell=cell(length(aprimeFnParams),1);
for ii=1:length(aprimeFnParams)
    if ~isscalar(aprimeFnParams(ii))
        error('Using experienceassetu does not allow for any of aprimeFn parameters to be anything but a scalar')
    end
    ParamCell(ii,1)={aprimeFnParams(ii)};
end

N_a1=prod(n_a1);
if N_a1==0
    N_a=prod(n_a2);
else
    N_a=prod([n_a1,n_a2]);
end
N_u=prod(n_u);

l_dexp=length(whichisdforexpasset);
l_u=length(n_u);

if nargin(aprimeFn)~=l_dexp+1+l_u+length(aprimeFnParams)
    error('Number of inputs to aprimeFn does not fit with size of aprimeFnParams')
end

if N_z==0 % To save writing a separate script for without z
    if l_dexp>=1
        if whichisdforexpasset(1)==1
            d1grid=d_grid(1:n_d(1));
        else
            d1grid=d_grid(sum(n_d(1:whichisdforexpasset(1)-1))+1:sum(n_d(1:whichisdforexpasset(1))));
        end
        d1vals=reshape(d1grid(Policy(whichisdforexpasset(1),:)),[N_a,1]);
        if l_dexp>=2
            d2grid=d_grid(sum(n_d(1:whichisdforexpasset(2)-1))+1:sum(n_d(1:whichisdforexpasset(2))));
            d2vals=reshape(d2grid(Policy(whichisdforexpasset(2),:)),[N_a,1]);
            if l_dexp>=3
                d3grid=d_grid(sum(n_d(1:whichisdforexpasset(3)-1))+1:sum(n_d(1:whichisdforexpasset(3))));
                d3vals=reshape(d3grid(Policy(whichisdforexpasset(3),:)),[N_a,1]);
                if l_dexp>=4
                    d4grid=d_grid(sum(n_d(1:whichisdforexpasset(4)-1))+1:sum(n_d(1:whichisdforexpasset(4))));
                    d4vals=reshape(d4grid(Policy(whichisdforexpasset(4),:)),[N_a,1]);
                end
            end
        end
    end
else
    if l_dexp>=1
        if whichisdforexpasset(1)==1
            d1grid=d_grid(1:n_d(1));
        else
            d1grid=d_grid(sum(n_d(1:whichisdforexpasset(1)-1))+1:sum(n_d(1:whichisdforexpasset(1))));
        end
        d1vals=reshape(d1grid(Policy(whichisdforexpasset(1),:,:)),[N_a*N_z,1]);
        if l_dexp>=2
            d2grid=d_grid(sum(n_d(1:whichisdforexpasset(2)-1))+1:sum(n_d(1:whichisdforexpasset(2))));
            d2vals=reshape(d2grid(Policy(whichisdforexpasset(2),:,:)),[N_a*N_z,1]);
            if l_dexp>=3
                d3grid=d_grid(sum(n_d(1:whichisdforexpasset(3)-1))+1:sum(n_d(1:whichisdforexpasset(3))));
                d3vals=reshape(d3grid(Policy(whichisdforexpasset(3),:,:)),[N_a*N_z,1]);
                if l_dexp>=4
                    d4grid=d_grid(sum(n_d(1:whichisdforexpasset(4)-1))+1:sum(n_d(1:whichisdforexpasset(4))));
                    d4vals=reshape(d4grid(Policy(whichisdforexpasset(4),:,:)),[N_a*N_z,1]);
                end
            end
        end
    end
end

if N_z==0
    if N_a1==0
        a2vals=a2_grid;
    else
        a2vals=kron(a2_grid,ones(N_a1,1));
    end
else
    if N_a1==0
        a2vals=kron(ones(N_z,1),a2_grid);
    else
        a2vals=kron(ones(N_z,1),kron(a2_grid,ones(N_a1,1)));
    end
end

u_gridvals=CreateGridvals(n_u,u_grid,1);
if l_u>=1
    u1vals=shiftdim(u_gridvals(:,1),-1);
    if l_u>=2
        u2vals=shiftdim(u_gridvals(:,2),-1);
        if l_u>=3
            u3vals=shiftdim(u_gridvals(:,3),-1);
            if l_u>=4
                u4vals=shiftdim(u_gridvals(:,4),-1);
                if l_u>=5
                    error('Max of four u variables supported (contact if you need more)')
                end
            end
        end
    end
end

% Note: the relevant d for experienceassetu is just the 'whichisdforexpasset' d (this is typically just the last if using just experienceassetu)
% expassetu: aprime(d,a2,u)
% Removed: a2vals=a2vals.*ones(1,1,1,'gpuArray'); % was here to fool matlab which otherwise threw an error; restore this line if functionality breaks
if l_u==1
    if l_dexp==1
        a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, u1vals, ParamCell{:});
    elseif l_dexp==2
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, u1vals, ParamCell{:});
    elseif l_dexp==3
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, u1vals, ParamCell{:});
    elseif l_dexp==4
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, u1vals, ParamCell{:});
    end
elseif l_u==2
    if l_dexp==1
        a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, u1vals, u2vals, ParamCell{:});
    elseif l_dexp==2
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, u1vals, u2vals, ParamCell{:});
    elseif l_dexp==3
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, u1vals, u2vals, ParamCell{:});
    elseif l_dexp==4
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, u1vals, u2vals, ParamCell{:});
    end
elseif l_u==3
    if l_dexp==1
        a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, u1vals, u2vals, u3vals, ParamCell{:});
    elseif l_dexp==2
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, u1vals, u2vals, u3vals, ParamCell{:});
    elseif l_dexp==3
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, u1vals, u2vals, u3vals, ParamCell{:});
    elseif l_dexp==4
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, u1vals, u2vals, u3vals, ParamCell{:});
    end
elseif l_u==4
    if l_dexp==1
        a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, u1vals, u2vals, u3vals, u4vals, ParamCell{:});
    elseif l_dexp==2
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, u1vals, u2vals, u3vals, u4vals, ParamCell{:});
    elseif l_dexp==3
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, u1vals, u2vals, u3vals, u4vals, ParamCell{:});
    elseif l_dexp==4
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, u1vals, u2vals, u3vals, u4vals, ParamCell{:});
    end
end

%% Calcuate grid indexes and probs from the values
if N_z==0
    a2primeVals=reshape(a2primeVals,[1,N_a*N_u]);
else
    a2primeVals=reshape(a2primeVals,[1,N_a*N_z*N_u]);
end

a2_griddiff=a2_grid(2:end)-a2_grid(1:end-1); % Distance between point and the next point

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

if N_z==0
    a2primeIndexes=reshape(a2primeIndexes,[N_a,N_u]); % Index of lower grid point
    a2primeProbs=reshape(a2primeProbs,[N_a,N_u]); % Probability of lower grid point
else
    a2primeIndexes=reshape(a2primeIndexes,[N_a,N_z,N_u]); % Index of lower grid point
    a2primeProbs=reshape(a2primeProbs,[N_a,N_z,N_u]); % Probability of lower grid point
end

end
