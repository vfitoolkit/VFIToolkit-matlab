function [a2primeIndexes, a2primeProbs]=CreateaprimePolicyExperienceAssetze(Policy,aprimeFn, whichisdforexpasset, n_d, n_a1,n_a2, n_z, n_e, d_grid, a2_grid, z_gridvals, e_gridvals, aprimeFnParams)
% For experienceassetze: compute a2prime=aprimeFn(d, a2, z, e) using the
% Policy-chosen d for each state (one d per state), used in simulation /
% agent-distribution. Note: e is i.i.d. drawn at the START of the period,
% so Policy DOES depend on e (Policy has shape [..., N_a, N_z, N_e]).
% Because the true value of a2prime will (almost always) lie between two
% consecutive points in a2_grid, it is linearly interpolated back on to
% a2_grid. Thus the continuous a2prime is represented by (index of lower
% grid point in a2primeIndexes, probability of lower grid point in
% a2primeProbs) on a2_grid; the upper index is implicitly lower+1 with
% prob 1-minus-prob-of-lower.
%
% The input Policy will contain aprime (except for the experience asset)
% and the decision variables (d2, and where applicable d1). The output is
% just the Policy for a2prime (the experience asset). As well as the
% related probabilities.
%
% Companion file CreateExperienceAssetzeFnMatrix.m does the same but for
% ALL d (not just the Policy-chosen one), used during value-function
% iteration to find Policy. This file is used afterwards, once Policy has
% been chosen, for simulation / agent-distribution.
%
% Output sizes:
%   a2primeIndexes - [N_a, N_z, N_e]
%   a2primeProbs   - [N_a, N_z, N_e]

ParamCell=cell(length(aprimeFnParams),1);
for ii=1:length(aprimeFnParams)
    if ~isscalar(aprimeFnParams(ii))
        error('Using experienceassetze does not allow for any of aprimeFn parameters to be anything but a scalar')
    end
    ParamCell(ii,1)={aprimeFnParams(ii)};
end

N_a1=prod(n_a1);
if N_a1==0
    N_a=prod(n_a2);
else
    N_a=prod([n_a1,n_a2]);
end
N_z=prod(n_z);
N_e=prod(n_e);
N_aze=N_a*N_z*N_e;

l_dexp=length(whichisdforexpasset);
l_z=length(n_z);
l_e=length(n_e);

if nargin(aprimeFn)~=l_dexp+1+l_z+l_e+length(aprimeFnParams)
    error('Number of inputs to aprimeFn does not fit with size of aprimeFnParams')
end

if l_dexp>=1
    if whichisdforexpasset(1)==1
        d1grid=d_grid(1:n_d(1));
    else
        d1grid=d_grid(sum(n_d(1:whichisdforexpasset(1)-1))+1:sum(n_d(1:whichisdforexpasset(1))));
    end
    d1vals=reshape(d1grid(Policy(whichisdforexpasset(1),:,:,:)),[N_aze,1]);
    if l_dexp>=2
        d2grid=d_grid(sum(n_d(1:whichisdforexpasset(2)-1))+1:sum(n_d(1:whichisdforexpasset(2))));
        d2vals=reshape(d2grid(Policy(whichisdforexpasset(2),:,:,:)),[N_aze,1]);
        if l_dexp>=3
            d3grid=d_grid(sum(n_d(1:whichisdforexpasset(3)-1))+1:sum(n_d(1:whichisdforexpasset(3))));
            d3vals=reshape(d3grid(Policy(whichisdforexpasset(3),:,:,:)),[N_aze,1]);
            if l_dexp>=4
                d4grid=d_grid(sum(n_d(1:whichisdforexpasset(4)-1))+1:sum(n_d(1:whichisdforexpasset(4))));
                d4vals=reshape(d4grid(Policy(whichisdforexpasset(4),:,:,:)),[N_aze,1]);
            end
        end
    end
end

% Layout: a varies fastest, then z, then e (matching Policy's [N_a,N_z,N_e] reshape)
if N_a1==0
    a2vals=kron(ones(N_e*N_z,1),a2_grid);
else
    a2vals=kron(ones(N_e*N_z,1),kron(a2_grid,ones(N_a1,1)));
end

% z values: each (a,z,e)-row gets z_gridvals row for its z
if l_z>=1
    z1vals=kron(ones(N_e,1),kron(z_gridvals(:,1),ones(N_a,1)));
    if l_z>=2
        z2vals=kron(ones(N_e,1),kron(z_gridvals(:,2),ones(N_a,1)));
        if l_z>=3
            z3vals=kron(ones(N_e,1),kron(z_gridvals(:,3),ones(N_a,1)));
            if l_z>=4
                z4vals=kron(ones(N_e,1),kron(z_gridvals(:,4),ones(N_a,1)));
                if l_z>=5
                    error('Max of four z variables supported in CreateaprimePolicyExperienceAssetze (contact if you need more)')
                end
            end
        end
    end
end

% e values: each (a,z,e)-row gets e_gridvals row for its e
if l_e>=1
    e1vals=kron(e_gridvals(:,1),ones(N_a*N_z,1));
    if l_e>=2
        e2vals=kron(e_gridvals(:,2),ones(N_a*N_z,1));
        if l_e>=3
            e3vals=kron(e_gridvals(:,3),ones(N_a*N_z,1));
            if l_e>=4
                e4vals=kron(e_gridvals(:,4),ones(N_a*N_z,1));
                if l_e>=5
                    error('Max of four e variables supported in CreateaprimePolicyExperienceAssetze (contact if you need more)')
                end
            end
        end
    end
end

% Removed: a2vals=a2vals.*ones(1,1,1,'gpuArray'); % was in the analogous non-z file to fool matlab which otherwise threw an error; restore if functionality breaks

% Pack z and e args in order (z1,...z_lz, e1,...e_le) for arrayfun
zecell={};
if l_z>=1, zecell{end+1}=z1vals; end
if l_z>=2, zecell{end+1}=z2vals; end
if l_z>=3, zecell{end+1}=z3vals; end
if l_z>=4, zecell{end+1}=z4vals; end
if l_e>=1, zecell{end+1}=e1vals; end
if l_e>=2, zecell{end+1}=e2vals; end
if l_e>=3, zecell{end+1}=e3vals; end
if l_e>=4, zecell{end+1}=e4vals; end

% expassetze: aprime(d,a2,z,e)
if l_dexp==1
    a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, zecell{:}, ParamCell{:});
elseif l_dexp==2
    a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, zecell{:}, ParamCell{:});
elseif l_dexp==3
    a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, zecell{:}, ParamCell{:});
elseif l_dexp==4
    a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, zecell{:}, ParamCell{:});
end

%% Calcuate grid indexes and probs from the values
a2primeVals=reshape(a2primeVals,[1,N_aze]);

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

a2primeIndexes=reshape(a2primeIndexes,[N_a,N_z,N_e]); % Index of lower grid point
a2primeProbs=reshape(a2primeProbs,[N_a,N_z,N_e]); % Probability of lower grid point

end
