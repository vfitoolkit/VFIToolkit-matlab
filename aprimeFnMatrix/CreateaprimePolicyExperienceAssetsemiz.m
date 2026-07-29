function [a2primeIndexes, a2primeProbs]=CreateaprimePolicyExperienceAssetsemiz(Policy,aprimeFn, whichisdforexpasset, n_d, n_a1,n_a2, n_semiz, N_semiz,N_z,N_e, d_grid, a2_grid, semiz_gridvals, aprimeFnParams)
% For experienceassetsemiz: compute a2prime=aprimeFn(d, a2, semiz) using the
% Policy-chosen d for each state (one d per state), used in simulation /
% agent-distribution. Because the true value of a2prime will (almost
% always) lie between two consecutive points in a2_grid, it is linearly
% interpolated back on to a2_grid. Thus the continuous a2prime is
% represented by (index of lower grid point in a2primeIndexes, probability
% of lower grid point in a2primeProbs) on a2_grid; the upper index is
% implicitly lower+1 with prob 1-minus-prob-of-lower.
%
% Note: the experience asset is driven by the semi-exogenous state semiz. In
% the joint exogenous ordering bothz=[semiz,z], semiz is the fast index, so
% in the Policy layout [N_a, N_semiz, N_z, N_e] semiz occupies dimension 2
% (in contrast to experienceassetz, where the driving z is in dimension 3).
%
% The input Policy will contain aprime (except for the experience asset)
% and the decision variables (d2, and where applicable d1). The output is
% just the Policy for a2prime (the experience asset). As well as the
% related probabilities.
%
% Companion file CreateExperienceAssetsemizFnMatrix.m does the same but for ALL
% d (not just the Policy-chosen one), used during value-function iteration
% to find Policy. This file is used afterwards, once Policy has been
% chosen, for simulation / agent-distribution.
%
% Output sizes:
%   a2primeIndexes - [N_a, N_semizze]
%   a2primeProbs   - [N_a, N_semizze]
%
% Note: N_semizze is just the 'size' of Policy

ParamCell=cell(length(aprimeFnParams),1);
for ii=1:length(aprimeFnParams)
    if ~isscalar(aprimeFnParams(ii))
        error('Using experienceassetsemiz does not allow for any of aprimeFn parameters to be anything but a scalar')
    end
    ParamCell(ii,1)={aprimeFnParams(ii)};
end

N_a1=prod(n_a1);
if N_a1==0
    N_a=prod(n_a2);
else
    N_a=prod([n_a1,n_a2]);
end

l_dexp=length(whichisdforexpasset);
l_semiz=length(n_semiz);

if nargin(aprimeFn)~=l_dexp+1+l_semiz+length(aprimeFnParams)
    error('Number of inputs to aprimeFn does not fit with size of aprimeFnParams')
end

if l_semiz>=5
    error('Max of four semiz variables supported in CreateaprimePolicyExperienceAssetsemiz (contact if you need more)')
end
if N_semiz==0
    error('experienceassetsemiz requires N_semiz>0 (the semi-exogenous state drives the experience asset)')
end

if l_dexp>=1 % WHY I AM DOING THIS, PRETTY SURE YOU CANNOT NOT SATISFY THIS???
    if whichisdforexpasset(1)==1
        d1grid=d_grid(1:n_d(1));
    else
        d1grid=d_grid(sum(n_d(1:whichisdforexpasset(1)-1))+1:sum(n_d(1:whichisdforexpasset(1))));
    end

    if N_e==0
        N_semizze=N_semiz*N_z;
        d1vals=reshape(d1grid(Policy(whichisdforexpasset(1),:,:)),[N_a,N_semiz,N_z]);
        if l_dexp>=2
            d2grid=d_grid(sum(n_d(1:whichisdforexpasset(2)-1))+1:sum(n_d(1:whichisdforexpasset(2))));
            d2vals=reshape(d2grid(Policy(whichisdforexpasset(2),:,:)),[N_a,N_semiz,N_z]);
            if l_dexp>=3
                d3grid=d_grid(sum(n_d(1:whichisdforexpasset(3)-1))+1:sum(n_d(1:whichisdforexpasset(3))));
                d3vals=reshape(d3grid(Policy(whichisdforexpasset(3),:,:)),[N_a,N_semiz,N_z]);
                if l_dexp>=4
                    d4grid=d_grid(sum(n_d(1:whichisdforexpasset(4)-1))+1:sum(n_d(1:whichisdforexpasset(4))));
                    d4vals=reshape(d4grid(Policy(whichisdforexpasset(4),:,:)),[N_a,N_semiz,N_z]);
                end
            end
        end
    else % N_e>0
        N_semizze=N_semiz*N_z*N_e;
        d1vals=reshape(d1grid(Policy(whichisdforexpasset(1),:,:)),[N_a,N_semiz,N_z,N_e]);
        if l_dexp>=2
            d2grid=d_grid(sum(n_d(1:whichisdforexpasset(2)-1))+1:sum(n_d(1:whichisdforexpasset(2))));
            d2vals=reshape(d2grid(Policy(whichisdforexpasset(2),:,:)),[N_a,N_semiz,N_z,N_e]);
            if l_dexp>=3
                d3grid=d_grid(sum(n_d(1:whichisdforexpasset(3)-1))+1:sum(n_d(1:whichisdforexpasset(3))));
                d3vals=reshape(d3grid(Policy(whichisdforexpasset(3),:,:)),[N_a,N_semiz,N_z,N_e]);
                if l_dexp>=4
                    d4grid=d_grid(sum(n_d(1:whichisdforexpasset(4)-1))+1:sum(n_d(1:whichisdforexpasset(4))));
                    d4vals=reshape(d4grid(Policy(whichisdforexpasset(4),:,:)),[N_a,N_semiz,N_z,N_e]);
                end
            end
        end
    end
end

if N_a1==0
    a2vals=a2_grid;
else
    a2vals=repelem(a2_grid,N_a1,1);
end


%% expassetsemiz: aprime(d,a2,semiz); semiz varies along dimension 2 (shiftdim -1)
if l_semiz==1
    if l_dexp==1
        a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, shiftdim(semiz_gridvals(:,1),-1), ParamCell{:});
    elseif l_dexp==2
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, shiftdim(semiz_gridvals(:,1),-1), ParamCell{:});
    elseif l_dexp==3
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, shiftdim(semiz_gridvals(:,1),-1), ParamCell{:});
    elseif l_dexp==4
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, shiftdim(semiz_gridvals(:,1),-1), ParamCell{:});
    end
elseif l_semiz==2
    if l_dexp==1
        a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, shiftdim(semiz_gridvals(:,1),-1), shiftdim(semiz_gridvals(:,2),-1), ParamCell{:});
    elseif l_dexp==2
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, shiftdim(semiz_gridvals(:,1),-1), shiftdim(semiz_gridvals(:,2),-1), ParamCell{:});
    elseif l_dexp==3
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, shiftdim(semiz_gridvals(:,1),-1), shiftdim(semiz_gridvals(:,2),-1), ParamCell{:});
    elseif l_dexp==4
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, shiftdim(semiz_gridvals(:,1),-1), shiftdim(semiz_gridvals(:,2),-1), ParamCell{:});
    end
elseif l_semiz==3
    if l_dexp==1
        a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, shiftdim(semiz_gridvals(:,1),-1), shiftdim(semiz_gridvals(:,2),-1), shiftdim(semiz_gridvals(:,3),-1), ParamCell{:});
    elseif l_dexp==2
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, shiftdim(semiz_gridvals(:,1),-1), shiftdim(semiz_gridvals(:,2),-1), shiftdim(semiz_gridvals(:,3),-1), ParamCell{:});
    elseif l_dexp==3
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, shiftdim(semiz_gridvals(:,1),-1), shiftdim(semiz_gridvals(:,2),-1), shiftdim(semiz_gridvals(:,3),-1), ParamCell{:});
    elseif l_dexp==4
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, shiftdim(semiz_gridvals(:,1),-1), shiftdim(semiz_gridvals(:,2),-1), shiftdim(semiz_gridvals(:,3),-1), ParamCell{:});
    end
elseif l_semiz==4
    if l_dexp==1
        a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, shiftdim(semiz_gridvals(:,1),-1), shiftdim(semiz_gridvals(:,2),-1), shiftdim(semiz_gridvals(:,3),-1), shiftdim(semiz_gridvals(:,4),-1), ParamCell{:});
    elseif l_dexp==2
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, shiftdim(semiz_gridvals(:,1),-1), shiftdim(semiz_gridvals(:,2),-1), shiftdim(semiz_gridvals(:,3),-1), shiftdim(semiz_gridvals(:,4),-1), ParamCell{:});
    elseif l_dexp==3
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, shiftdim(semiz_gridvals(:,1),-1), shiftdim(semiz_gridvals(:,2),-1), shiftdim(semiz_gridvals(:,3),-1), shiftdim(semiz_gridvals(:,4),-1), ParamCell{:});
    elseif l_dexp==4
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, shiftdim(semiz_gridvals(:,1),-1), shiftdim(semiz_gridvals(:,2),-1), shiftdim(semiz_gridvals(:,3),-1), shiftdim(semiz_gridvals(:,4),-1), ParamCell{:});
    end
end


%% Calcuate grid indexes and probs from the values
a2primeVals=reshape(a2primeVals,[1,N_a*N_semizze]);

a2_griddiff=a2_grid(2:end)-a2_grid(1:end-1); % Distance between point and the next point

a2primeIndexes=discretize(a2primeVals,a2_grid); % Finds the lower grid point
% Have to have special treatment for trying to leave the ends of the grid

% Those points which tried to leave the bottom of the grid have probability 0 of the 'upper' point (1 of lower point)
offBottomOfGrid=(a2primeVals<=a2_grid(1));
a2primeIndexes(offBottomOfGrid)=1; % Has already been handled
% Those points which tried to leave the top of the grid have probability 1 of the 'upper' point (0 of lower point)
offTopOfGrid=(a2primeVals>=a2_grid(end));
a2primeIndexes(offTopOfGrid)=n_a2-1; % lower grid point is the one before the end point
a2primeIndexes=reshape(a2primeIndexes,[N_a*N_semizze,1]);

% Now, find the probabilities
aprime_residual=a2primeVals'-a2_grid(a2primeIndexes);
% Probability of the 'lower' points
a2primeProbs=1-aprime_residual./a2_griddiff(a2primeIndexes);
% And clean up the ends of the grid
a2primeProbs(offBottomOfGrid)=1;
a2primeProbs(offTopOfGrid)=0;

a2primeIndexes=reshape(a2primeIndexes,[N_a,N_semizze]); % Index of lower grid point
a2primeProbs=reshape(a2primeProbs,[N_a,N_semizze]); % Probability of lower grid point


end
