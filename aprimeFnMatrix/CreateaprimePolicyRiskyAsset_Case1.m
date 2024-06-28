function [a2primeIndexes, a2primeProbs]=CreateaprimePolicyRiskyAsset_Case1(Policy,aprimeFn, whichisdforriskyasset, n_d, n_a1,n_a2, N_z, n_u, d_grid, a2_grid, u_grid, aprimeFnParams)
% The input Policy will contain aprime (except for the risky asset)
% and the decision variables (d2, and where applicable d1). The output is
% just the Policy for a2prime (the risky asset). As well as the
% related probabilities.

% Note: aprimeIndex is [N_a,N_z,N_u], so is aprimeProbs which only reports the probability for the lower grid point.
%
% Creates the grid points and their 'interpolation' probabilities.
% Note: aprimeIndexes is always the 'lower' point (the upper points are
% just aprimeIndexes+1, so no need to waste memory storing them), and the
% aprimeProbs are the probability of this lower point (prob of upper point
% is just 1 minus this).
%
%
% Remark: This is like CreateRiskyAssetFnMatrix_Case1(), except
% instead of looking at all possible d, we only care about those in Policy.
% The Policy based ones are needed for simulation, while those for all
% possible d were needed for value function (to find Policy).

ParamCell=cell(length(aprimeFnParams),1);
for ii=1:length(aprimeFnParams)
    if size(aprimeFnParams(ii))~=[1,1]
        error('Using GPU for the return fn does not allow for any of aprimeFn parameters to be anything but a scalar')
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

l_drisky=length(whichisdforriskyasset);
l_u=length(n_u);

if nargin(aprimeFn)~=l_drisky+l_u+length(aprimeFnParams)
    error('Number of inputs to aprimeFn does not fit with size of aprimeFnParams')
end

if N_z==0 % To save writing a seperate script for without z
    if l_drisky>=1
        if whichisdforriskyasset(1)==1
            d1_grid=d_grid(1:n_d(1));
        else
            d1_grid=d_grid(sum(n_d(1:whichisdforriskyasset(1)-1))+1:sum(n_d(1:whichisdforriskyasset(1))));
        end
        d1vals=reshape(d1_grid(Policy(whichisdforriskyasset(1),:)),[N_a,1]);
        if l_drisky>=2
            d2grid=d_grid(sum(n_d(1:whichisdforriskyasset(2)-1))+1:sum(n_d(1:whichisdforriskyasset(2))));
            d2vals=reshape(d2grid(Policy(whichisdforriskyasset(2),:)),[N_a,1]);
            if l_drisky>=3
                d3grid=d_grid(sum(n_d(1:whichisdforriskyasset(3)-1))+1:sum(n_d(1:whichisdforriskyasset(3))));
                d3vals=reshape(d3grid(Policy(whichisdforriskyasset(3),:)),[N_a,1]);
                if l_drisky>=4
                    d4grid=d_grid(sum(n_d(1:whichisdforriskyasset(4)-1))+1:sum(n_d(1:whichisdforriskyasset(4))));
                    d4vals=reshape(d4grid(Policy(whichisdforriskyasset(4),:)),[N_a,1]);
                end
            end
        end
    end
else
    if l_drisky>=1
        if whichisdforriskyasset(1)==1
            d1_grid=d_grid(1:n_d(1));
        else
            d1_grid=d_grid(sum(n_d(1:whichisdforriskyasset(1)-1))+1:sum(n_d(1:whichisdforriskyasset(1))));
        end
        d1vals=reshape(d1_grid(Policy(whichisdforriskyasset(1),:,:)),[N_a*N_z,1]);
        if l_drisky>=2
            d2grid=d_grid(sum(n_d(1:whichisdforriskyasset(2)-1))+1:sum(n_d(1:whichisdforriskyasset(2))));
            d2vals=reshape(d2grid(Policy(whichisdforriskyasset(2),:,:)),[N_a*N_z,1]);
            if l_drisky>=3
                d3grid=d_grid(sum(n_d(1:whichisdforriskyasset(3)-1))+1:sum(n_d(1:whichisdforriskyasset(3))));
                d3vals=reshape(d3grid(Policy(whichisdforriskyasset(3),:,:)),[N_a*N_z,1]);
                if l_drisky>=4
                    d4grid=d_grid(sum(n_d(1:whichisdforriskyasset(4)-1))+1:sum(n_d(1:whichisdforriskyasset(4))));
                    d4vals=reshape(d4grid(Policy(whichisdforriskyasset(4),:,:)),[N_a*N_z,1]);
                end
            end
        end
    end
end

u_gridvals=CreateGridvals(n_u,u_grid,1);
if l_u>=1
    u1vals=shiftdim(u_gridvals(:,1),-1);
    if l_u>=2
        u2vals=shiftdim(u_gridvals(:,2),-1);
        if l_u>=3
            error('Max of two u variables supported (contact if you need more)')
        end
    end
end

% Note: the relevant d for risky asset is just the 'whichisdforriskyasset' d (this is typically all d)
% riskyasset: aprime(d,u)
if l_u==1
    if l_drisky==1
        a2primeVals=arrayfun(aprimeFn, d1vals, u1vals, ParamCell{:});
    elseif l_drisky==2
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, u1vals, ParamCell{:});
    elseif l_drisky==3
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, u1vals, ParamCell{:});
    elseif l_drisky==4
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, u1vals, ParamCell{:});
    end
elseif l_u==2
    if l_drisky==1
        a2primeVals=arrayfun(aprimeFn, d1vals, u1vals, u2vals, ParamCell{:});
    elseif l_drisky==2
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, u1vals, u2vals, ParamCell{:});
    elseif l_drisky==3
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, u1vals, u2vals, ParamCell{:});
    elseif l_drisky==4
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, u1vals, u2vals, ParamCell{:});
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
    a2primeIndexes=reshape(a2primeIndexes,[N_a,1,N_u]); % Index of lower grid point
    a2primeProbs=reshape(a2primeProbs,[N_a,1,N_u]); % Probability of lower grid point
else
    a2primeIndexes=reshape(a2primeIndexes,[N_a,N_z,N_u]); % Index of lower grid point
    a2primeProbs=reshape(a2primeProbs,[N_a,N_z,N_u]); % Probability of lower grid point
end


end
