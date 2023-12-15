function [a2primeIndexes, a2primeProbs]=CreateaprimePolicyExperienceAsset_Case1(Policy,aprimeFn, whichisdforexpasset, n_d, n_a1,n_a2, N_z, d_grid, a2_grid, aprimeFnParams)
% The input Policy will contain aprime (except for the experience asset)
% and the decision variables (d2, and where applicable d1). The output is
% just the Policy for a2prime (the experience asset). As well as the
% related probabilities.

% Note: aprimeIndex is [N_a,N_z], so is aprimeProbs which only reports the probability for the lower grid point.
%
% Creates the grid points and their 'interpolation' probabilities.
% Note: aprimeIndexes is always the 'lower' point (the upper points are
% just aprimeIndexes+1, so no need to waste memory storing them), and the
% aprimeProbs are the probability of this lower point (prob of upper point
% is just 1 minus this).
%
%
% Remark: This is like CreateExperienceAssetFnMatrix_Case1(), except
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

l_dexp=length(whichisdforexpasset);

if nargin(aprimeFn)~=l_dexp+1+length(aprimeFnParams)
    error('ERROR: Number of inputs to aprimeFn does not fit with size of aprimeFnParams')
end

if N_z==0 % To save writing a seperate script for without z
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

if N_a1==0
    if N_z==0
        a2vals=a2_grid;
    else
        a2vals=kron(ones(N_z,1),a2_grid);
    end
else
    if N_z==0
        a2vals=kron(a2_grid,ones(N_a1,1));
    else
        a2vals=kron(ones(N_z,1),kron(a2_grid,ones(N_a1,1)));
    end
end


% Note: the relevant d for experience asset is just the 'whichisdforexpasset' d (this is typically just the last if using just experience asset, but
% needs to be something else, e.g., when combining experience asset with semi-exogenous state)
% expasset: aprime(d,a2)
a2vals=a2vals.*ones(1,1,1,'gpuArray'); % this is just to fool matlab which otherwise throws an error
if l_dexp==1
    a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, ParamCell{:});
elseif l_dexp==2
    a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, ParamCell{:});
elseif l_dexp==3
    a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, ParamCell{:});
elseif l_dexp==4
    a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, ParamCell{:});
end


%% Calcuate grid indexes and probs from the values
if N_z==0
    a2primeVals=reshape(a2primeVals,[1,N_a]);
else
    a2primeVals=reshape(a2primeVals,[1,N_a*N_z]);
end

expasset_grid=a2_grid;

a_griddiff=expasset_grid(2:end)-expasset_grid(1:end-1); % Distance between point and the next point

% temp=expasset_grid-aprimeVals;
% temp(temp>0)=1; % Equals 1 when a_grid is greater than aprimeVals
% 
% [~,a2primeIndexes]=max(temp,[],1); % Keep the dimension corresponding to aprimeVals, minimize over the a_grid dimension
[~,a2primeIndexes]=max((expasset_grid-a2primeVals>0),[],1); % Keep the dimension corresponding to aprimeVals, minimize over the a_grid dimension
% Note, this is going to find the 'first' grid point such that aprimeVals is smaller than or equal to that grid point
% This is the 'upper' grid point

% Switch to lower grid point index
a2primeIndexes=a2primeIndexes-1;
a2primeIndexes(a2primeIndexes==0)=1;

% Now, find the probabilities
aprime_residual=a2primeVals'-expasset_grid(a2primeIndexes);
% Probability of the 'lower' points
a2primeProbs=1-aprime_residual./a_griddiff(a2primeIndexes);

% Those points which tried to leave the top of the grid have probability 1 of the 'upper' point (0 of lower point)
offTopOfGrid=(a2primeVals>=expasset_grid(end));
a2primeProbs(offTopOfGrid)=0;
% Those points which tried to leave the bottom of the grid have probability 0 of the 'upper' point (1 of lower point)
offBottomOfGrid=(a2primeVals<=expasset_grid(1));
a2primeProbs(offBottomOfGrid)=1;

if N_z==0
    a2primeIndexes=reshape(a2primeIndexes,[N_a,1]); % Index of lower grid point
    a2primeProbs=reshape(a2primeProbs,[N_a,1]); % Probability of lower grid point
else
    a2primeIndexes=reshape(a2primeIndexes,[N_a,N_z]); % Index of lower grid point
    a2primeProbs=reshape(a2primeProbs,[N_a,N_z]); % Probability of lower grid point
end


end
