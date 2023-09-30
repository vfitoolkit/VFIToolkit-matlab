function [aprimeIndexes, aprimeProbs]=CreateaprimePolicyExperienceAsset_Case1(Policy,aprimeFn, whichisdforexpasset, n_a, N_z, a_grid, aprimeFnParams)
% The input Policy will contain aprime (except for the experience asset)
% and the decision variables (d2, and where applicable d1). The output is
% just the Policy for aprime (for all endogenous states). As well as the
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

N_a=prod(n_a);

if nargin(aprimeFn)~=1+1+length(aprimeFnParams)
    error('ERROR: Number of inputs to aprimeFn does not fit with size of aprimeFnParams')
end

% Note: hardcodes that the relevant a for experience asset is just the last a
if length(n_a)==1
    a2vals=a_grid;
else
    a2vals=kron(a_grid(end-n_a(end)+1:end), ones(sum(n_a(1:end-1)),1) ); % Experience asset is the last asset
end

% Note: the relevant d for experience asset is just the
% 'whichisdforexpasset' d (this is l_d if using just experience asset, but
% needs to be something else, e.g., when combining experience asset with
% semi-exogenous state)
if N_z==1
    aprimeVals=arrayfun(aprimeFn, shiftdim(Policy(whichisdforexpasset,:),1), a2vals, ParamCell{:});  % [N_a,1]
else
    aprimeVals=arrayfun(aprimeFn, shiftdim(Policy(whichisdforexpasset,:,:),1), a2vals, ParamCell{:});  % [N_a,N_z]
end

%% Calcuate grid indexes and probs from the values
aprimeVals=reshape(aprimeVals,[1,N_a*N_z]);

expasset_grid=a_grid(end-n_a(end)+1:end);

a_griddiff=expasset_grid(2:end)-expasset_grid(1:end-1); % Distance between point and the next point

temp=expasset_grid-aprimeVals;
temp(temp>0)=1; % Equals 1 when a_grid is greater than aprimeVals

[~,aprimeIndexes]=max(temp,[],1); % Keep the dimension corresponding to aprimeVals, minimize over the a_grid dimension
% Note, this is going to find the 'first' grid point such that aprimeVals is smaller than or equal to that grid point
% This is the 'upper' grid point

% Switch to lower grid point index
aprimeIndexes=aprimeIndexes-1;
aprimeIndexes(aprimeIndexes==0)=1;

% Now, find the probabilities
aprime_residual=aprimeVals'-expasset_grid(aprimeIndexes);
% Probability of the 'lower' points
aprimeProbs=1-aprime_residual./a_griddiff(aprimeIndexes);

% Those points which tried to leave the top of the grid have probability 1 of the 'upper' point (0 of lower point)
offTopOfGrid=(aprimeVals>=expasset_grid(end));
aprimeProbs(offTopOfGrid)=0;
% Those points which tried to leave the bottom of the grid have probability 0 of the 'upper' point (1 of lower point)
offBottomOfGrid=(aprimeVals<=expasset_grid(1));
aprimeProbs(offBottomOfGrid)=1;

aprimeIndexes=reshape(aprimeIndexes,[N_a,N_z]); % Index of lower grid point

aprimeProbs=reshape(aprimeProbs,[N_a,N_z]); % Probability of lower grid point

end
