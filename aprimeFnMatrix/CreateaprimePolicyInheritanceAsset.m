function [a2primeIndexes, a2primeProbs]=CreateaprimePolicyInheritanceAsset(Policy,aprimeFn, whichisdforinheritasset, n_d, n_a1,n_a2, n_z, n_zprime, d_grid, a2_grid, z_gridvals, zprime_gridvals, aprimeFnParams)
% The input Policy will contain aprime (except for the inheritance asset)
% and the decision variables (d2, and where applicable d1). The output is
% just the Policy for a2prime (the inheritance asset). As well as the
% related probabilities.

% Note: aprimeIndex is [N_a,N_z,N_zprime], so is aprimeProbs which only reports the probability for the lower grid point.
%
% Creates the grid points and their 'interpolation' probabilities.
% Note: aprimeIndexes is always the 'lower' point (the upper points are
% just aprimeIndexes+1, so no need to waste memory storing them), and the
% aprimeProbs are the probability of this lower point (prob of upper point
% is just 1 minus this).
%
%
% Remark: This is like CreateInheritanceAssetFnMatrix_Case1(), except
% instead of looking at all possible d, we only care about those in Policy.
% The Policy based ones are needed for simulation, while those for all
% possible d were needed for value function (to find Policy).

ParamCell=cell(length(aprimeFnParams),1);
for ii=1:length(aprimeFnParams)
    if ~isscalar(aprimeFnParams(ii))
        error('Using inheritance asset does not allow for any of aprimeFn parameters to be anything but a scalar')
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
N_zprime=prod(n_zprime);

l_dinherit=length(whichisdforinheritasset); % number of decision variables used for the inheritance asset

l_z=length(n_z);
l_zprime=length(n_zprime);

if nargin(aprimeFn)~=l_dinherit+l_z+l_zprime+length(aprimeFnParams)
    error('Number of inputs to aprimeFn does not fit with size of aprimeFnParams')
end

if l_dinherit>=1
    if whichisdforinheritasset(1)==1
        d1grid=d_grid(1:n_d(1));
    else
        d1grid=d_grid(sum(n_d(1:whichisdforinheritasset(1)-1))+1:sum(n_d(1:whichisdforinheritasset(1))));
    end
    d1vals=reshape(d1grid(Policy(whichisdforinheritasset(1),:,:)),[N_a,N_z]);
    if l_dinherit>=2
        d2grid=d_grid(sum(n_d(1:whichisdforinheritasset(2)-1))+1:sum(n_d(1:whichisdforinheritasset(2))));
        d2vals=reshape(d2grid(Policy(whichisdforinheritasset(2),:,:)),[N_a,N_z]);
        if l_dinherit>=3
            d3grid=d_grid(sum(n_d(1:whichisdforinheritasset(3)-1))+1:sum(n_d(1:whichisdforinheritasset(3))));
            d3vals=reshape(d3grid(Policy(whichisdforinheritasset(3),:,:)),[N_a,N_z]);
            if l_dinherit>=4
                d4grid=d_grid(sum(n_d(1:whichisdforinheritasset(4)-1))+1:sum(n_d(1:whichisdforinheritasset(4))));
                d4vals=reshape(d4grid(Policy(whichisdforinheritasset(4),:,:)),[N_a,N_z]);
            end
        end
    end
end


% Note: the relevant d for experience asset is just the 'whichisdforinheritasset' d (this is typically just the last if using just inheritance asset, but
% needs to be something else, e.g., when combining inheritance asset with semi-exogenous state)
% inheritanceasset: aprime(d,z,zprime)

% Note: Following are different to how they are in CreateInheritanceAssetFnMatrix_Case1()
z_gridvals=shiftdim(z_gridvals,-1);
zprime_gridvals=shiftdim(zprime_gridvals,-2);

if l_dinherit==1
    if l_z==1
        a2primeVals=arrayfun(aprimeFn, d1vals, z_gridvals(1,:,1), zprime_gridvals(1,1,:,1), ParamCell{:});
    elseif l_z==2
        a2primeVals=arrayfun(aprimeFn, d1vals, z_gridvals(1,:,1),z_gridvals(1,:,2), zprime_gridvals(1,1,:,1),zprime_gridvals(1,1,:,2), ParamCell{:});
    elseif l_z==3
        a2primeVals=arrayfun(aprimeFn, d1vals, z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3), zprime_gridvals(1,1,:,1),zprime_gridvals(1,1,:,2),zprime_gridvals(1,1,:,3), ParamCell{:});
    elseif l_z==4
        a2primeVals=arrayfun(aprimeFn, d1vals, z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3),z_gridvals(1,:,4), zprime_gridvals(1,1,:,1),zprime_gridvals(1,1,:,2),zprime_gridvals(1,1,:,3),zprime_gridvals(1,1,:,4), ParamCell{:});
    end
elseif l_dinherit==2
    if l_z==1
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, z_gridvals(1,:,1), zprime_gridvals(1,1,:,1), ParamCell{:});
    elseif l_z==2
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, z_gridvals(1,:,1),z_gridvals(1,:,2), zprime_gridvals(1,1,:,1),zprime_gridvals(1,1,:,2), ParamCell{:});
    elseif l_z==3
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3), zprime_gridvals(1,1,:,1),zprime_gridvals(1,1,:,2),zprime_gridvals(1,1,:,3), ParamCell{:});
    elseif l_z==4
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3),z_gridvals(1,:,4), zprime_gridvals(1,1,:,1),zprime_gridvals(1,1,:,2),zprime_gridvals(1,1,:,3),zprime_gridvals(1,1,:,4), ParamCell{:});
    end
elseif l_dinherit==3
    if l_z==1
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, z_gridvals(1,:,1), zprime_gridvals(1,1,:,1), ParamCell{:});
    elseif l_z==2
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, z_gridvals(1,:,1),z_gridvals(1,:,2), zprime_gridvals(1,1,:,1),zprime_gridvals(1,1,:,2), ParamCell{:});
    elseif l_z==3
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3), zprime_gridvals(1,1,:,1),zprime_gridvals(1,1,:,2),zprime_gridvals(1,1,:,3), ParamCell{:});
    elseif l_z==4
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3),z_gridvals(1,:,4), zprime_gridvals(1,1,:,1),zprime_gridvals(1,1,:,2),zprime_gridvals(1,1,:,3),zprime_gridvals(1,1,:,4), ParamCell{:});
    end
elseif l_dinherit==4
    if l_z==1
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, z_gridvals(1,:,1), zprime_gridvals(1,1,:,1), ParamCell{:});
    elseif l_z==2
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, z_gridvals(1,:,1),z_gridvals(1,:,2), zprime_gridvals(1,1,:,1),zprime_gridvals(1,1,:,2), ParamCell{:});
    elseif l_z==3
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3), zprime_gridvals(1,1,:,1),zprime_gridvals(1,1,:,2),zprime_gridvals(1,1,:,3), ParamCell{:});
    elseif l_z==4
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3),z_gridvals(1,:,4), zprime_gridvals(1,1,:,1),zprime_gridvals(1,1,:,2),zprime_gridvals(1,1,:,3),zprime_gridvals(1,1,:,4), ParamCell{:});
    end
end
% a2primeVals is (d,z,zprime)


%% Calcuate grid indexes and probs from the values
a2primeVals=reshape(a2primeVals,[1,N_a*N_z*N_zprime]);

a2_griddiff=a2_grid(2:end)-a2_grid(1:end-1); % Distance between point and the next point

a2primeIndexes=discretize(a2primeVals,a2_grid); % Finds the lower grid point
% Have to have special treatment for trying to leave the ends of the grid

% Those points which tried to leave the bottom of the grid have probability 0 of the 'upper' point (1 of lower point)
offBottomOfGrid=(a2primeVals<=a2_grid(1));
a2primeIndexes(offBottomOfGrid)=1; % Has already been handled
% Those points which tried to leave the top of the grid have probability 1 of the 'upper' point (0 of lower point)
offTopOfGrid=(a2primeVals>=a2_grid(end));
a2primeIndexes(offTopOfGrid)=n_a2-1; % lower grid point is the one before the end point
a2primeIndexes=reshape(a2primeIndexes,[N_a*N_z*N_zprime,1]);

% Now, find the probabilities
aprime_residual=a2primeVals'-a2_grid(a2primeIndexes);
% Probability of the 'lower' points
a2primeProbs=1-aprime_residual./a2_griddiff(a2primeIndexes);
% And clean up the ends of the grid
a2primeProbs(offBottomOfGrid)=1;
a2primeProbs(offTopOfGrid)=0;

a2primeIndexes=reshape(a2primeIndexes,[N_a,N_z,N_zprime]); % Index of lower grid point
a2primeProbs=reshape(a2primeProbs,[N_a,N_z,N_zprime]); % Probability of lower grid point


end
