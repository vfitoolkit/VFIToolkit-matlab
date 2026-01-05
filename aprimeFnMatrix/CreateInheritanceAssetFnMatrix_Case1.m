function [a2primeIndexes,a2primeProbs]=CreateInheritanceAssetFnMatrix_Case1(aprimeFn, n_d, n_a2, n_z, n_zprime, d_grid, a2_grid, z_gridvals, zprime_gridvals, aprimeFnParams)
% Note: aprimeIndex is [N_d,N_zprime,N_z], and aprimeProbs is [N_d,N_zprime,N_z]
%
% Creates the grid points and their 'interpolation' probabilities
% Note: a2primeIndexes is always the 'lower' point (the upper points are
% just a2primeIndexes+1, so no need to waste memory storing them), and the
% a2primeProbs are the probability of this lower point (prob of upper point
% is just 1 minus this).

ParamCell=cell(length(aprimeFnParams),1);
for ii=1:length(aprimeFnParams)
    if ~isscalar(aprimeFnParams(ii))
        error('Using inheritanceasset does not allow for any of aprimeFn parameters to be anything but a scalar')
    end
    ParamCell(ii,1)={aprimeFnParams(ii)};
end

N_d=prod(n_d);
N_a2=prod(n_a2);
N_z=prod(n_z);
N_zprime=prod(n_zprime);

l_d=length(n_d);
if N_d==0
    l_d=0;
end
if l_d>4
    error('Inheritance asset does not allow for more than four of d variable (you have length(n_d)>4)')
end
l_a2=1; % hardcoded that there is only one inheritance asset
l_z=length(n_z);
if l_z>4
    error('Inheritance asset does not allow for more than four of z variable (you have length(n_z)>4)')
end

if nargin(aprimeFn)~=l_d+l_z+l_z+length(aprimeFnParams)
    error('Number of inputs to aprimeFn does not fit with size of aprimeFnParams')
end

if l_d>=1
    d1vals=d_grid(1:n_d(1));
    if l_d>=2
        d2vals=shiftdim(d_grid(n_d(1)+1:sum(n_d(1:2))),-1);
        if l_d>=3
            d3vals=shiftdim(d_grid(sum(n_d(1:2))+1:sum(n_d(1:3))),-2);
            if l_d>=4
                d4vals=shiftdim(d_grid(sum(n_d(1:3))+1:sum(n_d(1:4))),-3);
            end
        end
    end
end
% Note: By using the following sizes, the resulting a2primeVals will actually be (d,zprime,z), even though the ordering of inputs is d,z,zprime
zprime_gridvals=shiftdim(zprime_gridvals,-1);
z_gridvals=shiftdim(z_gridvals,-2);

if l_d==1
    if l_z==1
        a2primeVals=arrayfun(aprimeFn, d1vals, z_gridvals(1,1,:,1), zprime_gridvals(1,:,1), ParamCell{:});
    elseif l_z==2
        a2primeVals=arrayfun(aprimeFn, d1vals, z_gridvals(1,1,:,1),z_gridvals(1,1,:,2), zprime_gridvals(1,:,1),zprime_gridvals(1,:,2), ParamCell{:});
    elseif l_z==3
        a2primeVals=arrayfun(aprimeFn, d1vals, z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3), zprime_gridvals(1,:,1),zprime_gridvals(1,:,2),zprime_gridvals(1,:,3), ParamCell{:});
    elseif l_z==4
        a2primeVals=arrayfun(aprimeFn, d1vals, z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3),z_gridvals(1,1,:,4), zprime_gridvals(1,:,1),zprime_gridvals(1,:,2),zprime_gridvals(1,:,3),zprime_gridvals(1,:,4), ParamCell{:});
    end
elseif l_d==2
    if l_z==1
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, z_gridvals(1,1,:,1), zprime_gridvals(1,:,1), ParamCell{:});
    elseif l_z==2
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, z_gridvals(1,1,:,1),z_gridvals(1,1,:,2), zprime_gridvals(1,:,1),zprime_gridvals(1,:,2), ParamCell{:});
    elseif l_z==3
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3), zprime_gridvals(1,:,1),zprime_gridvals(1,:,2),zprime_gridvals(1,:,3), ParamCell{:});
    elseif l_z==4
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3),z_gridvals(1,1,:,4), zprime_gridvals(1,:,1),zprime_gridvals(1,:,2),zprime_gridvals(1,:,3),zprime_gridvals(1,:,4), ParamCell{:});
    end
elseif l_d==3
    if l_z==1
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, z_gridvals(1,1,:,1), zprime_gridvals(1,:,1), ParamCell{:});
    elseif l_z==2
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, z_gridvals(1,1,:,1),z_gridvals(1,1,:,2), zprime_gridvals(1,:,1),zprime_gridvals(1,:,2), ParamCell{:});
    elseif l_z==3
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3), zprime_gridvals(1,:,1),zprime_gridvals(1,:,2),zprime_gridvals(1,:,3), ParamCell{:});
    elseif l_z==4
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3),z_gridvals(1,1,:,4), zprime_gridvals(1,:,1),zprime_gridvals(1,:,2),zprime_gridvals(1,:,3),zprime_gridvals(1,:,4), ParamCell{:});
    end
elseif l_d==4
    if l_z==1
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, z_gridvals(1,1,:,1), zprime_gridvals(1,:,1), ParamCell{:});
    elseif l_z==2
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, z_gridvals(1,1,:,1),z_gridvals(1,1,:,2), zprime_gridvals(1,:,1),zprime_gridvals(1,:,2), ParamCell{:});
    elseif l_z==3
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3), zprime_gridvals(1,:,1),zprime_gridvals(1,:,2),zprime_gridvals(1,:,3), ParamCell{:});
    elseif l_z==4
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, z_gridvals(1,1,:,1),z_gridvals(1,1,:,2),z_gridvals(1,1,:,3),z_gridvals(1,1,:,4), zprime_gridvals(1,:,1),zprime_gridvals(1,:,2),zprime_gridvals(1,:,3),zprime_gridvals(1,:,4), ParamCell{:});
    end
end
% a2primeVals is (d,zprime,z). Notice that the size of zprime_gridvals and
% z_gridvals delivers this even though the order of the inputs does not
% look like this.


%% Calcuate grid indexes and probs from the values
if l_a2==1
    a2primeVals=reshape(a2primeVals,[1,N_d*N_zprime*N_z]);

    a2_griddiff=a2_grid(2:end)-a2_grid(1:end-1); % Distance between point and the next point
    
    % For small aprimeVals and a_grid, max() is faster than discretize()
    % http://discourse.vfitoolkit.com/t/example-attanasio-low-sanchez-marcos-2008/257/25
    if N_d*N_zprime*N_z*N_a2<1000000
        [~,a2primeIndexes]=max((a2_grid>a2primeVals),[],1); % Keep the dimension corresponding to aprimeVals, minimize over the a_grid dimension
        % Note, this is going to find the 'first' grid point which is bigger than a2primeVals
        % This is the 'upper' grid point
        % Have to have special treatment for trying to leave the ends of the grid (I fix these below)

        % Switch to lower grid point index
        a2primeIndexes=a2primeIndexes-1;
        a2primeIndexes(a2primeIndexes==0)=1;
        a2primeIndexes=reshape(a2primeIndexes,[N_d*N_zprime*N_z,1]);

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
    
    a2primeIndexes=reshape(a2primeIndexes,[N_d,N_zprime,N_z]);
    a2primeProbs=reshape(a2primeProbs,[N_d,N_zprime,N_z]);
end


