function [a2primeIndexes,a2primeProbs]=CreateaprimeFnMatrix_RiskyAsset(aprimeFn, n_d, n_a2, n_u, d_grid, a2_grid, u_grid, aprimeFnParams, aprimeIndexAsColumn)
% Note: a2primeIndex is [N_a2*N_u,1], whereas a2primeProbs is [N_a2,N_u]
%
% Creates the grid points and their 'interpolation' probabilities
% Note: a2primeIndexes is always the 'lower' point (the upper points are
% just a2primeIndexes+1, so no need to waste memory storing them), and the
% a2primeProbs are the probability of this lower point (prob of upper point
% is just 1 minus this).

ParamCell=cell(length(aprimeFnParams),1);
for ii=1:length(aprimeFnParams)
    if ~isscalar(aprimeFnParams(ii))
        error('Using riskyasset does not allow for any of aprimeFn parameters to be anything but a scalar')
    end
    ParamCell(ii,1)={aprimeFnParams(ii)};
end

N_d=prod(n_d);
N_a2=prod(n_a2);
N_u=prod(n_u);

l_d=length(n_d);
if N_d==0
    l_d=0;
end
l_a2=length(n_a2);
l_u=length(n_u);
if l_d>4
    error('The aprimeFn does not allow for more than four of d variable (you have length(n_d)>4)')
end
if l_u>5
    error('The aprimeFn does not allow for more than five of u variable (you have length(n_u)>5)')
end

if nargin(aprimeFn)~=l_d+l_u+length(aprimeFnParams)
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
if all(size(u_grid)==[sum(n_u),1]) % kroneker product u_grid
    if l_u>=1
        u1vals=shiftdim(u_grid(1:n_u(1)),-l_d);
        if l_u>=2
            u2vals=shiftdim(u_grid(n_u(1)+1:n_u(1)+n_u(2)),-l_d-1);
            if l_u>=3
                u3vals=shiftdim(u_grid(sum(n_u(1:2))+1:sum(n_u(1:3))),-l_d-2);
                if l_u>=4
                    u4vals=shiftdim(u_grid(sum(n_u(1:3))+1:sum(n_u(1:4))),-l_d-3);
                    if l_u>=5
                        u5vals=shiftdim(u_grid(sum(n_u(1:4))+1:sum(n_u(1:5))),-l_d-4);
                    end
                end
            end
        end
    end
elseif all(size(u_grid)==[prod(n_u),l_u]) % joint u_grid
    if l_u>=1
        u1vals=shiftdim(u_grid(:,1),-l_d);
        if l_u>=2
            u2vals=shiftdim(u_grid(:,2),-l_d);
            if l_u>=3
                u3vals=shiftdim(u_grid(:,3),-l_d);
                if l_u>=4
                    u4vals=shiftdim(u_grid(:,4),-l_d);
                    if l_u>=5
                        u5vals=shiftdim(u_grid(:,5),-l_d);
                    end
                end
            end
        end
    end
end

if l_d==1
    if l_u==1
        d1vals(1,1,1,1)=d_grid(1); % Requires special treatment
        a2primeVals=arrayfun(aprimeFn, d1vals, u1vals, ParamCell{:});
    elseif l_u==2
        a2primeVals=arrayfun(aprimeFn, d1vals, u1vals,u2vals, ParamCell{:});
    elseif l_u==3
        a2primeVals=arrayfun(aprimeFn, d1vals, u1vals,u2vals,u3vals, ParamCell{:});
    elseif l_u==4
        a2primeVals=arrayfun(aprimeFn, d1vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
    elseif l_u==5
        a2primeVals=arrayfun(aprimeFn, d1vals, u1vals,u2vals,u3vals,u4vals,u5vals, ParamCell{:});
    end
elseif l_d==2
    if l_u==1
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, u1vals, ParamCell{:});
    elseif l_u==2
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, u1vals,u2vals, ParamCell{:});
    elseif l_u==3
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, u1vals,u2vals,u3vals, ParamCell{:});
    elseif l_u==4
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
    elseif l_u==5
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, u1vals,u2vals,u3vals,u4vals,u5vals, ParamCell{:});
    end
elseif l_d==3
    if l_u==1
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, u1vals, ParamCell{:});
    elseif  l_u==2
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, u1vals,u2vals, ParamCell{:});
    elseif  l_u==3
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, u1vals,u2vals,u3vals, ParamCell{:});
    elseif  l_u==4
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
    elseif  l_u==5
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, u1vals,u2vals,u3vals,u4vals,u5vals, ParamCell{:});
    end
elseif l_d==4
    if l_u==1
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, u1vals, ParamCell{:});
    elseif l_u==2
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, u1vals,u2vals, ParamCell{:});
    elseif l_u==3
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, u1vals,u2vals,u3vals, ParamCell{:});
    elseif l_u==4
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
    elseif l_u==5
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, u1vals,u2vals,u3vals,u4vals,u5vals, ParamCell{:});
    end
end


%% Calcuate grid indexes and probs from the values
if l_a2==1
    a2primeVals=reshape(a2primeVals,[1,N_d*N_u]);

    a2_griddiff=a2_grid(2:end)-a2_grid(1:end-1); % Distance between point and the next point
    
    % For small aprimeVals and a_grid, max() is faster than discretize()
    % http://discourse.vfitoolkit.com/t/example-attanasio-low-sanchez-marcos-2008/257/25
    if N_d*N_u*N_a2<1000000
        [~,a2primeIndexes]=max((a2_grid>a2primeVals),[],1); % Keep the dimension corresponding to aprimeVals, minimize over the a_grid dimension
        % Note, this is going to find the 'first' grid point which is bigger than aprimeVals
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
    
    if aprimeIndexAsColumn==1 % value fn codes want column, simulation codes want matrix
%     aprimeIndexes=reshape(aprimeIndexes,[N_d*N_u,1]);
        a2primeIndexes=a2primeIndexes'; % This is just doing the commented out reshape above
    else
        a2primeIndexes=reshape(a2primeIndexes,[N_d,N_u]);
    end
    a2primeProbs=reshape(a2primeProbs,[N_d,N_u]);
end


