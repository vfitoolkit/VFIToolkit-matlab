function [a2primeIndexes,a2primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d, n_a2, n_e, d_gridvals, a2_grid, e_gridvals, aprimeFnParams, aprimeIndexAsColumn)  % since a2 is one-dimensional, can be a2_grid or a2_gridvals
% For experienceassete: enumerate a2prime=aprimeFn(d, a2, e) over ALL d
% (used during value-function iteration). Because the true value of a2prime
% will (almost always) lie between two consecutive points in a2_grid, it is
% linearly interpolated back on to a2_grid. Thus the continuous a2prime is
% represented by (index of lower grid point in a2primeIndexes, probability
% of lower grid point in a2primeProbs) on a2_grid; the upper index is
% implicitly lower+1 with prob 1-minus-prob-of-lower.
%
% Companion file CreateaprimePolicyExperienceAssete.m does the same but only
% for the Policy-chosen d (one d per state), used in simulation /
% agent-distribution. This file is used during value-function iteration
% where every d must be evaluated.
%
% Output sizes:
%   a2primeIndexes - shape depends on aprimeIndexAsColumn:
%                      1 => column vector [N_d*N_a2*N_e, 1]
%                      2 => matrix [N_d, N_a2, N_e]
%                      3 => matrix [N_d*N_a2, N_e]
%   a2primeProbs   - [N_d, N_a2, N_e]

ParamCell=cell(length(aprimeFnParams),1);
for ii=1:length(aprimeFnParams)
    if size(aprimeFnParams(ii))~=[1,1]
        error('Using experienceassete does not allow for any of aprimeFn parameters to be anything but a scalar')
    end
    ParamCell(ii,1)={aprimeFnParams(ii)};
end

N_d=prod(n_d);
N_a2=prod(n_a2);
N_e=prod(n_e);

l_d=length(n_d);
if N_d==0
    l_d=0;
end
l_a2=length(n_a2);
if l_d>4
    error('experienceassete does not allow for more than four of d variable (you have length(n_d)>4)')
end
l_e=length(n_e);

if nargin(aprimeFn)~=l_d+l_a2+l_e+length(aprimeFnParams)
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
if l_e>=1
    e1vals=shiftdim(e_gridvals(:,1),-1-l_a2);
    if l_e>=2
        e2vals=shiftdim(e_gridvals(:,2),-1-l_a2);
        if l_e>=3
            e3vals=shiftdim(e_gridvals(:,3),-1-l_a2);
            if l_e>=4
                e4vals=shiftdim(e_gridvals(:,4),-1-l_a2);
            end
        end
    end
end

ecell={};
if l_e>=1, ecell{end+1}=e1vals; end
if l_e>=2, ecell{end+1}=e2vals; end
if l_e>=3, ecell{end+1}=e3vals; end
if l_e>=4, ecell{end+1}=e4vals; end

if l_d==1
    a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, ecell{:}, ParamCell{:});
elseif l_d==2
    a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, ecell{:}, ParamCell{:});
elseif l_d==3
    a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, ecell{:}, ParamCell{:});
elseif l_d==4
    a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, ecell{:}, ParamCell{:});
end

%% Calcuate grid indexes and probs from the values
if l_a2==1
    a2primeVals=reshape(a2primeVals,[1,N_d*N_a2*N_e]);

    a2_griddiff=a2_grid(2:end)-a2_grid(1:end-1); % Distance between point and the next point

    % For small aprimeVals and a_grid, max() is faster than discretize()
    if N_d*N_a2*N_e*N_a2<1000000
        [~,a2primeIndexes]=max((a2_grid>a2primeVals),[],1);
        a2primeIndexes=a2primeIndexes-1;
        a2primeIndexes(a2primeIndexes==0)=1;

        aprime_residual=a2primeVals'-a2_grid(a2primeIndexes);
        a2primeProbs=1-aprime_residual./a2_griddiff(a2primeIndexes);

        offTopOfGrid=(a2primeVals>=a2_grid(end));
        a2primeIndexes(offTopOfGrid)=n_a2-1;
        a2primeProbs(offTopOfGrid)=0;
        offBottomOfGrid=(a2primeVals<=a2_grid(1));
        a2primeProbs(offBottomOfGrid)=1;
    else
        a2primeIndexes=discretize(a2primeVals,a2_grid);

        offBottomOfGrid=(a2primeVals<=a2_grid(1));
        a2primeIndexes(offBottomOfGrid)=1;
        offTopOfGrid=(a2primeVals>=a2_grid(end));
        a2primeIndexes(offTopOfGrid)=n_a2-1;

        aprime_residual=a2primeVals'-a2_grid(a2primeIndexes);
        a2primeProbs=1-aprime_residual./a2_griddiff(a2primeIndexes);
        a2primeProbs(offBottomOfGrid)=1;
        a2primeProbs(offTopOfGrid)=0;
    end

    if aprimeIndexAsColumn==1
        a2primeIndexes=a2primeIndexes';
    elseif aprimeIndexAsColumn==3
        a2primeIndexes=reshape(a2primeIndexes,[N_d*N_a2,N_e]);
    else
        a2primeIndexes=reshape(a2primeIndexes,[N_d,N_a2,N_e]);
    end
    a2primeProbs=reshape(a2primeProbs,[N_d,N_a2,N_e]);
end


end
