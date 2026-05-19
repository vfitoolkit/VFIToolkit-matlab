function [a2primeIndexes,a2primeProbs]=CreateExperienceAssetuFnMatrix_J(aprimeFn, n_d, n_a2, n_u, N_j, d_gridvals, a2_grid, u_gridvals, aprimeFnParams, aprimeIndexAsColumn) % since a2 is one-dimensional, can be a2_grid or a2_gridvals
% Age-dependent (_J) version of CreateExperienceAssetuFnMatrix: enumerate
% a2prime=aprimeFn(d, a2, u) over ALL d (used during value-function
% iteration). Continuous a2prime is linearly interpolated back on to
% a2_grid; output (lower-grid-index, prob-of-lower-grid).
%
% Companion file CreateaprimePolicyExperienceAssetu_J.m does the same but
% only for the Policy-chosen d, used in simulation / agent-distribution.
%
% Dimension layout: d in dim 1, a2 in dim 2, u in dim 3, j in dim 4.
% aprimeFnParams is passed as a [N_j, n_params] matrix; each column is
% shifted via shiftdim(...,-3) so j lives in dim 4 during arrayfun.
%
% Output sizes:
%   a2primeIndexes - shape depends on aprimeIndexAsColumn:
%                      1 => matrix [N_d*N_a2, N_u, N_j]
%                      2 => matrix [N_d, N_a2, N_u, N_j]
%   a2primeProbs   - [N_d, N_a2, N_u, N_j]

ParamCell=cell(size(aprimeFnParams,2),1);
for ii=1:size(aprimeFnParams,2)
    ParamCell(ii,1)={shiftdim(aprimeFnParams(:,ii),-3)}; % j is fourth dimension
end

N_d=prod(n_d);
N_a2=prod(n_a2);
N_u=prod(n_u);

l_d=length(n_d);
if N_d==0
    l_d=0;
end
l_a2=length(n_a2);
if l_d>4
    error('experienceassetu does not allow for more than four of d variable (you have length(n_d)>4)')
end
l_u=length(n_u);

if nargin(aprimeFn)~=l_d+l_a2+l_u+size(aprimeFnParams,2)
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
if l_u>=1
    u1vals=shiftdim(u_gridvals(:,1),-1-l_a2);
    if l_u>=2
        u2vals=shiftdim(u_gridvals(:,2),-1-l_a2);
        if l_u>=3
            u3vals=shiftdim(u_gridvals(:,3),-1-l_a2);
            if l_u>=4
                u4vals=shiftdim(u_gridvals(:,4),-1-l_a2);
                if l_u>=5
                    error('Max of four u variables supported (contact if you need more)')
                end
            end
        end
    end
end

if l_u==1
    if l_d==1
        a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, u1vals, ParamCell{:});
    elseif l_d==2
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, u1vals, ParamCell{:});
    elseif l_d==3
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, u1vals, ParamCell{:});
    elseif l_d==4
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, u1vals, ParamCell{:});
    end
elseif l_u==2
    if l_d==1
        a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, u1vals,u2vals, ParamCell{:});
    elseif l_d==2
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, u1vals,u2vals, ParamCell{:});
    elseif l_d==3
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, u1vals,u2vals, ParamCell{:});
    elseif l_d==4
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, u1vals,u2vals, ParamCell{:});
    end
elseif l_u==3
    if l_d==1
        a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, u1vals,u2vals,u3vals, ParamCell{:});
    elseif l_d==2
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, u1vals,u2vals,u3vals, ParamCell{:});
    elseif l_d==3
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, u1vals,u2vals,u3vals, ParamCell{:});
    elseif l_d==4
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, u1vals,u2vals,u3vals, ParamCell{:});
    end
elseif l_u==4
    if l_d==1
        a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
    elseif l_d==2
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
    elseif l_d==3
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
    elseif l_d==4
        a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, u1vals,u2vals,u3vals,u4vals, ParamCell{:});
    end
end

%% Calcuate grid indexes and probs from the values
if l_a2==1
    a2primeVals=reshape(a2primeVals,[1,N_d*N_a2*N_u,N_j]);

    a2_griddiff=a2_grid(2:end)-a2_grid(1:end-1);

    if N_d*N_a2*N_u*N_j*N_a2<1000000
        [~,a2primeIndexes]=max((a2_grid>a2primeVals),[],1);
        a2primeIndexes=a2primeIndexes-1;
        a2primeIndexes(a2primeIndexes==0)=1;
        a2primeIndexes=reshape(a2primeIndexes,[N_d*N_a2*N_u,N_j]);

        aprime_residual=reshape(a2primeVals,[N_d*N_a2*N_u,N_j])-a2_grid(a2primeIndexes);
        a2primeProbs=1-aprime_residual./a2_griddiff(a2primeIndexes);

        offTopOfGrid=(a2primeVals>=a2_grid(end));
        a2primeIndexes(offTopOfGrid)=n_a2-1;
        a2primeProbs(offTopOfGrid)=0;
        offBottomOfGrid=(a2primeVals<=a2_grid(1));
        a2primeProbs(offBottomOfGrid)=1;
    else
        a2primeIndexes=discretize(a2primeVals,a2_grid);
        a2primeIndexes=reshape(a2primeIndexes,[N_d*N_a2*N_u,N_j]);

        offBottomOfGrid=(a2primeVals<=a2_grid(1));
        a2primeIndexes(offBottomOfGrid)=1;
        offTopOfGrid=(a2primeVals>=a2_grid(end));
        a2primeIndexes(offTopOfGrid)=n_a2-1;

        aprime_residual=reshape(a2primeVals,[N_d*N_a2*N_u,N_j])-a2_grid(a2primeIndexes);
        a2primeProbs=1-aprime_residual./a2_griddiff(a2primeIndexes);
        a2primeProbs(offBottomOfGrid)=1;
        a2primeProbs(offTopOfGrid)=0;
    end

    if aprimeIndexAsColumn==1
        a2primeIndexes=reshape(a2primeIndexes,[N_d*N_a2,N_u,N_j]);
    else
        a2primeIndexes=reshape(a2primeIndexes,[N_d,N_a2,N_u,N_j]);
    end
    a2primeProbs=reshape(a2primeProbs,[N_d,N_a2,N_u,N_j]);
end


end
