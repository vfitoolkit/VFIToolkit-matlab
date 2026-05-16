function [a2primeIndexes,a2primeProbs]=CreateExperienceAssetzeFnMatrix_J(aprimeFn, n_d, n_a2, n_z, n_e, N_j, d_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, aprimeFnParams, aprimeIndexAsColumn) % since a2 is one-dimensional, can be a2_grid or a2_gridvals
% Age-dependent (_J) version of CreateExperienceAssetzeFnMatrix: enumerate
% a2prime=aprimeFn(d, a2, z, e) over ALL d (used during value-function
% iteration).
%
% Companion file CreateaprimePolicyExperienceAssetze_J.m does the same but
% only for the Policy-chosen d, used in simulation / agent-distribution.
%
% Dimension layout: d in dim 1, a2 in dim 2, z in dim 3, e in dim 4, j in dim 5.
% aprimeFnParams is a [N_j, n_params] matrix; each column is shifted via
% shiftdim(...,-4) so j lives in dim 5 during arrayfun.
%
% Output sizes:
%   a2primeIndexes - shape depends on aprimeIndexAsColumn:
%                      1 => matrix [N_d*N_a2, N_z, N_e, N_j]
%                      2 => matrix [N_d, N_a2, N_z, N_e, N_j]
%   a2primeProbs   - [N_d, N_a2, N_z, N_e, N_j]

ParamCell=cell(size(aprimeFnParams,2),1);
for ii=1:size(aprimeFnParams,2)
    ParamCell(ii,1)={shiftdim(aprimeFnParams(:,ii),-4)}; % j is fifth dimension
end

N_d=prod(n_d);
N_a2=prod(n_a2);
N_z=prod(n_z);
N_e=prod(n_e);

l_d=length(n_d);
if N_d==0
    l_d=0;
end
l_a2=length(n_a2);
if l_d>4
    error('experienceassetze does not allow for more than four of d variable (you have length(n_d)>4)')
end
l_z=length(n_z);
l_e=length(n_e);

if nargin(aprimeFn)~=l_d+l_a2+l_z+l_e+size(aprimeFnParams,2)
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
% permute once: [N_z, l_z, N_j] -> [1, 1, N_z, 1, N_j, l_z]; then index final dim per z column
z_gridvals_Jp=permute(z_gridvals_J,[4,5,1,6,3,2]);
if l_z>=1
    z1vals=z_gridvals_Jp(:,:,:,:,:,1);   % [1, 1, N_z, 1, N_j]
    if l_z>=2
        z2vals=z_gridvals_Jp(:,:,:,:,:,2);
        if l_z>=3
            z3vals=z_gridvals_Jp(:,:,:,:,:,3);
            if l_z>=4
                z4vals=z_gridvals_Jp(:,:,:,:,:,4);
            end
        end
    end
end
% permute once: [N_e, l_e, N_j] -> [1, 1, 1, N_e, N_j, l_e]; then index final dim per e column
e_gridvals_Jp=permute(e_gridvals_J,[4,5,6,1,3,2]);
if l_e>=1
    e1vals=e_gridvals_Jp(:,:,:,:,:,1);   % [1, 1, 1, N_e, N_j]
    if l_e>=2
        e2vals=e_gridvals_Jp(:,:,:,:,:,2);
        if l_e>=3
            e3vals=e_gridvals_Jp(:,:,:,:,:,3);
            if l_e>=4
                e4vals=e_gridvals_Jp(:,:,:,:,:,4);
            end
        end
    end
end

zecell={};
if l_z>=1, zecell{end+1}=z1vals; end
if l_z>=2, zecell{end+1}=z2vals; end
if l_z>=3, zecell{end+1}=z3vals; end
if l_z>=4, zecell{end+1}=z4vals; end
if l_e>=1, zecell{end+1}=e1vals; end
if l_e>=2, zecell{end+1}=e2vals; end
if l_e>=3, zecell{end+1}=e3vals; end
if l_e>=4, zecell{end+1}=e4vals; end

if l_d==1
    a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, zecell{:}, ParamCell{:});
elseif l_d==2
    a2primeVals=arrayfun(aprimeFn, d1vals,d2vals, a2vals, zecell{:}, ParamCell{:});
elseif l_d==3
    a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals, a2vals, zecell{:}, ParamCell{:});
elseif l_d==4
    a2primeVals=arrayfun(aprimeFn, d1vals,d2vals,d3vals,d4vals, a2vals, zecell{:}, ParamCell{:});
end

%% Calcuate grid indexes and probs from the values
if l_a2==1
    a2primeVals=reshape(a2primeVals,[1,N_d*N_a2*N_z*N_e,N_j]);

    a2_griddiff=a2_grid(2:end)-a2_grid(1:end-1);

    if N_d*N_a2*N_z*N_e*N_j*N_a2<1000000
        [~,a2primeIndexes]=max((a2_grid>a2primeVals),[],1);
        a2primeIndexes=a2primeIndexes-1;
        a2primeIndexes(a2primeIndexes==0)=1;
        a2primeIndexes=reshape(a2primeIndexes,[N_d*N_a2*N_z*N_e,N_j]);

        aprime_residual=reshape(a2primeVals,[N_d*N_a2*N_z*N_e,N_j])-a2_grid(a2primeIndexes);
        a2primeProbs=1-aprime_residual./a2_griddiff(a2primeIndexes);

        offTopOfGrid=(a2primeVals>=a2_grid(end));
        a2primeIndexes(offTopOfGrid)=n_a2-1;
        a2primeProbs(offTopOfGrid)=0;
        offBottomOfGrid=(a2primeVals<=a2_grid(1));
        a2primeProbs(offBottomOfGrid)=1;
    else
        a2primeIndexes=discretize(a2primeVals,a2_grid);
        a2primeIndexes=reshape(a2primeIndexes,[N_d*N_a2*N_z*N_e,N_j]);

        offBottomOfGrid=(a2primeVals<=a2_grid(1));
        a2primeIndexes(offBottomOfGrid)=1;
        offTopOfGrid=(a2primeVals>=a2_grid(end));
        a2primeIndexes(offTopOfGrid)=n_a2-1;

        aprime_residual=reshape(a2primeVals,[N_d*N_a2*N_z*N_e,N_j])-a2_grid(a2primeIndexes);
        a2primeProbs=1-aprime_residual./a2_griddiff(a2primeIndexes);
        a2primeProbs(offBottomOfGrid)=1;
        a2primeProbs(offTopOfGrid)=0;
    end

    if aprimeIndexAsColumn==1
        a2primeIndexes=reshape(a2primeIndexes,[N_d*N_a2,N_z,N_e,N_j]);
    else
        a2primeIndexes=reshape(a2primeIndexes,[N_d,N_a2,N_z,N_e,N_j]);
    end
    a2primeProbs=reshape(a2primeProbs,[N_d,N_a2,N_z,N_e,N_j]);
end


end
