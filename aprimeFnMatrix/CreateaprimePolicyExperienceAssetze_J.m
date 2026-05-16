function [a2primeIndexes, a2primeProbs]=CreateaprimePolicyExperienceAssetze_J(Policy,aprimeFn, whichisdforexpasset, n_d, n_a1,n_a2, n_z, n_e, N_j, d_grid, a2_grid, z_gridvals_J, e_gridvals_J, aprimeFnParams, fastOLG)
% Age-dependent (_J) version of CreateaprimePolicyExperienceAssetze:
% compute a2prime=aprimeFn(d, a2, z, e) using the Policy-chosen d for each
% state, used in simulation / agent-distribution. z enters aprimeFn directly,
% and e (drawn at the START of the period) enters Policy as well.
%
% The input Policy will contain aprime (except for the experience asset)
% and the decision variables (d2, and where applicable d1). The output is
% just the Policy for a2prime (the experience asset). As well as the
% related probabilities.
%
% Companion file CreateExperienceAssetzeFnMatrix_J.m does the same for ALL d
% (not just the Policy-chosen one), used during value-function iteration.
%
% Two layout modes controlled by the fastOLG flag (j ordered before z,e):
%   fastOLG==0 :  state order is (a, z, e, j). Policy has shape
%                 [L, N_a, N_z, N_e, N_j]; index Policy(k,:,:) (Matlab
%                 linearises trailing dims); reshape to [N_a*N_z*N_e, N_j].
%   fastOLG==1 :  state order is (a, j, z, e). Policy has shape
%                 [L, N_a, N_j, N_z, N_e]; index Policy(k,:,:,:,:);
%                 reshape to [N_a, N_j, N_z, N_e].
%
% aprimeFnParams is passed as a [N_j, n_params] matrix and each column is
% shifted via shiftdim(...,-1) so j lives in dim 2 during arrayfun.
%
% Output sizes:
%   fastOLG==0 : [N_a, N_z, N_e, N_j]
%   fastOLG==1 : [N_a, N_j, N_z, N_e]

ParamCell=cell(size(aprimeFnParams,2),1);
for ii=1:size(aprimeFnParams,2)
    ParamCell(ii,1)={shiftdim(aprimeFnParams(:,ii),-1)}; % j is second dimension
end

N_a1=prod(n_a1);
if N_a1==0
    N_a=prod(n_a2);
else
    N_a=prod([n_a1,n_a2]);
end
N_z=prod(n_z);
N_e=prod(n_e);
N_aze=N_a*N_z*N_e;

l_dexp=length(whichisdforexpasset);
l_z=length(n_z);
l_e=length(n_e);

if nargin(aprimeFn)~=l_dexp+1+l_z+l_e+size(aprimeFnParams,2)
    error('Number of inputs to aprimeFn does not fit with size of aprimeFnParams')
end


if fastOLG==0 % state order (a, z, e, j)

    if l_dexp>=1
        if whichisdforexpasset(1)==1
            d1grid=d_grid(1:n_d(1));
        else
            d1grid=d_grid(sum(n_d(1:whichisdforexpasset(1)-1))+1:sum(n_d(1:whichisdforexpasset(1))));
        end
        d1vals=reshape(d1grid(Policy(whichisdforexpasset(1),:,:)),[N_aze,N_j]);
        if l_dexp>=2
            d2grid=d_grid(sum(n_d(1:whichisdforexpasset(2)-1))+1:sum(n_d(1:whichisdforexpasset(2))));
            d2vals=reshape(d2grid(Policy(whichisdforexpasset(2),:,:)),[N_aze,N_j]);
            if l_dexp>=3
                d3grid=d_grid(sum(n_d(1:whichisdforexpasset(3)-1))+1:sum(n_d(1:whichisdforexpasset(3))));
                d3vals=reshape(d3grid(Policy(whichisdforexpasset(3),:,:)),[N_aze,N_j]);
                if l_dexp>=4
                    d4grid=d_grid(sum(n_d(1:whichisdforexpasset(4)-1))+1:sum(n_d(1:whichisdforexpasset(4))));
                    d4vals=reshape(d4grid(Policy(whichisdforexpasset(4),:,:)),[N_aze,N_j]);
                end
            end
        end
    end

    % Layout: a fastest, then z, then e, then j
    if N_a1==0
        a2vals=kron(ones(N_e*N_z,1),a2_grid);
    else
        a2vals=kron(ones(N_e*N_z,1),kron(a2_grid,ones(N_a1,1)));
    end

    % permute once: [N_z, l_z, N_j] -> [N_z, N_j, l_z]; then index final dim per z column
    z_gridvals_Jp=permute(z_gridvals_J,[1,3,2]);
    if l_z>=1
        z1vals=kron(ones(N_e,1),kron(z_gridvals_Jp(:,:,1),ones(N_a,1)));   % [N_a*N_z*N_e, N_j]
        if l_z>=2
            z2vals=kron(ones(N_e,1),kron(z_gridvals_Jp(:,:,2),ones(N_a,1)));
            if l_z>=3
                z3vals=kron(ones(N_e,1),kron(z_gridvals_Jp(:,:,3),ones(N_a,1)));
                if l_z>=4
                    z4vals=kron(ones(N_e,1),kron(z_gridvals_Jp(:,:,4),ones(N_a,1)));
                    if l_z>=5
                        error('Max of four z variables supported (contact if you need more)')
                    end
                end
            end
        end
    end

    % permute once: [N_e, l_e, N_j] -> [N_e, N_j, l_e]; then index final dim per e column
    e_gridvals_Jp=permute(e_gridvals_J,[1,3,2]);
    if l_e>=1
        e1vals=kron(e_gridvals_Jp(:,:,1),ones(N_a*N_z,1));   % [N_a*N_z*N_e, N_j]
        if l_e>=2
            e2vals=kron(e_gridvals_Jp(:,:,2),ones(N_a*N_z,1));
            if l_e>=3
                e3vals=kron(e_gridvals_Jp(:,:,3),ones(N_a*N_z,1));
                if l_e>=4
                    e4vals=kron(e_gridvals_Jp(:,:,4),ones(N_a*N_z,1));
                    if l_e>=5
                        error('Max of four e variables supported (contact if you need more)')
                    end
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

    % Removed: a2vals=a2vals.*ones(1,1,1,'gpuArray'); % was here to fool matlab which otherwise threw an error; restore this line if functionality breaks
    if l_dexp==1
        a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, zecell{:}, ParamCell{:});
    elseif l_dexp==2
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, zecell{:}, ParamCell{:});
    elseif l_dexp==3
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, zecell{:}, ParamCell{:});
    elseif l_dexp==4
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, zecell{:}, ParamCell{:});
    end

    a2primeVals=reshape(a2primeVals,[1,N_aze*N_j]);

    a2_griddiff=a2_grid(2:end)-a2_grid(1:end-1);

    a2primeIndexes=discretize(a2primeVals,a2_grid);
    offBottomOfGrid=(a2primeVals<=a2_grid(1));
    a2primeIndexes(offBottomOfGrid)=1;
    offTopOfGrid=(a2primeVals>=a2_grid(end));
    a2primeIndexes(offTopOfGrid)=n_a2-1;

    aprime_residual=a2primeVals'-a2_grid(a2primeIndexes);
    a2primeProbs=1-aprime_residual./a2_griddiff(a2primeIndexes);
    a2primeProbs(offBottomOfGrid)=1;
    a2primeProbs(offTopOfGrid)=0;

    a2primeIndexes=reshape(a2primeIndexes,[N_a,N_z,N_e,N_j]);
    a2primeProbs=reshape(a2primeProbs,[N_a,N_z,N_e,N_j]);


elseif fastOLG==1 % state order (a, j, z, e)

    if l_dexp>=1
        if whichisdforexpasset(1)==1
            d1grid=d_grid(1:n_d(1));
        else
            d1grid=d_grid(sum(n_d(1:whichisdforexpasset(1)-1))+1:sum(n_d(1:whichisdforexpasset(1))));
        end
        d1vals=reshape(d1grid(Policy(whichisdforexpasset(1),:,:,:,:)),[N_a,N_j,N_z,N_e]);
        if l_dexp>=2
            d2grid=d_grid(sum(n_d(1:whichisdforexpasset(2)-1))+1:sum(n_d(1:whichisdforexpasset(2))));
            d2vals=reshape(d2grid(Policy(whichisdforexpasset(2),:,:,:,:)),[N_a,N_j,N_z,N_e]);
            if l_dexp>=3
                d3grid=d_grid(sum(n_d(1:whichisdforexpasset(3)-1))+1:sum(n_d(1:whichisdforexpasset(3))));
                d3vals=reshape(d3grid(Policy(whichisdforexpasset(3),:,:,:,:)),[N_a,N_j,N_z,N_e]);
                if l_dexp>=4
                    d4grid=d_grid(sum(n_d(1:whichisdforexpasset(4)-1))+1:sum(n_d(1:whichisdforexpasset(4))));
                    d4vals=reshape(d4grid(Policy(whichisdforexpasset(4),:,:,:,:)),[N_a,N_j,N_z,N_e]);
                end
            end
        end
    end

    if N_a1==0
        a2vals=a2_grid;
    else
        a2vals=repelem(a2_grid,N_a1,1);
    end

    % permute once: [N_z, l_z, N_j] -> [1, N_j, N_z, l_z]; then index final dim per z column
    z_gridvals_Jp=permute(z_gridvals_J,[4,3,1,2]);
    if l_z>=1
        z1vals=z_gridvals_Jp(:,:,:,1);   % [1, N_j, N_z]
        if l_z>=2
            z2vals=z_gridvals_Jp(:,:,:,2);
            if l_z>=3
                z3vals=z_gridvals_Jp(:,:,:,3);
                if l_z>=4
                    z4vals=z_gridvals_Jp(:,:,:,4);
                    if l_z>=5
                        error('Max of four z variables supported (contact if you need more)')
                    end
                end
            end
        end
    end

    % permute once: [N_e, l_e, N_j] -> [1, N_j, 1, N_e, l_e]; then index final dim per e column
    e_gridvals_Jp=permute(e_gridvals_J,[4,3,5,1,2]);
    if l_e>=1
        e1vals=e_gridvals_Jp(:,:,:,:,1);   % [1, N_j, 1, N_e]
        if l_e>=2
            e2vals=e_gridvals_Jp(:,:,:,:,2);
            if l_e>=3
                e3vals=e_gridvals_Jp(:,:,:,:,3);
                if l_e>=4
                    e4vals=e_gridvals_Jp(:,:,:,:,4);
                    if l_e>=5
                        error('Max of four e variables supported (contact if you need more)')
                    end
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

    % Removed: a2vals=a2vals.*ones(1,1,1,'gpuArray'); % was here to fool matlab which otherwise threw an error; restore this line if functionality breaks
    if l_dexp==1
        a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, zecell{:}, ParamCell{:});
    elseif l_dexp==2
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, zecell{:}, ParamCell{:});
    elseif l_dexp==3
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, zecell{:}, ParamCell{:});
    elseif l_dexp==4
        a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, zecell{:}, ParamCell{:});
    end

    a2primeVals=reshape(a2primeVals,[1,N_a*N_j*N_z*N_e]);

    a2_griddiff=a2_grid(2:end)-a2_grid(1:end-1);

    a2primeIndexes=discretize(a2primeVals,a2_grid);
    offBottomOfGrid=(a2primeVals<=a2_grid(1));
    a2primeIndexes(offBottomOfGrid)=1;
    offTopOfGrid=(a2primeVals>=a2_grid(end));
    a2primeIndexes(offTopOfGrid)=n_a2-1;

    aprime_residual=a2primeVals'-a2_grid(a2primeIndexes);
    a2primeProbs=1-aprime_residual./a2_griddiff(a2primeIndexes);
    a2primeProbs(offBottomOfGrid)=1;
    a2primeProbs(offTopOfGrid)=0;

    a2primeIndexes=reshape(a2primeIndexes,[N_a,N_j,N_z,N_e]);
    a2primeProbs=reshape(a2primeProbs,[N_a,N_j,N_z,N_e]);


end

end
