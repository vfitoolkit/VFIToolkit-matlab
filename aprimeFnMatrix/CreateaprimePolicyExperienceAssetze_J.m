function [a2primeIndexes, a2primeProbs]=CreateaprimePolicyExperienceAssetze_J(Policy,aprimeFn, whichisdforexpasset, n_d, n_a1,n_a2, n_z, n_e,N_semiz,N_z,N_e, N_j, d_grid, a2_grid, z_gridvals_J, e_gridvals_J, aprimeFnParams, fastOLG)
% Age-dependent (_J) version of CreateaprimePolicyExperienceAssetze:
% compute a2prime=aprimeFn(d, a2, z, e) using the Policy-chosen d for each
% state, used in simulation / agent-distribution. z enters aprimeFn
% directly, and e (drawn at the START of the period) enters Policy as well.
%
% The input Policy will contain aprime (except for the experience asset)
% and the decision variables (d2, and where applicable d1). The output is
% just the Policy for a2prime (the experience asset). As well as the
% related probabilities.
%
% Companion file CreateExperienceAssetzeFnMatrix_J.m does the same for ALL
% d (not just the Policy-chosen one), used during value-function iteration.
%
% Two layout modes controlled by the fastOLG flag:
%   fastOLG==0 :  state order is (a, semiz, z, e, j).
%   fastOLG==1 :  state order is (a, j, semiz, z, e).
%
% aprimeFnParams is passed as a [N_j, n_params] matrix and each column is
% shifted so j lives at the appropriate dimension for the chosen fastOLG.
%
% Output sizes:
%   l_a2==1 (legacy):
%     fastOLG==0 : a2primeIndexes/Probs [N_a, N_semizze, N_j]
%     fastOLG==1 : a2primeIndexes/Probs [N_a, N_j, N_semizze]
%   l_a2==2 (multi-dim, per-dim factored; fastOLG==0 only at present):
%     fastOLG==0 : a2primeIndexes/Probs [N_a, l_a2, N_semizze, N_j]
%     a2primeIndexes(:,k,:,:) = lower-grid index in a2_k dim
%     a2primeProbs(:,k,:,:)   = probability of lower grid point in a2_k dim
%     Caller does nested 2-corner interp (Kron-fold to 4 corners).
%
% Note: N_semizze = N_semiz * N_z * N_e (with size-1 substituted for any
% absent dim) is just the 'size' of Policy beyond a and j.

N_a1=prod(n_a1);
if N_a1==0
    N_a=prod(n_a2);
else
    N_a=prod([n_a1,n_a2]);
end

l_dexp=length(whichisdforexpasset);
l_a2=length(n_a2);
l_z=length(n_z);
l_e=length(n_e);

if l_a2>2
    error('experienceassetze currently supports length(n_a2) in {1,2}')
end
if l_a2>1 && fastOLG==1
    error('experienceassetze l_a2==2 is not yet supported in fastOLG==1 mode')
end

if nargin(aprimeFn)~=l_dexp+l_a2+l_z+l_e+(l_a2>=2)+size(aprimeFnParams,2)
    % When l_a2>=2, aprimeFn takes an extra 'whicha' integer selector slot
    % between the z,e inputs and the parameter inputs.
    error('Number of inputs to aprimeFn does not fit with size of aprimeFnParams')
end

if l_z>=5
    error('Max of four z variables supported in CreateaprimePolicyExperienceAssetze_J (contact if you need more)')
end
if l_e>=5
    error('Max of four e variables supported in CreateaprimePolicyExperienceAssetze_J (contact if you need more)')
end


if fastOLG==0 % state order (a, semiz, z, e, j)

    % j is at dim 5
    ParamCell=cell(size(aprimeFnParams,2),1);
    for ii=1:size(aprimeFnParams,2)
        ParamCell(ii,1)={shiftdim(aprimeFnParams(:,ii),-4)};
    end

    if l_dexp>=1
        if whichisdforexpasset(1)==1
            d1grid=d_grid(1:n_d(1));
        else
            d1grid=d_grid(sum(n_d(1:whichisdforexpasset(1)-1))+1:sum(n_d(1:whichisdforexpasset(1))));
        end

        if N_semiz==0
            N_semizze=N_z*N_e;
            d1vals=reshape(d1grid(Policy(whichisdforexpasset(1),:,:)),[N_a,1,N_z,N_e,N_j]);
            if l_dexp>=2
                d2grid=d_grid(sum(n_d(1:whichisdforexpasset(2)-1))+1:sum(n_d(1:whichisdforexpasset(2))));
                d2vals=reshape(d2grid(Policy(whichisdforexpasset(2),:,:)),[N_a,1,N_z,N_e,N_j]);
                if l_dexp>=3
                    d3grid=d_grid(sum(n_d(1:whichisdforexpasset(3)-1))+1:sum(n_d(1:whichisdforexpasset(3))));
                    d3vals=reshape(d3grid(Policy(whichisdforexpasset(3),:,:)),[N_a,1,N_z,N_e,N_j]);
                    if l_dexp>=4
                        d4grid=d_grid(sum(n_d(1:whichisdforexpasset(4)-1))+1:sum(n_d(1:whichisdforexpasset(4))));
                        d4vals=reshape(d4grid(Policy(whichisdforexpasset(4),:,:)),[N_a,1,N_z,N_e,N_j]);
                    end
                end
            end
        elseif N_semiz>0
            N_semizze=N_semiz*N_z*N_e;
            d1vals=reshape(d1grid(Policy(whichisdforexpasset(1),:,:)),[N_a,N_semiz,N_z,N_e,N_j]);
            if l_dexp>=2
                d2grid=d_grid(sum(n_d(1:whichisdforexpasset(2)-1))+1:sum(n_d(1:whichisdforexpasset(2))));
                d2vals=reshape(d2grid(Policy(whichisdforexpasset(2),:,:)),[N_a,N_semiz,N_z,N_e,N_j]);
                if l_dexp>=3
                    d3grid=d_grid(sum(n_d(1:whichisdforexpasset(3)-1))+1:sum(n_d(1:whichisdforexpasset(3))));
                    d3vals=reshape(d3grid(Policy(whichisdforexpasset(3),:,:)),[N_a,N_semiz,N_z,N_e,N_j]);
                    if l_dexp>=4
                        d4grid=d_grid(sum(n_d(1:whichisdforexpasset(4)-1))+1:sum(n_d(1:whichisdforexpasset(4))));
                        d4vals=reshape(d4grid(Policy(whichisdforexpasset(4),:,:)),[N_a,N_semiz,N_z,N_e,N_j]);
                    end
                end
            end
        end
    end

    if l_a2==1
        if N_a1==0
            a2vals=a2_grid;
        else
            a2vals=repelem(a2_grid,N_a1,1);
        end
    elseif l_a2==2
        n_a2_1=n_a2(1); n_a2_2=n_a2(2);
        a2_grid_1=a2_grid(1:n_a2_1);
        a2_grid_2=a2_grid(n_a2_1+1:n_a2_1+n_a2_2);
        a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
        if N_a1==0
            a2vals_1=a2_gridvals(:,1);
            a2vals_2=a2_gridvals(:,2);
        else
            a2vals_1=repelem(a2_gridvals(:,1),N_a1,1);
            a2vals_2=repelem(a2_gridvals(:,2),N_a1,1);
        end
    end

    % permute once: [N_z, l_z, N_j] -> [1, 1, N_z, 1, N_j, l_z]; index final dim per z column
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

    % permute once: [N_e, l_e, N_j] -> [1, 1, 1, N_e, N_j, l_e]; index final dim per e column
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

    %% expassetze_J: aprime(d, a2, z, e) [plus whicha when l_a2>=2]
    if l_a2==1
    if l_dexp==1
        if l_z==1
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        elseif l_z==2
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        elseif l_z==3
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        elseif l_z==4
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        end
    elseif l_dexp==2
        if l_z==1
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        elseif l_z==2
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals,z2vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        elseif l_z==3
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        elseif l_z==4
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        end
    elseif l_dexp==3
        if l_z==1
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        elseif l_z==2
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals,z2vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        elseif l_z==3
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        elseif l_z==4
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        end
    elseif l_dexp==4
        if l_z==1
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        elseif l_z==2
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals,z2vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        elseif l_z==3
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        elseif l_z==4
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        end
    end

    a2primeVals=reshape(a2primeVals,[1,N_a*N_semizze*N_j]);

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

    a2primeIndexes=reshape(a2primeIndexes,[N_a,N_semizze,N_j]);
    a2primeProbs=reshape(a2primeProbs,[N_a,N_semizze,N_j]);

    elseif l_a2==2
        %% Multi-dim a2 (l_a2=2) in fastOLG==0: bilinear, per-dim factored
        % Build arrayfun arg list dynamically (avoids 4*4*4*2 enumeration).
        args=cell(0,1);
        args{end+1}=d1vals;
        if l_dexp>=2, args{end+1}=d2vals; end
        if l_dexp>=3, args{end+1}=d3vals; end
        if l_dexp>=4, args{end+1}=d4vals; end
        args{end+1}=a2vals_1;
        args{end+1}=a2vals_2;
        if l_z>=1, args{end+1}=z1vals; end
        if l_z>=2, args{end+1}=z2vals; end
        if l_z>=3, args{end+1}=z3vals; end
        if l_z>=4, args{end+1}=z4vals; end
        if l_e>=1, args{end+1}=e1vals; end
        if l_e>=2, args{end+1}=e2vals; end
        if l_e>=3, args{end+1}=e3vals; end
        if l_e>=4, args{end+1}=e4vals; end

        args1=[args; {1}; ParamCell];
        args2=[args; {2}; ParamCell];
        a2pVals_1=arrayfun(aprimeFn, args1{:});
        a2pVals_2=arrayfun(aprimeFn, args2{:});

        [loIdx_1, prob_1]=local_interp1d(a2pVals_1, a2_grid_1, n_a2_1);
        [loIdx_2, prob_2]=local_interp1d(a2pVals_2, a2_grid_2, n_a2_2);

        a2primeIndexes=zeros(N_a,l_a2,N_semizze,N_j,'gpuArray');
        a2primeProbs=zeros(N_a,l_a2,N_semizze,N_j,'gpuArray');
        a2primeIndexes(:,1,:,:)=reshape(loIdx_1,[N_a,1,N_semizze,N_j]);
        a2primeIndexes(:,2,:,:)=reshape(loIdx_2,[N_a,1,N_semizze,N_j]);
        a2primeProbs(:,1,:,:)=reshape(prob_1,[N_a,1,N_semizze,N_j]);
        a2primeProbs(:,2,:,:)=reshape(prob_2,[N_a,1,N_semizze,N_j]);
    end


elseif fastOLG==1 % state order (a, j, semiz, z, e)

    % j is at dim 2
    ParamCell=cell(size(aprimeFnParams,2),1);
    for ii=1:size(aprimeFnParams,2)
        ParamCell(ii,1)={shiftdim(aprimeFnParams(:,ii),-1)};
    end

    if l_dexp>=1
        if whichisdforexpasset(1)==1
            d1grid=d_grid(1:n_d(1));
        else
            d1grid=d_grid(sum(n_d(1:whichisdforexpasset(1)-1))+1:sum(n_d(1:whichisdforexpasset(1))));
        end

        if N_semiz==0
            N_semizze=N_z*N_e;
            d1vals=reshape(d1grid(Policy(whichisdforexpasset(1),:,:,:,:)),[N_a,N_j,1,N_z,N_e]);
            if l_dexp>=2
                d2grid=d_grid(sum(n_d(1:whichisdforexpasset(2)-1))+1:sum(n_d(1:whichisdforexpasset(2))));
                d2vals=reshape(d2grid(Policy(whichisdforexpasset(2),:,:,:,:)),[N_a,N_j,1,N_z,N_e]);
                if l_dexp>=3
                    d3grid=d_grid(sum(n_d(1:whichisdforexpasset(3)-1))+1:sum(n_d(1:whichisdforexpasset(3))));
                    d3vals=reshape(d3grid(Policy(whichisdforexpasset(3),:,:,:,:)),[N_a,N_j,1,N_z,N_e]);
                    if l_dexp>=4
                        d4grid=d_grid(sum(n_d(1:whichisdforexpasset(4)-1))+1:sum(n_d(1:whichisdforexpasset(4))));
                        d4vals=reshape(d4grid(Policy(whichisdforexpasset(4),:,:,:,:)),[N_a,N_j,1,N_z,N_e]);
                    end
                end
            end
        elseif N_semiz>0
            N_semizze=N_semiz*N_z*N_e;
            d1vals=reshape(d1grid(Policy(whichisdforexpasset(1),:,:,:,:)),[N_a,N_j,N_semiz,N_z,N_e]);
            if l_dexp>=2
                d2grid=d_grid(sum(n_d(1:whichisdforexpasset(2)-1))+1:sum(n_d(1:whichisdforexpasset(2))));
                d2vals=reshape(d2grid(Policy(whichisdforexpasset(2),:,:,:,:)),[N_a,N_j,N_semiz,N_z,N_e]);
                if l_dexp>=3
                    d3grid=d_grid(sum(n_d(1:whichisdforexpasset(3)-1))+1:sum(n_d(1:whichisdforexpasset(3))));
                    d3vals=reshape(d3grid(Policy(whichisdforexpasset(3),:,:,:,:)),[N_a,N_j,N_semiz,N_z,N_e]);
                    if l_dexp>=4
                        d4grid=d_grid(sum(n_d(1:whichisdforexpasset(4)-1))+1:sum(n_d(1:whichisdforexpasset(4))));
                        d4vals=reshape(d4grid(Policy(whichisdforexpasset(4),:,:,:,:)),[N_a,N_j,N_semiz,N_z,N_e]);
                    end
                end
            end
        end
    end

    if N_a1==0
        a2vals=a2_grid;
    else
        a2vals=repelem(a2_grid,N_a1,1);
    end

    % permute once: [N_z, l_z, N_j] -> [1, N_j, 1, N_z, 1, l_z]; index final dim per z column
    z_gridvals_Jp=permute(z_gridvals_J,[4,3,5,1,6,2]);
    if l_z>=1
        z1vals=z_gridvals_Jp(:,:,:,:,:,1);   % [1, N_j, 1, N_z, 1]
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

    % permute once: [N_e, l_e, N_j] -> [1, N_j, 1, 1, N_e, l_e]; index final dim per e column
    e_gridvals_Jp=permute(e_gridvals_J,[4,3,5,6,1,2]);
    if l_e>=1
        e1vals=e_gridvals_Jp(:,:,:,:,:,1);   % [1, N_j, 1, 1, N_e]
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

    %% expassetze_J: aprime(d, a2, z, e)
    if l_dexp==1
        if l_z==1
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        elseif l_z==2
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        elseif l_z==3
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        elseif l_z==4
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        end
    elseif l_dexp==2
        if l_z==1
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        elseif l_z==2
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals,z2vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        elseif l_z==3
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        elseif l_z==4
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        end
    elseif l_dexp==3
        if l_z==1
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        elseif l_z==2
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals,z2vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        elseif l_z==3
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        elseif l_z==4
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        end
    elseif l_dexp==4
        if l_z==1
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        elseif l_z==2
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals,z2vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        elseif l_z==3
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        elseif l_z==4
            if l_e==1
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
            elseif l_e==2
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
            elseif l_e==3
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
            elseif l_e==4
                a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
            end
        end
    end

    a2primeVals=reshape(a2primeVals,[1,N_a*N_j*N_semizze]);

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

    a2primeIndexes=reshape(a2primeIndexes,[N_a,N_j,N_semizze]);
    a2primeProbs=reshape(a2primeProbs,[N_a,N_j,N_semizze]);


end

end


function [loIdx, prob]=local_interp1d(aprimeVals, grid, n_grid)
% 1D linear-interp: lower-grid index in 1..n_grid and prob of lower point.
apvals=aprimeVals(:);
N=numel(apvals);
griddiff=grid(2:end)-grid(1:end-1);

if N*n_grid<1000000
    [~,upIdx]=max((grid>apvals'),[],1);
    loIdx=upIdx-1;
    loIdx(loIdx==0)=1;
    loIdx=loIdx(:);
    residual=apvals-grid(loIdx);
    prob=1-residual./griddiff(loIdx);
    offTop=(apvals>=grid(end));
    loIdx(offTop)=n_grid-1;
    prob(offTop)=0;
    offBottom=(apvals<=grid(1));
    prob(offBottom)=1;
else
    loIdx=discretize(apvals,grid);
    loIdx=loIdx(:);
    offBottom=(apvals<=grid(1));
    loIdx(offBottom)=1;
    offTop=(apvals>=grid(end));
    loIdx(offTop)=n_grid-1;
    residual=apvals-grid(loIdx);
    prob=1-residual./griddiff(loIdx);
    prob(offBottom)=1;
    prob(offTop)=0;
end
end
