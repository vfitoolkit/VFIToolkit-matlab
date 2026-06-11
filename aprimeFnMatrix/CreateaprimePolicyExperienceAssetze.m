function [a2primeIndexes, a2primeProbs]=CreateaprimePolicyExperienceAssetze(Policy,aprimeFn, whichisdforexpasset, n_d, n_a1,n_a2, n_z, n_e,N_semiz,N_z,N_e, d_grid, a2_grid, z_gridvals, e_gridvals, aprimeFnParams)
% For experienceassetze: compute a2prime=aprimeFn(d, a2, z, e) using the
% Policy-chosen d for each state (one d per state), used in simulation /
% agent-distribution. Note: e is i.i.d. drawn at the START of the period,
% so Policy DOES depend on e (Policy has shape [..., N_a, N_z, N_e]).
% Because the true value of a2prime will (almost always) lie between two
% consecutive points in a2_grid, it is linearly interpolated back on to
% a2_grid. Thus the continuous a2prime is represented by (index of lower
% grid point in a2primeIndexes, probability of lower grid point in
% a2primeProbs) on a2_grid; the upper index is implicitly lower+1 with
% prob 1-minus-prob-of-lower.
%
% The input Policy will contain aprime (except for the experience asset)
% and the decision variables (d2, and where applicable d1). The output is
% just the Policy for a2prime (the experience asset). As well as the
% related probabilities.
%
% Companion file CreateExperienceAssetzeFnMatrix.m does the same but for
% ALL d (not just the Policy-chosen one), used during value-function
% iteration to find Policy. This file is used afterwards, once Policy has
% been chosen, for simulation / agent-distribution.
%
% Output sizes:
%   l_a2==1 (legacy):
%     a2primeIndexes - [N_a, N_semizze]
%     a2primeProbs   - [N_a, N_semizze]
%   l_a2==2 (multi-dim, per-dim factored):
%     a2primeIndexes - [N_a, l_a2, N_semizze]
%     a2primeProbs   - [N_a, l_a2, N_semizze]
%     a2primeIndexes(:,k,:) = lower-grid index in a2_k dim
%     a2primeProbs(:,k,:)   = probability of lower grid point in a2_k dim
%     Caller does nested 2-corner interp (Kron-fold to 4 corners).
%
% Note: N_semizze is just the 'size' of Policy

ParamCell=cell(length(aprimeFnParams),1);
for ii=1:length(aprimeFnParams)
    if ~isscalar(aprimeFnParams(ii))
        error('Using experienceassetze does not allow for any of aprimeFn parameters to be anything but a scalar')
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
l_a2=length(n_a2);
l_z=length(n_z);
l_e=length(n_e);

if l_a2>2
    error('experienceassetze currently supports length(n_a2) in {1,2}')
end

if nargin(aprimeFn)~=l_dexp+l_a2+l_z+l_e+(l_a2>=2)+length(aprimeFnParams)
    % When l_a2>=2, aprimeFn takes an extra 'whicha' integer selector slot
    % between the z,e inputs and the parameter inputs.
    error('Number of inputs to aprimeFn does not fit with size of aprimeFnParams')
end

if l_z>=5
    error('Max of four z variables supported in CreateaprimePolicyExperienceAssetze (contact if you need more)')
end
if l_e>=5
    error('Max of four e variables supported in CreateaprimePolicyExperienceAssetze (contact if you need more)')
end


if l_dexp>=1 % WHY I AM DOING THIS, PRETTY SURE YOU CANNOT NOT SATISFY THIS???
    if whichisdforexpasset(1)==1
        d1grid=d_grid(1:n_d(1));
    else
        d1grid=d_grid(sum(n_d(1:whichisdforexpasset(1)-1))+1:sum(n_d(1:whichisdforexpasset(1))));
    end

    if N_semiz==0
        N_semizze=N_z*N_e;
        d1vals=reshape(d1grid(Policy(whichisdforexpasset(1),:,:)),[N_a,1,N_z,N_e]);
        if l_dexp>=2
            d2grid=d_grid(sum(n_d(1:whichisdforexpasset(2)-1))+1:sum(n_d(1:whichisdforexpasset(2))));
            d2vals=reshape(d2grid(Policy(whichisdforexpasset(2),:,:)),[N_a,1,N_z,N_e]);
            if l_dexp>=3
                d3grid=d_grid(sum(n_d(1:whichisdforexpasset(3)-1))+1:sum(n_d(1:whichisdforexpasset(3))));
                d3vals=reshape(d3grid(Policy(whichisdforexpasset(3),:,:)),[N_a,1,N_z,N_e]);
                if l_dexp>=4
                    d4grid=d_grid(sum(n_d(1:whichisdforexpasset(4)-1))+1:sum(n_d(1:whichisdforexpasset(4))));
                    d4vals=reshape(d4grid(Policy(whichisdforexpasset(4),:,:)),[N_a,1,N_z,N_e]);
                end
            end
        end
    elseif N_semiz>0
        N_semizze=N_semiz*N_z*N_e;
        d1vals=reshape(d1grid(Policy(whichisdforexpasset(1),:,:)),[N_a,N_semiz,N_z,N_e]);
        if l_dexp>=2
            d2grid=d_grid(sum(n_d(1:whichisdforexpasset(2)-1))+1:sum(n_d(1:whichisdforexpasset(2))));
            d2vals=reshape(d2grid(Policy(whichisdforexpasset(2),:,:)),[N_a,N_semiz,N_z,N_e]);
            if l_dexp>=3
                d3grid=d_grid(sum(n_d(1:whichisdforexpasset(3)-1))+1:sum(n_d(1:whichisdforexpasset(3))));
                d3vals=reshape(d3grid(Policy(whichisdforexpasset(3),:,:)),[N_a,N_semiz,N_z,N_e]);
                if l_dexp>=4
                    d4grid=d_grid(sum(n_d(1:whichisdforexpasset(4)-1))+1:sum(n_d(1:whichisdforexpasset(4))));
                    d4vals=reshape(d4grid(Policy(whichisdforexpasset(4),:,:)),[N_a,N_semiz,N_z,N_e]);
                end
            end
        end
    end
end


% Layout: a varies fastest, then z, then e (matching Policy's [N_a,N_z,N_e] reshape)
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
    a2_gridvals=CreateGridvals(n_a2,a2_grid,1); % [N_a2, 2]
    if N_a1==0
        a2vals_1=a2_gridvals(:,1);
        a2vals_2=a2_gridvals(:,2);
    else
        a2vals_1=repelem(a2_gridvals(:,1),N_a1,1);
        a2vals_2=repelem(a2_gridvals(:,2),N_a1,1);
    end
end


%% expassetze: aprime(d,a2,z,e) [plus whicha selector when l_a2>=2]
if l_a2==1
if l_dexp==1
    if l_z==1
        if l_e==1
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), ParamCell{:});
        elseif l_e==2
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), ParamCell{:});
        elseif l_e==3
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), ParamCell{:});
        elseif l_e==4
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), ParamCell{:});
        end
    elseif l_z==2
        if l_e==1
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), ParamCell{:});
        elseif l_e==2
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), ParamCell{:});
        elseif l_e==3
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), ParamCell{:});
        elseif l_e==4
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), ParamCell{:});
        end
    elseif l_z==3
        if l_e==1
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), ParamCell{:});
        elseif l_e==2
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), ParamCell{:});
        elseif l_e==3
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), ParamCell{:});
        elseif l_e==4
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), ParamCell{:});
        end
    elseif l_z==4
        if l_e==1
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), ParamCell{:});
        elseif l_e==2
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), ParamCell{:});
        elseif l_e==3
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), ParamCell{:});
        elseif l_e==4
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), ParamCell{:});
        end
    end
elseif l_dexp==2
    if l_z==1
        if l_e==1
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), ParamCell{:});
        elseif l_e==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), ParamCell{:});
        elseif l_e==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), ParamCell{:});
        elseif l_e==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), ParamCell{:});
        end
    elseif l_z==2
        if l_e==1
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), ParamCell{:});
        elseif l_e==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), ParamCell{:});
        elseif l_e==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), ParamCell{:});
        elseif l_e==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), ParamCell{:});
        end
    elseif l_z==3
        if l_e==1
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), ParamCell{:});
        elseif l_e==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), ParamCell{:});
        elseif l_e==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), ParamCell{:});
        elseif l_e==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), ParamCell{:});
        end
    elseif l_z==4
        if l_e==1
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), ParamCell{:});
        elseif l_e==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), ParamCell{:});
        elseif l_e==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), ParamCell{:});
        elseif l_e==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), ParamCell{:});
        end
    end
elseif l_dexp==3
    if l_z==1
        if l_e==1
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), ParamCell{:});
        elseif l_e==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), ParamCell{:});
        elseif l_e==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), ParamCell{:});
        elseif l_e==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), ParamCell{:});
        end
    elseif l_z==2
        if l_e==1
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), ParamCell{:});
        elseif l_e==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), ParamCell{:});
        elseif l_e==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), ParamCell{:});
        elseif l_e==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), ParamCell{:});
        end
    elseif l_z==3
        if l_e==1
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), ParamCell{:});
        elseif l_e==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), ParamCell{:});
        elseif l_e==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), ParamCell{:});
        elseif l_e==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), ParamCell{:});
        end
    elseif l_z==4
        if l_e==1
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), ParamCell{:});
        elseif l_e==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), ParamCell{:});
        elseif l_e==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), ParamCell{:});
        elseif l_e==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), ParamCell{:});
        end
    end
elseif l_dexp==4
    if l_z==1
        if l_e==1
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), ParamCell{:});
        elseif l_e==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), ParamCell{:});
        elseif l_e==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), ParamCell{:});
        elseif l_e==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), ParamCell{:});
        end
    elseif l_z==2
        if l_e==1
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), ParamCell{:});
        elseif l_e==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), ParamCell{:});
        elseif l_e==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), ParamCell{:});
        elseif l_e==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), ParamCell{:});
        end
    elseif l_z==3
        if l_e==1
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), ParamCell{:});
        elseif l_e==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), ParamCell{:});
        elseif l_e==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), ParamCell{:});
        elseif l_e==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), ParamCell{:});
        end
    elseif l_z==4
        if l_e==1
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), ParamCell{:});
        elseif l_e==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), ParamCell{:});
        elseif l_e==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), ParamCell{:});
        elseif l_e==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), ParamCell{:});
        end
    end
end


    %% Calcuate grid indexes and probs from the values (l_a2==1)
    a2primeVals=reshape(a2primeVals,[1,N_a*N_semizze]);

    a2_griddiff=a2_grid(2:end)-a2_grid(1:end-1); % Distance between point and the next point

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

    a2primeIndexes=reshape(a2primeIndexes,[N_a,N_semizze]); % Index of lower grid point
    a2primeProbs=reshape(a2primeProbs,[N_a,N_semizze]); % Probability of lower grid point

elseif l_a2==2
    %% Multi-dim a2 (l_a2=2): bilinear interp, per-dim factored output
    % Build arrayfun arg list dynamically (avoids 4*4*4*2 = 128-case enumeration).
    % All shape contracts match the l_a2==1 path:
    %   d*vals    : [N_a, 1 or N_semiz, N_z, N_e] (Policy-chosen per state)
    %   a2vals_k  : [N_a, 1]            (state-indexed; no new dim)
    %   z col k   : shiftdim(...,-2)    (dim 3)
    %   e col k   : shiftdim(...,-3)    (dim 4)
    %% Full nested if-else enumeration over (l_dexp, l_z, l_e) x (whicha=1,2). Matches the l_a2==1 cascade style above.
    if l_dexp==1
        if l_z==1
            if l_e==1
                a2pVals_1=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), 2, ParamCell{:});
            elseif l_e==2
                a2pVals_1=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 2, ParamCell{:});
            elseif l_e==3
                a2pVals_1=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 2, ParamCell{:});
            elseif l_e==4
                a2pVals_1=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 2, ParamCell{:});
            end
        elseif l_z==2
            if l_e==1
                a2pVals_1=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), 2, ParamCell{:});
            elseif l_e==2
                a2pVals_1=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 2, ParamCell{:});
            elseif l_e==3
                a2pVals_1=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 2, ParamCell{:});
            elseif l_e==4
                a2pVals_1=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 2, ParamCell{:});
            end
        elseif l_z==3
            if l_e==1
                a2pVals_1=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), 2, ParamCell{:});
            elseif l_e==2
                a2pVals_1=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 2, ParamCell{:});
            elseif l_e==3
                a2pVals_1=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 2, ParamCell{:});
            elseif l_e==4
                a2pVals_1=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 2, ParamCell{:});
            end
        elseif l_z==4
            if l_e==1
                a2pVals_1=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), 2, ParamCell{:});
            elseif l_e==2
                a2pVals_1=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 2, ParamCell{:});
            elseif l_e==3
                a2pVals_1=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 2, ParamCell{:});
            elseif l_e==4
                a2pVals_1=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 2, ParamCell{:});
            end
        end
    elseif l_dexp==2
        if l_z==1
            if l_e==1
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), 2, ParamCell{:});
            elseif l_e==2
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 2, ParamCell{:});
            elseif l_e==3
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 2, ParamCell{:});
            elseif l_e==4
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 2, ParamCell{:});
            end
        elseif l_z==2
            if l_e==1
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), 2, ParamCell{:});
            elseif l_e==2
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 2, ParamCell{:});
            elseif l_e==3
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 2, ParamCell{:});
            elseif l_e==4
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 2, ParamCell{:});
            end
        elseif l_z==3
            if l_e==1
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), 2, ParamCell{:});
            elseif l_e==2
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 2, ParamCell{:});
            elseif l_e==3
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 2, ParamCell{:});
            elseif l_e==4
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 2, ParamCell{:});
            end
        elseif l_z==4
            if l_e==1
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), 2, ParamCell{:});
            elseif l_e==2
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 2, ParamCell{:});
            elseif l_e==3
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 2, ParamCell{:});
            elseif l_e==4
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 2, ParamCell{:});
            end
        end
    elseif l_dexp==3
        if l_z==1
            if l_e==1
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), 2, ParamCell{:});
            elseif l_e==2
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 2, ParamCell{:});
            elseif l_e==3
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 2, ParamCell{:});
            elseif l_e==4
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 2, ParamCell{:});
            end
        elseif l_z==2
            if l_e==1
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), 2, ParamCell{:});
            elseif l_e==2
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 2, ParamCell{:});
            elseif l_e==3
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 2, ParamCell{:});
            elseif l_e==4
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 2, ParamCell{:});
            end
        elseif l_z==3
            if l_e==1
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), 2, ParamCell{:});
            elseif l_e==2
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 2, ParamCell{:});
            elseif l_e==3
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 2, ParamCell{:});
            elseif l_e==4
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 2, ParamCell{:});
            end
        elseif l_z==4
            if l_e==1
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), 2, ParamCell{:});
            elseif l_e==2
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 2, ParamCell{:});
            elseif l_e==3
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 2, ParamCell{:});
            elseif l_e==4
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 2, ParamCell{:});
            end
        end
    elseif l_dexp==4
        if l_z==1
            if l_e==1
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), 2, ParamCell{:});
            elseif l_e==2
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 2, ParamCell{:});
            elseif l_e==3
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 2, ParamCell{:});
            elseif l_e==4
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 2, ParamCell{:});
            end
        elseif l_z==2
            if l_e==1
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), 2, ParamCell{:});
            elseif l_e==2
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 2, ParamCell{:});
            elseif l_e==3
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 2, ParamCell{:});
            elseif l_e==4
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 2, ParamCell{:});
            end
        elseif l_z==3
            if l_e==1
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), 2, ParamCell{:});
            elseif l_e==2
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 2, ParamCell{:});
            elseif l_e==3
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 2, ParamCell{:});
            elseif l_e==4
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 2, ParamCell{:});
            end
        elseif l_z==4
            if l_e==1
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), 2, ParamCell{:});
            elseif l_e==2
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), 2, ParamCell{:});
            elseif l_e==3
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), 2, ParamCell{:});
            elseif l_e==4
                a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 1, ParamCell{:});
                a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), 2, ParamCell{:});
            end
        end
    end

    %% Per-dim grid indexes and probs (inlined 1D linear-interp; mirrors the linter-style inlining in CreateExperienceAssetzeFnMatrix.m)
    a2_griddiff_1=a2_grid_1(2:end)-a2_grid_1(1:end-1);
    a2_griddiff_2=a2_grid_2(2:end)-a2_grid_2(1:end-1);
    N_total=numel(a2pVals_1);

    % --- a2 dim 1 ---
    if N_total*n_a2_1<1000000
        [~,loIdx_1]=max((a2_grid_1>a2pVals_1(:)'),[],1);
        loIdx_1=loIdx_1-1;
        loIdx_1(loIdx_1==0)=1;
        loIdx_1=loIdx_1(:); % force column: when n_a2_1==2, a2_griddiff_1 is scalar and would otherwise return shape of loIdx_1
        aprime_residual_1=a2pVals_1(:)-a2_grid_1(loIdx_1);
        prob_1=1-aprime_residual_1./a2_griddiff_1(loIdx_1);
        offTopOfGrid_1=(a2pVals_1(:)>=a2_grid_1(end));
        loIdx_1(offTopOfGrid_1)=n_a2_1-1;
        prob_1(offTopOfGrid_1)=0;
        offBottomOfGrid_1=(a2pVals_1(:)<=a2_grid_1(1));
        prob_1(offBottomOfGrid_1)=1;
    else
        loIdx_1=discretize(a2pVals_1,a2_grid_1);
        offBottomOfGrid_1=(a2pVals_1(:)<=a2_grid_1(1));
        loIdx_1(offBottomOfGrid_1)=1;
        offTopOfGrid_1=(a2pVals_1(:)>=a2_grid_1(end));
        loIdx_1(offTopOfGrid_1)=n_a2_1-1;
        loIdx_1=loIdx_1(:); % force column (see above)
        aprime_residual_1=a2pVals_1(:)-a2_grid_1(loIdx_1);
        prob_1=1-aprime_residual_1./a2_griddiff_1(loIdx_1);
        prob_1(offBottomOfGrid_1)=1;
        prob_1(offTopOfGrid_1)=0;
    end

    % --- a2 dim 2 ---
    if N_total*n_a2_2<1000000
        [~,loIdx_2]=max((a2_grid_2>a2pVals_2(:)'),[],1);
        loIdx_2=loIdx_2-1;
        loIdx_2(loIdx_2==0)=1;
        loIdx_2=loIdx_2(:); % force column: when n_a2_2==2, a2_griddiff_2 is scalar (see dim 1 note above)
        aprime_residual_2=a2pVals_2(:)-a2_grid_2(loIdx_2);
        prob_2=1-aprime_residual_2./a2_griddiff_2(loIdx_2);
        offTopOfGrid_2=(a2pVals_2(:)>=a2_grid_2(end));
        loIdx_2(offTopOfGrid_2)=n_a2_2-1;
        prob_2(offTopOfGrid_2)=0;
        offBottomOfGrid_2=(a2pVals_2(:)<=a2_grid_2(1));
        prob_2(offBottomOfGrid_2)=1;
    else
        loIdx_2=discretize(a2pVals_2,a2_grid_2);
        offBottomOfGrid_2=(a2pVals_2(:)<=a2_grid_2(1));
        loIdx_2(offBottomOfGrid_2)=1;
        offTopOfGrid_2=(a2pVals_2(:)>=a2_grid_2(end));
        loIdx_2(offTopOfGrid_2)=n_a2_2-1;
        loIdx_2=loIdx_2(:); % force column (see above)
        aprime_residual_2=a2pVals_2(:)-a2_grid_2(loIdx_2);
        prob_2=1-aprime_residual_2./a2_griddiff_2(loIdx_2);
        prob_2(offBottomOfGrid_2)=1;
        prob_2(offTopOfGrid_2)=0;
    end

    a2primeIndexes=zeros(N_a,l_a2,N_semizze,'gpuArray');
    a2primeProbs=zeros(N_a,l_a2,N_semizze,'gpuArray');
    a2primeIndexes(:,1,:)=reshape(loIdx_1,[N_a,1,N_semizze]);
    a2primeIndexes(:,2,:)=reshape(loIdx_2,[N_a,1,N_semizze]);
    a2primeProbs(:,1,:)=reshape(prob_1,[N_a,1,N_semizze]);
    a2primeProbs(:,2,:)=reshape(prob_2,[N_a,1,N_semizze]);
end

end
