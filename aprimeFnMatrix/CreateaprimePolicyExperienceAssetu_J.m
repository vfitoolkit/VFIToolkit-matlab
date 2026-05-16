function [a2primeIndexes, a2primeProbs]=CreateaprimePolicyExperienceAssetu_J(Policy,aprimeFn, whichisdforexpasset, n_d, n_a1,n_a2, N_z, n_u, N_j, d_grid, a2_grid, u_grid, aprimeFnParams, fastOLG)
% Age-dependent (_J) version of CreateaprimePolicyExperienceAssetu: compute
% a2prime=aprimeFn(d, a2, u) using the Policy-chosen d for each state, used
% in simulation / agent-distribution. u is i.i.d. drawn BETWEEN periods, so
% Policy does NOT depend on u; u only enters when computing a2prime from
% the already-chosen d.
%
% The input Policy will contain aprime (except for the experience asset)
% and the decision variables (d2, and where applicable d1). The output is
% just the Policy for a2prime (the experience asset). As well as the
% related probabilities.
%
% Companion file CreateExperienceAssetuFnMatrix_J.m does the same for ALL d
% (not just the Policy-chosen one), used during value-function iteration.
%
% Two layout modes controlled by the fastOLG flag (j ordered last vs second):
%   fastOLG==0 :  state order is (a, z, j).
%                 N_z==0: Policy is [L, N_a, N_j]; index Policy(k,:,:);
%                 reshape to [N_a, N_j].
%                 N_z>0:  Policy is [L, N_a, N_z, N_j]; index Policy(k,:,:)
%                 (Matlab linearises trailing dims); reshape to [N_a*N_z, N_j].
%   fastOLG==1 :  state order is (a, j, z).
%                 N_z==0: Policy is [L, N_a, N_j]; reshape to [N_a, N_j].
%                 N_z>0:  Policy is [L, N_a, N_j, N_z]; index Policy(k,:,:,:);
%                 reshape to [N_a, N_j, N_z].
%
% aprimeFnParams is passed as a [N_j, n_params] matrix and each column is
% shifted via shiftdim(...,-1) so that j is the SECOND dimension during
% arrayfun broadcasting.
%
% Output sizes:
%   fastOLG==0, N_z==0 : [N_a, N_j, N_u]
%   fastOLG==0, N_z>0  : [N_a, N_z, N_j, N_u]
%   fastOLG==1, N_z==0 : [N_a, N_j, N_u]
%   fastOLG==1, N_z>0  : [N_a, N_j, N_z, N_u]

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
N_u=prod(n_u);

l_dexp=length(whichisdforexpasset);
l_u=length(n_u);

if nargin(aprimeFn)~=l_dexp+1+l_u+size(aprimeFnParams,2)
    error('Number of inputs to aprimeFn does not fit with size of aprimeFnParams')
end

u_gridvals=CreateGridvals(n_u,u_grid,1);
if l_u>=5
    error('Max of four u variables supported (contact if you need more)')
end

if fastOLG==0 % state order (a, z, j)

    if N_z==0
        if l_dexp>=1
            if whichisdforexpasset(1)==1
                d1grid=d_grid(1:n_d(1));
            else
                d1grid=d_grid(sum(n_d(1:whichisdforexpasset(1)-1))+1:sum(n_d(1:whichisdforexpasset(1))));
            end
            d1vals=reshape(d1grid(Policy(whichisdforexpasset(1),:,:)),[N_a,N_j]);
            if l_dexp>=2
                d2grid=d_grid(sum(n_d(1:whichisdforexpasset(2)-1))+1:sum(n_d(1:whichisdforexpasset(2))));
                d2vals=reshape(d2grid(Policy(whichisdforexpasset(2),:,:)),[N_a,N_j]);
                if l_dexp>=3
                    d3grid=d_grid(sum(n_d(1:whichisdforexpasset(3)-1))+1:sum(n_d(1:whichisdforexpasset(3))));
                    d3vals=reshape(d3grid(Policy(whichisdforexpasset(3),:,:)),[N_a,N_j]);
                    if l_dexp>=4
                        d4grid=d_grid(sum(n_d(1:whichisdforexpasset(4)-1))+1:sum(n_d(1:whichisdforexpasset(4))));
                        d4vals=reshape(d4grid(Policy(whichisdforexpasset(4),:,:)),[N_a,N_j]);
                    end
                end
            end
        end
    else
        if l_dexp>=1
            if whichisdforexpasset(1)==1
                d1grid=d_grid(1:n_d(1));
            else
                d1grid=d_grid(sum(n_d(1:whichisdforexpasset(1)-1))+1:sum(n_d(1:whichisdforexpasset(1))));
            end
            d1vals=reshape(d1grid(Policy(whichisdforexpasset(1),:,:)),[N_a*N_z,N_j]);
            if l_dexp>=2
                d2grid=d_grid(sum(n_d(1:whichisdforexpasset(2)-1))+1:sum(n_d(1:whichisdforexpasset(2))));
                d2vals=reshape(d2grid(Policy(whichisdforexpasset(2),:,:)),[N_a*N_z,N_j]);
                if l_dexp>=3
                    d3grid=d_grid(sum(n_d(1:whichisdforexpasset(3)-1))+1:sum(n_d(1:whichisdforexpasset(3))));
                    d3vals=reshape(d3grid(Policy(whichisdforexpasset(3),:,:)),[N_a*N_z,N_j]);
                    if l_dexp>=4
                        d4grid=d_grid(sum(n_d(1:whichisdforexpasset(4)-1))+1:sum(n_d(1:whichisdforexpasset(4))));
                        d4vals=reshape(d4grid(Policy(whichisdforexpasset(4),:,:)),[N_a*N_z,N_j]);
                    end
                end
            end
        end
    end

    if N_a1==0
        if N_z==0
            a2vals=a2_grid;
        else
            a2vals=kron(ones(N_z,1),a2_grid);
        end
    else
        if N_z==0
            a2vals=kron(a2_grid,ones(N_a1,1));
        else
            a2vals=kron(ones(N_z,1),kron(a2_grid,ones(N_a1,1)));
        end
    end

    % u broadcasts in dim 3 (after a*z=dim1, j=dim2)
    if l_u>=1
        u1vals=shiftdim(u_gridvals(:,1),-2);
        if l_u>=2
            u2vals=shiftdim(u_gridvals(:,2),-2);
            if l_u>=3
                u3vals=shiftdim(u_gridvals(:,3),-2);
                if l_u>=4
                    u4vals=shiftdim(u_gridvals(:,4),-2);
                end
            end
        end
    end

    % expassetu_J: aprime(d, a2, u)
    % Removed: a2vals=a2vals.*ones(1,1,1,'gpuArray'); % was here to fool matlab which otherwise threw an error; restore this line if functionality breaks
    if l_u==1
        if l_dexp==1
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, u1vals, ParamCell{:});
        elseif l_dexp==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, u1vals, ParamCell{:});
        elseif l_dexp==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, u1vals, ParamCell{:});
        elseif l_dexp==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, u1vals, ParamCell{:});
        end
    elseif l_u==2
        if l_dexp==1
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, u1vals, u2vals, ParamCell{:});
        elseif l_dexp==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, u1vals, u2vals, ParamCell{:});
        elseif l_dexp==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, u1vals, u2vals, ParamCell{:});
        elseif l_dexp==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, u1vals, u2vals, ParamCell{:});
        end
    elseif l_u==3
        if l_dexp==1
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, u1vals, u2vals, u3vals, ParamCell{:});
        elseif l_dexp==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, u1vals, u2vals, u3vals, ParamCell{:});
        elseif l_dexp==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, u1vals, u2vals, u3vals, ParamCell{:});
        elseif l_dexp==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, u1vals, u2vals, u3vals, ParamCell{:});
        end
    elseif l_u==4
        if l_dexp==1
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, u1vals, u2vals, u3vals, u4vals, ParamCell{:});
        elseif l_dexp==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, u1vals, u2vals, u3vals, u4vals, ParamCell{:});
        elseif l_dexp==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, u1vals, u2vals, u3vals, u4vals, ParamCell{:});
        elseif l_dexp==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, u1vals, u2vals, u3vals, u4vals, ParamCell{:});
        end
    end

    if N_z==0
        a2primeVals=reshape(a2primeVals,[1,N_a*N_j*N_u]);
    else
        a2primeVals=reshape(a2primeVals,[1,N_a*N_z*N_j*N_u]);
    end

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

    if N_z==0
        a2primeIndexes=reshape(a2primeIndexes,[N_a,N_j,N_u]);
        a2primeProbs=reshape(a2primeProbs,[N_a,N_j,N_u]);
    else
        a2primeIndexes=reshape(a2primeIndexes,[N_a,N_z,N_j,N_u]);
        a2primeProbs=reshape(a2primeProbs,[N_a,N_z,N_j,N_u]);
    end


elseif fastOLG==1 % state order (a, j, z)

    if N_z==0
        if l_dexp>=1
            if whichisdforexpasset(1)==1
                d1grid=d_grid(1:n_d(1));
            else
                d1grid=d_grid(sum(n_d(1:whichisdforexpasset(1)-1))+1:sum(n_d(1:whichisdforexpasset(1))));
            end
            d1vals=reshape(d1grid(Policy(whichisdforexpasset(1),:,:)),[N_a,N_j]);
            if l_dexp>=2
                d2grid=d_grid(sum(n_d(1:whichisdforexpasset(2)-1))+1:sum(n_d(1:whichisdforexpasset(2))));
                d2vals=reshape(d2grid(Policy(whichisdforexpasset(2),:,:)),[N_a,N_j]);
                if l_dexp>=3
                    d3grid=d_grid(sum(n_d(1:whichisdforexpasset(3)-1))+1:sum(n_d(1:whichisdforexpasset(3))));
                    d3vals=reshape(d3grid(Policy(whichisdforexpasset(3),:,:)),[N_a,N_j]);
                    if l_dexp>=4
                        d4grid=d_grid(sum(n_d(1:whichisdforexpasset(4)-1))+1:sum(n_d(1:whichisdforexpasset(4))));
                        d4vals=reshape(d4grid(Policy(whichisdforexpasset(4),:,:)),[N_a,N_j]);
                    end
                end
            end
        end
    else
        if l_dexp>=1
            if whichisdforexpasset(1)==1
                d1grid=d_grid(1:n_d(1));
            else
                d1grid=d_grid(sum(n_d(1:whichisdforexpasset(1)-1))+1:sum(n_d(1:whichisdforexpasset(1))));
            end
            d1vals=reshape(d1grid(Policy(whichisdforexpasset(1),:,:,:)),[N_a,N_j,N_z]);
            if l_dexp>=2
                d2grid=d_grid(sum(n_d(1:whichisdforexpasset(2)-1))+1:sum(n_d(1:whichisdforexpasset(2))));
                d2vals=reshape(d2grid(Policy(whichisdforexpasset(2),:,:,:)),[N_a,N_j,N_z]);
                if l_dexp>=3
                    d3grid=d_grid(sum(n_d(1:whichisdforexpasset(3)-1))+1:sum(n_d(1:whichisdforexpasset(3))));
                    d3vals=reshape(d3grid(Policy(whichisdforexpasset(3),:,:,:)),[N_a,N_j,N_z]);
                    if l_dexp>=4
                        d4grid=d_grid(sum(n_d(1:whichisdforexpasset(4)-1))+1:sum(n_d(1:whichisdforexpasset(4))));
                        d4vals=reshape(d4grid(Policy(whichisdforexpasset(4),:,:,:)),[N_a,N_j,N_z]);
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

    % u broadcasts depending on whether z is present
    if N_z==0
        if l_u>=1
            u1vals=shiftdim(u_gridvals(:,1),-2); % dim 3
            if l_u>=2
                u2vals=shiftdim(u_gridvals(:,2),-2);
                if l_u>=3
                    u3vals=shiftdim(u_gridvals(:,3),-2);
                    if l_u>=4
                        u4vals=shiftdim(u_gridvals(:,4),-2);
                    end
                end
            end
        end
    else
        if l_u>=1
            u1vals=shiftdim(u_gridvals(:,1),-3); % dim 4 (a=1, j=2, z=3, u=4)
            if l_u>=2
                u2vals=shiftdim(u_gridvals(:,2),-3);
                if l_u>=3
                    u3vals=shiftdim(u_gridvals(:,3),-3);
                    if l_u>=4
                        u4vals=shiftdim(u_gridvals(:,4),-3);
                    end
                end
            end
        end
    end

    % Removed: a2vals=a2vals.*ones(1,1,1,'gpuArray'); % was here to fool matlab which otherwise threw an error; restore this line if functionality breaks
    if l_u==1
        if l_dexp==1
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, u1vals, ParamCell{:});
        elseif l_dexp==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, u1vals, ParamCell{:});
        elseif l_dexp==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, u1vals, ParamCell{:});
        elseif l_dexp==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, u1vals, ParamCell{:});
        end
    elseif l_u==2
        if l_dexp==1
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, u1vals, u2vals, ParamCell{:});
        elseif l_dexp==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, u1vals, u2vals, ParamCell{:});
        elseif l_dexp==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, u1vals, u2vals, ParamCell{:});
        elseif l_dexp==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, u1vals, u2vals, ParamCell{:});
        end
    elseif l_u==3
        if l_dexp==1
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, u1vals, u2vals, u3vals, ParamCell{:});
        elseif l_dexp==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, u1vals, u2vals, u3vals, ParamCell{:});
        elseif l_dexp==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, u1vals, u2vals, u3vals, ParamCell{:});
        elseif l_dexp==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, u1vals, u2vals, u3vals, ParamCell{:});
        end
    elseif l_u==4
        if l_dexp==1
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, u1vals, u2vals, u3vals, u4vals, ParamCell{:});
        elseif l_dexp==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, u1vals, u2vals, u3vals, u4vals, ParamCell{:});
        elseif l_dexp==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, u1vals, u2vals, u3vals, u4vals, ParamCell{:});
        elseif l_dexp==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, u1vals, u2vals, u3vals, u4vals, ParamCell{:});
        end
    end

    if N_z==0
        a2primeVals=reshape(a2primeVals,[1,N_a*N_j*N_u]);
    else
        a2primeVals=reshape(a2primeVals,[1,N_a*N_j*N_z*N_u]);
    end

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

    if N_z==0
        a2primeIndexes=reshape(a2primeIndexes,[N_a,N_j,N_u]);
        a2primeProbs=reshape(a2primeProbs,[N_a,N_j,N_u]);
    else
        a2primeIndexes=reshape(a2primeIndexes,[N_a,N_j,N_z,N_u]);
        a2primeProbs=reshape(a2primeProbs,[N_a,N_j,N_z,N_u]);
    end


end

end
