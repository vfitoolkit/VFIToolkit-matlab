function [a2primeIndexes, a2primeProbs]=CreateaprimePolicyExperienceAsset_J(Policy,aprimeFn, whichisdforexpasset, n_d, n_a1,n_a2, N_semizze, N_j, d_grid, a2_grid, aprimeFnParams, fastOLG)
% Age-dependent (_J) version of CreateaprimePolicyExperienceAsset.
%
% Output sizes:
%   l_a2==1 (legacy):
%     N_semizze==0 : [N_a, N_j]
%     fastOLG==0   : [N_a, N_semizze, N_j]
%     fastOLG==1   : [N_a, N_j, N_semizze]
%   l_a2==2 (multi-dim, per-dim factored): same layout with l_a2 dim inserted just before N_j:
%     N_semizze==0 : [N_a, l_a2, N_j]
%     fastOLG==0   : [N_a, l_a2, N_semizze, N_j]
%     fastOLG==1   : [N_a, l_a2, N_j, N_semizze]
%   a2primeIndexes(:,k,...) = lower-grid index in a2_k dim; a2primeProbs(:,k,...) = prob of lower.
%   Caller does nested 2-corner interp with skipinterp at each level.

ParamCell=cell(size(aprimeFnParams,2),1);
if fastOLG==0 && N_semizze>0
    % d1vals shape is [N_a, N_semizze, N_j] in this branch; N_j sits at dim 3
    paramShift=-2;
else
    paramShift=-1;
end
for ii=1:size(aprimeFnParams,2)
    ParamCell(ii,1)={shiftdim(aprimeFnParams(:,ii),paramShift)};
end

N_a1=prod(n_a1);
if N_a1==0
    N_a=prod(n_a2);
else
    N_a=prod([n_a1,n_a2]);
end
N_a2=prod(n_a2);

l_dexp=length(whichisdforexpasset);
l_a2=length(n_a2);

if l_a2>2
    error('experienceasset currently supports length(n_a2) in {1,2}')
end

if nargin(aprimeFn)~=l_dexp+l_a2+(l_a2>=2)+size(aprimeFnParams,2)
    % When l_a2>=2, aprimeFn takes an extra 'whicha' integer selector slot after the a2 inputs.
    error('Number of inputs to aprimeFn does not fit with size of aprimeFnParams')
end


if l_a2==1
    % ===================== legacy l_a2==1 (unchanged) =====================

    if fastOLG==0 % (a,z,j)
        if N_semizze==0
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
                d1vals=reshape(d1grid(Policy(whichisdforexpasset(1),:,:)),[N_a,N_semizze,N_j]);
                if l_dexp>=2
                    d2grid=d_grid(sum(n_d(1:whichisdforexpasset(2)-1))+1:sum(n_d(1:whichisdforexpasset(2))));
                    d2vals=reshape(d2grid(Policy(whichisdforexpasset(2),:,:)),[N_a,N_semizze,N_j]);
                    if l_dexp>=3
                        d3grid=d_grid(sum(n_d(1:whichisdforexpasset(3)-1))+1:sum(n_d(1:whichisdforexpasset(3))));
                        d3vals=reshape(d3grid(Policy(whichisdforexpasset(3),:,:)),[N_a,N_semizze,N_j]);
                        if l_dexp>=4
                            d4grid=d_grid(sum(n_d(1:whichisdforexpasset(4)-1))+1:sum(n_d(1:whichisdforexpasset(4))));
                            d4vals=reshape(d4grid(Policy(whichisdforexpasset(4),:,:)),[N_a,N_semizze,N_j]);
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

        a2vals=a2vals.*ones(1,1,1,'gpuArray');
        if l_dexp==1
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, ParamCell{:});
        elseif l_dexp==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, ParamCell{:});
        elseif l_dexp==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, ParamCell{:});
        elseif l_dexp==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, ParamCell{:});
        end


        if N_semizze==0
            a2primeVals=reshape(a2primeVals,[1,N_a,N_j]);
        else
            a2primeVals=reshape(a2primeVals,[1,N_a*N_semizze,N_j]);
        end

        a2_griddiff=a2_grid(2:end)-a2_grid(1:end-1);

        a2primeIndexes=discretize(a2primeVals,a2_grid);

        offBottomOfGrid=(a2primeVals<=a2_grid(1));
        a2primeIndexes(offBottomOfGrid)=1;
        offTopOfGrid=(a2primeVals>=a2_grid(end));
        a2primeIndexes(offTopOfGrid)=n_a2-1;
        if N_semizze==0
            a2primeIndexes=reshape(a2primeIndexes,[N_a,N_j]);
        else
            a2primeIndexes=reshape(a2primeIndexes,[N_a*N_semizze,N_j]);
        end

        aprime_residual=shiftdim(a2primeVals,1)-a2_grid(a2primeIndexes);
        a2primeProbs=1-aprime_residual./a2_griddiff(a2primeIndexes);
        a2primeProbs(offBottomOfGrid)=1;
        a2primeProbs(offTopOfGrid)=0;

        if N_semizze==0
            a2primeIndexes=reshape(a2primeIndexes,[N_a,N_j]);
            a2primeProbs=reshape(a2primeProbs,[N_a,N_j]);
        else
            a2primeIndexes=reshape(a2primeIndexes,[N_a,N_semizze,N_j]);
            a2primeProbs=reshape(a2primeProbs,[N_a,N_semizze,N_j]);
        end

    elseif fastOLG==1 % (a,j,z)
        if N_semizze==0
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
                d1vals=reshape(d1grid(Policy(whichisdforexpasset(1),:,:,:)),[N_a,N_j,N_semizze]);
                if l_dexp>=2
                    d2grid=d_grid(sum(n_d(1:whichisdforexpasset(2)-1))+1:sum(n_d(1:whichisdforexpasset(2))));
                    d2vals=reshape(d2grid(Policy(whichisdforexpasset(2),:,:,:)),[N_a,N_j,N_semizze]);
                    if l_dexp>=3
                        d3grid=d_grid(sum(n_d(1:whichisdforexpasset(3)-1))+1:sum(n_d(1:whichisdforexpasset(3))));
                        d3vals=reshape(d3grid(Policy(whichisdforexpasset(3),:,:,:)),[N_a,N_j,N_semizze]);
                        if l_dexp>=4
                            d4grid=d_grid(sum(n_d(1:whichisdforexpasset(4)-1))+1:sum(n_d(1:whichisdforexpasset(4))));
                            d4vals=reshape(d4grid(Policy(whichisdforexpasset(4),:,:,:)),[N_a,N_j,N_semizze]);
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

        a2vals=a2vals.*ones(1,1,1,'gpuArray');
        if l_dexp==1
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, ParamCell{:});
        elseif l_dexp==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, ParamCell{:});
        elseif l_dexp==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, ParamCell{:});
        elseif l_dexp==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, ParamCell{:});
        end


        if N_semizze==0
            a2primeVals=reshape(a2primeVals,[1,N_a,N_j]);
        else
            a2primeVals=reshape(a2primeVals,[1,N_a,N_j,N_semizze]);
        end

        a2_griddiff=a2_grid(2:end)-a2_grid(1:end-1);

        a2primeIndexes=discretize(a2primeVals,a2_grid);

        offBottomOfGrid=(a2primeVals<=a2_grid(1));
        a2primeIndexes(offBottomOfGrid)=1;
        offTopOfGrid=(a2primeVals>=a2_grid(end));
        a2primeIndexes(offTopOfGrid)=n_a2-1;
        if N_semizze==0
            a2primeIndexes=reshape(a2primeIndexes,[N_a,N_j]);
        else
            a2primeIndexes=reshape(a2primeIndexes,[N_a,N_j,N_semizze]);
        end

        aprime_residual=shiftdim(a2primeVals,1)-a2_grid(a2primeIndexes);
        a2primeProbs=1-aprime_residual./a2_griddiff(a2primeIndexes);
        a2primeProbs(offBottomOfGrid)=1;
        a2primeProbs(offTopOfGrid)=0;

        if N_semizze==0
            a2primeIndexes=reshape(a2primeIndexes,[N_a,N_j]);
            a2primeProbs=reshape(a2primeProbs,[N_a,N_j]);
        else
            a2primeIndexes=reshape(a2primeIndexes,[N_a,N_j,N_semizze]);
            a2primeProbs=reshape(a2primeProbs,[N_a,N_j,N_semizze]);
        end
    end

elseif l_a2==2
    % ===================== multi-dim a2 (l_a2==2): Kaprimepts=4 corners =====================
    n_a2_1=n_a2(1); n_a2_2=n_a2(2);
    a2_grid_1=a2_grid(1:n_a2_1);
    a2_grid_2=a2_grid(n_a2_1+1:n_a2_1+n_a2_2);
    a2_gridvals=CreateGridvals(n_a2,a2_grid,1); % [N_a2, 2]

    if N_a1==0
        a2vals_1=a2_gridvals(:,1);
        a2vals_2=a2_gridvals(:,2);
    else
        a2vals_1=repelem(a2_gridvals(:,1),N_a1,1); % [N_a,1]
        a2vals_2=repelem(a2_gridvals(:,2),N_a1,1);
    end

    if fastOLG==0 % (a,z,j); d*vals shape [N_a,N_j] or [N_a,N_semizze,N_j]
        if N_semizze==0
            d_shape={[N_a,N_j]};
            policy_idx_args={':' ,':'};
        else
            d_shape={[N_a,N_semizze,N_j]};
            policy_idx_args={':',':',':'};
        end
    else % fastOLG==1 (a,j,z); d*vals shape [N_a,N_j] or [N_a,N_j,N_semizze]
        if N_semizze==0
            d_shape={[N_a,N_j]};
            policy_idx_args={':',':'};
        else
            d_shape={[N_a,N_j,N_semizze]};
            policy_idx_args={':',':',':'};
        end
    end
    target_shape=d_shape{1};

    % Build per-d Policy slices (mirrors legacy logic, using target_shape)
    if l_dexp>=1
        if whichisdforexpasset(1)==1
            d1grid=d_grid(1:n_d(1));
        else
            d1grid=d_grid(sum(n_d(1:whichisdforexpasset(1)-1))+1:sum(n_d(1:whichisdforexpasset(1))));
        end
        d1vals=reshape(d1grid(Policy(whichisdforexpasset(1),policy_idx_args{:})),target_shape);
        if l_dexp>=2
            d2grid=d_grid(sum(n_d(1:whichisdforexpasset(2)-1))+1:sum(n_d(1:whichisdforexpasset(2))));
            d2vals=reshape(d2grid(Policy(whichisdforexpasset(2),policy_idx_args{:})),target_shape);
            if l_dexp>=3
                d3grid=d_grid(sum(n_d(1:whichisdforexpasset(3)-1))+1:sum(n_d(1:whichisdforexpasset(3))));
                d3vals=reshape(d3grid(Policy(whichisdforexpasset(3),policy_idx_args{:})),target_shape);
                if l_dexp>=4
                    d4grid=d_grid(sum(n_d(1:whichisdforexpasset(4)-1))+1:sum(n_d(1:whichisdforexpasset(4))));
                    d4vals=reshape(d4grid(Policy(whichisdforexpasset(4),policy_idx_args{:})),target_shape);
                end
            end
        end
    end

    % Broadcast a2vals_* to target_shape
    a2vals_1=a2vals_1.*ones(1,1,1,'gpuArray');
    a2vals_2=a2vals_2.*ones(1,1,1,'gpuArray');

    % GPU arrayfun requires scalar output; call once per a2 dim with whicha selector
    if l_dexp==1
        a2pVals_1=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, 1, ParamCell{:});
        a2pVals_2=arrayfun(aprimeFn, d1vals, a2vals_1, a2vals_2, 2, ParamCell{:});
    elseif l_dexp==2
        a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, 1, ParamCell{:});
        a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, a2vals_1, a2vals_2, 2, ParamCell{:});
    elseif l_dexp==3
        a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, 1, ParamCell{:});
        a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals_1, a2vals_2, 2, ParamCell{:});
    elseif l_dexp==4
        a2pVals_1=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, 1, ParamCell{:});
        a2pVals_2=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals_1, a2vals_2, 2, ParamCell{:});
    end

    [loIdx_1, prob_1]=local_interp1d(a2pVals_1, a2_grid_1, n_a2_1);
    [loIdx_2, prob_2]=local_interp1d(a2pVals_2, a2_grid_2, n_a2_2);

    % Per-dim factored output (NOT Kron-folded):
    %   a2primeIndexes(:,k,...) = lower-grid index in a2_k dim
    %   a2primeProbs(:,k,...)   = probability of lower grid point in a2_k dim
    Ntot=numel(loIdx_1);
    a2primeIndexes_flat=zeros(Ntot,l_a2,'gpuArray');
    a2primeProbs_flat=zeros(Ntot,l_a2,'gpuArray');
    a2primeIndexes_flat(:,1)=loIdx_1(:);
    a2primeIndexes_flat(:,2)=loIdx_2(:);
    a2primeProbs_flat(:,1)=prob_1(:);
    a2primeProbs_flat(:,2)=prob_2(:);

    % Reshape: l_a2 dim inserted after N_a, before remaining dims (mirrors prior Kaprimepts layout)
    if fastOLG==0
        if N_semizze==0
            a2primeIndexes=reshape(a2primeIndexes_flat,[N_a,N_j,l_a2]); a2primeIndexes=permute(a2primeIndexes,[1,3,2]); % [N_a,l_a2,N_j]
            a2primeProbs=reshape(a2primeProbs_flat,[N_a,N_j,l_a2]); a2primeProbs=permute(a2primeProbs,[1,3,2]);
        else
            a2primeIndexes=reshape(a2primeIndexes_flat,[N_a,N_semizze,N_j,l_a2]); a2primeIndexes=permute(a2primeIndexes,[1,4,2,3]); % [N_a,l_a2,N_semizze,N_j]
            a2primeProbs=reshape(a2primeProbs_flat,[N_a,N_semizze,N_j,l_a2]); a2primeProbs=permute(a2primeProbs,[1,4,2,3]);
        end
    else % fastOLG==1
        if N_semizze==0
            a2primeIndexes=reshape(a2primeIndexes_flat,[N_a,N_j,l_a2]); a2primeIndexes=permute(a2primeIndexes,[1,3,2]); % [N_a,l_a2,N_j]
            a2primeProbs=reshape(a2primeProbs_flat,[N_a,N_j,l_a2]); a2primeProbs=permute(a2primeProbs,[1,3,2]);
        else
            a2primeIndexes=reshape(a2primeIndexes_flat,[N_a,N_j,N_semizze,l_a2]); a2primeIndexes=permute(a2primeIndexes,[1,4,2,3]); % [N_a,l_a2,N_j,N_semizze]
            a2primeProbs=reshape(a2primeProbs_flat,[N_a,N_j,N_semizze,l_a2]); a2primeProbs=permute(a2primeProbs,[1,4,2,3]);
        end
    end
end


end


function [loIdx, prob]=local_interp1d(aprimeVals, grid, n_grid)
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
