function [a2primeIndexes, a2primeProbs]=CreateaprimePolicyExperienceAssete_J(Policy,aprimeFn, whichisdforexpasset, n_d, n_a1,n_a2, n_e,N_semiz,N_z,N_e, N_j, d_grid, a2_grid, e_gridvals_J, aprimeFnParams, fastOLG)
% Age-dependent (_J) version of CreateaprimePolicyExperienceAssete: compute
% a2prime=aprimeFn(d, a2, e) using the Policy-chosen d for each state, used
% in simulation / agent-distribution. e is i.i.d. drawn at the START of the
% period (so Policy DOES depend on e), and e enters aprimeFn directly.
%
% The input Policy will contain aprime (except for the experience asset)
% and the decision variables (d2, and where applicable d1). The output is
% just the Policy for a2prime (the experience asset). As well as the
% related probabilities.
%
% Companion file CreateExperienceAsseteFnMatrix_J.m does the same for ALL d
% (not just the Policy-chosen one), used during value-function iteration.
%
% Two layout modes controlled by the fastOLG flag:
%   fastOLG==0 :  state order is (a, semiz, z, e, j).
%   fastOLG==1 :  state order is (a, j, semiz, z, e).
%
% aprimeFnParams is passed as a [N_j, n_params] matrix and each column is
% shifted so j lives at the appropriate dimension for the chosen fastOLG.
%
% Output sizes:
%   fastOLG==0 : [N_a, N_semizze, N_j]
%   fastOLG==1 : [N_a, N_j, N_semizze]
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
l_e=length(n_e);

if nargin(aprimeFn)~=l_dexp+1+l_e+size(aprimeFnParams,2)
    error('Number of inputs to aprimeFn does not fit with size of aprimeFnParams')
end

if l_e>=5
    error('Max of four e variables supported in CreateaprimePolicyExperienceAssete_J (contact if you need more)')
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

        if N_semiz==0 && N_z==0
            N_semizze=N_e;
            d1vals=reshape(d1grid(Policy(whichisdforexpasset(1),:,:)),[N_a,1,1,N_e,N_j]);
            if l_dexp>=2
                d2grid=d_grid(sum(n_d(1:whichisdforexpasset(2)-1))+1:sum(n_d(1:whichisdforexpasset(2))));
                d2vals=reshape(d2grid(Policy(whichisdforexpasset(2),:,:)),[N_a,1,1,N_e,N_j]);
                if l_dexp>=3
                    d3grid=d_grid(sum(n_d(1:whichisdforexpasset(3)-1))+1:sum(n_d(1:whichisdforexpasset(3))));
                    d3vals=reshape(d3grid(Policy(whichisdforexpasset(3),:,:)),[N_a,1,1,N_e,N_j]);
                    if l_dexp>=4
                        d4grid=d_grid(sum(n_d(1:whichisdforexpasset(4)-1))+1:sum(n_d(1:whichisdforexpasset(4))));
                        d4vals=reshape(d4grid(Policy(whichisdforexpasset(4),:,:)),[N_a,1,1,N_e,N_j]);
                    end
                end
            end
        elseif N_semiz==0 && N_z>0
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
        elseif N_semiz>0 && N_z==0
            N_semizze=N_semiz*N_e;
            d1vals=reshape(d1grid(Policy(whichisdforexpasset(1),:,:)),[N_a,N_semiz,1,N_e,N_j]);
            if l_dexp>=2
                d2grid=d_grid(sum(n_d(1:whichisdforexpasset(2)-1))+1:sum(n_d(1:whichisdforexpasset(2))));
                d2vals=reshape(d2grid(Policy(whichisdforexpasset(2),:,:)),[N_a,N_semiz,1,N_e,N_j]);
                if l_dexp>=3
                    d3grid=d_grid(sum(n_d(1:whichisdforexpasset(3)-1))+1:sum(n_d(1:whichisdforexpasset(3))));
                    d3vals=reshape(d3grid(Policy(whichisdforexpasset(3),:,:)),[N_a,N_semiz,1,N_e,N_j]);
                    if l_dexp>=4
                        d4grid=d_grid(sum(n_d(1:whichisdforexpasset(4)-1))+1:sum(n_d(1:whichisdforexpasset(4))));
                        d4vals=reshape(d4grid(Policy(whichisdforexpasset(4),:,:)),[N_a,N_semiz,1,N_e,N_j]);
                    end
                end
            end
        elseif N_semiz>0 && N_z>0
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

    if N_a1==0
        a2vals=a2_grid;
    else
        a2vals=repelem(a2_grid,N_a1,1);
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

    %% expassete_J: aprime(d, a2, e)
    if l_dexp==1
        if l_e==1
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, e1vals, ParamCell{:});
        elseif l_e==2
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, e1vals, e2vals, ParamCell{:});
        elseif l_e==3
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, e1vals, e2vals, e3vals, ParamCell{:});
        elseif l_e==4
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
        end
    elseif l_dexp==2
        if l_e==1
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, e1vals, ParamCell{:});
        elseif l_e==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, e1vals, e2vals, ParamCell{:});
        elseif l_e==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, e1vals, e2vals, e3vals, ParamCell{:});
        elseif l_e==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
        end
    elseif l_dexp==3
        if l_e==1
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, e1vals, ParamCell{:});
        elseif l_e==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, e1vals, e2vals, ParamCell{:});
        elseif l_e==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, e1vals, e2vals, e3vals, ParamCell{:});
        elseif l_e==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
        end
    elseif l_dexp==4
        if l_e==1
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, e1vals, ParamCell{:});
        elseif l_e==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, e1vals, e2vals, ParamCell{:});
        elseif l_e==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, e1vals, e2vals, e3vals, ParamCell{:});
        elseif l_e==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
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

        if N_semiz==0 && N_z==0
            N_semizze=N_e;
            d1vals=reshape(d1grid(Policy(whichisdforexpasset(1),:,:,:)),[N_a,N_j,1,1,N_e]);
            if l_dexp>=2
                d2grid=d_grid(sum(n_d(1:whichisdforexpasset(2)-1))+1:sum(n_d(1:whichisdforexpasset(2))));
                d2vals=reshape(d2grid(Policy(whichisdforexpasset(2),:,:,:)),[N_a,N_j,1,1,N_e]);
                if l_dexp>=3
                    d3grid=d_grid(sum(n_d(1:whichisdforexpasset(3)-1))+1:sum(n_d(1:whichisdforexpasset(3))));
                    d3vals=reshape(d3grid(Policy(whichisdforexpasset(3),:,:,:)),[N_a,N_j,1,1,N_e]);
                    if l_dexp>=4
                        d4grid=d_grid(sum(n_d(1:whichisdforexpasset(4)-1))+1:sum(n_d(1:whichisdforexpasset(4))));
                        d4vals=reshape(d4grid(Policy(whichisdforexpasset(4),:,:,:)),[N_a,N_j,1,1,N_e]);
                    end
                end
            end
        elseif N_semiz==0 && N_z>0
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
        elseif N_semiz>0 && N_z==0
            N_semizze=N_semiz*N_e;
            d1vals=reshape(d1grid(Policy(whichisdforexpasset(1),:,:,:,:)),[N_a,N_j,N_semiz,1,N_e]);
            if l_dexp>=2
                d2grid=d_grid(sum(n_d(1:whichisdforexpasset(2)-1))+1:sum(n_d(1:whichisdforexpasset(2))));
                d2vals=reshape(d2grid(Policy(whichisdforexpasset(2),:,:,:,:)),[N_a,N_j,N_semiz,1,N_e]);
                if l_dexp>=3
                    d3grid=d_grid(sum(n_d(1:whichisdforexpasset(3)-1))+1:sum(n_d(1:whichisdforexpasset(3))));
                    d3vals=reshape(d3grid(Policy(whichisdforexpasset(3),:,:,:,:)),[N_a,N_j,N_semiz,1,N_e]);
                    if l_dexp>=4
                        d4grid=d_grid(sum(n_d(1:whichisdforexpasset(4)-1))+1:sum(n_d(1:whichisdforexpasset(4))));
                        d4vals=reshape(d4grid(Policy(whichisdforexpasset(4),:,:,:,:)),[N_a,N_j,N_semiz,1,N_e]);
                    end
                end
            end
        elseif N_semiz>0 && N_z>0
            N_semizze=N_semiz*N_z*N_e;
            d1vals=reshape(d1grid(Policy(whichisdforexpasset(1),:,:,:,:,:)),[N_a,N_j,N_semiz,N_z,N_e]);
            if l_dexp>=2
                d2grid=d_grid(sum(n_d(1:whichisdforexpasset(2)-1))+1:sum(n_d(1:whichisdforexpasset(2))));
                d2vals=reshape(d2grid(Policy(whichisdforexpasset(2),:,:,:,:,:)),[N_a,N_j,N_semiz,N_z,N_e]);
                if l_dexp>=3
                    d3grid=d_grid(sum(n_d(1:whichisdforexpasset(3)-1))+1:sum(n_d(1:whichisdforexpasset(3))));
                    d3vals=reshape(d3grid(Policy(whichisdforexpasset(3),:,:,:,:,:)),[N_a,N_j,N_semiz,N_z,N_e]);
                    if l_dexp>=4
                        d4grid=d_grid(sum(n_d(1:whichisdforexpasset(4)-1))+1:sum(n_d(1:whichisdforexpasset(4))));
                        d4vals=reshape(d4grid(Policy(whichisdforexpasset(4),:,:,:,:,:)),[N_a,N_j,N_semiz,N_z,N_e]);
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

    %% expassete_J: aprime(d, a2, e)
    if l_dexp==1
        if l_e==1
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, e1vals, ParamCell{:});
        elseif l_e==2
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, e1vals, e2vals, ParamCell{:});
        elseif l_e==3
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, e1vals, e2vals, e3vals, ParamCell{:});
        elseif l_e==4
            a2primeVals=arrayfun(aprimeFn, d1vals, a2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
        end
    elseif l_dexp==2
        if l_e==1
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, e1vals, ParamCell{:});
        elseif l_e==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, e1vals, e2vals, ParamCell{:});
        elseif l_e==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, e1vals, e2vals, e3vals, ParamCell{:});
        elseif l_e==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, a2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
        end
    elseif l_dexp==3
        if l_e==1
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, e1vals, ParamCell{:});
        elseif l_e==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, e1vals, e2vals, ParamCell{:});
        elseif l_e==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, e1vals, e2vals, e3vals, ParamCell{:});
        elseif l_e==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, a2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
        end
    elseif l_dexp==4
        if l_e==1
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, e1vals, ParamCell{:});
        elseif l_e==2
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, e1vals, e2vals, ParamCell{:});
        elseif l_e==3
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, e1vals, e2vals, e3vals, ParamCell{:});
        elseif l_e==4
            a2primeVals=arrayfun(aprimeFn, d1vals, d2vals, d3vals, d4vals, a2vals, e1vals, e2vals, e3vals, e4vals, ParamCell{:});
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
