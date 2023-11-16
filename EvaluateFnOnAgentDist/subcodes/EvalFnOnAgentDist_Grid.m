function Values=EvalFnOnAgentDist_Grid(FnToEvaluate,FnToEvaluateParams,PolicyValues,l_daprime,n_a,n_z,a_gridvals,z_gridvals)
% Note: This also handles e and semiz, just put them together with the z as for this function there is no difference
% Note: z_grid needs to be a joint-grid

N_a=prod(n_a);
N_z=prod(n_z);
l_a=length(n_a);
if N_z==0
    l_z=0;
else
    l_z=length(n_z);
end

ParamCell=cell(length(FnToEvaluateParams),1);
for ii=1:length(FnToEvaluateParams)
    ParamCell(ii,1)={FnToEvaluateParams(ii)};
end

if N_z==0
    PolicyValues=reshape(PolicyValues,[size(PolicyValues,1),N_a]);
    PolicyValues=PolicyValues'; %[N_a,l_daprime]
else
    PolicyValues=reshape(PolicyValues,[size(PolicyValues,1),N_a,N_z]);
    PolicyValues=permute(PolicyValues,[2,3,1]); %[N_a,N_z,l_daprime]
end

if N_z==0
    if l_daprime>=1
        daprime1vals=PolicyValues(:,1);
        if l_daprime>=2
            daprime2vals=PolicyValues(:,2);
            if l_daprime>=3
                daprime3vals=PolicyValues(:,3);
                if l_daprime>=4
                    daprime4vals=PolicyValues(:,4);
                    if l_daprime>=5
                        daprime5vals=PolicyValues(:,5);
                    end
                end
            end
        end
    end
else
    % z_gridvals=shiftdim(CreateGridvals(n_z,z_grid,1),1);
    % assumes joint z_grid
    z_gridvals=shiftdim(z_gridvals,-1);

    if l_daprime>=1
        daprime1vals=PolicyValues(:,:,1);
        if l_daprime>=2
            daprime2vals=PolicyValues(:,:,2);
            if l_daprime>=3
                daprime3vals=PolicyValues(:,:,3);
                if l_daprime>=4
                    daprime4vals=PolicyValues(:,:,4);
                    if l_daprime>=5
                        daprime5vals=PolicyValues(:,:,5);
                    end
                end
            end
        end
    end
end


if l_daprime==1
    if l_a==1 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals, a_gridvals(:,1), ParamCell{:});
    elseif l_daprime==1 && l_a==1 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals, a_gridvals(:,1), z_gridvals(1,:,1), ParamCell{:});
    elseif l_daprime==1 && l_a==1 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals, a_gridvals(:,1), z_gridvals(1,:,1),z_gridvals(1,:,2), ParamCell{:});
    elseif l_daprime==1 && l_a==1 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals, a_gridvals(:,1), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3), ParamCell{:});
    elseif l_daprime==1 && l_a==1 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals, a_gridvals(:,1), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3),z_gridvals(1,:,4), ParamCell{:});
    elseif l_daprime==1 && l_a==2 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals, a_gridvals(:,1),a_gridvals(:,2), ParamCell{:});
    elseif l_daprime==1 && l_a==2 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals, a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,:,1), ParamCell{:});
    elseif l_daprime==1 && l_a==2 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals, a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,:,1),z_gridvals(1,:,2), ParamCell{:});
    elseif l_daprime==1 && l_a==2 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals, a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3), ParamCell{:});
    elseif l_daprime==1 && l_a==2 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals, a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3),z_gridvals(1,:,4), ParamCell{:});
    elseif l_daprime==1 && l_a==3 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), ParamCell{:});
    elseif l_daprime==1 && l_a==3 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,:,1), ParamCell{:});
    elseif l_daprime==1 && l_a==3 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,:,1),z_gridvals(1,:,2), ParamCell{:});
    elseif l_daprime==1 && l_a==3 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3), ParamCell{:});
    elseif l_daprime==1 && l_a==3 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3),z_gridvals(1,:,4), ParamCell{:});
    elseif l_daprime==1 && l_a==4 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), ParamCell{:});
    elseif l_daprime==1 && l_a==4 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,:,1), ParamCell{:});
    elseif l_daprime==1 && l_a==4 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,:,1),z_gridvals(1,:,2), ParamCell{:});
    elseif l_daprime==1 && l_a==4 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3), ParamCell{:});
    elseif l_daprime==1 && l_a==4 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3),z_gridvals(1,:,4), ParamCell{:});
    end
elseif l_daprime==2
    if l_a==1 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_gridvals(:,1), ParamCell{:});
    elseif l_daprime==2 && l_a==1 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_gridvals(:,1), z_gridvals(1,:,1), ParamCell{:});
    elseif l_daprime==2 && l_a==1 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_gridvals(:,1), z_gridvals(1,:,1),z_gridvals(1,:,2), ParamCell{:});
    elseif l_daprime==2 && l_a==1 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_gridvals(:,1), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3), ParamCell{:});
    elseif l_daprime==2 && l_a==1 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_gridvals(:,1), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3),z_gridvals(1,:,4), ParamCell{:});
    elseif l_daprime==2 && l_a==2 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_gridvals(:,1),a_gridvals(:,2), ParamCell{:});
    elseif l_daprime==2 && l_a==2 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,:,1), ParamCell{:});
    elseif l_daprime==2 && l_a==2 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,:,1),z_gridvals(1,:,2), ParamCell{:});
    elseif l_daprime==2 && l_a==2 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3), ParamCell{:});
    elseif l_daprime==2 && l_a==2 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3),z_gridvals(1,:,4), ParamCell{:});
    elseif l_daprime==2 && l_a==3 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), ParamCell{:});
    elseif l_daprime==2 && l_a==3 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,:,1), ParamCell{:});
    elseif l_daprime==2 && l_a==3 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,:,1),z_gridvals(1,:,2), ParamCell{:});
    elseif l_daprime==2 && l_a==3 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3), ParamCell{:});
    elseif l_daprime==2 && l_a==3 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3),z_gridvals(1,:,4), ParamCell{:});
    elseif l_daprime==2 && l_a==4 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), ParamCell{:});
    elseif l_daprime==2 && l_a==4 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,:,1), ParamCell{:});
    elseif l_daprime==2 && l_a==4 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,:,1),z_gridvals(1,:,2), ParamCell{:});
    elseif l_daprime==2 && l_a==4 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3), ParamCell{:});
    elseif l_daprime==2 && l_a==4 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3),z_gridvals(1,:,4), ParamCell{:});
    end
elseif l_daprime==3
    if l_a==1 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_gridvals(:,1), ParamCell{:});
    elseif l_daprime==2 && l_a==1 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_gridvals(:,1), z_gridvals(1,:,1), ParamCell{:});
    elseif l_daprime==2 && l_a==1 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_gridvals(:,1), z_gridvals(1,:,1),z_gridvals(1,:,2), ParamCell{:});
    elseif l_daprime==2 && l_a==1 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_gridvals(:,1), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3), ParamCell{:});
    elseif l_daprime==2 && l_a==1 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_gridvals(:,1), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3),z_gridvals(1,:,4), ParamCell{:});
    elseif l_a==2 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_gridvals(:,1),a_gridvals(:,2), ParamCell{:});
    elseif l_daprime==3 && l_a==2 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,:,1), ParamCell{:});
    elseif l_daprime==3 && l_a==2 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,:,1),z_gridvals(1,:,2), ParamCell{:});
    elseif l_daprime==3 && l_a==2 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3), ParamCell{:});
    elseif l_daprime==3 && l_a==2 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3),z_gridvals(1,:,4), ParamCell{:});
    elseif l_daprime==3 && l_a==3 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), ParamCell{:});
    elseif l_daprime==3 && l_a==3 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,:,1), ParamCell{:});
    elseif l_daprime==3 && l_a==3 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,:,1),z_gridvals(1,:,2), ParamCell{:});
    elseif l_daprime==3 && l_a==3 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3), ParamCell{:});
    elseif l_daprime==3 && l_a==3 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3),z_gridvals(1,:,4), ParamCell{:});
    elseif l_daprime==3 && l_a==4 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), ParamCell{:});
    elseif l_daprime==3 && l_a==4 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,:,1), ParamCell{:});
    elseif l_daprime==3 && l_a==4 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,:,1),z_gridvals(1,:,2), ParamCell{:});
    elseif l_daprime==3 && l_a==4 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3), ParamCell{:});
    elseif l_daprime==3 && l_a==4 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3),z_gridvals(1,:,4), ParamCell{:});
    end
elseif l_daprime==4
    if l_a==1 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_gridvals(:,1), ParamCell{:});
    elseif l_daprime==2 && l_a==1 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_gridvals(:,1), z_gridvals(1,:,1), ParamCell{:});
    elseif l_daprime==2 && l_a==1 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_gridvals(:,1), z_gridvals(1,:,1),z_gridvals(1,:,2), ParamCell{:});
    elseif l_daprime==2 && l_a==1 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_gridvals(:,1), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3), ParamCell{:});
    elseif l_daprime==2 && l_a==1 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_gridvals(:,1), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3),z_gridvals(1,:,4), ParamCell{:});
    elseif l_a==2 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_gridvals(:,1),a_gridvals(:,2), ParamCell{:});
    elseif l_daprime==4 && l_a==2 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,:,1), ParamCell{:});
    elseif l_daprime==4 && l_a==2 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,:,1),z_gridvals(1,:,2), ParamCell{:});
    elseif l_daprime==4 && l_a==2 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3), ParamCell{:});
    elseif l_daprime==4 && l_a==2 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3),z_gridvals(1,:,4), ParamCell{:});
    elseif l_daprime==4 && l_a==3 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), ParamCell{:});
    elseif l_daprime==4 && l_a==3 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,:,1), ParamCell{:});
    elseif l_daprime==4 && l_a==3 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,:,1),z_gridvals(1,:,2), ParamCell{:});
    elseif l_daprime==4 && l_a==3 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3), ParamCell{:});
    elseif l_daprime==4 && l_a==3 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3),z_gridvals(1,:,4), ParamCell{:});
    elseif l_daprime==4 && l_a==4 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), ParamCell{:});
    elseif l_daprime==4 && l_a==4 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,:,1), ParamCell{:});
    elseif l_daprime==4 && l_a==4 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,:,1),z_gridvals(1,:,2), ParamCell{:});
    elseif l_daprime==4 && l_a==4 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3), ParamCell{:});
    elseif l_daprime==4 && l_a==4 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3),z_gridvals(1,:,4), ParamCell{:});
    end
elseif l_daprime==5
    if l_a==1 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_gridvals(:,1), ParamCell{:});
    elseif l_daprime==2 && l_a==1 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_gridvals(:,1), z_gridvals(1,:,1), ParamCell{:});
    elseif l_daprime==2 && l_a==1 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_gridvals(:,1), z_gridvals(1,:,1),z_gridvals(1,:,2), ParamCell{:});
    elseif l_daprime==2 && l_a==1 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_gridvals(:,1), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3), ParamCell{:});
    elseif l_daprime==2 && l_a==1 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_gridvals(:,1), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3),z_gridvals(1,:,4), ParamCell{:});
    elseif l_a==2 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_gridvals(:,1),a_gridvals(:,2), ParamCell{:});
    elseif l_daprime==5 && l_a==2 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,:,1), ParamCell{:});
    elseif l_daprime==5 && l_a==2 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,:,1),z_gridvals(1,:,2), ParamCell{:});
    elseif l_daprime==5 && l_a==2 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3), ParamCell{:});
    elseif l_daprime==5 && l_a==2 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_gridvals(:,1),a_gridvals(:,2), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3),z_gridvals(1,:,4), ParamCell{:});
    elseif l_daprime==5 && l_a==3 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), ParamCell{:});
    elseif l_daprime==5 && l_a==3 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,:,1), ParamCell{:});
    elseif l_daprime==5 && l_a==3 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,:,1),z_gridvals(1,:,2), ParamCell{:});
    elseif l_daprime==5 && l_a==3 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3), ParamCell{:});
    elseif l_daprime==5 && l_a==3 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3),z_gridvals(1,:,4), ParamCell{:});
    elseif l_daprime==5 && l_a==4 && l_z==0
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), ParamCell{:});
    elseif l_daprime==5 && l_a==4 && l_z==1
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,:,1), ParamCell{:});
    elseif l_daprime==5 && l_a==4 && l_z==2
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,:,1),z_gridvals(1,:,2), ParamCell{:});
    elseif l_daprime==5 && l_a==4 && l_z==3
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3), ParamCell{:});
    elseif l_daprime==5 && l_a==4 && l_z==4
        Values=arrayfun(FnToEvaluate, daprime1vals,daprime2vals,daprime3vals,daprime4vals,daprime5vals, a_gridvals(:,1),a_gridvals(:,2),a_gridvals(:,3),a_gridvals(:,4), z_gridvals(1,:,1),z_gridvals(1,:,2),z_gridvals(1,:,3),z_gridvals(1,:,4), ParamCell{:});
    end
end

% Note: No need to reshape the output as it is already the appropriate size
% if N_z==0
%     Values=reshape(Values,[N_a,1]);
% else
%     Values=reshape(Values,[N_a,N_z]);
% end



end