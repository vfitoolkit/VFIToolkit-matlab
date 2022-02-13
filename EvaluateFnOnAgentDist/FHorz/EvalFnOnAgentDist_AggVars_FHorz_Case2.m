function AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case2(StationaryDist, PolicyIndexes, FnsToEvaluateFn, Parameters,FnsToEvaluateParamNames, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, simoptions, AgeDependentGridParamNames) %pi_z,p_val
% Evaluates the aggregate value (weighted sum/integral) for each element of FnsToEvaluateFn
% options and AgeDependentGridParamNames is only needed when you are using Age Dependent Grids, otherwise this is not a required input.

if isa(StationaryDist,'struct')
    % Using Age Dependent Grids so send there
    % Note that in this case: d_grid is d_gridfn, a_grid is a_gridfn,
    % z_grid is z_gridfn. Parallel is options. AgeDependentGridParamNames is also needed. 
    AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case2_AgeDepGrids(StationaryDist, PolicyIndexes, FnsToEvaluateFn, Parameters,FnsToEvaluateParamNames, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, simoptions, AgeDependentGridParamNames);
    return
end

% l_d=length(n_d);
l_a=length(n_a);
l_z=length(n_z);
N_a=prod(n_a);
N_z=prod(n_z);

%% This implementation is slightly inefficient when shocks are not age dependent, but speed loss is fairly trivial
if exist('simoptions','var')
    if isfield(simoptions,'ExogShockFn') % If using ExogShockFn then figure out the parameter names
        simoptions.ExogShockFnParamNames=getAnonymousFnInputNames(simoptions.ExogShockFn);
    end
end
eval('fieldexists_ExogShockFn=1;simoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;simoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')
eval('fieldexists_pi_z_J=1;simoptions.pi_z_J;','fieldexists_pi_z_J=0;')

if fieldexists_pi_z_J==1
    z_grid_J=simoptions.z_grid_J;
elseif fieldexists_ExogShockFn==1
    z_grid_J=zeros(sum(n_z),N_j);
    for jj=1:N_j
        if fieldexists_ExogShockFnParamNames==1
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
            ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
            for ii=1:length(ExogShockFnParamsVec)
                ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
            end
            [z_grid,~]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
        else
            [z_grid,~]=simoptions.ExogShockFn(jj);
        end
        z_grid_J(:,jj)=z_grid;
    end
else
    z_grid_J=repmat(z_grid,1,N_j);
end
if Parallel==2
    z_grid_J=gpuArray(z_grid_J);
end

%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
        if length(temp)>(l_d+l_a+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(AggVarNames{ff});
    end    
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end

%%
if isa(StationaryDist,'gpuArray')% Parallel==2
    AggVars=zeros(length(FnsToEvaluateFn),1,'gpuArray');
    
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z,N_j]);
    
    PolicyValues=PolicyInd2Val_FHorz_Case2(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,2);
    permuteindexes=[1+(1:1:(l_a+l_z)),1,1+l_a+l_z+1];    
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]

    for i=1:length(FnsToEvaluateFn)
        Values=nan(N_a*N_z,N_j,'gpuArray');
        for jj=1:N_j
            z_grid=z_grid_J(:,jj);
            
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames) %|| strcmp(SSvalueParamNames(i).Names(1),'')) % check for 'SSvalueParamNames={} or SSvalueParamNames={''}'
                FnToEvaluateParamsVec=[];
            else
                FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names,jj);
            end
            Values(:,jj)=reshape(EvalFnOnAgentDist_Grid_Case2(FnsToEvaluateFn{i}, FnToEvaluateParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,2),[N_a*N_z,1]);
        end
%         Values=reshape(Values,[N_a*N_z,N_j]);
        AggVars(i)=sum(sum(Values.*StationaryDistVec));
    end
    
else
    AggVars=zeros(length(FnsToEvaluateFn),1);
%     d_val=zeros(l_d,1);
%     a_val=zeros(l_a,1);
%     z_val=zeros(l_z,1);
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z*N_j,1]);
    
    a_gridvals=CreateGridvals(n_a,a_grid,1);
    z_gridvals=CreateGridvals(n_z,z_grid,1);
    dPolicy_gridvals=zeros(N_a*N_z,N_j);
    for jj=1:N_j
        dPolicy_gridvals(:,jj)=CreateGridvals_Policy(PolicyIndexes(:,:,jj),n_d,[],n_a,n_z,d_grid,[],2,1);
    end
    
    for i=1:length(FnsToEvaluateFn)
        Values=zeros(N_a,N_z,N_j);
        for jj=1:N_j
            z_grid=z_grid_J(:,jj);
            z_gridvals=CreateGridvals(n_z,z_grid,2);
            
            for a_c=1:N_a
                a_val=a_gridvals(a_c,:);
                for z_c=1:N_z
                    z_val=z_gridvals(z_c,:);
                    az_c=sub2ind_homemade([N_a,N_z],[a_c,z_c]);
                    d_val=dPolicy_gridvals(az_c,jj);
                    % Includes check for cases in which no parameters are actually required
                    if isempty(FnsToEvaluateParamNames(i).Names)
                        tempv=[d_val,a_val,z_val];
                        tempcell=cell(1,length(tempv));
                        for temp_c=1:length(tempv)
                            tempcell{temp_c}=tempv(temp_c);
                        end
                    else
                        FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names,jj);
                        tempv=[d_val,a_val,z_val,FnToEvaluateParamsVec];
                        tempcell=cell(1,length(tempv));
                        for temp_c=1:length(tempv)
                            tempcell{temp_c}=tempv(temp_c);
                        end
                    end
                    Values(a_c,z_c,jj)=FnsToEvaluateFn{i}(tempcell{:});
                end
            end
        end
        Values=reshape(Values,[N_a*N_z*N_j,1]);
        
        AggVars(i)=sum(Values.*StationaryDistVec);
    end

end


end

