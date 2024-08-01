function ValuesOnGrid=EvalFnOnAgentDist_ValuesOnGrid_FHorz_Case1(PolicyIndexes, FnsToEvaluate,Parameters,FnsToEvaluateParamNames,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions)

if ~exist('simoptions','var')
    simoptions.lowmemory=0;
else
    if ~isfield(simoptions,'lowmemory')
        simoptions.lowmemory=0;
    end
end


%%
l_a=length(n_a);
N_a=prod(n_a);

a_gridvals=CreateGridvals(n_a,a_grid,1);

%% Exogenous shock grids
% Create the combination of (semiz,z,e) as all three are the same for FnsToEvaluate 
[n_z,z_gridvals_J,N_z,l_z,simoptions]=CreateGridvals_FnsToEvaluate_FHorz(n_z,z_grid,N_j,simoptions);


%% Implement new way of handling FnsToEvaluate
% Figure out l_daprime from Policy
l_daprime=size(PolicyIndexes,1);

% Note: l_z includes e and semiz (when appropriate)
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
        if length(temp)>(l_daprime+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_daprime+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(AggVarNames{ff});
    end    
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end
if isfield(simoptions,'outputasstructure')
    if simoptions.outputasstructure==1
        FnsToEvaluateStruct=1;
        AggVarNames=simoptions.AggVarNames;
    elseif simoptions.outputasstructure==0
        FnsToEvaluateStruct=0;
    end
end

%%
if N_z==0
    PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,0,N_j,d_grid,a_grid,simoptions,1);
    PolicyValuesPermute=permute(PolicyValues,[2,3,1]); % (N_a,N_j,l_daprime)

    for ff=1:length(FnsToEvaluate)
        CellOverAgeOfParamValues=CreateCellOverAgeFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,N_j,2); % j in 2nd dimension: (a,j,l_d+l_a), so we want j to be after N_a
        ValuesOnGrid.(AggVarNames{ff})=EvalFnOnAgentDist_Grid_J(FnsToEvaluate{ff},CellOverAgeOfParamValues,PolicyValuesPermute,l_daprime,n_a,0,a_gridvals,[]);
    end
else % N_z
    PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,1);
    PolicyValuesPermute=permute(PolicyValues,[2,3,4,1]); % (N_a,N_z,N_j,l_daprime)

    for ff=1:length(FnsToEvaluate)
        CellOverAgeOfParamValues=CreateCellOverAgeFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,N_j,3); % j in 3nd dimension: (a,z,j,l_d+l_a), so we want j to be after N_a and N_z
        ValuesOnGrid.(AggVarNames{ff})=EvalFnOnAgentDist_Grid_J(FnsToEvaluate{ff},CellOverAgeOfParamValues,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals_J);
    end
end


%%
if FnsToEvaluateStruct==0
    % Output as matrix instead of structure
    ValuesOnGrid2=ValuesOnGrid;
    clear ValuesOnGrid
    if N_z==0
        ValuesOnGrid=zeros(N_a,N_j,length(FnsToEvaluate),'gpuArray');
        for ff=1:length(AggVarNames)
            ValuesOnGrid(:,:,ff)=reshape(ValuesOnGrid2.(AggVarNames{ff}),[N_a,N_j]);
        end
    else
        ValuesOnGrid=zeros(N_a*N_z,N_j,length(FnsToEvaluate),'gpuArray');
        for ff=1:length(AggVarNames)
            ValuesOnGrid(:,:,ff)=reshape(ValuesOnGrid2.(AggVarNames{ff}),[N_a*N_z,N_j]);
        end
    end
else
    % Reshape to be appropriate size
    if N_z==0
        for ff=1:length(AggVarNames)
            ValuesOnGrid.(AggVarNames{ff})=reshape(ValuesOnGrid.(AggVarNames{ff}),[n_a,N_j]);
        end
    else
        for ff=1:length(AggVarNames)
            ValuesOnGrid.(AggVarNames{ff})=reshape(ValuesOnGrid.(AggVarNames{ff}),[n_a,n_z,N_j]);
        end
    end
end




end
