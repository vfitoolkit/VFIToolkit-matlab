function ValuesOnGrid=EvalFnOnAgentDist_ValuesOnGrid_FHorz_Case2_noz(PolicyIndexes, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, N_j, d_grid, a_grid, Parallel,simoptions)

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
N_a=prod(n_a);


%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
        if length(temp)>(l_d+l_a)
            FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(AggVarNames{ff});
    end    
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end
if isfield(simoptions,'keepoutputasmatrix')
    if simoptions.keepoutputasmatrix==1
        FnsToEvaluateStruct=0;
    elseif simoptions.keepoutputasmatrix==2
        FnsToEvaluateStruct=2;
    end
end


%%
ValuesOnGrid=zeros(N_a,N_j,length(FnsToEvaluate),'gpuArray');

PolicyValues=PolicyInd2Val_FHorz_Case2_noz(PolicyIndexes,n_d,n_a,N_j,d_grid);
permuteindexes=[1+(1:1:l_a),1,1+l_a+1];
PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,l_d,N_j]

PolicyValuesPermuteVec=reshape(PolicyValuesPermute,[N_a*l_d,N_j]);
for ff=1:length(FnsToEvaluate)
    Values=nan(N_a,N_j,'gpuArray');
    for jj=1:N_j
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(ff).Names) % || strcmp(FnsToEvaluateParamNames(1),'')) % check for 'FnsToEvaluateParamNames={}'
            FnToEvaluateParamsVec=[];
        else
            FnToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj));
        end
        Values(:,jj)=reshape(EvalFnOnAgentDist_Grid_Case2_noz(FnsToEvaluate{ff}, FnToEvaluateParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,l_d]),n_d,n_a,a_grid,Parallel),[N_a,1]);
    end
    ValuesOnGrid(:,:,ff)=Values;
end

if FnsToEvaluateStruct==1
    ValuesOnGrid2=ValuesOnGrid;
    clear ValuesOnGrid
    ValuesOnGrid=struct();
    for ff=1:length(AggVarNames)
        ValuesOnGrid.(AggVarNames{ff})=reshape(ValuesOnGrid2(:,:,ff),[n_a,N_j]);
        % Change the ordering and size so that ProbDensityFns has same kind of shape as StationaryDist
    end
elseif FnsToEvaluateStruct==0
    % Change the ordering and size so that ProbDensityFns has same kind of
    % shape as StationaryDist, except first dimension indexes the
    % 'FnsToEvaluate'.
    ValuesOnGrid=permute(ValuesOnGrid,[3,1,2]);
    ValuesOnGrid=reshape(ValuesOnGrid,[length(FnsToEvaluate),n_a,N_j]);
elseif FnsToEvaluateStruct==2
    % Just output ValuesOnGrid as is
end

end
