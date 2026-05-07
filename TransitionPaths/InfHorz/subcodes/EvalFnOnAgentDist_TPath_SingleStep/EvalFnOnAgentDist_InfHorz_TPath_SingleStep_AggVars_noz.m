function AggVars=EvalFnOnAgentDist_InfHorz_TPath_SingleStep_AggVars_noz(AgentDist, PolicyValuesPermute, FnsToEvaluateCell, Parameters, FnsToEvaluateParamNames, FnsToEvaluateNames, n_a, a_gridvals, outputastruct)
% Evaluates the aggregate value (weighted sum/integral) for each element of FnsToEvaluate
% For internal use only

N_a=prod(n_a);

l_daprime=size(PolicyValuesPermute,2); % PolicyValuesPermute is [N_a,l_d+l_aprime]

%%
if outputastruct==0
    AggVars=zeros(length(FnsToEvaluateCell),1,'gpuArray');
else
    AggVars=struct();
end

for ff=1:length(FnsToEvaluateNames)
    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names);
    Values=EvalFnOnAgentDist_Grid(FnsToEvaluateCell{ff}, FnToEvaluateParamsCell,PolicyValuesPermute,l_daprime,n_a,0,a_gridvals,[]);
    Values=reshape(Values,[N_a,1]);
    % When evaluating value function (which may sometimes give -Inf values) on StationaryDistVec (which at those points will be 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
    temp=Values.*AgentDist;
    val=sum(temp(~isnan(temp)));
    
    if outputastruct==0
        AggVars(ff)=val;
    else % if outputastruct==1
        AggVars.(FnsToEvaluateNames{ff}).Mean=val;
    end

end


end
