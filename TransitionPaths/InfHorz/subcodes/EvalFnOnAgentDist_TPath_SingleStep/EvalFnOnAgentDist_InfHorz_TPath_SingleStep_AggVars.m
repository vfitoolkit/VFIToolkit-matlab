function AggVars=EvalFnOnAgentDist_InfHorz_TPath_SingleStep_AggVars(AgentDist, PolicyValuesPermute, FnsToEvaluateCell, Parameters, FnsToEvaluateParamNames, FnsToEvaluateNames, n_a, n_z, a_gridvals, z_gridvals, outputastruct)
% Evaluates the aggregate value (weighted sum/integral) for each element of FnsToEvaluate
% For internal use only

N_a=prod(n_a);
N_z=prod(n_z);

l_daprime=size(PolicyValuesPermute,3); % PolicyValuesPermute is [N_a,N_z,l_d+l_aprime]

%%
if outputastruct==0
    AggVars=zeros(length(FnsToEvaluateCell),1,'gpuArray');
else
    AggVars=struct();
end

for ff=1:length(FnsToEvaluateNames)
    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names);
    Values=EvalFnOnAgentDist_Grid(FnsToEvaluateCell{ff}, FnToEvaluateParamsCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);
    Values=reshape(Values,[N_a*N_z,1]);
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
