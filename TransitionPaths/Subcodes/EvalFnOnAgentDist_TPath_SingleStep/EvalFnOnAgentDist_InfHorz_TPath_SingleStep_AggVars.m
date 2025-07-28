function AggVars=EvalFnOnAgentDist_InfHorz_TPath_SingleStep_AggVars(AgentDist, PolicyValues, FnsToEvaluateCell, Parameters, FnsToEvaluateParamNames, n_a, n_z, a_gridvals, z_gridvals)
% Evaluates the aggregate value (weighted sum/integral) for each element of FnsToEvaluate
% For internal use only

N_a=prod(n_a);
N_z=prod(n_z);

l_daprime=size(PolicyValues,1);

%%
AggVars=zeros(length(FnsToEvaluateCell),1,'gpuArray');

PolicyValuesPermute=permute(reshape(PolicyValues,[size(PolicyValues,1),N_a,N_z]),[2,3,1]); %[N_a,N_z,l_d+l_a]

for ff=1:length(FnsToEvaluateCell)
    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names);
    Values=EvalFnOnAgentDist_Grid(FnsToEvaluateCell{ff}, FnToEvaluateParamsCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);
    Values=reshape(Values,[N_a*N_z,1]);
    % When evaluating value function (which may sometimes give -Inf values) on StationaryDistVec (which at those points will be 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
    temp=Values.*AgentDist;
    AggVars(ff)=sum(temp(~isnan(temp)));
end


end
