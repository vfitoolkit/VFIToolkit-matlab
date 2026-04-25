function AggVars=TransitionPath_InfHorz_substeps_Step4tt_AggVars(AgentDist,PolicyValuesPath_tt,tt,FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,n_a,n_z,n_e,N_z,N_e,a_gridvals,ze_gridvals,transpathoptions)
% Maybe should just take PolicyValuesPath_tt as input instead of PolicyValuesPath

if N_z>0 && N_e>0 && transpathoptions.zepathtrivial==0
    ze_gridvals=transpathoptions.ze_gridvals_T(:,:,tt);
elseif N_z>0 && transpathoptions.zpathtrivial==0
    ze_gridvals=transpathoptions.z_gridvals_T(:,:,tt);
elseif N_e>0 && transpathoptions.epathtrivial==0
    ze_gridvals=transpathoptions.e_gridvals_T(:,:,tt);
end

outputastruct=1;

if N_z==0 && N_e==0
    AggVars=EvalFnOnAgentDist_InfHorz_TPath_SingleStep_AggVars_noz(AgentDist, PolicyValuesPath_tt, FnsToEvaluateCell, Parameters, FnsToEvaluateParamNames, AggVarNames, n_a, a_gridvals, outputastruct);
elseif N_z>0 && N_e==0
    AggVars=EvalFnOnAgentDist_InfHorz_TPath_SingleStep_AggVars(AgentDist, PolicyValuesPath_tt, FnsToEvaluateCell, Parameters, FnsToEvaluateParamNames, AggVarNames, n_a, n_z, a_gridvals, ze_gridvals, outputastruct);
elseif N_z==0 && N_e>0
    AggVars=EvalFnOnAgentDist_InfHorz_TPath_SingleStep_AggVars(AgentDist, PolicyValuesPath_tt, FnsToEvaluateCell, Parameters, FnsToEvaluateParamNames, AggVarNames, n_a, n_e, a_gridvals, ze_gridvals, outputastruct);
elseif N_z>0 && N_e>0
    AggVars=EvalFnOnAgentDist_InfHorz_TPath_SingleStep_AggVars(AgentDist, PolicyValuesPath_tt, FnsToEvaluateCell, Parameters, FnsToEvaluateParamNames, AggVarNames, n_a, [n_z,n_e], a_gridvals, ze_gridvals, outputastruct);
end








end
