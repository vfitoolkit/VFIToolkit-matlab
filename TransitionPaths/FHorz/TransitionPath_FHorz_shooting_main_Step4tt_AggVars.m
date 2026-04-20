function AggVars=TransitionPath_FHorz_shooting_main_Step4tt_AggVars(AgentDist,AgeWeights,PolicyValuesPath,tt,FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_aprime,l_a,l_z,l_e,N_d,N_a,N_z,N_e,a_gridvals,ze_gridvals_J_fastOLG,transpathoptions)
% Maybe should just take PolicyValuesPath_tt as input instead of PolicyValuesPath

if N_z>0 && N_e>0 && transpathoptions.zepathtrivial==0
    ze_gridvals_J_fastOLG=transpathoptions.ze_gridvals_J_fastOLG(:,:,:,tt);
elseif N_z>0 && transpathoptions.zpathtrivial==0
    ze_gridvals_J_fastOLG=transpathoptions.z_gridvals_J_fastOLG(:,:,:,tt);
elseif N_e>0 && transpathoptions.epathtrivial==0
    ze_gridvals_J_fastOLG=transpathoptions.z_gridvals_J_fastOLG(:,:,:,tt);
end

if N_z==0 && N_e==0
    if N_d==0
        AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG_noz(AgentDist.*AgeWeights, [], PolicyValuesPath(:,:,:,tt), FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,0,l_aprime,l_a,N_a,a_gridvals,1);
    else
        AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG_noz(AgentDist.*AgeWeights,PolicyValuesPath(:,:,1:l_d,tt), PolicyValuesPath(:,:,l_d+1:end,tt), FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_aprime,l_a,N_a,a_gridvals,1);
    end
elseif N_z>0 && N_e==0
    if N_d==0
        AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist.*AgeWeights, [], PolicyValuesPath(:,:,:,:,tt), FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,0,l_aprime,l_a,l_z,N_a,N_z,a_gridvals,ze_gridvals_J_fastOLG,1);
    else
        AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist.*AgeWeights, PolicyValuesPath(:,:,:,1:l_d,tt), PolicyValuesPath(:,:,:,l_d+1:end,tt), FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_aprime,l_a,l_z,N_a,N_z,a_gridvals,ze_gridvals_J_fastOLG,1);
    end
elseif N_z==0 && N_e>0
    if N_d==0
        AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist.*AgeWeights,[], PolicyValuesPath(:,:,:,:,tt), FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,0,l_aprime,l_a,l_e,N_a,N_e,a_gridvals,ze_gridvals_J_fastOLG,1);
    else
        AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist.*AgeWeights, PolicyValuesPath(:,:,:,1:l_d,tt), PolicyValuesPath(:,:,:,l_d+1:end,tt), FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_aprime,l_a,l_e,N_a,N_e,a_gridvals,ze_gridvals_J_fastOLG,1);
    end
elseif N_z>0 && N_e>0
    if N_d==0
        AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist.*AgeWeights, [], PolicyValuesPath(:,:,:,:,tt), FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,0,l_aprime,l_a,l_z+l_e,N_a,N_z*N_e,a_gridvals,ze_gridvals_J_fastOLG,1);
    else
        AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist.*AgeWeights, PolicyValuesPath(:,:,:,1:l_d,tt), PolicyValuesPath(:,:,:,l_d+1:end,tt), FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_aprime,l_a,l_z+l_e,N_a,N_ze,a_gridvals,ze_gridvals_J_fastOLG,1);
    end
end









end