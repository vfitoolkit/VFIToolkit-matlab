function AggVars=TransitionPath_FHorz_substeps_Step4tt_AggVars(AgentDist,AgeWeights,PolicyValuesPath_tt,tt,FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_aprime,l_a,l_z,l_e,N_d,N_a,N_z,N_e,a_gridvals,ze_gridvals_J_fastOLG,transpathoptions)
% Maybe should just take PolicyValuesPath_tt as input instead of PolicyValuesPath

if N_z>0 && N_e>0 && transpathoptions.zepathtrivial==0
    ze_gridvals_J_fastOLG=transpathoptions.ze_gridvals_J_fastOLG(:,:,:,tt);
elseif N_z>0 && transpathoptions.zpathtrivial==0
    ze_gridvals_J_fastOLG=transpathoptions.z_gridvals_J_fastOLG(:,:,:,tt);
elseif N_e>0 && transpathoptions.epathtrivial==0
    ze_gridvals_J_fastOLG=transpathoptions.e_gridvals_J_fastOLG(:,:,:,tt);
end

if transpathoptions.fastOLG==0
    % What's the right thing here?
    WeightedAgentDist=AgentDist.*AgeWeights';
    if N_e==0
        if N_z==0
            WeightedAgentDist=reshape(WeightedAgentDist,[N_a*N_j,1]);
        else
            WeightedAgentDist=reshape(WeightedAgentDist,[N_a*N_j*N_z,1]);
        end
    else
        if N_z==0
            WeightedAgentDist=reshape(WeightedAgentDist,[N_a*N_j,N_e]);
        else
            WeightedAgentDist=reshape(WeightedAgentDist,[N_a*N_j*N_z,N_e]);
        end
    end
else
    WeightedAgentDist=AgentDist.*AgeWeights;
end

if N_z==0 && N_e==0
    if N_d==0
        AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG_noz(WeightedAgentDist, [], PolicyValuesPath_tt, FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,0,l_aprime,l_a,N_a,a_gridvals,1);
    else
        AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG_noz(WeightedAgentDist,PolicyValuesPath_tt(:,:,1:l_d), PolicyValuesPath_tt(:,:,l_d+1:end), FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_aprime,l_a,N_a,a_gridvals,1);
    end
elseif N_z>0 && N_e==0
    if N_d==0
        AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(WeightedAgentDist, [], PolicyValuesPath_tt, FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,0,l_aprime,l_a,l_z,N_a,N_z,a_gridvals,ze_gridvals_J_fastOLG,1);
    else
        AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(WeightedAgentDist, PolicyValuesPath_tt(:,:,:,1:l_d), PolicyValuesPath_tt(:,:,:,l_d+1:end), FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_aprime,l_a,l_z,N_a,N_z,a_gridvals,ze_gridvals_J_fastOLG,1);
    end
elseif N_z==0 && N_e>0
    if N_d==0
        AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(WeightedAgentDist,[], PolicyValuesPath_tt, FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,0,l_aprime,l_a,l_e,N_a,N_e,a_gridvals,ze_gridvals_J_fastOLG,1);
    else
        AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(WeightedAgentDist, PolicyValuesPath_tt(:,:,:,1:l_d), PolicyValuesPath_tt(:,:,:,l_d+1:end), FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_aprime,l_a,l_e,N_a,N_e,a_gridvals,ze_gridvals_J_fastOLG,1);
    end
elseif N_z>0 && N_e>0
    if N_d==0
        AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(WeightedAgentDist, [], PolicyValuesPath_tt, FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,0,l_aprime,l_a,l_z+l_e,N_a,N_z*N_e,a_gridvals,ze_gridvals_J_fastOLG,1);
    else
        N_ze=N_z*N_e;
        AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(WeightedAgentDist, PolicyValuesPath_tt(:,:,:,1:l_d), PolicyValuesPath_tt(:,:,:,l_d+1:end), FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_aprime,l_a,l_z+l_e,N_a,N_ze,a_gridvals,ze_gridvals_J_fastOLG,1);
    end
end









end
