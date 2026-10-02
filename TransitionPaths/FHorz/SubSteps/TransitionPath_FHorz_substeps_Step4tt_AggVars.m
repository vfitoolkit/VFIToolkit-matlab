function AggVars=TransitionPath_FHorz_substeps_Step4tt_AggVars(AgentDist,AgeWeights,PolicyValuesPath_tt,tt,FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_aprime,l_a,l_semiz,l_z,l_e,N_d,N_a,N_semiz,N_z,N_e,a_gridvals,semizze_gridvals_J_fastOLG,transpathoptions,distfastOLG)
% The FnsToEvaluate are evaluated over (a,j,z) [z standing for whatever exogenous states there are,
% semiz then z then e], and aggregated as sum(Values(:).*AgentDist(:)), so the age-weighted dist must be
% in that same linear order. It is under simoptions.fastOLG=1 (the default), where the dist is (a,j,z) with
% e trailing. Under simoptions.fastOLG=0 it is (a,z,j) [N_a*N_z,N_j], so distfastOLG=0 puts it into
% (a,j,z) first. distfastOLG is optional, and defaults to 1.
if ~exist('distfastOLG','var')
    distfastOLG=1;
end
if distfastOLG==0
    AgentDist=AgentDist.*AgeWeights;
    AgentDist=reshape(permute(reshape(AgentDist,[N_a,numel(AgentDist)/(N_a*N_j),N_j]),[1,3,2]),[numel(AgentDist),1]); % (a,z,j) -> (a,j,z)
    AgeWeights=1; % already applied
end
% Maybe should just take PolicyValuesPath_tt as input instead of PolicyValuesPath

if N_semiz==0
    if N_z>0 && N_e>0 && transpathoptions.zepathtrivial==0
        semizze_gridvals_J_fastOLG=transpathoptions.ze_gridvals_J_fastOLG(:,:,:,tt);
    elseif N_z>0 && transpathoptions.zpathtrivial==0
        semizze_gridvals_J_fastOLG=transpathoptions.z_gridvals_J_fastOLG(:,:,:,tt);
    elseif N_e>0 && transpathoptions.epathtrivial==0
        semizze_gridvals_J_fastOLG=transpathoptions.e_gridvals_J_fastOLG(:,:,:,tt);
    end
else
    if N_z>0 && N_e>0 && transpathoptions.zepathtrivial==0
        semizze_gridvals_J_fastOLG=transpathoptions.semizze_gridvals_J_fastOLG(:,:,:,tt);
    elseif N_z>0 && transpathoptions.zpathtrivial==0
        semizze_gridvals_J_fastOLG=transpathoptions.semizz_gridvals_J_fastOLG(:,:,:,tt);
    elseif N_e>0 && transpathoptions.epathtrivial==0
        semizze_gridvals_J_fastOLG=transpathoptions.semize_gridvals_J_fastOLG(:,:,:,tt);
    end
end

if N_semiz==0
    if N_z==0 && N_e==0
        if N_d==0
            AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG_noz(AgentDist.*AgeWeights, [], PolicyValuesPath_tt, FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,0,l_aprime,l_a,N_a,a_gridvals,1);
        else
            AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG_noz(AgentDist.*AgeWeights,PolicyValuesPath_tt(:,:,1:l_d), PolicyValuesPath_tt(:,:,l_d+1:end), FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_aprime,l_a,N_a,a_gridvals,1);
        end
    elseif N_z>0 && N_e==0
        if N_d==0
            AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist.*AgeWeights, [], PolicyValuesPath_tt, FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,0,l_aprime,l_a,l_z,N_a,N_z,a_gridvals,semizze_gridvals_J_fastOLG,1);
        else
            AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist.*AgeWeights, PolicyValuesPath_tt(:,:,:,1:l_d), PolicyValuesPath_tt(:,:,:,l_d+1:end), FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_aprime,l_a,l_z,N_a,N_z,a_gridvals,semizze_gridvals_J_fastOLG,1);
        end
    elseif N_z==0 && N_e>0
        if N_d==0
            AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist.*AgeWeights,[], PolicyValuesPath_tt, FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,0,l_aprime,l_a,l_e,N_a,N_e,a_gridvals,semizze_gridvals_J_fastOLG,1);
        else
            AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist.*AgeWeights, PolicyValuesPath_tt(:,:,:,1:l_d), PolicyValuesPath_tt(:,:,:,l_d+1:end), FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_aprime,l_a,l_e,N_a,N_e,a_gridvals,semizze_gridvals_J_fastOLG,1);
        end
    elseif N_z>0 && N_e>0
        if N_d==0
            AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist.*AgeWeights, [], PolicyValuesPath_tt, FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,0,l_aprime,l_a,l_z+l_e,N_a,N_z*N_e,a_gridvals,semizze_gridvals_J_fastOLG,1);
        else
            AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist.*AgeWeights, PolicyValuesPath_tt(:,:,:,1:l_d), PolicyValuesPath_tt(:,:,:,l_d+1:end), FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_aprime,l_a,l_z+l_e,N_a,N_z*N_e,a_gridvals,semizze_gridvals_J_fastOLG,1);
        end
    end
else
    if N_z==0 && N_e==0
        if N_d==0
            AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist.*AgeWeights, [], PolicyValuesPath_tt, FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,0,l_aprime,l_a,l_semiz,N_a,N_semiz,a_gridvals,semizze_gridvals_J_fastOLG,1);
        else
            AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist.*AgeWeights,PolicyValuesPath_tt(:,:,:,1:l_d), PolicyValuesPath_tt(:,:,:,l_d+1:end), FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_aprime,l_a,l_semiz,N_a,N_semiz,a_gridvals,semizze_gridvals_J_fastOLG,1);
        end
    elseif N_z>0 && N_e==0
        if N_d==0
            AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist.*AgeWeights, [], PolicyValuesPath_tt, FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,0,l_aprime,l_a,l_semiz+l_z,N_a,N_semiz*N_z,a_gridvals,semizze_gridvals_J_fastOLG,1);
        else
            AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist.*AgeWeights, PolicyValuesPath_tt(:,:,:,1:l_d), PolicyValuesPath_tt(:,:,:,l_d+1:end), FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_aprime,l_a,l_semiz+l_z,N_a,N_semiz*N_z,a_gridvals,semizze_gridvals_J_fastOLG,1);
        end
    elseif N_z==0 && N_e>0
        if N_d==0
            AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist.*AgeWeights,[], PolicyValuesPath_tt, FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,0,l_aprime,l_a,l_semiz+l_e,N_a,N_semiz*N_e,a_gridvals,semizze_gridvals_J_fastOLG,1);
        else
            AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist.*AgeWeights, PolicyValuesPath_tt(:,:,:,1:l_d), PolicyValuesPath_tt(:,:,:,l_d+1:end), FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_aprime,l_a,l_semiz+l_e,N_a,N_semiz*N_e,a_gridvals,semizze_gridvals_J_fastOLG,1);
        end
    elseif N_z>0 && N_e>0
        if N_d==0
            AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist.*AgeWeights, [], PolicyValuesPath_tt, FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,0,l_aprime,l_a,l_semiz+l_z+l_e,N_a,N_semiz*N_z*N_e,a_gridvals,semizze_gridvals_J_fastOLG,1);
        else
            AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist.*AgeWeights, PolicyValuesPath_tt(:,:,:,1:l_d), PolicyValuesPath_tt(:,:,:,l_d+1:end), FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_aprime,l_a,l_semiz+l_z+l_e,N_a,N_semiz*N_z*N_e,a_gridvals,semizze_gridvals_J_fastOLG,1);
        end
    end
end







end
