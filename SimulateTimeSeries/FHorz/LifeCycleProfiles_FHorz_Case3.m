function AgeConditionalStats=LifeCycleProfiles_FHorz_Case3(StationaryDist,Policy,FnsToEvaluate,FnsToEvaluateParamNames,Parameters,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions, AgeDependentGridParamNames)
% Because Case3 policy is same as for Case2 there is no difference in
% EvalFnOnAgentDist between Case2 and Case3. The Case3 command is therefore
% just a front to redirect to Case2. This function exists solely so that
% the user does not have to understand VFI Toolkit so deeply that they know
% Case2 and Case3, while very different for value fn and agent dist, are
% just the same to EvalFnOnAgentDist.

if exist('AgeDependentGridParamNames','var')
    AgeConditionalStats=LifeCycleProfiles_FHorz_Case2(StationaryDist,Policy,FnsToEvaluate,FnsToEvaluateParamNames,Parameters,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions, AgeDependentGridParamNames);
else
    AgeConditionalStats=LifeCycleProfiles_FHorz_Case2(StationaryDist,Policy,FnsToEvaluate,FnsToEvaluateParamNames,Parameters,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);
end

end

