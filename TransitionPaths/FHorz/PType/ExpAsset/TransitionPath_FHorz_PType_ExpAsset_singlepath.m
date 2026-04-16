function AggVarsPath=TransitionPath_FHorz_PType_ExpAsset_singlepath(PricePathOld, ParamPath, PricePathNames,ParamPathNames,T,V_final,AgentDist_initial,jequalOneDist_T,AgeWeights_T, ...
    l_d,n_d1,n_d2,n_a1,n_a2,N_z,n_z,N_e,n_e,N_j,d2_grid,a1_gridvals,a2_grid,d_gridvals,aprime_gridvals,a_gridvals,z_gridvals_J, pi_z_J,pi_z_J_sim,e_gridvals_J,pi_e_J,pi_e_J_sim,ze_gridvals_J_fastOLG, ...
    ReturnFn, aprimeFn, FnsToEvaluateCell, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, FnsToEvaluateParamNames, AggVarNames, ...
    PricePathSizeVec, ParamPathSizeVec, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, II1orII, II2, exceptlastj,exceptfirstj,justfirstj, ...
    transpathoptions, vfoptions, simoptions)
% When doing shooting alogrithm on TPath FHorz PType, this is for a given ptype, and does the steps of back-iterate to get policy, then forward to get agent dist and agg vars.
% The only output is the agg vars path.

% AggVarsPath=zeros(length(FnsToEvaluate),T-1,'gpuArray');
if transpathoptions.fastOLG==0
    if N_z==0
        if N_e==0
            AggVarsPath=TransitionPath_FHorz_PType_ExpAsset_singlepath_noz_raw(PricePathOld, ParamPath, PricePathNames,ParamPathNames,T,V_final,AgentDist_initial,jequalOneDist_T,AgeWeights_T,l_d,n_d1,n_d2,n_a1,n_a2,N_z,n_z,N_j,d2_grid,a1_gridvals,a2_grid,d_gridvals,a_gridvals,z_gridvals_J, pi_z_J,pi_z_J_sim,ze_gridvals_J_fastOLG,ReturnFn, aprimeFn, FnsToEvaluateCell, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, FnsToEvaluateParamNames, AggVarNames, PricePathSizeVec, ParamPathSizeVec, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, II1orII, II2, exceptlastj,exceptfirstj,justfirstj, transpathoptions, vfoptions, simoptions);
        else
            AggVarsPath=TransitionPath_FHorz_PType_ExpAsset_singlepath_noz_e_raw(PricePathOld, ParamPath, PricePathNames,ParamPathNames,T,V_final,AgentDist_initial,jequalOneDist_T,AgeWeights_T,l_d,n_d1,n_d2,n_a1,n_a2,N_z,n_z,N_e,n_e,N_j,d2_grid,a1_gridvals,a2_grid,d_gridvals,a_gridvals,z_gridvals_J, pi_z_J,pi_z_J_sim,e_gridvals_J,pi_e_J,pi_e_J_sim,ze_gridvals_J_fastOLG,ReturnFn, aprimeFn, FnsToEvaluateCell, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, FnsToEvaluateParamNames, AggVarNames, PricePathSizeVec, ParamPathSizeVec, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, II1orII, II2, exceptlastj,exceptfirstj,justfirstj, transpathoptions, vfoptions, simoptions);
        end
    else
        if N_e==0
            % Principal test case...
            AggVarsPath=TransitionPath_FHorz_PType_ExpAsset_singlepath_raw(PricePathOld, ParamPath, PricePathNames,ParamPathNames,T,V_final,AgentDist_initial,jequalOneDist_T,AgeWeights_T,l_d,n_d1,n_d2,n_a1,n_a2,N_z,n_z,N_j,d2_grid,a1_gridvals,a2_grid,d_gridvals,a_gridvals,z_gridvals_J, pi_z_J,pi_z_J_sim,ze_gridvals_J_fastOLG,ReturnFn, aprimeFn, FnsToEvaluateCell, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, FnsToEvaluateParamNames, AggVarNames, PricePathSizeVec, ParamPathSizeVec, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, II1orII, II2, exceptlastj,exceptfirstj,justfirstj, transpathoptions, vfoptions, simoptions);
        else
            AggVarsPath=TransitionPath_FHorz_PType_ExpAsset_singlepath_e_raw(PricePathOld, ParamPath, PricePathNames,ParamPathNames,T,V_final,AgentDist_initial,jequalOneDist_T,AgeWeights_T,l_d,n_d1,n_d2,n_a1,n_a2,N_z,n_z,N_e,n_e,N_j,d2_grid,a1_gridvals,a2_grid,d_gridvals,a_gridvals,z_gridvals_J, pi_z_J,pi_z_J_sim,e_gridvals_J,pi_e_J,pi_e_J_sim,ze_gridvals_J_fastOLG,ReturnFn, aprimeFn, FnsToEvaluateCell, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, FnsToEvaluateParamNames, AggVarNames, PricePathSizeVec, ParamPathSizeVec, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, II1orII, II2, exceptlastj,exceptfirstj,justfirstj, transpathoptions, vfoptions, simoptions);
        end
    end
elseif transpathoptions.fastOLG==1
    if N_z==0
        if N_e==0
            AggVarsPath=TransitionPath_FHorz_PType_ExpAsset_singlepath_fastOLG_noz_raw(PricePathOld, ParamPath, PricePathNames,ParamPathNames,T,V_final,AgentDist_initial,jequalOneDist_T,AgeWeights_T,l_d,N_d1,N_d2,n_d1,n_d2,N_a1,N_a2,n_a1,n_a2,N_z,n_z,N_e,n_e,N_j,d1_grid,d2_grid,a1_grid,a2_grid,d_gridvals,aprime_gridvals,a_gridvals,z_gridvals_J, pi_z_J,pi_z_J_sim,e_gridvals_J,pi_e_J,pi_e_J_sim,ze_gridvals_J_fastOLG,ReturnFn, aprimeFn, FnsToEvaluateCell, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, FnsToEvaluateParamNames, AggVarNames, PricePathSizeVec, ParamPathSizeVec, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, II1orII, II2, exceptlastj,exceptfirstj,justfirstj, transpathoptions, vfoptions, simoptions);
        else
            AggVarsPath=TransitionPath_FHorz_PType_ExpAsset_singlepath_fastOLG_noz_e_raw(PricePathOld, ParamPath, PricePathNames,ParamPathNames,T,V_final,AgentDist_initial,jequalOneDist_T,AgeWeights_T,l_d,N_d1,N_d2,n_d1,n_d2,N_a1,N_a2,n_a1,n_a2,N_z,n_z,N_e,n_e,N_j,d1_grid,d2_grid,a1_grid,a2_grid,d_gridvals,aprime_gridvals,a_gridvals,z_gridvals_J, pi_z_J,pi_z_J_sim,e_gridvals_J,pi_e_J,pi_e_J_sim,ze_gridvals_J_fastOLG,ReturnFn, aprimeFn, FnsToEvaluateCell, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, FnsToEvaluateParamNames, AggVarNames, PricePathSizeVec, ParamPathSizeVec, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, II1orII, II2, exceptlastj,exceptfirstj,justfirstj, transpathoptions, vfoptions, simoptions);
        end
    else
        if N_e==0
            AggVarsPath=TransitionPath_FHorz_PType_ExpAsset_singlepath_fastOLG_raw(PricePathOld, ParamPath, PricePathNames,ParamPathNames,T,V_final,AgentDist_initial,jequalOneDist_T,AgeWeights_T,l_d,N_d1,N_d2,n_d1,n_d2,N_a1,N_a2,n_a1,n_a2,N_z,n_z,N_e,n_e,N_j,d1_grid,d2_grid,a1_grid,a2_grid,d_gridvals,aprime_gridvals,a_gridvals,z_gridvals_J, pi_z_J,pi_z_J_sim,e_gridvals_J,pi_e_J,pi_e_J_sim,ze_gridvals_J_fastOLG,ReturnFn, aprimeFn, FnsToEvaluateCell, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, FnsToEvaluateParamNames, AggVarNames, PricePathSizeVec, ParamPathSizeVec, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, II1orII, II2, exceptlastj,exceptfirstj,justfirstj, transpathoptions, vfoptions, simoptions);
        else
            AggVarsPath=TransitionPath_FHorz_PType_ExpAsset_singlepath_fastOLG_e_raw(PricePathOld, ParamPath, PricePathNames,ParamPathNames,T,V_final,AgentDist_initial,jequalOneDist_T,AgeWeights_T,l_d,N_d1,N_d2,n_d1,n_d2,N_a1,N_a2,n_a1,n_a2,N_z,n_z,N_e,n_e,N_j,d1_grid,d2_grid,a1_grid,a2_grid,d_gridvals,aprime_gridvals,a_gridvals,z_gridvals_J, pi_z_J,pi_z_J_sim,e_gridvals_J,pi_e_J,pi_e_J_sim,ze_gridvals_J_fastOLG,ReturnFn, aprimeFn, FnsToEvaluateCell, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, FnsToEvaluateParamNames, AggVarNames, PricePathSizeVec, ParamPathSizeVec, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, II1orII, II2, exceptlastj,exceptfirstj,justfirstj, transpathoptions, vfoptions, simoptions);
        end
    end
end


end
