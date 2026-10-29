function [GEcondns,aprimeReferencePath]=TransitionPath_FHorz_Anderson_resid(pvec,aprimeReferencePath,uselocalsearch,nPrices, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_z,n_e,N_j, N_d,N_a,N_semiz,N_z,N_e, l_d,l_aprime,l_a,l_semiz,l_z,l_e, d_gridvals, aprime_gridvals,a_gridvals,a_grid,semiz_gridvals_J,z_gridvals_J,e_gridvals_J,semizze_gridvals_J_fastOLG, pi_semiz_J, pi_z_J,pi_e_J,pi_e_J_vf,pi_semiz_J_sim,pi_z_J_sim,pi_e_J_sim, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, stockvarsNames, stockvarInPricePathNames, vfoptions, simoptions, transpathoptions, itercounter, PolicyIndexesPath, N_probs, II1, II2, exceptlastj, exceptfirstj, justfirstj)
% The residual AndersonAccelerationTPath evaluates with vfoptions.localsearch, as a TWO-OUTPUT call so
% that the reference policy is carried in and out. The FHorz counterpart of
% TransitionPath_InfHorz_Anderson_resid, see there. It is a file and not an anonymous function because
% an anonymous function forwards its body's outputs IN ORDER, and what is wanted is outputs 1 and 5 of
% TransitionPath_FHorz_singlepathiter.
% The argument list is TransitionPath_FHorz_singlepathiter's, GENERATED from its signature (PricePathOld
% replaced by pvec, aprimeReferencePath moved to the front), not transcribed.
%
% ONE FUNCTION SERVES BOTH CLOSURES: uselocalsearch=1 is the restricted residual Anderson iterates on,
% uselocalsearch=0 the VerifyFn that certifies convergence. Both leave a reference behind for the next
% restricted sweep, the unrestricted one by way of localsearchseed.
% An empty reference forces an unrestricted sweep: the first sweep of a solve is what builds it.
if isempty(aprimeReferencePath)
    uselocalsearch=0;
end
vfoptions.localsearch=uselocalsearch;
vfoptions.localsearchseed=1-uselocalsearch;

[GEcondnPath,~,~,~,aprimeReferencePath]=TransitionPath_FHorz_singlepathiter(reshape(pvec,T,nPrices), PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_z,n_e,N_j, N_d,N_a,N_semiz,N_z,N_e, l_d,l_aprime,l_a,l_semiz,l_z,l_e, d_gridvals, aprime_gridvals,a_gridvals,a_grid,semiz_gridvals_J,z_gridvals_J,e_gridvals_J,semizze_gridvals_J_fastOLG, pi_semiz_J, pi_z_J,pi_e_J,pi_e_J_vf,pi_semiz_J_sim,pi_z_J_sim,pi_e_J_sim, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, stockvarsNames, stockvarInPricePathNames, vfoptions, simoptions, transpathoptions, itercounter, PolicyIndexesPath, aprimeReferencePath, N_probs, II1, II2, exceptlastj, exceptfirstj, justfirstj);

GEcondns=reshape(GEcondnPath,[],1);

end
