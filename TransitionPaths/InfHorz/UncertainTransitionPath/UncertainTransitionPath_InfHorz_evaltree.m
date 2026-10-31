function [GEcondnvec,GEcondnTree,AggVarsTree,VPathTree,PolicyPathTree,AgentDistPathTree]=UncertainTransitionPath_InfHorz_evaltree(p, nNodes, nPrices, PricePathNames, notlast, ParamPath, ParamPathNodes, ParamPathNames, T, T1, T2, pi_S, S_initial, V_final, Policy_final, AgentDist_initial, n_d,n_a,n_z, d_grid,a_grid,z_grid, pi_z, ReturnFn, FnsToEvaluate, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, transpathoptions, simoptions, vfoptions)
% One evaluation of the tree of an uncertain transition path at the prices p, for UncertainTransitionPath_InfHorz:
% the value fn, agent dist and aggregate variables on every node, and the general eqm conditions at every node.
% The counterpart of TransitionPath_InfHorz_singlepathiter, used where the algorithm needs the general eqm conditions
% as a function of the prices (Anderson acceleration).
%   p is the prices stacked as nNodes-by-nPrices (in the order of PricePathNames), as a column vector
%   GEcondnvec is the general eqm conditions at the nodes before period T (the notlast nodes), stacked as
%     nnotlast-by-nGEeqns, as a column vector
%   GEcondnTree is the general eqm conditions at every node, nNodes-by-nGEeqns

PricePathOld=reshape(gather(p),[nNodes,nPrices]);
PricePathStruct=struct();
for pp=1:nPrices
    PricePathStruct.(PricePathNames{pp})=PricePathOld(:,pp)';
end
[VPathTree,PolicyPathTree]=ValueFnOnUncertainTransPath_InfHorz(PricePathStruct, ParamPath, T, T1, T2, pi_S, S_initial, V_final, Policy_final, Parameters, n_d,n_a,n_z, d_grid,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, transpathoptions, vfoptions);
AgentDistPathTree=AgentDistOnUncertainTransPath_InfHorz(AgentDist_initial, PricePathStruct, ParamPath, PolicyPathTree, T, T1, T2, pi_S, S_initial, n_d,n_a,n_z,pi_z, Parameters, transpathoptions, simoptions);
AggVarsTree=EvalFnOnUncertainTransPath_AggVars_InfHorz(FnsToEvaluate, AgentDistPathTree, PolicyPathTree, PricePathStruct, ParamPath, Parameters, T, T1, T2, pi_S, S_initial, n_d,n_a,n_z, d_grid,a_grid,z_grid, transpathoptions, simoptions);
AggVarNames=fieldnames(AggVarsTree);

% The general eqm conditions at every node, evaluated as updatePricePathNew_TPath_tt does in each period
nGEeqns=length(GeneralEqmEqnsCell);
GEcondnTree=zeros(nNodes,nGEeqns);
for kk=1:nNodes
    Parameters_k=Parameters;
    for pp=1:length(ParamPathNames)
        Parameters_k.(ParamPathNames{pp})=ParamPathNodes.(ParamPathNames{pp})(kk);
    end
    for pp=1:nPrices
        Parameters_k.(PricePathNames{pp})=PricePathOld(kk,pp);
    end
    for ff=1:length(AggVarNames)
        Parameters_k.(AggVarNames{ff})=AggVarsTree.(AggVarNames{ff}).Mean(kk);
    end
    for gg=1:nGEeqns
        GEcondnTree(kk,gg)=real(GeneralEqmConditions_Case1_v3(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg).Names, Parameters_k));
    end
end
GEcondnvec=reshape(GEcondnTree(notlast,:),[],1);

end
