function [GEcondnvec,GEcondnTree,AggVarsTree,VPathTree,PolicyPathTree,AgentDistPathTree]=UncertainTransitionPath_FHorz_evaltree(p, nNodes, nPrices, PricePathNames, notlast, ParamPath, ParamPathNodes, ParamPathNames, T, T1, T2, pi_S, S_initial, V_final, Policy_final, AgentDist_initial, jequaloneDist, n_d,n_a,n_z,N_j, d_grid,a_grid,z_grid, pi_z, ReturnFn, FnsToEvaluate, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeightsParamNames, transpathoptions, simoptions, vfoptions)
% One evaluation of the tree of an uncertain transition path of a finite horizon (OLG) model at the prices p, for
% UncertainTransitionPath_FHorz: the value fn, agent dist and aggregate variables on every node, and the general eqm
% conditions at every node. The FHorz counterpart of UncertainTransitionPath_InfHorz_evaltree, and of
% TransitionPath_FHorz_singlepathiter.
%   p is the prices stacked as nNodes-by-nPrices (in the order of PricePathNames), as a column vector
%   GEcondnvec is the general eqm conditions at the nodes before period T (the notlast nodes), stacked as
%     nnotlast-by-nGEeqns, as a column vector
%   GEcondnTree is the general eqm conditions at every node, nNodes-by-nGEeqns
% A parameter on the tree is k-by-nNodes (k=N_j when it depends on age); at a node it is put in Parameters as a row,
% 1-by-k, as TransitionPath_FHorz_singlepathiter puts a period of the ParamPath matrix.

PricePathOld=reshape(gather(p),[nNodes,nPrices]);
PricePathStruct=struct();
for pp=1:nPrices
    PricePathStruct.(PricePathNames{pp})=PricePathOld(:,pp)';
end
[VPathTree,PolicyPathTree]=ValueFnOnUncertainTransPath_FHorz(PricePathStruct, ParamPath, T, T1, T2, pi_S, S_initial, V_final, Policy_final, Parameters, n_d,n_a,n_z,N_j, d_grid,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, transpathoptions, vfoptions);
AgentDistPathTree=AgentDistOnUncertainTransPath_FHorz(AgentDist_initial, jequaloneDist, PricePathStruct, ParamPath, PolicyPathTree, AgeWeightsParamNames, T, T1, T2, pi_S, S_initial, n_d,n_a,n_z,N_j,pi_z, Parameters, transpathoptions, simoptions);
AggVarsTree=EvalFnOnUncertainTransPath_AggVars_FHorz(FnsToEvaluate, AgentDistPathTree, PolicyPathTree, PricePathStruct, ParamPath, Parameters, T, T1, T2, pi_S, S_initial, n_d,n_a,n_z,N_j, d_grid,a_grid,z_grid, transpathoptions, simoptions);
AggVarNames=fieldnames(AggVarsTree);

% The general eqm conditions at every node, evaluated as TransitionPath_FHorz_singlepathiter does in each period
nGEeqns=length(GeneralEqmEqnsCell);
GEcondnTree=zeros(nNodes,nGEeqns);
for kk=1:nNodes
    Parameters_k=Parameters;
    for pp=1:length(ParamPathNames)
        Parameters_k.(ParamPathNames{pp})=ParamPathNodes.(ParamPathNames{pp})(:,kk)';
    end
    for pp=1:nPrices
        Parameters_k.(PricePathNames{pp})=PricePathOld(kk,pp);
    end
    for ff=1:length(AggVarNames)
        Parameters_k.(AggVarNames{ff})=AggVarsTree.(AggVarNames{ff}).Mean(kk);
    end
    for gg=1:nGEeqns
        GEcondnTree(kk,gg)=GeneralEqmConditions_Case1_v3(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg).Names, Parameters_k);
    end
end
GEcondnvec=reshape(GEcondnTree(notlast,:),[],1);

end
