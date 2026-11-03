function [GEcondnvec,GEcondnTree,AggVarsTree,VPathTree,PolicyPathTree,AgentDistPathTree,AggVarsGivenChildTree]=UncertainTransitionPath_InfHorz_evaltree(p, nNodes, nPrices, PricePathNames, notlast, Tree, ParamPath, ParamPathNodes, ParamPathNames, T, T1, T2, pi_S, S_initial, V_final, Policy_final, AgentDist_initial, n_d,n_a,n_z, d_grid,a_grid,z_grid, pi_z, ReturnFn, FnsToEvaluate, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, GEusestplus1, tplus1priceNames, tplus1pricePathkk, tplus1FnNames, tminus1priceNames, tminus1paramNames, tminus1AggVarsNames, Parameters, DiscountFactorParamNames, transpathoptions, simoptions, vfoptions)
% One evaluation of the tree of an uncertain transition path at the prices p, for UncertainTransitionPath_InfHorz:
% the value fn, agent dist and aggregate variables on every node, and the general eqm conditions at every node.
% The counterpart of TransitionPath_InfHorz_singlepathiter, used by both the shooting algorithm and Anderson acceleration.
%   p is the prices stacked as nNodes-by-nPrices (in the order of PricePathNames), as a column vector
%   GEcondnvec is the general eqm conditions at the nodes before period T (the notlast nodes), stacked as
%     nnotlast-by-nGEeqns, as a column vector
%   GEcondnTree is the general eqm conditions at every node, nNodes-by-nGEeqns
% _tminus1 is the value at the parent node (transpathoptions.initialvalues at the period 1 node). A general eqm condition
% that uses _tplus1 (GEusestplus1) is evaluated once for each child of the node, with the child's prices and the
% aggregates given that child (AggVarsGivenChildTree), and the condition at the node is the expectation over the
% children (see UncertainTransitionPath_InfHorz). At a node with one child that is the child's prices, and at a period
% T node the node's own prices (the final eqm), as on a deterministic path.

PricePathOld=reshape(gather(p),[nNodes,nPrices]);
PricePathStruct=struct();
for pp=1:nPrices
    PricePathStruct.(PricePathNames{pp})=PricePathOld(:,pp)';
end
[VPathTree,PolicyPathTree]=ValueFnOnUncertainTransPath_InfHorz(PricePathStruct, ParamPath, T, T1, T2, pi_S, S_initial, V_final, Policy_final, Parameters, n_d,n_a,n_z, d_grid,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, transpathoptions, vfoptions);
AgentDistPathTree=AgentDistOnUncertainTransPath_InfHorz(AgentDist_initial, PricePathStruct, ParamPath, PolicyPathTree, T, T1, T2, pi_S, S_initial, n_d,n_a,n_z,pi_z, Parameters, transpathoptions, simoptions);
[AggVarsTree,AggVarsGivenChildTree]=EvalFnOnUncertainTransPath_AggVars_InfHorz(FnsToEvaluate, AgentDistPathTree, PolicyPathTree, PricePathStruct, ParamPath, Parameters, T, T1, T2, pi_S, S_initial, n_d,n_a,n_z, d_grid,a_grid,z_grid, transpathoptions, simoptions);
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
    % _tminus1: the values at the parent node, as the deterministic path takes them from the period before (and the
    % prices from the same prices, PricePathOld)
    parentnode=Tree.nodeparent(kk);
    for pp=1:length(tminus1priceNames)
        if parentnode>0
            Parameters_k.([tminus1priceNames{pp},'_tminus1'])=PricePathOld(parentnode,strcmp(PricePathNames,tminus1priceNames{pp}));
        else
            Parameters_k.([tminus1priceNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1priceNames{pp});
        end
    end
    for pp=1:length(tminus1paramNames)
        if parentnode>0
            Parameters_k.([tminus1paramNames{pp},'_tminus1'])=ParamPathNodes.(tminus1paramNames{pp})(parentnode);
        else
            Parameters_k.([tminus1paramNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1paramNames{pp});
        end
    end
    for ff=1:length(tminus1AggVarsNames)
        if parentnode>0
            Parameters_k.([tminus1AggVarsNames{ff},'_tminus1'])=AggVarsTree.(tminus1AggVarsNames{ff}).Mean(parentnode);
        else
            Parameters_k.([tminus1AggVarsNames{ff},'_tminus1'])=transpathoptions.initialvalues.(tminus1AggVarsNames{ff});
        end
    end
    % The conditions that do not use _tplus1
    for gg=find(~GEusestplus1)
        GEcondnTree(kk,gg)=real(GeneralEqmConditions_Case1_v3(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg).Names, Parameters_k));
    end
    % The conditions that use _tplus1, in expectation over the children. At a node with one child (and at a period T
    % node, which is its own next period) the weight is exactly one, so this is the deterministic condition.
    if any(GEusestplus1)
        if Tree.nodenchild(kk)==0
            children=kk;
            w=1;
        elseif Tree.nodenchild(kk)==1
            children=Tree.nodefirstchild(kk);
            w=1;
        else
            children=Tree.nodefirstchild(kk)+(0:Tree.nodenchild(kk)-1);
            w=Tree.nodecondprob(children);
        end
        for cc=1:length(children)
            for pp=1:length(tplus1priceNames)
                Parameters_k.([tplus1priceNames{pp},'_tplus1'])=PricePathOld(children(cc),tplus1pricePathkk(pp));
            end
            if Tree.nodenchild(kk)>0
                for ff=1:length(tplus1FnNames)
                    Parameters_k.(tplus1FnNames{ff})=AggVarsGivenChildTree.(tplus1FnNames{ff})(children(cc));
                end
            end
            for gg=find(GEusestplus1)
                GEcondnTree(kk,gg)=GEcondnTree(kk,gg)+w(cc)*real(GeneralEqmConditions_Case1_v3(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg).Names, Parameters_k));
            end
        end
    end
end
GEcondnvec=reshape(GEcondnTree(notlast,:),[],1);

end
