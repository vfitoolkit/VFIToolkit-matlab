function [GEcondnvec,GEcondnTree,AggVarsTree,VPathTree,PolicyPathTree,AgentDistPathTree,AggVarsGivenChildTree]=UncertainTransitionPath_FHorz_evaltree(p, nNodes, nPrices, PricePathNames, notlast, Tree, ParamPath, ParamPathNodes, ParamPathNames, T, T1, T2, pi_S, S_initial, V_final, Policy_final, AgentDist_initial, jequaloneDist, n_d,n_a,n_z,N_j, d_grid,a_grid,z_grid, pi_z, ReturnFn, FnsToEvaluate, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, GEusestplus1, tplus1priceNames, tplus1pricePathkk, tplus1FnNames, tminus1priceNames, tminus1paramNames, tminus1AggVarsNames, Parameters, DiscountFactorParamNames, AgeWeightsParamNames, transpathoptions, simoptions, vfoptions)
% One evaluation of the tree of an uncertain transition path of a finite horizon (OLG) model at the prices p, for
% UncertainTransitionPath_FHorz: the value fn, agent dist and aggregate variables on every node, and the general eqm
% conditions at every node. The FHorz counterpart of UncertainTransitionPath_InfHorz_evaltree, and of
% TransitionPath_FHorz_singlepathiter. Used by both the shooting algorithm and Anderson acceleration.
%   p is the prices stacked as nNodes-by-nPrices (in the order of PricePathNames), as a column vector
%   GEcondnvec is the general eqm conditions at the nodes before period T (the notlast nodes), stacked as
%     nnotlast-by-nGEeqns, as a column vector
%   GEcondnTree is the general eqm conditions at every node, nNodes-by-nGEeqns
% A parameter on the tree is k-by-nNodes (k=N_j when it depends on age); at a node it is put in Parameters as a row,
% 1-by-k, as TransitionPath_FHorz_singlepathiter puts a period of the ParamPath matrix.
% _tminus1 is the value at the parent node (transpathoptions.initialvalues at the period 1 node). A general eqm condition
% that uses _tplus1 (GEusestplus1) is evaluated once for each child of the node, with the child's prices and the
% aggregates given that child (AggVarsGivenChildTree), and the condition at the node is the expectation over the
% children (see UncertainTransitionPath_FHorz). At a node with one child that is the child's prices, and at a period
% T node the node's own prices (the final eqm), as on a deterministic path.
% INTERMEDIATE EQNS (transpathoptions.intermediateEqns) are evaluated at each node, in order, after the aggregates and
% before the general eqm conditions. One that uses _tplus1 (transpathoptions.intermediateEqnsusetplus1, set by
% UncertainTransitionPath_FHorz) depends on the child, so it is evaluated once for each child, with that child's
% values, inside the loop over the children.

PricePathOld=reshape(gather(p),[nNodes,nPrices]);
PricePathStruct=struct();
for pp=1:nPrices
    PricePathStruct.(PricePathNames{pp})=PricePathOld(:,pp)';
end
[VPathTree,PolicyPathTree]=ValueFnOnUncertainTransPath_FHorz(PricePathStruct, ParamPath, T, T1, T2, pi_S, S_initial, V_final, Policy_final, Parameters, n_d,n_a,n_z,N_j, d_grid,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, transpathoptions, vfoptions);
AgentDistPathTree=AgentDistOnUncertainTransPath_FHorz(AgentDist_initial, jequaloneDist, PricePathStruct, ParamPath, PolicyPathTree, AgeWeightsParamNames, T, T1, T2, pi_S, S_initial, n_d,n_a,n_z,N_j,pi_z, Parameters, transpathoptions, simoptions);
[AggVarsTree,AggVarsGivenChildTree]=EvalFnOnUncertainTransPath_AggVars_FHorz(FnsToEvaluate, AgentDistPathTree, PolicyPathTree, PricePathStruct, ParamPath, Parameters, T, T1, T2, pi_S, S_initial, n_d,n_a,n_z,N_j, d_grid,a_grid,z_grid, transpathoptions, simoptions);
AggVarNames=fieldnames(AggVarsTree);

% The intermediate eqns: those that do not use _tplus1 at the node, those that do once per child
nIntEqns=0;
if transpathoptions.useintermediateEqns==1
    intEqnNames=fieldnames(transpathoptions.intermediateEqns);
    nIntEqns=length(intEqnNames);
end

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
            Parameters_k.([tminus1paramNames{pp},'_tminus1'])=ParamPathNodes.(tminus1paramNames{pp})(:,parentnode)';
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
    % The intermediate eqns that do not use _tplus1, in order
    for gg=1:nIntEqns
        if transpathoptions.intermediateEqnsusetplus1(gg)==0
            Parameters_k.(intEqnNames{gg})=GeneralEqmConditions_Case1_v3(transpathoptions.intermediateEqnsCell{gg}, transpathoptions.intermediateEqnParamNames(gg).Names, Parameters_k);
        end
    end
    % The conditions that do not use _tplus1
    for gg=find(~GEusestplus1)
        GEcondnTree(kk,gg)=GeneralEqmConditions_Case1_v3(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg).Names, Parameters_k);
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
            % The intermediate eqns that use _tplus1, in order, with this child's values
            for gg=1:nIntEqns
                if transpathoptions.intermediateEqnsusetplus1(gg)==1
                    Parameters_k.(intEqnNames{gg})=GeneralEqmConditions_Case1_v3(transpathoptions.intermediateEqnsCell{gg}, transpathoptions.intermediateEqnParamNames(gg).Names, Parameters_k);
                end
            end
            for gg=find(GEusestplus1)
                GEcondnTree(kk,gg)=GEcondnTree(kk,gg)+w(cc)*GeneralEqmConditions_Case1_v3(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg).Names, Parameters_k);
            end
        end
    end
end
GEcondnvec=reshape(GEcondnTree(notlast,:),[],1);

end
