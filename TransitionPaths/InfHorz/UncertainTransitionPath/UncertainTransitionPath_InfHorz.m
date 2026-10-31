function [PricePathTree, UncertainTPathSummary]=UncertainTransitionPath_InfHorz(PricePathShaper, ParamPath, T, T1, T2, pi_S, S_initial, AgentDist_initial, n_d,n_a,n_z, d_grid,a_grid,z_grid, pi_z, ReturnFn, FnsToEvaluate, GeneralEqmEqns, GeneralEqmEqns_Transition, Parameters, DiscountFactorParamNames, transpathoptions, simoptions, vfoptions, heteroagentoptions, simoptions_finaleqm, vfoptions_finaleqm)
% The general eqm transition path of an infinite horizon model with an aggregate shock S. T1,...,T2 are the periods
% in which agents face uncertainty: in each of them next period's S is drawn from pi_S(S_t,:). So S_t=S_initial up to
% T1, and S is frozen from T2+1 on. Prices depend on the history of S, so
% general eqm is solved on the TREE of histories (see UncertainTransPathTree for the tree, the timing of pi_S, and
% the forms PricePath and ParamPath can take). Everything is held at the final stationary eqm of its S in period T.
%
% INPUTS (as for TransitionPath_InfHorz, and as MultipleRevealTransitionPath_Case1_FHorz for the final eqms)
%   PricePathShaper.(price) is 1-by-T: the initial guess at a node in period t is p_initial+(p_final(S)-p_initial)*shaper(t),
%     with p_initial the value in Parameters (the eqm before the path) and p_final(S) the final stationary eqm of the
%     node's S. transpathoptions.PricePathTree0 (any form UncertainTransPathTree accepts) overrides it, e.g. the
%     PricePathTree of an earlier solve on the same tree.
%   ParamPath.(name) is N_S-by-T for a parameter that depends on S, 1-by-T for one that does not.
%   AgentDist_initial is the agent distribution in period 1.
%   GeneralEqmEqns are those of the final stationary eqms; GeneralEqmEqns_Transition those of the path.
%   heteroagentoptions, simoptions_finaleqm and vfoptions_finaleqm are for the final stationary eqms.
% The final stationary eqm of each S that a history can be in at period T is solved here, with
% HeteroAgentStationaryEqm_InfHorz, at that S's period-T parameters; its value fn (with vfoptions, the path options)
% is the V_final of that S.
%
% ALGORITHM: shooting (transpathoptions.GEnewprice=3, with GEnewprice3.howtoupdate as in TransitionPath_InfHorz).
% Each iteration evaluates the whole tree at the current prices (ValueFnOnUncertainTransPath_InfHorz,
% AgentDistOnUncertainTransPath_InfHorz, EvalFnOnUncertainTransPath_AggVars_InfHorz), then updates the prices at
% every node from the general eqm conditions at that node, exactly as the deterministic shooting algorithm does in
% each period. Converged when, over every node before period T, the prices move by less than toleranceGEprices and
% the general eqm conditions are within toleranceGEcondns (the same norms as TransitionPath_InfHorz).
%
% OUTPUTS
%   PricePathTree.(price) is 1-by-nNodes, the prices on the tree (at convergence, the prices the summary was evaluated at).
%   UncertainTPathSummary: .Tree (from UncertainTransPathTree), .GEcondnTree.(eqn) (1-by-nNodes), .AggVarsTree,
%     .VPathTree, .PolicyPathTree, .AgentDistPathTree, .finaleqm.S<k> (p_eqm, GeneralEqmCondnValues, V_final,
%     Policy_final), .iterations, .PricePathDist, .GEcondnDist.
%   The prices of history h are PricePathTree.(price)(UncertainTPathSummary.Tree.histories(h,:)).
%
% Not yet: Anderson (GEnewprice=2), price constraints, intermediateEqns, _tminus1/_tplus1 in the general eqm
% conditions, local search, permanent types.

%% Options
if ~exist('transpathoptions','var') || isempty(transpathoptions)
    transpathoptions=struct();
end
if ~exist('simoptions','var') || isempty(simoptions)
    simoptions=struct();
end
if ~exist('vfoptions','var') || isempty(vfoptions)
    vfoptions=struct();
end
if ~exist('heteroagentoptions','var') || isempty(heteroagentoptions)
    heteroagentoptions=struct();
end
if ~exist('simoptions_finaleqm','var') || isempty(simoptions_finaleqm)
    simoptions_finaleqm=simoptions;
end
if ~exist('vfoptions_finaleqm','var') || isempty(vfoptions_finaleqm)
    vfoptions_finaleqm=vfoptions;
end
if ~isfield(transpathoptions,'GEnewprice')
    error('transpathoptions.GEnewprice must be set: =3 for the shooting algorithm')
elseif transpathoptions.GEnewprice==1
    error('transpathoptions.GEnewprice=1 (quasi-Newton) is not implemented for uncertain transition paths, use =3 (the shooting algorithm)')
elseif transpathoptions.GEnewprice==2
    error('transpathoptions.GEnewprice=2 (Anderson acceleration) is not yet implemented for uncertain transition paths, use =3 (the shooting algorithm)')
elseif transpathoptions.GEnewprice~=3
    error('transpathoptions.GEnewprice must be 3 (the shooting algorithm) for uncertain transition paths')
end
if ~isfield(transpathoptions,'GEnewprice3') || ~isfield(transpathoptions.GEnewprice3,'howtoupdate')
    error('transpathoptions.GEnewprice=3 (the shooting algorithm) requires transpathoptions.GEnewprice3.howtoupdate')
end
if ~isfield(transpathoptions,'oldpathweight')
    transpathoptions.oldpathweight=0; % as TransitionPath_InfHorz under GEnewprice=3: the step is in howtoupdate
end
if ~isfield(transpathoptions,'maxiter')
    transpathoptions.maxiter=1000;
end
if ~isfield(transpathoptions,'verbose')
    transpathoptions.verbose=0;
end
if ~isfield(transpathoptions,'toleranceGEprices')
    transpathoptions.toleranceGEprices=Inf; % off by default, as in TransitionPath_InfHorz
end
if ~isfield(transpathoptions,'toleranceGEcondns')
    transpathoptions.toleranceGEcondns=1e-4;
end
if ~isfield(transpathoptions,'multiGEcriterion')
    transpathoptions.multiGEcriterion=1;
end
if ~isfield(transpathoptions,'multiGEweights')
    transpathoptions.multiGEweights=ones(1,length(fieldnames(GeneralEqmEqns_Transition)));
end
if ~isfield(transpathoptions,'updateaccuracycutoff')
    transpathoptions.updateaccuracycutoff=10^(-9);
end
if ~isfield(vfoptions,'localsearch')
    vfoptions.localsearch=0;
end
% Not yet implemented, refused by name rather than ignored
if vfoptions.localsearch~=0
    error('UncertainTransitionPath_InfHorz: vfoptions.localsearch is not yet implemented for uncertain transition paths')
end
if isfield(transpathoptions,'intermediateEqns')
    error('UncertainTransitionPath_InfHorz: transpathoptions.intermediateEqns is not yet implemented for uncertain transition paths')
end
if (isfield(transpathoptions,'constrainpositive') && ~isempty(transpathoptions.constrainpositive)) || (isfield(transpathoptions,'constrain0to1') && ~isempty(transpathoptions.constrain0to1)) || (isfield(transpathoptions,'constrainAtoB') && ~isempty(transpathoptions.constrainAtoB))
    error('UncertainTransitionPath_InfHorz: price constraints (constrainpositive, constrain0to1, constrainAtoB) are not yet implemented for uncertain transition paths')
end
GEeqnNames=fieldnames(GeneralEqmEqns_Transition);
nGEeqns=length(GEeqnNames);
GeneralEqmEqnsCell=cell(1,nGEeqns);
for gg=1:nGEeqns
    temp=getAnonymousFnInputNames(GeneralEqmEqns_Transition.(GEeqnNames{gg}));
    for ii=1:length(temp)
        if endsWith(temp{ii},'_tminus1') || endsWith(temp{ii},'_tplus1')
            error('UncertainTransitionPath_InfHorz: GeneralEqmEqns_Transition.%s uses %s; _tminus1 and _tplus1 are not yet implemented for uncertain transition paths',GEeqnNames{gg},temp{ii})
        end
    end
    GeneralEqmEqnParamNames(gg).Names=temp;
    GeneralEqmEqnsCell{gg}=GeneralEqmEqns_Transition.(GEeqnNames{gg});
end

%% The tree (prices are put on it once the final eqms are known)
[Tree,~,ParamPathNodes]=UncertainTransPathTree(T,T1,T2,pi_S,S_initial,[],ParamPath,transpathoptions);
nNodes=Tree.nNodes;
ParamPathNames=fieldnames(ParamPathNodes);
PricePathNames=fieldnames(PricePathShaper);
nPrices=length(PricePathNames);
transpathoptions=setupGEnewprice3_shooting(transpathoptions,GeneralEqmEqns_Transition,PricePathNames);

%% The final stationary eqm of each S that a history can be in at period T
finalS=unique(Tree.nodeS(Tree.nodedate==T))';
p_initial=zeros(1,nPrices);
for pp=1:nPrices
    p_initial(pp)=Parameters.(PricePathNames{pp});
end
p_final=nan(Tree.N_S,nPrices); % row S, for the S that have a final eqm
V_final=struct();
Policy_final=struct();
for SS=finalS
    Sstr=['S',num2str(SS)];
    Parameters_S=Parameters;
    % the period T parameters of a history in S
    kk=find(Tree.nodedate==T & Tree.nodeS==SS,1);
    for pp=1:length(ParamPathNames)
        Parameters_S.(ParamPathNames{pp})=ParamPathNodes.(ParamPathNames{pp})(kk);
    end
    if transpathoptions.verbose>=1
        fprintf('UncertainTransitionPath_InfHorz: solving the final stationary eqm of S=%i \n',SS)
    end
    [p_eqm_S,GEcondns_S]=HeteroAgentStationaryEqm_InfHorz(n_d, n_a, n_z, 0, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters_S, DiscountFactorParamNames, [], [], [], PricePathNames, heteroagentoptions, simoptions_finaleqm, vfoptions_finaleqm);
    for pp=1:nPrices
        Parameters_S.(PricePathNames{pp})=p_eqm_S.(PricePathNames{pp});
        p_final(SS,pp)=p_eqm_S.(PricePathNames{pp});
    end
    [V_final.(Sstr),Policy_final.(Sstr)]=ValueFnIter_InfHorz(n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z,ReturnFn,Parameters_S,DiscountFactorParamNames,[],vfoptions);
    UncertainTPathSummary.finaleqm.(Sstr).p_eqm=p_eqm_S;
    UncertainTPathSummary.finaleqm.(Sstr).GeneralEqmCondnValues=GEcondns_S;
    UncertainTPathSummary.finaleqm.(Sstr).V_final=V_final.(Sstr);
    UncertainTPathSummary.finaleqm.(Sstr).Policy_final=Policy_final.(Sstr);
end

%% Initial guess on the tree: PricePathOld is nNodes-by-nPrices
PricePathOld=zeros(nNodes,nPrices);
if isfield(transpathoptions,'PricePathTree0')
    [~,PricePathNodes0]=UncertainTransPathTree(T,T1,T2,pi_S,S_initial,transpathoptions.PricePathTree0,[],transpathoptions);
    for pp=1:nPrices
        PricePathOld(:,pp)=PricePathNodes0.(PricePathNames{pp})';
    end
else
    for pp=1:nPrices
        shaper=PricePathShaper.(PricePathNames{pp});
        if numel(shaper)~=T
            error('UncertainTransitionPath_InfHorz: PricePathShaper.%s must be 1-by-T',PricePathNames{pp})
        end
        shaper=shaper(:);
        for kk=1:nNodes
            pfin=p_final(Tree.nodeS(kk),pp);
            if isnan(pfin)
                pfin=p_initial(pp); % an S with no final eqm (none of its histories is still in it at period T)
            end
            PricePathOld(kk,pp)=p_initial(pp)+(pfin-p_initial(pp))*shaper(Tree.nodedate(kk));
        end
    end
end
% Period T is held at the final eqm of its S
lastnodes=find(Tree.nodedate==T)';
for kk=lastnodes
    PricePathOld(kk,:)=p_final(Tree.nodeS(kk),:);
end
notlast=(Tree.nodedate<T); % the nodes whose prices are solved for

%% Shooting
PricePathDist=Inf;
GEcondnDist=Inf;
itercounter=1;
while itercounter<=transpathoptions.maxiter
    PricePathStruct=struct();
    for pp=1:nPrices
        PricePathStruct.(PricePathNames{pp})=PricePathOld(:,pp)';
    end
    [VPathTree,PolicyPathTree]=ValueFnOnUncertainTransPath_InfHorz(PricePathStruct, ParamPath, T, T1, T2, pi_S, S_initial, V_final, Policy_final, Parameters, n_d,n_a,n_z, d_grid,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, transpathoptions, vfoptions);
    AgentDistPathTree=AgentDistOnUncertainTransPath_InfHorz(AgentDist_initial, PricePathStruct, ParamPath, PolicyPathTree, T, T1, T2, pi_S, S_initial, n_d,n_a,n_z,pi_z, Parameters, transpathoptions, simoptions);
    AggVarsTree=EvalFnOnUncertainTransPath_AggVars_InfHorz(FnsToEvaluate, AgentDistPathTree, PolicyPathTree, PricePathStruct, ParamPath, Parameters, T, T1, T2, pi_S, S_initial, n_d,n_a,n_z, d_grid,a_grid,z_grid, transpathoptions, simoptions);
    AggVarNames=fieldnames(AggVarsTree);

    % The general eqm conditions, and the shooting update, at every node (as the deterministic shooting algorithm does in
    % each period, with updatePricePathNew_TPath_tt)
    PricePathNew=PricePathOld;
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
        [PricePathNew_k,GEcondnTree(kk,:)]=updatePricePathNew_TPath_tt(Parameters_k,GeneralEqmEqnsCell,GeneralEqmEqnParamNames,PricePathOld(kk,:),itercounter,transpathoptions);
        if notlast(kk)
            PricePathNew(kk,:)=PricePathNew_k;
        end
    end

    if any(~isfinite(GEcondnTree(notlast,:)),'all')
        error('UncertainTransitionPath_InfHorz: the general eqm conditions are not finite at some node (iteration %i)',itercounter)
    end

    % Distances, over the nodes before period T, in the norms TransitionPath_InfHorz uses
    PricePathDist=max(abs(PricePathNew(notlast,:)-PricePathOld(notlast,:)),[],'all');
    if transpathoptions.multiGEcriterion==0
        GEcondnDist=max(sum(abs(transpathoptions.multiGEweights.*GEcondnTree(notlast,:)),2));
    elseif transpathoptions.multiGEcriterion==1
        GEcondnDist=max(sqrt(sum(transpathoptions.multiGEweights.*(GEcondnTree(notlast,:).^2),2)));
    end
    if transpathoptions.verbose>=1
        fprintf('UncertainTransitionPath_InfHorz: iteration %i, price distance %8.6f, general eqm conditions %8.6f \n',itercounter,PricePathDist,GEcondnDist)
    end

    % Converged on the prices just evaluated: stop BEFORE the update, so the summary is of the returned prices
    if PricePathDist<=transpathoptions.toleranceGEprices && GEcondnDist<=transpathoptions.toleranceGEcondns
        break
    end
    if itercounter==transpathoptions.maxiter
        break
    end

    % setupGEnewprice3_shooting makes oldpathweight one weight per price (zero for every shooting row, so this is just the
    % new prices); as a row it applies to each price's column
    oldw=reshape(transpathoptions.oldpathweight,[1,nPrices]);
    PricePathOld(notlast,:)=oldw.*PricePathOld(notlast,:)+(1-oldw).*PricePathNew(notlast,:);
    itercounter=itercounter+1;
end
if PricePathDist>transpathoptions.toleranceGEprices || GEcondnDist>transpathoptions.toleranceGEcondns
    warning('UncertainTransitionPath_InfHorz: reached maxiter (%i) without convergence; the general eqm conditions are %8.6f from zero, against toleranceGEcondns=%g',transpathoptions.maxiter,GEcondnDist,transpathoptions.toleranceGEcondns)
end

%% Outputs
PricePathTree=struct();
for pp=1:nPrices
    PricePathTree.(PricePathNames{pp})=PricePathOld(:,pp)';
end
UncertainTPathSummary.Tree=Tree;
for gg=1:nGEeqns
    UncertainTPathSummary.GEcondnTree.(GEeqnNames{gg})=GEcondnTree(:,gg)';
end
UncertainTPathSummary.AggVarsTree=AggVarsTree;
UncertainTPathSummary.VPathTree=VPathTree;
UncertainTPathSummary.PolicyPathTree=PolicyPathTree;
UncertainTPathSummary.AgentDistPathTree=AgentDistPathTree;
UncertainTPathSummary.iterations=itercounter;
UncertainTPathSummary.PricePathDist=PricePathDist;
UncertainTPathSummary.GEcondnDist=GEcondnDist;

end
