function [PricePathTree, UncertainTPathSummary]=UncertainTransitionPath_FHorz(PricePathShaper, ParamPath, T, T1, T2, pi_S, S_initial, AgentDist_initial, jequaloneDist, n_d,n_a,n_z,N_j, d_grid,a_grid,z_grid, pi_z, ReturnFn, FnsToEvaluate, GeneralEqmEqns, GeneralEqmEqns_Transition, Parameters, DiscountFactorParamNames, AgeWeightsParamNames, transpathoptions, simoptions, vfoptions, heteroagentoptions, simoptions_finaleqm, vfoptions_finaleqm)
% The general eqm transition path of a finite horizon (OLG) model with an aggregate shock S. The FHorz counterpart of
% UncertainTransitionPath_InfHorz: T1,...,T2 are the periods in which agents face uncertainty, in each of them next
% period's S is drawn from pi_S(S_t,:). So S_t=S_initial up to T1, and S is frozen from T2+1 on. Prices depend on the
% history of S, so general eqm is solved on the TREE of histories (see UncertainTransPathTree for the tree, the timing
% of pi_S, and the forms PricePath and ParamPath can take). Everything is held at the final stationary eqm of its S in
% period T.
%
% INPUTS (as for TransitionPath_Case1_FHorz, and as MultipleRevealTransitionPath_Case1_FHorz for the final eqms)
%   PricePathShaper.(price) is 1-by-T: the initial guess at a node in period t is p_initial+(p_final(S)-p_initial)*shaper(t),
%     with p_initial the value in Parameters (the eqm before the path) and p_final(S) the final stationary eqm of the
%     node's S. transpathoptions.PricePathTree0 (any form UncertainTransPathTree accepts) overrides it, e.g. the
%     PricePathTree of an earlier solve on the same tree. Prices are scalars (one value per node).
%   ParamPath.(name) is N_S-by-T for a scalar that depends on S, 1-by-T for one that does not, N_j-by-T for one that
%     depends on age, and ParamPath.(name).S1, .S2, ... (each any of those forms) for one that depends on age and S
%     (see ValueFnOnUncertainTransPath_FHorz).
%   AgentDist_initial is the agent distribution in period 1, [n_a,n_z,N_j].
%   jequaloneDist, the newborns, as for AgentDistOnUncertainTransPath_FHorz: [n_a,n_z], [n_a,n_z,T], or per S as
%     jequaloneDist.S1, .S2, ... The age weights (AgeWeightsParamNames) can be on the ParamPath, per S.
%   GeneralEqmEqns are those of the final stationary eqms; GeneralEqmEqns_Transition those of the path.
%   heteroagentoptions, simoptions_finaleqm and vfoptions_finaleqm are for the final stationary eqms.
% The final stationary eqm of each S that a history can be in at period T is solved here, with
% HeteroAgentStationaryEqm_Case1_FHorz, at that S's period-T parameters and newborns; its value fn (with vfoptions, the
% path options) is the V_final of that S.
%
% ALGORITHM, as in TransitionPath_Case1_FHorz:
%   GEnewprice=3, shooting (with GEnewprice3.howtoupdate). Each iteration evaluates the whole tree at the current
%     prices (ValueFnOnUncertainTransPath_FHorz, AgentDistOnUncertainTransPath_FHorz,
%     EvalFnOnUncertainTransPath_AggVars_FHorz), then updates the prices at every node from the general eqm
%     conditions at that node, exactly as the deterministic shooting algorithm does in each period. Converged when,
%     over every node before period T, the prices move by less than toleranceGEprices and the general eqm conditions
%     are within toleranceGEcondns (the same norms as TransitionPath_Case1_FHorz).
%
% OUTPUTS
%   PricePathTree.(price) is 1-by-nNodes, the prices on the tree (at convergence, the prices the summary was evaluated at).
%   UncertainTPathSummary: .Tree (from UncertainTransPathTree), .GEcondnTree.(eqn) (1-by-nNodes), .AggVarsTree,
%     .VPathTree, .PolicyPathTree, .AgentDistPathTree, .finaleqm.S<k> (p_eqm, GeneralEqmCondnValues, V_final,
%     Policy_final), .iterations, .PricePathDist, .GEcondnDist.
%   The prices of history h are PricePathTree.(price)(UncertainTPathSummary.Tree.histories(h,:)).
%
% Not yet: Anderson acceleration (GEnewprice=2), price constraints, _tplus1 and _tminus1, intermediateEqns, a
% discount factor read at the child (_tplus1 in DiscountFactorParamNames), local search, transpathoptions.fastOLG=2,
% stockvars, semi-exogenous states, experience assets, exoticpreferences, permanent types.

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
    error('UncertainTransitionPath_FHorz: transpathoptions.GEnewprice=2 (Anderson acceleration) is not yet implemented, use =3 (the shooting algorithm)')
elseif transpathoptions.GEnewprice~=3
    error('transpathoptions.GEnewprice must be 3 (the shooting algorithm) for UncertainTransitionPath_FHorz')
end
if ~isfield(transpathoptions,'GEnewprice3') || ~isfield(transpathoptions.GEnewprice3,'howtoupdate')
    error('transpathoptions.GEnewprice=3 (the shooting algorithm) requires transpathoptions.GEnewprice3.howtoupdate')
end
if ~isfield(transpathoptions,'oldpathweight')
    transpathoptions.oldpathweight=0; % as TransitionPath_Case1_FHorz under GEnewprice=3: the step is in howtoupdate
end
if ~isfield(transpathoptions,'maxiter')
    transpathoptions.maxiter=1000;
end
if ~isfield(transpathoptions,'verbose')
    transpathoptions.verbose=0;
end
if ~isfield(transpathoptions,'toleranceGEprices')
    transpathoptions.toleranceGEprices=Inf; % off by default, as in UncertainTransitionPath_InfHorz
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
if ~isfield(transpathoptions,'fastOLG')
    transpathoptions.fastOLG=0;
end
% Not yet implemented for uncertain FHorz transition paths
if transpathoptions.fastOLG==2
    error('UncertainTransitionPath_FHorz: transpathoptions.fastOLG=2 is not yet implemented (it needs local search), use =0 or =1')
end
if isfield(vfoptions,'localsearch') && vfoptions.localsearch~=0
    error('UncertainTransitionPath_FHorz: vfoptions.localsearch is not yet implemented')
end
if isfield(transpathoptions,'stockvars') && ~isempty(transpathoptions.stockvars)
    error('UncertainTransitionPath_FHorz: transpathoptions.stockvars is not yet implemented for uncertain transition paths')
end
if isfield(transpathoptions,'intermediateEqns')
    error('UncertainTransitionPath_FHorz: transpathoptions.intermediateEqns is not yet implemented')
end
if (isfield(transpathoptions,'constrainpositive') && ~isempty(transpathoptions.constrainpositive)) || (isfield(transpathoptions,'constrain0to1') && ~isempty(transpathoptions.constrain0to1)) || (isfield(transpathoptions,'constrainAtoB') && ~isempty(transpathoptions.constrainAtoB))
    error('UncertainTransitionPath_FHorz: constraints on the prices (transpathoptions.constrainpositive, constrain0to1, constrainAtoB) are not yet implemented')
end
if isfield(vfoptions,'n_semiz') && prod(vfoptions.n_semiz)>0
    error('UncertainTransitionPath_FHorz: semi-exogenous states (vfoptions.n_semiz) are not yet implemented for uncertain transition paths')
end
if (isfield(vfoptions,'experienceasset') && vfoptions.experienceasset>=1) || (isfield(vfoptions,'experienceassetz') && vfoptions.experienceassetz>=1)
    error('UncertainTransitionPath_FHorz: experience assets are not yet implemented for uncertain transition paths')
end
if isfield(vfoptions,'exoticpreferences') && ~strcmp(vfoptions.exoticpreferences,'None')
    error('UncertainTransitionPath_FHorz: vfoptions.exoticpreferences is not yet implemented for uncertain transition paths')
end
GEeqnNames=fieldnames(GeneralEqmEqns_Transition);
nGEeqns=length(GEeqnNames);
GeneralEqmEqnsCell=cell(1,nGEeqns);
for gg=1:nGEeqns
    GeneralEqmEqnParamNames(gg).Names=getAnonymousFnInputNames(GeneralEqmEqns_Transition.(GEeqnNames{gg}));
    GeneralEqmEqnsCell{gg}=GeneralEqmEqns_Transition.(GEeqnNames{gg});
    if any(endsWith(GeneralEqmEqnParamNames(gg).Names,'_tplus1')) || any(endsWith(GeneralEqmEqnParamNames(gg).Names,'_tminus1'))
        error('UncertainTransitionPath_FHorz: GeneralEqmEqns_Transition.%s uses a _tplus1 or _tminus1 input, which is not yet implemented',GEeqnNames{gg})
    end
end
FnNames=fieldnames(FnsToEvaluate);
for ff=1:length(FnNames)
    temp=getAnonymousFnInputNames(FnsToEvaluate.(FnNames{ff}));
    if any(endsWith(temp,'_tplus1')) || any(endsWith(temp,'_tminus1'))
        error('UncertainTransitionPath_FHorz: FnsToEvaluate.%s uses a _tplus1 or _tminus1 input, which is not yet implemented',FnNames{ff})
    end
end
if ischar(DiscountFactorParamNames)
    DiscountFactorParamNames={DiscountFactorParamNames};
end
for DFparam_c=1:length(DiscountFactorParamNames)
    if endsWith(DiscountFactorParamNames{DFparam_c},'_tplus1')
        error('UncertainTransitionPath_FHorz: a discount factor read at the child (%s in DiscountFactorParamNames) is not yet implemented',DiscountFactorParamNames{DFparam_c})
    end
end

%% The tree (prices are put on it once the final eqms are known); parameters can be vectors (age-dependent)
transpathoptions.uncertaintreevectorparams=1;
[Tree,~,ParamPathNodes]=UncertainTransPathTree(T,T1,T2,pi_S,S_initial,[],ParamPath,transpathoptions);
nNodes=Tree.nNodes;
ParamPathNames=fieldnames(ParamPathNodes);
PricePathNames=fieldnames(PricePathShaper);
nPrices=length(PricePathNames);
for pp=1:nPrices
    if numel(Parameters.(PricePathNames{pp}))~=1
        error('UncertainTransitionPath_FHorz: Parameters.%s, a price, is not a scalar; prices that depend on age are not implemented for uncertain transition paths',PricePathNames{pp})
    end
end
transpathoptions=setupGEnewprice3_shooting(transpathoptions,GeneralEqmEqns_Transition,PricePathNames);

%% The final stationary eqm of each S that a history can be in at period T
finalS=unique(Tree.nodeS(Tree.nodedate==T))';
p_initial=zeros(1,nPrices);
for pp=1:nPrices
    p_initial(pp)=Parameters.(PricePathNames{pp});
end
% The newborns of the final eqm of S: its period T jequaloneDist, in the shape of AgentDist_initial without the age
Distshape=size(AgentDist_initial);
jeqshape=[Distshape(1:end-1),1];
NJ1=prod(Distshape(1:end-1));
p_final=nan(Tree.N_S,nPrices); % row S, for the S that have a final eqm
V_final=struct();
Policy_final=struct();
for SS=finalS
    Sstr=['S',num2str(SS)];
    Parameters_S=Parameters;
    % the period T parameters of a history in S (a row, as the deterministic commands hold an age-dependent parameter)
    kk=find(Tree.nodedate==T & Tree.nodeS==SS,1);
    for pp=1:length(ParamPathNames)
        Parameters_S.(ParamPathNames{pp})=ParamPathNodes.(ParamPathNames{pp})(:,kk)';
    end
    if isstruct(jequaloneDist)
        if ~isfield(jequaloneDist,Sstr)
            error('UncertainTransitionPath_FHorz: jequaloneDist is given per S, but has no field %s (some history is in S=%i at period T)',Sstr,SS)
        end
        temp=jequaloneDist.(Sstr);
    else
        temp=jequaloneDist;
    end
    if numel(temp)==NJ1
        jequaloneDist_S=reshape(temp,jeqshape);
    elseif numel(temp)==NJ1*T
        temp=reshape(temp,[NJ1,T]);
        jequaloneDist_S=reshape(temp(:,T),jeqshape);
    else
        error('UncertainTransitionPath_FHorz: jequaloneDist must be [n_a,n_z] (the same every period), [n_a,n_z,T], or given per S as jequaloneDist.S1, .S2, ..., each of those')
    end
    if transpathoptions.verbose>=1
        fprintf('UncertainTransitionPath_FHorz: solving the final stationary eqm of S=%i \n',SS)
    end
    [p_eqm_S,~,GEcondns_S]=HeteroAgentStationaryEqm_Case1_FHorz(jequaloneDist_S,AgeWeightsParamNames,n_d, n_a, n_z, N_j, 0, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters_S, DiscountFactorParamNames, [], [], [], PricePathNames, heteroagentoptions, simoptions_finaleqm, vfoptions_finaleqm);
    for pp=1:nPrices
        Parameters_S.(PricePathNames{pp})=p_eqm_S.(PricePathNames{pp});
        p_final(SS,pp)=p_eqm_S.(PricePathNames{pp});
    end
    [V_final.(Sstr),Policy_final.(Sstr)]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,pi_z,ReturnFn,Parameters_S,DiscountFactorParamNames,[],vfoptions);
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
            error('UncertainTransitionPath_FHorz: PricePathShaper.%s must be 1-by-T',PricePathNames{pp})
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
nNL=sum(notlast);
while itercounter<=transpathoptions.maxiter
    % The whole tree at the current prices, and the general eqm conditions at every node
    [~,GEcondnTree,AggVarsTree,VPathTree,PolicyPathTree,AgentDistPathTree]=UncertainTransitionPath_FHorz_evaltree(reshape(PricePathOld,[],1), nNodes, nPrices, PricePathNames, notlast, ParamPath, ParamPathNodes, ParamPathNames, T, T1, T2, pi_S, S_initial, V_final, Policy_final, AgentDist_initial, jequaloneDist, n_d,n_a,n_z,N_j, d_grid,a_grid,z_grid, pi_z, ReturnFn, FnsToEvaluate, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeightsParamNames, transpathoptions, simoptions, vfoptions);

    % The shooting update of the nodes before period T, from the general eqm conditions at each of them. It is the
    % deterministic one (updatePricePathNew_TPath_T, a per-period formula, so the order of the rows does not matter),
    % applied to those nodes as if they were the periods 1,...,T-1 of a path; the extra row given to it is the 'period
    % T' it passes through, and is dropped.
    PricePathNew=PricePathOld;
    Pnotlastnew=updatePricePathNew_TPath_T(GEcondnTree(notlast,:),[PricePathOld(notlast,:); zeros(1,nPrices)],nNL+1,itercounter,transpathoptions);
    PricePathNew(notlast,:)=Pnotlastnew(1:nNL,:);

    if any(~isfinite(GEcondnTree(notlast,:)),'all')
        error('UncertainTransitionPath_FHorz: the general eqm conditions are not finite at some node (iteration %i)',itercounter)
    end

    % Distances, over the nodes before period T, in the norms TransitionPath_Case1_FHorz uses
    PricePathDist=max(abs(PricePathNew(notlast,:)-PricePathOld(notlast,:)),[],'all');
    if transpathoptions.multiGEcriterion==0
        GEcondnDist=max(sum(abs(transpathoptions.multiGEweights.*GEcondnTree(notlast,:)),2));
    elseif transpathoptions.multiGEcriterion==1
        GEcondnDist=max(sqrt(sum(transpathoptions.multiGEweights.*(GEcondnTree(notlast,:).^2),2)));
    end
    if transpathoptions.verbose>=1
        fprintf('UncertainTransitionPath_FHorz: iteration %i, price distance %8.6f, general eqm conditions %8.6f \n',itercounter,PricePathDist,GEcondnDist)
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
    warning('UncertainTransitionPath_FHorz: reached maxiter (%i) without convergence; the general eqm conditions are %8.6f from zero, against toleranceGEcondns=%g',transpathoptions.maxiter,GEcondnDist,transpathoptions.toleranceGEcondns)
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
