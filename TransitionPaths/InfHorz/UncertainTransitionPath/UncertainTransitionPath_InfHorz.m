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
% ALGORITHMS, as in TransitionPath_InfHorz:
%   GEnewprice=3, shooting (with GEnewprice3.howtoupdate). Each iteration evaluates the whole tree at the current
%     prices (ValueFnOnUncertainTransPath_InfHorz, AgentDistOnUncertainTransPath_InfHorz,
%     EvalFnOnUncertainTransPath_AggVars_InfHorz), then updates the prices at every node from the general eqm
%     conditions at that node, exactly as the deterministic shooting algorithm does in each period. Converged when,
%     over every node before period T, the prices move by less than toleranceGEprices and the general eqm conditions
%     are within toleranceGEcondns (the same norms as TransitionPath_InfHorz).
%   GEnewprice=2, Anderson acceleration (with GEnewprice2.howtoupdate, and the options in transpathoptions.anderson)
%     of that same shooting map, with the prices of every node stacked into one vector, by AndersonAcceleration (as
%     TransitionPath_InfHorz_Anderson). Converged when the general eqm conditions are within toleranceGEcondns.
% PRICE CONSTRAINTS (transpathoptions.constrainpositive, constrain0to1, constrainAtoB with constrainAtoBlimits, and
% constraintstepfraction), as in TransitionPath_InfHorz: the same bounds at every node, and an update that would take a
% price to or past its bound moves only constraintstepfraction of the way there. The initial guess (every node,
% including period T, the final eqms) must be strictly inside the bounds.
%
% OUTPUTS
%   PricePathTree.(price) is 1-by-nNodes, the prices on the tree (at convergence, the prices the summary was evaluated at).
%   UncertainTPathSummary: .Tree (from UncertainTransPathTree), .GEcondnTree.(eqn) (1-by-nNodes), .AggVarsTree,
%     .VPathTree, .PolicyPathTree, .AgentDistPathTree, .finaleqm.S<k> (p_eqm, GeneralEqmCondnValues, V_final,
%     Policy_final), .iterations, .PricePathDist, .GEcondnDist.
%   The prices of history h are PricePathTree.(price)(UncertainTPathSummary.Tree.histories(h,:)).
%
% Not yet: intermediateEqns, _tminus1/_tplus1 in the general eqm conditions, local search, permanent types.

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
    error('transpathoptions.GEnewprice must be set: =3 for the shooting algorithm, =2 for Anderson acceleration')
elseif transpathoptions.GEnewprice==1
    error('transpathoptions.GEnewprice=1 (quasi-Newton) is not implemented for uncertain transition paths, use =3 (the shooting algorithm) or =2 (Anderson acceleration)')
elseif transpathoptions.GEnewprice~=2 && transpathoptions.GEnewprice~=3
    error('transpathoptions.GEnewprice must be 3 (the shooting algorithm) or 2 (Anderson acceleration) for uncertain transition paths')
end
if transpathoptions.GEnewprice==3 && (~isfield(transpathoptions,'GEnewprice3') || ~isfield(transpathoptions.GEnewprice3,'howtoupdate'))
    error('transpathoptions.GEnewprice=3 (the shooting algorithm) requires transpathoptions.GEnewprice3.howtoupdate')
end
if transpathoptions.GEnewprice==2 && (~isfield(transpathoptions,'GEnewprice2') || ~isfield(transpathoptions.GEnewprice2,'howtoupdate'))
    % Anderson acceleration accelerates the shooting map, so it needs the same instructions the shooting algorithm does
    error('transpathoptions.GEnewprice=2 (Anderson acceleration) requires transpathoptions.GEnewprice2.howtoupdate (same format as GEnewprice3.howtoupdate)')
end
if ~isfield(transpathoptions,'anderson')
    transpathoptions.anderson=struct(); % GEnewprice=2 only: the AndersonAcceleration options; defaults are set there, except safeguard
end
if ~isfield(transpathoptions.anderson,'safeguard')
    transpathoptions.anderson.safeguard=0; % as TransitionPath_InfHorz
end
if ~isfield(transpathoptions,'constrainpositive')
    transpathoptions.constrainpositive={};
end
if ~isfield(transpathoptions,'constrain0to1')
    transpathoptions.constrain0to1={};
end
if ~isfield(transpathoptions,'constrainAtoB')
    transpathoptions.constrainAtoB={};
end
if ~isfield(transpathoptions,'constraintstepfraction')
    transpathoptions.constraintstepfraction=0.5; % as TransitionPath_InfHorz for shooting and Anderson
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

%% Constraints on the prices, as in TransitionPath_InfHorz (see there): one bound per price, the same at every node
pricelowerbound=-Inf(1,nPrices);
priceupperbound=Inf(1,nPrices);
useconstraints=0;
constrainednames={};
for cc=1:3
    if cc==1
        connames=transpathoptions.constrainpositive;
        conname='constrainpositive';
    elseif cc==2
        connames=transpathoptions.constrain0to1;
        conname='constrain0to1';
    else
        connames=transpathoptions.constrainAtoB;
        conname='constrainAtoB';
    end
    if ischar(connames)
        connames={connames};
    end
    for nn=1:length(connames)
        pp=find(strcmp(connames{nn},PricePathNames));
        if isempty(pp)
            error('transpathoptions.%s contains ''%s'', which is not a price in PricePathShaper (only prices can be constrained on a transition path)',conname,connames{nn})
        end
        if any(strcmp(connames{nn},constrainednames))
            error('The price ''%s'' appears in more than one of transpathoptions.constrainpositive, constrain0to1 and constrainAtoB',connames{nn})
        end
        constrainednames=[constrainednames,connames(nn)];
        if cc==1
            lb=0; ub=Inf;
        elseif cc==2
            lb=0; ub=1;
        else
            if ~isfield(transpathoptions,'constrainAtoBlimits') || ~isfield(transpathoptions.constrainAtoBlimits,connames{nn})
                error('transpathoptions.constrainAtoB contains ''%s'', so you must set transpathoptions.constrainAtoBlimits.%s=[A,B]',connames{nn},connames{nn})
            end
            temp=transpathoptions.constrainAtoBlimits.(connames{nn});
            if numel(temp)~=2 || temp(2)<=temp(1)
                error('transpathoptions.constrainAtoBlimits.%s must be [A,B] with A<B',connames{nn})
            end
            lb=temp(1); ub=temp(2);
        end
        pricelowerbound(pp)=lb;
        priceupperbound(pp)=ub;
        useconstraints=1;
    end
end
if useconstraints==1
    stepfractionvec=0.5*ones(1,nPrices);
    if isstruct(transpathoptions.constraintstepfraction)
        temp=fieldnames(transpathoptions.constraintstepfraction);
        for nn=1:length(temp)
            if ~any(strcmp(temp{nn},constrainednames))
                error('transpathoptions.constraintstepfraction.%s is set, but ''%s'' is not a constrained price (it is not in constrainpositive, constrain0to1 or constrainAtoB)',temp{nn},temp{nn})
            end
            stepfractionvec(strcmp(temp{nn},PricePathNames))=transpathoptions.constraintstepfraction.(temp{nn});
        end
    elseif isscalar(transpathoptions.constraintstepfraction)
        stepfractionvec(:)=transpathoptions.constraintstepfraction;
    else
        error('transpathoptions.constraintstepfraction must be a scalar, or a struct with one field per constrained price (e.g. transpathoptions.constraintstepfraction.r=0.25)')
    end
    if ~all(stepfractionvec>0 & stepfractionvec<1)
        error('transpathoptions.constraintstepfraction must be strictly between 0 and 1 (for every price it is set for)')
    end
    % The initial guess has to be strictly inside the bounds at every node, including period T (the final eqms, which
    % are never updated), else the rule has no interior point to step from
    for pp=1:nPrices
        if any(PricePathOld(:,pp)<=pricelowerbound(pp)) || any(PricePathOld(:,pp)>=priceupperbound(pp))
            error('UncertainTransitionPath_InfHorz: the initial guess for ''%s'' is not strictly inside its constraint (bounds [%g,%g]) at every node (including the final eqms in period T)',PricePathNames{pp},pricelowerbound(pp),priceupperbound(pp))
        end
    end
end
nconstraintiters=0;
nconstraintcuts=0;

if transpathoptions.GEnewprice==3
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
    PricePathProposed=oldw.*PricePathOld(notlast,:)+(1-oldw).*PricePathNew(notlast,:);
    if useconstraints==1
        % Fraction-to-the-boundary, as in TransitionPath_InfHorz_shooting: any (node,price) that the update would take to
        % or past its bound moves only constraintstepfraction of the way from its current value to that bound. By
        % logical indexing, never by multiplying by an indicator (an unconstrained price has bounds -Inf and Inf).
        nNL=sum(notlast);
        Theta=repmat(stepfractionvec,nNL,1);
        LB=repmat(pricelowerbound,nNL,1);
        UB=repmat(priceupperbound,nNL,1);
        Pold=PricePathOld(notlast,:);
        belowlb=(PricePathProposed<=LB);
        aboveub=(PricePathProposed>=UB);
        PricePathProposed(belowlb)=Pold(belowlb)+Theta(belowlb).*(LB(belowlb)-Pold(belowlb));
        PricePathProposed(aboveub)=Pold(aboveub)+Theta(aboveub).*(UB(aboveub)-Pold(aboveub));
        nconstraintcuts=sum(belowlb(:))+sum(aboveub(:));
        if nconstraintcuts>0
            nconstraintiters=nconstraintiters+1;
        end
    end
    PricePathOld(notlast,:)=PricePathProposed;
    itercounter=itercounter+1;
end
if PricePathDist>transpathoptions.toleranceGEprices || GEcondnDist>transpathoptions.toleranceGEcondns
    warning('UncertainTransitionPath_InfHorz: reached maxiter (%i) without convergence; the general eqm conditions are %8.6f from zero, against toleranceGEcondns=%g',transpathoptions.maxiter,GEcondnDist,transpathoptions.toleranceGEcondns)
end
if useconstraints==1
    fprintf('Constraints on the prices: cut back %i of the %i updates \n',nconstraintiters,itercounter-1)
    if nconstraintcuts>0
        warning('UncertainTransitionPath_InfHorz: the constraints on the prices cut back the last update (%i elements), so the prices may be pressing against a bound',nconstraintcuts)
    end
end

elseif transpathoptions.GEnewprice==2
%% Anderson acceleration of the shooting map, on the prices of every node stacked into one vector
% p is the nNodes-by-nPrices prices, as a column. The general eqm conditions are those of the nodes before period T,
% and the shooting map is the deterministic one (updatePricePathNew_TPath_T) applied to those nodes, with the period T
% nodes (the final eqms) passed through unchanged, as the deterministic shooting map passes period T through.
nNL=sum(notlast);
GEcondnsFn=@(p) UncertainTransitionPath_InfHorz_evaltree(p, nNodes, nPrices, PricePathNames, notlast, ParamPath, ParamPathNodes, ParamPathNames, T, T1, T2, pi_S, S_initial, V_final, Policy_final, AgentDist_initial, n_d,n_a,n_z, d_grid,a_grid,z_grid, pi_z, ReturnFn, FnsToEvaluate, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, transpathoptions, simoptions, vfoptions);
ShootingMapFn=@(p,GEc) UncertainTransitionPath_InfHorz_shootingmap(p,GEc,nNodes,nPrices,nGEeqns,notlast,transpathoptions);
if transpathoptions.multiGEcriterion==0
    DistanceFn=@(GEc) max(sum(abs(transpathoptions.multiGEweights.*reshape(GEc,nNL,nGEeqns)),2));
elseif transpathoptions.multiGEcriterion==1
    DistanceFn=@(GEc) max(sqrt(sum(transpathoptions.multiGEweights.*(reshape(GEc,nNL,nGEeqns).^2),2)));
end
andersonoptions=transpathoptions.anderson;
andersonoptions.verbose=transpathoptions.verbose;
if ~isfield(andersonoptions,'maxiter')
    andersonoptions.maxiter=transpathoptions.maxiter; % so that transpathoptions.maxiter means what it does for every other algorithm
end
if useconstraints==1
    andersonoptions.lowerbound=reshape(repmat(pricelowerbound,nNodes,1),[],1);
    andersonoptions.upperbound=reshape(repmat(priceupperbound,nNodes,1),[],1);
    andersonoptions.stepfraction=reshape(repmat(stepfractionvec,nNodes,1),[],1);
end
[p,GEcondns,andersonoutput]=AndersonAcceleration(GEcondnsFn,ShootingMapFn,DistanceFn,reshape(PricePathOld,[],1),transpathoptions.toleranceGEcondns,andersonoptions);
PricePathOld=reshape(gather(p),[nNodes,nPrices]);
GEcondnDist=DistanceFn(GEcondns);
PricePathDist=NaN; % Anderson is converged on the general eqm conditions only
itercounter=andersonoutput.iterations;
if useconstraints==1
    fprintf('Constraints on the prices: over %i iterations, cut back the shooting step %i time(s) (%i elements in all), and %i Anderson step(s) left the bounds and were not taken \n',andersonoutput.iterations,andersonoutput.nconstraintiters,andersonoutput.nconstraintcuts,andersonoutput.ninfeasiblesteps)
end
if GEcondnDist>transpathoptions.toleranceGEcondns
    warning('UncertainTransitionPath_InfHorz: Anderson acceleration did not converge; the general eqm conditions are %8.6f from zero, against toleranceGEcondns=%g',GEcondnDist,transpathoptions.toleranceGEcondns)
end
% The tree at the returned prices, for the summary (one more evaluation; Anderson keeps only the conditions)
[~,GEcondnTree,AggVarsTree,VPathTree,PolicyPathTree,AgentDistPathTree]=GEcondnsFn(reshape(PricePathOld,[],1));
UncertainTPathSummary.andersonoutput=andersonoutput;
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
