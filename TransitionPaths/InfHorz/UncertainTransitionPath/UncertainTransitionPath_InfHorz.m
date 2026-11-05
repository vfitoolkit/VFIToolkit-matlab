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
%     .AggVarsGivenChildTree (see EvalFnOnUncertainTransPath_AggVars_InfHorz), .VPathTree, .PolicyPathTree, .AgentDistPathTree, .finaleqm.S<k> (p_eqm, GeneralEqmCondnValues, V_final,
%     Policy_final), .iterations, .PricePathDist, .GEcondnDist.
%   The prices of history h are PricePathTree.(price)(UncertainTPathSummary.Tree.histories(h,:)).
%
% _tplus1 AND _tminus1, in GeneralEqmEqns_Transition and FnsToEvaluate (as in TransitionPath_InfHorz: _tplus1 of
% prices, _tminus1 of prices, parameters and aggregates, with transpathoptions.initialvalues for period 1).
%   _tminus1 is the value at the parent node.
%   _tplus1 at a node with one child is the child's price, and at a period T node the node's own (the final eqm). At a
%     branching node next period's price is not yet known, and a general eqm condition that uses _tplus1 holds IN
%     EXPECTATION: it is evaluated with the prices of each child c, and the condition at the node is
%     sum_c Tree.nodecondprob(c)*condition_c. A FnsToEvaluate that uses _tplus1 is the same: its aggregate at the node
%     given child c is UncertainTPathSummary.AggVarsGivenChildTree.(name)(c), a general eqm condition takes that value
%     in condition_c, and AggVarsTree.(name).Mean is the expectation over the children.
%   The expectation is of the condition exactly as written: q-(d_tplus1+q_tplus1)/(1+r_tplus1) and
%   q*(1+r_tplus1)-(d_tplus1+q_tplus1) are the same condition on a deterministic path, but not in expectation.
%   Not implemented: _tminus1 of an aggregate whose FnsToEvaluate uses _tplus1 (its value at a branching node depends
%   on the child), and transpathoptions.stockvars.
%
% _tplus1 IN DiscountFactorParamNames: a stochastic discount factor (see ValueFnOnUncertainTransPath_InfHorz).
%   'name' is read at the node, 'name_tplus1' at the next node, where name is a price or a ParamPath parameter. E.g. a
%   firm owned by a representative household, with price m and general eqm condition m-beta*C_tminus1/C on the path,
%   uses DiscountFactorParamNames={'betafirm','m_tplus1'}: at a branching node the next period value fn is
%   sum_c nodecondprob(c)*m_c*V_c. In the final stationary eqms S is frozen, so the next node's value is the node's
%   own, and they are solved with the _tplus1 removed ({'betafirm','m'}), with m from GeneralEqmEqns (e.g. m-beta).
%
% INTERMEDIATE EQNS (transpathoptions.intermediateEqns), as in TransitionPath_InfHorz: at each node, in order, after
%   the aggregates and before the general eqm conditions, and usable by later intermediate eqns and by the conditions.
%   One that uses _tplus1 (or a FnsToEvaluate that uses _tplus1, or an earlier one that does) depends on the child at a
%   branching node, so it is evaluated once per child c with c's values, and a condition that uses it is one that holds
%   in expectation (above). Intermediate eqns of the final stationary eqms go in heteroagentoptions.intermediateEqns.
%
% LOCAL SEARCH (vfoptions.localsearch=1, with vfoptions.nlocalsearch, nlocalsearchup, nlocalsearchdown), as in
%   TransitionPath_InfHorz, for both shooting and Anderson, with a reference policy per node: the first sweep is
%   standard, later ones restrict the aprime search at every node to a window around the previous answer there, and a
%   converged restricted solve is certified by one standard sweep at the same prices (resuming if that moves it). So
%   the answer, and everything in the summary, is from a standard sweep. UncertainTPathSummary.localsearch reports
%   the counts.
%
% Not yet: stockvars, permanent types.

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
% LOCAL SEARCH, as in TransitionPath_InfHorz (see TransitionPath_InfHorz_shooting for the scheme), with a reference per
% node: only 0 and 1 are meaningful here (2 is ValueFnOnTransPath_InfHorz's 'force it'), and the checks are the same
if vfoptions.localsearch~=0 && vfoptions.localsearch~=1
    error('vfoptions.localsearch must be 0 or 1 here; the value 2 is only meaningful to ValueFnOnTransPath_InfHorz and ValueFnOnUncertainTransPath_InfHorz, which use it to force a restricted search they would otherwise ignore')
elseif vfoptions.localsearch==1
    if ~isfield(vfoptions,'nlocalsearch')
        error('When using vfoptions.localsearch=1 you must set vfoptions.nlocalsearch')
    end
    if vfoptions.nlocalsearch<1
        error('vfoptions.nlocalsearch must be at least 1')
    end
    % nlocalsearch is the STARTING window, ratcheted as in TransitionPath_InfHorz_shooting
    if ~isfield(vfoptions,'nlocalsearchup')
        vfoptions.nlocalsearchup=2;
    end
    if ~isfield(vfoptions,'nlocalsearchdown')
        vfoptions.nlocalsearchdown=1;
    end
    if vfoptions.nlocalsearchup<1 || vfoptions.nlocalsearchdown<1
        error('vfoptions.nlocalsearchup and vfoptions.nlocalsearchdown must both be at least 1')
    end
    if isfield(vfoptions,'n_semiz') && prod(vfoptions.n_semiz)>0
        error('vfoptions.localsearch=1 is not implemented with a semi-exogenous state')
    end
    if isfield(vfoptions,'experienceasset') && vfoptions.experienceasset>=1
        error('vfoptions.localsearch=1 is not implemented with an experience asset')
    end
    if n_a(1)<2*vfoptions.nlocalsearch+1
        error('vfoptions.nlocalsearch is too large for the grid: need n_a>=2*nlocalsearch+1')
    end
end
if isfield(transpathoptions,'stockvars') && ~isempty(transpathoptions.stockvars)
    error('UncertainTransitionPath_InfHorz: transpathoptions.stockvars is not yet implemented for uncertain transition paths')
end
GEeqnNames=fieldnames(GeneralEqmEqns_Transition);
nGEeqns=length(GEeqnNames);
GeneralEqmEqnsCell=cell(1,nGEeqns);
for gg=1:nGEeqns
    GeneralEqmEqnParamNames(gg).Names=getAnonymousFnInputNames(GeneralEqmEqns_Transition.(GEeqnNames{gg}));
    GeneralEqmEqnsCell{gg}=GeneralEqmEqns_Transition.(GEeqnNames{gg});
end
% Intermediate eqns, as in TransitionPath_InfHorz: evaluated at each node, in order, after the aggregates and before
% the general eqm conditions, and their values can be used by later intermediate eqns and by the conditions
transpathoptions.useintermediateEqns=0;
if isfield(transpathoptions,'intermediateEqns')
    transpathoptions.useintermediateEqns=1;
    intEqnNames=fieldnames(transpathoptions.intermediateEqns);
    nIntEqns=length(intEqnNames);
    transpathoptions.intermediateEqnsCell=cell(1,nIntEqns);
    for gg=1:nIntEqns
        transpathoptions.intermediateEqnParamNames(gg).Names=getAnonymousFnInputNames(transpathoptions.intermediateEqns.(intEqnNames{gg}));
        transpathoptions.intermediateEqnsCell{gg}=transpathoptions.intermediateEqns.(intEqnNames{gg});
    end
end

%% The tree (prices are put on it once the final eqms are known)
[Tree,~,ParamPathNodes]=UncertainTransPathTree(T,T1,T2,pi_S,S_initial,[],ParamPath,transpathoptions);
nNodes=Tree.nNodes;
ParamPathNames=fieldnames(ParamPathNodes);
PricePathNames=fieldnames(PricePathShaper);
nPrices=length(PricePathNames);
transpathoptions=setupGEnewprice3_shooting(transpathoptions,GeneralEqmEqns_Transition,PricePathNames);

%% _tplus1 and _tminus1
% _tplus1 (prices only): at a node with one child it is the child's price, and at a period T node the node's own price
% (the final eqm), exactly as on a deterministic path. At a branching node next period's price is not known, and a
% general eqm condition that uses _tplus1 is required to hold in expectation over the children: it is evaluated once for
% each child c, with c's prices, and GEcondn=sum_c nodecondprob(c)*GEcondn_c. A FnsToEvaluate that uses _tplus1 is
% treated the same way (its aggregate given child c is in AggVarsGivenChildTree, its AggVarsTree is the expectation),
% and a general eqm condition that uses such an aggregate takes the value given child c in GEcondn_c.
% _tminus1 (prices, parameters, aggregates): the value at the parent node, and transpathoptions.initialvalues at the
% period 1 node.
FnNames=fieldnames(FnsToEvaluate);
tplus1FnNames={}; % the FnsToEvaluate that use a _tplus1 input
usetminus1=0;
for ff=1:length(FnNames)
    temp=getAnonymousFnInputNames(FnsToEvaluate.(FnNames{ff}));
    if any(endsWith(temp,'_tplus1'))
        tplus1FnNames=[tplus1FnNames,FnNames(ff)];
    end
    if any(endsWith(temp,'_tminus1'))
        usetminus1=1;
    end
end
% An intermediate eqn that uses _tplus1, a FnsToEvaluate that uses _tplus1, or an earlier intermediate eqn that does,
% depends on the child at a branching node, and is evaluated once per child (see UncertainTransitionPath_InfHorz_evaltree)
tplus1IntEqnNames={};
if transpathoptions.useintermediateEqns==1
    transpathoptions.intermediateEqnsusetplus1=zeros(1,nIntEqns);
    for gg=1:nIntEqns
        temp=transpathoptions.intermediateEqnParamNames(gg).Names;
        if any(endsWith(temp,'_tplus1')) || any(ismember(temp,tplus1FnNames)) || any(ismember(temp,tplus1IntEqnNames))
            transpathoptions.intermediateEqnsusetplus1(gg)=1;
            tplus1IntEqnNames=[tplus1IntEqnNames,intEqnNames(gg)];
        end
        if any(endsWith(temp,'_tminus1'))
            usetminus1=1;
        end
    end
end
GEusestplus1=false(1,nGEeqns); % the general eqm conditions that hold in expectation at a branching node
for gg=1:nGEeqns
    temp=GeneralEqmEqnParamNames(gg).Names;
    GEusestplus1(gg)=any(endsWith(temp,'_tplus1')) || any(ismember(temp,tplus1FnNames)) || any(ismember(temp,tplus1IntEqnNames));
    if any(endsWith(temp,'_tminus1'))
        usetminus1=1;
    end
end
if usetminus1==1 && ~isfield(transpathoptions,'initialvalues')
    error('UncertainTransitionPath_InfHorz: a _tminus1 input is used, so transpathoptions.initialvalues must be given (it supplies the period 1 values)')
end
[tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tminus1paramNames,tplus1pricePathkk]=inputsFindtplus1tminus1(FnsToEvaluate,GeneralEqmEqns_Transition,PricePathNames,ParamPathNames,{},transpathoptions);
temp=intersect(tminus1AggVarsNames,tplus1FnNames);
if ~isempty(temp)
    error('UncertainTransitionPath_InfHorz: %s_tminus1 is used, but FnsToEvaluate.%s uses _tplus1, so at a branching node its value depends on which child comes next and its _tminus1 is not defined; not implemented for uncertain transition paths',temp{1},temp{1})
end

%% Discount factors: 'name' read at the node, 'name_tplus1' at the next node (a stochastic discount factor)
% Checked here, before the final eqms are solved; ValueFnOnUncertainTransPath_InfHorz applies them.
if ischar(DiscountFactorParamNames)
    DiscountFactorParamNames={DiscountFactorParamNames};
end
DiscountFactorParamNames_finaleqm=DiscountFactorParamNames; % S is frozen in a final eqm: the next node's value is the node's own
for DFparam_c=1:length(DiscountFactorParamNames)
    DFname=DiscountFactorParamNames{DFparam_c};
    if endsWith(DFname,'_tplus1')
        DFname=DFname(1:end-7);
        DiscountFactorParamNames_finaleqm{DFparam_c}=DFname;
        if any(strcmp(DFname,DiscountFactorParamNames))
            error('UncertainTransitionPath_InfHorz: DiscountFactorParamNames contains both %s and %s_tplus1; %s would be applied twice, at the node and at the child',DFname,DFname,DFname)
        end
        if ~any(strcmp(DFname,PricePathNames)) && ~any(strcmp(DFname,ParamPathNames))
            error('UncertainTransitionPath_InfHorz: DiscountFactorParamNames contains %s_tplus1, but %s is not a price in PricePathShaper or a parameter in ParamPath (a discount factor read at the child must take a value at each node)',DFname,DFname)
        end
        if isfield(vfoptions,'exoticpreferences') && ~strcmp(vfoptions.exoticpreferences,'None')
            error('UncertainTransitionPath_InfHorz: a discount factor read at the child (_tplus1 in DiscountFactorParamNames) is not implemented with vfoptions.exoticpreferences')
        end
    elseif ~any(strcmp(DFname,PricePathNames)) && ~any(strcmp(DFname,ParamPathNames)) && numel(Parameters.(DFname))>1
        % The value fn takes the product of all the elements of the discount factors, so a vector would silently become
        % its product; a discount factor that depends on S is an N_S-by-T ParamPath
        error('UncertainTransitionPath_InfHorz: Parameters.%s, in DiscountFactorParamNames, is not a scalar (a discount factor that depends on S goes in ParamPath, N_S-by-T)',DFname)
    end
end

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
    [p_eqm_S,GEcondns_S]=HeteroAgentStationaryEqm_InfHorz(n_d, n_a, n_z, 0, pi_z, d_grid, a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters_S, DiscountFactorParamNames_finaleqm, [], [], [], PricePathNames, heteroagentoptions, simoptions_finaleqm, vfoptions_finaleqm);
    for pp=1:nPrices
        Parameters_S.(PricePathNames{pp})=p_eqm_S.(PricePathNames{pp});
        p_final(SS,pp)=p_eqm_S.(PricePathNames{pp});
    end
    [V_final.(Sstr),Policy_final.(Sstr)]=ValueFnIter_InfHorz(n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z,ReturnFn,Parameters_S,DiscountFactorParamNames_finaleqm,[],vfoptions);
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
nNL=sum(notlast);
% LOCAL SEARCH, the scheme of TransitionPath_InfHorz_shooting with a reference per node: iteration 1 is a standard
% sweep that builds the reference; later iterations restrict the aprime search of every node to a window around it,
% refreshing it as they go, and nlocalsearch ratchets (up by nlocalsearchup when an answer sat on a window edge
% anywhere on the tree, excluding the grid's own ends, down by nlocalsearchdown otherwise, reset to the input after
% every standard sweep). When the restricted iterations converge, ONE standard sweep re-evaluates the SAME prices: if
% they are still converged the answer is certified, else local search resumes from the new reference. So the
% restriction never changes the answer, only how fast it is reached, and the summary is always of a standard sweep.
uselocalsearch=vfoptions.localsearch;
aprimeReferenceTree=[];
lsmodenow=uselocalsearch; % 0 no local search, 1 restricted (standard while there is no reference), 2 verification
nverify=0; nrestricted=0; lsnup=0; lsndown=0; lsnedge=0; lsntotal=0; lsmovemax=0;
if uselocalsearch==1
    nlocalsearchstart=vfoptions.nlocalsearch;
    nlocalsearchcap=floor((prod(n_a)-1)/2); % at the cap the window is the whole grid
    lsnmin=Inf; lsnmax=0;
end
while itercounter<=transpathoptions.maxiter
    % The whole tree at the current prices, and the general eqm conditions at every node
    wasrestricted=(lsmodenow==1 && ~isempty(aprimeReferenceTree));
    [~,aprimeReferenceTree,GEcondnTree,AggVarsTree,VPathTree,PolicyPathTree,AgentDistPathTree,AggVarsGivenChildTree,lsdiag]=UncertainTransitionPath_InfHorz_evaltree(reshape(PricePathOld,[],1), aprimeReferenceTree, lsmodenow, nNodes, nPrices, PricePathNames, notlast, Tree, ParamPath, ParamPathNodes, ParamPathNames, T, T1, T2, pi_S, S_initial, V_final, Policy_final, AgentDist_initial, n_d,n_a,n_z, d_grid,a_grid,z_grid, pi_z, ReturnFn, FnsToEvaluate, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, GEusestplus1, tplus1priceNames, tplus1pricePathkk, tplus1FnNames, tminus1priceNames, tminus1paramNames, tminus1AggVarsNames, Parameters, DiscountFactorParamNames, transpathoptions, simoptions, vfoptions);
    if wasrestricted
        % The ratchet on nlocalsearch, from the edge counts of this restricted sweep over the whole tree
        nrestricted=nrestricted+1;
        lsnmin=min(lsnmin,vfoptions.nlocalsearch);
        lsnmax=max(lsnmax,vfoptions.nlocalsearch);
        lsnedge=lsnedge+lsdiag.nedge;
        lsntotal=lsntotal+lsdiag.ntotal;
        lsmovemax=max(lsmovemax,lsdiag.movemax);
        if lsdiag.nedge>0
            vfoptions.nlocalsearch=min(vfoptions.nlocalsearch+vfoptions.nlocalsearchup,nlocalsearchcap);
            lsnup=lsnup+1;
        else
            vfoptions.nlocalsearch=max(vfoptions.nlocalsearch-vfoptions.nlocalsearchdown,1);
            lsndown=lsndown+1;
        end
    end

    % The shooting update of the nodes before period T, from the general eqm conditions at each of them. It is the
    % deterministic one (updatePricePathNew_TPath_T, a per-period formula, so the order of the rows does not matter),
    % applied to those nodes as if they were the periods 1,...,T-1 of a path; the extra row given to it is the 'period
    % T' it passes through, and is dropped.
    PricePathNew=PricePathOld;
    Pnotlastnew=updatePricePathNew_TPath_T(GEcondnTree(notlast,:),[PricePathOld(notlast,:); zeros(1,nPrices)],nNL+1,itercounter,transpathoptions);
    PricePathNew(notlast,:)=Pnotlastnew(1:nNL,:);

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
        if ~wasrestricted
            break % converged under a standard sweep, so the answer is certified
        else
            % Converged under local search: re-evaluate THESE SAME prices with a standard sweep (not updated, or the
            % check would be of different prices)
            lsmodenow=2;
            nverify=nverify+1;
            if transpathoptions.verbose>=1
                fprintf('UncertainTransitionPath_InfHorz: local search converged at iteration %i, checking it with a standard sweep \n',itercounter)
            end
            itercounter=itercounter+1;
            continue
        end
    end
    if itercounter==transpathoptions.maxiter
        break
    end
    if uselocalsearch==1 && ~wasrestricted
        % A reference exists now, so the restricted search starts (again), from the input window
        lsmodenow=1;
        vfoptions.nlocalsearch=nlocalsearchstart;
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
if uselocalsearch==1
    % Printed whenever local search was used, as in TransitionPath_InfHorz_shooting
    fprintf('Local search: %i iterations, of which %i restricted and %i standard verification sweep(s) \n',itercounter,nrestricted,nverify)
    if nrestricted>0
        fprintf('Local search: nlocalsearch started at %i, ranged %i to %i, ended at %i, with %i ratchet(s) up and %i down \n',nlocalsearchstart,lsnmin,lsnmax,vfoptions.nlocalsearch,lsnup,lsndown)
        fprintf('Local search: the answer sat on a window edge, excluding the grid ends, at %.2f%% of state-nodes; the largest move from a reference was %i grid points, which the window bounds \n',100*lsnedge/max(lsntotal,1),lsmovemax)
    end
    UncertainTPathSummary.localsearch.nrestricted=nrestricted;
    UncertainTPathSummary.localsearch.nverify=nverify;
    UncertainTPathSummary.localsearch.lastsweeprestricted=wasrestricted;
end

elseif transpathoptions.GEnewprice==2
%% Anderson acceleration of the shooting map, on the prices of every node stacked into one vector
% p is the nNodes-by-nPrices prices, as a column. The general eqm conditions are those of the nodes before period T,
% and the shooting map is the deterministic one (updatePricePathNew_TPath_T) applied to those nodes, with the period T
% nodes (the final eqms) passed through unchanged, as the deterministic shooting map passes period T through.
nNL=sum(notlast);
% lsmode 0: a standard sweep. With local search the residual carries the reference per node, so it goes to
% AndersonAccelerationTPath (as TransitionPath_InfHorz_Anderson does): the restricted residual (lsmode 1, the first
% evaluation builds the reference) and an unrestricted VerifyFn (lsmode 2) that certifies convergence. The first two
% outputs of evaltree are the conditions and the reference, which is the two-output form both of them need.
GEcondnsFn=@(p) UncertainTransitionPath_InfHorz_evaltree(p, [], 0, nNodes, nPrices, PricePathNames, notlast, Tree, ParamPath, ParamPathNodes, ParamPathNames, T, T1, T2, pi_S, S_initial, V_final, Policy_final, AgentDist_initial, n_d,n_a,n_z, d_grid,a_grid,z_grid, pi_z, ReturnFn, FnsToEvaluate, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, GEusestplus1, tplus1priceNames, tplus1pricePathkk, tplus1FnNames, tminus1priceNames, tminus1paramNames, tminus1AggVarsNames, Parameters, DiscountFactorParamNames, transpathoptions, simoptions, vfoptions);
GEcondnsFnLS=@(p,ref) UncertainTransitionPath_InfHorz_evaltree(p, ref, 1, nNodes, nPrices, PricePathNames, notlast, Tree, ParamPath, ParamPathNodes, ParamPathNames, T, T1, T2, pi_S, S_initial, V_final, Policy_final, AgentDist_initial, n_d,n_a,n_z, d_grid,a_grid,z_grid, pi_z, ReturnFn, FnsToEvaluate, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, GEusestplus1, tplus1priceNames, tplus1pricePathkk, tplus1FnNames, tminus1priceNames, tminus1paramNames, tminus1AggVarsNames, Parameters, DiscountFactorParamNames, transpathoptions, simoptions, vfoptions);
VerifyFnLS=@(p) UncertainTransitionPath_InfHorz_evaltree(p, [], 2, nNodes, nPrices, PricePathNames, notlast, Tree, ParamPath, ParamPathNodes, ParamPathNames, T, T1, T2, pi_S, S_initial, V_final, Policy_final, AgentDist_initial, n_d,n_a,n_z, d_grid,a_grid,z_grid, pi_z, ReturnFn, FnsToEvaluate, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, GEusestplus1, tplus1priceNames, tplus1pricePathkk, tplus1FnNames, tminus1priceNames, tminus1paramNames, tminus1AggVarsNames, Parameters, DiscountFactorParamNames, transpathoptions, simoptions, vfoptions);
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
if vfoptions.localsearch==1
    [p,GEcondns,andersonoutput]=AndersonAccelerationTPath(GEcondnsFnLS,ShootingMapFn,DistanceFn,VerifyFnLS,[],reshape(PricePathOld,[],1),transpathoptions.toleranceGEcondns,andersonoptions);
    fprintf('Local search: %i Anderson iterations, %i unrestricted certification sweep(s), of which %i put the solve back outside tolerance \n',andersonoutput.iterations,andersonoutput.nverify,andersonoutput.nverifyfailed)
    UncertainTPathSummary.localsearch.nverify=andersonoutput.nverify;
else
    [p,GEcondns,andersonoutput]=AndersonAcceleration(GEcondnsFn,ShootingMapFn,DistanceFn,reshape(PricePathOld,[],1),transpathoptions.toleranceGEcondns,andersonoptions);
end
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
% The tree at the returned prices, for the summary (one more evaluation, a standard sweep; Anderson keeps only the
% conditions)
[~,~,GEcondnTree,AggVarsTree,VPathTree,PolicyPathTree,AgentDistPathTree,AggVarsGivenChildTree]=GEcondnsFn(reshape(PricePathOld,[],1));
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
UncertainTPathSummary.AggVarsGivenChildTree=AggVarsGivenChildTree;
UncertainTPathSummary.VPathTree=VPathTree;
UncertainTPathSummary.PolicyPathTree=PolicyPathTree;
UncertainTPathSummary.AgentDistPathTree=AgentDistPathTree;
UncertainTPathSummary.iterations=itercounter;
UncertainTPathSummary.PricePathDist=PricePathDist;
UncertainTPathSummary.GEcondnDist=GEcondnDist;

end
