function varargout=TransitionPath_InfHorz_PType(PricePath0, ParamPath, T, V_final, AgentDist_initial, n_d,n_a,n_z, Names_i, d_grid,a_grid,z_grid, pi_z, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, PTypeDistParamNames, transpathoptions, simoptions, vfoptions)
% PricePath0 is a structure with fields names being the Prices and each field containing a T-by-1 path. It is the initial guess for the PricePath.
% ParamPath is a structure with fields names being the parameter names of those parameters which change over the path and each field containing a T-by-1 path.
%
% Each permanent type is set up exactly as TransitionPath_InfHorz sets up a model without permanent types,
% and everything about it is kept in PTypeStructure.(iistr), including its own copy of transpathoptions
% (the exogenous shock setup writes the type's z and e paths into transpathoptions, so one shared copy
% would leave every type using the last type's shocks). One iteration of the path then solves each type
% with TransitionPath_InfHorz_singlepathiter, the same code as without permanent types, and only the
% general eqm conditions are evaluated across types, on the aggregates pooled with the ptweights. See
% TransitionPath_InfHorz_PType_singlepathiter. The same design as TransitionPath_Case1_FHorz_PType.
%
% transpathoptions.GEnewprice means the same as in TransitionPath_InfHorz: =2 is Anderson acceleration,
% =3 is the shooting algorithm. =1 (quasi-Newton) is not implemented with permanent types.

% Create N_i so I can use it
if iscell(Names_i)
    N_i=length(Names_i);
else
    N_i=Names_i;
end


%% Check which transpathoptions have been used, set all others to defaults
if exist('transpathoptions','var')==0
    % transpathoptions.GEnewprice must be set explicitly: =2 is Anderson acceleration, =3 is the shooting
    % algorithm (=1, quasi-Newton, exists only without permanent types). There is deliberately no default.
    % GEnewprice=1 used to be the default here and meant that the GeneralEqmEqns were price-updating
    % formulae; that mode is gone, and the numbers now mean the same as in TransitionPath_InfHorz.
    error('transpathoptions.GEnewprice must be set: =2 for Anderson acceleration, =3 for the shooting algorithm')
else
    %Check transpathoptions for missing fields, if there are some fill them with the defaults
    if isfield(transpathoptions,'tolerance')
        error('Old transpathoptions.tolerance, has now been renamed and you should use transpathoptions.toleranceGEcondns instead')
    end
    if ~isfield(transpathoptions,'toleranceGEprices')
        transpathoptions.toleranceGEprices=Inf; % convergence criterion for GE prices, set =Inf to turn this off (it is off by default)
    end
    if ~isfield(transpathoptions,'toleranceGEcondns')
        transpathoptions.toleranceGEcondns=1e-4; % convergence criterion for GE condns
    end
    if ~isfield(transpathoptions,'multiGEcriterion')
        transpathoptions.multiGEcriterion=1;
    end
    if ~isfield(transpathoptions,'multiGEweights')
        transpathoptions.multiGEweights=ones(1,length(fieldnames(GeneralEqmEqns))); % One weight per general eqm eqn (the ptype-dependent ones get duplicated later)
    end
    if ~isfield(transpathoptions,'updateaccuracycutoff')
        transpathoptions.updateaccuracycutoff=10^(-9);
    end
    if ~isfield(transpathoptions,'parallel')
        transpathoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(transpathoptions,'GEnewprice')
        % No default, see the comment in the branch above where transpathoptions is not given at all
        error('transpathoptions.GEnewprice must be set: =2 for Anderson acceleration, =3 for the shooting algorithm')
    end
    if ~isfield(transpathoptions,'GEptype')
        transpathoptions.GEptype={}; % 1 indicates that this general eqm condition is 'conditional on permanent type'
    end
    if ~isfield(transpathoptions,'PricePathptype_vectoroutput')
        transpathoptions.PricePathptype_vectoroutput=0; % PricePath that depends on ptype defaults to being output as a structure
    end
    if ~isfield(transpathoptions,'anderson')
        transpathoptions.anderson=struct(); % GEnewprice=2 only: the Anderson acceleration options, all documented in AndersonAcceleration(). Defaults are set there, except for safeguard just below
    end
    if ~isfield(transpathoptions.anderson,'safeguard')
        transpathoptions.anderson.safeguard=0; % GEnewprice=2 only: 0: take every Anderson step. 1: evaluate the general eqm conditions at the trial point too, and fall back to a plain shooting step if the distance got worse, which costs a second path solve on every Anderson iteration
    end
    if ~isfield(transpathoptions,'oldpathweight')
        if transpathoptions.GEnewprice==3
            transpathoptions.oldpathweight=0; % user has to specify them as part of setup
        else
            transpathoptions.oldpathweight=0.9;
        end
    end
    if ~isfield(transpathoptions,'weightscheme')
        transpathoptions.weightscheme=1;
    end
    if ~isfield(transpathoptions,'Ttheta')
        transpathoptions.Ttheta=1;
    end
    if ~isfield(transpathoptions,'maxiter')
        transpathoptions.maxiter=1000;
    end
    if ~isfield(transpathoptions,'verbose')
        transpathoptions.verbose=0;
    end
    if ~isfield(transpathoptions,'graphpricepath')
        transpathoptions.graphpricepath=0; % 1: creates a graph of the 'current' price path which updates each iteration.
    end
    if ~isfield(transpathoptions,'graphaggvarspath')
        transpathoptions.graphaggvarspath=0; % 1: creates a graph of the 'current' aggregate variables which updates each iteration.
    end
    if ~isfield(transpathoptions,'graphGEcondns')
        transpathoptions.graphGEcondns=0;  % 1: creates a graph of the 'current' general eqm conditions which updates each iteration.
    end
    if ~isfield(transpathoptions,'historyofpricepath')
        transpathoptions.historyofpricepath=0;
    end
    % Constraints on the prices (see the setup below where PricePath becomes a matrix)
    if ~isfield(transpathoptions,'constrainpositive')
        transpathoptions.constrainpositive={}; % names of prices constrained to be positive
    end
    if ~isfield(transpathoptions,'constrain0to1')
        transpathoptions.constrain0to1={}; % names of prices constrained to be between 0 and 1
    end
    if ~isfield(transpathoptions,'constrainAtoB')
        transpathoptions.constrainAtoB={}; % names of prices constrained to be between A and B, with transpathoptions.constrainAtoBlimits.(name)=[A,B]
    end
    if ~isfield(transpathoptions,'constraintstepfraction')
        transpathoptions.constraintstepfraction=0.5; % when an update would take a constrained price past its bound, it moves this fraction of the way to the bound instead
    end
    if ~isfield(transpathoptions,'stockvars')
        transpathoptions.stockvars={}; % 'stockvars' are prices where you write '_tminus1' and it should cumulate (to there will be a general eqm eqn that relates the _tminus1 to the t for a price in PricePath)
    end
    if ~isfield(transpathoptions,'tanimprovement')
        transpathoptions.tanimprovement=1;
    end
end
if transpathoptions.parallel~=2
    error('Transition paths can only be solved if you have a GPU')
end
if transpathoptions.GEnewprice==1
    error('transpathoptions.GEnewprice=1 (quasi-Newton) is not implemented for permanent types, use =2 (Anderson acceleration) or =3 (the shooting algorithm)')
elseif transpathoptions.GEnewprice~=2 && transpathoptions.GEnewprice~=3
    error('transpathoptions.GEnewprice must be 2 (Anderson acceleration) or 3 (the shooting algorithm)')
end
if transpathoptions.GEnewprice==2 && (~isfield(transpathoptions,'GEnewprice2') || ~isfield(transpathoptions.GEnewprice2,'howtoupdate'))
    % Anderson acceleration accelerates the shooting map, so it needs the same instructions the shooting algorithm does
    error('transpathoptions.GEnewprice=2 (Anderson acceleration) requires transpathoptions.GEnewprice2.howtoupdate (same format as GEnewprice3.howtoupdate)')
end
% updatepert=1 updates each price inside the loop over t. With permanent types the general eqm conditions
% need the aggregates of every type, so they are only evaluated once every type has been solved, and the
% update is always the whole-path one (updatepert=0). updatepert=1 only matters for stockvars, which
% cannot be used with permanent types.
if isfield(transpathoptions,'updatepert')
    if transpathoptions.updatepert==1
        error('transpathoptions.updatepert=1 is not implemented for permanent types (it is only needed for stockvars, which cannot be used with permanent types)')
    end
end
transpathoptions.updatepert=0;

%% Reformat transpathoptions.GEptype from cell of names into vector of 1s and 0s
if isempty(transpathoptions.GEptype)
    transpathoptions.GEptype=zeros(1,length(fieldnames(GeneralEqmEqns))); % 1 indicates that this general eqm condition is 'conditional on permanent type'
else
    temp=transpathoptions.GEptype;
    transpathoptions.GEptype=zeros(1,length(fieldnames(GeneralEqmEqns))); % 1 indicates that this general eqm condition is 'conditional on permanent type'
    GEeqnNames=fieldnames(GeneralEqmEqns);
    for gg1=1:length(temp)
        for gg2=1:length(GEeqnNames)
            if strcmp(temp{gg1},GEeqnNames{gg2})
                transpathoptions.GEptype(gg2)=1;
            end
        end
    end
end


%% vfoptions and simoptions are set up per permanent type (below), make sure they exist
if exist('vfoptions','var')==0
    vfoptions=struct();
end
if exist('simoptions','var')==0
    simoptions=struct();
end


%% Create PTypeStructure

% PTypeStructure.Names_i never really gets used. Just makes things easier
% to read when you are looking at PTypeStructure (which only ever exists
% internally to the VFI Toolkit)
if iscell(Names_i)
    PTypeStructure.Names_i=Names_i;
    PTypeStructure.N_i=length(Names_i);
else
    PTypeStructure.N_i=Names_i;
    PTypeStructure.Names_i={'ptype001'};
    for ii=2:PTypeStructure.N_i
        if ii<10
            PTypeStructure.Names_i{ii}=['ptype00',num2str(ii)];
        elseif ii<100
            PTypeStructure.Names_i{ii}=['ptype0',num2str(ii)];
        elseif ii<1000
            PTypeStructure.Names_i{ii}=['ptype',num2str(ii)];
        end
    end
end
PTypeStructure.ParametersRaw=Parameters; % For use in General eqm conditions (as we might want them across ptypes for some purposes)

FnNames=fieldnames(FnsToEvaluate);
PTypeStructure.numFnsToEvaluate=length(FnNames);


%% The permanent type weights, PTypeStructure.ptweights_T (N_i-by-T)
% Each ptype's agent distribution has mass one, the ptype weights only enter when the aggregates are
% pooled across ptypes. They are those of the initial agent distribution in every period. (Type masses
% that change over the path, which TransitionPath_Case1_FHorz_PType has, are not yet implemented here.)
if isfield(ParamPath,PTypeDistParamNames{1})
    error('The permanent type weights (%s) on the ParamPath are not yet implemented for InfHorz transition paths with permanent types (email me if you want this)',PTypeDistParamNames{1})
end
PTypeStructure.ptweights_T=repmat(reshape(gather(AgentDist_initial.ptweights),[],1),1,T);
if isfield(Parameters,PTypeDistParamNames{1})
    if numel(Parameters.(PTypeDistParamNames{1}))~=PTypeStructure.N_i
        error('Parameter for PTypeDistParamNames does not have the same number of permanent types as N_i/Names_i')
    end
    if max(abs(reshape(Parameters.(PTypeDistParamNames{1}),[],1)-reshape(gather(AgentDist_initial.ptweights),[],1)))>10^(-9)
        warning('TransitionPath_InfHorz_PType: the permanent type weights in Parameters (%s) differ from AgentDist_initial.ptweights, by up to %g. The transition path uses AgentDist_initial.ptweights.',PTypeDistParamNames{1},max(abs(reshape(Parameters.(PTypeDistParamNames{1}),[],1)-reshape(gather(AgentDist_initial.ptweights),[],1))))
    end
end
PTypeStructure.ptweights_T=gpuArray(PTypeStructure.ptweights_T);


%% PricePath: internally a matrix of size T-by-'number of prices'
% A price depends on ptype when it is a structure, with a field for every permanent type (named by
% Names_i), or when it is not a structure and has N_i rows (row ii is ptype ii). Each ptype then sees
% its own column. The columns are put in Names_i order, whatever the order of the fields of the structure.
PricePathNames=fieldnames(PricePath0);
nPrices=length(PricePathNames);
PTypeStructure.PricePath_Idependsonptype=zeros(1,nPrices);
PTypeStructure.PricePathWidth_ii=zeros(1,nPrices); % the number of columns each ptype sees
PricePathCell=cell(1,nPrices);
for pp=1:nPrices
    temp=PricePath0.(PricePathNames{pp});
    if isstruct(temp)
        tempnames=fieldnames(temp);
        if ~all(ismember(tempnames,PTypeStructure.Names_i))
            error('PricePath.%s is a structure, so its fields must be permanent type names (Names_i), but %s is not one',PricePathNames{pp},tempnames{find(~ismember(tempnames,PTypeStructure.Names_i),1)})
        end
        if ~all(ismember(PTypeStructure.Names_i,tempnames))
            error('PricePath.%s is a structure, so it must have a field for every permanent type, but it has none for %s',PricePathNames{pp},PTypeStructure.Names_i{find(~ismember(PTypeStructure.Names_i,tempnames),1)})
        end
        PTypeStructure.PricePath_Idependsonptype(pp)=1;
        PricePathCell{pp}=zeros(T,0);
        for ii=1:PTypeStructure.N_i % in Names_i order
            temp_ii=temp.(PTypeStructure.Names_i{ii});
            if size(temp_ii,1)~=T
                temp_ii=temp_ii';
            end
            if ii==1
                PTypeStructure.PricePathWidth_ii(pp)=size(temp_ii,2);
            elseif size(temp_ii,2)~=PTypeStructure.PricePathWidth_ii(pp)
                error('PricePath.%s: the path of every permanent type must be the same size',PricePathNames{pp})
            end
            PricePathCell{pp}=[PricePathCell{pp},temp_ii];
        end
    else
        if size(temp,1)~=T
            temp=temp';
        end
        PricePathCell{pp}=temp;
        % (with N_i=1 every price would pass the width test, but none of them depends on ptype)
        if size(temp,2)==PTypeStructure.N_i && PTypeStructure.N_i>1
            PTypeStructure.PricePath_Idependsonptype(pp)=1;
            PTypeStructure.PricePathWidth_ii(pp)=1;
        else
            PTypeStructure.PricePathWidth_ii(pp)=size(temp,2);
        end
    end
    if size(PricePathCell{pp},1)~=T
        error('PricePath.%s appears to be the wrong size (it should be T periods long)',PricePathNames{pp})
    end
end
PricePath0=gpuArray([PricePathCell{:}]);
temp=zeros(1,nPrices);
for pp=1:nPrices
    temp(pp)=size(PricePathCell{pp},2);
end
PricePathSizeVec=[cumsum(temp)-temp+1; cumsum(temp)]; % the columns of each price in PricePath (all N_i of a price that depends on ptype)
PricePathSizeVec_ii=[cumsum(PTypeStructure.PricePathWidth_ii)-PTypeStructure.PricePathWidth_ii+1; cumsum(PTypeStructure.PricePathWidth_ii)]; % the columns of each price in the PricePath seen by one ptype

%% Constraints on the prices
% transpathoptions.constrainpositive, constrain0to1 and constrainAtoB (with constrainAtoBlimits.(name)=[A,B]) take
% the names of PricePath prices, with the same meanings as the heteroagentoptions of the same names, and apply to
% every period (and every column of a price that has more than one; so a price that depends on ptype has the same
% bounds for every ptype). The bounds are open: the prices must stay
% strictly inside them.
% They are NOT handled by a change of variables, as the stationary general eqm does. The transition path
% algorithms are not optimizers: their updates are defined in price units (howtoupdate moves a price by a factor
% times its general eqm residual), and a change of variables would change what every update means, even where no
% constraint is anywhere near binding. Instead the algorithm proposes its update as usual, and any (period,price)
% that the update would take past a bound moves only transpathoptions.constraintstepfraction of the way from its
% current value to the bound (a fraction-to-the-boundary rule, as interior-point methods use). Elements that stay
% inside are untouched, so a path whose constraints never bind is computed exactly as without them, and the bound
% is approached geometrically when the solution is near it.
% So there is nothing to choose between a log and a softplus transform here, and constrainpositivemethod (which
% heteroagentoptions has) is ignored.
% transpathoptions.constraintstepfraction is a scalar (the same for every constrained price), or a struct by price name,
% e.g. constraintstepfraction.r=0.25, like constrainAtoBlimits; a constrained price that is not named gets 0.5.
% The bounds are put in transpathoptions.pricepathlowerbound and pricepathupperbound, and the step fractions in
% transpathoptions.constraintstepfractionvec, one entry per column of PricePath (-Inf and Inf, and 0.5, where
% unconstrained), which is all the solvers need.
nPricePathCols=size(PricePath0,2);
transpathoptions.pricepathlowerbound=-Inf(1,nPricePathCols);
transpathoptions.pricepathupperbound=Inf(1,nPricePathCols);
transpathoptions.useconstraints=0;
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
            error('transpathoptions.%s contains ''%s'', which is not a price in PricePath (only prices can be constrained on a transition path)',conname,connames{nn})
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
        transpathoptions.pricepathlowerbound(PricePathSizeVec(1,pp):PricePathSizeVec(2,pp))=lb;
        transpathoptions.pricepathupperbound(PricePathSizeVec(1,pp):PricePathSizeVec(2,pp))=ub;
        transpathoptions.useconstraints=1;
    end
end
if transpathoptions.useconstraints==1
    % The step fraction, per column of PricePath
    % A constrained price not named in a struct gets the default, 0.5 (quasi-Newton, which defaults to 0.99, is not
    % implemented with permanent types)
    transpathoptions.constraintstepfractionvec=0.5*ones(1,nPricePathCols);
    if isstruct(transpathoptions.constraintstepfraction)
        temp=fieldnames(transpathoptions.constraintstepfraction);
        for nn=1:length(temp)
            if ~any(strcmp(temp{nn},constrainednames))
                error('transpathoptions.constraintstepfraction.%s is set, but ''%s'' is not a constrained price (it is not in constrainpositive, constrain0to1 or constrainAtoB)',temp{nn},temp{nn})
            end
            pp=find(strcmp(temp{nn},PricePathNames));
            transpathoptions.constraintstepfractionvec(PricePathSizeVec(1,pp):PricePathSizeVec(2,pp))=transpathoptions.constraintstepfraction.(temp{nn});
        end
    elseif isscalar(transpathoptions.constraintstepfraction)
        transpathoptions.constraintstepfractionvec(:)=transpathoptions.constraintstepfraction;
    else
        error('transpathoptions.constraintstepfraction must be a scalar, or a struct with one field per constrained price (e.g. transpathoptions.constraintstepfraction.r=0.25)')
    end
    if ~all(transpathoptions.constraintstepfractionvec>0 & transpathoptions.constraintstepfractionvec<1)
        error('transpathoptions.constraintstepfraction must be strictly between 0 and 1 (for every price it is set for)')
    end
    % The initial guess has to be strictly inside the bounds, in every period including the final one (which the
    % solvers never update), else the rule has no interior point to step from
    for pp=1:length(PricePathNames)
        cols=PricePathSizeVec(1,pp):PricePathSizeVec(2,pp);
        temp=gather(PricePath0(:,cols));
        if any(temp(:)<=min(transpathoptions.pricepathlowerbound(cols))) || any(temp(:)>=max(transpathoptions.pricepathupperbound(cols)))
            error('The initial PricePath for ''%s'' is not strictly inside its constraint (bounds [%g,%g]) in every period',PricePathNames{pp},transpathoptions.pricepathlowerbound(cols(1)),transpathoptions.pricepathupperbound(cols(1)))
        end
    end
    % Both algorithms apply the constraints: shooting (GEnewprice=3) and Anderson (=2, both types) cut back each element
    % of the update by constraintstepfraction (see TransitionPath_InfHorz_PType_shooting and _PType_Anderson)
end



%% ParamPath
% Each ptype gets its own ParamPath (set up in the loop over ptypes below). A parameter on the ParamPath
% can depend on ptype as a structure, with fields named by Names_i, and then need not be given for
% every ptype: a ptype that it has no field for is not on the path for that parameter (and the
% parameter takes its value in Parameters, for that ptype, in every period). It can also depend on
% ptype as a non-structure with N_i rows (row ii is ptype ii).
% ParamPath itself is then just the parameters that are not structures, and is what the general eqm
% conditions that do not depend on ptype see.
ParamPathIn=ParamPath;
ParamPathNamesAll=fieldnames(ParamPathIn);
ParamPathNames={};
ParamPathCell={};
for pp=1:length(ParamPathNamesAll)
    temp=ParamPathIn.(ParamPathNamesAll{pp});
    if isstruct(temp)
        tempnames=fieldnames(temp);
        if ~all(ismember(tempnames,PTypeStructure.Names_i))
            error('ParamPath.%s is a structure, so its fields must be permanent type names (Names_i), but %s is not one',ParamPathNamesAll{pp},tempnames{find(~ismember(tempnames,PTypeStructure.Names_i),1)})
        end
    else
        if size(temp,1)~=T
            temp=temp';
        end
        if size(temp,1)~=T
            error('ParamPath.%s appears to be the wrong size (it should be T periods long)',ParamPathNamesAll{pp})
        end
        ParamPathNames{end+1}=ParamPathNamesAll{pp};
        ParamPathCell{end+1}=temp;
    end
end
if isempty(ParamPathNames)
    ParamPath=zeros(T,0);
    ParamPathSizeVec=zeros(2,0);
else
    ParamPath=[ParamPathCell{:}];
    temp=zeros(1,length(ParamPathCell));
    for pp=1:length(ParamPathCell)
        temp(pp)=size(ParamPathCell{pp},2);
    end
    ParamPathSizeVec=[cumsum(temp)-temp+1; cumsum(temp)];
end

if transpathoptions.verbose>1
    PricePathNames
    ParamPathNamesAll
end


%% Check some inputs
if isstruct(GeneralEqmEqns)
    if length(PricePathNames)~=length(fieldnames(GeneralEqmEqns))
        fprintf('length(PricePathNames)=%i and length(fieldnames(GeneralEqmEqns))=%i (relates to following error) \n', length(PricePathNames), length(fieldnames(GeneralEqmEqns)))
        error('Initial PricePath contains less variables than GeneralEqmEqns (structure) \n')
    end
else
    if length(PricePathNames)~=length(GeneralEqmEqns)
        error('Initial PricePath contains less variables than GeneralEqmEqns')
    end
end


%% Fill out all of PTypeStructure
if transpathoptions.verbose==1
    fprintf('Setting up the permanent types for transition \n')
end

for ii=1:PTypeStructure.N_i

    iistr=PTypeStructure.Names_i{ii};
    PTypeStructure.iistr{ii}=iistr;

    %% Which parts of PricePath relate to ptype ii
    % A price that depends on ptype is N_i blocks of PricePathWidth_ii columns, in Names_i order; ptype ii sees block ii
    PTypeStructure.(iistr).RelevantPricePath=false(1,size(PricePath0,2));
    for pp=1:length(PricePathNames)
        if PTypeStructure.PricePath_Idependsonptype(pp)==1
            PTypeStructure.(iistr).RelevantPricePath(PricePathSizeVec(1,pp)+(ii-1)*PTypeStructure.PricePathWidth_ii(pp):PricePathSizeVec(1,pp)+ii*PTypeStructure.PricePathWidth_ii(pp)-1)=true;
        else
            PTypeStructure.(iistr).RelevantPricePath(PricePathSizeVec(1,pp):PricePathSizeVec(2,pp))=true;
        end
    end

    %% This ptype's ParamPath
    % A structure is read by ptype name, and skipped if it has no field for this ptype (the parameter is
    % then not on the path for this ptype); a non-structure with N_i rows gives row ii
    ParamPathNames_ii={};
    ParamPathCell_ii={};
    for pp=1:length(ParamPathNamesAll)
        temp=ParamPathIn.(ParamPathNamesAll{pp});
        if isstruct(temp)
            if ~isfield(temp,iistr)
                continue
            end
            temp=temp.(iistr);
            if size(temp,1)~=T
                temp=temp';
            end
        else
            if size(temp,1)~=T
                temp=temp';
            end
            if size(temp,2)==PTypeStructure.N_i && PTypeStructure.N_i>1
                temp=temp(:,ii);
            end
        end
        if size(temp,1)~=T
            error('ParamPath.%s (permanent type %s) appears to be the wrong size (it should be T periods long)',ParamPathNamesAll{pp},iistr)
        end
        ParamPathNames_ii{end+1}=ParamPathNamesAll{pp};
        ParamPathCell_ii{end+1}=temp;
    end
    if isempty(ParamPathNames_ii)
        ParamPath_ii=zeros(T,0);
        ParamPathSizeVec_ii=zeros(2,0);
    else
        ParamPath_ii=[ParamPathCell_ii{:}];
        temp=zeros(1,length(ParamPathCell_ii));
        for pp=1:length(ParamPathCell_ii)
            temp(pp)=size(ParamPathCell_ii{pp},2);
        end
        ParamPathSizeVec_ii=[cumsum(temp)-temp+1; cumsum(temp)];
    end
    ParamPath_ii=gpuArray(ParamPath_ii);

    %% This type's transpathoptions
    % Its own copy, because the shock setup below writes the z and e paths of this type into it
    transpathoptions_ii=transpathoptions;
    if ~isfield(transpathoptions_ii,'localsearchdiagnostic')
        transpathoptions_ii.localsearchdiagnostic=0;
    end

    %% This type's vfoptions, with defaults (as in TransitionPath_InfHorz)
    vfoptions_ii=PType_Options(vfoptions,iistr); % some vfoptions will differ by permanent type
    if ~isfield(vfoptions_ii,'lowmemory')
        vfoptions_ii.lowmemory=0;
    end
    if ~isfield(vfoptions_ii,'verbose')
        vfoptions_ii.verbose=0;
    end
    % Model setup
    if ~isfield(vfoptions_ii,'exoticpreferences')
        vfoptions_ii.exoticpreferences='None';
    elseif ~strcmp(vfoptions_ii.exoticpreferences,'None')
        error('transition paths with permanent types cannot yet handle exoticpreferences')
    end
    if ~isfield(vfoptions_ii,'experienceasset')
        vfoptions_ii.experienceasset=0;
    end
    % Exogenous shocks
    if ~isfield(vfoptions_ii,'n_semiz')
        vfoptions_ii.n_semiz=0;
    end
    if ~isfield(vfoptions_ii,'n_e')
        vfoptions_ii.n_e=0;
    end
    % Algorithm to use
    if ~isfield(vfoptions_ii,'solnmethod')
        vfoptions_ii.solnmethod='purediscretization'; % Currently this does nothing
    end
    if ~isfield(vfoptions_ii,'gridinterplayer')
        vfoptions_ii.gridinterplayer=0;
    elseif vfoptions_ii.gridinterplayer==1
        if ~isfield(vfoptions_ii,'ngridinterp')
            error('When using vfoptions.gridinterplayer=1 you must set vfoptions.ngridinterp')
        end
    end
    % Local search needs a reference carried across GE iterations and a verification sweep, which only
    % TransitionPath_InfHorz_shooting and _Anderson have; the PType solvers would reach Step1 restricted with no reference
    if ~isfield(vfoptions_ii,'localsearch')
        vfoptions_ii.localsearch=0;
    elseif vfoptions_ii.localsearch~=0
        error('vfoptions.localsearch is not yet implemented on a transition path with permanent types (email me if you want this)')
    end
    vfoptions_ii.parallel=2; % transition path is GPU only

    %% Combinations that are not implemented on a transition path
    % The substeps key off experienceasset and n_semiz only, so anything else silently reaches the
    % standard-endogenous-state code and returns the wrong model.
    if isfield(vfoptions_ii,'riskyasset')
        if vfoptions_ii.riskyasset>=1
            error('vfoptions.riskyasset is not yet implemented on a transition path (email me if you want this)')
        end
    end
    if isfield(vfoptions_ii,'residualasset')
        if vfoptions_ii.residualasset>=1
            error('vfoptions.residualasset is not yet implemented on a transition path (email me if you want this)')
        end
    end

    %% This type's simoptions, with defaults (as in TransitionPath_InfHorz)
    simoptions_ii=PType_Options(simoptions,iistr); % some simoptions will differ by permanent type
    if ~isfield(simoptions_ii,'tolerance')
        simoptions_ii.tolerance=10^(-9);
    end
    if ~isfield(simoptions_ii,'verbose')
        simoptions_ii.verbose=0;
    end
    if ~isfield(simoptions_ii,'experienceasset')
        simoptions_ii.experienceasset=0;
    end
    if ~isfield(simoptions_ii,'gridinterplayer')
        simoptions_ii.gridinterplayer=0;
    elseif simoptions_ii.gridinterplayer==1
        if ~isfield(simoptions_ii,'ngridinterp')
            error('When using simoptions.gridinterplayer=1 you must set simoptions.ngridinterp')
        end
    end
    if ~isfield(simoptions_ii,'n_e')
        simoptions_ii.n_e=0;
    end
    simoptions_ii.parallel=2; % transition path is GPU only

    %% Go through everything which might be dependent on permanent type (PType)
    [n_d_ii,n_a_ii,d_grid_ii,a_grid_ii]=PType_setup_da(iistr,n_d,n_a,d_grid,a_grid);
    % divideandconquer default needs n_a, so is set here rather than with the other vfoptions
    if ~isfield(vfoptions_ii,'divideandconquer')
        vfoptions_ii.divideandconquer=0;
    elseif vfoptions_ii.divideandconquer==1
        if ~isfield(vfoptions_ii,'level1n')
            if isscalar(n_a_ii)
                vfoptions_ii.level1n=floor(sqrt(n_a_ii(1)));
            elseif length(n_a_ii)==2
                vfoptions_ii.level1n=[floor(sqrt(n_a_ii(1))),n_a_ii(2)]; % default DC2A: level1n(2)==n_a(2) triggers DC2A branch
            end
        end
        if n_a_ii(1)<5
            error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
        end
    end
    % Exogenous shocks: z (and e, in vfoptions and simoptions)
    [n_z_ii,z_grid_ii,pi_z_ii,vfoptions_ii]=PType_setup_ExogShocks(ii,iistr,N_i,n_z,z_grid,pi_z,vfoptions_ii,3);
    [~,~,~,simoptions_ii]=PType_setup_ExogShocks(ii,iistr,N_i,n_z,z_grid,pi_z,simoptions_ii,3);
    % ReturnFn and DiscountFactor
    [ReturnFn_ii, DiscountFactorParamNames_ii]=PType_setup_ReturnFnDiscountFactor(iistr,ReturnFn,DiscountFactorParamNames);
    % Parameters
    Parameters_ii=PType_setup_Parameters(ii,iistr,N_i,Parameters,3);

    %% Make sure all the relevant inputs are GPU arrays (not standard arrays)
    pi_z_ii=gpuArray(pi_z_ii);
    d_grid_ii=gpuArray(d_grid_ii);
    a_grid_ii=gpuArray(a_grid_ii);
    z_grid_ii=gpuArray(z_grid_ii);
    V_final_ii=gpuArray(V_final.(iistr));
    % Tan improvement means we want agent dist on cpu
    AgentDist_initial_ii=gather(AgentDist_initial.(iistr));

    %% Check the sizes of some of the inputs
    N_d_ii=prod(n_d_ii);
    N_a_ii=prod(n_a_ii);
    N_z_ii=prod(n_z_ii);
    N_e_ii=prod(vfoptions_ii.n_e);
    % vfoptions.lowmemory=2 loops over both z and e, so it is only possible in models with both z and e
    if vfoptions_ii.lowmemory==2 && (N_z_ii==0 || N_e_ii==0)
        error('vfoptions.lowmemory=2 is only possible in models with both z and e (it loops over both), use vfoptions.lowmemory=1')
    end
    if N_e_ii>0
        % The e shocks are set up from simoptions below (ExogShockSetup_InfHorz_TPath), so if they were only
        % given in vfoptions, copy them across
        if prod(simoptions_ii.n_e)==0
            simoptions_ii.n_e=vfoptions_ii.n_e;
            if isfield(vfoptions_ii,'EiidShockFn')
                simoptions_ii.EiidShockFn=vfoptions_ii.EiidShockFn;
            else
                simoptions_ii.e_grid=vfoptions_ii.e_grid;
                simoptions_ii.pi_e=vfoptions_ii.pi_e;
            end
        end
        if vfoptions_ii.experienceasset>=1
            error('Transition paths with an experience asset are not yet implemented with an e variable')
        end
    end

    if N_d_ii==0
        l_d_ii=0;
    else
        l_d_ii=length(n_d_ii);
    end
    l_a_ii=length(n_a_ii);
    l_aprime_ii=l_a_ii;
    if vfoptions_ii.experienceasset>=1
        l_aprime_ii=l_aprime_ii-1;
    end
    if N_z_ii==0
        l_z_ii=0;
    else
        l_z_ii=length(n_z_ii);
    end
    if N_e_ii==0
        l_e_ii=0;
    else
        l_e_ii=length(vfoptions_ii.n_e);
    end
    N_semiz_ii=prod(vfoptions_ii.n_semiz);
    if N_semiz_ii==0
        l_semiz_ii=0;
    else
        l_semiz_ii=length(vfoptions_ii.n_semiz);
    end

    %% Semi-exogenous states, vfoptions.n_semiz (the same refusals as TransitionPath_InfHorz)
    if N_semiz_ii>0
        if N_d_ii==0
            error('Transition paths with semi-exogenous states (vfoptions.n_semiz) require a decision variable: the last vfoptions.l_dsemiz decision variables determine the semiz transitions')
        end
        if length(n_a_ii)>1
            error('Transition paths with semi-exogenous states (vfoptions.n_semiz) only allow a single endogenous state (cannot have length(n_a)>1)')
        end
        if vfoptions_ii.experienceasset>=1
            error('Transition paths with semi-exogenous states (vfoptions.n_semiz) are not yet implemented together with an experience asset')
        end
    end

    %% Implement new way of handling ReturnFn inputs
    ReturnFnParamNames_ii=ReturnFnParamNamesFn(ReturnFn_ii,n_d_ii,n_a_ii,n_z_ii,0,vfoptions_ii,Parameters_ii);

    %% Set up exogenous shock processes
    [z_gridvals_ii, pi_z_ii, pi_z_sparse_ii, e_gridvals_ii, pi_e_ii, pi_e_sparse_ii, ze_gridvals_ii, transpathoptions_ii, simoptions_ii]=ExogShockSetup_InfHorz_TPath(n_z_ii,z_grid_ii,pi_z_ii,Parameters_ii,PricePathNames,ParamPath_ii,ParamPathNames_ii,ParamPathSizeVec_ii,T,transpathoptions_ii,simoptions_ii,4);
    % Convert z and e to joint-grids and transition matrix
    % Sets up transpathoptions_ii.zpathtrivial, .epathtrivial, .gridsinGE (and the _T paths when they vary over the path)

    %% Set up semi-exogenous states (as TransitionPath_InfHorz)
    if N_semiz_ii>0
        [semiz_gridvals_ii, pi_semiz_ii, transpathoptions_ii, vfoptions_ii]=SemiExogShockSetup_InfHorz_TPath(n_d_ii,d_grid_ii,Parameters_ii,PricePathNames,ParamPath_ii,ParamPathNames_ii,ParamPathSizeVec_ii,T,transpathoptions_ii,vfoptions_ii,4);
        vfoptions_ii.setup_semiexo.semiz_gridvals=semiz_gridvals_ii;
        vfoptions_ii.setup_semiexo.pi_semiz=pi_semiz_ii;
        simoptions_ii.n_semiz=vfoptions_ii.n_semiz;
        simoptions_ii.l_dsemiz=vfoptions_ii.l_dsemiz;
        simoptions_ii.setup_semiexo=vfoptions_ii.setup_semiexo;
        simoptions_ii.setup_semiexo.pi_semiz=gather(pi_semiz_ii); % agent distribution iteration is performed on cpu
        % Joint grid on (semiz,z,e), semiz fastest, used for AggVars (semiz disguised as part of z)
        if N_z_ii==0 && N_e_ii==0
            simoptions_ii.setup_semiexo.semizze_gridvals=semiz_gridvals_ii;
        else
            N_ze_ii=max(N_z_ii,1)*max(N_e_ii,1);
            simoptions_ii.setup_semiexo.semizze_gridvals=[repmat(semiz_gridvals_ii,N_ze_ii,1),repelem(ze_gridvals_ii,N_semiz_ii,1)];
            if transpathoptions_ii.zepathtrivial==0
                % z and/or e grids vary over the path (the semiz grid cannot): [N_semiz*N_ze,l_semiz+l_z+l_e,T]
                transpathoptions_ii.semizze_gridvals_T=[repmat(semiz_gridvals_ii,N_ze_ii,1,T),repelem(transpathoptions_ii.ze_gridvals_T,N_semiz_ii,1,1)];
            end
        end
    else
        if isfield(simoptions_ii,'n_semiz') && prod(simoptions_ii.n_semiz)>0
            error('Transition paths: simoptions.n_semiz is set but vfoptions.n_semiz is not. The semi-exogenous state setup (n_semiz, semiz_grid, SemiExoStateFn or pi_semiz, l_dsemiz) is read from vfoptions')
        end
        simoptions_ii.n_semiz=0;
    end

    %% If using any non-standard endogenous states, setup for those
    [vfoptions_ii,simoptions_ii]=SetupNonStandardEndoStates_InfHorz_TPath(n_d_ii,n_a_ii,d_grid_ii,a_grid_ii,vfoptions_ii,simoptions_ii);

    %% Setup for V_final
    % Note: I keep Policy as having a first dimension (even if it is just 1)
    if N_semiz_ii>0 % V keeps the composite bothz=(semiz,z) dimension, e separate
        if N_e_ii==0
            V_final_ii=reshape(V_final_ii,[N_a_ii,N_semiz_ii*max(N_z_ii,1)]);
        else
            V_final_ii=reshape(V_final_ii,[N_a_ii,N_semiz_ii*max(N_z_ii,1),N_e_ii]);
        end
    elseif N_e_ii==0
        if N_z_ii==0
            V_final_ii=reshape(V_final_ii,[N_a_ii,1]);
        else
            V_final_ii=reshape(V_final_ii,[N_a_ii,N_z_ii]);
        end
    else
        if N_z_ii==0
            V_final_ii=reshape(V_final_ii,[N_a_ii,N_e_ii]);
        else
            V_final_ii=reshape(V_final_ii,[N_a_ii,N_z_ii,N_e_ii]);
        end
    end

    %% Setup for AgentDist_initial
    if N_semiz_ii>0 % (a,semiz,z,e), semiz fastest
        AgentDist_initial_ii=reshape(AgentDist_initial_ii,[N_a_ii*N_semiz_ii*max(N_z_ii,1)*max(N_e_ii,1),1]);
    elseif N_e_ii==0  % no z, no e
        if N_z_ii==0
            AgentDist_initial_ii=reshape(AgentDist_initial_ii,[N_a_ii,1]);
        else % z, no e
            AgentDist_initial_ii=reshape(AgentDist_initial_ii,[N_a_ii*N_z_ii,1]);
        end
    else
        if N_z_ii==0 % no z, e
            AgentDist_initial_ii=reshape(AgentDist_initial_ii,[N_a_ii*N_e_ii,1]);
        else % z & e
            AgentDist_initial_ii=reshape(AgentDist_initial_ii,[N_a_ii*N_z_ii*N_e_ii,1]);
        end
    end

    %% Figure out which functions are actually relevant to the present PType. And then change to FnsToEvaluate as cell so that it is not being recomputed all the time
    % A FnsToEvaluate that is a structure has one entry per ptype that it applies to; one that is a
    % function applies to all ptypes. Only the relevant ones are evaluated for this ptype.
    % WhichFnsForCurrentPType(kk) is the position of FnsToEvaluate kk in this ptype's list (0 if not relevant).
    l_daprime_ii=l_d_ii+l_a_ii;
    if vfoptions_ii.experienceasset>=1
        l_daprime_ii=l_daprime_ii-1;
    end
    FnsToEvaluate_ii=struct();
    WhichFnsForCurrentPType_ii=zeros(PTypeStructure.numFnsToEvaluate,1);
    jj=1; % jj indexes the FnsToEvaluate that are relevant to the current PType
    for kk=1:PTypeStructure.numFnsToEvaluate
        if isa(FnsToEvaluate.(FnNames{kk}),'struct')
            if isfield(FnsToEvaluate.(FnNames{kk}), iistr)
                FnsToEvaluate_ii.(FnNames{kk})=FnsToEvaluate.(FnNames{kk}).(iistr);
                WhichFnsForCurrentPType_ii(kk)=jj; jj=jj+1;
            end
            % otherwise this FnsToEvaluate is not relevant for the current PType
        else
            % If the Fn is not a structure (if it is a function) it is assumed to be relevant to all PTypes.
            FnsToEvaluate_ii.(FnNames{kk})=FnsToEvaluate.(FnNames{kk});
            WhichFnsForCurrentPType_ii(kk)=jj; jj=jj+1;
        end
    end
    AggVarNames_ii=fieldnames(FnsToEvaluate_ii);
    FnsToEvaluateCell_ii=cell(1,length(AggVarNames_ii));
    FnsToEvaluateParamNames_ii=struct('Names',{});
    for ff=1:length(AggVarNames_ii)
        temp=getAnonymousFnInputNames(FnsToEvaluate_ii.(AggVarNames_ii{ff}));
        if length(temp)>(l_daprime_ii+l_a_ii+l_semiz_ii+l_z_ii+l_e_ii)
            FnsToEvaluateParamNames_ii(ff).Names={temp{l_daprime_ii+l_a_ii+l_semiz_ii+l_z_ii+l_e_ii+1:end}}; % the first inputs will always be (d,aprime,a,semiz,z,e)
        else
            FnsToEvaluateParamNames_ii(ff).Names={};
        end
        FnsToEvaluateCell_ii{ff}=FnsToEvaluate_ii.(AggVarNames_ii{ff});
    end
    % Change FnsToEvaluate out of structure form, but want to still create AggVars as a structure
    simoptions_ii.outputasstructure=1;

    %% Set up Gridvals (used by FnsToEvaluate, among others)
    a_gridvals_ii=CreateGridvals(n_a_ii,a_grid_ii,1); % a_gridvals is [N_a,l_a]

    if N_d_ii>0
        % Gridvals: switch to joint-grids
        if all(size(d_grid_ii)==[sum(n_d_ii),1]) % if stacked-column grid
            d_gridvals_ii=CreateGridvals(n_d_ii,gpuArray(d_grid_ii),1);
        elseif all(size(d_grid_ii)==[prod(n_d_ii),length(n_d_ii)]) % if joint-grid
            d_gridvals_ii=gpuArray(d_grid_ii);
        end
    else
        d_gridvals_ii=[];
    end

    if vfoptions_ii.gridinterplayer==0
        aprime_gridvals_ii=a_gridvals_ii;
    elseif vfoptions_ii.gridinterplayer==1
        % use fine grid for aprime_gridvals
        if isscalar(n_a_ii)
            n_aprime_ii=n_a_ii+(n_a_ii-1)*vfoptions_ii.ngridinterp;
            aprime_grid_ii=interp1(gpuArray(1:1:N_a_ii)',a_grid_ii,gpuArray(linspace(1,N_a_ii,n_aprime_ii))');
            aprime_gridvals_ii=CreateGridvals(n_aprime_ii,aprime_grid_ii,1);
        else
            a1_grid_ii=a_grid_ii(1:n_a_ii(1));
            n_a1prime_ii=n_a_ii(1)+(n_a_ii(1)-1)*vfoptions_ii.ngridinterp;
            n_aprime_ii=[n_a1prime_ii,n_a_ii(2:end)];
            a1prime_grid_ii=interp1(gpuArray(1:1:n_a_ii(1))',a1_grid_ii,gpuArray(linspace(1,n_a_ii(1),n_a1prime_ii))');
            aprime_grid_ii=[a1prime_grid_ii; a_grid_ii(n_a_ii(1)+1:end)];
            aprime_gridvals_ii=CreateGridvals(n_aprime_ii,aprime_grid_ii,1);
        end
        vfoptions_ii.policyind2val_finegridinput=1; % aprime_gridvals contains the fine grid for the first asset (tells PolicyInd2Val_InfHorz_TPath)
    end

    %% Setup, the shapes of various of these objects vary depending on the setting (as in TransitionPath_InfHorz_shooting)
    [PolicyIndexesPath_ii,N_probs_ii,II1_ii,II2_ii]=TransitionPath_InfHorz_substeps_Step0_setup(l_d_ii,l_aprime_ii,N_a_ii,N_z_ii,N_e_ii,T,transpathoptions_ii,vfoptions_ii,simoptions_ii);

    %% Each type is solved with TransitionPath_InfHorz_singlepathiter, but evaluates no general eqm conditions
    % itself (they need the aggregates of every type), and the intermediateEqns belong with them. It
    % returns the aggregate variables of the path, which it only keeps when graphaggvarspath=1. The
    % graphs themselves are drawn by the PType solver, not by the single path iteration.
    transpathoptions_ii.useintermediateEqns=0;
    transpathoptions_ii.graphaggvarspath=1;
    transpathoptions_ii.updatepert=0;

    %% Store everything about this ptype
    PTypeStructure.(iistr).n_d=n_d_ii;
    PTypeStructure.(iistr).n_a=n_a_ii;
    PTypeStructure.(iistr).n_z=n_z_ii;
    PTypeStructure.(iistr).n_e=vfoptions_ii.n_e;
    PTypeStructure.(iistr).N_a=N_a_ii;
    PTypeStructure.(iistr).N_z=N_z_ii;
    PTypeStructure.(iistr).N_e=N_e_ii;
    PTypeStructure.(iistr).l_d=l_d_ii;
    PTypeStructure.(iistr).l_aprime=l_aprime_ii;
    PTypeStructure.(iistr).d_gridvals=d_gridvals_ii;
    PTypeStructure.(iistr).aprime_gridvals=aprime_gridvals_ii;
    PTypeStructure.(iistr).a_gridvals=a_gridvals_ii;
    PTypeStructure.(iistr).a_grid=a_grid_ii;
    PTypeStructure.(iistr).z_gridvals=z_gridvals_ii;
    PTypeStructure.(iistr).e_gridvals=e_gridvals_ii;
    PTypeStructure.(iistr).ze_gridvals=ze_gridvals_ii;
    PTypeStructure.(iistr).pi_z=pi_z_ii;
    PTypeStructure.(iistr).pi_z_sparse=pi_z_sparse_ii;
    PTypeStructure.(iistr).pi_e=pi_e_ii;
    PTypeStructure.(iistr).ReturnFn=ReturnFn_ii;
    PTypeStructure.(iistr).ReturnFnParamNames=ReturnFnParamNames_ii;
    PTypeStructure.(iistr).DiscountFactorParamNames=DiscountFactorParamNames_ii;
    PTypeStructure.(iistr).Parameters=Parameters_ii;
    PTypeStructure.(iistr).FnsToEvaluateCell=FnsToEvaluateCell_ii;
    PTypeStructure.(iistr).FnsToEvaluateParamNames=FnsToEvaluateParamNames_ii;
    PTypeStructure.(iistr).AggVarNames=AggVarNames_ii;
    PTypeStructure.(iistr).WhichFnsForCurrentPType=WhichFnsForCurrentPType_ii;
    PTypeStructure.(iistr).V_final=V_final_ii;
    PTypeStructure.(iistr).AgentDist_initial=AgentDist_initial_ii;
    PTypeStructure.(iistr).PolicyIndexesPath=PolicyIndexesPath_ii;
    PTypeStructure.(iistr).N_probs=N_probs_ii;
    PTypeStructure.(iistr).II1=II1_ii;
    PTypeStructure.(iistr).II2=II2_ii;
    PTypeStructure.(iistr).vfoptions=vfoptions_ii;
    PTypeStructure.(iistr).simoptions=simoptions_ii;
    PTypeStructure.(iistr).transpathoptions=transpathoptions_ii;
    PTypeStructure.(iistr).ParamPath=ParamPath_ii;
    PTypeStructure.(iistr).ParamPathNames=ParamPathNames_ii;
    PTypeStructure.(iistr).ParamPathSizeVec=ParamPathSizeVec_ii;
end

%% If using intermediateEqns, switch from structure to cell setup
transpathoptions.useintermediateEqns=0;
if isfield(transpathoptions,'intermediateEqns')
    transpathoptions.useintermediateEqns=1;
    intEqnNames=fieldnames(transpathoptions.intermediateEqns);
    nIntEqns=length(intEqnNames);

    transpathoptions.intermediateEqnsCell=cell(1,nIntEqns);
    for gg=1:nIntEqns
        temp=getAnonymousFnInputNames(transpathoptions.intermediateEqns.(intEqnNames{gg}));
        transpathoptions.intermediateEqnParamNames(gg).Names=temp;
        transpathoptions.intermediateEqnsCell{gg}=transpathoptions.intermediateEqns.(intEqnNames{gg});
    end
    % Now:
    %  transpathoptions.intermediateEqns is still the structure
    %  transpathoptions.intermediateEqnsCell is cell
    %  transpathoptions.intermediateEqnParamNames(gg).Names contains the names
end

%% GE eqns, switch from structure to cell setup
GEeqnNames=fieldnames(GeneralEqmEqns);
nGeneralEqmEqns=length(GEeqnNames);
nGeneralEqmEqns_acrossptypes=sum(transpathoptions.GEptype==0)+N_i*sum(transpathoptions.GEptype==1);
% transpathoptions.multiGEweights is one weight per general eqm eqn. A general eqm condition that
% depends on ptype is evaluated once per ptype and so occupies N_i consecutive entries of
% GEcondnPath, in eqn order, so duplicate its weight N_i times to line the two up.
if ~all(transpathoptions.GEptype==0)
    transpathoptions.multiGEweights=repelem(transpathoptions.multiGEweights,1+(N_i-1)*(transpathoptions.GEptype==1));
end

GeneralEqmEqnsCell=cell(1,nGeneralEqmEqns);
for gg=1:nGeneralEqmEqns
    temp=getAnonymousFnInputNames(GeneralEqmEqns.(GEeqnNames{gg}));
    GeneralEqmEqnParamNames(gg).Names=temp;
    GeneralEqmEqnsCell{gg}=GeneralEqmEqns.(GEeqnNames{gg});
end
% Now:
%  GeneralEqmEqns is still the structure
%  GeneralEqmEqnsCell is cell
%  GeneralEqmEqnParamNames(ff).Names contains the names


%% Set up the shooting update (used by both GEnewprice=3, and by GEnewprice=2 which accelerates the same map)
transpathoptions=setupGEnewprice3_shooting(transpathoptions,GeneralEqmEqns,PricePathNames,N_i,PricePathSizeVec);


if transpathoptions.verbose==1
    fprintf('Completed setup, beginning transition computation \n')
end

%% Check if using _tminus1 and/or _tplus1 variables.
[tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tminus1paramNames,tplus1pricePathkk,use_tplus1price,use_tminus1price,use_tminus1params,use_tminus1AggVars]=inputsFindtplus1tminus1(FnsToEvaluate,GeneralEqmEqns,PricePathNames,ParamPathNamesAll,PTypeStructure.Names_i,transpathoptions);

if ~isempty(transpathoptions.stockvars)
    error('Cannot yet use transpathoptions.stockvars with permanent types')
end

% An initial value that depends on ptype is a structure (with a field for every ptype, named by Names_i)
% or a vector of length N_i.
tempnames=[tminus1priceNames(:);tminus1paramNames(:);tminus1AggVarsNames(:)];
for pp=1:length(tempnames)
    if isstruct(transpathoptions.initialvalues.(tempnames{pp}))
        if ~all(ismember(PTypeStructure.Names_i,fieldnames(transpathoptions.initialvalues.(tempnames{pp}))))
            error('transpathoptions.initialvalues.%s is a structure, so it must have a field for every permanent type (it has none for %s)',tempnames{pp},PTypeStructure.Names_i{find(~ismember(PTypeStructure.Names_i,fieldnames(transpathoptions.initialvalues.(tempnames{pp}))),1)})
        end
    end
end
% The _tminus1 prices and parameters are also seen by each ptype on its own (they can be in the ReturnFn
% or FnsToEvaluate), and an initial value that depends on ptype has to be given to each ptype as just its own value.
for ii=1:N_i
    iistr=PTypeStructure.Names_i{ii};
    if use_tminus1price==1
        for pp=1:length(tminus1priceNames)
            temp=transpathoptions.initialvalues.(tminus1priceNames{pp});
            if isstruct(temp)
                PTypeStructure.(iistr).transpathoptions.initialvalues.(tminus1priceNames{pp})=temp.(iistr);
            elseif length(temp)==N_i && N_i>1
                PTypeStructure.(iistr).transpathoptions.initialvalues.(tminus1priceNames{pp})=temp(ii);
            end
        end
    end
    if use_tminus1params==1
        for pp=1:length(tminus1paramNames)
            temp=transpathoptions.initialvalues.(tminus1paramNames{pp});
            if isstruct(temp)
                PTypeStructure.(iistr).transpathoptions.initialvalues.(tminus1paramNames{pp})=temp.(iistr);
            elseif length(temp)==N_i && N_i>1
                PTypeStructure.(iistr).transpathoptions.initialvalues.(tminus1paramNames{pp})=temp(ii);
            end
        end
    end
end


if transpathoptions.verbose>=1
    transpathoptions
end

%% Solve for the general eqm path
% GEnewprice: =2 Anderson acceleration, =3 the shooting algorithm
if transpathoptions.GEnewprice==2 % Anderson acceleration of the shooting update
    [PricePath,GEcondnPathmatrix]=TransitionPath_InfHorz_PType_Anderson(PricePath0, PricePathNames, PricePathSizeVec, PricePathSizeVec_ii, ParamPath, ParamPathNames, ParamPathSizeVec, T, FnNames, GEeqnNames, nGeneralEqmEqns_acrossptypes, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, transpathoptions, PTypeStructure);
elseif transpathoptions.GEnewprice==3 % the shooting algorithm
    [PricePath,GEcondnPathmatrix]=TransitionPath_InfHorz_PType_shooting(PricePath0, PricePathNames, PricePathSizeVec, PricePathSizeVec_ii, ParamPath, ParamPathNames, ParamPathSizeVec, T, FnNames, GEeqnNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, transpathoptions, PTypeStructure);
end

%% Switch the solution into structure for output.
% A price is a 1-by-T path. A price that depends on ptype is a structure with one such path per ptype,
% or with PricePathptype_vectoroutput=1 the N_i of them stacked in Names_i order (N_i-by-T).
for pp=1:length(PricePathNames)
    if PTypeStructure.PricePath_Idependsonptype(pp)==0
        PricePathStruct.(PricePathNames{pp})=PricePath(:,PricePathSizeVec(1,pp):PricePathSizeVec(2,pp))';
    else
        if transpathoptions.PricePathptype_vectoroutput==1
            PricePathStruct.(PricePathNames{pp})=PricePath(:,PricePathSizeVec(1,pp):PricePathSizeVec(2,pp))';
        elseif transpathoptions.PricePathptype_vectoroutput==0
            for ii=1:N_i
                PricePathStruct.(PricePathNames{pp}).(PTypeStructure.Names_i{ii})=PricePath(:,PricePathSizeVec(1,pp)+(ii-1)*PTypeStructure.PricePathWidth_ii(pp):PricePathSizeVec(1,pp)+ii*PTypeStructure.PricePathWidth_ii(pp)-1)';
            end
        end
    end
end
% GEcondnPath: a general eqm condition that depends on ptype has N_i columns, one per ptype, in eqn order
gg_c=0;
for gg=1:length(GEeqnNames)
    if transpathoptions.GEptype(gg)==0
        gg_c=gg_c+1;
        GEcondnPath.(GEeqnNames{gg})=GEcondnPathmatrix(:,gg_c)';
    else
        for ii=1:N_i
            gg_c=gg_c+1;
            GEcondnPath.(GEeqnNames{gg}).(PTypeStructure.Names_i{ii})=GEcondnPathmatrix(:,gg_c)';
        end
    end
end

if nargout==1
    varargout={PricePathStruct};
elseif nargout==2
    varargout={PricePathStruct,GEcondnPath};
end


end
