function varargout=TransitionPath_Case1_FHorz_PType(PricePath0, ParamPath, T, V_final, AgentDist_initial, jequalOneDist, n_d, n_a, n_z, N_j, Names_i, d_grid,a_grid,z_grid, pi_z, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, AgeWeightsParamNames, PTypeDistParamNames, transpathoptions, simoptions, vfoptions)
% This code will work for all transition paths except those that involve at
% change in the transition matrix pi_z (can handle a change in pi_z, but
% only if it is a 'surprise', not anticipated changes)
%
% PricePathOld is a structure with fields names being the Prices and each field containing a T-by-1 path.
% ParamPath is a structure with fields names being the parameter names of those parameters which change over the path and each field containing a T-by-1 path.
%
% Each permanent type is set up exactly as TransitionPath_Case1_FHorz sets up a model without permanent
% types, and everything about it is kept in PTypeStructure.(iistr), including its own copy of
% transpathoptions (the exogenous shock setup writes the type's z and e paths into transpathoptions, so
% one shared copy would leave every type using the last type's shocks). One iteration of the path then
% solves each type with TransitionPath_FHorz_singlepathiter, the same code as without permanent types,
% and only the general eqm conditions are evaluated across types, on the aggregates pooled with the
% ptweights. See TransitionPath_FHorz_PType_singlepathiter.

% Remark to self: No real need for T as input, as this is anyway the length of PricePathOld

% Create N_i so I can use it
if iscell(Names_i)
    N_i=length(Names_i);
else
    N_i=Names_i;
end


%% Check which transpathoptions have been used, set all others to defaults
if exist('transpathoptions','var')==0
    disp('No transpathoptions given, using defaults')
    % transpathoptions.GEnewprice must be set explicitly: =2 is Anderson acceleration, =3 is the shooting
    % algorithm (=1, quasi-Newton, exists only without permanent types). There is deliberately no default.
    % GEnewprice=1 used to be the default here and meant that the GeneralEqmEqns were price-updating
    % formulae; that mode is gone, and the numbers now mean the same as in TransitionPath_Case1_FHorz.
    error('transpathoptions.GEnewprice must be set: =2 for Anderson acceleration, =3 for the shooting algorithm')
else
    %Check transpathoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(transpathoptions,'fastOLG')
        transpathoptions.fastOLG=0; % fastOLG is done as (a,j,z), rather than standard (a,z,j)
    end
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
    if ~isfield(transpathoptions,'parallel')
        transpathoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(transpathoptions,'GEnewprice')
        % No default, see the comment in the branch above where transpathoptions is not given at all
        error('transpathoptions.GEnewprice must be set: =2 for Anderson acceleration, =3 for the shooting algorithm')
    end
    if ~isfield(transpathoptions,'GEptype')
        transpathoptions.GEptype={}; %zeros(1,length(fieldnames(GeneralEqmEqns))); % 1 indicates that this general eqm condition is 'conditional on permanent type'
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
        transpathoptions.oldpathweight=0.9;
        % Note that when using transpathoptions.GEnewprice==3
        % Implicitly it is setting transpathoptions.oldpathweight=0
        % because the user anyway has to specify them as part of setup
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
    if ~isfield(transpathoptions,'stockvars')
        transpathoptions.stockvars={}; % 'stockvars' are prices where you write '_tminus1' and it should cumulate (to there will be a general eqm eqn that relates the _tminus1 to the t for a price in PricePath)
    end
    % transpathoptions.updateageweights %Don't declare if not being used
end

if transpathoptions.parallel~=2
    error('Sorry but transition paths are not implemented for cpu, you will need a gpu to use them')
end
if transpathoptions.GEnewprice==1
    error('transpathoptions.GEnewprice=1 (quasi-Newton) is not implemented for permanent types, use =2 (Anderson acceleration) or =3 (the shooting algorithm)')
elseif transpathoptions.GEnewprice~=2 && transpathoptions.GEnewprice~=3
    error('transpathoptions.GEnewprice must be 2 (Anderson acceleration) or 3 (the shooting algorithm)')
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

%% Get AgeWeights from Parameters
if ~isfield(Parameters,AgeWeightsParamNames{1})
    error(['Failed to find parameter ', AgeWeightsParamNames{1}])
end
% Later, when creating PTypeStructure, we get the ptype-specific versions out of this and create AgeWeights_T for each ptype


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
PTypeStructure.ptweights=AgentDist_initial.ptweights;

FnNames=fieldnames(FnsToEvaluate);
PTypeStructure.numFnsToEvaluate=length(FnNames);
PTypeStructure.FnsAndPTypeIndicator=zeros(PTypeStructure.numFnsToEvaluate,PTypeStructure.N_i,'gpuArray');


%% The mass of each permanent type cannot change over the transition path
% The aggregates are pooled with AgentDist_initial.ptweights in every period
if isfield(ParamPath,PTypeDistParamNames{1})
    error('The permanent type weights (%s) cannot be on the ParamPath: the mass of each permanent type cannot (yet) change over a transition path',PTypeDistParamNames{1})
end


%% PricePath: internally a matrix of size T-by-'number of prices'
% A price depends on ptype when it is a structure, with a field for every permanent type (named by
% Names_i), or when it is not a structure and has N_i rows (row ii is ptype ii). Each ptype then sees
% its own block of columns: one column, or N_j for an age-dependent price. The blocks are put in
% Names_i order, whatever the order of the fields of the structure.
PricePathNames=fieldnames(PricePath0);
nPrices=length(PricePathNames);
PTypeStructure.PricePath_Idependsonptype=zeros(1,nPrices);
PTypeStructure.PricePathWidth_ii=zeros(1,nPrices); % the number of columns each ptype sees (1, or N_j for an age-dependent price)
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
PricePathSizeVec=[cumsum(temp)-temp+1; cumsum(temp)]; % the columns of each price in PricePath (all N_i blocks of a price that depends on ptype)
PricePathSizeVec_ii=[cumsum(PTypeStructure.PricePathWidth_ii)-PTypeStructure.PricePathWidth_ii+1; cumsum(PTypeStructure.PricePathWidth_ii)]; % the columns of each price in the PricePath seen by one ptype


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

    %% This type's transpathoptions
    % Its own copy, because the shock setup below writes the z and e paths of this type into it
    transpathoptions_ii=transpathoptions;

    %% This type's vfoptions, with defaults (as in TransitionPath_Case1_FHorz)
    vfoptions_ii=PType_Options(vfoptions,iistr); % some vfoptions will differ by permanent type
    if ~isfield(vfoptions_ii,'gridinterplayer')
        vfoptions_ii.gridinterplayer=0;
    elseif vfoptions_ii.gridinterplayer==1
        if ~isfield(vfoptions_ii,'ngridinterp')
            error('You have vfoptions.gridinterplayer, so must also set vfoptions.ngridinterp')
        end
    end
    if ~isfield(vfoptions_ii,'lowmemory')
        vfoptions_ii.lowmemory=0;
    end
    % Feedback options
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
    if ~isfield(vfoptions_ii,'experienceassetz')
        vfoptions_ii.experienceassetz=0;
    end
    % Exogenous shocks
    if ~isfield(vfoptions_ii,'n_semiz')
        vfoptions_ii.n_semiz=0;
    end
    if ~isfield(vfoptions_ii,'n_e')
        vfoptions_ii.n_e=0;
    end
    vfoptions_ii.parallel=2; % transition path is GPU only
    vfoptions_ii.EVpre=0; % =1 is used by 'Matched Expecations Path', for TPath we want =0 (this relates to details of fastOLG=1 value fn code)

    %% This type's simoptions, with defaults (as in TransitionPath_Case1_FHorz)
    simoptions_ii=PType_Options(simoptions,iistr); % some simoptions will differ by permanent type
    if transpathoptions_ii.fastOLG==1
        simoptions_ii.fastOLG=1;
    end
    if ~isfield(simoptions_ii,'verbose')
        simoptions_ii.verbose=0;
    end
    if ~isfield(simoptions_ii,'fastOLG')
        simoptions_ii.fastOLG=1;
    end
    if ~isfield(simoptions_ii,'gridinterplayer')
        simoptions_ii.gridinterplayer=0;
    end
    % Model setup
    if ~isfield(simoptions_ii,'experienceasset')
        simoptions_ii.experienceasset=0;
    end
    if ~isfield(simoptions_ii,'experienceassetz')
        simoptions_ii.experienceassetz=0;
    end
    simoptions_ii.parallel=2; % transition path is GPU only

    if transpathoptions_ii.fastOLG==1 && simoptions_ii.fastOLG==0
        error('If you set transpathoptions.fastOLG=1 then you must have simoptions.fastOLG=1 (I just have not implemented simoptions.fastOLG=0 for this')
    end
    if transpathoptions_ii.fastOLG==0 && vfoptions_ii.lowmemory>0
        error('On transtion paths you can only use vfoptions.lowmemory>0 when using transpathoptions.fastOLG=1, because otherwise the runtimes will anyway be so slow as to be essentially unusable')
    end

    %% Combinations that are not implemented on a transition path
    % The substeps key off experienceasset/experienceassetz and n_semiz only, so anything
    % else silently reaches the standard-endogenous-state code and returns the wrong model.
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
    if prod(vfoptions_ii.n_semiz)>0 && (vfoptions_ii.experienceasset>=1 || vfoptions_ii.experienceassetz>=1)
        error('experienceasset (or experienceassetz) together with a semi-exogenous state is not yet implemented on a transition path (email me if you want this)')
    end

    %% Go through everything which might be dependent on permanent type (PType)
    [n_d_ii,n_a_ii,d_grid_ii,a_grid_ii]=PType_setup_da(iistr,n_d,n_a,d_grid,a_grid);
    if isstruct(N_j)
        N_j_ii=N_j.(iistr);
    else
        N_j_ii=N_j;
    end
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

    if N_d_ii==0
        l_d_ii=0;
    else
        l_d_ii=length(n_d_ii);
    end
    l_a_ii=length(n_a_ii);
    l_aprime_ii=l_a_ii;
    if vfoptions_ii.experienceasset>=1 || vfoptions_ii.experienceassetz>=1
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
    if prod(vfoptions_ii.n_semiz)==0
        l_semiz_ii=0;
    else
        l_semiz_ii=length(vfoptions_ii.n_semiz);
    end

    %% Implement new way of handling ReturnFn inputs
    ReturnFnParamNames_ii=ReturnFnParamNamesFn(ReturnFn_ii,n_d_ii,n_a_ii,n_z_ii,N_j_ii,vfoptions_ii,Parameters_ii);

    %% Set up exogenous shock processes
    [z_gridvals_J_ii, pi_z_J_ii, pi_z_J_sim_ii, e_gridvals_J_ii, pi_e_J_ii, pi_e_J_sim_ii, ze_gridvals_J_fastOLG_ii, transpathoptions_ii, simoptions_ii]=ExogShockSetup_FHorz_TPath(n_z_ii,z_grid_ii,pi_z_ii,N_a_ii,N_j_ii,T,Parameters_ii,PricePathNames,ParamPathNames_ii,transpathoptions_ii,simoptions_ii,4);
    % Convert z and e to age-dependent joint-grids and transition matrix
    % Sets up transpathoptions_ii.zpathtrivial, .epathtrivial, .gridsinGE (and the _T paths when they vary over the path)

    %% Semi-exogenous states
    % The semi-exogenous transition (semiz->semiz') depends on the decision d2, so semiz is carried as part of the
    % composite exogenous state bothz=(semiz,z) [semiz indexes fastest, e kept separate] in the value function, and
    % bothze=(semiz,z,e) in the agent distribution. Setup is stashed in vfoptions/simoptions; the Step1/Step2/Step3tt/Step4tt
    % substeps dispatch to their SemiExo variants when prod(simoptions.n_semiz)>0.
    [semiz_gridvals_J_ii, pi_semiz_J_ii, pi_semiz_J_sim_ii, transpathoptions_ii, vfoptions_ii, simoptions_ii]=SemiExogShockSetup_FHorz_TPath(n_d_ii,N_j_ii,d_grid_ii,Parameters_ii,PricePathNames,ParamPath_ii,ParamPathNames_ii,ParamPathSizeVec_ii,T,transpathoptions_ii,vfoptions_ii,simoptions_ii,4);
    N_semiz_ii=prod(vfoptions_ii.n_semiz);
    if N_semiz_ii>0
        % Composite sizes used by the semiz reshape branches below
        N_asemiz_ii=N_a_ii*N_semiz_ii;
        if N_z_ii==0
            N_bothz_ii=N_semiz_ii;
        else
            N_bothz_ii=N_semiz_ii*N_z_ii;
        end
        if N_e_ii==0
            N_bothze_ii=N_bothz_ii;
        else
            N_bothze_ii=N_bothz_ii*N_e_ii;
        end

        % Add semiz to ze_gridvals_J_fastOLG -> semizze_gridvals_J_fastOLG (disguise semiz as part of z for AggVars; semiz indexes fastest)
        semiz_gridvals_J_fastOLG_ii=shiftdim(permute(vfoptions_ii.semiz_gridvals_J,[3,1,2]),-1); % [1,N_j,N_semiz,l_semiz]
        if isempty(ze_gridvals_J_fastOLG_ii) % no z, no e
            semizze_gridvals_J_fastOLG_ii=semiz_gridvals_J_fastOLG_ii;
        else
            N_ze_ii=size(ze_gridvals_J_fastOLG_ii,3);
            semizze_gridvals_J_fastOLG_ii=cat(4, repmat(semiz_gridvals_J_fastOLG_ii,1,1,N_ze_ii,1), repelem(ze_gridvals_J_fastOLG_ii,1,1,N_semiz_ii,1));
        end

        if transpathoptions_ii.fastOLG==1 && N_z_ii>0
            % The fastOLG SemiExo value fn raws form joint bothz=(semiz,z) expectations over all N_j slices, so
            % pi_z_J needs the same appended j=N_j zero row as pi_semiz_J (no continuation value in the final
            % period; the plain no-semiz fastOLG raws instead receive N_j-1 slices and append in-file)
            pi_z_J_ii=cat(1,pi_z_J_ii,zeros(1,N_z_ii,N_z_ii,'gpuArray'));
            if transpathoptions_ii.zpathtrivial==0
                transpathoptions_ii.pi_z_J_T=cat(1,transpathoptions_ii.pi_z_J_T,zeros(1,N_z_ii,N_z_ii,T,'gpuArray')); % so the per-period overrides in Step1 arrive pre-appended
            end
        end
    else
        semizze_gridvals_J_fastOLG_ii=ze_gridvals_J_fastOLG_ii;
    end

    %% If using any non-standard endogenous states, setup for those
    [vfoptions_ii,simoptions_ii]=SetupNonStandardEndoStates_FHorz_TPath(n_d_ii,n_a_ii,d_grid_ii,a_grid_ii,vfoptions_ii,simoptions_ii,n_z_ii,z_grid_ii,N_j_ii,Parameters_ii);
    if vfoptions_ii.experienceassetz>=1 && transpathoptions_ii.zpathtrivial==0
        error('experienceassetz with z varying over the transition path is not yet implemented (email me if you want this)')
    end

    %% Setup for V_final
    % Note: I keep Policy as having a first dimension (even if it is just 1)
    if N_semiz_ii==0
        if N_e_ii==0
            if N_z_ii==0
                V_final_ii=reshape(V_final_ii,[N_a_ii,N_j_ii]);
            else
                if transpathoptions_ii.fastOLG==0
                    V_final_ii=reshape(V_final_ii,[N_a_ii,N_z_ii,N_j_ii]);
                else % vfoptions.fastOLG==1
                    V_final_ii=reshape(permute(reshape(V_final_ii,[N_a_ii,N_z_ii,N_j_ii]),[1,3,2]),[N_a_ii*N_j_ii,N_z_ii]);
                end
            end
        else
            if N_z_ii==0
                if transpathoptions_ii.fastOLG==0
                    V_final_ii=reshape(V_final_ii,[N_a_ii,N_e_ii,N_j_ii]);
                else % vfoptions.fastOLG==1
                    V_final_ii=reshape(permute(reshape(V_final_ii,[N_a_ii,N_e_ii,N_j_ii]),[1,3,2]),[N_a_ii*N_j_ii,N_e_ii]);
                end
            else
                if transpathoptions_ii.fastOLG==0
                    V_final_ii=reshape(V_final_ii,[N_a_ii,N_z_ii,N_e_ii,N_j_ii]);
                else % vfoptions.fastOLG==1
                    V_final_ii=reshape(permute(reshape(V_final_ii,[N_a_ii,N_z_ii,N_e_ii,N_j_ii]),[1,4,2,3]),[N_a_ii*N_j_ii,N_z_ii,N_e_ii]);
                end
            end
        end
    else % semiz: V keeps the composite bothz=(semiz,z) dimension (e separate)
        if N_e_ii==0
            if transpathoptions_ii.fastOLG==0
                V_final_ii=reshape(V_final_ii,[N_a_ii,N_bothz_ii,N_j_ii]);
            else % fastOLG==1
                V_final_ii=reshape(permute(reshape(V_final_ii,[N_a_ii,N_bothz_ii,N_j_ii]),[1,3,2]),[N_a_ii*N_j_ii,N_bothz_ii]);
            end
        else
            if transpathoptions_ii.fastOLG==0
                V_final_ii=reshape(V_final_ii,[N_a_ii,N_bothz_ii,N_e_ii,N_j_ii]);
            else % fastOLG==1
                V_final_ii=reshape(permute(reshape(V_final_ii,[N_a_ii,N_bothz_ii,N_e_ii,N_j_ii]),[1,4,2,3]),[N_a_ii*N_j_ii,N_bothz_ii,N_e_ii]);
            end
        end
    end

    %% Get the age weights, check if they depend on path, and make sure they are the right shape
    % It is assumed there is only one Age Weight Parameter (name)). It can depend on ptype (structure, or N_i in one dimension)
    AgeWeights_ii=gpuArray(Parameters_ii.(AgeWeightsParamNames{1}));
    % If the AgeWeights do not vary over the transition, then we will just set them up now.
    transpathoptions_ii.ageweightstrivial=1;
    if all(size(AgeWeights_ii)==[N_j_ii,1])
        % Does not depend on transition path period
    elseif all(size(AgeWeights_ii)==[1,N_j_ii])
        % Does not depend on transition path period
        % Make AgeWeights a column vector
        AgeWeights_ii=AgeWeights_ii';
    else
        fprintf('Following error applies to agent permanent type: %s \n',iistr)
        error('The age weights parameter seems to be the wrong size')
    end
    AgeWeights_T_ii=repelem(AgeWeights_ii,1,T); % N_j-by-T
    % Check ParamPath to see if the AgeWeights vary over the transition
    temp=strcmp(ParamPathNames_ii,AgeWeightsParamNames{1});
    if any(temp)
        transpathoptions_ii.ageweightstrivial=0; % AgeWeights vary over the transition
        [~,kk]=max(temp); % Get index for the AgeWeightsParamNames{1} in ParamPathNames_ii
        % Create AgeWeights_T
        AgeWeights_T_ii=ParamPath_ii(:,ParamPathSizeVec_ii(1,kk):ParamPathSizeVec_ii(2,kk))'; % N_j-by-T
    end
    AgeWeights_path1_ii=AgeWeights_T_ii(:,1); % period 1 of the age weights, kept for the check against the initial agent distribution (AgeWeights_T is reshaped below)

    % Turn AgeWeights_T into appropriate size so that we can always just do AgentDist.*AgeWeights
    % Currently it is N_j-by-T
    if N_semiz_ii==0
        if simoptions_ii.fastOLG==0
            if N_e_ii==0
                if N_z_ii==0
                    AgeWeights_T_ii=repelem(shiftdim(AgeWeights_T_ii,-1),N_a_ii,1,1); % [N_a,N_j,T]
                else
                    AgeWeights_T_ii=repelem(shiftdim(AgeWeights_T_ii,-1),N_a_ii*N_z_ii,1,1); % [N_a*N_z,N_j,T]
                end
            else % N_e>0
                if N_z_ii==0
                    AgeWeights_T_ii=repelem(shiftdim(AgeWeights_T_ii,-1),N_a_ii*N_e_ii,1,1); % [N_a*N_e,N_j,T]
                else
                    AgeWeights_T_ii=repelem(shiftdim(AgeWeights_T_ii,-1),N_a_ii*N_z_ii*N_e_ii,1,1); % [N_a*N_z*N_e,N_j,T]
                end
            end
        elseif simoptions_ii.fastOLG==1
            if N_e_ii==0
                if N_z_ii==0
                    AgeWeights_T_ii=repelem(AgeWeights_T_ii,N_a_ii,1); % [N_a*N_j,T]
                else
                    AgeWeights_T_ii=repmat(repelem(AgeWeights_T_ii,N_a_ii,1),N_z_ii,1); % [N_a*N_j*N_z,T]
                end
            else % N_e>0
                if N_z_ii==0
                    AgeWeights_T_ii=repelem(reshape(AgeWeights_T_ii,[N_j_ii,1,T]),N_a_ii,N_e_ii); % [N_a*N_j,N_e,T]
                else
                    AgeWeights_T_ii=repmat(repelem(reshape(AgeWeights_T_ii,[N_j_ii,1,T]),N_a_ii,1),N_z_ii,N_e_ii); % [N_a*N_j*N_z,N_e,T]
                end
            end
        end
    else % semiz: dist carries the composite bothze=(semiz,z,e); shaped as without semiz, with bothz=(semiz,z) in place of z
        if simoptions_ii.fastOLG==0
            AgeWeights_T_ii=repelem(shiftdim(AgeWeights_T_ii,-1),N_a_ii*N_bothze_ii,1,1); % [N_a*N_bothze,N_j,T]
        elseif simoptions_ii.fastOLG==1
            % fastOLG dist is (a,j,bothz) with e trailing, the same as without semiz but with bothz=(semiz,z) in place of z
            if N_e_ii==0
                AgeWeights_T_ii=repmat(repelem(AgeWeights_T_ii,N_a_ii,1),N_bothz_ii,1); % [N_a*N_j*N_bothz,T]
            else % N_e>0
                AgeWeights_T_ii=repmat(repelem(reshape(AgeWeights_T_ii,[N_j_ii,1,T]),N_a_ii,1),N_bothz_ii,N_e_ii); % [N_a*N_j*N_bothz,N_e,T]
            end
        end
    end

    %% Setup for AgentDist_initial (includes a check that if AgeWeights do not change over the transition then they should match the initial agent distribution)
    if N_semiz_ii==0
        if N_e_ii==0  % no z, no e
            if N_z_ii==0
                AgentDist_initial_ii=reshape(AgentDist_initial_ii,[N_a_ii,N_j_ii]);
                AgeWeights_initial_ii=sum(AgentDist_initial_ii,1); % [1,N_j]
                if simoptions_ii.fastOLG==1
                    AgentDist_initial_ii=reshape(AgentDist_initial_ii,[N_a_ii*N_j_ii,1]);
                end
            else % z, no e
                AgentDist_initial_ii=reshape(AgentDist_initial_ii,[N_a_ii*N_z_ii,N_j_ii]);
                AgeWeights_initial_ii=sum(AgentDist_initial_ii,1); % [1,N_j]
                if simoptions_ii.fastOLG==1
                    AgentDist_initial_ii=reshape(AgentDist_initial_ii,[N_a_ii,N_z_ii,N_j_ii]);
                    AgentDist_initial_ii=permute(AgentDist_initial_ii,[1,3,2]);
                    AgentDist_initial_ii=reshape(AgentDist_initial_ii,[N_a_ii*N_j_ii*N_z_ii,1]);
                end
            end
        else
            if N_z_ii==0 % no z, e
                AgentDist_initial_ii=reshape(AgentDist_initial_ii,[N_a_ii*N_e_ii,N_j_ii]);
                AgeWeights_initial_ii=sum(AgentDist_initial_ii,1); % [1,N_j]
                if simoptions_ii.fastOLG==1
                    AgentDist_initial_ii=reshape(AgentDist_initial_ii,[N_a_ii,N_e_ii,N_j_ii]);
                    AgentDist_initial_ii=permute(AgentDist_initial_ii,[1,3,2]);
                    AgentDist_initial_ii=reshape(AgentDist_initial_ii,[N_a_ii*N_j_ii,N_e_ii]);
                end
            else % z & e
                AgentDist_initial_ii=reshape(AgentDist_initial_ii,[N_a_ii*N_z_ii*N_e_ii,N_j_ii]);
                AgeWeights_initial_ii=sum(AgentDist_initial_ii,1); % [1,N_j]
                if simoptions_ii.fastOLG==1
                    AgentDist_initial_ii=reshape(AgentDist_initial_ii,[N_a_ii,N_z_ii,N_e_ii,N_j_ii]);
                    AgentDist_initial_ii=permute(AgentDist_initial_ii,[1,4,2,3]);
                    AgentDist_initial_ii=reshape(AgentDist_initial_ii,[N_a_ii*N_j_ii*N_z_ii,N_e_ii]);
                end
            end
        end
    else % semiz: dist carries the composite bothze=(semiz,z,e); fastOLG uses (a,j,bothz) with e trailing, bothz=(semiz,z), as without semiz but with bothz in place of z
        AgentDist_initial_ii=reshape(AgentDist_initial_ii,[N_a_ii*N_bothze_ii,N_j_ii]);
        AgeWeights_initial_ii=sum(AgentDist_initial_ii,1); % [1,N_j]
        if simoptions_ii.fastOLG==1
            if N_e_ii==0
                AgentDist_initial_ii=reshape(permute(reshape(AgentDist_initial_ii,[N_a_ii,N_bothz_ii,N_j_ii]),[1,3,2]),[N_a_ii*N_j_ii*N_bothz_ii,1]);
            else
                AgentDist_initial_ii=reshape(permute(reshape(AgentDist_initial_ii,[N_a_ii,N_bothz_ii,N_e_ii,N_j_ii]),[1,4,2,3]),[N_a_ii*N_j_ii*N_bothz_ii,N_e_ii]);
            end
        end
    end
    if transpathoptions_ii.ageweightstrivial==1
        if max(abs(AgeWeights_initial_ii-AgeWeights_ii'))>10^(-9) % was 10^(-13), but this was problematic with numerical rounding errors
            % Note: AgeWeights_inital is [1,N_j], while AgeWeights is [N_j,1], hence we need the transpose on AgeWeights
            fprintf('Following error applies to agent permanent type: %s \n',iistr)
            fprintf('AgeWeights are: \n')
            AgeWeights_ii
            fprintf('AgeWeights implicit in the initial agent distribution are: \n')
            AgeWeights_initial_ii
            error('AgeWeights differs from the weights implicit in the initial agent distribution')
        end
    elseif max(abs(AgeWeights_initial_ii(:)-AgeWeights_path1_ii))>10^(-9)
        % The age weights are on the ParamPath, so period 1 aggregates use their first period. If the initial agent
        % distribution has other age weights, those are not used (it is normalised within each age first), which
        % is legitimate if the population is meant to jump at the start of the path, but is more often a mistake.
        warning('TransitionPath_Case1_FHorz_PType: for permanent type %s, the age weights in period 1 of the ParamPath (%s) differ from those implicit in the initial agent distribution, by up to %g. Period 1 uses the ParamPath ones; the initial distribution is only used within each age.',iistr,AgeWeightsParamNames{1},max(abs(AgeWeights_initial_ii(:)-AgeWeights_path1_ii)))
    end

    %% Remove the age weights and do all the iterations. Only put the age weights back in when performing FnsToEvaluate (faster as saves putting weights in and then removing them T times)
    % Weights are all in AgeWeights_T
    if N_semiz_ii==0
        if N_e_ii==0
            if N_z_ii==0
                if simoptions_ii.fastOLG==0
                    AgentDist_initial_ii=AgentDist_initial_ii./AgeWeights_initial_ii; % remove age weights
                elseif simoptions_ii.fastOLG==1
                    AgentDist_initial_ii=AgentDist_initial_ii./repelem(AgeWeights_initial_ii',N_a_ii,1); % remove age weights
                end
            else % N_z>0
                if simoptions_ii.fastOLG==0
                    AgentDist_initial_ii=AgentDist_initial_ii./AgeWeights_initial_ii; % remove age weights
                elseif simoptions_ii.fastOLG==1
                    AgentDist_initial_ii=AgentDist_initial_ii./repmat(repelem(AgeWeights_initial_ii',N_a_ii,1),N_z_ii,1); % remove age weights
                end
            end
        else % N_e>0
            if N_z_ii==0
                if simoptions_ii.fastOLG==0
                    AgentDist_initial_ii=AgentDist_initial_ii./AgeWeights_initial_ii; % remove age weights
                elseif simoptions_ii.fastOLG==1
                    AgentDist_initial_ii=AgentDist_initial_ii./repelem(AgeWeights_initial_ii',N_a_ii,1); % remove age weights
                end
            else % N_z>0
                if simoptions_ii.fastOLG==0
                    AgentDist_initial_ii=AgentDist_initial_ii./AgeWeights_initial_ii; % remove age weights
                elseif simoptions_ii.fastOLG==1
                    AgentDist_initial_ii=AgentDist_initial_ii./repmat(repelem(AgeWeights_initial_ii',N_a_ii,1),N_z_ii,1); % remove age weights
                end
            end
        end
    else % semiz
        if simoptions_ii.fastOLG==0
            AgentDist_initial_ii=AgentDist_initial_ii./AgeWeights_initial_ii; % remove age weights ([N_a*N_bothze,N_j])
        elseif simoptions_ii.fastOLG==1
            AgentDist_initial_ii=AgentDist_initial_ii./repmat(repelem(AgeWeights_initial_ii',N_a_ii,1),N_bothz_ii,1); % remove age weights ((a,j,bothz), e in columns)
        end
    end

    %% Check if jequalOneDistPath is a path or not (and reshape appropriately)
    % jequalOneDist can be common to all ptypes, or a structure with one per ptype
    if isstruct(jequalOneDist)
        jequalOneDist_ii=gpuArray(jequalOneDist.(iistr));
    else
        jequalOneDist_ii=gpuArray(jequalOneDist);
    end
    temp=size(jequalOneDist_ii);
    % Note: simoptions.fastOLG is handled via 'justfirstj', rather than via shape of jequalOneDist
    if temp(end)==T % jequalOneDist depends on T
        transpathoptions_ii.trivialjequalonedist=0;
        if N_semiz_ii>0
            error('jequalOneDist as a path (depending on t) is not yet implemented for semi-exogenous state transition paths')
        end
        if N_z_ii==0
            if N_e_ii==0
                jequalOneDist_ii=reshape(jequalOneDist_ii,[N_a_ii,T]);
            else
                jequalOneDist_ii=reshape(jequalOneDist_ii,[N_a_ii*N_e_ii,T]);
            end
        else
            if N_e_ii==0
                jequalOneDist_ii=reshape(jequalOneDist_ii,[N_a_ii*N_z_ii,T]);
            else
                jequalOneDist_ii=reshape(jequalOneDist_ii,[N_a_ii*N_z_ii*N_e_ii,T]);
            end
        end
        % As in TransitionPath_Case1_FHorz, jequalOneDist then holds the whole path (TransitionPath_FHorz_singlepathiter reads it as jequalOneDist_T)
    else
        transpathoptions_ii.trivialjequalonedist=1;
        if N_semiz_ii==0
            if N_z_ii==0
                if N_e_ii==0
                    jequalOneDist_ii=reshape(jequalOneDist_ii,[N_a_ii,1]);
                else
                    jequalOneDist_ii=reshape(jequalOneDist_ii,[N_a_ii*N_e_ii,1]);
                end
            else
                if N_e_ii==0
                    jequalOneDist_ii=reshape(jequalOneDist_ii,[N_a_ii*N_z_ii,1]);
                else
                    jequalOneDist_ii=reshape(jequalOneDist_ii,[N_a_ii*N_z_ii*N_e_ii,1]);
                end
            end
        else % semiz: age-1 dist over the composite state, listed (a,semiz,(z),(e)); the fastOLG dist raws place it at j=1 of their (a,j,semiz,z) storage
            if simoptions_ii.fastOLG==0
                jequalOneDist_ii=reshape(jequalOneDist_ii,[N_a_ii*N_bothze_ii,1]);
            else
                if N_z_ii==0 && N_e_ii==0
                    jequalOneDist_ii=reshape(jequalOneDist_ii,[N_asemiz_ii,1]);
                elseif N_e_ii==0
                    jequalOneDist_ii=reshape(jequalOneDist_ii,[N_asemiz_ii*N_z_ii,1]);
                elseif N_z_ii==0
                    jequalOneDist_ii=reshape(jequalOneDist_ii,[N_asemiz_ii*N_e_ii,1]);
                else
                    jequalOneDist_ii=reshape(jequalOneDist_ii,[N_asemiz_ii*N_z_ii*N_e_ii,1]);
                end
            end
        end
    end

    %% Figure out which functions are actually relevant to the present PType. And then change to FnsToEvaluate as cell so that it is not being recomputed all the time
    % A FnsToEvaluate that is a structure has one entry per ptype that it applies to; one that is a
    % function applies to all ptypes. Only the relevant ones are evaluated for this ptype.
    % WhichFnsForCurrentPType(kk) is the position of FnsToEvaluate kk in this ptype's list (0 if not relevant).
    l_daprime_ii=l_d_ii+l_a_ii;
    if vfoptions_ii.experienceasset>=1 || vfoptions_ii.experienceassetz>=1
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
                PTypeStructure.FnsAndPTypeIndicator(kk,ii)=1;
            end
            % otherwise this FnsToEvaluate is not relevant for the current PType
        else
            % If the Fn is not a structure (if it is a function) it is assumed to be relevant to all PTypes.
            FnsToEvaluate_ii.(FnNames{kk})=FnsToEvaluate.(FnNames{kk});
            WhichFnsForCurrentPType_ii(kk)=jj; jj=jj+1;
            PTypeStructure.FnsAndPTypeIndicator(kk,ii)=1;
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
        vfoptions_ii.policyind2val_finegridinput=1; % aprime_gridvals contains the fine grid for the first asset (tells PolicyInd2Val_FHorz_TPath)
    end

    %% Setup, the shapes of various of these objects vary depending on the setting (as in TransitionPath_FHorz_shooting)
    [PolicyIndexesPath_ii,N_probs_ii,II1_ii,II2_ii,exceptlastj_ii,exceptfirstj_ii,justfirstj_ii]=TransitionPath_FHorz_substeps_Step0_setup(l_d_ii,l_aprime_ii,N_a_ii,N_semiz_ii,N_z_ii,N_e_ii,N_j_ii,T,transpathoptions_ii,vfoptions_ii,simoptions_ii);
    % Semi-exogenous state: the form of pi_e_J the SemiExo value fn wants
    pi_e_J_vf_ii=pi_e_J_ii;
    if N_semiz_ii>0
        if transpathoptions_ii.fastOLG==1 && N_e_ii>0 && N_z_ii==0
            pi_e_J_vf_ii=reshape(pi_e_J_ii,[N_a_ii*N_j_ii,1,N_e_ii]); % SemiExo value fn keeps the bothz dim, so needs (a,j)-by-1-by-e even when N_z==0
        end
    end

    %% Each type is solved with TransitionPath_FHorz_singlepathiter, but evaluates no general eqm conditions
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
    PTypeStructure.(iistr).N_j=N_j_ii;
    PTypeStructure.(iistr).N_d=N_d_ii;
    PTypeStructure.(iistr).N_a=N_a_ii;
    PTypeStructure.(iistr).N_semiz=N_semiz_ii;
    PTypeStructure.(iistr).N_z=N_z_ii;
    PTypeStructure.(iistr).N_e=N_e_ii;
    PTypeStructure.(iistr).l_d=l_d_ii;
    PTypeStructure.(iistr).l_aprime=l_aprime_ii;
    PTypeStructure.(iistr).l_a=l_a_ii;
    PTypeStructure.(iistr).l_semiz=l_semiz_ii;
    PTypeStructure.(iistr).l_z=l_z_ii;
    PTypeStructure.(iistr).l_e=l_e_ii;
    PTypeStructure.(iistr).d_gridvals=d_gridvals_ii;
    PTypeStructure.(iistr).aprime_gridvals=aprime_gridvals_ii;
    PTypeStructure.(iistr).a_gridvals=a_gridvals_ii;
    PTypeStructure.(iistr).a_grid=a_grid_ii;
    PTypeStructure.(iistr).semiz_gridvals_J=semiz_gridvals_J_ii;
    PTypeStructure.(iistr).z_gridvals_J=z_gridvals_J_ii;
    PTypeStructure.(iistr).e_gridvals_J=e_gridvals_J_ii;
    PTypeStructure.(iistr).semizze_gridvals_J_fastOLG=semizze_gridvals_J_fastOLG_ii;
    PTypeStructure.(iistr).pi_semiz_J=pi_semiz_J_ii;
    PTypeStructure.(iistr).pi_z_J=pi_z_J_ii;
    PTypeStructure.(iistr).pi_e_J=pi_e_J_ii;
    PTypeStructure.(iistr).pi_e_J_vf=pi_e_J_vf_ii;
    PTypeStructure.(iistr).pi_semiz_J_sim=pi_semiz_J_sim_ii;
    PTypeStructure.(iistr).pi_z_J_sim=pi_z_J_sim_ii;
    PTypeStructure.(iistr).pi_e_J_sim=pi_e_J_sim_ii;
    PTypeStructure.(iistr).ReturnFn=ReturnFn_ii;
    PTypeStructure.(iistr).ReturnFnParamNames=ReturnFnParamNames_ii;
    PTypeStructure.(iistr).DiscountFactorParamNames=DiscountFactorParamNames_ii;
    PTypeStructure.(iistr).Parameters=Parameters_ii;
    PTypeStructure.(iistr).FnsToEvaluateCell=FnsToEvaluateCell_ii;
    PTypeStructure.(iistr).FnsToEvaluateParamNames=FnsToEvaluateParamNames_ii;
    PTypeStructure.(iistr).AggVarNames=AggVarNames_ii;
    PTypeStructure.(iistr).WhichFnsForCurrentPType=WhichFnsForCurrentPType_ii;
    PTypeStructure.(iistr).AgeWeights_T=AgeWeights_T_ii;
    PTypeStructure.(iistr).V_final=V_final_ii;
    PTypeStructure.(iistr).AgentDist_initial=AgentDist_initial_ii;
    PTypeStructure.(iistr).jequalOneDist=jequalOneDist_ii;
    PTypeStructure.(iistr).PolicyIndexesPath=PolicyIndexesPath_ii;
    PTypeStructure.(iistr).N_probs=N_probs_ii;
    PTypeStructure.(iistr).II1=II1_ii;
    PTypeStructure.(iistr).II2=II2_ii;
    PTypeStructure.(iistr).exceptlastj=exceptlastj_ii;
    PTypeStructure.(iistr).exceptfirstj=exceptfirstj_ii;
    PTypeStructure.(iistr).justfirstj=justfirstj_ii;
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
    [PricePath,GEcondnPathmatrix]=TransitionPath_FHorz_PType_Anderson(PricePath0, PricePathNames, PricePathSizeVec, PricePathSizeVec_ii, ParamPath, ParamPathNames, ParamPathSizeVec, T, FnNames, GEeqnNames, nGeneralEqmEqns_acrossptypes, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, transpathoptions, PTypeStructure);
elseif transpathoptions.GEnewprice==3 % the shooting algorithm
    [PricePath,GEcondnPathmatrix]=TransitionPath_Case1_FHorz_PType_shooting(PricePath0, PricePathNames, PricePathSizeVec, PricePathSizeVec_ii, ParamPath, ParamPathNames, ParamPathSizeVec, T, FnNames, GEeqnNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tplus1pricePathkk, tminus1AggVarsNames, transpathoptions, PTypeStructure);
end

%% Switch the solution into structure for output.
% A price is width-by-T (width is 1, or N_j for an age-dependent price). A price that depends on ptype
% is a structure with one such path per ptype, or with PricePathptype_vectoroutput=1 the N_i of them
% stacked in Names_i order ((N_i*width)-by-T).
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
