function varargout=TransitionPath_Case1_MixHorz_PType(PricePath0, ParamPath, T, V_final, AgentDist_initial, jequalOneDist, n_d, n_a, n_z, N_j, Names_i, d_grid,a_grid,z_grid, pi_z, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, AgeWeightsParamNames, PTypeDistParamNames, transpathoptions, simoptions, vfoptions)
% This code will work for all transition paths except those that involve at
% change in the transition matrix pi_z (can handle a change in pi_z, but
% only if it is a 'surprise', not anticipated changes) 
%
% PricePathOld is a structure with fields names being the Prices and each field containing a T-by-1 path.
% ParamPath is a structure with fields names being the parameter names of those parameters which change over the path and each field containing a T-by-1 path.

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
    %If transpathoptions is not given, just use all the defaults
    transpathoptions.tolerance=10^(-4);
    transpathoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    transpathoptions.GEnewprice=1; % 1 is shooting algorithm, 0 is that the GE should evaluate to zero and the 'new' is the old plus the "non-zero" (for each time period seperately); 
                                   % 2 is to do optimization routine with 'distance between old and new path', 3 is just same as 0, but easier to set up
    transpathoptions.GEptype={}; % zeros(1,length(fieldnames(GeneralEqmEqns))); % 1 indicates that this general eqm condition is 'conditional on permanent type' [input should be a cell of names; it gets reformatted internally to be this form]
    transpathoptions.PricePathptype_vectoroutput=0; % PricePath that depends on ptype defaults to being output as a structure
    transpathoptions.oldpathweight=0.9; % default =0.9
    transpathoptions.weightscheme=1; % default =1
    transpathoptions.Ttheta=1;
    transpathoptions.maxiter=500; % Based on personal experience anything that hasn't converged well before this is just hung-up on trying to get the 4th decimal place (typically because the number of grid points was not large enough to allow this level of accuracy).
    transpathoptions.verbose=0;
    transpathoptions.graphpricepath=0; % 1: creates a graph of the 'current' price path which updates each iteration.
    transpathoptions.graphaggvarspath=0; % 1: creates a graph of the 'current' aggregate variables which updates each iteration.
    transpathoptions.graphGEcondns=0;  % 1: creates a graph of the 'current' general eqm conditions which updates each iteration.
    transpathoptions.historyofpricepath=0;
    transpathoptions.stockvars=0;
    transpathoptions.fastOLG=0; % fastOLG is done as (a,j,z), rather than standard (a,z,j)
    % transpathoptions.updateageweights % Don't declare if not being used
else
    %Check transpathoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(transpathoptions,'tolerance')
        transpathoptions.tolerance=10^(-4);
    end
    if ~isfield(transpathoptions,'parallel')
        transpathoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(transpathoptions,'GEnewprice')
        transpathoptions.GEnewprice=1; % 0 is that the GE should evaluate to zero and the 'new' is the old plus the "non-zero" (for each time period seperately);
                                       % 1 is shooting algorithm, 
                                       % 2 is to do optimization routine with 'distance between old and new path'
                                       % 3 is just same as 0, but easier to set 
    end
    if ~isfield(transpathoptions,'GEptype')
        transpathoptions.GEptype={}; %zeros(1,length(fieldnames(GeneralEqmEqns))); % 1 indicates that this general eqm condition is 'conditional on permanent type'
    end
    if ~isfield(transpathoptions,'PricePathptype_vectoroutput')
        transpathoptions.PricePathptype_vectoroutput=0; % PricePath that depends on ptype defaults to being output as a structure
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
    if ~isfield(transpathoptions,'stockvars') % stockvars is solely for internal use, the user does not need to set it
        if ~isfield(transpathoptions,'stockvarinit') && ~isfield(transpathoptions,'stockvars') && ~isfield(transpathoptions,'stockvars')
            transpathoptions.stockvars=0;
        else
            transpathoptions.stockvars=1; % If stockvars has not itself been declared, but at least one of the stock variable options has then set stockvars to 1.
        end
    end
    if transpathoptions.stockvars==1 % Note: If this is not inputted then it is created by the above lines.
        if ~isfield(transpathoptions,'stockvarinit')
            error('transpathoptions includes some Stock Variable options but is missing stockvarinit \n')
        elseif ~isfield(transpathoptions,'stockvarpath0')
            error('transpathoptions includes some Stock Variable options but is missing stockvarpath0 \n')
        elseif ~isfield(transpathoptions,'stockvareqns')
            error('transpathoptions includes some Stock Variable options but is missing stockvareqns \n')
        end
    end
    if ~isfield(transpathoptions,'fastOLG')
        transpathoptions.fastOLG=0; % fastOLG is done as (a,j,z), rather than standard (a,z,j)
    end
    % transpathoptions.updateageweights %Don't declare if not being used
end

if transpathoptions.parallel~=2
    error('Sorry but transition paths are not implemented for cpu, you will need a gpu to use them')
end

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


%% Some internal commands require a few vfoptions and simoptions to be set
vfoptions.EVpre=0; % Not actually an option that can be used here
if ~isfield(vfoptions,'verbose')
    vfoptions.verbose=0;
end
if ~isfield(vfoptions,'experienceasset')
    for ii=1:N_i
        vfoptions.experienceasset.(Names_i{ii})=0;
    end
else
    % User created vfoptions.PType.experienceasset; we have Names_i
    for ii=1:N_i
        if isfield(vfoptions.experienceasset, Names_i{ii})
            if ~isfield(vfoptions,'aprimeFn') || ~isfield(vfoptions.aprimeFn, Names_i{ii})
                error('To use an experience asset you must define vfoptions.aprimeFn')
            end
        else
            vfoptions.experienceasset.(Names_i{ii})=0;
        end
    end
end
if ~isfield(vfoptions,'lowmemory')
    for ii=1:N_i
        vfoptions.lowmemory.(Names_i{ii})=0;
    end
else
    % User created vfoptions.PType.lowmemory; we have Names_i
    for ii=1:N_i
        if ~isfield(vfoptions.lowmemory, Names_i{ii})
            vfoptions.lowmemory.(Names_i{ii})=0;
        end
    end
end

%% Get AgeWeights from Parameters   
if isstruct(AgeWeightsParamNames)
    for ii=1:N_i
        try
            AgeWeights=Parameters.(AgeWeightsParamNames.(Names_i{ii}){1});
            if isstruct(N_j)
                if length(AgeWeights)~=N_j.(Names_i{ii})
                    error('Ageweights does not have age-like length')
                else
                    break
                end
            elseif isscalar(N_j)
                if length(AgeWeights)~=N_j
                    error('Ageweights does not have age-like length')
                else
                    break
                end
            else
                if length(AgeWeights)~=N_j(ii)
                    error('Ageweights does not have age-like length')
                else
                    break
                end
            end
        catch
            if ii==N_i
                error(['Failed to find parameter ', AgeWeightsParamNames.(Names_i{ii}){1}])
            end
        end
    end
else
    try
        AgeWeights=Parameters.(AgeWeightsParamNames{1});
    catch
        error(['Failed to find parameter ', AgeWeightsParamNames{1}])
    end
end
% Later, when creating PTypeStructure, we get the ptype-specific versions out of this and create an AgeWeights_T structure.

%%
[PricePath0,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec,PricePathSizeVec_ii,ParamPathSizeVec_ii]=PricePathParamPath_MixHorz_StructToMatrix(PricePath0,ParamPath,N_j,T,N_i);

if transpathoptions.verbose>1
    PricePathNames
    ParamPathNames
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


%% Fill out all of PTypeStructure
PTypeStructure.FnsAndPTypeIndicator=zeros(length(FnsToEvaluate),PTypeStructure.N_i,'gpuArray');

if transpathoptions.verbose==1
    fprintf('Setting up the permanent types for transition \n')
end

for ii=1:PTypeStructure.N_i

    iistr=PTypeStructure.Names_i{ii};
    PTypeStructure.iistr{ii}=iistr;
    
    if exist('vfoptions','var') % vfoptions.verbose (allowed to depend on permanent type)
        PTypeStructure.(iistr).vfoptions=PType_Options(vfoptions,Names_i,ii); % some vfoptions will differ by permanent type, will clean these up as we go before they are passed
    end
    
    if exist('simoptions','var') % vfoptions.verbose (allowed to depend on permanent type)
        PTypeStructure.(iistr).simoptions=PType_Options(simoptions,Names_i,ii); % some simoptions will differ by permanent type, will clean these up as we go before they are passed
    end
    
    % The parameter names can be made to depend on the permanent-type
    PTypeStructure.(iistr).DiscountFactorParamNames=DiscountFactorParamNames;
    if isa(DiscountFactorParamNames,'struct')
        PTypeStructure.(iistr).DiscountFactorParamNames=DiscountFactorParamNames.(Names_i{ii});
    end

    % Need to fill in some defaults
    PTypeStructure.(iistr).vfoptions.parallel=2; % hardcode
    PTypeStructure.(iistr).simoptions.parallel=2; % hardcode
    if ~isfield(PTypeStructure.(iistr).vfoptions,'n_e')
        PTypeStructure.(iistr).n_e=0;
    else
        PTypeStructure.(iistr).n_e=PTypeStructure.(iistr).vfoptions.n_e;
    end
    if ~isfield(PTypeStructure.(iistr).vfoptions,'divideandconquer')
        PTypeStructure.(iistr).vfoptions.divideandconquer=0; %default
    else
        if PTypeStructure.(iistr).vfoptions.divideandconquer==1
            PTypeStructure.(iistr).vfoptions.level1n=ceil(n_a.(iistr)/50); % default
        end
    end
    if ~isfield(PTypeStructure.(iistr).vfoptions,'gridinterplayer')
        PTypeStructure.(iistr).vfoptions.gridinterplayer=0; %default
    end
    if ~isfield(PTypeStructure.(iistr).simoptions,'gridinterplayer')
        PTypeStructure.(iistr).simoptions.gridinterplayer=0; %default
    end
    % Model setup
    if ~isfield(PTypeStructure.(iistr).vfoptions,'exoticpreferences')
        PTypeStructure.(iistr).vfoptions.exoticpreferences='None'; % not yet implemented, so hardcodes None
    else
        if ~strcmp(PTypeStructure.(iistr).vfoptions.exoticpreferences,'None')
            error('transition paths cannot yet handle exoticpreferences')
        end
    end
    if ~isfield(PTypeStructure.(iistr).vfoptions,'experienceasset')
        PTypeStructure.(iistr).vfoptions.experienceasset=0;
    end
    if ~isfield(PTypeStructure.(iistr).simoptions,'experienceasset')
        PTypeStructure.(iistr).simoptions.experienceasset=0;
    end

    % Go through everything which might be dependent on permanent type (PType)
    % Notice that the way this is coded the grids (etc.) could be either
    % fixed, or a function (that depends on age, and possibly on permanent
    % type), or they could be a structure. Only in the case where they are
    % a structure is there a need to take just a specific part and send
    % only that to the 'non-PType' version of the command.
    
    % Start with those that determine whether the current permanent type is finite or
    % infinite horizon, and whether it is Case 1 or Case 2
    % Figure out which case is relevant to the current PType. This is done
    % using N_j which for the current type will evaluate to 'Inf' if it is
    % infinite horizon and a finite number for any other finite horizon.
    % First, check if it is a structure, and otherwise just get the
    % relevant value.
    
    % Horizon is determined via N_j
    if isstruct(N_j)
        PTypeStructure.(iistr).N_j=N_j.(iistr);
    elseif isscalar(N_j)
        PTypeStructure.(iistr).N_j=N_j;
    else
        PTypeStructure.(iistr).N_j=N_j(ii);
    end
    
    if isa(n_d,'struct')
        PTypeStructure.(iistr).n_d=n_d.(iistr);
    else
        PTypeStructure.(iistr).n_d=n_d;
    end
    N_d=prod(PTypeStructure.(iistr).n_d);
    PTypeStructure.(iistr).N_d=N_d;
    if N_d==0
        PTypeStructure.(iistr).l_d=0;
    else
        PTypeStructure.(iistr).l_d=length(PTypeStructure.(iistr).n_d);
    end
    if isa(n_a,'struct')
        PTypeStructure.(iistr).n_a=n_a.(iistr);
    else
        PTypeStructure.(iistr).n_a=n_a;
    end
    N_a=prod(PTypeStructure.(iistr).n_a);
    PTypeStructure.(iistr).N_a=N_a;
    PTypeStructure.(iistr).l_a=length(PTypeStructure.(iistr).n_a);
    PTypeStructure.(iistr).l_aprime=PTypeStructure.(iistr).l_a;
    if PTypeStructure.(iistr).vfoptions.experienceasset==1
        PTypeStructure.(iistr).l_aprime=PTypeStructure.(iistr).l_aprime-1;
    end
    if isa(n_z,'struct')
        PTypeStructure.(iistr).n_z=n_z.(iistr);
    else
        PTypeStructure.(iistr).n_z=n_z;
    end
    N_z=prod(PTypeStructure.(iistr).n_z);
    PTypeStructure.(iistr).N_z=N_z;
    if N_z==0
        PTypeStructure.(iistr).l_z=0;
    else
        PTypeStructure.(iistr).l_z=length(n_z);
    end
    N_e=prod(PTypeStructure.(iistr).n_e);
    PTypeStructure.(iistr).N_e=N_e;
    if N_e==0
        PTypeStructure.(iistr).l_e=0;
    else
        PTypeStructure.(iistr).l_e=length(PTypeStructure.(iistr).n_e);
    end

    if isa(d_grid,'struct')
        PTypeStructure.(iistr).d_grid=gpuArray(d_grid.(iistr));
    else
        PTypeStructure.(iistr).d_grid=gpuArray(d_grid);
    end
    if isa(a_grid,'struct')
        PTypeStructure.(iistr).a_grid=gpuArray(a_grid.(iistr));
    else
        PTypeStructure.(iistr).a_grid=gpuArray(a_grid);
    end
    if isa(z_grid,'struct')
        PTypeStructure.(iistr).z_grid=gpuArray(z_grid.(iistr));
    else
        PTypeStructure.(iistr).z_grid=gpuArray(z_grid);
    end

    % to be able to EvalFnsOnAgentDist using fastOLG we also need
    PTypeStructure.(iistr).a_gridvals=gpuArray(CreateGridvals(PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).a_grid,1)); % a_grivdals is [N_a,l_a]
    % use fine grid for aprime_gridvals
    if PTypeStructure.(iistr).vfoptions.gridinterplayer==0
        PTypeStructure.(iistr).aprime_gridvals=PTypeStructure.(iistr).a_gridvals;
    elseif PTypeStructure.(iistr).vfoptions.gridinterplayer==1
        if isscalar(PTypeStructure.(iistr).n_a)
            n_aprime=PTypeStructure.(iistr).n_a+(PTypeStructure.(iistr).n_a-1)*PTypeStructure.(iistr).vfoptions.ngridinterp;
            aprime_grid=interp1(gpuArray(1:1:PTypeStructure.(iistr).N_a)',PTypeStructure.(iistr).a_grid,gpuArray(linspace(1,PTypeStructure.(iistr).N_a,n_aprime))');
            PTypeStructure.(iistr).aprime_gridvals=CreateGridvals(n_aprime,aprime_grid,1);
        else
            a1_grid=PTypeStructure.(iistr).a_grid(1:PTypeStructure.(iistr).n_a(1));
            n_a1prime=PTypeStructure.(iistr).n_a(1)+(PTypeStructure.(iistr).n_a(1)-1)*PTypeStructure.(iistr).vfoptions.ngridinterp;
            n_aprime=[n_a1prime,PTypeStructure.(iistr).n_a(2:end)];
            a1prime_grid=interp1(gpuArray(1:1:PTypeStructure.(iistr).n_a(1))',a1_grid,gpuArray(linspace(1,PTypeStructure.(iistr).n_a(1),n_a1prime))');
            aprime_grid=[a1prime_grid; PTypeStructure.(iistr).a_grid(PTypeStructure.(iistr).n_a(1)+1:end)];
            PTypeStructure.(iistr).aprime_gridvals=CreateGridvals(n_aprime,aprime_grid,1);
        end
    end
    PTypeStructure.(iistr).d_gridvals=CreateGridvals(PTypeStructure.(iistr).n_d,gpuArray(PTypeStructure.(iistr).d_grid),1);
    % if N_d==0
    %     PTypeStructure.(iistr).daprime_gridvals=gpuArray(PTypeStructure.(iistr).a_gridvals);
    % else
    %     PTypeStructure.(iistr).daprime_gridvals=gpuArray([kron(ones(N_a,1),CreateGridvals(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).d_grid,1)), kron(PTypeStructure.(iistr).a_gridvals,ones(PTypeStructure.(iistr).N_d,1))]); % daprime_gridvals is [N_d*N_aprime,l_d+l_aprime]
    % end

    if isa(pi_z,'struct')
        PTypeStructure.(iistr).pi_z=pi_z.(Names_i{ii}); % Different grids by permanent type, but not depending on age. (same as the case just above; this case can occour with or without the existence of vfoptions, as long as there is no vfoptions.agedependentgrids)
    else
        PTypeStructure.(iistr).pi_z=pi_z;
    end

    % If using any non-standard endogenous states, setup for those (both FHorz and InfHorz btw)
    [PTypeStructure.(iistr).vfoptions,PTypeStructure.(iistr).simoptions]=SetupNonStandardEndoStates_FHorz_TPath(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).d_grid,PTypeStructure.(iistr).a_grid,PTypeStructure.(iistr).vfoptions,PTypeStructure.(iistr).simoptions);

    PTypeStructure.(iistr).ReturnFn=ReturnFn;
    if isa(ReturnFn,'struct')
        PTypeStructure.(iistr).ReturnFn=ReturnFn.(Names_i{ii});
    end
    
    % Parameters are allowed to be given as structure, or as vector/matrix (in terms of their dependence on permanent type). 
    % So go through each of these in term.
    % ie. Parameters.alpha=[0;1]; or Parameters.alpha.ptype1=0; Parameters.alpha.ptype2=1;
    PTypeStructure.(iistr).Parameters=Parameters;
    FullParamNames=fieldnames(Parameters); % all the different parameters
    nFields=length(FullParamNames);
    for kField=1:nFields
        if isa(Parameters.(FullParamNames{kField}), 'struct') % Check the current parameter for permanent type in structure form
            % Check if this parameter is used for the current permanent type (it may or may not be, some parameters are only used be a subset of permanent types)
            if isfield(Parameters.(FullParamNames{kField}),Names_i{ii})
                PTypeStructure.(iistr).Parameters.(FullParamNames{kField})=Parameters.(FullParamNames{kField}).(Names_i{ii});
            end
        elseif sum(size(Parameters.(FullParamNames{kField}))==PTypeStructure.N_i)>=1 % Check for permanent type in vector/matrix form.
            temp=Parameters.(FullParamNames{kField});
            [~,ptypedim]=max(size(Parameters.(FullParamNames{kField}))==PTypeStructure.N_i); % Parameters as vector/matrix can be at most two dimensional, figure out which relates to PType, it should be the row dimension, if it is not then give a warning.
            if ptypedim==1
                PTypeStructure.(iistr).Parameters.(FullParamNames{kField})=temp(ii,:);
            elseif ptypedim==2
                if ~strcmp(FullParamNames{kField},PTypeDistParamNames{1})
                    sprintf('Possible Warning: some parameters appear to have been imputted with dependence on permanent type indexed by column rather than row \n')
                    sprintf(['Specifically, parameter: ', FullParamNames{kField}, ' \n'])
                    sprintf('(it is possible this is just a coincidence of number of columns) \n')
                    dbstack
                end
            end
        end
    end
    % THIS TREATMENT OF PARAMETERS COULD BE IMPROVED TO BETTER DETECT INPUT SHAPE ERRORS.
    PTypeStructure.ParametersRaw=Parameters; % For use in General eqm conditions (as we might want them across ptypes for some purposes)
    
    
    % The parameter names can be made to depend on the permanent-type
    PTypeStructure.(iistr).DiscountFactorParamNames=DiscountFactorParamNames;
    if isa(DiscountFactorParamNames,'struct')
        PTypeStructure.(iistr).DiscountFactorParamNames=DiscountFactorParamNames.(Names_i{ii});
    end

    % Implement new way of handling ReturnFn inputs (note l_d, l_a, l_z are just created for this and then not used for anything else later)
    if PTypeStructure.(iistr).n_d(1)==0
        l_d=0;
    else
        l_d=length(PTypeStructure.(iistr).n_d);
    end
    if PTypeStructure.(iistr).vfoptions.experienceasset
        l_aprime=length(PTypeStructure.(iistr).vfoptions.setup_experienceasset.n_a1);
        l_a=l_aprime+length(PTypeStructure.(iistr).vfoptions.setup_experienceasset.n_a2);
    else
        l_aprime=length(PTypeStructure.(iistr).n_a);
        l_a=l_aprime;
    end
    l_z=length(PTypeStructure.(iistr).n_z);
    if PTypeStructure.(iistr).N_z==0
        l_z=0;
    end
    if isfield(PTypeStructure.(iistr).vfoptions,'SemiExoStateFn')
        l_z=l_z+length(PTypeStructure.(iistr).vfoptions.n_semiz);
    end
    l_e=PTypeStructure.(iistr).l_e;

    %% Implement new way of handling ReturnFn inputs
    ReturnFnParamNames=ReturnFnParamNamesFn(PTypeStructure.(iistr).ReturnFn,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).vfoptions,Parameters);
    PTypeStructure.(iistr).ReturnFnParamNames=ReturnFnParamNames;
    

    %% Figure out which functions are actually relevant to the present PType. And then change to FnsToEvaluate as cell so that it is not being recomputed all the time 
    % Only the relevant ones need to be evaluated.
    % The dependence of FnsToEvaluateFn and FnsToEvaluateFnParamNames are necessarily the same.

    FnNames=fieldnames(FnsToEvaluate);
    PTypeStructure.numFnsToEvaluate=length(fieldnames(FnsToEvaluate));
    PTypeStructure.(iistr).WhichFnsForCurrentPType=zeros(PTypeStructure.numFnsToEvaluate,1);
    jj=1; % jj indexes the FnsToEvaluate that are relevant to the current PType
    for kk=1:PTypeStructure.numFnsToEvaluate
        if isa(FnsToEvaluate.(FnNames{kk}),'struct')
            if isfield(FnsToEvaluate.(FnNames{kk}), Names_i{ii})
                PTypeStructure.(iistr).FnsToEvaluate.(FnNames{kk})=FnsToEvaluate.(FnNames{kk}).(Names_i{ii});
                % % Figure out FnsToEvaluateParamNames
                % temp=getAnonymousFnInputNames(FnsToEvaluate.(FnNames{kk}).(Names_i{ii}));
                % PTypeStructure.(iistr).FnsToEvaluateParamNames(jj).Names={temp{l_d+l_a+l_a+l_z+l_e+1:end}}; % the first inputs will always be (d,aprime,a,z)
                PTypeStructure.(iistr).WhichFnsForCurrentPType(kk)=jj; jj=jj+1;
                PTypeStructure.FnsAndPTypeIndicator(kk,ii)=1;
                % else
                %  % do nothing as this FnToEvaluate is not relevant for the current PType
                % % Implicitly, PTypeStructure.(iistr).WhichFnsForCurrentPType(kk)=0
            end
        else
            % If the Fn is not a structure (if it is a function) it is assumed to be relevant to all PTypes.
            PTypeStructure.(iistr).FnsToEvaluate.(FnNames{kk})=FnsToEvaluate.(FnNames{kk});
            % Figure out FnsToEvaluateParamNames
            temp=getAnonymousFnInputNames(FnsToEvaluate.(FnNames{kk}));
            PTypeStructure.(iistr).FnsToEvaluateParamNames(jj).Names={temp{l_d+l_a+l_a+l_z+l_e+1:end}}; % the first inputs will always be (d,aprime,a,z)
            PTypeStructure.(iistr).WhichFnsForCurrentPType(kk)=jj; jj=jj+1;
            PTypeStructure.FnsAndPTypeIndicator(kk,ii)=1;
        end
    end
    % Now that all the relevant FnsToEvaluate for type ii are in PTypeStructure.(iistr).FnsToEvaluate
    PTypeStructure.(iistr).l_daprime=PTypeStructure.(iistr).l_d+PTypeStructure.(iistr).l_aprime;
    PTypeStructure.(iistr).AggVarNames=fieldnames(PTypeStructure.(iistr).FnsToEvaluate);
    PTypeStructure.(iistr).FnsToEvaluateCell=cell(1,length(PTypeStructure.(iistr).AggVarNames));
    for ff=1:length(PTypeStructure.(iistr).AggVarNames)
        temp=getAnonymousFnInputNames(PTypeStructure.(iistr).FnsToEvaluate.(PTypeStructure.(iistr).AggVarNames{ff}));
        if length(temp)>(PTypeStructure.(iistr).l_daprime+PTypeStructure.(iistr).l_a+PTypeStructure.(iistr).l_z+PTypeStructure.(iistr).l_e)
            PTypeStructure.(iistr).FnsToEvaluateParamNames(ff).Names={temp{PTypeStructure.(iistr).l_daprime+PTypeStructure.(iistr).l_a+PTypeStructure.(iistr).l_z+PTypeStructure.(iistr).l_e+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            PTypeStructure.(iistr).FnsToEvaluateParamNames(ff).Names={};
        end
        PTypeStructure.(iistr).FnsToEvaluateCell{ff}=PTypeStructure.(iistr).FnsToEvaluate.(PTypeStructure.(iistr).AggVarNames{ff});
    end
    % Change FnsToEvaluate out of structure form, but want to still create AggVars as a structure
    PTypeStructure.(iistr).simoptions.outputasstructure=1;

    %% Check if pi_z and z_grid can be pre-computed
    % Note: cannot handle that whether not not they can be pre-computed differs across ptypes
    transpathoptions.zpathprecomputed=1;
    transpathoptions.zpathtrivial=1; % z_grid_J and pi_z_J are not varying over the path
    if isfield(PTypeStructure.(iistr).vfoptions,'ExogShockFn')
        transpathoptions.zpathprecomputed=0;
        N_z=prod(PTypeStructure.(iistr).n_z);
        % Note: If ExogShockFn depends on the path, it must be done via a parameter
        % that depends on the path (i.e., via ParamPath or PricePath)
        PTypeStructure.(iistr).vfoptions.ExogShockFnParamNames=getAnonymousFnInputNames(PTypeStructure.(iistr).vfoptions.ExogShockFn);
        overlap=0;
        for pp=1:length(PTypeStructure.(iistr).vfoptions.ExogShockFnParamNames)
            if strcmp(PTypeStructure.(iistr).vfoptions.ExogShockFnParamNames{pp},PricePathNames)
                overlap=1;
            end
        end
        if overlap==0
            transpathoptions.zpathprecomputed=1;
            % If ExogShockFn does not depend on any of the prices (in PricePath), then
            % we can simply create it now rather than within each 'subfn' or 'p_grid'

            % Check if it depends on the ParamPath
            transpathoptions.zpathtrivial=1;
            for pp=1:length(PTypeStructure.(iistr).vfoptions.ExogShockFnParamNames)
                if strcmp(PTypeStructure.(iistr).vfoptions.ExogShockFnParamNames{pp},ParamPathNames)
                    transpathoptions.zpathtrivial=0;
                end
            end
            if transpathoptions.zpathtrivial==1
                ExogShockFnParamsVec=CreateVectorFromParams(PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).vfoptions.ExogShockFnParamNames);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for pp=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(pp,1)={ExogShockFnParamsVec(pp)};
                end
                [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
            elseif transpathoptions.zpathtrivial==0
                % z_grid and/or pi_z varies along the transition path (but only depending on ParamPath, not PricePath
                transpathoptions.(iistr).pi_z_T=zeros(N_z,N_z,T,'gpuArray');
                transpathoptions.(iistr).z_grid_T=zeros(sum(n_z),T,'gpuArray');
                for tt=1:T
                    for pp=1:length(ParamPathNames)
                        PTypeStructure.(iistr).Parameters.(ParamPathNames{pp})=ParamPathStruct.(ParamPathNames{pp});
                    end
                    % Note, we know the PricePath is irrelevant for the current purpose
                    ExogShockFnParamsVec=CreateVectorFromParams(PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).vfoptions.ExogShockFnParamNames);
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for pp=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(pp,1)={ExogShockFnParamsVec(pp)};
                    end
                    [z_grid,pi_z]=PTypeStructure.(iistr).vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
                    transpathoptions.(iistr).pi_z_T(:,:,tt)=pi_z;
                    transpathoptions.(iistr).z_grid_T(:,tt)=z_grid;
                end
            end
        end
    end
    %% If used, check if pi_e and e_grid can be pre-computed
    % Note: cannot handle that whether not not they can be pre-computed differs across ptypes
    if l_e>0
        % Check if e_grid and/or pi_e depend on prices. If not then create pi_e_J and e_grid_J for the entire transition before we start
        transpathoptions.epathprecomputed=1;
        transpathoptions.epathtrivial=1; % e_grid_J and pi_e_J are not varying over the path
        if isfield(PTypeStructure.(iistr).vfoptions,'EiidShockFn')
            transpathoptions.epathprecomputed=0;
            % Note: If EiidShockFn depends on the path, it must be done via a parameter
            % that depends on the path (i.e., via ParamPath or PricePath)
            PTypeStructure.(iistr).vfoptions.EiidShockFnParamNames=getAnonymousFnInputNames(PTypeStructure.(iistr).vfoptions.EiidShockFn);
            overlap=0;
            for pp=1:length(PTypeStructure.(iistr).vfoptions.EiidShockFnParamNames)
                if strcmp(PTypeStructure.(iistr).vfoptions.EiidShockFnParamNames{pp},PricePathNames)
                    overlap=1;
                end
            end
            if overlap==0
                transpathoptions.epathprecomputed=1;
                % If ExogShockFn does not depend on any of the prices (in PricePath), then
                % we can simply create it now rather than within each 'subfn' or 'p_grid'

                % Check if it depends on the ParamPath
                transpathoptions.epathtrivial=1;
                for pp=1:length(PTypeStructure.(iistr).vfoptions.EiidShockFnParamNames)
                    if strcmp(PTypeStructure.(iistr).vfoptions.EiidShockFnParamNames{pp},ParamPathNames)
                        transpathoptions.epathtrivial=0;
                    end
                end
                if transpathoptions.epathtrivial==1
                    EiidShockFnParamsVec=CreateVectorFromParams(PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).vfoptions.EiidShockFnParamNames);
                    EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                    for pp=1:length(EiidShockFnParamsVec)
                        EiidShockFnParamsCell(pp,1)={EiidShockFnParamsVec(pp)};
                    end
                    [e_grid,pi_e]=PTypeStructure.(iistr).vfoptions.EiidShockFn(EiidShockFnParamsCell{:});
                    % Now store them in vfoptions and simoptions
                    PTypeStructure.(iistr).vfoptions.pi_e=pi_e;
                    PTypeStructure.(iistr).vfoptions.e_grid=e_grid;
                    PTypeStructure.(iistr).simoptions.pi_e=pi_e;
                    PTypeStructure.(iistr).simoptions.e_grid=e_grid;
                elseif transpathoptions.epathtrivial==0
                    % e_grid_T and/or pi_e_T varies along the transition path (but only depending on ParamPath, not PricePath)
                    transpathoptions.(iistr).pi_e_T=zeros(N_e,T,'gpuArray');
                    transpathoptions.(iistr).e_grid_T=zeros(sum(PTypeStructure.(iistr).n_e),T,'gpuArray');
                    for tt=1:T
                        for pp=1:length(ParamPathNames)
                            PTypeStructure.(iistr).Parameters.(ParamPathNames{pp})=ParamPathStruct.(ParamPathNames{pp});
                        end
                        % Note, we know the PricePath is irrelevant for the current purpose
                        EiidShockFnParamsVec=CreateVectorFromParams(PTypeStructure.(iistr).Parameters, vPTypeStructure.(iistr).foptions.EiidShockFnParamNames);
                        EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                        for pp=1:length(ExogShockFnParamsVec)
                            EiidShockFnParamsCell(pp,1)={EiidShockFnParamsVec(pp)};
                        end
                        [e_grid,pi_e]=PTypeStructure.(iistr).vfoptions.ExogShockFn(EiidShockFnParamsCell{:});

                        transpathoptions.(iistr).pi_e_T(:,tt)=pi_e;
                        transpathoptions.(iistr).e_grid_T(:,tt)=e_grid;
                    end
                end
            end
        end
    end

    %% If using any non-standard endogenous states, setup for those (both FHorz and InfHorz btw)
    [PTypeStructure.(iistr).vfoptions,PTypeStructure.(iistr).simoptions]=SetupNonStandardEndoStates_FHorz_TPath(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).d_grid,PTypeStructure.(iistr).a_grid,PTypeStructure.(iistr).vfoptions,PTypeStructure.(iistr).simoptions);

    N_j_temp=PTypeStructure.(iistr).N_j;
    if ~isfinite(N_j_temp)
        % Reshape V_final
        % If no z, then N_z=1 here
        V_final.(iistr)=reshape(V_final.(iistr),[N_a,N_z]);

        % Presume all InfHorz ExogShock functions happen only with respect to transition paths (i.e., z_grid_T/e_grid_T and pi_z_T/pi_e_T
    else
        %% Set up exogenous shock processes
        [PTypeStructure.(iistr).z_gridvals_J, PTypeStructure.(iistr).pi_z_J, PTypeStructure.(iistr).pi_z_J_sim, PTypeStructure.(iistr).e_gridvals_J, PTypeStructure.(iistr).pi_e_J, PTypeStructure.(iistr).pi_e_J_sim, PTypeStructure.(iistr).ze_gridvals_J_fastOLG, transpathoptions, PTypeStructure.(iistr).simoptions]=ExogShockSetup_TPath_FHorz(PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).z_grid,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).N_a,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).Parameters,PricePathNames,ParamPathNames,transpathoptions,PTypeStructure.(iistr).simoptions,4);
        % Convert z and e to age-dependent joint-grids and transtion matrix
        % output: z_gridvals_J, pi_z_J, e_gridvals_J, pi_e_J, transpathoptions,vfoptions,simoptions

        %% Organise V_final and AgentDist_initial
        % Reshape V_final
        if N_z==0
            if N_e==0
                V_final.(iistr)=reshape(V_final.(iistr),[N_a,N_j_temp]);
            else
                V_final.(iistr)=reshape(V_final.(iistr),[N_a,N_e,N_j_temp]);
            end
        else
            if N_e==0
                V_final.(iistr)=reshape(V_final.(iistr),[N_a,N_z,N_j_temp]);
            else
                V_final.(iistr)=reshape(V_final.(iistr),[N_a,N_z,N_e,N_j_temp]);
            end
        end
        if transpathoptions.fastOLG==1
            if N_z==0
                if N_e==0
                    % Already reshaped
                    % V_final.(iistr)=reshape(V_final.(iistr),[N_a,N_j_temp]);
                else
                    V_final.(iistr)=reshape(permute(V_final.(iistr),[1,3,2]),[N_a*N_j_temp,N_e]);
                end
            else
                if N_e==0
                    V_final.(iistr)=reshape(permute(V_final.(iistr),[1,3,2]),[N_a*N_j_temp,N_z]);
                else
                    V_final.(iistr)=reshape(permute(V_final.(iistr),[1,4,2,3]),[N_a*N_j_temp,N_z,N_e]);
                end
            end
        end

        % Reshape AgentDist_initial and turn AgeWeights_T into appropriate size so that we can always just do AgentDist.*AgeWeights
        % Note when simoptions.fastOLG==1 we have shapes of [N_a*N_j_temp*whatever,1-or-N_e] instead of [N_a,whatever-and-maybe-N_e,N_j_temp]
        AgentDist_init=AgentDist_initial.(iistr);
        if N_z==0
            if N_e==0
                AgentDist_init=reshape(AgentDist_init,[N_a,N_j_temp]); % if simoptions.fastOLG==0
                AgeWeights_init=sum(AgentDist_init,1); % [1,N_j]
                if PTypeStructure.(iistr).simoptions.fastOLG
                    AgentDist_init=reshape(AgentDist_init,[N_a*N_j_temp,1]);
                    AgeWeights_init=repelem(AgeWeights_init',N_a,1);
                end
            else
                if N_e==0
                    V_final.(iistr)=reshape(permute(V_final.(iistr),[1,3,2]),[N_a*N_j_temp,N_z]);
                else
                    V_final.(iistr)=reshape(permute(V_final.(iistr),[1,4,2,3]),[N_a*N_j_temp,N_z,N_e]);
                end
            end
        end
        % Reshape AgentDist_initial
        if N_z==0
            if N_e==0
                AgentDist_initial.(iistr)=reshape(AgentDist_initial.(iistr),[N_a,N_j_temp]); % if simoptions.fastOLG==0
                AgeWeights_init.(iistr)=sum(AgentDist_initial.(iistr),1); % [1,N_j_ii]
                if PTypeStructure.(iistr).simoptions.fastOLG==1
                    AgentDist_initial.(iistr)=reshape(AgentDist_initial.(iistr),[N_a*N_j_temp,1]);
                    AgeWeights_init.(iistr)=repelem(AgeWeights_init.(iistr)',N_a,1);
                end
            else
                AgentDist_initial.(iistr)=reshape(AgentDist_initial.(iistr),[N_a*N_e,N_j_temp]); % if simoptions.fastOLG==0
                AgeWeights_init.(iistr)=sum(AgentDist_initial.(iistr),1); % [1,N_j_ii]
                if PTypeStructure.(iistr).simoptions.fastOLG==1 % simoptions.fastOLG==1, so AgentDist is treated as : (a,j,z)-by-1
                    AgentDist_initial.(iistr)=reshape(permute(reshape(AgentDist_initial.(iistr),[N_a,N_e,N_j_temp]),[1,3,2]),[N_a*N_j_temp*N_e,1]);
                    AgeWeights_init.(iistr)=repelem(AgeWeights_init.(iistr)',N_a,1);
                end
            end
        else
            if N_e==0
                AgentDist_initial.(iistr)=reshape(AgentDist_initial.(iistr),[N_a*N_z,N_j_temp]); % if simoptions.fastOLG==0
                AgeWeights_init.(iistr)=sum(AgentDist_initial.(iistr),1); % [1,N_j_ii]
                if PTypeStructure.(iistr).simoptions.fastOLG==1 % simoptions.fastOLG==1, so AgentDist is treated as : (a,j,z)-by-1
                    AgentDist_initial.(iistr)=reshape(permute(reshape(AgentDist_initial.(iistr),[N_a,N_z,N_j_temp]),[1,3,2]),[N_a*N_j_temp*N_z,1]);
                    AgeWeights_init.(iistr)=repelem(AgeWeights_init.(iistr)',N_a,1);
                end
            else
                AgentDist_initial.(iistr)=reshape(AgentDist_initial.(iistr),[N_a*N_z*N_e,N_j_temp]); % if simoptions.fastOLG==0
                AgeWeights_init.(iistr)=sum(AgentDist_initial.(iistr),1); % [1,N_j_ii]
                if PTypeStructure.(iistr).simoptions.fastOLG==1 % simoptions.fastOLG==1, so AgentDist is treated as : (a,j,z)-by-1
                    AgentDist_initial.(iistr)=reshape(permute(reshape(AgentDist_initial.(iistr),[N_a,N_z,N_e,N_j_temp]),[1,4,2,3]),[N_a*N_j_temp*N_z,N_e]);
                    AgeWeights_init.(iistr)=repelem(AgeWeights_init.(iistr)',N_a,1);
                end
            end
        end
        % Get AgeWeights and switch into the transpathoptions.ageweightstrivial=0 setup (and this is what subfns hardcode when doing PTypes)
        % It is assumed there is only one Age Weight Parameter (name))
        % AgeWeights_T is (a,j,z)-by-T (create as j-by-T to start, then switch)
        if isstruct(AgeWeights)
            AgeWeights_ii=AgeWeights.(iistr);
            if all(size(AgeWeights_ii)==[N_j_temp,1])
                % Does not depend on transition path period
                PTypeStructure.(iistr).AgeWeights_T=gather(AgeWeights_ii.*ones(1,T));
            elseif all(size(AgeWeights)==[1,N_j_temp])
                % Does not depend on transition path period
                PTypeStructure.(iistr).AgeWeights_T=gather(AgeWeights_ii'.*ones(1,T));
            else
                fprintf('Following error applies to agent permanent type: %s \n',Names_i{ii})
                error('The age weights parameter seems to be the wrong size')
            end
        else % not a structure, so must apply to all permanent types
            if all(size(AgeWeights)==[N_j_temp,1])
                % Does not depend on transition path period
                PTypeStructure.(iistr).AgeWeights_T=gather(AgeWeights.*ones(1,T));
            elseif all(size(AgeWeights)==[1,N_j_temp])
                % Does not depend on transition path period
                PTypeStructure.(iistr).AgeWeights_T=gather(AgeWeights'.*ones(1,T));
            else
                error('The age weights parameter seems to be the wrong size')
            end
        end
        % Check ParamPath to see if the AgeWeights vary over the transition
        % (and overwrite PTypeStructure.(iistr).AgeWeights_T if it does)
        if isstruct(AgeWeightsParamNames)
            temp=strcmp(ParamPathNames,AgeWeightsParamNames.(iistr){1});
        else
            temp=strcmp(ParamPathNames,AgeWeightsParamNames{1});
        end
        if any(temp)
            transpathoptions.ageweightstrivial=0; % AgeWeights vary over the transition
            [~,kk]=max(temp); % Get index for the AgeWeightsParamNames in ParamPathNames
            % Create AgeWeights_T
            PTypeStructure.(iistr).AgeWeights_T=ParamPath(:,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk))'; % This will always be N_j_ii-by-T (as transpose)
            % Note: still leave it in ParamPath just in case it is used in AggVars or somesuch
        end
        % Because ptypes hardcodes transpathoptions.ageweightstrivial=0, we need
        if PTypeStructure.(iistr).simoptions.fastOLG==1
            if N_z==0
                PTypeStructure.(iistr).AgeWeights_T=repelem(PTypeStructure.(iistr).AgeWeights_T,N_a,1); % simoptions.fastOLG=1 so this is (a,j)-by-1
            else
                PTypeStructure.(iistr).AgeWeights_T=repmat(repelem(PTypeStructure.(iistr).AgeWeights_T,N_a,1),N_z,1); % simoptions.fastOLG=1 so this is (a,j,z)-by-1
            end
        end
    
        %% Set up jequalOneDist_T.(iistr) [hardcodes transpathoptions.trivialjequalonedist=0 and simoptions.fastOLG=1]
        if ~isstruct(jequalOneDist)
            jequalOneDist_temp=gpuArray(jequalOneDist);
        else % jequalOneDist is a structure
            jequalOneDist_temp=gpuArray(jequalOneDist.(iistr));
        end
        % Check if jequalOneDistPath is a path or not (and reshape appropriately)
        temp=size(jequalOneDist_temp);
        if temp(end)==T % jequalOneDist depends on T
            transpathoptions.(iistr).trivialjequalonedist=0;
            if N_z==0
                if N_e==0
                    jequalOneDist_temp=reshape(jequalOneDist_temp,[N_a,T]);
                else
                    jequalOneDist_temp=reshape(jequalOneDist_temp,[N_a*N_e,T]); % simoptions.fastOLG==1
                end
            else
                if N_e==0
                    jequalOneDist_temp=reshape(jequalOneDist_temp,[N_a*N_z,T]); % simoptions.fastOLG==1
                else
                    jequalOneDist_temp=reshape(jequalOneDist_temp,[N_a*N_z*N_e,T]); % simoptions.fastOLG==1
                end
            end
            PTypeStructure.(iistr).jequalOneDist_T=jequalOneDist_temp;
        else
            transpathoptions.(iistr).trivialjequalonedist=1;
            if N_z==0
                if N_e==0
                    jequalOneDist_temp=reshape(jequalOneDist_temp,[N_a,1]);
                else
                    jequalOneDist_temp=reshape(jequalOneDist_temp,[N_a*N_e,1]); % simoptions.fastOLG==1
                end
            else
                if N_e==0
                    jequalOneDist_temp=reshape(jequalOneDist_temp,[N_a*N_z,1]); % simoptions.fastOLG==1
                else
                    jequalOneDist_temp=reshape(jequalOneDist_temp,[N_a*N_z*N_e,1]); % simoptions.fastOLG==1
                end
            end
            PTypeStructure.(iistr).jequalOneDist=jequalOneDist_temp;
        end
    end

    %% Which parts of ParamPath and PricePath relate to ptype ii
    % Some ParamPath and PricePath parameters may depend on ptype
    PTypeStructure.(iistr).RelevantPricePath=ones(1,size(PricePath0,2)); % start will all relevant
    for pp=1:length(PricePathNames)
        if PricePathSizeVec(2,pp)-PricePathSizeVec(1,pp)+1==PTypeStructure.N_i
            % This depends on ii, so only keep the ii-th one
            PTypeStructure.(iistr).RelevantPricePath(PricePathSizeVec(1,pp):PricePathSizeVec(2,pp))=zeros(1,PTypeStructure.N_i);
            PTypeStructure.(iistr).RelevantPricePath(PricePathSizeVec(1,pp)+ii-1)=1;
        end
    end
    PTypeStructure.(iistr).RelevantPricePath=logical(PTypeStructure.(iistr).RelevantPricePath); % logical, so easier to use later
    PTypeStructure.(iistr).RelevantParamPath=ones(1,size(ParamPath,2)); % start will all relevant
    for pp=1:length(ParamPathNames)
        if ParamPathSizeVec(2,pp)-ParamPathSizeVec(1,pp)+1==PTypeStructure.N_i
            % This depends on ii, so only keep the ii-th one
            PTypeStructure.(iistr).RelevantParamPath(ParamPathSizeVec(1,pp):ParamPathSizeVec(2,pp))=zeros(1,PTypeStructure.N_i);
            PTypeStructure.(iistr).RelevantParamPath(ParamPathSizeVec(1,pp)+ii-1)=1;
        end
    end
    PTypeStructure.(iistr).RelevantParamPath=logical(PTypeStructure.(iistr).RelevantParamPath); % logical, so easier to use later


    if ii==1
        PTypeStructure.PricePath_Idependsonptype=zeros(1,length(PricePathNames));
        for pp=1:length(PricePathNames)
            if PricePathSizeVec(2,pp)-PricePathSizeVec(1,pp)+1==PTypeStructure.N_i
                PTypeStructure.PricePath_Idependsonptype(pp)=1; % This depends on ii
            end
        end
        PTypeStructure.ParamPath_Idependsonptype=zeros(1,length(ParamPathNames));
        for pp=1:length(ParamPathNames)
            if ParamPathSizeVec(2,pp)-ParamPathSizeVec(1,pp)+1==PTypeStructure.N_i
                PTypeStructure.ParamPath_Idependsonptype(pp)=1; % This depends on ii
            end
        end
    end
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



%%
if transpathoptions.stockvars==1 
    error('transpathoptions.stockvars=1 not yet implemented with PType \n')
end

if transpathoptions.verbose==1
    fprintf('Completed setup, beginning transition computation \n')
end


%% If using a shooting algorithm, set that up
transpathoptions=setupGEnewprice3_shooting(transpathoptions,GeneralEqmEqns,PricePathNames,N_i,PricePathSizeVec);

%% Check if using _tminus1 and/or _tplus1 variables.
if isstruct(FnsToEvaluate) && isstruct(GeneralEqmEqns)
    [tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tminus1paramNames,tplus1pricePathkk]=inputsFindtplus1tminus1(FnsToEvaluate,GeneralEqmEqns,PricePathNames,ParamPathNames,Names_i);
else
    tplus1priceNames=[];
    tminus1priceNames=[];
    tminus1paramNames=[];
    tminus1AggVarsNames=[];
    tplus1pricePathkk=[];  % I cannot remember what this was even for (how is it different rom tplus1priceNames??)
end

use_tplus1price=0;
if ~isempty(tplus1priceNames)
    use_tplus1price=1;
end
use_tminus1price=0;
if ~isempty(tminus1priceNames)
    use_tminus1price=1;
    for ii=1:length(tminus1priceNames)
        if ~isfield(transpathoptions.initialvalues,tminus1priceNames{ii})
            error('Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1priceNames{ii})
        end
    end
end
use_tminus1params=0;
if ~isempty(tminus1paramNames)
    use_tminus1params=1;
    for ii=1:length(tminus1paramNames)
        if ~isfield(transpathoptions.initialvalues,tminus1paramNames{ii})
            error('Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1paramNames{ii})
        end
    end
end
use_tminus1AggVars=0;
if ~isempty(tminus1AggVarsNames)
    use_tminus1AggVars=1;
    for ii=1:length(tminus1AggVarsNames)
        if ~isfield(transpathoptions.initialvalues,tminus1AggVarsNames{ii})
            error('Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1AggVarsNames{ii})
        end
    end
end
% Note: I used this approach (rather than just creating _tplus1 and _tminus1 for everything) as it will be same computation.

if transpathoptions.verbose>1
    use_tplus1price
    use_tminus1price
    use_tminus1params
    use_tminus1AggVars
    % tplus1pricePathkk
end


if transpathoptions.verbose>=1
    transpathoptions
end

%% Shooting algorithm
if transpathoptions.GEnewprice~=2
    % For permanent type, there is just one shooting command,
    % because things like z,e, and fastOLG are handled on a per-PType basis (to permit that they differ across ptype)
    [PricePath,GEcondnPath]=TransitionPath_Case1_MixHorz_PType_shooting(PricePath0, PricePathNames, ParamPath, ParamPathNames, T, V_final, AgentDist_initial, FnsToEvaluate, GeneralEqmEqns, PricePathSizeVec, ParamPathSizeVec, PricePathSizeVec_ii, ParamPathSizeVec_ii, GEeqnNames,nGeneralEqmEqns,nGeneralEqmEqns_acrossptypes,GeneralEqmEqnsCell,GeneralEqmEqnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, transpathoptions, PTypeStructure);

    % Switch the solution into structure for output.
    pp_indexinpricepath=zeros(1,length(PricePathNames));
    pp_c=0;
    for pp=1:length(PricePathNames)
        if PTypeStructure.PricePath_Idependsonptype(pp)==0
            pp_c=pp_c+1;
            pp_indexinpricepath(pp)=pp_c;
        else
            pp_c=pp_c+1;
            pp_indexinpricepath(pp)=pp_c;
            pp_c=pp_c+(PTypeStructure.N_i-1);
        end
    end

    for pp=1:length(PricePathNames)
        if PTypeStructure.PricePath_Idependsonptype(pp)==0
            PricePathStruct.(PricePathNames{pp})=PricePath(:,pp_indexinpricepath(pp));
        else
            if transpathoptions.PricePathptype_vectoroutput==1
                PricePathStruct.(PricePathNames{pp})=PricePath(:,pp_indexinpricepath(pp):pp_indexinpricepath(pp)+PTypeStructure.N_i-1);
            elseif transpathoptions.PricePathptype_vectoroutput==0
                for ii=1:N_i
                    PricePathStruct.(PricePathNames{pp}).(Names_i{ii})=PricePath(:,pp_indexinpricepath(pp)+ii-1);
                end
            end
        end
    end

    if nargout==1
        varargout={PricePathStruct};
    elseif nargout==2
        varargout={PricePathStruct,GEcondnPath};
    end

    return
end


%%
if transpathoptions.GEnewprice==2 % Function minimization
    % Have not attempted implementing this for PType yet, no point until I
    % get it to be useful without PType
    error('Have not yet implemented transpathoptions.GEnewprice=2')

    if nargout==1
        varargout={PricePathStruct};
    elseif nargout==2
        varargout={PricePathStruct,GEcondnPath};
    end

end


end