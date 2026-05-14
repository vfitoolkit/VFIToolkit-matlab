function varargout=TransitionPath_InfHorz_PType(PricePath0, ParamPath, T, V_final, AgentDist_initial, n_d,n_a,n_z, Names_i, d_grid,a_grid,z_grid, pi_z, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, PTypeDistParamNames, transpathoptions, simoptions, vfoptions)
% PricePathOld is a structure with fields names being the Prices and each field containing a T-by-1 path.
% ParamPath is a structure with fields names being the parameter names of those parameters which change over the path and each field containing a T-by-1 path.

% Remark to self: No real need for T as input, as this is anyway the length of PricePathOld

% PricePathOld, ParamPath, T, V_final, StationaryDist_init, GeneralEqmEqns, GeneralEqmEqnParamNames

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
    transpathoptions.oldpathweight=0.9; % default =0.9
    transpathoptions.weightscheme=1; % default =1
    transpathoptions.Ttheta=1;
    transpathoptions.maxiter=500; % Based on personal experience anything that hasn't converged well before this is just hung-up on trying to get the 4th decimal place (typically because the number of grid points was not large enough to allow this level of accuracy).
    transpathoptions.verbose=0;
    transpathoptions.graphpricepath=0; % 1: creates a graph of the 'current' price path which updates each iteration.
    transpathoptions.graphaggvarspath=0; % 1: creates a graph of the 'current' aggregate variables which updates each iteration.
    transpathoptions.graphGEcondns=0;  % 1: creates a graph of the 'current' general eqm conditions which updates each iteration.
    transpathoptions.historyofpricepath=0;
    transpathoptions.stockvars={}; % 'stockvars' are prices where you write '_tminus1' and it should cumulate (to there will be a general eqm eqn that relates the _tminus1 to the t for a price in PricePath)
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
    if ~isfield(transpathoptions,'stockvars')
        transpathoptions.stockvars={}; % 'stockvars' are prices where you write '_tminus1' and it should cumulate (to there will be a general eqm eqn that relates the _tminus1 to the t for a price in PricePath)
    end
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
if exist('vfoptions','var')==0
    vfoptions.exoticpreferences='None';
    vfoptions.divideandconquer=0;
    vfoptions.lowmemory=0;
else
    if ~isfield(vfoptions,'exoticpreferences')
        vfoptions.exoticpreferences='None';
    end
    if ~isfield(vfoptions,'divideandconquer')
        vfoptions.divideandconquer=0;
    end
    if ~isfield(vfoptions,'lowmemory')
        vfoptions.lowmemory=0;
    end
end
if vfoptions.divideandconquer==1
    if isscalar(n_a)
        vfoptions.level1n=11;
    end
end


%% Internally PricePath is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'.
[PricePath0,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec,PricePathSizeVec_ii,ParamPathSizeVec_ii]=PricePathParamPath_StructToMatrix(PricePath0,ParamPath,T,N_i);


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

    %% Sort out vfoptions and simoptions
    PTypeStructure.(iistr).vfoptions=PType_Options(vfoptions,Names_i,ii); % some vfoptions will differ by permanent type, will clean these up as we go before they are passed
    PTypeStructure.(iistr).simoptions=PType_Options(simoptions,Names_i,ii); % some vfoptions will differ by permanent type, will clean these up as we go before they are passed
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
            PTypeStructure.(iistr).vfoptions.level1n=ceil(n_a/50); % default
        end
    end
    if ~isfield(PTypeStructure.(iistr).vfoptions,'gridinterplayer')
        PTypeStructure.(iistr).vfoptions.gridinterplayer=0; %default
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

    %%
    % Go through everything which might be dependent on permanent type (PType)
    % Notice that the way this is coded the grids (etc.) could be either
    % fixed, or a function (that depends on age, and possibly on permanent
    % type), or they could be a structure. Only in the case where they are
    % a structure is there a need to take just a specific part and send
    % only that to the 'non-PType' version of the command.

    if isa(n_d,'struct')
        PTypeStructure.(iistr).n_d=n_d.(Names_i{ii});
    else
        PTypeStructure.(iistr).n_d=n_d;
    end
    N_d=prod(PTypeStructure.(iistr).n_d);
    PTypeStructure.(iistr).N_d=N_d;
    if N_d==0
        PTypeStructure.(iistr).l_d=0;
    else
        PTypeStructure.(iistr).l_d=length(n_d);
    end
    if isa(n_a,'struct')
        PTypeStructure.(iistr).n_a=n_a.(Names_i{ii});
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
        PTypeStructure.(iistr).n_z=n_z.(Names_i{ii});
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
        PTypeStructure.(iistr).l_e=length(n_e);
    end

    if isa(d_grid,'struct')
        PTypeStructure.(iistr).d_grid=gpuArray(d_grid.(Names_i{ii}));
    else
        PTypeStructure.(iistr).d_grid=gpuArray(d_grid);
    end
    if isa(a_grid,'struct')
        PTypeStructure.(iistr).a_grid=gpuArray(a_grid.(Names_i{ii}));
    else
        PTypeStructure.(iistr).a_grid=gpuArray(a_grid);
    end
    if isa(z_grid,'struct')
        PTypeStructure.(iistr).z_grid=gpuArray(z_grid.(Names_i{ii}));
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
        PTypeStructure.(iistr).pi_z=pi_z.(Names_i{ii}); % Different grids by permanent type, but not depending on age. (same as the case just above; this case can occur with or without the existence of vfoptions, as long as there is no vfoptions.agedependentgrids)
    else
        PTypeStructure.(iistr).pi_z=pi_z;
    end

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
                PTypeStructure.(iistr).Parameters.(FullParamNames{kField})=temp(:,ii);
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
    l_a=length(PTypeStructure.(iistr).n_a);
    l_z=length(PTypeStructure.(iistr).n_z);
    if PTypeStructure.(iistr).N_z==0
        l_z=0;
    end
    if isfield(PTypeStructure.(iistr).vfoptions,'SemiExoStateFn')
        l_z=l_z+length(PTypeStructure.(iistr).vfoptions.n_semiz);
    end
    l_e=0;
    if isfield(PTypeStructure.(iistr).vfoptions,'n_e')
        if PTypeStructure.(iistr).vfoptions.n_e(1)~=0
            l_e=length(PTypeStructure.(iistr).vfoptions.n_e);
        end
    end
    % Figure out ReturnFnParamNames from ReturnFn
    temp=getAnonymousFnInputNames(PTypeStructure.(iistr).ReturnFn);
    if length(temp)>(l_d+l_a+l_a+l_z+l_e) % This is largely pointless, the ReturnFn is always going to have some parameters
        ReturnFnParamNames={temp{l_d+l_a+l_a+l_z+l_e+1:end}}; % the first inputs will always be (d,aprime,a,z)
    else
        ReturnFnParamNames={};
    end
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

    %% Set up exogenous shock processes
    [PTypeStructure.(iistr).z_gridvals, PTypeStructure.(iistr).pi_z, PTypeStructure.(iistr).pi_z_sparse, PTypeStructure.(iistr).e_gridvals, PTypeStructure.(iistr).pi_e, PTypeStructure.(iistr).pi_e_sparse, PTypeStructure.(iistr).ze_gridvals, transpathoptions, PTypeStructure.(iistr).vfoptions]=ExogShockSetup_TPath_InfHorz(PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).z_grid,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).N_a,PTypeStructure.(iistr).Parameters,PricePathNames,ParamPathNames,transpathoptions,PTypeStructure.(iistr).vfoptions,4);
    % Convert z and e to joint-grids and transtion matrix
    % output: z_gridvals, pi_z, e_gridvals, pi_e, transpathoptions,vfoptions,simoptions


    %% Organise V_final and AgentDist_initial
    % Reshape V_final
    if N_z==0
        if N_e==0
            V_final.(iistr)=reshape(V_final.(iistr),[N_a,1]);
        else
            V_final.(iistr)=reshape(V_final.(iistr),[N_a,N_e]);
        end
    else
        if N_e==0
            V_final.(iistr)=reshape(V_final.(iistr),[N_a,N_z]);
        else
            V_final.(iistr)=reshape(V_final.(iistr),[N_a,N_z,N_e]);
        end
    end
    % Reshape AgentDist_initial
    if N_z==0
        if N_e==0
            AgentDist_initial.(iistr)=reshape(AgentDist_initial.(iistr),[N_a,1]);
        else
            AgentDist_initial.(iistr)=reshape(AgentDist_initial.(iistr),[N_a*N_e,1]);
        end
    else
        if N_e==0
            AgentDist_initial.(iistr)=reshape(AgentDist_initial.(iistr),[N_a*N_z,1]);
        else
            AgentDist_initial.(iistr)=reshape(AgentDist_initial.(iistr),[N_a*N_z*N_e,1]);
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


%%
if transpathoptions.parallel==2
   PricePath0=gpuArray(PricePath0);
end

% GeneralEqmEqnNames=fieldnames(GeneralEqmEqns);
% for gg=1:length(GeneralEqmEqnNames)
%     GeneralEqmEqnParamNames{gg}=getAnonymousFnInputNames(GeneralEqmEqns.(GeneralEqmEqnNames{gg}));
% end

if any(strcmp(PTypeDistParamNames,PricePathNames)) || any(strcmp(PTypeDistParamNames,ParamPathNames))
    error('Cannot yet handle PTypeDistParamNames changing values over path, ask on forum if you need this')
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


%% If using a shooting algorithm, set that up
transpathoptions=setupGEnewprice3_shooting(transpathoptions,GeneralEqmEqns,PricePathNames,N_i,PricePathSizeVec);


if transpathoptions.verbose==1
    fprintf('Completed setup, beginning transition computation \n')
end


%% Check if using _tminus1 and/or _tplus1 variables.
[tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tminus1paramNames,tplus1pricePathkk,use_tplus1price,use_tminus1price,use_tminus1params,use_tminus1AggVars]=inputsFindtplus1tminus1(FnsToEvaluate,GeneralEqmEqns,PricePathNames,ParamPathNames,PTypeStructure.Names_i,transpathoptions);

% Following lines remove transpathoptions.stockvars from tminus1priceNames, and update use_tminus1price if necessary
if ~isempty(transpathoptions.stockvars)
    use_stockvars=1;
    stockvarsNames=transpathoptions.stockvars;
    transpathoptions=rmfield(transpathoptions,'stockvars');


    % how to find stockvars in PricePathNames
    stockvarsInPricePathNames=zeros(length(stockvarsNames),1); %% the pp index in PricePathNames that corresponds to each stockvar
    for kk=1:length(stockvarsNames)
        % throw error if the stockvar is not in PriceParamNames
        if ~any(strcmp(stockvarsNames{kk},PricePathNames))
            fprintf('Following error relates to stockvar: %s \n', stockvarsNames{kk})
            error('Cannot use a transpathoptions.stockvar which is not in PricePath')
        end
        % otherwise, find the matching index
        for pp=1:length(PricePathNames)
            if strcmp(stockvarsNames{kk},PricePathNames{pp})
                stockvarsInPricePathNames(kk)=pp;
            end
        end
    end

    % remove from stockvars from tminus1priceNames [stockvars have _tminus1 in name, but they 'cumulate' so have to be treated separately]
    for pp=1:length(stockvarsNames)
        if ~any(strcmp(stockvarsNames{pp},tminus1priceNames))
            error('transpathoptions.stockvars must appear as prices that are used with _tminus1')
        else
            tminus1priceNames(strcmp(tminus1priceNames, stockvarsNames{pp})) = [];
        end
    end
    if isempty(tminus1priceNames)
        use_tminus1params=0;
    end
else
    use_stockvars=0;
    stockvarsNames=[];
    stockvarsInPricePathNames=[];
end

if use_stockvars==1
    error('Cannot yet use ptype for stockvars')
    % stockvars does not yet work with ptype because in the _shooting command the tt loop over V, Dist, AggVars is done separately from the tt loop over GE condns. Need to rework so that they are all in one tt loop.
end


if transpathoptions.verbose>=1
    transpathoptions
end



%% Shooting algorithm
if transpathoptions.GEnewprice~=2
    % For permanent type, there is just one shooting command,
    % because things like z,e, and fastOLG are handled on a per-PType basis (to permit that they differ across ptype)
    [PricePath,GEcondnPathmatrix]=TransitionPath_InfHorz_PType_shooting(PricePath0, PricePathNames, ParamPath, ParamPathNames, T, V_final, AgentDist_initial, PTypeDistParamNames, FnsToEvaluate, GeneralEqmEqns, PricePathSizeVec, ParamPathSizeVec, PricePathSizeVec_ii, ParamPathSizeVec_ii, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, GEeqnNames, nGeneralEqmEqns, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, stockvarsNames, stockvarsInPricePathNames, transpathoptions, PTypeStructure);

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
            PricePathStruct.(PricePathNames{pp})=PricePath(:,pp_indexinpricepath(pp))';
        else
            if transpathoptions.PricePathptype_vectoroutput==1
                PricePathStruct.(PricePathNames{pp})=PricePath(:,pp_indexinpricepath(pp):pp_indexinpricepath(pp)+PTypeStructure.N_i-1)';
            elseif transpathoptions.PricePathptype_vectoroutput==0
                for ii=1:N_i
                    PricePathStruct.(PricePathNames{pp}).(Names_i{ii})=PricePath(:,pp_indexinpricepath(pp)+ii-1)';
                end
            end
        end
    end
    for gg=1:length(GEeqnNames)
        GEcondnPath.(GEeqnNames{gg})=GEcondnPathmatrix(:,gg)';
    end

    if nargout==1
        varargout={PricePathStruct};
    elseif nargout==2
        varargout={PricePathStruct,GEcondnPath};
    end

    return
end



%% As optimization problem
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
