function PricePath=TransitionPath_Case1_FHorz_PType(PricePathOld, ParamPath, T, V_final, AgentDist_init, jequalOneDist, n_d, n_a, n_z, N_j, Names_i, d_grid,a_grid,z_grid, pi_z, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, AgeWeightsParamNames, PTypeDistParamNames, transpathoptions, simoptions, vfoptions)
% This code will work for all transition paths except those that involve at
% change in the transition matrix pi_z (can handle a change in pi_z, but
% only if it is a 'surprise', not anticipated changes) 
%
% PricePathOld is a structure with fields names being the Prices and each field containing a T-by-1 path.
% ParamPath is a structure with fields names being the parameter names of those parameters which change over the path and each field containing a T-by-1 path.

% Remark to self: No real need for T as input, as this is anyway the length of PricePathOld

%% PType hardcodes some things that are options when not using PTypes
% Namely, 
% hardcodes simoptions.fastOLG=1
% hardcodes transpathoptions.ageweightstrivial=0 (don't overwrite 
% In both cases, mainly done so I don't have to handle this differing by PType

%%
% HARDCODE N_e=0. NEED TO FIX THIS LATER!
using_e_var=0


%% Check which transpathoptions have been used, set all others to defaults 
if exist('transpathoptions','var')==0
    disp('No transpathoptions given, using defaults')
    %If transpathoptions is not given, just use all the defaults
    transpathoptions.tolerance=10^(-4);
    transpathoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    transpathoptions.GEnewprice=1; % 1 is shooting algorithm, 0 is that the GE should evaluate to zero and the 'new' is the old plus the "non-zero" (for each time period seperately); 
                                   % 2 is to do optimization routine with 'distance between old and new path', 3 is just same as 0, but easier to set up
    transpathoptions.GEptype=zeros(1,length(fieldnames(GeneralEqmEqns))); % 1 indicates that this general eqm condition is 'conditional on permanent type'
    transpathoptions.oldpathweight=0.9; % default =0.9
    transpathoptions.weightscheme=1; % default =1
    transpathoptions.Ttheta=1;
    transpathoptions.maxiter=500; % Based on personal experience anything that hasn't converged well before this is just hung-up on trying to get the 4th decimal place (typically because the number of grid points was not large enough to allow this level of accuracy).
    transpathoptions.verbose=0;
    transpathoptions.graphpricepath=0;
    transpathoptions.graphaggvarspath=0;
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
        transpathoptions.GEptype=zeros(1,length(fieldnames(GeneralEqmEqns))); % 1 indicates that this general eqm condition is 'conditional on permanent type'
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
        transpathoptions.graphpricepath=0;
    end
    if ~isfield(transpathoptions,'graphaggvarspath')
        transpathoptions.graphaggvarspath=0;
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

%% Some internal commands require a few vfoptions and simoptions to be set
if exist('vfoptions','var')==0
    vfoptions.policy_forceintegertype=0;
else
    if isfield(vfoptions,'policy_forceintegertype')==0
        vfoptions.policy_forceintegertype=0;
    end
end

%% Get AgeWeights from Parameters   
try
    AgeWeights=Parameters.(AgeWeightsParamNames{1});
catch
    error(['Failed to find parameter ', AgeWeightsParamNames{1}])
end
% Later, when creating PTypeStructure, we get the ptype-specific versions out of this and create an AgeWeights_T structure.

%% ptype just hardcodes that this is non-trivial, so I will use
jequalOneDist_T=struct();

%% Note: Internally PricePathOld is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'. 
% Actually, some of those prices are 1-by-N_j, so is more subtle than this.
PricePathNames=fieldnames(PricePathOld);
PricePathStruct=PricePathOld; 
PricePathSizeVec=zeros(1,length(PricePathNames)); % Allows for a given price param to depend on age (or permanent type)
for ii=1:length(PricePathNames)
    temp=PricePathStruct.(PricePathNames{ii});
    tempsize=size(temp);
    PricePathSizeVec(ii)=tempsize(tempsize~=T); % Get the dimension which is not T
end
PricePathSizeVec=cumsum(PricePathSizeVec);
if length(PricePathNames)>1
    PricePathSizeVec=[[1,PricePathSizeVec(1:end-1)+1];PricePathSizeVec];
else
    PricePathSizeVec=[1;PricePathSizeVec];
end
PricePathOld=zeros(T,PricePathSizeVec(2,end));% Do this seperately afterwards so that can preallocate the memory
for ii=1:length(PricePathNames)
    if size(PricePathStruct.(PricePathNames{ii}),1)==T
        PricePathOld(:,PricePathSizeVec(1,ii):PricePathSizeVec(2,ii))=PricePathStruct.(PricePathNames{ii});
    else % Need to transpose
        PricePathOld(:,PricePathSizeVec(1,ii):PricePathSizeVec(2,ii))=PricePathStruct.(PricePathNames{ii})';
    end
end

ParamPathNames=fieldnames(ParamPath);
ParamPathStruct=ParamPath;
ParamPathSizeVec=zeros(1,length(ParamPathNames)); % Allows for a given price param to depend on age (or permanent type)
for ii=1:length(ParamPathNames)
    temp=ParamPathStruct.(ParamPathNames{ii});
    tempsize=size(temp);
    ParamPathSizeVec(ii)=tempsize(tempsize~=T); % Get the dimension which is not T
end
ParamPathSizeVec=cumsum(ParamPathSizeVec);
if length(ParamPathNames)>1
    ParamPathSizeVec=[[1,ParamPathSizeVec(1:end-1)+1];ParamPathSizeVec];
else
    ParamPathSizeVec=[1;ParamPathSizeVec];
end
ParamPath=zeros(T,ParamPathSizeVec(2,end));% Do this seperately afterwards so that can preallocate the memory
for ii=1:length(ParamPathNames)
    if size(ParamPathStruct.(ParamPathNames{ii}),1)==T
        ParamPath(:,ParamPathSizeVec(1,ii):ParamPathSizeVec(2,ii))=ParamPathStruct.(ParamPathNames{ii});
    else % Need to transpose
        ParamPath(:,ParamPathSizeVec(1,ii):ParamPathSizeVec(2,ii))=ParamPathStruct.(ParamPathNames{ii})';
    end
end

PricePath=struct();

if transpathoptions.verbose>1
    PricePathNames
    ParamPathNames
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
    
    % Need to fill in some defaults
    PTypeStructure.(iistr).vfoptions.parallel=2; % hardcode
    PTypeStructure.(iistr).simoptions.parallel=2; % hardcode
    PTypeStructure.(iistr).simoptions.iterate=1; % hardcode
    PTypeStructure.(iistr).simoptions.fastOLG=1; % hardcode 
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
    if ~isfield(PTypeStructure.(iistr).vfoptions,'exoticpreferences')
        PTypeStructure.(iistr).vfoptions.exoticpreferences='None'; % not yet implemented, so hardcodes None
    else
        if ~strcmp(PTypeStructure.(iistr).vfoptions.exoticpreferences,'None')
            error('transition paths cannot yet handle exoticpreferences')
        end
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
        PTypeStructure.(iistr).N_j=N_j.(Names_i{ii});
    elseif isscalar(N_j)
        PTypeStructure.(iistr).N_j=N_j;
    else
        PTypeStructure.(iistr).N_j=N_j(ii);
    end
    
    PTypeStructure.(iistr).n_d=n_d;
    if isa(n_d,'struct')
        PTypeStructure.(iistr).n_d=n_d.(Names_i{ii});
    end
    N_d=prod(PTypeStructure.(iistr).n_d);
    PTypeStructure.(iistr).N_d=N_d;
    PTypeStructure.(iistr).n_a=n_a;
    if isa(n_a,'struct')
        PTypeStructure.(iistr).n_a=n_a.(Names_i{ii});
    end
    N_a=prod(PTypeStructure.(iistr).n_a);
    PTypeStructure.(iistr).N_a=N_a;
    PTypeStructure.(iistr).n_z=n_z;
    if isa(n_z,'struct')
        PTypeStructure.(iistr).n_z=n_z.(Names_i{ii});
    end
    N_z=prod(PTypeStructure.(iistr).n_z);
    PTypeStructure.(iistr).N_z=N_z;
    N_e=prod(PTypeStructure.(iistr).n_e);
    PTypeStructure.(iistr).N_e=N_e;

    PTypeStructure.(iistr).d_grid=d_grid;
    if isa(d_grid,'struct')
        PTypeStructure.(iistr).d_grid=d_grid.(Names_i{ii});
    end
    PTypeStructure.(iistr).a_grid=a_grid;
    if isa(a_grid,'struct')
        PTypeStructure.(iistr).a_grid=a_grid.(Names_i{ii});
    end
    PTypeStructure.(iistr).z_grid=z_grid;
    if isa(z_grid,'struct')
        PTypeStructure.(iistr).z_grid=z_grid.(Names_i{ii});
    end
    % to be able to EvalFnsOnAgentDist using fastOLG we also need
    PTypeStructure.(iistr).a_gridvals=CreateGridvals(PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).a_grid,1); % a_grivdals is [N_a,l_a]
    PTypeStructure.(iistr).daprime_gridvals=gpuArray([kron(ones(N_a,1),CreateGridvals(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).d_grid,1)), kron(PTypeStructure.(iistr).a_gridvals,ones(PTypeStructure.(iistr).N_d,1))]); % daprime_gridvals is [N_d*N_aprime,l_d+l_aprime]
    
    PTypeStructure.(iistr).pi_z=pi_z;
    % If using 'agedependentgrids' then pi_z will actually be the AgeDependentGridParamNames, which is a structure. 
    % Following gets complicated as pi_z being a structure could be because
    % it depends just on age, or on permanent type, or on both.
    if exist('vfoptions','var')
        if isfield(vfoptions,'agedependentgrids')
            if isa(vfoptions.agedependentgrids, 'struct')
                if isfield(vfoptions.agedependentgrids, Names_i{ii})
                    PTypeStructure.(iistr).vfoptions.agedependentgrids=vfoptions.agedependentgrids.(Names_i{ii});
                    PTypeStructure.(iistr).simoptions.agedependentgrids=simoptions.agedependentgrids.(Names_i{ii});
                    % In this case AgeDependentGridParamNames must be set up as, e.g., AgeDependentGridParamNames.ptype1.d_grid
                    PTypeStructure.(iistr).pi_z=pi_z.(Names_i{ii});
                else
                    % The current permanent type does not use age dependent grids.
                    PTypeStructure.(iistr).vfoptions=rmfield(PTypeStructure.(iistr).vfoptions,'agedependentgrids');
                    PTypeStructure.(iistr).simoptions=rmfield(PTypeStructure.(iistr).simoptions,'agedependentgrids');
                    % Different grids by permanent type (some of them must be using agedependentgrids even though not the current permanent type), but not depending on age.
                    PTypeStructure.(iistr).pi_z=pi_z.(Names_i{ii});
                end
            else
                temp=size(vfoptions.agedependentgrids);
                if temp(1)>1 % So different permanent types use different settings for age dependent grids
                    if prod(temp(ii,:))>0
                        PTypeStructure.(iistr).vfoptions.agedependentgrids=vfoptions.agedependentgrids(ii,:);
                        PTypeStructure.(iistr).simoptions.agedependentgrids=simoptions.agedependentgrids(ii,:);
                    else
                        PTypeStructure.(iistr).vfoptions=rmfield(PTypeStructure.(iistr).vfoptions,'agedependentgrids');
                        PTypeStructure.(iistr).simoptions=rmfield(PTypeStructure.(iistr).simoptions,'agedependentgrids');
                    end
                    % In this case AgeDependentGridParamNames must be set up as, e.g., AgeDependentGridParamNames.ptype1.d_grid
                    PTypeStructure.(iistr).pi_z=pi_z.(Names_i{ii});
                else % Grids depend on age, but not on permanent type (at least the function does not, you could set it up so that this is handled by the same function but a parameter whose value differs by permanent type
                    PTypeStructure.(iistr).pi_z=pi_z;
                end
            end
        elseif isa(pi_z,'struct')
            PTypeStructure.(iistr).pi_z=pi_z.(Names_i{ii}); % Different grids by permanent type, but not depending on age.
        end
    elseif isa(pi_z,'struct')
        PTypeStructure.(iistr).pi_z=pi_z.(Names_i{ii}); % Different grids by permanent type, but not depending on age. (same as the case just above; this case can occour with or without the existence of vfoptions, as long as there is no vfoptions.agedependentgrids)
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
    if PTypeStructure.(iistr).n_z(1)==0
        l_z=0;
    end
    if isfield(PTypeStructure.(iistr).vfoptions,'SemiExoStateFn')
        l_z=l_z+length(PTypeStructure.(iistr).vfoptions.n_semiz);
    end
    l_e=0;
    if isfield(PTypeStructure.(iistr).vfoptions,'n_e')
        if PTypeStructure.(iistr).vfoptions.n_e(1)~=0
            l_e=length(PTypeStructure.(iistr).vfoptions.n_e);
            using_e_var=1;
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
    

    %% Figure out which functions are actually relevant to the present PType. 
    % Only the relevant ones need to be evaluated.
    % The dependence of FnsToEvaluateFn and FnsToEvaluateFnParamNames are necessarily the same.
    PTypeStructure.(iistr).FnsToEvaluate={};

    FnNames=fieldnames(FnsToEvaluate);
    PTypeStructure.numFnsToEvaluate=length(fieldnames(FnsToEvaluate));
    PTypeStructure.(iistr).WhichFnsForCurrentPType=zeros(PTypeStructure.numFnsToEvaluate,1);
    jj=1; % jj indexes the FnsToEvaluate that are relevant to the current PType
    for kk=1:PTypeStructure.numFnsToEvaluate
        if isa(FnsToEvaluate.(FnNames{kk}),'struct')
            if isfield(FnsToEvaluate.(FnNames{kk}), Names_i{ii})
                PTypeStructure.(iistr).FnsToEvaluate{jj}=FnsToEvaluate.(FnNames{kk}).(Names_i{ii});
                % Figure out FnsToEvaluateParamNames
                temp=getAnonymousFnInputNames(FnsToEvaluate.(FnNames{kk}).(Names_i{ii}));
                PTypeStructure.(iistr).FnsToEvaluateParamNames(jj).Names={temp{l_d+l_a+l_a+l_z+l_e+1:end}}; % the first inputs will always be (d,aprime,a,z)
                PTypeStructure.(iistr).WhichFnsForCurrentPType(kk)=jj; jj=jj+1;
                PTypeStructure.FnsAndPTypeIndicator(kk,ii)=1;
                % else
                %  % do nothing as this FnToEvaluate is not relevant for the current PType
                % % Implicitly, PTypeStructure.(iistr).WhichFnsForCurrentPType(kk)=0
            end
        else
            % If the Fn is not a structure (if it is a function) it is assumed to be relevant to all PTypes.
            PTypeStructure.(iistr).FnsToEvaluate{jj}=FnsToEvaluate.(FnNames{kk});
            % Figure out FnsToEvaluateParamNames
            temp=getAnonymousFnInputNames(FnsToEvaluate.(FnNames{kk}));
            PTypeStructure.(iistr).FnsToEvaluateParamNames(jj).Names={temp{l_d+l_a+l_a+l_z+l_e+1:end}}; % the first inputs will always be (d,aprime,a,z)
            PTypeStructure.(iistr).WhichFnsForCurrentPType(kk)=jj; jj=jj+1;
            PTypeStructure.FnsAndPTypeIndicator(kk,ii)=1;
        end
    end
    PTypeStructure.(iistr).AggVarNames=PTypeStructure.(iistr).FnsToEvaluate;


    %% Check if pi_z and z_grid can be precomputed
    % Note: cannot handle that whether not not they can be precomputed differs across ptypes
    transpathoptions.zpathprecomputed=0;
    if isfield(PTypeStructure.(iistr).vfoptions,'pi_z_J')
        transpathoptions.zpathprecomputed=1;
        transpathoptions.zpathtrivial=1; % z_grid_J and pi_z_J are not varying over the path
    elseif isfield(PTypeStructure.(iistr).vfoptions,'ExogShockFn')
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
                pi_z_J=zeros(N_z,N_z,N_j,'gpuArray');
                z_grid_J=zeros(sum(n_z),N_j,'gpuArray');
                for jj=1:N_j
                    ExogShockFnParamsVec=CreateVectorFromParams(PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).vfoptions.ExogShockFnParamNames,jj);
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for pp=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(pp,1)={ExogShockFnParamsVec(pp)};
                    end
                    [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
                    pi_z_J(:,:,jj)=gpuArray(pi_z);
                    z_grid_J(:,jj)=gpuArray(z_grid);
                end
                % Now store them in vfoptions and simoptions
                PTypeStructure.(iistr).vfoptions.pi_z_J=pi_z_J;
                PTypeStructure.(iistr).vfoptions.z_grid_J=z_grid_J;
                PTypeStructure.(iistr).simoptions.pi_z_J=pi_z_J;
                PTypeStructure.(iistr).simoptions.z_grid_J=z_grid_J;
            elseif transpathoptions.zpathtrivial==0
                % z_grid_J and/or pi_z_J varies along the transition path (but only depending on ParamPath, not PricePath
                transpathoptions.(iistr).pi_z_J_T=zeros(N_z,N_z,N_j,T,'gpuArray');
                transpathoptions.(iistr).z_grid_J_T=zeros(sum(n_z),N_j,T,'gpuArray');
                pi_z_J=zeros(N_z,N_z,N_j,'gpuArray');
                z_grid_J=zeros(sum(n_z),N_j,'gpuArray');
                for tt=1:T
                    for pp=1:length(ParamPathNames)
                        PTypeStructure.(iistr).Parameters.(ParamPathNames{pp})=ParamPathStruct.(ParamPathNames{pp});
                    end
                    % Note, we know the PricePath is irrelevant for the current purpose
                    for jj=1:N_j
                        ExogShockFnParamsVec=CreateVectorFromParams(PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).vfoptions.ExogShockFnParamNames,jj);
                        ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                        for pp=1:length(ExogShockFnParamsVec)
                            ExogShockFnParamsCell(pp,1)={ExogShockFnParamsVec(pp)};
                        end
                        [z_grid,pi_z]=PTypeStructure.(iistr).vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
                        pi_z_J(:,:,jj)=gpuArray(pi_z);
                        z_grid_J(:,jj)=gpuArray(z_grid);
                    end
                    transpathoptions.(iistr).pi_z_J_T(:,:,:,tt)=pi_z_J;
                    transpathoptions.(iistr).z_grid_J_T(:,:,tt)=z_grid_J;
                end
            end
        end
    end
    %% If used, check if pi_e and e_grid can be procomputed
    % Note: cannot handle that whether not not they can be precomputed differs across ptypes
    if using_e_var==1
        % Check if e_grid and/or pi_e depend on prices. If not then create pi_e_J and e_grid_J for the entire transition before we start

        transpathoptions.epathprecomputed=0;
        if isfield(PTypeStructure.(iistr).vfoptions,'pi_e_J')
            transpathoptions.epathprecomputed=1;
            transpathoptions.epathtrivial=1; % e_grid_J and pi_e_J are not varying over the path
        elseif isfield(PTypeStructure.(iistr).vfoptions,'EiidShockFn')
            N_e=prod(PTypeStructure.(iistr).vfoptions.n_e);
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
                    pi_e_J=zeros(N_e,N_j,'gpuArray');
                    e_grid_J=zeros(sum(PTypeStructure.(iistr).vfoptions.n_e),N_j,'gpuArray');
                    for jj=1:N_j
                        EiidShockFnParamsVec=CreateVectorFromParams(PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).vfoptions.EiidShockFnParamNames,jj);
                        EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                        for pp=1:length(EiidShockFnParamsVec)
                            EiidShockFnParamsCell(pp,1)={EiidShockFnParamsVec(pp)};
                        end
                        [e_grid,pi_e]=PTypeStructure.(iistr).vfoptions.EiidShockFn(EiidShockFnParamsCell{:});
                        pi_e_J(:,jj)=gpuArray(pi_e);
                        e_grid_J(:,jj)=gpuArray(e_grid);
                    end
                    % Now store them in vfoptions and simoptions
                    PTypeStructure.(iistr).vfoptions.pi_e_J=pi_e_J;
                    PTypeStructure.(iistr).vfoptions.e_grid_J=e_grid_J;
                    PTypeStructure.(iistr).simoptions.pi_e_J=pi_e_J;
                    PTypeStructure.(iistr).simoptions.e_grid_J=e_grid_J;
                elseif transpathoptions.epathtrivial==0
                    % e_grid_J and/or pi_e_J varies along the transition path (but only depending on ParamPath, not PricePath)
                    transpathoptions.(iistr).pi_e_J_T=zeros(N_e,N_j,T,'gpuArray');
                    transpathoptions.(iistr).e_grid_J_T=zeros(sum(PTypeStructure.(iistr).vfoptions.n_e),N_j,T,'gpuArray');
                    pi_e_J=zeros(N_e,N_j,'gpuArray');
                    e_grid_J=zeros(sum(PTypeStructure.(iistr).vfoptions.n_e),N_j,'gpuArray');
                    for tt=1:T
                        for pp=1:length(ParamPathNames)
                            PTypeStructure.(iistr).Parameters.(ParamPathNames{pp})=ParamPathStruct.(ParamPathNames{pp});
                        end
                        % Note, we know the PricePath is irrelevant for the current purpose
                        for jj=1:N_j
                            EiidShockFnParamsVec=CreateVectorFromParams(PTypeStructure.(iistr).Parameters, vPTypeStructure.(iistr).foptions.EiidShockFnParamNames,jj);
                            EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                            for pp=1:length(ExogShockFnParamsVec)
                                EiidShockFnParamsCell(pp,1)={EiidShockFnParamsVec(pp)};
                            end
                            [e_grid,pi_e]=PTypeStructure.(iistr).vfoptions.ExogShockFn(EiidShockFnParamsCell{:});
                            pi_e_J(:,jj)=gpuArray(pi_e);
                            e_grid_J(:,jj)=gpuArray(e_grid);
                        end
                        transpathoptions.(iistr).pi_e_J_T(:,:,tt)=pi_e_J;
                        transpathoptions.(iistr).e_grid_J_T(:,:,tt)=e_grid_J;
                    end
                end
            end
        end
    end

    %% Organise V_final and AgentDist_initial
    % Reshape V_final
    if transpathoptions.fastOLG==0
        if N_z==0
            if N_e==0
                V_final.(iistr)=reshape(V_final.(iistr),[N_a,N_j]);
            else
                V_final.(iistr)=reshape(V_final.(iistr),[N_a,N_e,N_j]);
            end
        else
            if N_e==0
                V_final.(iistr)=reshape(V_final.(iistr),[N_a,N_z,N_j]);
            else
                V_final.(iistr)=reshape(V_final.(iistr),[N_a,N_z,N_e,N_j]);
            end
        end
    else
        if N_z==0
            if N_e==0
                V_final.(iistr)=reshape(V_final.(iistr),[N_a,N_j]);
            else
                V_final.(iistr)=reshape(permute(V_final.(iistr),[1,3,2]),[N_a*N_j,N_e]);
            end
        else
            if N_e==0
                V_final.(iistr)=reshape(permute(V_final.(iistr),[1,3,2]),[N_a*N_j,N_z]);
            else
                V_final.(iistr)=reshape(permute(V_final.(iistr),[1,4,2,3]),[N_a*N_j,N_z,N_e]);
            end
        end
    end
    % Reshape AgentDist_initial
    if N_z==0
        if N_e==0
            AgentDist_init.(iistr)=reshape(AgentDist_init.(iistr),[N_a,N_j]); % if simoptions.fastOLG==0
            AgeWeights_initial.(iistr)=sum(AgentDist_init.(iistr),1); % [1,N_j]
            if PTypeStructure.(iistr).simoptions.fastOLG==1
                AgentDist_init.(iistr)=reshape(AgentDist_init.(iistr),[N_a*N_j,1]);
                AgeWeights_initial.(iistr)=repelem(AgeWeights_initial.(iistr)',N_a,1);
            end
        else
            AgentDist_init.(iistr)=reshape(AgentDist_init.(iistr),[N_a*N_e,N_j]); % if simoptions.fastOLG==0
            AgeWeights_initial.(iistr)=sum(AgeWeights_initial.(iistr),1); % [1,N_j]
            if PTypeStructure.(iistr).simoptions.fastOLG==1 % simoptions.fastOLG==1, so AgentDist is treated as : (a,j,z)-by-1
                AgentDist_initial.(iistr)=reshape(permute(reshape(AgentDist_initial.(iistr),[N_a,N_e,N_j]),[1,3,2]),[N_a*N_j*N_e,1]);
                AgeWeights_initial.(iistr)=repelem(AgeWeights_initial.(iistr)',N_a,1);
            end
        end
    else
        if N_e==0
            AgentDist_init.(iistr)=reshape(AgentDist_init.(iistr),[N_a*N_z,N_j]); % if simoptions.fastOLG==0
            AgeWeights_initial.(iistr)=sum(AgeWeights_initial.(iistr),1); % [1,N_j]
            if PTypeStructure.(iistr).simoptions.fastOLG==1 % simoptions.fastOLG==1, so AgentDist is treated as : (a,j,z)-by-1
                AgentDist_initial.(iistr)=reshape(permute(reshape(AgentDist_initial.(iistr),[N_a,N_z,N_j]),[1,3,2]),[N_a*N_j*N_z,1]);
                AgeWeights_initial.(iistr)=repelem(AgeWeights_initial.(iistr)',N_a,1);
            end
        else
            AgentDist_init.(iistr)=reshape(AgentDist_init.(iistr),[N_a*N_z*N_e,N_j]); % if simoptions.fastOLG==0
            AgeWeights_initial.(iistr)=sum(AgeWeights_initial.(iistr),1); % [1,N_j]
            if PTypeStructure.(iistr).simoptions.fastOLG==1 % simoptions.fastOLG==1, so AgentDist is treated as : (a,j,z)-by-1
                AgentDist_initial.(iistr)=reshape(permute(reshape(AgentDist_initial.(iistr),[N_a,N_z,N_e,N_j]),[1,4,2,3]),[N_a*N_j*N_z,N_e]);
                AgeWeights_initial.(iistr)=repelem(AgeWeights_initial.(iistr)',N_a,1);
            end
        end
    end
    % Get AgeWeights and switch into the transpathoptions.ageweightstrivial=0 setup (and this is what subfns hardcode when doing PTypes)
    % It is assumed there is only one Age Weight Parameter (name))
    % AgeWeights_T is (a,j)-by-T (create as j-by-T to start, then switch)
    if isstruct(AgeWeights)
        AgeWeights_ii=AgeWeights.(iistr);
        if all(size(AgeWeights_ii)==[N_j,1])
            % Does not depend on transition path period
            AgeWeights_T.(iistr)=gather(AgeWeights_ii.*ones(1,T));
        elseif all(size(AgeWeights)==[1,N_j])
            % Does not depend on transition path period
            AgeWeights_T.(iistr)=gather(AgeWeights_ii'.*ones(1,T));
        else
            fprintf('Following error applies to agent permanent type: %s \n',Names_i{ii})
            error('The age weights parameter seems to be the wrong size')
        end
    else % not a structure, so must apply to all permanent types
        if all(size(AgeWeights)==[N_j,1])
            % Does not depend on transition path period
            AgeWeights_T.(iistr)=gather(AgeWeights.*ones(1,T));
        elseif all(size(AgeWeights)==[1,N_j])
            % Does not depend on transition path period
            AgeWeights_T.(iistr)=gather(AgeWeights'.*ones(1,T));
        else
            error('The age weights parameter seems to be the wrong size')
        end
    end
    % Check ParamPath to see if the AgeWeights vary over the transition
    % (and overwrite AgeWeights_T.(iistr) if it does)
    temp=strcmp(ParamPathNames,AgeWeightsParamNames{1});
    if any(temp)
        transpathoptions.ageweightstrivial=0; % AgeWeights vary over the transition
        [~,kk]=max(temp); % Get index for the AgeWeightsParamNames{1} in ParamPathNames
        % Create AgeWeights_T
        AgeWeights_T.(iistr)=ParamPath(:,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk))'; % This will always be N_j-by-T (as transpose)
        % Note: still leave it in ParamPath just in case it is used in AggVars or somesuch
    end
    % Because ptypes hardcodes transpathoptions.ageweightstrivial=0 and fastOLG=1, we need
    AgeWeights_T.(iistr)=repelem(AgeWeights_T.(iistr),N_a,1); % simoptions.fastOLG=1 so this is (a,j)-by-1


    %% Set up jequalOneDist_T.(iistr) [hardcodes transpathoptions.trivialjequalonedist=0 and simoptions.fastOLG=1]
    if ~isstruct(jequalOneDist)
        jequalOneDist_temp=gpuArray(jequalOneDist);
    else % jequalOneDist is a structure
        jequalOneDist_temp=gpuArray(jequalOneDist.(iistr));
    end
    % Check if jequalOneDistPath is a path or not (and reshape appropriately)
    temp=size(jequalOneDist_temp);
    if temp(end)==T % jequalOneDist depends on T
        % transpathoptions.trivialjequalonedist=0; hardcoded for ptypes
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
        jequalOneDist_T.(iistr)=jequalOneDist_temp;
    else
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
        jequalOneDist_T.(iistr)=jequalOneDist_temp.*ones(1,T,'gpuArray');
    end

end


%%
if transpathoptions.parallel==2 
   PricePathOld=gpuArray(PricePathOld);
end

% GeneralEqmEqnNames=fieldnames(GeneralEqmEqns);
% for gg=1:length(GeneralEqmEqnNames)
%     GeneralEqmEqnParamNames{gg}=getAnonymousFnInputNames(GeneralEqmEqns.(GeneralEqmEqnNames{gg}));
% end

%%
if transpathoptions.stockvars==1 
    error('transpathoptions.stockvars=1 not yet implemented with PType \n')
end

if transpathoptions.verbose==1
    fprintf('Completed setup, beginning transition computination \n')
end

%% Set up GEnewprice==3 (if relevant)
if transpathoptions.GEnewprice==3
    transpathoptions.weightscheme=0; % Don't do any weightscheme, is already taken care of by GEnewprice=3
    
    if isstruct(GeneralEqmEqns) 
        % Need to make sure that order of rows in transpathoptions.GEnewprice3.howtoupdate
        % Is same as order of fields in GeneralEqmEqns
        % I do this by just reordering rows of transpathoptions.GEnewprice3.howtoupdate
        temp=transpathoptions.GEnewprice3.howtoupdate;
        GEeqnNames=fieldnames(GeneralEqmEqns);
        for tt=1:length(GEeqnNames)
            for jj=1:size(temp,1)
                if strcmp(temp{jj,1},GEeqnNames{tt}) % Names match
                    transpathoptions.GEnewprice3.howtoupdate{tt,1}=temp{jj,1};
                    transpathoptions.GEnewprice3.howtoupdate{tt,2}=temp{jj,2};
                    transpathoptions.GEnewprice3.howtoupdate{tt,3}=temp{jj,3};
                    transpathoptions.GEnewprice3.howtoupdate{tt,4}=temp{jj,4};
                end
            end
        end
        nGeneralEqmEqns=length(GEeqnNames);
    else
        nGeneralEqmEqns=length(GeneralEqmEqns);
    end
    transpathoptions.GEnewprice3.add=[transpathoptions.GEnewprice3.howtoupdate{:,3}];
    transpathoptions.GEnewprice3.factor=[transpathoptions.GEnewprice3.howtoupdate{:,4}];
    transpathoptions.GEnewprice3.keepold=ones(size(transpathoptions.GEnewprice3.factor));
    transpathoptions.GEnewprice3.keepold=ones(size(transpathoptions.GEnewprice3.factor));
    tempweight=transpathoptions.oldpathweight;
    transpathoptions.oldpathweight=zeros(size(transpathoptions.GEnewprice3.factor));
    for tt=1:length(transpathoptions.GEnewprice3.factor)
        if transpathoptions.GEnewprice3.factor(tt)==Inf
            transpathoptions.GEnewprice3.factor(tt)=1;
            transpathoptions.GEnewprice3.keepold(tt)=0;
            transpathoptions.oldpathweight(tt)=tempweight;
        end
    end
    if size(transpathoptions.GEnewprice3.howtoupdate,1)==nGeneralEqmEqns && nGeneralEqmEqns==length(PricePathNames)
        % do nothing, this is how things should be
    else
        fprintf('ERROR: transpathoptions.GEnewprice3.howtoupdate does not fit with GeneralEqmEqns (different number of conditions/prices) \n')
    end
    transpathoptions.GEnewprice3.permute=zeros(size(transpathoptions.GEnewprice3.howtoupdate,1),1);
    for tt=1:size(transpathoptions.GEnewprice3.howtoupdate,1) % number of rows is the number of prices (and number of GE conditions)
        for jj=1:length(PricePathNames)
            if strcmp(transpathoptions.GEnewprice3.howtoupdate{tt,2},PricePathNames{jj})
                transpathoptions.GEnewprice3.permute(tt)=jj;
            end
        end
    end
    if isfield(transpathoptions,'updateaccuracycutoff')==0
        transpathoptions.updateaccuracycutoff=0; % No cut-off (only changes in the price larger in magnitude that this will be made (can be set to, e.g., 10^(-6) to help avoid changes at overly high precision))
    end
end


%% Check if using _tminus1 and/or _tplus1 variables.
if isstruct(FnsToEvaluate) && isstruct(GeneralEqmEqns)
    [tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tminus1paramNames,tplus1pricePathkk]=inputsFindtplus1tminus1(FnsToEvaluate,GeneralEqmEqns,PricePathNames,ParamPathNames);
    if transpathoptions.verbose>1
        tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tminus1paramNames,tplus1pricePathkk
    end
else
    tplus1priceNames=[];
    tminus1priceNames=[];
    tminus1paramNames=[];
    tminus1AggVarsNames=[];
    tplus1pricePathkk=[];
end
 
use_tplus1price=0;
if length(tplus1priceNames)>0
    use_tplus1price=1;
end
use_tminus1price=0;
if length(tminus1priceNames)>0
    use_tminus1price=1;
    for ii=1:length(tminus1priceNames)
        if ~isfield(transpathoptions.initialvalues,tminus1priceNames{ii})
            error('Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1priceNames{ii})
        end
    end
end
use_tminus1params=0;
if length(tminus1paramNames)>0
    use_tminus1params=1;
    for ii=1:length(tminus1paramNames)
        if ~isfield(transpathoptions.initialvalues,tminus1paramNames{ii})
            error('Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1paramNames{ii})
        end
    end
end
use_tminus1AggVars=0;
if length(tminus1AggVarsNames)>0
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
end



%%

if transpathoptions.verbose==1
    transpathoptions
end

if transpathoptions.GEnewprice~=2
    % For permanent type, there is just one shooting command,
    % because things like z,e, and fastOLG are handled on a per-PType basis (to permit that they differ across ptype)
    PricePathOld=TransitionPath_Case1_FHorz_PType_shooting(PricePathOld, PricePathNames, ParamPath, ParamPathNames, T, V_final, AgentDist_init, jequalOneDist_T, AgeWeights_T, FnsToEvaluate, GeneralEqmEqns, PricePathSizeVec, ParamPathSizeVec, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, transpathoptions, PTypeStructure);
    % Switch the solution into structure for output.
    for ii=1:length(PricePathNames)
        PricePath.(PricePathNames{ii})=PricePathOld(:,ii);
    end
    if transpathoptions.stockvars==1
        for ii=1:length(StockVarsPathNames)
            PricePath.(StockVarsPathNames{ii})=StockVarsPathOld(:,ii);
        end
    end
    return
end


%%
if transpathoptions.GEnewprice==2 % Function minimization
    % Have not attempted implementing this for PType yet, no point until I
    % get it to be useful without PType
end


end