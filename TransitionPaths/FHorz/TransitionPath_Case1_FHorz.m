function PricePath=TransitionPath_Case1_FHorz(PricePathOld, ParamPath, T, V_final, StationaryDist_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, AgeWeightsParamNames, transpathoptions, simoptions, vfoptions)
% This code will work for all transition paths except those that involve at
% change in the transition matrix pi_z (can handle a change in pi_z, but
% only if it is a 'surprise', not anticipated changes) 
%
% PricePathOld is a structure with fields names being the Prices and each field containing a T-by-1 path.
% ParamPath is a structure with fields names being the parameter names of those parameters which change over the path and each field containing a T-by-1 path.
%
% Only works for v2

% Remark to self: No real need for T as input, as this is anyway the length of PricePathOld


%% Check which transpathoptions have been used, set all others to defaults 
if exist('transpathoptions','var')==0
    disp('No transpathoptions given, using defaults')
    %If transpathoptions is not given, just use all the defaults
    transpathoptions.tolerance=10^(-4);
    transpathoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    transpathoptions.GEnewprice=1; % 1 is shooting algorithm, 0 is that the GE should evaluate to zero and the 'new' is the old plus the "non-zero" (for each time period seperately); 
                                   % 2 is to do optimization routine with 'distance between old and new path', 3 is just same as 0, but easier to set up
    transpathoptions.oldpathweight=0.9; % default =0.9
    transpathoptions.weightscheme=1; % default =1
    transpathoptions.Ttheta=1;
    transpathoptions.maxiterations=500; % Based on personal experience anything that hasn't converged well before this is just hung-up on trying to get the 4th decimal place (typically because the number of grid points was not large enough to allow this level of accuracy).
    transpathoptions.verbose=0;
    transpathoptions.verbosegraphs=0;
    transpathoptions.graphpricepath=0;
    transpathoptions.graphaggvarspath=0;
    transpathoptions.historyofpricepath=0;
    transpathoptions.stockvars=0;
    transpathoptions.fastOLG=0;
    % transpathoptions.updateageweights % Don't declare if not being used
else
    %Check transpathoptions for missing fields, if there are some fill them with the defaults
    if isfield(transpathoptions,'tolerance')==0
        transpathoptions.tolerance=10^(-4);
    end
    if isfield(transpathoptions,'parallel')==0
        transpathoptions.parallel=1+(gpuDeviceCount>0);
    end
    if isfield(transpathoptions,'GEnewprice')==0
        transpathoptions.GEnewprice=1; % 0 is that the GE should evaluate to zero and the 'new' is the old plus the "non-zero" (for each time period seperately);
                                       % 1 is shooting algorithm, 
                                       % 2 is to do optimization routine with 'distance between old and new path'
                                       % 3 is just same as 0, but easier to set 
    end
    if isfield(transpathoptions,'oldpathweight')==0
        transpathoptions.oldpathweight=0.9;
        % Note that when using transpathoptions.GEnewprice==3
        % Implicitly it is setting transpathoptions.oldpathweight=0
        % because the user anyway has to specify them as part of setup
    end
    if isfield(transpathoptions,'weightscheme')==0
        transpathoptions.weightscheme=1;
    end
    if isfield(transpathoptions,'Ttheta')==0
        transpathoptions.Ttheta=1;
    end
    if isfield(transpathoptions,'maxiterations')==0
        transpathoptions.maxiterations=500;
    end
    if isfield(transpathoptions,'verbose')==0
        transpathoptions.verbose=0;
    end
    if isfield(transpathoptions,'verbosegraphs')==0
        transpathoptions.verbosegraphs=0;
    end
    if isfield(transpathoptions,'graphpricepath')==0
        transpathoptions.graphpricepath=0;
    end
    if isfield(transpathoptions,'graphaggvarspath')==0
        transpathoptions.graphaggvarspath=0;
    end
    if isfield(transpathoptions,'historyofpricepath')==0
        transpathoptions.historyofpricepath=0;
    end
    if isfield(transpathoptions,'usestockvars')==0 % usestockvars is solely for internal use, the user does not need to set it
        if isfield(transpathoptions,'stockvarinit')==0 && isfield(transpathoptions,'usestockvars')==0 && isfield(transpathoptions,'usestockvars')==0
            transpathoptions.usestockvars=0;
        else
            transpathoptions.usestockvars=1; % If usestockvars has not itself been declared, but at least one of the stock variable options has then set usestockvars to 1.
        end
    end
    if transpathoptions.usestockvars==1 % Note: If this is not inputted then it is created by the above lines.
        if isfield(transpathoptions,'stockvarinit')==0
            error('ERROR: transpathoptions includes some Stock Variable options but is missing stockvarinit \n')
        elseif isfield(transpathoptions,'stockvarpath0')==0
            error('ERROR: transpathoptions includes some Stock Variable options but is missing stockvarpath0 \n')
        elseif isfield(transpathoptions,'stockvareqns')==0
            error('ERROR: transpathoptions includes some Stock Variable options but is missing stockvareqns \n')
        end
    end
    if isfield(transpathoptions,'fastOLG')==0
        transpathoptions.fastOLG=0;
    end
    % transpathoptions.updateageweights %Don't declare if not being used
end

if isfield(transpathoptions,'p_eqm_init')
    p_eqm_init=transpathoptions.p_eqm_init;
    use_p_eqm_init=1;
else
    use_p_eqm_init=0;
end


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

PricePathNames
ParamPathNames


%% Check which vfoptions have been used, set all others to defaults 
if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
    vfoptions.parallel=transpathoptions.parallel;
    vfoptions.returnmatrix=2;
    vfoptions.verbose=0;
    vfoptions.lowmemory=0;
    vfoptions.exoticpreferences='None';
    vfoptions.polindorval=1;
    vfoptions.policy_forceintegertype=0;
    vfoptions.endotype=0;
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if isfield(vfoptions,'parallel')==0
        vfoptions.parallel=transpathoptions.parallel; % GPU where available, otherwise parallel CPU.
    end
    if vfoptions.parallel==2
        vfoptions.returnmatrix=2; % On GPU, must use this option
    end
    if isfield(vfoptions,'lowmemory')==0
        vfoptions.lowmemory=0;
    end
    if isfield(vfoptions,'verbose')==0
        vfoptions.verbose=0;
    end
    if isfield(vfoptions,'returnmatrix')==0
        if isa(ReturnFn,'function_handle')==1
            vfoptions.returnmatrix=0;
        else
            vfoptions.returnmatrix=1;
        end
    end
    if isfield(vfoptions,'exoticpreferences')==0
        vfoptions.exoticpreferences='None';
    end
    if strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
        if ~isfield(vfoptions,'quasi_hyperbolic')
            vfoptions.quasi_hyperbolic='Naive'; % This is the default, alternative is 'Sophisticated'.
        elseif ~strcmp(vfoptions.quasi_hyperbolic,'Naive') && ~strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
            fprintf('ERROR: when using Quasi-Hyperbolic discounting vfoptions.quasi_hyperbolic must be either Naive or Sophisticated \n')
            dbstack
            return
        end
    end
    if isfield(vfoptions,'polindorval')==0
        vfoptions.polindorval=1;
    end
    if isfield(vfoptions,'policy_forceintegertype')==0
        vfoptions.policy_forceintegertype=0;
    end
    if isfield(vfoptions,'endotype')==0
        vfoptions.endotype=0;
    end
end

%% Check which simoptions have been used, set all others to defaults 
if isfield(transpathoptions,'simoptions')==1
    simoptions=transpathoptions.simoptions;
end
if exist('simoptions','var')==0
    simoptions.nsims=10^4;
    simoptions.parallel=transpathoptions.parallel; % GPU where available, otherwise parallel CPU.
    simoptions.verbose=0;
    try 
        PoolDetails=gcp;
        simoptions.ncores=PoolDetails.NumWorkers;
    catch
        simoptions.ncores=1;
    end
    simoptions.iterate=1;
    simoptions.tolerance=10^(-9);
else
    %Check vfoptions for missing fields, if there are some fill them with
    %the defaults
    if isfield(simoptions,'tolerance')==0
        simoptions.tolerance=10^(-9);
    end
    if isfield(simoptions,'nsims')==0
        simoptions.nsims=10^4;
    end
    if isfield(simoptions,'parallel')==0
        simoptions.parallel=transpathoptions.parallel;
    end
    if isfield(simoptions,'verbose')==0
        simoptions.verbose=0;
    end
    if isfield(simoptions,'ncores')==0
        try
            PoolDetails=gcp;
            simoptions.ncores=PoolDetails.NumWorkers;
        catch
            simoptions.ncores=1;
        end
    end
    if isfield(simoptions,'iterate')==0
        simoptions.iterate=1;
    end
end

%% Check the sizes of some of the inputs
if isempty(n_d)
    N_d=0;
else
    N_d=prod(n_d);
end
N_z=prod(n_z);
N_a=prod(n_a);

if N_d>0
    if size(d_grid)~=[N_d, 1]
        fprintf('ERROR: d_grid is not the correct shape (should be of size N_d-by-1) \n')
        fprintf('       d_grid is of size: %i by % i, while N_d is %i \n',size(d_grid,1),size(d_grid,2),N_d)
        dbstack
        return
    end
end
if size(a_grid)~=[N_a, 1]
    fprintf('ERROR: a_grid is not the correct shape (should be of size N_a-by-1) \n')
    fprintf('       a_grid is of size: %i by % i, while N_a is %i \n',size(a_grid,1),size(a_grid,2),N_a)
    dbstack
    return
elseif size(z_grid)~=[N_z, 1]
    fprintf('ERROR: z_grid is not the correct shape (should be of size N_z-by-1) \n')
    fprintf('       z_grid is of size: %i by % i, while N_z is %i \n',size(z_grid,1),size(z_grid,2),N_z)
    dbstack
    return
elseif size(pi_z)~=[N_z, N_z]
    fprintf('ERROR: pi is not of size N_z-by-N_z \n')
    fprintf('       pi is of size: %i by % i, while N_z is %i \n',size(pi_z,1),size(pi_z,2),N_z)
    dbstack
    return
end
if isstruct(GeneralEqmEqns)
    if length(PricePathNames)~=length(fieldnames(GeneralEqmEqns))
        fprintf('ERROR: Initial PricePath contains less variables than GeneralEqmEqns (structure) \n')
        fprintf('       They are: %i and % i respectively \n',length(PricePathNames), length(fieldnames(GeneralEqmEqns)))
        dbstack
        return
    end
else
    if length(PricePathNames)~=length(GeneralEqmEqns)
        disp('ERROR: Initial PricePath contains less variables than GeneralEqmEqns')
        fprintf('       They are: %i and % i respectively \n',length(PricePathNames), length(GeneralEqmEqns))
        dbstack
        return
    end
end

%%
if transpathoptions.parallel==2 
   % If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
   pi_z=gpuArray(pi_z);
   if N_d>0
       d_grid=gpuArray(d_grid);
   end
   a_grid=gpuArray(a_grid);
   z_grid=gpuArray(z_grid);
   PricePathOld=gpuArray(PricePathOld);
else
   % If using CPU make sure all the relevant inputs are CPU arrays (not standard arrays)
   % This may be completely unnecessary.
   pi_z=gather(pi_z);
   if N_d>0
       d_grid=gather(d_grid);
   end
   a_grid=gather(a_grid);
   z_grid=gather(z_grid);
   PricePathOld=gather(PricePathOld);
end

%%
if transpathoptions.usestockvars==1 
    % Get the stock variable objects out of transpathoptions.
    StockVariable_init=transpathoptions.stockvarinit;
    StockVariableEqns=transpathoptions.stockvareqns.lawofmotion;
    StockVarsPathNames=fieldnames(StockVariableEqns);
    StockVarsPathOld=zeros(T,length(StockVarsPathNames));
    for ss=1:length(StockVarsPathNames)
%         temp=getAnonymousFnInputNames(StockVariableEqns.(StockVarsPathNames{ss}));
%         StockVariableEqnParamNames(ss).Names={temp{:}}; % the first inputs will always be (d,aprime,a,z)
%         StockVariableEqns2{ss}=StockVariableEqnsStruct.(StockVarsPathNames{ss});
        StockVarsPathOld(:,ss)=transpathoptions.stockvarpath0.(StockVarsPathNames{ss});
    end    
%     StockVariableEqns=StockVariableEqns2;
    if transpathoptions.parallel==2
        StockVarsPathOld=gpuArray(StockVarsPathOld);
    end
end

%% Check if z_grid and/or pi_z depend on prices. If not then create pi_z_J and z_grid_J for the entire transition before we start
% If 'exogenous shock fn' is used, then precompute it to save evaluating it numerous times
% Check if using 'exogenous shock fn' (exogenous state has a grid and transition matrix that depends on age)

transpathoptions.zpathprecomputed=0;
if isfield(vfoptions,'pi_z_J')
    transpathoptions.zpathprecomputed=1;
    transpathoptions.zpathtrivial=1; % z_grid_J and pi_z_J are not varying over the path
elseif isfield(vfoptions,'ExogShockFn')
    % Note: If ExogShockFn depends on the path, it must be done via a parameter
    % that depends on the path (i.e., via ParamPath or PricePath)
    overlap=0;
    for ii=1:length(vfoptions.ExogShockFnParamNames)
        if strcmp(vfoptions.ExogShockFnParamNames{ii},PricePathNames)
            overlap=1;
        end
    end
    if overlap==0
        transpathoptions.zpathprecomputed=1;
        % If ExogShockFn does not depend on any of the prices (in PricePath), then
        % we can simply create it now rather than within each 'subfn' or 'p_grid'
        
        % Check if it depends on the ParamPath
        transpathoptions.zpathtrivial=1;
        for ii=1:length(vfoptions.ExogShockFnParamNames)
            if strcmp(vfoptions.ExogShockFnParamNames{ii},ParamPathNames)
                transpathoptions.zpathtrivial=0;
            end
        end
        if transpathoptions.zpathtrivial==1
            pi_z_J=zeros(N_z,N_z,N_j,'gpuArray');
            z_grid_J=zeros(N_z,N_j,'gpuArray');
            for jj=1:N_j
                if isfield(vfoptions,'ExogShockFnParamNames')
                    ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for ii=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                    end
                    [z_grid,pi_z]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
                else
                    [z_grid,pi_z]=simoptions.ExogShockFn(jj);
                end
                pi_z_J(:,:,jj)=gpuArray(pi_z);
                z_grid_J(:,jj)=gpuArray(z_grid);
            end
            % Now store them in vfoptions and simoptions
            vfoptions.pi_z_J=pi_z_J;
            vfoptions.z_grid_J=z_grid_J;
            simoptions.pi_z_J=pi_z_J;
            simoptions.z_grid_J=z_grid_J;
        elseif transpathoptions.zpathtrivial==0
            % z_grid_J and/or pi_z_J varies along the transition path (but only depending on ParamPath, not PricePath
            transpathoptions.pi_z_J_T=zeros(N_z,N_z,N_j,T,'gpuArray');
            transpathoptions.z_grid_J_T=zeros(sum(n_z),N_j,T,'gpuArray');
            pi_z_J=zeros(N_z,N_z,N_j,'gpuArray');
            z_grid_J=zeros(sum(n_z),N_j,'gpuArray');
            for tt=1:T
                for ii=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{ii})=ParamPathStruct.(ParamPathNames{ii});
                end
                % Note, we know the PricePath is irrelevant for the current purpose
                for jj=1:N_j
                    if isfield(vfoptions,'ExogShockFnParamNames')
                        ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                        ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                        for ii=1:length(ExogShockFnParamsVec)
                            ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                        end
                        [z_grid,pi_z]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
                    else
                        [z_grid,pi_z]=simoptions.ExogShockFn(jj);
                    end
                    pi_z_J(:,:,jj)=gpuArray(pi_z);
                    z_grid_J(:,jj)=gpuArray(z_grid);
                end
                transpathoptions.pi_z_J_T(:,:,:,tt)=pi_z_J;
                transpathoptions.z_grid_J_T(:,:,tt)=z_grid_J;
            end
        end
    end
end


%% Handle ReturnFn and FnsToEvaluate structures
l_d=length(n_d);
if n_d(1)==0
    l_d=0;
end
l_a=length(n_a);
l_z=length(n_z);
l_a_temp=l_a;
l_z_temp=l_z;
if max(vfoptions.endotype)==1
    l_a_temp=l_a-sum(vfoptions.endotype);
    l_z_temp=l_z+sum(vfoptions.endotype);
end
% Create ReturnFnParamNames
temp=getAnonymousFnInputNames(ReturnFn);
if length(temp)>(l_d+l_a_temp+l_a_temp+l_z_temp)
    ReturnFnParamNames={temp{l_d+l_a_temp+l_a_temp+l_z_temp+1:end}}; % the first inputs will always be (d,aprime,a,z)
else
    ReturnFnParamNames={};
end

if ~isstruct(FnsToEvaluate)
    error('Transition paths only work with version 2+ (FnsToEvaluate has to be a structure)')
end


%%
transpathoptions
if transpathoptions.GEnewprice~=2
    if transpathoptions.parallel==2
        if transpathoptions.usestockvars==0
            if transpathoptions.fastOLG==0
                PricePathOld=TransitionPath_Case1_FHorz_shooting(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, StationaryDist_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, AgeWeightsParamNames, ReturnFnParamNames, vfoptions, simoptions,transpathoptions);
            else % use fastOLG setting
                PricePathOld=TransitionPath_Case1_FHorz_shooting_fastOLG(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, StationaryDist_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, AgeWeightsParamNames, ReturnFnParamNames, vfoptions, simoptions,transpathoptions);
            end
        else % transpathoptions.usestockvars==1
            warning('StockVars does not yet work correctly')
            if transpathoptions.fastOLG==0                
                [PricePathOld,StockVarsPathOld]=TransitionPath_Case1_FHorz_StockVar_shooting(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, StockVarsPathOld, StockVarsPathNames, T, V_final, StationaryDist_init, StockVariable_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, StockVariableEqns, Parameters, DiscountFactorParamNames, AgeWeightsParamNames, ReturnFnParamNames, vfoptions, simoptions,transpathoptions);
            else % use fastOLG setting
                [PricePathOld,StockVarsPathOld]=TransitionPath_Case1_FHorz_StockVar_shooting_fastOLG(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, StockVarsPathOld, StockVarsPathNames, T, V_final, StationaryDist_init, StockVariable_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, StockVariableEqns, Parameters, DiscountFactorParamNames, AgeWeightsParamNames, ReturnFnParamNames, vfoptions, simoptions,transpathoptions);
            end
        end
    else
        error('VFI Toolkit does not offer transition path without gpu. Would be too slow to be useful.')
    end
    % Switch the solution into structure for output.
    for ii=1:length(PricePathNames)
        PricePath.(PricePathNames{ii})=PricePathOld(:,ii);
    end
    if transpathoptions.usestockvars==1
        for ii=1:length(StockVarsPathNames)
            PricePath.(StockVarsPathNames{ii})=StockVarsPathOld(:,ii);
        end
    end
    return
end


%% Set up transition path as minimization of a function (default is to use as objective the weighted sum of squares of the general eqm conditions)
l_p=size(PricePathOld,2);
if transpathoptions.verbose==1
    transpathoptions
end
if transpathoptions.verbose==1
    DiscountFactorParamNames
    ReturnFnParamNames
    ParamPathNames
    PricePathNames
end

PricePathVec=gather(reshape(PricePathOld,[T*length(PricePathNames),1])); % Has to be vector of fminsearch. Additionally, provides a double check on sizes.

% I HAVEN'T GOTTEN THIS TO WORK WELL ENOUGH THAT I AM COMFORTABLE LEAVING IT ENABLED
if transpathoptions.GEnewprice==2 % Function minimization
    error('transpathoptions.GEnewprice==2 not currently enabled')
%     if n_d(1)==0
%         GeneralEqmConditionsPathFn=@(pricepath) TransitionPath_Case1_FHorz_no_d_subfn(pricepathPricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, AgeWeightsParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, vfoptions, simoptions,transpathoptions);
%     else
%         GeneralEqmConditionsPathFn=@(pricepath) TransitionPath_Case1_FHorz_subfn(pricepath, PricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, AgeWeightsParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, vfoptions, simoptions,transpathoptions);
%     end
end

% if transpathoptions.GEnewprice2algo==0
% [PricePath,~]=fminsearch(GeneralEqmConditionsPathFn,PricePathVec);
% else
%     [PricePath,~]=fminsearch(GeneralEqmConditionsPathFn,PricePathOld);
% end
%
% LOOK INTO USING 'SURROGATE OPTIMIZATION'

if transpathoptions.parallel==2
    PricePath=gpuArray(reshape(PricePath,[T,length(PricePathNames)])); % Switch back to appropriate shape (out of the vector required to use fminsearch)
else
    PricePath=gather(reshape(PricePath,[T,length(PricePathNames)])); % Switch back to appropriate shape (out of the vector required to use fminsearch)    
end

for ii=1:length(PricePathNames)
    PricePath.(PricePathNames{ii})=PricePathOld(:,ii);
end


end