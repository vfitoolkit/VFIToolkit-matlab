function PricePath=TransitionPath_Case1_FHorz(PricePathOld, ParamPath, T, V_final, StationaryDist_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, AgeWeightsParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, transpathoptions, simoptions, vfoptions)
% This code will work for all transition paths except those that involve at
% change in the transition matrix pi_z (can handle a change in pi_z, but
% only if it is a 'surprise', not anticipated changes) 
%
% PricePathOld is a structure with fields names being the Prices and each field containing a T-by-1 path.
% ParamPath is a structure with fields names being the parameter names of those parameters which change over the path and each field containing a T-by-1 path.

% Remark to self: No real need for T as input, as this is anyway the length of PricePathOld

% Note: Internally PricePathOld is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'. 
PricePathNames=fieldnames(PricePathOld);
PricePathStruct=PricePathOld; 
PricePathOld=zeros(T,length(PricePathNames));
for ii=1:length(PricePathNames)
    PricePathOld(:,ii)=PricePathStruct.(PricePathNames{ii});
end
ParamPathNames=fieldnames(ParamPath);
ParamPathStruct=ParamPath; 
ParamPath=zeros(T,length(ParamPathNames));
for ii=1:length(ParamPathNames)
    ParamPath(:,ii)=ParamPathStruct.(ParamPathNames{ii});
end

PricePath=struct();

%% Check which transpathoptions have been used, set all others to defaults 
if exist('transpathoptions','var')==0
    disp('No transpathoptions given, using defaults')
    %If transpathoptions is not given, just use all the defaults
    transpathoptions.tolerance=10^(-4);
    transpathoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    transpathoptions.fastOLG=0;
    transpathoptions.exoticpreferences=0;
    transpathoptions.oldpathweight=0.9; % default =0.9
    transpathoptions.weightscheme=1; % default =1
    transpathoptions.Ttheta=1;
    transpathoptions.maxiterations=500; % Based on personal experience anything that hasn't converged well before this is just hung-up on trying to get the 4th decimal place (typically because the number of grid points was not large enough to allow this level of accuracy).
    transpathoptions.verbose=0;
    transpathoptions.verbosegraphs=0;
    transpathoptions.GEnewprice=2;
    transpathoptions.historyofpricepath=0;
    transpathoptions.stockvars=0;
else
    %Check transpathoptions for missing fields, if there are some fill them with the defaults
    if isfield(transpathoptions,'tolerance')==0
        transpathoptions.tolerance=10^(-4);
    end
    if isfield(transpathoptions,'parallel')==0
        transpathoptions.parallel=1+(gpuDeviceCount>0);
    end
    if isfield(transpathoptions,'fastOLG')==0
        transpathoptions.fastOLG=0;
    end
    if isfield(transpathoptions,'exoticpreferences')==0
        transpathoptions.exoticpreferences=0;
    end
    if isfield(transpathoptions,'oldpathweight')==0
        transpathoptions.oldpathweight=0.9;
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
    if isfield(transpathoptions,'GEnewprice')==0
        transpathoptions.GEnewprice=2;
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
            fprintf('ERROR: transpathoptions includes some Stock Variable options but is missing stockvarinit \n')
            dbstack
            return
        elseif isfield(transpathoptions,'stockvarpath0')==0
            fprintf('ERROR: transpathoptions includes some Stock Variable options but is missing stockvarpath0 \n')
            dbstack
            return
        elseif isfield(transpathoptions,'stockvareqns')==0
            fprintf('ERROR: transpathoptions includes some Stock Variable options but is missing stockvareqns \n')
            dbstack
            return
        end
    end
end

%% Check which vfoptions have been used, set all others to defaults 
if isfield(transpathoptions,'vfoptions')==1
    vfoptions=transpathoptions.vfoptions;
end


%% Check which vfoptions have been used, set all others to defaults 
if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
%     vfoptions.exoticpreferences=0;
    vfoptions.parallel=transpathoptions.parallel;
    vfoptions.returnmatrix=2;
    vfoptions.verbose=0;
    vfoptions.lowmemory=0;
    vfoptions.exoticpreferences=0;
    vfoptions.polindorval=1;
    vfoptions.policy_forceintegertype=0;
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
        vfoptions.exoticpreferences=0;
    end
    if isfield(vfoptions,'polindorval')==0
        vfoptions.polindorval=1;
    end
    if isfield(vfoptions,'policy_forceintegertype')==0
        vfoptions.policy_forceintegertype=0;
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
N_d=prod(n_d);
N_z=prod(n_z);
N_a=prod(n_a);

if N_d>0
    if size(d_grid)~=[N_d, 1]
        disp('ERROR: d_grid is not the correct shape (should be of size N_d-by-1)')
        dbstack
        return
    end
end
if size(a_grid)~=[N_a, 1]
    disp('ERROR: a_grid is not the correct shape (should be of size N_a-by-1)')
    dbstack
    return
elseif size(z_grid)~=[N_z, 1]
    disp('ERROR: z_grid is not the correct shape (should be of size N_z-by-1)')
    dbstack
    return
elseif size(pi_z)~=[N_z, N_z]
    disp('ERROR: pi is not of size N_z-by-N_z')
    dbstack
    return
end
if length(PricePathNames)~=length(GeneralEqmEqns)
    disp('ERROR: Initial PricePath contains less variables than GeneralEqmEqns')
    dbstack
    return
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
%     StockVariableEqns=transpathoptions.stockvareqns.lawofmotion;
    % and switch from structure/names to matrix for VFI Toolkit internals.
    StockVarsPathNames=fieldnames(transpathoptions.stockvarpath0);
    StockVarsStruct=transpathoptions.stockvarpath0;
    StockVarsPathOld=zeros(T,length(StockVarsPathNames));
    StockVariableEqns=cell(1,length(StockVarsPathNames));
    for ii=1:length(StockVarsPathNames)
        StockVarsPathOld(:,ii)=StockVarsStruct.(StockVarsPathNames{ii});
        StockVariableEqnParamNames(ii).Names=transpathoptions.stockvareqns.Names.(StockVarsPathNames{ii}); % Really, what I should do is redesign the way the GeneralEqm equations and names work so they are in a structure, and then just leave this in it's structure form as well.
        StockVariableEqns{ii}=transpathoptions.stockvareqns.lawofmotion.(StockVarsPathNames{ii});
    end
    if transpathoptions.parallel==2
        StockVarsPathOld=gpuArray(StockVarsPathOld);
    end
end

%%
if transpathoptions.exoticpreferences~=0
    disp('ERROR: Only transpathoptions.exoticpreferences==0 is supported by TransitionPath_Case1')
    dbstack
end

if transpathoptions.GEnewprice==1
    if transpathoptions.parallel==2
        if transpathoptions.usestockvars==0
            if transpathoptions.fastOLG==0
                PricePathOld=TransitionPath_Case1_FHorz_shooting(PricePathOld, PricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, AgeWeightsParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, vfoptions, simoptions, transpathoptions);
            else % use fastOLG setting
%                 if vfoptions.lowmemory==0
                    PricePathOld=TransitionPath_Case1_FHorz_shooting_fastOLG(PricePathOld, PricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, AgeWeightsParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, vfoptions, simoptions, transpathoptions);
%                 elseif vfoptions.lowmemory==1
%                     PricePathOld=TransitionPath_Case1_FHorz_shooting_fastOLG_lowmem(PricePathOld, PricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, AgeWeightsParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, vfoptions, simoptions, transpathoptions);
%                 end
            end
        else % transpathoptions.usestockvars==1
            if transpathoptions.fastOLG==0
                % NOT YET IMPLEMENTED
%                 PricePathOld=TransitionPath_Case1_FHorz_shooting(PricePathOld, PricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, AgeWeightsParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, vfoptions, simoptions, transpathoptions);
            else % use fastOLG setting
                [PricePathOld,StockVarsPathOld]=TransitionPath_Case1_FHorz_StockVar_shooting_fastOLG(PricePathOld, PricePathNames, StockVarsPathOld, StockVarsPathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, StockVariable_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, StockVariableEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, AgeWeightsParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, StockVariableEqnParamNames, vfoptions, simoptions, transpathoptions);
            end
        end
    else
        PricePathOld=TransitionPath_Case1_FHorz_Par1_shooting(PricePathOld, PricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, AgeWeightsParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, vfoptions, simoptions, transpathoptions);
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

%% Set up transition path as minimization of a function (default is to use as objective the weighted sum of squares of the general eqm conditions)
PricePathVec=gather(reshape(PricePathOld,[T*length(PricePathNames),1])); % Has to be vector of fminsearch. Additionally, provides a double check on sizes.

if transpathoptions.GEnewprice==2 % Function minimization
    % CURRENTLY ONLY WORKS WITH GPU (PARALLEL==2)
    if n_d(1)==0
        GeneralEqmConditionsPathFn=@(pricepath) TransitionPath_Case1_FHorz_no_d_subfn(pricepathPricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, AgeWeightsParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, vfoptions, simoptions,transpathoptions);
    else
        GeneralEqmConditionsPathFn=@(pricepath) TransitionPath_Case1_FHorz_subfn(pricepath, PricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, AgeWeightsParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, vfoptions, simoptions,transpathoptions);
    end
end

% if transpathoptions.GEnewprice2algo==0
[PricePath,~]=fminsearch(GeneralEqmConditionsPathFn,PricePathVec);
% else
%     [PricePath,~]=fminsearch(GeneralEqmConditionsPathFn,PricePathOld);
% end

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