function PricePath=TransitionPath_Case1(PricePathOld, ParamPath, T, V_final, AgentDist_initial, n_d, n_a, n_z, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, transpathoptions, vfoptions, simoptions, EntryExitParamNames)
% This code will work for all transition paths except those that involve at
% change in the transition matrix pi_z (can handle a change in pi_z, but
% only if it is a 'surprise', not anticipated changes) 
%
% PricePathOld is a structure with fields names being the Prices and each field containing a T-by-1 path.
% ParamPath is a structure with fields names being the parameter names of those parameters which change over the path and each field containing a T-by-1 path.
%
% transpathoptions is not a required input.

% Remark to self: No real need for T as input, as this is anyway the length of PricePathOld

%%
% Note: Internally PricePathOld is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'. 
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
    %     PricePathOld(:,ii)=PricePathStruct.(PricePathNames{ii});
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
%     ParamPath(:,ii)=ParamPathStruct.(ParamPathNames{ii});
end

PricePath=struct();

PricePathNames
ParamPathNames


%% Check which transpathoptions have been used, set all others to defaults 
if exist('transpathoptions','var')==0
    disp('No transpathoptions given, using defaults')
    %If transpathoptions is not given, just use all the defaults
    transpathoptions.tolerance=10^(-5);
    transpathoptions.updateaccuracycutoff=10^(-9); % If the suggested update is less than this then don't bother; 10^(-9) is decent odds to be numerical error anyway (currently only works for transpathoptions.GEnewprice=3)
    transpathoptions.parallel=1+(gpuDeviceCount>0);
    transpathoptions.lowmemory=0;
    transpathoptions.GEnewprice=1; % 1 is shooting algorithm, 0 is that the GE should evaluate to zero and the 'new' is the old plus the "non-zero" (for each time period seperately); 
                                   % 2 is to do optimization routine with 'distance between old and new path', 3 is just same as 0, but easier to set up
    transpathoptions.oldpathweight=0.9; % default =0.9
    transpathoptions.weightscheme=1; % default =1
    transpathoptions.Ttheta=1;
    transpathoptions.maxiterations=1000;
    transpathoptions.verbose=0;
    transpathoptions.graphpricepath=0; % 1: creates a graph of the 'current' price path which updates each iteration.
    transpathoptions.graphaggvarspath=0; % 1: creates a graph of the 'current' aggregate variables which updates each iteration.
    transpathoptions.graphGEcondns=0;  % 1: creates a graph of the 'current' general eqm conditions which updates each iteration.
    transpathoptions.historyofpricepath=0;
    transpathoptions.stockvars=0;
    transpathoptions.weightsforpath=ones(T,length(GeneralEqmEqns)); % Won't actually be used under the defaults, but am still setting it.
    transpathoptions.tanimprovement=1;
else
    %Check transpathoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(transpathoptions,'tolerance')
        transpathoptions.tolerance=10^(-5);
    end
    if ~isfield(transpathoptions,'updateaccuracycutoff')
        transpathoptions.updateaccuracycutoff=10^(-9);
    end
    if ~isfield(transpathoptions,'parallel')
        transpathoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(transpathoptions,'lowmemory')
        transpathoptions.lowmemory=0;
    end
    if ~isfield(transpathoptions,'GEnewprice')
        transpathoptions.GEnewprice=1; % 1 is shooting algorithm, 0 is that the GE should evaluate to zero and the 'new' is the old plus the "non-zero" (for each time period seperately);
                                       % 2 is to do optimization routine with 'distance between old and new path', 3 is just same as 0, but easier to set up
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
    if ~isfield(transpathoptions,'maxiterations')
        transpathoptions.maxiterations=1000;
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
        transpathoptions.stockvars=0;
    end
    if ~isfield(transpathoptions,'weightsforpath')
        transpathoptions.weightsforpath=ones(T,length(GeneralEqmEqns));
    end
    if ~isfield(transpathoptions,'tanimprovement')
        transpathoptions.tanimprovement=1;
    end
end
if transpathoptions.parallel~=2
    error('Transition paths can only be solved if you have a GPU')
end

if isfield(transpathoptions,'p_eqm_init')
    p_eqm_init=transpathoptions.p_eqm_init;
    use_p_eqm_init=1;
else
    use_p_eqm_init=0;
end

if transpathoptions.graphGEcondns==1
    if transpathoptions.GEnewprice~=3
        error('Can only use transpathoptions.graphGEcondns=1 when using transpathoptions.GEnewprice=3')
    end
end

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
    vfoptions.solnmethod='purediscretization';
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
    if isfield(vfoptions,'solnmethod')==0
        vfoptions.solnmethod='purediscretization';
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
N_a=prod(n_a);
N_z=prod(n_z);

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

% If there is entry and exit, then send to relevant command
if isfield(simoptions,'agententryandexit')==1 % isfield(transpathoptions,'agententryandexit')==1
    error('ERROR: have not yet implemented transition path for models with entry/exit \n')
%     if ~exist('EntryExitParamNames','var')
%         fprintf('ERROR: need to input EntryExitParamNames to TransitionPath_Case1() \n')
%         PricePath=[];
%         return
%     end
%     if simoptions.agententryandexit==1% transpathoptions.agententryandexit==1
%         PricePath=TransitionPath_Case1_EntryExit(PricePathOld, PricePathNames, ParamPath, ParamPathNames, T, V_final, AgentDist_initial, n_d, n_a, n_z, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnInputNames, EntryExitParamNames, transpathoptions, vfoptions, simoptions);
%         return
% %     elseif transpathoptions.agententryandexit==2
% %         PricePath=TransitionPath_Case1_EntryExit2(PricePathOld, PricePathNames, ParamPath, ParamPathNames, T, V_final, AgentDist_initial, n_d, n_a, n_z, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnInputNames, EntryExitParamNames, transpathoptions, vfoptions, simoptions);
% %         return
%     end
end


%%
% If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
pi_z=gpuArray(pi_z);
if N_d>0
    d_grid=gpuArray(d_grid);
end
a_grid=gpuArray(a_grid);
z_grid=gpuArray(z_grid);
PricePathOld=gpuArray(PricePathOld);


%% Implement new way of handling ReturnFn inputs
ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,0,vfoptions,Parameters);

%%
if transpathoptions.GEnewprice~=2
    if transpathoptions.lowmemory==1
        % The lowmemory option is going to use gpu (but loop over z instead of parallelize) for value fn, and then use sparse matrices on cpu when iterating on agent dist.
        % Note: Just using vfoptions.lowmemory=1 will do the loop over z for value fn, but would not include the sparse matrix for agent distribtion
        if N_d==0
            PricePath=TransitionPath_Case1_nod_lowmem(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, n_a, n_z, pi_z, a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, simoptions,transpathoptions);
        else
            PricePath=TransitionPath_Case1_lowmem(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, n_d, n_a, n_z, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, simoptions,transpathoptions);
        end
    else
        if N_d==0
            PricePath=TransitionPath_Case1_shooting_nod(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, n_a, n_z, pi_z, a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, simoptions,transpathoptions);
        else
            PricePath=TransitionPath_Case1_shooting(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, n_d, n_a, n_z, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, simoptions,transpathoptions);
        end
    end
end

if transpathoptions.GEnewprice==2
    warning('Have not yet implemented transpathoptions.GEnewprice==2 for infinite horizon transition paths (2 is to treat path as a fixed-point problem) ')
end

end