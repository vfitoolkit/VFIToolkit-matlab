function varargout=TransitionPath_InfHorz(PricePath0, ParamPath, T, V_final, AgentDist_initial, n_d,n_a,n_z, d_grid,a_grid,z_grid, pi_z, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, transpathoptions, simoptions, vfoptions, EntryExitParamNames)
% This code will work for all transition paths except those that involve at
% change in the transition matrix pi_z (can handle a change in pi_z, but
% only if it is a 'surprise', not anticipated changes)
%
% PricePath0 is a structure with fields names being the Prices and each field containing a T-by-1 path. It is the initial guess for the PricePath.
% ParamPath is a structure with fields names being the parameter names of those parameters which change over the path and each field containing a T-by-1 path.
%
% transpathoptions is not a required input.

% Remark to self: No real need for T as input, as this is anyway the length of PricePathOld

if all(size(d_grid)==[prod(n_z),prod(n_z)])
    error('Check input order: pi_z comes after z_grid') % Keep this error message until end of 2007, can remove after that
end

%% Check which transpathoptions have been used, set all others to defaults
if exist('transpathoptions','var')==0
    disp('No transpathoptions given, using defaults')
    % If transpathoptions is not given, just use all the defaults
    transpathoptions.tolerance=10^(-5);
    transpathoptions.updateaccuracycutoff=10^(-9); % If the suggested update is less than this then don't bother; 10^(-9) is decent odds to be numerical error anyway (currently only works for transpathoptions.GEnewprice=3)
    transpathoptions.parallel=1+(gpuDeviceCount>0);
    transpathoptions.GEnewprice=1; % 1 is shooting algorithm, 0 is that the GE should evaluate to zero and the 'new' is the old plus the "non-zero" (for each time period seperately);
                                   % 2 is to do optimization routine with 'distance between old and new path', 3 is just same as 0, but easier to set up
    transpathoptions.oldpathweight=0.9; % default =0.9
    transpathoptions.weightscheme=1; % default =1
    transpathoptions.Ttheta=1;
    transpathoptions.maxiter=1000;
    transpathoptions.verbose=0;
    transpathoptions.graphpricepath=0; % 1: creates a graph of the 'current' price path which updates each iteration.
    transpathoptions.graphaggvarspath=0; % 1: creates a graph of the 'current' aggregate variables which updates each iteration.
    transpathoptions.graphGEcondns=0;  % 1: creates a graph of the 'current' general eqm conditions which updates each iteration.
    transpathoptions.historyofpricepath=0;
    transpathoptions.stockvars={}; % 'stockvars' are prices where you write '_tminus1' and it should cumulate (to there will be a general eqm eqn that relates the _tminus1 to the t for a price in PricePath)
    transpathoptions.weightsforpath=ones(T,length(GeneralEqmEqns)); % Won't actually be used under the defaults, but am still setting it.
    transpathoptions.tanimprovement=1;
else
    % Check transpathoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(transpathoptions,'tolerance')
        transpathoptions.tolerance=10^(-5);
    end
    if ~isfield(transpathoptions,'updateaccuracycutoff')
        transpathoptions.updateaccuracycutoff=10^(-9);
    end
    if ~isfield(transpathoptions,'parallel')
        transpathoptions.parallel=1+(gpuDeviceCount>0);
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

if transpathoptions.graphGEcondns==1
    if transpathoptions.GEnewprice~=3
        error('Can only use transpathoptions.graphGEcondns=1 when using transpathoptions.GEnewprice=3')
    end
end

%% Check which vfoptions have been used, set all others to defaults
vfoptions.parallel=2; % GPU, has to be or transpath will already have thrown an error
if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
    vfoptions.verbose=0;
    vfoptions.lowmemory=0;
    % Model setup:
    vfoptions.exoticpreferences='None';
    vfoptions.experienceasset=0;
    % Exogenous shocks
    vfoptions.n_semiz=0;
    vfoptions.n_e=0;
    % Algorithm to use:
    vfoptions.solnmethod='purediscretization'; % Currently this does nothing
    vfoptions.divideandconquer=0;
    vfoptions.gridinterplayer=0;
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(vfoptions,'lowmemory')
        vfoptions.lowmemory=0;
    end
    if ~isfield(vfoptions,'verbose')
        vfoptions.verbose=0;
    end
    % Model setup:
    if ~isfield(vfoptions,'exoticpreferences')
        vfoptions.exoticpreferences='None';
    end
    if strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
        if ~isfield(vfoptions,'quasi_hyperbolic')
            vfoptions.quasi_hyperbolic='Naive'; % This is the default, alternative is 'Sophisticated'.
        elseif ~strcmp(vfoptions.quasi_hyperbolic,'Naive') && ~strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
            error('When using Quasi-Hyperbolic discounting vfoptions.quasi_hyperbolic must be either Naive or Sophisticated \n')
        end
    end
    if ~isfield(vfoptions,'experienceasset')
            vfoptions.experienceasset=0;
    end
    % Exogenous shocks
    if ~isfield(vfoptions,'n_semiz')
        vfoptions.n_semiz=0;
    end
    if ~isfield(vfoptions,'n_e')
        vfoptions.n_e=0;
    end
    % Algorithm to use:
    if ~isfield(vfoptions,'solnmethod')
        vfoptions.solnmethod='purediscretization'; % Currently this does nothing
    end
    if ~isfield(vfoptions,'divideandconquer')
        vfoptions.divideandconquer=0;
    end
    if ~isfield(vfoptions,'gridinterplayer')
        vfoptions.gridinterplayer=0;
    elseif vfoptions.gridinterplayer==1
        if ~isfield(vfoptions,'ngridinterp')
            error('When using vfoptions.gridinterplayer=1 you must set vfoptions.ngridinterp')
        end
    end
end

if vfoptions.divideandconquer==1
    if ~isfield(vfoptions,'level1n')
        if isscalar(n_a)
            vfoptions.level1n=max(ceil(n_a(1)/50),5); % minimum of 5
            if n_a(1)<5
                error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
            end
        elseif length(n_a)==2
            vfoptions.level1n=[max(ceil(n_a(1)/50),5),n_a(2)]; % default is DC2B, min of 5 points in level1 for a1
            if n_a(1)<5
                error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
            end
        end
        if vfoptions.verbose==1
            fprintf('Suggestion: When using vfoptions.divideandconquer it will be faster or slower if you set different values of vfoptions.level1n (for smaller models 7 or 9 is good, but for larger models something 15 or 21 can be better) \n')
        end
    end
end

%% Check which simoptions have been used, set all others to defaults
simoptions.parallel=2; % GPU, has to be or transpath will already have thrown an error
if exist('simoptions','var')==0
    simoptions.verbose=0;
    simoptions.tolerance=10^(-9);
    % Model setup
    simoptions.experienceasset=0;
    % Algorithm to use
    simoptions.gridinterplayer=0;
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(simoptions,'tolerance')
        simoptions.tolerance=10^(-9);
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
    end
    % Model setup
    if ~isfield(simoptions,'experienceasset')
        simoptions.experienceasset=0;
    end
    % Algorithm to use
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0;
    elseif simoptions.gridinterplayer==1
        if ~isfield(simoptions,'ngridinterp')
            error('When using simoptions.gridinterplayer=1 you must set simoptions.ngridinterp')
        end
    end
end

%% Check the sizes of some of the inputs
N_d=prod(n_d);
% N_a=prod(n_a);
N_z=prod(n_z);

if N_d>0
    if any(size(d_grid)~=[sum(n_d), 1]) && any(size(d_grid)~=[prod(n_d), length(n_d)]) % stacked-column-grid or joint-grid
        fprintf('d_grid is of size: %i by % i, while sum(n_d) is %i \n',size(d_grid,1),size(d_grid,2),sum(n_d))
        error('d_grid is not the correct shape [should be stacked-column of size sum(n_d)-by-1), or a joint-grid of size prod(n_z)-by-length(n_z) ] \n')
    end
end
if any(size(a_grid)~=[sum(n_a), 1])
    fprintf('a_grid is of size: %i by % i, while sum(n_a) is %i \n',size(a_grid,1),size(a_grid,2),sum(n_a))
    error('a_grid is not the correct shape (should be of size sum(n_a)-by-1) \n')
% check z_grid below when converting to z_gridvals
elseif any(size(pi_z)~=[N_z, N_z])
    fprintf('pi is of size: %i by % i, while N_z is %i \n',size(pi_z,1),size(pi_z,2),N_z)
    error('pi is not of size N_z-by-N_z \n')
end
if length(fieldnames(PricePath0))~=length(fieldnames(GeneralEqmEqns))
    fprintf('PricePath has %i prices and GeneralEqmEqns is % i eqns \n',length(fieldnames(PricePath0)), length(fieldnames(GeneralEqmEqns)))
    error('Initial PricePath contains less variables than GeneralEqmEqns (structure) \n')
end

%% Internally PricePath is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'.
[PricePath0,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec]=PricePathParamPath_StructToMatrix(PricePath0,ParamPath,T);

PricePathStruct=struct();

%%
% If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
pi_z=gpuArray(pi_z);
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
z_grid=gpuArray(z_grid);
PricePath0=gpuArray(PricePath0);
ParamPath=gpuArray(ParamPath);
V_final=gpuArray(V_final);
% Tan improvement means we want agent dist on cpu
AgentDist_initial=gather(AgentDist_initial);


%% Check the sizes of some of the inputs
N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);
if N_e>0
    error('Have not yet implemented i.i.d., e, shocks')
end

if N_d==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_aprime=l_a;
if vfoptions.experienceasset==1
    l_aprime=l_aprime-1;
end
if N_z==0
    l_z=0;
else
    l_z=length(n_z);
end
if N_e==0
    l_e=0;
else
    l_e=length(vfoptions.n_e);
end

%% Implement new way of handling ReturnFn inputs
ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,0,vfoptions,Parameters);

%% Set up exogenous shock processes
[z_gridvals, pi_z, pi_z_sparse, e_gridvals, pi_e, pi_e_sparse, ze_gridvals, transpathoptions, simoptions]=ExogShockSetup_InfHorz_TPath(n_z,z_grid,pi_z,Parameters,PricePathNames,ParamPathNames,transpathoptions,simoptions,4);
% Convert z and e to joint-grids and transition matrix
% output: z_gridvals, pi_z, e_gridvals, pi_e, transpathoptions,vfoptions,simoptions

% Sets up
% transpathoptions.zpathtrivial=1; % z_gridvals and pi_z are not varying over the path
%                              =0; % they vary over path, so z_gridvals_T and pi_z_T
% transpathoptions.epathtrivial=1; % e_gridvals and pi_e are not varying over the path
%                              =0; % they vary over path, so e_gridvals_T and pi_e_T
% and
% transpathoptions.gridsinGE=1; % grids depend on a GE parameter and so need to be recomputed every iteration
%                           =0; % grids are exogenous
%
% transpathoptions.zepathtrivial=0 when either of zpathtrival and epathtrivial both are zero

%% If using any non-standard endogenous states, setup for those
[vfoptions,simoptions]=SetupNonStandardEndoStates_InfHorz_TPath(n_d,n_a,d_grid,a_grid,vfoptions,simoptions);

%% Setup for V_final
% Note: I keep Policy as having a first dimension (even if it is just 1)
if N_e==0
    if N_z==0
        V_final=reshape(V_final,[N_a,1]);
    else
        V_final=reshape(V_final,[N_a,N_z]);
    end
else
    if N_z==0
        V_final=reshape(V_final,[N_a,N_e]);
    else
        V_final=reshape(V_final,[N_a,N_z,N_e]);
    end
end

%% Setup for AgentDist_initial
if N_e==0  % no z, no e
    if N_z==0
        AgentDist_initial=reshape(AgentDist_initial,[N_a,1]);
    else % z, no e
        AgentDist_initial=reshape(AgentDist_initial,[N_a*N_z,1]);
    end
else
    if N_z==0 % no z, e
        AgentDist_initial=reshape(AgentDist_initial,[N_a*N_e,1]);
    else % z & e
        AgentDist_initial=reshape(AgentDist_initial,[N_a*N_z*N_e,1]);
    end
end


%% Change to FnsToEvaluate as cell so that it is not being recomputed all the time
l_daprime=l_d+l_a;
if vfoptions.experienceasset==1
    l_daprime=l_daprime-1;
end

AggVarNames=fieldnames(FnsToEvaluate);
FnsToEvaluateCell=cell(1,length(AggVarNames));
for ff=1:length(AggVarNames)
    temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
    if length(temp)>(l_daprime+l_a+l_z+l_e)
        FnsToEvaluateParamNames(ff).Names={temp{l_daprime+l_a+l_z+l_e+1:end}}; % the first inputs will always be (d,aprime,a,z)
    else
        FnsToEvaluateParamNames(ff).Names={};
    end
    FnsToEvaluateCell{ff}=FnsToEvaluate.(AggVarNames{ff});
end
% Change FnsToEvaluate out of structure form, but want to still create AggVars as a structure
simoptions.outputasstructure=1;


%% Set up Gridvals (used by FnsToEvaluate, among others)
a_gridvals=CreateGridvals(n_a,a_grid,1); % a_gridvals is [N_a,l_a]

if N_d>0
    % Gridvals: switch to joint-grids
    if all(size(d_grid)==[sum(n_d),1]) % if stacked-column grid
        d_gridvals=CreateGridvals(n_d,gpuArray(d_grid),1);
    elseif all(size(d_grid)==[prod(n_d),length(n_d)]) % if joint-grid
        d_gridvals=gpuArray(d_grid);
    end
else
    d_gridvals=[];
end

if vfoptions.gridinterplayer==0
    aprime_gridvals=a_gridvals;
elseif vfoptions.gridinterplayer==1
    % use fine grid for aprime_gridvals
    if isscalar(n_a)
        n_aprime=n_a+(n_a-1)*vfoptions.ngridinterp;
        aprime_grid=interp1(gpuArray(1:1:N_a)',a_grid,gpuArray(linspace(1,N_a,n_aprime))');
        aprime_gridvals=CreateGridvals(n_aprime,aprime_grid,1);
    else
        a1_grid=a_grid(1:n_a(1));
        n_a1prime=n_a(1)+(n_a(1)-1)*vfoptions.ngridinterp;
        n_aprime=[n_a1prime,n_a(2:end)];
        a1prime_grid=interp1(gpuArray(1:1:n_a(1))',a1_grid,gpuArray(linspace(1,n_a(1),n_a1prime))');
        aprime_grid=[a1prime_grid; a_grid(n_a(1)+1:end)];
        aprime_gridvals=CreateGridvals(n_aprime,aprime_grid,1);
    end
end

%% GE eqns, switch from structure to cell setup
GEeqnNames=fieldnames(GeneralEqmEqns);
nGeneralEqmEqns=length(GEeqnNames);

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


%% If using a shooting algorithm, set that up
transpathoptions=setupGEnewprice3_shooting(transpathoptions,GeneralEqmEqns,PricePathNames);


%% Check if using _tminus1 and/or _tplus1 variables.
[tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tminus1paramNames,tplus1pricePathkk,use_tplus1price,use_tminus1price,use_tminus1params,use_tminus1AggVars]=inputsFindtplus1tminus1(FnsToEvaluate,GeneralEqmEqns,PricePathNames,ParamPathNames,{},transpathoptions);

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

%%
if transpathoptions.verbose>=1
    transpathoptions
end

if transpathoptions.verbose==2
    DiscountFactorParamNames
    ReturnFnParamNames
    ParamPathNames
    PricePathNames
end


%% If there is entry and exit, then send to relevant command
if isfield(simoptions,'agententryandexit')==1 % isfield(transpathoptions,'agententryandexit')==1
    error('Have not yet implemented transition path for models with entry/exit \n')
end

%%
if transpathoptions.GEnewprice~=2
    [PricePath,GEcondnPathmatrix]=TransitionPath_InfHorz_shooting(PricePath0, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, n_d,n_a,n_z,vfoptions.n_e, N_d,N_a,N_z,N_e, l_d,l_aprime,l_a,l_z,l_e, d_gridvals,aprime_gridvals,a_gridvals,a_grid,z_gridvals,e_gridvals,ze_gridvals,pi_z,pi_z_sparse,pi_e, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GEeqnNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, stockvarsNames, stockvarsInPricePathNames, vfoptions, simoptions,transpathoptions);

    % Switch to structure for output
    for pp=1:length(PricePathNames)
        PricePathStruct.(PricePathNames{pp})=PricePath(:,pp)';
    end
    for gg=1:length(GEeqnNames)
        GEcondnPath.(GEeqnNames{gg})=GEcondnPathmatrix(:,gg)';
    end

end

if transpathoptions.GEnewprice==2
    warning('Have not yet implemented transpathoptions.GEnewprice==2 for infinite horizon transition paths (2 is to treat path as a fixed-point problem) ')


    % Switch to structure for output
    for pp=1:length(PricePathNames)
        PricePathStruct.(PricePathNames{pp})=PricePath(:,pp)';
    end

end

if nargout==1
    varargout={PricePathStruct};
elseif nargout==2
    varargout={PricePathStruct,GEcondnPath};
end


end
