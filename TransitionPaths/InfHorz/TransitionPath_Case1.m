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
vfoptions.parallel=2; % GPU, has to be or transpath will already have thrown an error
if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
    vfoptions.verbose=0;
    vfoptions.lowmemory=0;
    vfoptions.polindorval=1;
    vfoptions.policy_forceintegertype=0;
    % Model setup:
    vfoptions.exoticpreferences='None';
    vfoptions.experienceasset=0;
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
    if ~isfield(vfoptions,'polindorval')
        vfoptions.polindorval=1;
    end
    if ~isfield(vfoptions,'policy_forceintegertype')
        vfoptions.policy_forceintegertype=0;
    end
    % Model setup:
    if ~isfield(vfoptions,'exoticpreferences')
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
    if ~isfield(vfoptions,'experienceasset')
            vfoptions.experienceasset=0;
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
            error('When using vfoptions.gridinterplayer=1 you must set vfoptions.gridinterplayer')
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
    simoptions.gridinterplayer=0;
else
    %Check vfoptions for missing fields, if there are some fill them with
    %the defaults
    if ~isfield(simoptions,'tolerance')
        simoptions.tolerance=10^(-9);
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
    end
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0;
    elseif simoptions.gridinterplayer==1
        if ~isfield(simoptions,'ngridinterp')
            error('When using simoptions.gridinterplayer=1 you must set simoptions.gridinterplayer')
        end
    end
end

%% Check the sizes of some of the inputs
N_d=prod(n_d);
% N_a=prod(n_a);
N_z=prod(n_z);

if N_d>0
    if any(size(d_grid)~=[sum(n_d), 1])
        fprintf('d_grid is of size: %i by % i, while sum(n_d) is %i \n',size(d_grid,1),size(d_grid,2),sum(n_d))
        dbstack
        error('d_grid is not the correct shape (should be of size sum(n_d)-by-1) \n')
    end
end
if any(size(a_grid)~=[sum(n_a), 1])
    fprintf('a_grid is of size: %i by % i, while sum(n_a) is %i \n',size(a_grid,1),size(a_grid,2),sum(n_a))
    dbstack
    error('a_grid is not the correct shape (should be of size sum(n_a)-by-1) \n')
% check z_grid below when converting to z_gridvals
elseif any(size(pi_z)~=[N_z, N_z])
    fprintf('pi is of size: %i by % i, while N_z is %i \n',size(pi_z,1),size(pi_z,2),N_z)
    dbstack
    error('pi is not of size N_z-by-N_z \n')
end
if length(PricePathNames)~=length(fieldnames(GeneralEqmEqns))
    fprintf('PricePath has %i prices and GeneralEqmEqns is % i eqns \n',length(PricePathNames), length(fieldnames(GeneralEqmEqns)))
    dbstack
    error('Initial PricePath contains less variables than GeneralEqmEqns (structure) \n')
end


%%
% If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
pi_z=gpuArray(pi_z);
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
z_grid=gpuArray(z_grid);
PricePathOld=gpuArray(PricePathOld);

%% Switch to z_gridvals
l_z=length(n_z);
if all(size(z_grid)==[sum(n_z),1])
    z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
elseif all(size(z_grid)==[prod(n_z),l_z])
    z_gridvals=z_grid;
else
    fprintf('       z_grid is of size: %i by % i, while N_z is %i \n',size(z_grid,1),size(z_grid,2),N_z)
    error('z_grid is not the correct shape (should be of size N_z-by-1) \n')
end


%% Implement new way of handling ReturnFn inputs
ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,0,vfoptions,Parameters);

GEeqnNames=fieldnames(GeneralEqmEqns);

%% If using a shooting algorithm, set that up
% Set up GEnewprice==3 (if relevant)
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
        error('transpathoptions.GEnewprice3.howtoupdate does not fit with GeneralEqmEqns (different number of conditions/prices) \n')
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

%% If there is entry and exit, then send to relevant command
if isfield(simoptions,'agententryandexit')==1 % isfield(transpathoptions,'agententryandexit')==1
    error('Have not yet implemented transition path for models with entry/exit \n')
end

%%
if transpathoptions.GEnewprice~=2
    if vfoptions.experienceasset==0
        if N_d==0
            PricePath=TransitionPath_InfHorz_shooting_nod(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, n_a, n_z, pi_z, a_grid,z_gridvals, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, GEeqnNames, vfoptions, simoptions,transpathoptions);
        else
            PricePath=TransitionPath_InfHorz_shooting(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, n_d, n_a, n_z, pi_z, d_grid,a_grid,z_gridvals, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, GEeqnNames, vfoptions, simoptions,transpathoptions);
        end
    elseif vfoptions.experienceasset==1
        % Split decision variables into the standard ones and the one relevant to the experience asset
        if isscalar(n_d)
            n_d1=0;
        else
            n_d1=n_d(1:end-1);
        end
        n_d2=n_d(end); % n_d2 is the decision variable that influences next period vale of the experience asset
        d1_grid=d_grid(1:sum(n_d1));
        d2_grid=d_grid(sum(n_d1)+1:end);
        % Split endogenous assets into the standard ones and the experience asset
        if isscalar(n_a)
            n_a1=0;
        else
            n_a1=n_a(1:end-1);
        end
        n_a2=n_a(end); % n_a2 is the experience asset
        a1_grid=a_grid(1:sum(n_a1));
        a2_grid=a_grid(sum(n_a1)+1:end);

        if isfield(vfoptions,'aprimeFn')
            aprimeFn=vfoptions.aprimeFn;
        else
            error('To use an experience asset you must define vfoptions.aprimeFn')
        end

        % aprimeFnParamNames in same fashion
        l_d2=length(n_d2);
        l_a2=length(n_a2);
        temp=getAnonymousFnInputNames(aprimeFn);
        if length(temp)>(l_d2+l_a2)
            aprimeFnParamNames={temp{l_d2+l_a2+1:end}}; % the first inputs will always be (d2,a2)
        else
            aprimeFnParamNames={};
        end

        N_a1=prod(n_a1);
        if N_a1==0
            error('Have not yet implemented TPath for InfHorz with experienceasset and no other (standard) asset, contact me if you want/need this')
        else
            PricePath=TransitionPath_InfHorz_shooting_ExpAsset(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, n_d1, n_d2, n_a1, n_a2, n_z, pi_z, d1_grid, d2_grid, a1_grid, a2_grid,z_gridvals, ReturnFn, vfoptions.aprimeFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, GEeqnNames, vfoptions, simoptions,transpathoptions);
        end
    end
end

if transpathoptions.GEnewprice==2
    warning('Have not yet implemented transpathoptions.GEnewprice==2 for infinite horizon transition paths (2 is to treat path as a fixed-point problem) ')
end

end
