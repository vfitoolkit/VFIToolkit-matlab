function SimPanelValues=SimPanelValues_TransPath_InfHorz(PolicyPath, PricePath, ParamPath, T, AgentDist_initial, n_d, n_a, n_z, pi_z, d_grid,a_grid,z_grid, FnsToEvaluate, Parameters, transpathoptions, simoptions)
% This code will work for all transition paths except those that involve at
% change in the transition matrix pi_z (can handle a change in pi_z, but
% only if it is a 'surprise', not anticipated changes)

% Simulates a panel based on PolicyPath of 'numbersims' agents of length
% 'T' beginning from randomly drawn InitialDist.
% SimPanelValues is a 3-dimensional matrix with first dimension being the
% number of 'variables' to be simulated, second dimension is FHorz, and
% third dimension is the number-of-simulations
%
% InitialDist can be inputted as over the finite time-horizon (j), or
% without a time-horizon in which case it is assumed to be an InitialDist
% for time j=1. (So InitialDist is either n_a-by-n_z-by-T, or n_a-by-n_z)
%
% No z is n_z=0 (z_grid=[], pi_z=[]). iid e shocks are given in simoptions (n_e, e_grid, pi_e), as for
% SimPanelValues_InfHorz. z_grid, pi_z, e_grid and pi_e may vary over the path: pi_z(:,:,t) is the transition from
% t to t+1, pi_e(:,t) the distribution of the e realized IN period t. AgentDist_initial is the period 1 agent
% distribution, [n_a,(n_z),(n_e)], and PolicyPath is [rows,n_a,(n_z),(n_e),T].
% Semi-exogenous states are given in simoptions (n_semiz, semiz_grid, SemiExoStateFn or pi_semiz, l_dsemiz), as for
% SimPanelValues_InfHorz; then AgentDist_initial is [n_a,n_semiz,(n_z),(n_e)], PolicyPath [rows,n_a,n_semiz,(n_z),(n_e),T],
% and FnsToEvaluate inputs are (d,aprime,a,semiz,z,e,...). Each period semiz' is drawn from pi_semiz(semiz,:,d2), at the
% d2 chosen that period (the last simoptions.l_dsemiz decision variables); pi_semiz may vary over the path
% (SemiExoStateFn depending on a ParamPath parameter, or a 4-D pi_semiz), slice t being the transition from t to t+1.

%% Check which simoptions have been declared, set all others to defaults
if ~exist('simoptions','var')
    % If simoptions is not given, just use all the defaults
    simoptions.verbose=0;
    simoptions.simperiods=T;
    simoptions.numbersims=10^3;
    simoptions.lowmemory=0; % setting to 1 slows the simulations, but reduces memory
    % Model setup
    simoptions.gridinterplayer=0;
    simoptions.experienceasset=0;
    simoptions.experienceassetu=0; % cannot actually be used in InfHorz, but need to set for some sub-commands
    simoptions.experienceassetz=0;   % cannot actually be used in InfHorz, but need to set for some sub-commands
    simoptions.experienceassetze=0;  % cannot actually be used in InfHorz, but need to set for some sub-commands
    simoptions.experienceassete=0;   % cannot actually be used in InfHorz, but need to set for some sub-commands
    simoptions.experienceassetsemiz=0; % cannot actually be used in InfHorz, but need to set for some sub-commands
    simoptions.riskyasset=0;         % cannot actually be used in InfHorz, but need to set for some sub-commands
    simoptions.n_semiz=0;
    simoptions.n_e=0;
    % When calling as a subcommand, the following is used internally
    simoptions.alreadygridvals=0;
    simoptions.alreadygridvals_semiexo=0;
    % For internal use only
    simoptions.keepoutputasmatrix=0;
    simoptions.simpanelindexkron=0;
else
    % Check simoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
    end
    if ~isfield(simoptions,'simperiods')
        simoptions.simperiods=T; % This can be made shorter, but not longer
    end
    if ~isfield(simoptions,'numbersims')
        simoptions.numbersims=10^3;
    end
    if ~isfield(simoptions,'lowmemory')
        simoptions.lowmemory=0; % setting to 1 slows the simulations, but reduces memory
    end
    % Model setup
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0;
    end
    if ~isfield(simoptions,'experienceasset')
        simoptions.experienceasset=0;
    end
    simoptions.experienceassetu=0; % cannot actually be used in InfHorz, but need to set for some sub-commands
    simoptions.experienceassetz=0;   % cannot actually be used in InfHorz, but need to set for some sub-commands
    simoptions.experienceassetze=0;  % cannot actually be used in InfHorz, but need to set for some sub-commands
    simoptions.experienceassete=0;   % cannot actually be used in InfHorz, but need to set for some sub-commands
    simoptions.experienceassetsemiz=0; % cannot actually be used in InfHorz, but need to set for some sub-commands
    simoptions.riskyasset=0;         % cannot actually be used in InfHorz, but need to set for some sub-commands
    if ~isfield(simoptions,'n_semiz')
        simoptions.n_semiz=0;
    end
    if ~isfield(simoptions,'n_e')
        simoptions.n_e=0;
    end
    % When calling as a subcommand, the following is used internally
    if ~isfield(simoptions,'alreadygridvals')
        simoptions.alreadygridvals=0;
    end
    if ~isfield(simoptions,'alreadygridvals_semiexo')
        simoptions.alreadygridvals_semiexo=0;
    end
    % For internal use only
    if ~isfield(simoptions,'keepoutputasmatrix')
        simoptions.keepoutputasmatrix=0;
    end
    if ~isfield(simoptions,'simpanelindexkron')
        simoptions.simpanelindexkron=0;
    end
end

if isfield(simoptions,'ExogShockFn') % If using ExogShockFn then figure out the parameter names
    simoptions.ExogShockFnParamNames=getAnonymousFnInputNames(simoptions.ExogShockFn);
end

simoptions.simperiods=gather(simoptions.simperiods);
simoptions.numbersims=gather(simoptions.numbersims); % This is just to deal with weird error that matlab decided simoptions.numbersims was on gpu and so couldn't be an input to rand()

%% Internally PricePath is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'.
[PricePath,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec]=PricePathParamPath_StructToMatrix(PricePath,ParamPath,T);


%%
if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);

N_d=prod(n_d);
N_a=prod(n_a);
% N_semizze set below

%% Simulate Panel Indexes
d_grid=gather(d_grid);
a_grid=gather(a_grid);

N_semiz=prod(simoptions.n_semiz);
N_z=prod(n_z);
N_e=prod(simoptions.n_e);

% Refused rather than simulated: the simulation below moves the a index with the policy alone, so it has no
% experience asset (aprimeFn) in it, and would give a wrong panel
if simoptions.experienceasset>=1
    error('SimPanelValues_TransPath_InfHorz: experienceasset is not yet implemented')
end
if N_semiz>0
    if N_d==0
        error('SimPanelValues_TransPath_InfHorz: semi-exogenous states (simoptions.n_semiz) require a decision variable: the last simoptions.l_dsemiz decision variables determine the semiz transitions')
    end
end

if N_z==0
    if N_e==0
        n_semizze=0;
    else
        n_semizze=simoptions.n_e;
    end
else
    if N_e==0
        n_semizze=n_z;
    else
        n_semizze=[n_z,simoptions.n_e];
    end
end
N_zeonly=max(N_z,1)*max(N_e,1); % (z,e) alone, without semiz
% semiz goes in front of (z,e), semiz fastest
if N_semiz>0
    if N_z==0 && N_e==0
        n_semizze=simoptions.n_semiz;
    else
        n_semizze=[simoptions.n_semiz,n_semizze];
    end
end
N_semizze=prod(n_semizze);

%% Exogenous shocks
% gridpiboth=4: both the grids (for the FnsToEvaluate) and the transition probabilities (for the simulation). The setup
% removes n_e, e_grid and pi_e from the options it is given, so it is given a copy and simoptions keeps them for the
% simulation. When z or e varies over the path the paths are in transpathoptions.pi_z_T, pi_e_T and ze_gridvals_T
% (ze_gridvals is the (z,e) grid, z varies first, which is the order of the kron (z,e) index of the simulation).
[~, pi_z, ~, ~, pi_e, ~, ze_gridvals, transpathoptions, ~]=ExogShockSetup_InfHorz_TPath(n_z,z_grid,pi_z,Parameters,PricePathNames,ParamPath,ParamPathNames,ParamPathSizeVec,T,transpathoptions,simoptions,4);

% The simulation is done as if a finite horizon of T periods: pi_z_T(:,:,t) is the transition from t to t+1, and
% simoptions.pi_e_J(:,t) the distribution of the e realized in period t (as pi_e_J in FHorz)
if N_z>0
    if transpathoptions.zpathtrivial==0
        pi_z_T=gather(transpathoptions.pi_z_T);
    else
        pi_z_T=repmat(gather(pi_z),1,1,T);
    end
else
    pi_z_T=[];
end
if N_e>0
    if transpathoptions.epathtrivial==0
        simoptions.pi_e_J=gather(transpathoptions.pi_e_T);
    else
        simoptions.pi_e_J=repmat(gather(reshape(pi_e,[N_e,1])),1,T);
    end
end

%% Semi-exogenous states
if N_semiz>0
    % semiz_gridvals [N_semiz,l_semiz] (the semiz grid is fixed over the path) and pi_semiz [N_semiz,N_semiz,N_dsemiz];
    % when transpathoptions.semizpathtrivial=0 the path is in transpathoptions.pi_semiz_T [N_semiz,N_semiz,N_dsemiz,T],
    % slice t being the transition from t to t+1
    [semiz_gridvals, pi_semiz, transpathoptions, simoptions]=SemiExogShockSetup_InfHorz_TPath(n_d,d_grid,Parameters,PricePathNames,ParamPath,ParamPathNames,ParamPathSizeVec,T,transpathoptions,simoptions,4);
    if transpathoptions.semizpathtrivial==0
        pi_semiz_T=gather(transpathoptions.pi_semiz_T);
    else
        pi_semiz_T=repmat(gather(pi_semiz),1,1,1,T);
    end
    % The (semiz,z,e) grid for the FnsToEvaluate, semiz fastest (the order of the kron exogenous index of the simulation)
    if N_z==0 && N_e==0
        ze_gridvals=semiz_gridvals;
    else
        ze_gridvals=[repmat(semiz_gridvals,N_zeonly,1),repelem(ze_gridvals,N_semiz,1)];
        if transpathoptions.zepathtrivial==0
            % z and/or e grids vary over the path (the semiz grid cannot): [N_semiz*N_z*N_e,l_semiz+l_z+l_e,T]
            transpathoptions.ze_gridvals_T=[repmat(semiz_gridvals,N_zeonly,1,T),repelem(transpathoptions.ze_gridvals_T,N_semiz,1,1)];
        end
    end
end

%% Simulate the indexes
% Can just pretend this is a standard FHorz model and use that to implement the simulation
simoptions.simpanelindexkron=1; % Keep the output as kron form as will want this later anyway for assigning the values
% (without this, SimPanelIndexes_FHorz returns the a index split across l_a rows and the z index
% split across l_z rows, but everything below reads row 1 as the kron a index and row 2 as the
% kron z index. Those coincide only when l_a==1 and l_z==1.)
if N_semiz==0
    % The initial draws are from AgentDist_initial over (a,z,e), and the kron output puts (z,e) together in row 2
    SimPanelIndexes=SimPanelIndexes_FHorz(gather(AgentDist_initial),gather(PolicyPath),n_d,n_a,n_z,T,pi_z_T, simoptions);
else
    % The FHorz semiz simulator, with the period-t semiz transitions as its age-t ones: each period semiz' is drawn from
    % pi_semiz(semiz,:,d2) at the d2 chosen that period. It sets up pi_semiz_J itself, from pi_semiz, so it is given a copy
    % of simoptions with the path pi_semiz [N_semiz,N_semiz,N_dsemiz,T] in place of SemiExoStateFn (whose parameters are
    % read from ParamPath above, which the FHorz setup knows nothing about), and with the d_grid input.
    % The initial draws are from AgentDist_initial over (a,semiz,z,e), and the kron output puts (semiz,z,e) together in row 2
    simoptions_semiz=simoptions;
    if isfield(simoptions_semiz,'SemiExoStateFn')
        simoptions_semiz=rmfield(simoptions_semiz,'SemiExoStateFn');
    end
    simoptions_semiz.pi_semiz=pi_semiz_T;
    simoptions_semiz.d_grid=d_grid;
    SimPanelIndexes=SimPanelIndexes_FHorz_semiz(gather(AgentDist_initial),gather(PolicyPath),n_d,n_a,n_z,T,pi_z_T, Parameters, simoptions_semiz);
end

%% Check if using _tminus1 and/or _tplus1 variables.
[tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tminus1paramNames,tplus1pricePathkk,use_tplus1price,use_tminus1price,use_tminus1params,use_tminus1AggVars]=inputsFindtplus1tminus1(FnsToEvaluate,struct(),PricePathNames,ParamPathNames,{},transpathoptions);

%% The _tminus1 and _tplus1 values come from transpathoptions.initialvalues
% Checked here rather than at first use, so that a missing initialvalues names itself
% instead of surfacing as 'Unrecognized field name' from inside the loop over t.
if (use_tminus1price==1 || use_tminus1params==1 || use_tminus1AggVars==1) && ~isfield(transpathoptions,'initialvalues')
    error('SimPanelValues_TransPath_InfHorz: a _tminus1 input is used, so transpathoptions.initialvalues must be given (it supplies the period-1 values)')
end
if use_tminus1AggVars==1
    % A _tminus1 aggregate is the population mean last period, which a simulated panel does not give
    % (its sample mean is a different object). Compute the path with EvalFnOnTransPath_AggVars_InfHorz
    % and put the lagged values on the ParamPath under a name of their own.
    error('SimPanelValues_TransPath_InfHorz: FnsToEvaluate cannot use the _tminus1 of an aggregate (%s_tminus1), as a simulated panel does not give the population aggregate. Compute the aggregate path with EvalFnOnTransPath_AggVars_InfHorz and pass its lag in on the ParamPath instead',tminus1AggVarsNames{1})
end

%% Exogenous shock grids: (z,e) are rolled together for the FnsToEvaluate (set up above)
if N_semizze==0
    l_semizze=0;
else
    l_semizze=length(n_semizze);
end


%% Implement new way of handling FnsToEvaluate
% Figure out l_daprime and l_aprime
l_daprime=size(PolicyPath,1)-2*simoptions.gridinterplayer; % gridinterplayer=1 carries an extra L2 index and L2flag
l_aprime=l_daprime-l_d;

% Note: l_semizze
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    FnsToEvalNames=fieldnames(FnsToEvaluate);
    for ff=1:length(FnsToEvalNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(FnsToEvalNames{ff}));
        if length(temp)>(l_daprime+l_a+l_semizze)
            FnsToEvaluateParamNames(ff).Names={temp{l_daprime+l_a+l_semizze+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(FnsToEvalNames{ff});
    end
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end
if isfield(simoptions,'keepoutputasmatrix')
    if simoptions.keepoutputasmatrix==1
        FnsToEvaluateStruct=0;
    end
end

nFnsToEvalute=length(FnsToEvaluate);


%% Precompute the gridvals vectors.
if N_semizze==0
    PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,1,T]);
else
    PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_semizze,T]);
end

% Note that dPolicy and aprimePolicy will depend on age
if N_d==0
    if N_semizze==0
        daprimePolicy_gridvals=zeros(N_a,l_aprime,T);
    else
        daprimePolicy_gridvals=zeros(N_a*N_semizze,l_aprime,T);
    end
else
    if N_semizze==0
        daprimePolicy_gridvals=zeros(N_a,l_d+l_aprime,T);
    else
        daprimePolicy_gridvals=zeros(N_a*N_semizze,l_d+l_aprime,T);
    end
end

for tt=1:T
    if N_d==0
        [~,aprimePolicy_gridvals_t]=CreateGridvals_Policy(PolicyPath(:,:,:,tt),n_d,n_a,n_a,n_semizze,[],a_grid,simoptions,1, 1); % handles gridinterplayer=1 in here
        daprimePolicy_gridvals(:,:,tt)=aprimePolicy_gridvals_t;
    else
        [dPolicy_gridvals_j,aprimePolicy_gridvals_t]=CreateGridvals_Policy(PolicyPath(:,:,:,tt),n_d,n_a,n_a,n_semizze,d_grid,a_grid,simoptions,1, 1); % handles gridinterplayer=1 in here
        daprimePolicy_gridvals(:,:,tt)=[dPolicy_gridvals_j, aprimePolicy_gridvals_t];
    end
end

a_gridvals=CreateGridvals(n_a,a_grid,1); % 1 at end indicates output as matrices.


%% Now switch everything to gpu so can use arrayfun() to evaluates all the FnsToEvaluate
daprimePolicy_gridvals=gpuArray(daprimePolicy_gridvals);
SimPanelIndexes=gpuArray(SimPanelIndexes);

SimPanelValues=nan(length(FnsToEvaluate), T, simoptions.numbersims,'gpuArray'); % needs to be NaN to permit that some people might be 'born' later than age j=1
% Note, having the whole T at this stage makes assigning the values based on the indexes vastly faster


%% Create PanelValues from PanelIndexes
if N_semizze>0
    for tt=1:T
        SimPanelIndexes_tt=SimPanelIndexes(:,tt,:);

        relevantindices=(~isnan(SimPanelIndexes_tt(1,1,:))); % Note, is just across the ii dimension
        sumrelevantindices=sum(relevantindices);

        % Update Parameters based on t
        % The _tminus1 values first, while Parameters still holds the previous period's prices and parameters
        % (setting them after period tt is written in would make every _tminus1 equal to period tt)
        if use_tminus1price==1
            for pp=1:length(tminus1priceNames)
                if tt>1
                    Parameters.([tminus1priceNames{pp},'_tminus1'])=Parameters.(tminus1priceNames{pp});
                else
                    Parameters.([tminus1priceNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1priceNames{pp});
                end
            end
        end
        if use_tminus1params==1
            for pp=1:length(tminus1paramNames)
                if tt>1
                    Parameters.([tminus1paramNames{pp},'_tminus1'])=Parameters.(tminus1paramNames{pp});
                else
                    Parameters.([tminus1paramNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1paramNames{pp});
                end
            end
        end

        for kk=1:length(PricePathNames)
            Parameters.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
        end
        for kk=1:length(ParamPathNames)
            Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
        end
        if use_tplus1price==1
            for pp=1:length(tplus1priceNames)
                kk=tplus1pricePathkk(pp);
                % Period T is the final stationary eqm, so the price after it is the same as at T
                Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePath(min(tt+1,T),PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
        end

        if sumrelevantindices>0 % Does the simulation even contain anyone in period tt?
            currentPanelIndexes_tt=SimPanelIndexes_tt(:,1,relevantindices);
            currentPanelValues_tt=zeros(sumrelevantindices,nFnsToEvalute); % transpose will be taken before storing

            az_ind=squeeze(currentPanelIndexes_tt(1,1,:)+N_a*(currentPanelIndexes_tt(2,1,:)-1));
            % a_ind=currentPanelIndexes_tt(1,1,:);
            % z_ind=currentPanelIndexes_tt(2,1,:); % this is semiz,z,e all together
            % j_ind=currentPanelIndexes_tt(3,1,:);

            a_val=a_gridvals(currentPanelIndexes_tt(1,1,:),:); % a_grid does depend on age
            if transpathoptions.zepathtrivial==0
                z_val=transpathoptions.ze_gridvals_T(currentPanelIndexes_tt(2,1,:),:,tt);
            else
                z_val=ze_gridvals(currentPanelIndexes_tt(2,1,:),:);
            end

            for vv=1:nFnsToEvalute
                if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'FnsToEvaluateParamNames={}'
                    ParamCell={};
                else
                    ValuesFnParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,tt);
                    ParamCell=num2cell(ValuesFnParamsVec);
                end
                daprime_val=daprimePolicy_gridvals(az_ind,:,tt);
                currentPanelValues_tt(:,vv)=EvalFnOnSimPanelIndex(FnsToEvaluate{vv},ParamCell,daprime_val,a_val,z_val,l_daprime,l_a,l_semizze);
            end
            SimPanelValues(:,tt,relevantindices)=reshape(currentPanelValues_tt',[nFnsToEvalute,1,sumrelevantindices]);

        end
    end
else % N_semizze==0
    for tt=1:T
        SimPanelIndexes_tt=SimPanelIndexes(:,tt,:);

        relevantindices=(~isnan(SimPanelIndexes_tt(1,1,:))); % Note, is just across the ii dimension
        sumrelevantindices=sum(relevantindices);

        % Update Parameters based on t
        % The _tminus1 values first, while Parameters still holds the previous period's prices and parameters
        % (setting them after period tt is written in would make every _tminus1 equal to period tt)
        if use_tminus1price==1
            for pp=1:length(tminus1priceNames)
                if tt>1
                    Parameters.([tminus1priceNames{pp},'_tminus1'])=Parameters.(tminus1priceNames{pp});
                else
                    Parameters.([tminus1priceNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1priceNames{pp});
                end
            end
        end
        if use_tminus1params==1
            for pp=1:length(tminus1paramNames)
                if tt>1
                    Parameters.([tminus1paramNames{pp},'_tminus1'])=Parameters.(tminus1paramNames{pp});
                else
                    Parameters.([tminus1paramNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1paramNames{pp});
                end
            end
        end

        for kk=1:length(PricePathNames)
            Parameters.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
        end
        for kk=1:length(ParamPathNames)
            Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
        end
        if use_tplus1price==1
            for pp=1:length(tplus1priceNames)
                kk=tplus1pricePathkk(pp);
                % Period T is the final stationary eqm, so the price after it is the same as at T
                Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePath(min(tt+1,T),PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
        end

        if sumrelevantindices>0 % Does the simulation even contain anyone in period tt?
            currentPanelIndexes_tt=SimPanelIndexes_tt(:,1,relevantindices);
            currentPanelValues_tt=zeros(sumrelevantindices,nFnsToEvalute); % transpose will be taken before storing

            a_ind=squeeze(currentPanelIndexes_tt(1,1,:));
            % j_ind=currentPanelIndexes_tt(2,1,:);

            a_val=a_gridvals(a_ind,:); % a_grid does depend on age

            for vv=1:nFnsToEvalute
                if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'FnsToEvaluateParamNames={}'
                    ParamCell={};
                else
                    ValuesFnParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,tt);
                    ParamCell=num2cell(ValuesFnParamsVec);
                end
                daprime_val=daprimePolicy_gridvals(a_ind,:,tt);
                currentPanelValues_tt(:,vv)=EvalFnOnSimPanelIndex(FnsToEvaluate{vv},ParamCell,daprime_val,a_val,[],l_daprime,l_a,0);
            end
            SimPanelValues(:,tt,relevantindices)=reshape(currentPanelValues_tt',[nFnsToEvalute,1,sumrelevantindices]);

        end
    end
end



%% Implement new way of handling FnsToEvaluate: convert results
if FnsToEvaluateStruct==1
    % Change the output into a structure
    SimPanelValues2=SimPanelValues;
    clear SimPanelValues
    SimPanelValues=struct();
    for ff=1:length(FnsToEvalNames)
        SimPanelValues.(FnsToEvalNames{ff})=shiftdim(SimPanelValues2(ff,:,:),1);
    end
end




end



