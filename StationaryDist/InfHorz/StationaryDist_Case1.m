function StationaryDist=StationaryDist_Case1(Policy,n_d,n_a,n_z,pi_z,simoptions,Parameters,EntryExitParamNames)
% Parameters and EntryExitParamNames are optional inputs, only needed/used when using entry-exit of agents (when simoptions.agententryandexit=1).
%
% StationaryDist will typically just be matrix containing pdf.
% When using entry-exit it is a structure: StationaryDist.pdf contains pdf,
% StationaryDist.mass contains the mass.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if exist('simoptions','var')==0
    simoptions.verbose=0;
    simoptions.parallel=1+(gpuDeviceCount>0);
    simoptions.maxit=10^6; % In my experience, after a simulation, if you need more than 10^6 iterations to reach the steady-state it is because something has gone wrong
    simoptions.tolerance=10^(-6); % I originally had this at 10^(-9) but this seems to have been overly strict as very hard to acheive and not needed for model accuracy, now set to 10^(-6) [note that this is the max of all error across the agent dist, the L-Infinity norm]
    simoptions.multiiter=50; % How many iteration steps before check tolerance
    simoptions.iterate=1;
    simoptions.tanimprovement=1; % Use Tan (2020) improvement to iteration (is hardcoded into everything but the most basic setting)
    simoptions.policyalreadykron=0; % Can specify that policy is already in kron form, used to speed up general eqm and transition path computations.
    simoptions.outputkron=0;
    % Options relating to simulation method
    simoptions.ncores=1;
    simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
    simoptions.simperiods=10^6; % I tried a few different things and this seems reasonable.
    simoptions.burnin=10^3; % Increasing this to 10^4 did not seem to impact the actual simulation agent distributions
    % Options relating to eigenvector method
    simoptions.eigenvector=0; % I implemented an eigenvector based approach. It is fast but not robust.
    % Alternative setups
    simoptions.agententryandexit=0;
    % simoptions.endogenousexit=0; % Not needed when simoptions.agententryandexit=0;
    % simoptions.SemiEndogShockFn % Undeclared by default (cannot be used with entry and exit)
    simoptions.experienceasset=0;
    % When calling as a subcommand, the following is used internally
    simoptions.alreadygridvals=0;
else
    %Check simoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
    end
    if ~isfield(simoptions,'parallel')
        simoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(simoptions, 'maxit')
        simoptions.maxit=10^6;
    end
    if ~isfield(simoptions, 'tolerance')
        simoptions.tolerance=10^(-6); % I originally had this at 10^(-9) but this seems to have been overly strict as very hard to acheive and not needed for model accuracy, now set to 10^(-6) [note that this is the max of all error across the agent dist, the L-Infinity norm]
    end
    if ~isfield(simoptions, 'multiiter')
        simoptions.multiiter=50; % How many iteration steps before check tolerance
    end
    if ~isfield(simoptions, 'iterate')
        simoptions.iterate=1;
    end
    if ~isfield(simoptions, 'tanimprovement')
        simoptions.tanimprovement=1; % Use Tan (2020) improvement to iteration (is hardcoded into everything but the most basic setting)
    end
    if ~isfield(simoptions, 'policyalreadykron')
        simoptions.policyalreadykron=0;
    end
    if ~isfield(simoptions, 'outputkron')
        simoptions.outputkron=0;
    end
    % Options relating to simulation method
    if ~isfield(simoptions,'ncores')
        simoptions.ncores=1;
    end
    if ~isfield(simoptions, 'seedpoint')
        simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
    end
    if ~isfield(simoptions, 'simperiods')
        simoptions.simperiods=10^6;  % I tried a few different things and this seems reasonable.
    end
    if ~isfield(simoptions, 'burnin')
        simoptions.burnin=10^3; % Increasing this to 10^4 did not seem to impact the actual simulation agent distributions
    end
    % Options relating to eigenvector method
    if ~isfield(simoptions,'eigenvector')
        simoptions.eigenvector=0; % I implemented an eigenvector based approach. It is fast but not robust.
    end
    % Alternative setups
    if ~isfield(simoptions, 'agententryandexit')
        simoptions.agententryandexit=0;
    else
        if simoptions.agententryandexit==1
            if isfield(simoptions, 'endogenousexit')==0
                simoptions.endogenousexit=0;
            end
        end
    end
    % simoptions.SemiEndogShockFn % Undeclared by default (cannot be used with entry and exit)
    if ~isfield(simoptions,'experienceasset')
        simoptions.experienceasset=0;
    end
    % When calling as a subcommand, the following is used internally
    if ~isfield(simoptions,'alreadygridvals')
        simoptions.alreadygridvals=0;
    end
end

%%
if simoptions.alreadygridvals==0
    if isfield(simoptions,'ExogShockFn')
        [~, pi_z, simoptions]=ExogShockSetup(n_z,[],pi_z,Parameters,simoptions,2);
    end
end

if simoptions.parallel==1 || simoptions.parallel==3 || simoptions.eigenvector==1 % Eigenvector only works for cpu
    Policy=gather(Policy);
    pi_z=gather(pi_z);
else
    Policy=gpuArray(Policy); % Note that in this instance it is very likely that the policy is anyway already on the gpu
    pi_z=gpuArray(pi_z);
end

if simoptions.policyalreadykron==0
    if simoptions.experienceasset==0
        PolicyKron=KronPolicyIndexes_Case1(Policy, n_d, n_a, n_z);
    end
    % simoptions.experienceasset==1, use Policy directly
end

%% Deal with entry and exit if that is being used
if simoptions.agententryandexit==1 % If there is entry and exit use the command for that, otherwise just continue as usual.
    % It is assumed that the 'entry' distribution is suitable initial guess
    % for stationary distribution (rather than usual approach of simulating a few agents)
    if isfield(EntryExitParamNames,'CondlEntryDecisions')==1
        % Temporarily modify the 'DistOfNewAgents' value in Parameters to be that conditional on entry decisions.
        Parameters.(EntryExitParamNames.DistOfNewAgents{1})=reshape(Parameters.(EntryExitParamNames.DistOfNewAgents{1}),[N_a*N_z,1]).*reshape(Parameters.(EntryExitParamNames.CondlEntryDecisions{1}),[N_a*N_z,1]);
        % Can then just do the rest of the computing the agents distribution exactly as normal.
    end
    
    StationaryDistKron.pdf=reshape(Parameters.(EntryExitParamNames.DistOfNewAgents{1}),[N_a*N_z,1]);
    StationaryDistKron.mass=Parameters.(EntryExitParamNames.MassOfNewAgents{1});
    [StationaryDist]=StationaryDist_Case1_Iteration_EntryExit_raw(StationaryDistKron,Parameters,EntryExitParamNames,PolicyKron,N_d,N_a,N_z,pi_z,simoptions);
    StationaryDist.pdf=reshape(StationaryDist.pdf,[n_a,n_z]);
    return
elseif simoptions.agententryandexit==2 % If there is exogenous entry and exit, but of trival nature so mass of agent distribution is unaffected.
    % (This is used in some infinite horizon models to control the distribution; avoid, e.g., some people/firms saving 'too much')
    % To create initial guess use ('middle' of) the newborns distribution for seed point and do no burnin and short simulations (ignoring exit).
    EntryDist=reshape(Parameters.(EntryExitParamNames.DistOfNewAgents{1}),[N_a*N_z,1]);
    [~,seedpoint_index]=max(abs(cumsum(EntryDist)-0.5));
    simoptions.seedpoint=ind2sub_homemade([N_a,N_z],seedpoint_index); % Would obviously be better initial guess to do a bunch of different simulations for variety of points in the 'EntryDist'.
    simoptions.simperiods=10^3;
    simoptions.burnin=0;
    if simoptions.parallel<=2
        StationaryDistKron=StationaryDist_Case1_Simulation_raw(PolicyKron,N_d,N_a,N_z,pi_z, simoptions);
    elseif simoptions.parallel>2
        StationaryDistKron=sparse(StationaryDist_Case1_Simulation_raw(PolicyKron,N_d,N_a,N_z,pi_z, simoptions));
    end
    if simoptions.verbose==1
        fprintf('Note: simoptions.iterate=1 is imposed/required when using simoptions.agententryandexit=2 \n')
    end
    ExitProb=Parameters.(EntryExitParamNames.ProbOfDeath{1});
    StationaryDist=StationaryDist_Case1_Iteration_EntryExit2_raw(StationaryDistKron,PolicyKron,N_d,N_a,N_z,pi_z,ExitProb,EntryDist,simoptions);
    StationaryDist=reshape(StationaryDist,[n_a,n_z]);
    return
end

%% Semi-endogenous state
% The transition matrix of the exogenous shocks depends on the value of the endogenous state.
if isfield(simoptions,'SemiEndogShockFn')
    if isa(simoptions.SemiEndogShockFn,'function_handle')==0
        pi_z_semiendog=simoptions.SemiEndogShockFn;
    else
        if ~isfield(simoptions,'SemiEndogShockFnParamNames')
            dbstack
            error('simoptions.SemiEndogShockFnParamNames is missing (is needed for simoptions.SemiEndogShockFn)')
        end
        pi_z_semiendog=zeros(N_a,N_z,N_z);
        a_gridvals=CreateGridvals(n_a,a_grid,2);
        SemiEndogParamsVec=CreateVectorFromParams(Parameters, simoptions.SemiEndogShockFnParamNames);
        SemiEndogParamsCell=cell(length(SemiEndogParamsVec),1);
        for ii=1:length(SemiEndogParamsVec)
            SemiEndogParamsCell(ii,1)={SemiEndogParamsVec(ii)};
        end
        parfor ii=1:N_a
            a_ii=a_gridvals(ii,:)';
            a_ii_SemiEndogParamsCell=[a_ii;SemiEndogParamsCell];
            [~,temp_pi_z]=SemiEndogShockFn(a_ii_SemiEndogParamsCell{:});
            pi_z_semiendog(ii,:,:)=temp_pi_z;
            % Note that temp_z_grid is just the same things for all k, and same as
            % z_grid created about 10 lines above, so I don't bother keeping it.
            % I only create it so you can double-check it is same as z_grid
        end
    end
    if simoptions.eigenvector==1
        StationaryDistKron=StationaryDist_Case1_LeftEigen_SemiEndog_raw(PolicyKron,N_d,N_a,N_z,pi_z_semiendog,simoptions);
        StationaryDist=reshape(StationaryDistKron,[n_a,n_z]);
        return
    else
        dbstack
        error('Only simoptions.eigenvector=1 is implemented for StationaryDist when using SemiEndogShockFn')
    end    
end


%% If there is an initial dist use that, otherwise set up a (basic but poor) initial guess
if simoptions.iterate==1
    % Iteration must start from an initial guess
    if isfield(simoptions, 'initialdist')
        StationaryDistKron=reshape(simoptions.initialdist,[N_a*N_z,1]);
    else
        % Just use a poor initial guesses
        StationaryDistKron=zeros(N_a,N_z);
        z_stat=ones(N_z,1)/N_z;
        for jj=1:10
            z_stat=pi_z'*z_stat;
        end
        StationaryDistKron(ceil(N_a/2),:)=z_stat';
        StationaryDistKron=reshape(StationaryDistKron,[N_a*N_z,1]);
        % Note for self: Tan improvement divides into 2 steps, first is
        % policy, second is pi_z. Can't you then go a step further and
        % divide whole thing to just do 'only second step' until dist over z fully
        % converges, and after than 'only do first step' until get full
        % convergence?
        % This initial guess is kind of trying to partially exploit this
        % idea.
    end
end

%% Experience asset
if simoptions.experienceasset==1
    if ~exist('Parameters','var')
        error('When using simoptions.experienceasset=1 you must include Parameter structure as input to StationaryDist_Case1 (input just after simoptions)')
    end
    % Iterate using Tan improvement
    StationaryDist=StationaryDist_InfHorz_ExpAsset(StationaryDistKron,Policy,n_d,n_a,n_z,pi_z,Parameters,simoptions);
    return
end


%% Down to just the baseline case, codes show a couple of possiblities. Only one is used, rest are legacy/demonstration.

%% The eigenvector method is never used as it seems to be both slower and often has problems (gives incorrect solutions, it struggles with markov chains in which chunks of the asymptotic distribution are zeros)
if simoptions.eigenvector==1
    StationaryDistKron=StationaryDist_Case1_LeftEigen_raw(PolicyKron,N_d,N_a,N_z,pi_z,simoptions);
    if numel(StationaryDistKron)==1
        % Has failed, so continue on below to simulation and iteration commands
        warning('Eigenvector method for simulating agent dist failed, going to use simulate/iterate instead')
    else
        StationaryDist=reshape(StationaryDistKron,[n_a,n_z]);
        return
    end
end
% Eigenvector method is terribly slow for sparse matrices, so if the Ptranspose matrix (transition matrix on AxZ) does not fit in memory as a
% full matrix then better to just use simulation and iteration (or for really big state space possibly just simulation).


%% Simulate agent distribution, unless there is an initaldist guess for the agent distribution in which case use that
if simoptions.iterate==0
    % Not something you want to do, just a demo of alternative way to compute
    StationaryDistKron=StationaryDist_Case1_Simulation_raw(PolicyKron,N_d,N_a,N_z,pi_z, simoptions);
    StationaryDist=reshape(StationaryDistKron,[n_a,n_z]);
    return
end

%% Iterate on the agent distribution, starts from the simulated agent distribution (or the initialdist)
if simoptions.iterate==1
    if simoptions.tanimprovement==0
        StationaryDistKron=StationaryDist_Case1_Iteration_raw(StationaryDistKron,PolicyKron,N_d,N_a,N_z,pi_z,simoptions);
    elseif simoptions.tanimprovement==1 % Improvement of Tan (2020)
        StationaryDistKron=StationaryDist_Case1_IterationTan_raw(StationaryDistKron,PolicyKron,N_d,N_a,N_z,pi_z,simoptions);
    end
end
StationaryDist=reshape(StationaryDistKron,[n_a,n_z]);




end
