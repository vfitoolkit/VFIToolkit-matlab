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
    % Options relating to simulation method
    simoptions.ncores=1;
    simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
    simoptions.simperiods=10^6; % I tried a few different things and this seems reasonable.
    simoptions.burnin=10^3; % Increasing this to 10^4 did not seem to impact the actual simulation agent distributions
    % Options for other solution methods, probably not things you want to play with
    simoptions.iterate=1;
    simoptions.tanimprovement=1; % Use Tan (2020) improvement to iteration (is hardcoded into everything but the most basic setting)
    simoptions.eigenvector=0; % I implemented an eigenvector based approach. It is fast but not robust.
    % Alternative setups
    simoptions.gridinterplayer=0;
    simoptions.agententryandexit=0;
    % simoptions.endogenousexit=0; % Not needed when simoptions.agententryandexit=0;
    % simoptions.SemiEndogShockFn % Undeclared by default (cannot be used with entry and exit)
    simoptions.experienceasset=0;
    simoptions.inheritanceasset=0;
    % Alternative Exogenous States
    simoptions.n_e=0;
    simoptions.n_semiz=0;
    % When calling as a subcommand, the following is used internally
    simoptions.outputkron=0;
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
    % Options for other solution methods, probably not things you want to play with
    if ~isfield(simoptions, 'iterate')
        simoptions.iterate=1;
    end
    if ~isfield(simoptions, 'tanimprovement')
        simoptions.tanimprovement=1; % Use Tan (2020) improvement to iteration (is hardcoded into everything but the most basic setting)
    end
    if ~isfield(simoptions,'eigenvector')
        simoptions.eigenvector=0; % I implemented an eigenvector based approach. It is fast but not robust.
    end
    % Alternative setups
    if ~isfield(simoptions, 'gridinterplayer')
        simoptions.gridinterplayer=0;
        if simoptions.gridinterplayer==1
            error('When using simoptions.gridinterplayer=1, you must set simoptions.ngridinterp')
        end
    end
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
    if ~isfield(simoptions,'inheritanceasset')
        simoptions.inheritanceasset=0;
    end
    % Alternative Exogenous States
    if ~isfield(simoptions,'n_e')
        simoptions.n_e=0;
    end
    if ~isfield(simoptions,'n_semiz')
        simoptions.n_semiz=0;
    end
    % When calling as a subcommand, the following is used internally
    if ~isfield(simoptions, 'outputkron')
        simoptions.outputkron=0;
    end
    if ~isfield(simoptions,'alreadygridvals')
        simoptions.alreadygridvals=0;
    end
end

%% Setup for Exogenous Shocks
if simoptions.alreadygridvals==0
    if isfield(simoptions,'ExogShockFn')
        [~, pi_z, simoptions]=ExogShockSetup(n_z,[],pi_z,Parameters,simoptions,2);
    end
end

N_e=prod(simoptions.n_e);
N_semiz=prod(simoptions.n_semiz);
if N_e>0
    error('Have not yet implemented e variables for InfHorz, ask on forum you need this')
end
if N_semiz>0
    error('Have not yet implemented semiz variables for InfHorz, ask on forum if you need this')
end


%% Deal with entry and exit if that is being used
if simoptions.agententryandexit==1 % If there is entry and exit use the command for that, otherwise just continue as usual.
    Policy=KronPolicyIndexes_Case1(Policy, n_d, n_a, n_z, simoptions);
    % It is assumed that the 'entry' distribution is suitable initial guess
    % for stationary distribution (rather than usual approach of simulating a few agents)
    if isfield(EntryExitParamNames,'CondlEntryDecisions')==1
        % Temporarily modify the 'DistOfNewAgents' value in Parameters to be that conditional on entry decisions.
        Parameters.(EntryExitParamNames.DistOfNewAgents{1})=reshape(Parameters.(EntryExitParamNames.DistOfNewAgents{1}),[N_a*N_z,1]).*reshape(Parameters.(EntryExitParamNames.CondlEntryDecisions{1}),[N_a*N_z,1]);
        % Can then just do the rest of the computing the agents distribution exactly as normal.
    end
    
    StationaryDist.pdf=reshape(Parameters.(EntryExitParamNames.DistOfNewAgents{1}),[N_a*N_z,1]);
    StationaryDist.mass=Parameters.(EntryExitParamNames.MassOfNewAgents{1});
    [StationaryDist]=StationaryDist_InfHorz_Iteration_EntryExit_raw(StationaryDist,Parameters,EntryExitParamNames,Policy,N_d,N_a,N_z,pi_z,simoptions);
    StationaryDist.pdf=reshape(StationaryDist.pdf,[n_a,n_z]);
    return
elseif simoptions.agententryandexit==2 % If there is exogenous entry and exit, but of trival nature so mass of agent distribution is unaffected.
    Policy=KronPolicyIndexes_Case1(Policy, n_d, n_a, n_z, simoptions);
    % (This is used in some infinite horizon models to control the distribution; avoid, e.g., some people/firms saving 'too much')
    % To create initial guess use ('middle' of) the newborns distribution for seed point and do no burnin and short simulations (ignoring exit).
    EntryDist=reshape(Parameters.(EntryExitParamNames.DistOfNewAgents{1}),[N_a*N_z,1]);
    [~,seedpoint_index]=max(abs(cumsum(EntryDist)-0.5));
    simoptions.seedpoint=ind2sub_homemade([N_a,N_z],seedpoint_index); % Would obviously be better initial guess to do a bunch of different simulations for variety of points in the 'EntryDist'.
    simoptions.simperiods=10^3;
    simoptions.burnin=0;
    if simoptions.parallel<=2
        StationaryDist=StationaryDist_Case1_Simulation_raw(Policy,N_d,N_a,N_z,pi_z, simoptions);
    elseif simoptions.parallel>2
        StationaryDist=sparse(StationaryDist_Case1_Simulation_raw(Policy,N_d,N_a,N_z,pi_z, simoptions));
    end
    if simoptions.verbose==1
        fprintf('Note: simoptions.iterate=1 is imposed/required when using simoptions.agententryandexit=2 \n')
    end
    ExitProb=Parameters.(EntryExitParamNames.ProbOfDeath{1});
    StationaryDist=StationaryDist_InfHorz_Iteration_EntryExit2_raw(StationaryDist,Policy,N_d,N_a,N_z,pi_z,ExitProb,EntryDist,simoptions);
    StationaryDist=reshape(StationaryDist,[n_a,n_z]);
    return
end

%% Semi-endogenous state
% The transition matrix of the exogenous shocks depends on the value of the endogenous state.
if isfield(simoptions,'SemiEndogShockFn')
    Policy=KronPolicyIndexes_Case1(Policy, n_d, n_a, n_z, simoptions);
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
        StationaryDist=StationaryDist_InfHorz_LeftEigen_SemiEndog_raw(Policy,N_d,N_a,N_z,pi_z_semiendog,simoptions);
        StationaryDist=reshape(StationaryDist,[n_a,n_z]);
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
        StationaryDist=reshape(simoptions.initialdist,[N_a*N_z,1]);
    else
        % Just use a poor initial guesses
        StationaryDist=zeros(N_a,N_z);
        z_stat=ones(N_z,1)/N_z;
        for jj=1:10
            z_stat=pi_z'*z_stat;
        end
        StationaryDist(ceil(N_a/2),:)=z_stat';
        StationaryDist=reshape(StationaryDist,[N_a*N_z,1]);
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
    StationaryDist=StationaryDist_InfHorz_ExpAsset(StationaryDist,Policy,n_d,n_a,n_z,pi_z,Parameters,simoptions);
    return
end

%% Inheritance asset
if simoptions.inheritanceasset==1
    if ~exist('Parameters','var')
        error('When using simoptions.inheritanceasset=1 you must include Parameter structure as input to StationaryDist_Case1 (input just after simoptions)')
    end
    % Iterate using Tan improvement
    StationaryDist=StationaryDist_InfHorz_InheritAsset(StationaryDist,Policy,n_d,n_a,n_z,pi_z,Parameters,simoptions);
    return
end

%% Down to just the baseline case, codes show a couple of possiblities. Only one is used, rest are legacy/demonstration.

%%
if simoptions.gridinterplayer==1
    if N_z==0
        if N_e==0
            StationaryDist=StationaryDist_InfHorz_GI_noz_raw(StationaryDist,Policy,n_d,n_a,N_a,simoptions);
            StationaryDist=reshape(StationaryDist,[n_a,1]);
        else
            StationaryDist=StationaryDist_InfHorz_GI_noz_e_raw(StationaryDist,Policy,n_d,n_a,N_a,N_e,pi_e,simoptions);
            StationaryDist=reshape(StationaryDist,[n_a,n_e]);
        end
    else
        if N_e==0
            StationaryDist=StationaryDist_InfHorz_GI_raw(StationaryDist,Policy,n_d,n_a,N_a,N_z,pi_z,simoptions);
            StationaryDist=reshape(StationaryDist,[n_a,n_z]);
        else
            StationaryDist=StationaryDist_InfHorz_GI_e_raw(StationaryDist,Policy,n_d,n_a,N_a,N_z,N_e,pi_z,pi_e,simoptions);
            StationaryDist=reshape(StationaryDist,[n_a,n_z,n_e]);
        end
    end

    return
end

%% Iterate on the agent distribution, starts from the simulated agent distribution (or the initialdist)
if simoptions.iterate==1
    if N_z==0
        if N_e==0
            Policy=KronPolicyIndexes_Case1_noz(Policy, n_d, n_a, simoptions);
        else
            Policy=KronPolicyIndexes_Case1(Policy, n_d, n_a, simoptions.n_e, simoptions);
        end
    else
        if N_e==0
            Policy=KronPolicyIndexes_Case1(Policy, n_d, n_a, n_z, simoptions);
        else
            Policy=KronPolicyIndexes_Case1(Policy, n_d, n_a, [n_z,simoptions.n_e], simoptions);
        end
    end

    if N_d==0
        Policy_aprime=Policy;
    else
        if N_z==0 && N_e==0
            Policy_aprime=shiftdim(Policy(2,:),1);
        else
            Policy_aprime=shiftdim(Policy(2,:,:),1);
        end
    end
    
    if N_z==0
        if N_e==0
            StationaryDist=StationaryDist_InfHorz_IterationTan_noz_raw(StationaryDist,Policy_aprime,N_a,simoptions);
            if simoptions.outputkron==0
                StationaryDist=reshape(StationaryDist,[n_a,1]);
            else
                StationaryDist=reshape(StationaryDist,[N_a,1]);
            end
        else
            StationaryDist=StationaryDist_InfHorz_IterationTan_noz_e_raw(StationaryDist,Policy_aprime,N_a,N_e,pi_e,simoptions);
            if simoptions.outputkron==0
                StationaryDist=reshape(StationaryDist,[n_a,n_e]);
            else
                StationaryDist=reshape(StationaryDist,[N_a,N_e]);
            end
        end
    else
        if N_e==0
            if simoptions.tanimprovement==0 % Note: not using the Tan improvement is only for the baseline case with z (no e)
                StationaryDist=StationaryDist_InfHorz_Iteration_raw(StationaryDist,Policy_aprime,N_a,N_z,pi_z,simoptions);
            elseif simoptions.tanimprovement==1 % Improvement of Tan (2020)
                StationaryDist=StationaryDist_InfHorz_IterationTan_raw(StationaryDist,Policy_aprime,N_a,N_z,pi_z,simoptions);
            end
            if simoptions.outputkron==0
                StationaryDist=reshape(StationaryDist,[n_a,n_z]);
            else
                StationaryDist=reshape(StationaryDist,[N_a,N_z]);
            end
        else
            StationaryDist=StationaryDist_InfHorz_IterationTan_e_raw(StationaryDist,Policy_aprime,N_a,N_z,N_e,pi_z,pi_e,simoptions);
            if simoptions.outputkron==0
                StationaryDist=reshape(StationaryDist,[n_a,n_z,n_e]);
            else
                StationaryDist=reshape(StationaryDist,[N_a,N_z,N_e]);
            end
        end
    end

    return
end

% The rest are only implemented with z (no e). They are just legacy that
% shows other ways you can compute the agent distribution, not things you
% are actually going to want to use.


%% The eigenvector method is never used as it seems to be both slower and often has problems (gives incorrect solutions, it struggles with markov chains in which chunks of the asymptotic distribution are zeros)
if simoptions.eigenvector==1
    Policy=KronPolicyIndexes_Case1(Policy, n_d, n_a, n_z, simoptions);
    StationaryDist=StationaryDist_InfHorz_LeftEigen_raw(gather(Policy),N_d,N_a,N_z,gather(pi_z),simoptions);
    if isscalar(StationaryDist)
        % Has failed, so continue on below to simulation and iteration commands
        warning('Eigenvector method for simulating agent dist failed, going to use simulate/iterate instead')
    else
        StationaryDist=reshape(StationaryDist,[n_a,n_z]);
        return
    end
end
% Eigenvector method is terribly slow for sparse matrices, so if the Ptranspose matrix (transition matrix on AxZ) does not fit in memory as a
% full matrix then better to just use simulation and iteration (or for really big state space possibly just simulation).


%% Simulate agent distribution, unless there is an initaldist guess for the agent distribution in which case use that
if simoptions.iterate==0
    % Not something you want to do, just a demo of alternative way to compute
    Policy=KronPolicyIndexes_Case1(Policy, n_d, n_a, n_z, simoptions);
    StationaryDist=StationaryDist_InfHorz_Simulation_raw(Policy,N_d,N_a,N_z,pi_z, simoptions);
    StationaryDist=reshape(StationaryDist,[n_a,n_z]);
    return
end



end
