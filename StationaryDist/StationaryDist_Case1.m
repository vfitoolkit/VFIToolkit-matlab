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
    simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
    simoptions.simperiods=10^6;
    simoptions.burnin=10^3;
    simoptions.parallel=1+(gpuDeviceCount>0);
    simoptions.verbose=0;
    try 
        PoolDetails=gcp;
        simoptions.ncores=PoolDetails.NumWorkers;
    catch
        simoptions.ncores=1;
    end
    simoptions.maxit=5*10^4; %In my experience, after a simulation, if you need more that 5*10^4 iterations to reach the steady-state it is because something has gone wrong
    simoptions.iterate=1;
    simoptions.agententryandexit=0;
    % simoptions.endogenousexit=0; % Not needed when simoptions.agententryandexit=0;
    simoptions.tolerance=10^(-9);
else
    %Check simoptions for missing fields, if there are some fill them with the defaults
    if isfield(simoptions, 'maxit')==0
        simoptions.maxit=5*10^4;
    end
    if isfield(simoptions, 'tolerance')==0
        simoptions.tolerance=10^(-9);
    end
    if isfield(simoptions, 'seedpoint')==0
        simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
    end
    if isfield(simoptions, 'simperiods')==0
        simoptions.simperiods=10^6;
    end
    if isfield(simoptions, 'burnin')==0
        simoptions.burnin=10^3;
    end
    if isfield(simoptions, 'parallel')==0
        simoptions.parallel=1+(gpuDeviceCount>0);
    end
    if isfield(simoptions, 'verbose')==0
        simoptions.verbose=0;
    end
    if isfield(simoptions, 'ncores')==0
        try
            PoolDetails=gcp;
            simoptions.ncores=PoolDetails.NumWorkers;
        catch
            simoptions.ncores=1;
        end
    end
    if isfield(simoptions, 'agententryandexit')==0
        simoptions.agententryandexit=0;
    else
        if simoptions.agententryandexit==1
            if isfield(simoptions, 'endogenousexit')==0
                simoptions.endogenousexit=0;
            end
        end
    end
    if isfield(simoptions, 'iterate')==0
        simoptions.iterate=1;
    end
end

%%
PolicyKron=KronPolicyIndexes_Case1(Policy, n_d, n_a, n_z);%,simoptions);

if simoptions.parallel~=2 % To cover the case when using gpu to solve value fn, but cpu to solve agent dist
    PolicyKron=gather(PolicyKron);
end

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
%     whos StationaryDistKron
    if simoptions.verbose==1
        fprintf('Note: simoptions.iterate=1 is imposed/required when using simoptions.agententryandexit=2 \n')
    end
    ExitProb=Parameters.(EntryExitParamNames.ProbOfDeath{1});
    StationaryDist=StationaryDist_Case1_Iteration_EntryExit2_raw(StationaryDistKron,PolicyKron,N_d,N_a,N_z,pi_z,ExitProb,EntryDist,simoptions);
    StationaryDist=reshape(StationaryDist,[n_a,n_z]);
    return
end

if isfield(simoptions, 'initialdist')==0
    StationaryDistKron=StationaryDist_Case1_Simulation_raw(PolicyKron,N_d,N_a,N_z,pi_z, simoptions);
else
    StationaryDistKron=reshape(simoptions.initialdist,[N_a,N_z]);
end
%%
% tic;
if simoptions.iterate==1
    StationaryDistKron=StationaryDist_Case1_Iteration_raw(StationaryDistKron,PolicyKron,N_d,N_a,N_z,pi_z,simoptions);
end
% toc

StationaryDist=reshape(StationaryDistKron,[n_a,n_z]);

% varargout={StationaryDistKron};

end
