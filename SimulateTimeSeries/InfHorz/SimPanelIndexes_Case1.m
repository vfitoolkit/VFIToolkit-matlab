function SimPanel=SimPanelIndexes_Case1(InitialDist,Policy,n_d,n_a,n_z,pi_z, simoptions, CondlProbOfSurvival)
% Simulates a panel based on PolicyIndexes of 'numbersims' agents of length
% 'simperiods' beginning from randomly drawn InitialDist. 
% CondlProbOfSurvival is an optional input. (only needed when using: simoptions.exitinpanel=1, there there is exit, either exog, endog or mix of both)
%
% InitialDist can be inputed as over the finite time-horizon (j), or
% without a time-horizon in which case it is assumed to be an InitialDist
% for time j=1. (So InitialDist is either n_a-by-n_z-by-n_j, or n_a-by-n_z)

N_a=prod(n_a);
N_z=prod(n_z);
N_d=prod(n_d);

%% Check which simoptions have been declared, set all others to defaults 
if exist('simoptions','var')==1
    %Check simoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(simoptions, 'polindorval')
        simoptions.polindorval=1;
    end
    if ~isfield(simoptions, 'simperiods')
        simoptions.simperiods=50;
    end
    if ~isfield(simoptions, 'numbersims')
        simoptions.numbersims=10^3;
    end
    if ~isfield(simoptions, 'parallel')
        simoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(simoptions, 'verbose')
        simoptions.verbose=0;
    end
    if ~isfield(simoptions, 'verbose')
        simoptions.verbose=0;
    end
    if ~isfield(simoptions, 'exitinpanel')
        simoptions.exitinpanel=0;
    end
else
    %If simoptions is not given, just use all the defaults
    simoptions.polindorval=1;
    simoptions.simperiods=50;
    simoptions.numbersims=10^3;
    simoptions.parallel=1+(gpuDeviceCount>0);
    simoptions.verbose=0;
    simoptions.exitinpanel=0;
end

if exist('CondlProbOfSurvival','var')==1
    simoptions.exitinpanel=1;
    CondlProbOfSurvivalKron=reshape(CondlProbOfSurvival,[N_a,N_z]);
    if ~isfield(simoptions, 'endogenousexit')
        simoptions.endogenousexit=0;  % Note: this will only be relevant if exitinpanel=1
    end
end

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

% Check if the inputted Policy is already in form of PolicyIndexesKron. If
% so this saves a big chunk of the run time of 'SimPanelIndexes_FHorz_Case1',
% Since this command is often called as a subcommand by functions where
% PolicyIndexesKron it saves a lot of run time.
%Policy is [l_d+l_a,n_a,n_z]
if (l_d==0 && ndims(Policy)==2) || ndims(Policy)==3
%     disp('Policy is alread Kron')
    PolicyIndexesKron=Policy;
else %    size(Policy)==[l_d+l_a,n_a,n_z]
    PolicyIndexesKron=KronPolicyIndexes_Case1(Policy, n_d, n_a, n_z);%,simoptions);
end

if simoptions.parallel==2
    % Get seedpoints from InitialDist while on gpu
    seedpoints=nan(simoptions.numbersims,2,'gpuArray'); % 2 as a,z (vectorized)
    cumsumInitialDistVec=cumsum(reshape(InitialDist,[N_a*N_z,1]));
    [~,seedpointvec]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims,1,'gpuArray'));
    for ii=1:simoptions.numbersims
        seedpoints(ii,:)=ind2sub_homemade_gpu([N_a,N_z],seedpointvec(ii));
    end
    seedpoints=round(seedpoints); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.
else
    InitialDist=gather(InitialDist); % Make sure it is not on gpu
    numbersims=gather(simoptions.numbersims); % This is just to deal with weird error that matlab decided simoptions.numbersims was on gpu and so couldn't be an input to rand()
    % Get seedpoints from InitialDist
    seedpoints=nan(simoptions.numbersims,2); % 3 as a,z (vectorized)
    cumsumInitialDistVec=cumsum(reshape(InitialDist,[N_a*N_z,1]));
    [~,seedpointvec]=max(cumsumInitialDistVec>rand(1,numbersims,1));
    for ii=1:simoptions.numbersims
        seedpoints(ii,:)=ind2sub_homemade([N_a,N_z],seedpointvec(ii));
    end
    seedpoints=round(seedpoints); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.
end

% fieldexists_ExogShockFn=0; % Needed below for use in SimLifeCycleIndexes_FHorz_Case1_raw()
cumsumpi_z=cumsum(pi_z,2);

MoveOutputtoGPU=0;
if simoptions.parallel==2
    % Simulation on GPU is really slow. So instead, switch to CPU, and then switch
    % back. For anything but ridiculously short simulations it is more than worth the overhead.
    PolicyIndexesKron=gather(PolicyIndexesKron);
    cumsumpi_z=gather(cumsumpi_z);
    seedpoints=gather(seedpoints);
    MoveOutputtoGPU=1;
    simoptions.simperiods=gather(simoptions.simperiods);
end

SimPanel=nan(l_a+l_z,simoptions.simperiods,simoptions.numbersims); % (a,z)
if simoptions.parallel==0
    for ii=1:simoptions.numbersims
        seedpoint=seedpoints(ii,:);
        %         SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_raw(PolicyIndexesKron,N_d,N_a,N_z,N_j,cumsumpi_z, seedpoint, simoptions.simperiods,fieldexists_ExogShockFn);
        if simoptions.exitinpanel==0
            SimTimeSeriesKron=SimTimeSeriesIndexes_Case1_raw(PolicyIndexesKron,N_d,N_a,N_z,cumsumpi_z,0,seedpoint,simoptions.simperiods,0); % 0: burnin, 0: use single CPU
        else
            if simoptions.endogenousexit==2 % Mixture of endogenous and exogenous exit
                SimTimeSeriesKron=SimTimeSeriesIndexes_Case1_Exit2_raw(PolicyIndexesKron, CondlProbOfSurvivalKron,N_d,N_a,N_z,cumsumpi_z,0,seedpoint,simoptions.simperiods,simoptions.exitprobabilities,0); % 0: burnin, 0: use single CPU
            else % Otherwise (either one of endogenous or exogenous exit; but not mixture)
                SimTimeSeriesKron=SimTimeSeriesIndexes_Case1_Exit_raw(PolicyIndexesKron, CondlProbOfSurvivalKron,N_d,N_a,N_z,cumsumpi_z,0,seedpoint,simoptions.simperiods,0); % 0: burnin, 0: use single CPU
            end
        end
        
        SimPanel_ii=nan(l_a+l_z,simoptions.simperiods);
        
        for t=1:simoptions.simperiods
            temp=SimTimeSeriesKron(:,t);
            if ~isnan(temp)
                a_c_vec=ind2sub_homemade([n_a],temp(1));
                z_c_vec=ind2sub_homemade([n_z],temp(2));
                for kk=1:l_a
                    SimPanel_ii(kk,t)=a_c_vec(kk);
                end
                for kk=1:l_z
                    SimPanel_ii(l_a+kk,t)=z_c_vec(kk);
                end
            end
        end
        SimPanel(:,:,ii)=SimPanel_ii;
    end
else
    exitinpanel=simoptions.exitinpanel; % reduce overhead with parfor
    simperiods=simoptions.simperiods; % reduce overhead with parfor
    endogenousexit=simoptions.endogenousexit; % reduce overhead with parfor
    if endogenousexit==2
        exitprobabilities=simoptions.exitprobabilities; % reduce overhead with parfor
    else
        exitprobabilities=0; % Not sure why, but Matlab was throwing error if this did not exist even when endogenousexit~=2, presumably something to do with figuring out the parallelization for parfor???
    end
    parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
        seedpoint=seedpoints(ii,:);
        if exitinpanel==0
            SimTimeSeriesKron=SimTimeSeriesIndexes_Case1_raw(PolicyIndexesKron,N_d,N_a,N_z,cumsumpi_z,0,seedpoint,simperiods,0); % 0: burnin, 0: use single CPU
        else
            if endogenousexit==2 % Mixture of endogenous and exogenous exit
                SimTimeSeriesKron=SimTimeSeriesIndexes_Case1_Exit2_raw(PolicyIndexesKron, CondlProbOfSurvivalKron,N_d,N_a,N_z,cumsumpi_z,0,seedpoint,simperiods,exitprobabilities,0); % 0: burnin, 0: use single CPU
            else % Otherwise (either one of endogenous or exogenous exit; but not mixture)
                SimTimeSeriesKron=SimTimeSeriesIndexes_Case1_Exit_raw(PolicyIndexesKron, CondlProbOfSurvivalKron,N_d,N_a,N_z,cumsumpi_z,0,seedpoint,simperiods,0); % 0: burnin, 0: use single CPU
            end
        end
        
        SimPanel_ii=nan(l_a+l_z,simperiods);
        
        for t=1:simperiods
            temp=SimTimeSeriesKron(:,t);
            if ~isnan(temp)
                a_c_vec=ind2sub_homemade([n_a],temp(1));
                z_c_vec=ind2sub_homemade([n_z],temp(2));
                for kk=1:l_a
                    SimPanel_ii(kk,t)=a_c_vec(kk);
                end
                for kk=1:l_z
                    SimPanel_ii(l_a+kk,t)=z_c_vec(kk);
                end
            end
        end
        SimPanel(:,:,ii)=SimPanel_ii;
    end
end

if MoveOutputtoGPU==1
    SimPanel=gpuArray(SimPanel);
end

end



