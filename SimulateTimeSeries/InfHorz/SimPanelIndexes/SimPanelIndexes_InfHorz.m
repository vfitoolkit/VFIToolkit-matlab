function SimPanel=SimPanelIndexes_InfHorz(InitialDist,Policy,n_d,n_a,n_z,pi_z, simoptions, CondlProbOfSurvival, Parameters)
% Input must already be on CPU
% Simulates a panel based on PolicyIndexes of 'numbersims' agents of length 'simperiods' beginning from randomly drawn InitialDist. 
%
% CondlProbOfSurvival is an optional input. Only needed when using: simoptions.exitinpanel=1, there there is exit, either exog, endog or mix of both.
% Parameters is an optional input. Only needed when you have mixed (endogenous and exogenous) exit.
%
% simoptions are already set, this is an internal use only command
% (SimPanelIndexes is called by SimPanelValues)

%%
simoptions.numbersims=gather(simoptions.numbersims); % This is just to deal with weird error that matlab decided simoptions.numbersims was on gpu and so couldn't be an input to rand()
simoptions.simperiods=gather(simoptions.simperiods);
simoptions.burnin=gather(simoptions.burnin);

%%
N_d=prod(n_d);
if N_d>0
    l_d=length(n_d);
else
    l_d=0;
end

N_a=prod(n_a);
l_a=length(n_a);

N_z=prod(n_z);
if N_z>0
    l_z=length(n_z);
    cumsumpi_z=cumsum(pi_z,2);
else
    l_z=0;
end
N_e=prod(simoptions.n_e);
if N_e>0
    l_e=length(simoptions.n_e);
    cumsumpi_e=gather(cumsum(simoptions.pi_e,1));
else
    l_e=0;
end

cumsumInitialDistVec=cumsum(InitialDist(:))/sum(InitialDist(:)); % Note: by using (:) I can ignore what the original dimensions were

%%
if exist('CondlProbOfSurvival','var')
    simoptions.exitinpanel=1;
    CondlProbOfSurvival=reshape(CondlProbOfSurvival,[N_a,N_z]);
    if ~isfield(simoptions, 'endogenousexit')
        simoptions.endogenousexit=0;  % Note: this will only be relevant if exitinpanel=1
    end
else
    CondlProbOfSurvival=0; % will be unused, but otherwise there was an error that it wasnt recognized
end


%%
if N_z==0
    if N_e==0
        N_bothze=0;
    else
        N_bothze=N_e;
    end
else
    if N_e==0
        N_bothze=N_z;
    else
        N_bothze=N_z*N_e;
    end
end

if N_bothze==0
    Policy=reshape(Policy,[size(Policy,1),N_a]);
    Policy_aprime=Policy(l_d+1:end,:,:);
else
    Policy=reshape(Policy,[size(Policy,1),N_a,N_bothze]);
    Policy_aprime=Policy(l_d+1:end,:,:,:);
end

if N_bothze==0
    Policy_aprime=KronPolicyIndexes_Case1_noz(Policy_aprime,0,n_a,simoptions);
    if simoptions.gridinterplayer==1
        CumPolicyProbs=ones([N_a,2]);
        CumPolicyProbs(:,1)=reshape(Policy_aprime(2,:),[N_a,1]); % L2 index
        CumPolicyProbs(:,1)=1-(CumPolicyProbs(:,1)-1)/(simoptions.ngridinterp+1); % prob of lower index
        % CumPolicyProbs(:,2) just leave this as ones
        Policy_aprime=repmat(reshape(Policy_aprime(1,:),[N_a,1]),1,2); % lower grid index
        Policy_aprime(:,2)=Policy_aprime(:,2)+1; % upper grid index
    end
else
    Policy_aprime=KronPolicyIndexes_Case1(Policy_aprime,0,n_a,N_bothze,simoptions);
    if simoptions.gridinterplayer==1
        CumPolicyProbs=ones([N_a,N_bothze,2]);
        CumPolicyProbs(:,:,1)=reshape(Policy_aprime(2,:,:),[N_a,N_bothze,1]); % L2 index
        CumPolicyProbs(:,:,1)=1-(CumPolicyProbs(:,:,1)-1)/(simoptions.ngridinterp+1); % prob of lower index
        % CumPolicyProbs(:,:,2) just leave this as ones
        Policy_aprime=repmat(reshape(Policy_aprime(1,:,:),[N_a,N_bothze,1]),1,1,2); % lower grid index
        Policy_aprime(:,:,2)=Policy_aprime(:,:,2)+1; % upper grid index
    end
end


%%

SimPanel=nan(l_a+l_z,simoptions.simperiods,simoptions.numbersims); % preallocate

exitinpanel=simoptions.exitinpanel; % reduce overhead with parfor

if exitinpanel==0
    % % Baseline setup
    % parfor ii=1:simoptions.numbersims % Parallel CPUs for the simulations
    %     seedpoint_ii=ind2sub_homemade([N_a,N_z],seedpointvec(ii));
    %     seedpoint_ii=round(seedpoint_ii); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.
    % 
    %     SimTimeSeriesKron=SimTimeSeriesIndexes_InfHorz_raw(Policy,l_d,n_a,cumsumpi_z,seedpoint_ii,simoptions);
    % 
    %     SimPanel_ii=nan(l_a+l_z,simoptions.simperiods);
    % 
    %     for t=1:simoptions.simperiods
    %         temp=SimTimeSeriesKron(:,t);
    %         if ~isnan(temp)
    %             a_c_vec=ind2sub_homemade([n_a],temp(1));
    %             z_c_vec=ind2sub_homemade([n_z],temp(2));
    %             for kk=1:l_a
    %                 SimPanel_ii(kk,t)=a_c_vec(kk);
    %             end
    %             for kk=1:l_z
    %                 SimPanel_ii(l_a+kk,t)=z_c_vec(kk);
    %             end
    %         end
    %     end
    %     SimPanel(:,:,ii)=SimPanel_ii;
    % end
    
    %% First do the case without e variables, otherwise do with e variables
    if N_z==0
        if N_e==0  % No z, No e
            if simoptions.gridinterplayer==0
                Policy_aprime=reshape(Policy_aprime,[N_a,1]);
            elseif simoptions.gridinterplayer==1
                Policy_aprime=reshape(Policy_aprime,[N_a,2]);
                CumPolicyProbs=reshape(CumPolicyProbs,[N_a,2]);
            end

            % Get seedpoints from InitialDist
            if simoptions.lowmemory==0
                [~,seedpointind]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims)); % will end up with simoptions.numbersims random draws from cumsumInitialDistVec
            else % simoptions.lowmemory==1
                seedpointind=zeros(1,simoptions.numbersims);
                parfor ii=1:simoptions.numbersims
                    [~,ind_ii]=max(cumsumInitialDistVec>rand(1,1));
                    seedpointind(ii)=ind_ii;
                end
            end
            seedpoints=[ind2sub_vec_homemade(N_a,seedpointind'),ones(simoptions.numbersims,1)];
            seedpoints=gather(floor(seedpoints)); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.

            % simoptions.simpanelindexkron==1 % Create the simulated data in kron form
            SimPanel=nan(1,simoptions.simperiods,simoptions.numbersims); % (a)
            if simoptions.gridinterplayer==0
                parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                    seedpoint=seedpoints(ii,:);
                    SimLifeCycleKron=SimTimeSeriesIndexes_InfHorz_noz_raw(Policy_aprime, simoptions, seedpoint);
                    SimPanel(:,:,ii)=SimLifeCycleKron;
                end
            elseif simoptions.gridinterplayer==1
                parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                    seedpoint=seedpoints(ii,:);
                    SimLifeCycleKron=SimTimeSeriesIndexes_InfHorz_PolicyProbs_noz_raw(Policy_aprime,CumPolicyProbs, simoptions, seedpoint);
                    SimPanel(:,:,ii)=SimLifeCycleKron;
                end
            end
            
            if simoptions.simpanelindexkron==0 % Convert results out of kron
                SimPanelKron=reshape(SimPanel,[1,simoptions.simperiods*simoptions.numbersims]);
                SimPanel=nan(l_a,simoptions.simperiods*simoptions.numbersims); % (a)

                SimPanel(1:l_a,:)=ind2sub_vec_homemade(n_a,SimPanelKron(1,:)')'; % a
                SimPanel=reshape(SimPanel,[1,simoptions.simperiods,simoptions.numbersims]);
            end

        else  % No z, with e
            if simoptions.gridinterplayer==0
                Policy_aprime=reshape(Policy_aprime,[N_a,N_e]);
            elseif simoptions.gridinterplayer==1
                Policy_aprime=reshape(Policy_aprime,[N_a,N_e,2]);
                CumPolicyProbs=reshape(CumPolicyProbs,[N_a,N_e,2]);
            end

            % Get seedpoints from InitialDist
            if simoptions.lowmemory==0
                [~,seedpointind]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims)); % will end up with simoptions.numbersims random draws from cumsumInitialDistVec
            else % simoptions.lowmemory==1
                seedpointind=zeros(1,simoptions.numbersims);
                parfor ii=1:simoptions.numbersims
                    [~,ind_ii]=max(cumsumInitialDistVec>rand(1,1));
                    seedpointind(ii)=ind_ii;
                end
            end
            seedpoints=[ind2sub_vec_homemade([N_a,N_e],seedpointind'),ones(simoptions.numbersims,1)];
            seedpoints=gather(floor(seedpoints)); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.

            % simoptions.simpanelindexkron==1 % Create the simulated data in kron form
            SimPanel=nan(2,simoptions.simperiods,simoptions.numbersims); % (a,e)
            if simoptions.gridinterplayer==0
                parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                    seedpoint=seedpoints(ii,:);
                    SimLifeCycleKron=SimTimeSeriesIndexes_InfHorz_noz_e_raw(Policy_aprime,cumsumpi_e, simoptions, seedpoint);
                    SimPanel(:,:,ii)=SimLifeCycleKron;
                end
            elseif simoptions.gridinterplayer==1
                parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                    seedpoint=seedpoints(ii,:);
                    SimLifeCycleKron=SimTimeSeriesIndexes_InfHorz_PolicyProbs_noz_e_raw(Policy_aprime,CumPolicyProbs,cumsumpi_e, simoptions, seedpoint);
                    SimPanel(:,:,ii)=SimLifeCycleKron;
                end
            end

            if simoptions.simpanelindexkron==0 % Convert results out of kron
                SimPanelKron=reshape(SimPanel,[2,simoptions.simperiods*simoptions.numbersims]);
                SimPanel=nan(l_a+l_e,simoptions.simperiods*simoptions.numbersims); % (a,e)

                SimPanel(1:l_a,:)=ind2sub_homemade(n_a,SimPanelKron(1,:)); % a
                SimPanel(l_a+1:l_a+l_e,:)=ind2sub_homemade(simoptions.n_e,SimPanelKron(2,:)); % e
                SimPanel=reshape(SimPanel,[2,simoptions.simperiods,simoptions.numbersims]);
            else
                % All exogenous states together
                % Only e, so already is
            end
        end
    else % N_z>0
        if N_e==0 % z, no e
            if simoptions.gridinterplayer==0
                Policy_aprime=reshape(Policy_aprime,[N_a,N_z]);
            elseif simoptions.gridinterplayer==1
                Policy_aprime=reshape(Policy_aprime,[N_a,N_z,2]);
                CumPolicyProbs=reshape(CumPolicyProbs,[N_a,N_z,2]);
            end

            % Get seedpoints from InitialDist
            if simoptions.lowmemory==0
                [~,seedpointind]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims)); % will end up with simoptions.numbersims random draws from cumsumInitialDistVec
            else % simoptions.lowmemory==1
                seedpointind=zeros(1,simoptions.numbersims);
                parfor ii=1:simoptions.numbersims
                    [~,ind_ii]=max(cumsumInitialDistVec>rand(1,1));
                    seedpointind(ii)=ind_ii;
                end
            end
            seedpoints=[ind2sub_vec_homemade([N_a,N_z],seedpointind'),ones(simoptions.numbersims,1)];
            seedpoints=gather(floor(seedpoints)); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.

            % simoptions.simpanelindexkron==1 % Create the simulated data in kron form
            SimPanel=nan(2,simoptions.simperiods,simoptions.numbersims); % (a,z)
            if simoptions.gridinterplayer==0
                parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                    seedpoint=seedpoints(ii,:);
                    SimLifeCycleKron=SimTimeSeriesIndexes_InfHorz_raw(Policy_aprime,cumsumpi_z, simoptions, seedpoint);
                    SimPanel(:,:,ii)=SimLifeCycleKron;
                end
            elseif simoptions.gridinterplayer==1
                parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                    seedpoint=seedpoints(ii,:);
                    SimLifeCycleKron=SimTimeSeriesIndexes_InfHorz_PolicyProbs_raw(Policy_aprime,CumPolicyProbs,cumsumpi_z, simoptions, seedpoint);
                    SimPanel(:,:,ii)=SimLifeCycleKron;
                end
            end

            if simoptions.simpanelindexkron==0 % Convert results out of kron
                SimPanelKron=reshape(SimPanel,[2,simoptions.simperiods*simoptions.numbersims]);
                SimPanel=nan(l_a+l_z,simoptions.simperiods*simoptions.numbersims); % (a,z)

                SimPanel(1:l_a,:)=ind2sub_vec_homemade(n_a,SimPanelKron(1,:)')'; % a
                SimPanel(l_a+1:l_a+l_z,:)=ind2sub_vec_homemade(n_z,SimPanelKron(2,:)')'; % z
                SimPanel=reshape(SimPanel,[2,simoptions.simperiods,simoptions.numbersims]);
            else
                % All exogenous states together
                % Only z, so already is
            end


        else % z and e
            if simoptions.gridinterplayer==0
                Policy_aprime=reshape(Policy_aprime,[N_a,N_z,N_e]);
            elseif simoptions.gridinterplayer==1
                Policy_aprime=reshape(Policy_aprime,[N_a,N_z,N_e,2]);
                CumPolicyProbs=reshape(CumPolicyProbs,[N_a,N_z,N_e,2]);
            end

            % Get seedpoints from InitialDist
            if simoptions.lowmemory==0
                [~,seedpointind]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims)); % will end up with simoptions.numbersims random draws from cumsumInitialDistVec
            else % simoptions.lowmemory==1
                seedpointind=zeros(1,simoptions.numbersims);
                parfor ii=1:simoptions.numbersims
                    [~,ind_ii]=max(cumsumInitialDistVec>rand(1,1));
                    seedpointind(ii)=ind_ii;
                end
            end
            seedpoints=[ind2sub_vec_homemade([N_a,N_z,N_e],seedpointind'),ones(simoptions.numbersims,1)];
            seedpoints=gather(floor(seedpoints)); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.

            % simoptions.simpanelindexkron==1 % Create the simulated data in kron form
            SimPanel=nan(3,simoptions.simperiods,simoptions.numbersims); % (a,z,e)
            if simoptions.gridinterplayer==0
                parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                    seedpoint=seedpoints(ii,:);
                    SimLifeCycleKron=SimTimeSeriesIndexes_InfHorz_e_raw(Policy_aprime,cumsumpi_z,cumsumpi_e, simoptions, seedpoint);
                    SimPanel(:,:,ii)=SimLifeCycleKron;
                end
            elseif simoptions.gridinterplayer==1
                parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                    seedpoint=seedpoints(ii,:);
                    SimLifeCycleKron=SimTimeSeriesIndexes_InfHorz_PolicyProbs_e_raw(Policy_aprime,CumPolicyProbs,cumsumpi_z,cumsumpi_e, simoptions, seedpoint);
                    SimPanel(:,:,ii)=SimLifeCycleKron;
                end
            end

            if simoptions.simpanelindexkron==0 % Convert results out of kron
                SimPanelKron=reshape(SimPanel,[3,simoptions.simperiods*simoptions.numbersims]);
                SimPanel=nan(l_a+l_z+l_e,simoptions.simperiods*simoptions.numbersims); % (a,z,e)

                SimPanel(1:l_a,:)=ind2sub_homemade(n_a,SimPanelKron(1,:)); % a
                SimPanel(l_a+1:l_a+l_z,:)=ind2sub_homemade(n_z,SimPanelKron(2,:)); % z
                SimPanel(l_a+l_z+1:l_a+l_z+l_e,:)=ind2sub_homemade(simoptions.n_e,SimPanelKron(3,:)); % e
                SimPanel=reshape(SimPanel,[3,simoptions.simperiods,simoptions.numbersims]);
            else
                % All exogenous states together
                SimPanel(2,:,:)=SimPanel(2,:,:)+N_z*(SimPanel(3,:,:)-1); % put z and e together
                SimPanel=SimPanel(1:2,:,:);
            end
        end
    end

else

    %% With exit and perhaps entry
    
    % Get seedpoints from InitialDist
    [~,seedpointvec]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims));

    endogenousexit=simoptions.endogenousexit; % reduce overhead with parfor
    if endogenousexit==2
        exitprobabilities=CreateVectorFromParams(Parameters, simoptions.exitprobabilities);
        exitprobs=[1-sum(exitprobabilities),exitprobabilities];
    else
        exitprobs=0; % Not sure why, but Matlab was throwing error if this did not exist even when endogenousexit~=2, presumably something to do with figuring out the parallelization for parfor???
    end

    parfor ii=1:simoptions.numbersims % Parallel CPUs for the simulations
        seedpoint_ii=ind2sub_homemade([N_a,N_z],seedpointvec(ii));
        seedpoint_ii=round(seedpoint_ii); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.

        if endogenousexit==2 % Mixture of endogenous and exogenous exit
            SimTimeSeriesKron=SimTimeSeriesIndexes_InfHorz_Exit2_raw(Policy, CondlProbOfSurvival,N_d,N_a,N_z,cumsumpi_z,simoptions.burnin,seedpoint_ii,simoptions.simperiods,exitprobs,0); % 0: burnin, 0: use single CPU
        else % Otherwise (either one of endogenous or exogenous exit; but not mixture)
            SimTimeSeriesKron=SimTimeSeriesIndexes_InfHorz_Exit_raw(Policy, CondlProbOfSurvival,N_d,N_a,N_z,cumsumpi_z,simoptions.burnin,seedpoint_ii,simoptions.simperiods,0); % 0: burnin, 0: use single CPU
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
end



