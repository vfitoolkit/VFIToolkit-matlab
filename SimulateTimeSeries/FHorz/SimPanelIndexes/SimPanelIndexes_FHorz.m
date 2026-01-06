function SimPanel=SimPanelIndexes_FHorz(InitialDist,Policy,n_d,n_a,n_z,N_j,pi_z_J, simoptions)
% Inputs should already be on cpu, output is on cpu
% 
% Intended to be called from SimPanelValues_FHorz_Case1()

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
    cumsumpi_z_J=cumsum(pi_z_J,2);
else
    l_z=0;
end
N_e=prod(simoptions.n_e);
if N_e>0
    l_e=length(simoptions.n_e);
    cumsumpi_e_J=gather(cumsum(simoptions.pi_e_J,1));
else
    l_e=0;
end

cumsumInitialDistVec=cumsum(InitialDist(:))/sum(InitialDist(:)); % Note: by using (:) I can ignore what the original dimensions were

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
    Policy=reshape(Policy,[size(Policy,1),N_a,N_j]);
    Policy_aprime=Policy(l_d+1:end,:,:);
else
    Policy=reshape(Policy,[size(Policy,1),N_a,N_bothze,N_j]);
    Policy_aprime=Policy(l_d+1:end,:,:,:);
end

if N_bothze==0
    Policy_aprime=KronPolicyIndexes_FHorz_Case1_noz(Policy_aprime,0,n_a,N_j,simoptions);
    if simoptions.gridinterplayer==1
        CumPolicyProbs=ones([N_a,2,N_j]);
        CumPolicyProbs(:,1,:)=reshape(Policy_aprime(2,:,:),[N_a,1,N_j]); % L2 index
        CumPolicyProbs(:,1,:)=1-(CumPolicyProbs(:,1,:)-1)/(simoptions.ngridinterp+1); % prob of lower index
        % CumPolicyProbs(:,2,:) just leave this as ones
        Policy_aprime=repmat(reshape(Policy_aprime(1,:,:),[N_a,1,N_j]),1,2,1); % lower grid index
        Policy_aprime(:,2,:)=Policy_aprime(:,2,:)+1; % upper grid index
    end
else
    Policy_aprime=KronPolicyIndexes_FHorz_Case1(Policy_aprime,0,n_a,N_bothze,N_j,simoptions);
    if simoptions.gridinterplayer==1
        CumPolicyProbs=ones([N_a,N_bothze,2,N_j]);
        CumPolicyProbs(:,:,1,:)=reshape(Policy_aprime(2,:,:,:),[N_a,N_bothze,1,N_j]); % L2 index
        CumPolicyProbs(:,:,1,:)=1-(CumPolicyProbs(:,:,1,:)-1)/(simoptions.ngridinterp+1); % prob of lower index
        % CumPolicyProbs(:,:,2,:) just leave this as ones
        Policy_aprime=repmat(reshape(Policy_aprime(1,:,:,:),[N_a,N_bothze,1,N_j]),1,1,2,1); % lower grid index
        Policy_aprime(:,:,2,:)=Policy_aprime(:,:,2,:)+1; % upper grid index
    end
end


%% First do the case without e variables, otherwise do with e variables
if N_z==0
    if N_e==0  % No z, No e
        if simoptions.gridinterplayer==0
            Policy_aprime=reshape(Policy_aprime,[N_a,N_j]);
        elseif simoptions.gridinterplayer==1
            Policy_aprime=reshape(Policy_aprime,[N_a,2,N_j]);
            CumPolicyProbs=reshape(CumPolicyProbs,[N_a,2,N_j]);
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
        if numel(InitialDist)==N_a % Has just been given for age j=1
            seedpoints=[ind2sub_vec_homemade(N_a,seedpointind'),ones(simoptions.numbersims,1)];
        else  % Distribution across ages as well
            seedpoints=[ind2sub_vec_homemade([N_a,N_j],seedpointind')]; %,ones(simoptions.numbersims,1)];
        end
        seedpoints=gather(floor(seedpoints)); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.
        
        % simoptions.simpanelindexkron==1 % Create the simulated data in kron form
        SimPanel=nan(2,N_j,simoptions.numbersims); % (a,j)
        if simoptions.gridinterplayer==0
            parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_noz_raw(Policy_aprime,N_j, simoptions, seedpoint);
                SimPanel(:,:,ii)=SimLifeCycleKron;
            end
        elseif simoptions.gridinterplayer==1
            parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_PolicyProbs_noz_raw(Policy_aprime,CumPolicyProbs,N_j, simoptions, seedpoint);
                SimPanel(:,:,ii)=SimLifeCycleKron;
            end
        end

        if simoptions.simpanelindexkron==0 % Convert results out of kron
            SimPanelKron=reshape(SimPanel,[2,N_j*simoptions.numbersims]);
            SimPanel=nan(l_a+1,N_j*simoptions.numbersims); % (a,j)

            SimPanel(1:l_a,:)=ind2sub_vec_homemade(n_a,SimPanelKron(1,:)')'; % a
            SimPanel(end,:)=SimPanelKron(2,:); % j

            SimPanel=reshape(SimPanel,[l_a+1,N_j,simoptions.numbersims]);
        end

    else  % No z, with e
        if simoptions.gridinterplayer==0
            Policy_aprime=reshape(Policy_aprime,[N_a,N_e,N_j]);
        elseif simoptions.gridinterplayer==1
            Policy_aprime=reshape(Policy_aprime,[N_a,N_e,2,N_j]);
            CumPolicyProbs=reshape(CumPolicyProbs,[N_a,N_e,2,N_j]);
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
        if numel(InitialDist)==N_a*N_e % Has just been given for age j=1
            seedpoints=[ind2sub_vec_homemade([N_a,N_e],seedpointind'),ones(simoptions.numbersims,1)];
        else  % Distribution across ages as well
            seedpoints=[ind2sub_vec_homemade([N_a,N_e,N_j],seedpointind'),ones(simoptions.numbersims,1)];
        end
        seedpoints=gather(floor(seedpoints)); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.

        % simoptions.simpanelindexkron==1 % Create the simulated data in kron form
        SimPanel=nan(3,N_j,simoptions.numbersims); % (a,e,j)
        if simoptions.gridinterplayer==0
            parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_noz_e_raw(Policy_aprime,N_j,cumsumpi_e_J, simoptions, seedpoint);
                SimPanel(:,:,ii)=SimLifeCycleKron;
            end
        elseif simoptions.gridinterplayer==1
            parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_PolicyProbs_noz_e_raw(Policy_aprime,CumPolicyProbs,N_j,cumsumpi_e_J, simoptions, seedpoint);
                SimPanel(:,:,ii)=SimLifeCycleKron;
            end
        end

        if simoptions.simpanelindexkron==0 % Convert results out of kron
            SimPanelKron=reshape(SimPanel,[3,N_j*simoptions.numbersims]);
            SimPanel=nan(l_a+l_e+1,N_j*simoptions.numbersims); % (a,e,j)

            SimPanel(1:l_a,:)=ind2sub_homemade(n_a,SimPanelKron(1,:)); % a
            SimPanel(l_a+1:l_a+l_e,:)=ind2sub_homemade(simoptions.n_e,SimPanelKron(2,:)); % e
            SimPanel(end,:)=SimPanelKron(3,:); % j

            SimPanel=reshape(SimPanel,[l_a+l_e+1,N_j,simoptions.numbersims]);
        else
            % All exogenous states together
            % Only e, so already is
        end
    end
else % N_z>0
    if N_e==0 % z, no e
        if simoptions.gridinterplayer==0
            Policy_aprime=reshape(Policy_aprime,[N_a,N_z,N_j]);
        elseif simoptions.gridinterplayer==1
            Policy_aprime=reshape(Policy_aprime,[N_a,N_z,2,N_j]);
            CumPolicyProbs=reshape(CumPolicyProbs,[N_a,N_z,2,N_j]);
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
        if numel(InitialDist)==N_a*N_z % Has just been given for age j=1
            seedpoints=[ind2sub_vec_homemade([N_a,N_z],seedpointind'),ones(simoptions.numbersims,1)];
        else  % Distribution across ages as well
            seedpoints=[ind2sub_vec_homemade([N_a,N_z,N_j],seedpointind')]; %,ones(simoptions.numbersims,1)];
        end
        seedpoints=gather(floor(seedpoints)); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.

        % simoptions.simpanelindexkron==1 % Create the simulated data in kron form
        SimPanel=nan(3,N_j,simoptions.numbersims); % (a,z,j)
        if simoptions.gridinterplayer==0
            parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_raw(Policy_aprime,N_j,cumsumpi_z_J, simoptions, seedpoint);
                SimPanel(:,:,ii)=SimLifeCycleKron;
            end
        elseif simoptions.gridinterplayer==1
            parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_PolicyProbs_raw(Policy_aprime,CumPolicyProbs,N_j,cumsumpi_z_J, simoptions, seedpoint);
                SimPanel(:,:,ii)=SimLifeCycleKron;
            end
        end

        if simoptions.simpanelindexkron==0 % Convert results out of kron
            SimPanelKron=reshape(SimPanel,[3,N_j*simoptions.numbersims]);
            SimPanel=nan(l_a+l_z+1,N_j*simoptions.numbersims); % (a,z,j)

            SimPanel(1:l_a,:)=ind2sub_vec_homemade(n_a,SimPanelKron(1,:)')'; % a
            SimPanel(l_a+1:l_a+l_z,:)=ind2sub_vec_homemade(n_z,SimPanelKron(2,:)')'; % z
            SimPanel(end,:)=SimPanelKron(3,:); % j

            SimPanel=reshape(SimPanel,[l_a+l_z+1,N_j,simoptions.numbersims]);
        else
            % All exogenous states together
            % Only z, so already is
        end


    else % z and e
        if simoptions.gridinterplayer==0
            Policy_aprime=reshape(Policy_aprime,[N_a,N_z,N_e,N_j]);
        elseif simoptions.gridinterplayer==1
            Policy_aprime=reshape(Policy_aprime,[N_a,N_z,N_e,2,N_j]);
            CumPolicyProbs=reshape(CumPolicyProbs,[N_a,N_z,N_e,2,N_j]);
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
        if numel(InitialDist)==N_a*N_z*N_e % Has just been given for age j=1
            seedpoints=[ind2sub_vec_homemade([N_a,N_z,N_e],seedpointind'),ones(simoptions.numbersims,1)];
        else  % Distribution across ages as well
            seedpoints=[ind2sub_vec_homemade([N_a,N_z,N_e,N_j],seedpointind'),ones(simoptions.numbersims,1)];
        end
        seedpoints=gather(floor(seedpoints)); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.
        
        % simoptions.simpanelindexkron==1 % Create the simulated data in kron form
        SimPanel=nan(4,N_j,simoptions.numbersims); % (a,z,e,j)
        if simoptions.gridinterplayer==0
            parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_e_raw(Policy_aprime,N_j,cumsumpi_z_J,cumsumpi_e_J, simoptions, seedpoint);
                SimPanel(:,:,ii)=SimLifeCycleKron;
            end
        elseif simoptions.gridinterplayer==1
            parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_PolicyProbs_e_raw(Policy_aprime,CumPolicyProbs,N_j,cumsumpi_z_J,cumsumpi_e_J, simoptions, seedpoint);
                SimPanel(:,:,ii)=SimLifeCycleKron;
            end
        end

        if simoptions.simpanelindexkron==0 % Convert results out of kron
            SimPanelKron=reshape(SimPanel,[4,N_j*simoptions.numbersims]);
            SimPanel=nan(l_a+l_z+l_e+1,N_j*simoptions.numbersims); % (a,z,e,j)

            SimPanel(1:l_a,:)=ind2sub_homemade(n_a,SimPanelKron(1,:)); % a
            SimPanel(l_a+1:l_a+l_z,:)=ind2sub_homemade(n_z,SimPanelKron(2,:)); % z
            SimPanel(l_a+l_z+1:l_a+l_z+l_e,:)=ind2sub_homemade(simoptions.n_e,SimPanelKron(3,:)); % e
            SimPanel(end,:)=SimPanelKron(4,:); % j

            SimPanel=reshape(SimPanel,[l_a+l_z+l_e+1,N_j,simoptions.numbersims]);
        else
            % All exogenous states together
            SimPanel(2,:,:)=SimPanel(2,:,:)+N_z*(SimPanel(3,:,:)-1); % put z and e together
            SimPanel(3,:,:)=SimPanel(4,:,:); % move j forward
            SimPanel=SimPanel(1:3,:,:);
        end
    end
end


end



