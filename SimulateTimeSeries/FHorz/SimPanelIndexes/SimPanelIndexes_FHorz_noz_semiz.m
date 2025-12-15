function SimPanel=SimPanelIndexes_FHorz_noz_semiz(InitialDist,PolicyKron,n_a, n_semiz, cumsumpi_semiz_J, N_j, simoptions)
% Intended to be called from SimPanelIndexes_FHorz_semiz
%
% PolicyKron already only contains d2

N_a=prod(n_a);
N_semiz=prod(n_semiz);

l_a=length(n_a);

%%
simperiods=gather(simoptions.simperiods);

cumsumInitialDistVec=cumsum(InitialDist(:))/sum(InitialDist(:)); % Note: by using (:) I can ignore what the original dimensions were

%% First do the case without e variables, otherwise do with e variables
if ~isfield(simoptions,'n_e')
        
    % Get seedpoints from InitialDist
    [~,seedpointind]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims)); % will end up with simoptions.numbersims random draws from cumsumInitialDistVec
    if numel(InitialDist)==N_a*N_semiz % Has just been given for age j=1
        seedpoints=[ind2sub_vec_homemade([N_a,N_semiz],seedpointind'),ones(simoptions.numbersims,1)];
    else  % Distribution across ages as well
        seedpoints=[ind2sub_vec_homemade([N_a,N_semiz,N_j],seedpointind'),ones(simoptions.numbersims,1)];
    end
    seedpoints=gather(floor(seedpoints)); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.

    SimPanel=nan(3,simperiods,simoptions.numbersims); % (a,semiz,j)
    if simoptions.parallel==0
        for ii=1:simoptions.numbersims
            seedpoint=seedpoints(ii,:);
            SimPanel(:,:,ii)=SimLifeCycleIndexes_FHorz_noz_semiz_raw(PolicyKron,N_j,cumsumpi_semiz_J, simoptions, seedpoint);
        end
    else
        parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
            seedpoint=seedpoints(ii,:);
            SimLifeCycleKron=SimLifeCycleIndexes_FHorz_noz_semiz_raw(PolicyKron,N_j,cumsumpi_semiz_J, simoptions, seedpoint);
            SimPanel(:,:,ii)=SimLifeCycleKron;
        end
    end


    if simoptions.simpanelindexkron==0 % Convert results out of kron
        SimPanelKron=reshape(SimPanel,[3,N_j*simoptions.numbersims]);
        SimPanel=nan(l_a+l_semiz+1,N_j*simoptions.numbersims); % (a,semiz,j)
        
        SimPanel(1:l_a,:)=ind2sub_homemade(n_a,SimPanelKron(1,:)); % a
        SimPanel(l_a+1:l_a+l_semiz,:)=ind2sub_homemade(n_semiz,SimPanelKron(2,:)); % semiz
        SimPanel(end,:)=SimPanelKron(3,:); % j

        SimPanel=reshape(SimPanel,[3,N_j,simoptions.numbersims]);
    end
        
    
else %if isfield(simoptions,'n_e')
    %% this time with e variables
    % If e variables are used they are treated seperately as this is faster/better
    N_e=prod(simoptions.n_e);
    l_e=length(simoptions.n_e);
    
    cumsumpi_e_J=gather(cumsum(simoptions.pi_e_J,1));
    
    % Get seedpoints from InitialDist
    [~,seedpointind]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims)); % will end up with simoptions.numbersims random draws from cumsumInitialDistVec
    if numel(InitialDist)==N_a*N_semiz*N_e % Has just been given for age j=1
        seedpoints=[ind2sub_vec_homemade([N_a,N_semiz,N_e],seedpointind'),ones(simoptions.numbersims,1)];
    else  % Distribution across ages as well
        seedpoints=[ind2sub_vec_homemade([N_a,N_semiz,N_e,N_j],seedpointind'),ones(simoptions.numbersims,1)];
    end
    seedpoints=gather(floor(seedpoints)); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.

    SimPanel=nan(4,simperiods,simoptions.numbersims); % (a,semiz,e,j)
    if simoptions.parallel==0
        for ii=1:simoptions.numbersims
            seedpoint=seedpoints(ii,:);
            SimPanel(:,:,ii)=SimLifeCycleIndexes_FHorz_noz_semiz_e_raw(PolicyKron,N_semiz,N_j,cumsumpi_semiz_J,cumsumpi_e_J, simoptions, seedpoint);
        end
    else
        parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
            seedpoint=seedpoints(ii,:);
            SimLifeCycleKron=SimLifeCycleIndexes_FHorz_noz_semiz_e_raw(PolicyKron,N_semiz,N_j,cumsumpi_semiz_J,cumsumpi_e_J, simoptions, seedpoint);
            SimPanel(:,:,ii)=SimLifeCycleKron;
        end
    end

    if simoptions.simpanelindexkron==0 % Convert results out of kron
        SimPanelKron=reshape(SimPanel,[4,N_j*simoptions.numbersims]);
        SimPanel=nan(l_a+l_semiz+l_e+1,N_j*simoptions.numbersims); % (a,semiz,e,j)
        
        SimPanel(1:l_a,:)=ind2sub_homemade(n_a,SimPanelKron(1,:)); % a
        SimPanel(l_a+1:l_a+l_semiz,:)=ind2sub_homemade(n_semiz,SimPanelKron(2,:)); % semiz
        SimPanel(l_a+l_semiz+1:l_a+l_semiz+l_e,:)=ind2sub_homemade(simoptions.n_e,SimPanelKron(3,:)); % e
        SimPanel(end,:)=SimPanelKron(4,:); % j

        SimPanel=reshape(SimPanel,[4,N_j,simoptions.numbersims]);
    end

end


end



