function SimPanel=SimPanelIndexes_FHorz_Case1(InitialDist,PolicyKron,n_d,n_a,n_z,N_j,pi_z_J, simoptions)
% Simulates a panel based on PolicyIndexes of 'numbersims' agents of length
% 'simperiods' beginning from randomly drawn InitialDist. (If you use the
% newbirths option you will get more than 'numbersims', due to the extra births)
%
% InitialDist can be inputed as over the finite time-horizon (j), or
% without a time-horizon in which case it is assumed to be an InitialDist
% for time j=1. (So InitialDist is either n_a-by-n_z-by-n_j, or n_a-by-n_z)


%% Check which simoptions have been declared, set all others to defaults 
if exist('simoptions','var')==1
    %Check simoptions for missing fields, if there are some fill them with
    %the defaults
    if ~isfield(simoptions, 'polindorval')
        simoptions.polindorval=1;
    end
    if ~isfield(simoptions, 'simperiods')
        simoptions.simperiods=N_j;
    end
    if ~isfield(simoptions, 'numbersims')
        simoptions.numbersims=10^3;
    end
    if ~isfield(simoptions, 'parallel')
        simoptions.parallel=1; % parallel CPU for panel data simulations
    end
    if ~isfield(simoptions, 'verbose')
        simoptions.verbose=0;
    end
    if isfield(simoptions,'ExogShockFn') % If using ExogShockFn then figure out the parameter names
        simoptions.ExogShockFnParamNames=getAnonymousFnInputNames(simoptions.ExogShockFn);
    end
    
    simoptions.newbirths=0; % It is assumed you do not want to add 'new births' to panel as you go. If you do you just tell it the 'birstdist' (sometimes just the same as InitialDist, but not often)
    if isfield(simoptions,'birthdist')
        simoptions.newbirths=1;
        % if you input simoptions.birthdist, you must also input
        % simoptions.birthrate (can be scalar, or vector of length
        % simoptions.simperiods)
        % I do not allow for the birthdist to change over time, only the
        % birthrate.
    end
    if ~isfield(simoptions, 'simpanelindexkron')
        simoptions.simpanelindexkron=0;
    end

else
    %If simoptions is not given, just use all the defaults
    simoptions.polindorval=1;
    simoptions.simperiods=N_j;
    simoptions.numbersims=10^3;
    simoptions.parallel=1; % parallel CPU for panel data simulations
    simoptions.verbose=0;
    simoptions.newbirths=0;
    simoptions.simpanelindexkron=0; % For some VFI Toolkit commands the kron is faster to use
end

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

N_a=prod(n_a);
N_z=prod(n_z);
N_d=prod(n_d);


%%
cumsumpi_z_J=cumsum(pi_z_J,2);
PolicyKron=gather(PolicyKron);
cumsumpi_z_J=gather(cumsumpi_z_J);

MoveOutputtoGPU=0;
if simoptions.parallel==2
    MoveOutputtoGPU=1;
    simoptions.parallel=1;
end


%% Some setups need to get sent off to alternative commands (specifically when using semiz variables)
if ~isfield(simoptions, 'n_semiz')
    simoptions.n_semiz=0;
end
if simoptions.n_semiz(1)>0
    if N_z==0
        if isfield(simoptions,'aggshock')
            SimPanel=SimPanelIndexes_FHorz_Case1_noz_semiz_aggshock(InitialDist,PolicyKron,n_d,n_a,N_j, simoptions);
            if MoveOutputtoGPU==1
                SimPanel=gpuArray(SimPanel);
            end
            return
        else
            SimPanel=SimPanelIndexes_FHorz_Case1_noz_semiz(InitialDist,PolicyKron,n_d,n_a,N_j, simoptions);
            if MoveOutputtoGPU==1
                SimPanel=gpuArray(SimPanel);
            end
            return
        end
    else
        SimPanel=SimPanelIndexes_FHorz_Case1_semiz(InitialDist,PolicyKron,n_d,n_a,n_z,N_j,cumsumpi_z_J, simoptions);
        if MoveOutputtoGPU==1
            SimPanel=gpuArray(SimPanel);
        end
        return
    end
end

%%
simperiods=gather(simoptions.simperiods);
numbersims=gather(simoptions.numbersims); % This is just to deal with weird error that matlab decided simoptions.numbersims was on gpu and so couldn't be an input to rand()

cumsumInitialDistVec=cumsum(InitialDist(:))/sum(InitialDist(:)); % Note: by using (:) I can ignore what the original dimensions were

%% First do the case without e variables, otherwise do with e variables
if ~isfield(simoptions,'n_e')
    
    % Get seedpoints from InitialDist
    if simoptions.lowmemory==0
        [~,seedpointind]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims)); % will end up with simoptions.numbersims random draws from cumsumInitialDistVec
    else % simoptions.lowmemory==1
        seedpointind=zeros(1,simoptions.numbersims);
        for ii=1:simoptions.numbersims
            [~,ind_ii]=max(cumsumInitialDistVec>rand(1,1));
            seedpointind(ii)=ind_ii;
        end
    end
    if numel(InitialDist)==N_a*N_z % Has just been given for age j=1
        seedpoints=[ind2sub_vec_homemade([N_a,N_z],seedpointind'),ones(simoptions.numbersims,1)];
    else  % Distribution across ages as well
        seedpoints=[ind2sub_vec_homemade([N_a,N_z,N_j],seedpointind'),ones(simoptions.numbersims,1)];
    end
    seedpoints=gather(floor(seedpoints)); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.
        
    % simoptions.simpanelindexkron==1 % Create the simulated data in kron form
    SimPanel=nan(3,N_j,simoptions.numbersims); % (a,z,j)
    if simoptions.parallel==0
        for ii=1:simoptions.numbersims
            seedpoint=seedpoints(ii,:);
            SimPanel(:,:,ii)=SimLifeCycleIndexes_FHorz_Case1_raw(PolicyKron,N_d,N_j,cumsumpi_z_J, seedpoint, simperiods);
        end
    else
        parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
            seedpoint=seedpoints(ii,:);
            SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_raw(PolicyKron,N_d,N_j,cumsumpi_z_J, seedpoint, simperiods);
            SimPanel(:,:,ii)=SimLifeCycleKron;
        end
    end


    if simoptions.newbirths==1
        cumulativebirthrate=cumprod(simoptions.birthrate.*ones(simperiods)+1)-1; % This works for scalar or vector simoptions.birthrate
        newbirthsvector=gather(round(simoptions.numbersims*cumulativebirthrate)); % Use rounding to decide how many new borns to do each period.
        BirthDist=gather(simoptions.birthdist);  % Make sure it is not on gpu

        SimPanel2=nan(l_a+l_z+1,simperiods,sum(newbirthsvector));
        for birthperiod=1:simperiods
            % Get seedpoints from birthdist
            seedpoints=nan(newbirthsvector(birthperiod),3); % 3 as a,z,j (vectorized)
            if numel(BirthDist)==N_a*N_z % Has just been given for age j=1
                cumsumBirthDistVec=cumsum(reshape(BirthDist,[N_a*N_z,1]));
                [~,seedpointind]=max(cumsumBirthDistVec>rand(1,numbersims,1));
                for ii=1:newbirthsvector(birthperiod)
                    seedpoints(ii,:)=[ind2sub_homemade([N_a,N_z],seedpointind(ii)),1];
                end
            else % Distribution across simperiods as well
                cumsumBirthDistVec=cumsum(reshape(BirthDist,[N_a*N_z*simperiods,1]));
                [~,seedpointind]=max(cumsumBirthDistVec>rand(1,simoptions.numbersims,1));
                for ii=1:newbirthsvector(birthperiod)
                    seedpoints(ii,:)=ind2sub_homemade([N_a,N_z,N_j],seedpointind(ii));
                end
            end
            seedpoints=floor(seedpoints);  % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.

            for ii=1:newbirthsvector(birthperiod)
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_raw(PolicyKron,N_d,N_j,cumsumpi_z_J, seedpoint, simperiods-birthperiod+1);
                SimPanel2(:,birthperiod:end,sum(newbirthsvector(1:(birthperiod-1)))+ii)=SimLifeCycleKron;
            end
        end
        SimPanel=[SimPanel;SimPanel2]; % Add the 'new borns' panel to the end of the main panel
    end

    if simoptions.simpanelindexkron==0 % Convert results out of kron
        SimPanelKron=reshape(SimPanel,[3,N_j*simoptions.numbersims]);
        SimPanel=nan(l_a+l_z+1,N_j*simoptions.numbersims); % (a,z,j)
        
        SimPanel(1:l_a,:)=ind2sub_homemade(n_a,SimPanelKron(1,:)); % a
        SimPanel(l_a+1:l_a+l_z,:)=ind2sub_homemade(n_z,SimPanelKron(2,:)); % z
        SimPanel(end,:)=SimPanelKron(3,:); % j
    end



else %if isfield(simoptions,'n_e')
    %% this time with e variables
    % If e variables are used they are treated seperately as this is faster/better
    N_e=prod(simoptions.n_e);
    l_e=length(simoptions.n_e);
    
    cumsumpi_e_J=gather(cumsum(simoptions.pi_e_J,1));
    
    % Get seedpoints from InitialDist
    if simoptions.lowmemory==0
        [~,seedpointind]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims)); % will end up with simoptions.numbersims random draws from cumsumInitialDistVec
    else % simoptions.lowmemory==1
        seedpointind=zeros(1,simoptions.numbersims);
        for ii=1:simoptions.numbersims
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
    SimPanel=nan(4,simperiods,simoptions.numbersims); % (a,z,e,j)
    if simoptions.parallel==0
        for ii=1:simoptions.numbersims
            seedpoint=seedpoints(ii,:);
            SimPanel(:,:,ii)=SimLifeCycleIndexes_FHorz_Case1_e_raw(PolicyKron,N_d,N_j,cumsumpi_z_J,cumsumpi_e_J,seedpoint,simperiods);
        end
    else
        parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
            seedpoint=seedpoints(ii,:);
            SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_e_raw(PolicyKron,N_d,N_j,cumsumpi_z_J,cumsumpi_e_J,seedpoint,simperiods);
            SimPanel(:,:,ii)=SimLifeCycleKron;
        end
    end


    if simoptions.newbirths==1
        cumulativebirthrate=cumprod(simoptions.birthrate.*ones(simperiods)+1)-1; % This works for scalar or vector simoptions.birthrate
        newbirthsvector=gather(round(simoptions.numbersims*cumulativebirthrate)); % Use rounding to decide how many new borns to do each period.
        BirthDist=gather(simoptions.birthdist);  % Make sure it is not on gpu

        SimPanel2=nan(l_a+l_z+l_e+1,simperiods,sum(newbirthsvector));
        for birthperiod=1:simperiods
            % Get seedpoints from birthdist
            seedpoints=nan(newbirthsvector(birthperiod),4); % 4 as a,z,e,j (vectorized)
            if numel(BirthDist)==N_a*N_z*N_e % Has just been given for age j=1
                cumsumBirthDistVec=cumsum(reshape(BirthDist,[N_a*N_z*N_e,1]));
                [~,seedpointvec]=max(cumsumBirthDistVec>rand(1,numbersims,1));
                for ii=1:newbirthsvector(birthperiod)
                    seedpoints(ii,:)=[ind2sub_homemade([N_a,N_z,N_e],seedpointvec(ii)),1];
                end
            else % Distribution across simperiods as well
                cumsumBirthDistVec=cumsum(reshape(BirthDist,[N_a*N_z*N_e*simperiods,1]));
                [~,seedpointvec]=max(cumsumBirthDistVec>rand(1,simoptions.numbersims,1));
                for ii=1:newbirthsvector(birthperiod)
                    seedpoints(ii,:)=ind2sub_homemade([N_a,N_z,N_e,N_j],seedpointvec(ii));
                end
            end
            seedpoints=floor(seedpoints);  % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.

            for ii=1:newbirthsvector(birthperiod)
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_e_raw(PolicyKron,N_d,N_j,cumsumpi_z_J,cumsumpi_e_J,seedpoint,simperiods);
                SimPanel2(:,birthperiod:end,sum(newbirthsvector(1:(birthperiod-1)))+ii)=SimLifeCycleKron;
            end
        end
        SimPanel=[SimPanel;SimPanel2]; % Add the 'new borns' panel to the end of the main panel
    end
        
    if simoptions.simpanelindexkron==0 % Convert results out of kron
        SimPanelKron=reshape(SimPanel,[4,N_j*simoptions.numbersims]);
        SimPanel=nan(l_a+l_z+l_e+1,N_j*simoptions.numbersims); % (a,z,e,j)
        
        SimPanel(1:l_a,:)=ind2sub_homemade(n_a,SimPanelKron(1,:)); % a
        SimPanel(l_a+1:l_a+l_z,:)=ind2sub_homemade(n_z,SimPanelKron(2,:)); % z
        SimPanel(l_a+l_z+1:l_a+l_z+l_e,:)=ind2sub_homemade(simoptions.n_e,SimPanelKron(3,:)); % e
        SimPanel(end,:)=SimPanelKron(4,:); % j
    end
        
end

if MoveOutputtoGPU==1
    SimPanel=gpuArray(SimPanel);
end

end



