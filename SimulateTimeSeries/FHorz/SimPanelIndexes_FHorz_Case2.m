function SimPanel=SimPanelIndexes_FHorz_Case2(InitialDist,Policy,n_d,n_a,n_z,N_j,pi_z,Phi_aprimeFn,Case2_Type,Parameters,PhiaprimeParamNames, simoptions)
% Simulates a panel based on PolicyIndexes of 'numbersims' agents of length
% 'simperiods' beginning from randomly drawn InitialDist. (If you use the
% newbirths option you will get more than 'numbersims', due to the extra births)
%
% InitialDist can be inputed as over the finite time-horizon (j), or
% without a time-horizon in which case it is assumed to be an InitialDist
% for time j=1. (So InitialDist is either n_a-by-n_z-by-n_j, or n_a-by-n_z)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

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
        simoptions.parallel=2;
    end
    if ~isfield(simoptions, 'verbose')
        simoptions.verbose=0;
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
else
    %If simoptions is not given, just use all the defaults
    simoptions.polindorval=1;
    simoptions.simperiods=N_j;
    simoptions.numbersims=10^3;
    simoptions.parallel=2;
    simoptions.verbose=0;
    simoptions.newbirths=0;
end

l_d=length(n_d);
l_a=length(n_a);
l_z=length(n_z);

% Check if the inputted Policy is already in form of PolicyIndexesKron. If
% so this saves a big chunk of the run time of 'SimPanelIndexes_FHorz_Case2',
% Since this command is often called as a subcommand by functions where
% PolicyIndexesKron it saves a lot of run time.
%Policy is [l_d,n_a,n_z,N_j]
if ndims(Policy)==3
%     disp('Policy is alread Kron')
    PolicyIndexesKron=Policy;
else %    size(Policy)==[l_d+l_a,n_a,n_z,N_j]
    PolicyIndexesKron=KronPolicyIndexes_FHorz_Case2(Policy, n_d, n_a, n_z, N_j);%,simoptions);
end

if simoptions.parallel==2
    % Get seedpoints from InitialDist while on gpu
    seedpoints=nan(simoptions.numbersims,3,'gpuArray'); % 3 as a,z,j (vectorized)
    if numel(InitialDist)==N_a*N_z % Has just been given for age j=1
        cumsumInitialDistVec=cumsum(reshape(InitialDist,[N_a*N_z,1]));
        [~,seedpointvec]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims,1,'gpuArray'));
        for ii=1:simoptions.numbersims
            seedpoints(ii,:)=[ind2sub_homemade_gpu([N_a,N_z],seedpointvec(ii)),1];
        end
    else % Distribution across ages as well
        cumsumInitialDistVec=cumsum(reshape(InitialDist,[N_a*N_z*N_j,1]));
        [~,seedpointvec]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims,1,'gpuArray'));
        for ii=1:simoptions.numbersims
            seedpoints(ii,:)=ind2sub_homemade_gpu([N_a,N_z,N_j],seedpointvec(ii));
        end
    end
    seedpoints=floor(seedpoints); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.
else
    InitialDist=gather(InitialDist); % Make sure it is not on gpu
    numbersims=gather(simoptions.numbersims); % This is just to deal with weird error that matlab decided simoptions.numbersims was on gpu and so couldn't be an input to rand()
    % Get seedpoints from InitialDist
    seedpoints=nan(simoptions.numbersims,3); % 3 as a,z,j (vectorized)
    if numel(InitialDist)==N_a*N_z % Has just been given for age j=1
        cumsumInitialDistVec=cumsum(reshape(InitialDist,[N_a*N_z,1]));
        [~,seedpointvec]=max(cumsumInitialDistVec>rand(1,numbersims,1));
        for ii=1:simoptions.numbersims
            seedpoints(ii,:)=[ind2sub_homemade([N_a,N_z],seedpointvec(ii)),1];
        end
    else % Distribution across ages as well
        cumsumInitialDistVec=cumsum(reshape(InitialDist,[N_a*N_z*N_j,1]));
        [~,seedpointvec]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims,1));
        for ii=1:simoptions.numbersims
            seedpoints(ii,:)=ind2sub_homemade([N_a,N_z,N_j],seedpointvec(ii));
        end
    end
    seedpoints=floor(seedpoints); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.
end

if isfield(simoptions,'ExogShockFn')==1
    fieldexists_ExogShockFn=1; % Needed below for use in SimLifeCycleIndexes_FHorz_Case2_raw()
    cumsumpi_z=nan(N_z,N_z,N_j);
    for jj=1:N_j
        if isfield(simoptions,'ExogShockFnParamNames')==1
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
            [~,pi_z_jj]=simoptions.ExogShockFn(ExogShockFnParamsVec);
        else
            [~,pi_z_jj]=simoptions.ExogShockFn(jj);
        end
        cumsumpi_z(:,:,jj)=gather(cumsum(pi_z_jj,2));
    end
else
    fieldexists_ExogShockFn=0; % Needed below for use in SimLifeCycleIndexes_FHorz_Case2_raw()
    cumsumpi_z=cumsum(pi_z,2);
end

if Case2_Type==1 % phi(d,a,z,z')
    disp('ERROR: StationaryDist_FHorz_Case2_Iteration_raw() not yet implemented for Case2_Type==1 (nor SimPanelIndexes_FHorz_Case2_raw)')
elseif Case2_Type==11 % phi(d,a,z')
    disp('ERROR: StationaryDist_FHorz_Case2_Iteration_raw() not yet implemented for Case2_Type==11 (nor SimPanelIndexes_FHorz_Case2_raw)')
elseif Case2_Type==12 % phi(d,a,z)
    Phi_of_Policy=zeros(N_a,N_z,N_j,'gpuArray'); %a'(a,z)
elseif Case2_Type==2  % phi(d,z',z)
    Phi_of_Policy=zeros(N_a,N_z,N_z,N_j,'gpuArray'); %a'(d,z',z)
end

if simoptions.phiaprimedependsonage==0
    PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
    if simoptions.lowmemory==0
        Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprimeFn, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
    end
end
for jj=1:N_j
    
    if simoptions.phiaprimedependsonage==1
        PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
        Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprimeFn, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
    end
    
    if Case2_Type==1 % phi(d,a,z,z')
        disp('ERROR: StationaryDist_FHorz_Case2_Iteration_raw() not yet implemented for Case2_Type==1 (nor SimPanelIndexes_FHorz_Case2_raw)')
    elseif Case2_Type==11 % phi(d,a,z')
        disp('ERROR: StationaryDist_FHorz_Case2_Iteration_raw() not yet implemented for Case2_Type==11 (nor SimPanelIndexes_FHorz_Case2_raw)')
    elseif Case2_Type==12 % phi(d,a,z)
        for z_c=1:N_z
            temp=[PolicyIndexesKron(:,z_c,jj),(1:N_a)',z_c*ones(N_a,1)];
            temp2=sub2ind([N_d,N_a,N_z],temp(:,1),temp(:,2),temp(:,3));
            Phi_of_Policy(:,z_c,jj)=Phi_aprimeMatrix(temp2); % WORK IN PROGRESS HERE
        end
    elseif Case2_Type==2  % phi(d,z',z)
        for z_c=1:N_z
            Phi_of_Policy(:,:,z_c,jj)=Phi_aprimeMatrix(PolicyIndexesKron(:,z_c,jj),:,z_c);
        end
    end    
end

MoveOutputtoGPU=0;
if simoptions.parallel==2
    % Simulation on GPU is really slow. So instead, switch to CPU, and then switch
    % back. For anything but ridiculously short simulations it is more than worth the overhead.
    Phi_of_Policy=gather(Phi_of_Policy);
    cumsumpi_z=gather(cumsumpi_z);
    seedpoints=gather(seedpoints);
    MoveOutputtoGPU=1;
    simoptions.simperiods=gather(simoptions.simperiods);
end


SimPanel=nan(l_a+l_z+1,simoptions.simperiods,simoptions.numbersims); % (a,z,j)
if simoptions.parallel==0
    for ii=1:simoptions.numbersims
        seedpoint=seedpoints(ii,:);
        SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case2_raw(Phi_of_Policy,Case2_Type,N_d,N_a,N_z,N_j,cumsumpi_z, seedpoint, simoptions.simperiods,fieldexists_ExogShockFn);
        
        SimPanel_ii=nan(l_a+l_z+1,simoptions.simperiods);
        
        j1=seedpoint(3);
        j2=min(N_j,j1+simoptions.simperiods);
        for t=1:(j2-j1+1)
            jj=t+j1-1;
            temp=SimLifeCycleKron(:,jj);
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
            SimPanel_ii(l_a+l_z+1,t)=jj;
        end
        SimPanel(:,:,ii)=SimPanel_ii;
    end
else
    parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
        seedpoint=seedpoints(ii,:);
        SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case2_raw(Phi_of_Policy,Case2_Type,N_d,N_a,N_z,N_j,cumsumpi_z, seedpoint, simoptions.simperiods,fieldexists_ExogShockFn);
        
        SimPanel_ii=nan(l_a+l_z+1,simoptions.simperiods);
        
        j1=seedpoint(3);
        j2=min(N_j,j1+simoptions.simperiods);
        for t=1:(j2-j1+1)
            jj=t+j1-1;
            temp=SimLifeCycleKron(:,jj);
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
            SimPanel_ii(l_a+l_z+1,t)=jj;
        end
        SimPanel(:,:,ii)=SimPanel_ii;
    end
end

if simoptions.newbirths==1
    cumulativebirthrate=cumprod(simoptions.birthrate.*ones(simoptions.simperiods)+1)-1; % This works for scalar or vector simoptions.birthrate
    newbirthsvector=gather(round(simoptions.numbersims*cumulativebirthrate)); % Use rounding to decide how many new borns to do each period.
    BirthDist=gather(simoptions.birthdist);  % Make sure it is not on gpu
    
    SimPanel2=nan(l_a+l_z+1,simoptions.simperiods,sum(newbirthsvector));
    for birthperiod=1:simoptions.simperiods
        % Get seedpoints from birthdist
        seedpoints=nan(newbirthsvector(birthperiod),3); % 3 as a,z,j (vectorized)
        if numel(BirthDist)==N_a*N_z % Has just been given for age j=1
            cumsumBirthDistVec=cumsum(reshape(BirthDist,[N_a*N_z,1]));
            [~,seedpointvec]=max(cumsumBirthDistVec>rand(1,numbersims,1));
            for ii=1:newbirthsvector(birthperiod)
                seedpoints(ii,:)=[ind2sub_homemade([N_a,N_z],seedpointvec(ii)),1];
            end
        else % Distribution across simoptions.simperiods as well
            cumsumBirthDistVec=cumsum(reshape(BirthDist,[N_a*N_z*simoptions.simperiods,1]));
            [~,seedpointvec]=max(cumsumBirthDistVec>rand(1,simoptions.numbersims,1));
            for ii=1:newbirthsvector(birthperiod)
                seedpoints(ii,:)=ind2sub_homemade([N_a,N_z,N_j],seedpointvec(ii));
            end
        end
        seedpoints=floor(seedpoints);  % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.
    
        for ii=1:newbirthsvector(birthperiod)
            seedpoint=seedpoints(ii,:);
            SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case2_raw(Phi_of_Policy,Case2_Type,N_d,N_a,N_z,N_j,cumsumpi_z, seedpoint, simoptions.simperiods-birthperiod+1,fieldexists_ExogShockFn);

            SimPanel_ii=nan(l_a+l_z+1,simoptions.simperiods);
            
            j1=seedpoint(3);
            j2=min(N_j,j1+(simoptions.simperiods-birthperiod+1));
            for t=1:(j2-j1+1)
                jj=t+j1-1;
                temp=SimLifeCycleKron(:,jj);
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
                SimPanel_ii(l_a+l_z+1,t)=jj;
            end
            SimPanel2(:,birthperiod:end,sum(newbirthsvector(1:(birthperiod-1)))+ii)=SimPanel_ii;
        end
    end
    SimPanel=[SimPanel;SimPanel2]; % Add the 'new borns' panel to the end of the main panel
end

if MoveOutputtoGPU==1
    SimPanel=gpuArray(SimPanel);
end

end



