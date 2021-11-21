function SimPanel=SimPanelIndexes_FHorz_Case2_AgeDepGrids_Dynasty(InitialDist,PolicyIndexesKron,daz_gridstructure,N_j,Phi_aprimeFn,Case2_Type,Parameters,PhiaprimeParamNames, simoptions)
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

%%
% Assumes inputted Policy is already in form of PolicyIndexesKron, as this saves a big chunk of the run time of 'SimPanelIndexes_FHorz_Case2',
% Since this command is only intended to be called as a subcommand by functions where PolicyIndexesKron it saves a lot of run time.

if simoptions.parallel==2
    N_a_j=daz_gridstructure.N_a.j001;
    N_z_j=daz_gridstructure.N_z.j001;
    % Get seedpoints from InitialDist while on gpu
    seedpoints=nan(simoptions.numbersims,3,'gpuArray'); % 3 as a,z,j (vectorized)
    if numel(InitialDist)==N_a_j*N_z_j % Has just been given for age j=1
        cumsumInitialDistVec=cumsum(reshape(InitialDist,[N_a_j*N_z_j,1]));
        [~,seedpointvec]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims,1,'gpuArray'));
        for ii=1:simoptions.numbersims
            seedpoints(ii,:)=[ind2sub_homemade_gpu([N_a_j,N_z_j],seedpointvec(ii)),1];
        end
    else % Distribution across ages as well
        % First, figure out the ages for each seedpointsvec (so that can do each age as a group, this will reduce computations).
        seeds_rand=rand(1,simoptions.numbersims,1,'gpuArray');
        [seeds_rand_sorted,sortIndex]=sort(seeds_rand);
        [~,seeds_age_sorted]=max(cumsum(InitialDist.AgeWeights')>seeds_rand_sorted);
        % There should be a smarter/faster way to implement the next five lines
        seeds_age_summary=zeros(N_j+1,1,'gpuArray');
        for jj=1:N_j
            [~,seeds_age_summary(jj)]=max((seeds_age_sorted-jj>=0));
        end
        seeds_age_summary(N_j+1)=simoptions.numbersims+1; % ie. =length(seeds_age_sorted)
        
%         seedpointvec=zeros(simoptions.numbersims,1,'gpuArray');
        AgeWeightsAdjustment=cumsum(InitialDist.AgeWeights)-InitialDist.AgeWeights(1);
        for jj=1:N_j
            seeds_rand_sorted_jj=seeds_rand_sorted(seeds_age_summary(jj):(seeds_age_summary(jj+1)-1))-AgeWeightsAdjustment(jj); % Get the seeds for the current age

            jstr=daz_gridstructure.jstr{jj};
            N_a_j=daz_gridstructure.N_a.(jstr(:));
            N_z_j=daz_gridstructure.N_z.(jstr(:));
            cumsumInitialDistVec=cumsum(reshape(InitialDist.(jstr),[N_a_j*N_z_j,1]));            
            % Use of max was creating an out-of-memory bottleneck
            for ii=1:(seeds_age_summary(jj+1)-seeds_age_summary(jj))
                [~,seedpointtemp]=max(cumsumInitialDistVec>seeds_rand_sorted_jj(ii));
                seedpoints(seeds_age_summary(jj)-1+ii,1:end-1)=ind2sub_homemade_gpu([N_a_j,N_z_j],seedpointtemp); % Just going through the 'max' for each ii one at a time. Slower, but avoids what was otherwise a potential memory bottleneck.
                seedpoints(seeds_age_summary(jj)-1+ii,end)=jj;
            end
        end
        % Now unsort the ages (technically no need for this, just 'nicer' if the resulting panel data set is not sorted by age at birth)
        unsortIndex(sortIndex)=1:1:simoptions.numbersims; % ie. 1:1:length(seeds_age_sorted)
        seedpoints=seedpoints(unsortIndex,:);
    end
    seedpoints=floor(seedpoints); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.
else
    N_a_j=daz_gridstructure.N_a.j001;
    N_z_j=daz_gridstructure.N_z.j001;
    InitialDist=gather(InitialDist); % Make sure it is not on gpu
    numbersims=gather(simoptions.numbersims); % This is just to deal with weird error that matlab decided simoptions.numbersims was on gpu and so couldn't be an input to rand()
    % Get seedpoints from InitialDist
    seedpoints=nan(simoptions.numbersims,3); % 3 as a,z,j (vectorized)
    if numel(InitialDist)==N_a_j*N_z_j % Has just been given for age j=1
        cumsumInitialDistVec=cumsum(reshape(InitialDist,[N_a_j*N_z_j,1]));
        [~,seedpointvec]=max(cumsumInitialDistVec>rand(1,numbersims,1));
        for ii=1:simoptions.numbersims
            seedpoints(ii,:)=[ind2sub_homemade([N_a_j,N_z_j],seedpointvec(ii)),1];
        end
    else % Distribution across ages as well
        % First, figure out the ages for each seedpointsvec (so that can do each age as a group, this will reduce computations).
        seeds_rand=rand(1,simoptions.numbersims,1);
        [seeds_rand_sorted,sortIndex]=sort(seeds_rand);
        [~,seeds_age_sorted]=max(cumsum(InitialDist.AgeWeights')>seeds_rand_sorted);
        % There should be a smarter/faster way to implement the next five lines
        seeds_age_summary=zeros(N_j+1,1);
        for jj=1:N_j
            [~,seeds_age_summary(jj)]=max((seeds_age_sorted-jj>=0));
        end
        seeds_age_summary(N_j+1)=simoptions.numbersims+1; % ie. =length(seeds_age_sorted)
        
        AgeWeightsAdjustment=cumsum(InitialDist.AgeWeights)-InitialDist.AgeWeights(1);
        for jj=1:N_j
            jstr=daz_gridstructure.jstr{jj};
            N_a_j=daz_gridstructure.N_a.(jstr(:));
            N_z_j=daz_gridstructure.N_z.(jstr(:));
            [uniq_cumsumInitialDistVec,uniq_index]=unique(cumsum(reshape(InitialDist.(jstr),[N_a_j*N_z_j,1])));
            uniq_cumsumInitialDistVec=gather(uniq_cumsumInitialDistVec);
            uniq_index=gather(uniq_index);
            [~,uniq_seedpointvectemp]=max(uniq_cumsumInitialDistVec>(seeds_rand_sorted-AgeWeightsAdjustment(jj)));
            seedpointvectemp=uniq_index(uniq_seedpointvectemp);
            for ii=1:(seeds_age_summary(jj+1)-seeds_age_summary(jj))
               seedpoints(seeds_age_summary(jj)-1+ii,1:end-1)=ind2sub_homemade([N_a_j,N_z_j],seedpointvectemp(ii));
               seedpoints(seeds_age_summary(jj)-1+ii,end)=jj;
            end
        end
        % Now unsort (technically no need for 'unsorting', just 'nicer' if the resulting panel data set is not sorted by age at birth)
        unsortIndex(sortIndex)=1:1:simoptions.numbersims; % ie. 1:1:length(seeds_age_sorted)
        seedpoints=seedpoints(unsortIndex,:);
    end
    seedpoints=floor(seedpoints); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.
end

%% Create Phi_of_Policy as this will be the basis for simulations.
% if Case2_Type==1 % phi(d,a,z,z')
%     disp('ERROR: StationaryDist_FHorz_Case2_Iteration_raw() not yet implemented for Case2_Type==1 (nor SimPanelIndexes_FHorz_Case2_raw)')
% elseif Case2_Type==11 % phi(d,a,z')
%     disp('ERROR: StationaryDist_FHorz_Case2_Iteration_raw() not yet implemented for Case2_Type==11 (nor SimPanelIndexes_FHorz_Case2_raw)')
% elseif Case2_Type==12 % phi(d,a,z)
%     Phi_of_Policy=zeros(N_a,N_z,N_j,'gpuArray'); %a'(a,z)
% elseif Case2_Type==2  % phi(d,z',z)
%     Phi_of_Policy=zeros(N_a,N_z,N_z,N_j,'gpuArray'); %a'(d,z',z)
% end
Phi_of_Policy=struct();

for jj=1:N_j
    jstr=daz_gridstructure.jstr{jj};
    n_d_j=daz_gridstructure.n_d.(jstr(:));
    n_a_j=daz_gridstructure.n_a.(jstr(:));
    n_z_j=daz_gridstructure.n_z.(jstr(:));
    N_d_j=daz_gridstructure.N_d.(jstr(:));
    N_a_j=daz_gridstructure.N_a.(jstr(:));
    N_z_j=daz_gridstructure.N_z.(jstr(:));
    N_z_jplus1=daz_gridstructure.N_z.(jstr(:));
    d_grid_j=daz_gridstructure.d_grid.(jstr(:));
    a_grid_j=daz_gridstructure.a_grid.(jstr(:));
    z_grid_j=daz_gridstructure.z_grid.(jstr(:));
    daz_gridstructure.cumsumpi_z.(jstr(:))=cumsum(daz_gridstructure.pi_z.(jstr(:)),2);
    
    PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
    Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprimeFn, Case2_Type, n_d_j, n_a_j, n_z_j, d_grid_j, a_grid_j, z_grid_j,PhiaprimeParamsVec);
    
    PolicyIndexesKron_jj=PolicyIndexesKron.(jstr(:));
    if Case2_Type==1 % phi(d,a,z,z')
        disp('ERROR: StationaryDist_FHorz_Case2_Iteration_raw() not yet implemented for Case2_Type==1 (nor SimPanelIndexes_FHorz_Case2_raw)')
    elseif Case2_Type==11 % phi(d,a,z')
        disp('ERROR: StationaryDist_FHorz_Case2_Iteration_raw() not yet implemented for Case2_Type==11 (nor SimPanelIndexes_FHorz_Case2_raw)')
    elseif Case2_Type==12 % phi(d,a,z)
        Phi_of_Policy_jj=zeros(N_a_j,N_z_j,'gpuArray');
        for z_c=1:N_z_j
            temp=[PolicyIndexesKron_jj(:,z_c),(1:N_a_j)',z_c*ones(N_a_j,1)];
            temp2=sub2ind([N_d_j,N_a_j,N_z_j],temp(:,1),temp(:,2),temp(:,3));
            Phi_of_Policy_jj(:,z_c)=Phi_aprimeMatrix(temp2); % WORK IN PROGRESS HERE
        end
    elseif Case2_Type==2  % phi(d,z',z)
        Phi_of_Policy_jj=zeros(N_a_j,N_z_jplus1,N_z_j,'gpuArray');
        for z_c=1:N_z_j
            Phi_of_Policy_jj(:,:,z_c)=Phi_aprimeMatrix(PolicyIndexesKron_jj(:,z_c),:,z_c);
        end
    end 
    Phi_of_Policy.(jstr(:))=Phi_of_Policy_jj;
end

MoveOutputtoGPU=0;
for jj=1:N_j
    jstr=daz_gridstructure.jstr{jj};
    Phi_of_Policy.(jstr(:))=gather(Phi_of_Policy.(jstr(:)));
    daz_gridstructure.cumsumpi_z.(jstr(:))=gather(daz_gridstructure.cumsumpi_z.(jstr(:)));
end
seedpoints=gather(seedpoints);
MoveOutputtoGPU=1;
simoptions.simperiods=gather(simoptions.simperiods);
if simoptions.parallel==2
    simoptions.parallel=1;
    MoveOutputtoGPU=1;
end


l_a=length(n_a_j);
l_z=length(n_z_j);
SimPanel=nan(l_a+l_z+1,simoptions.simperiods,simoptions.numbersims); % (a,z,j)
if simoptions.parallel==0
    for ii=1:simoptions.numbersims
        seedpoint=seedpoints(ii,:);
        SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case2_AgeDepGrids_Dynasty_raw(Phi_of_Policy,Case2_Type,daz_gridstructure, N_j, seedpoint, simoptions.simperiods);
          
        SimPanel_ii=nan(l_a+l_z+1,simoptions.simperiods);
        
        j1=seedpoint(3);
        j2=j1+simoptions.simperiods;
        for t=1:(j2-j1+1)
            jj=rem(t+j1-1,N_j); % This (and following three lines) is the other change for simoptions.dynasty=1
            if jj==0
                jj=N_j;
            end
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
        
        SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case2_AgeDepGrids_Dynasty_raw(Phi_of_Policy,Case2_Type,daz_gridstructure, N_j, seedpoint, simoptions.simperiods);
                
        SimPanel_ii=nan(l_a+l_z+1,simoptions.simperiods);
        
        j1=seedpoint(3);
        j2=j1+simoptions.simperiods-1;
        for t=1:(j2-j1+1)
            jj=rem(t-1+j1,N_j); % This (and following three lines) is the other change for simoptions.dynasty=1
            if jj==0
                jj=N_j;
            end
            jstr=daz_gridstructure.jstr{jj};
            n_a_j=daz_gridstructure.n_a.(jstr(:));
            n_z_j=daz_gridstructure.n_z.(jstr(:));
            temp=SimLifeCycleKron(:,j1+t-1);
            if ~isnan(temp)
                a_c_vec=ind2sub_homemade([n_a_j],temp(1));
                z_c_vec=ind2sub_homemade([n_z_j],temp(2));
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
    % NOT YET IMPLEMENTED THIS
%     cumulativebirthrate=cumprod(simoptions.birthrate.*ones(simoptions.simperiods)+1)-1; % This works for scalar or vector simoptions.birthrate
%     newbirthsvector=gather(round(simoptions.numbersims*cumulativebirthrate)); % Use rounding to decide how many new borns to do each period.
%     BirthDist=gather(simoptions.birthdist);  % Make sure it is not on gpu
%     
%     SimPanel2=nan(l_a+l_z+1,simoptions.simperiods,sum(newbirthsvector));
%     for birthperiod=1:simoptions.simperiods
%         % Get seedpoints from birthdist
%         seedpoints=nan(newbirthsvector(birthperiod),3); % 3 as a,z,j (vectorized)
%         if numel(BirthDist)==N_a*N_z % Has just been given for age j=1
%             cumsumBirthDistVec=cumsum(reshape(BirthDist,[N_a*N_z,1]));
%             [~,seedpointvec]=max(cumsumBirthDistVec>rand(1,numbersims,1));
%             for ii=1:newbirthsvector(birthperiod)
%                 seedpoints(ii,:)=[ind2sub_homemade([N_a,N_z],seedpointvec(ii)),1];
%             end
%         else % Distribution across simoptions.simperiods as well
%             cumsumBirthDistVec=cumsum(reshape(BirthDist,[N_a*N_z*simoptions.simperiods,1]));
%             [~,seedpointvec]=max(cumsumBirthDistVec>rand(1,simoptions.numbersims,1));
%             for ii=1:newbirthsvector(birthperiod)
%                 seedpoints(ii,:)=ind2sub_homemade([N_a,N_z,N_j],seedpointvec(ii));
%             end
%         end
%         seedpoints=floor(seedpoints);  % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.
%     
%         for ii=1:newbirthsvector(birthperiod)
%             seedpoint=seedpoints(ii,:);
%             SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case2_AgeDepGrids_raw(Phi_of_Policy,Case2_Type,daz_gridstructure, N_j, seedpoint, simoptions.simperiods);
% 
%             SimPanel_ii=nan(l_a+l_z+1,simoptions.simperiods);
%             
%             j1=seedpoint(3);
%             j2=min(N_j,j1+(simoptions.simperiods-birthperiod+1));
%             for t=1:(j2-j1+1)
%                 jj=t+j1-1;
%                 temp=SimLifeCycleKron(:,jj);
%                 if ~isnan(temp)
%                     a_c_vec=ind2sub_homemade([n_a],temp(1));
%                     z_c_vec=ind2sub_homemade([n_z],temp(2));
%                     for kk=1:l_a
%                         SimPanel_ii(kk,t)=a_c_vec(kk);
%                     end
%                     for kk=1:l_z
%                         SimPanel_ii(l_a+kk,t)=z_c_vec(kk);
%                     end
%                 end
%                 SimPanel_ii(l_a+l_z+1,t)=jj;
%             end
%             SimPanel2(:,birthperiod:end,sum(newbirthsvector(1:(birthperiod-1)))+ii)=SimPanel_ii;
%         end
%     end
%     SimPanel=[SimPanel;SimPanel2]; % Add the 'new borns' panel to the end of the main panel
end

if MoveOutputtoGPU==1
    SimPanel=gpuArray(SimPanel);
end

end



