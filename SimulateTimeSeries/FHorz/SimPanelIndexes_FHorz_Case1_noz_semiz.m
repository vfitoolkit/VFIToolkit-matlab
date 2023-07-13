function SimPanel=SimPanelIndexes_FHorz_Case1_noz_semiz(InitialDist,Policy,n_d,n_a,N_j, simoptions)
% Simulates a panel based on PolicyIndexes of 'numbersims' agents of length
% 'simperiods' beginning from randomly drawn InitialDist. (If you use the
% newbirths option you will get more than 'numbersims', due to the extra births)
%
% InitialDist can be inputed as over the finite time-horizon (j), or
% without a time-horizon in which case it is assumed to be an InitialDist
% for time j=1. (So InitialDist is either n_a-by-n_z-by-n_j, or n_a-by-n_z)

N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_d=prod(n_d);

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
        simoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
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
    simoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
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

%% Check the semiz options are all set
if ~isfield(simoptions,'semiz_grid')
    error('You have simoptions.n_semiz but have not setup simoptions.semiz_grid')
elseif ~isfield(simoptions,'pi_semiz')
    error('You have simoptions.n_semiz but have not setup simoptions.pi_semiz')
end

%% Setup related to semi-exogenous state (an exogenous state whose transition probabilities depend on a decision variable)
if ~isfield(simoptions,'n_semiz')
    error('When using simoptions.SemiExoShockFn you must declare simoptions.n_semiz')
end
if ~isfield(simoptions,'semiz_grid')
    error('When using simoptions.SemiExoShockFn you must declare simoptions.semiz_grid')
end
n_d1=n_d(1:end-1);
n_d2=n_d(end); % n_d2 is the decision variable that influences the transition probabilities of the semi-exogenous state
% d1_grid=simoptions.d_grid(1:sum(n_d1));
d2_grid=gpuArray(simoptions.d_grid(sum(n_d1)+1:end));
% Create the transition matrix in terms of (d,zprime,z) for the semi-exogenous states for each age
l_semiz=length(simoptions.n_semiz);
temp=getAnonymousFnInputNames(simoptions.SemiExoStateFn);
if length(temp)>(1+l_semiz+l_semiz) % This is largely pointless, the SemiExoShockFn is always going to have some parameters
    SemiExoStateFnParamNames={temp{1+l_semiz+l_semiz+1:end}}; % the first inputs will always be (d,semizprime,semiz)
else
    SemiExoStateFnParamNames={};
end
n_semiz=simoptions.n_semiz;
N_semiz=prod(n_semiz);
pi_semiz_J=zeros(N_semiz,N_semiz,n_d2,N_j);
for jj=1:N_j
    SemiExoStateFnParamValues=CreateVectorFromParams(Parameters,SemiExoStateFnParamNames,jj);
    pi_semiz_J(:,:,:,jj)=CreatePiSemiZ(n_d2,simoptions.n_semiz,d2_grid,simoptions.semiz_grid,simoptions.SemiExoStateFn,SemiExoStateFnParamValues);
end
cumsum_pi_semiz_J=gather(cumsum(pi_semiz_J,2));

%%
MoveOutputtoGPU=0;
if simoptions.parallel==2
    % Simulation on GPU is really slow. So instead, switch to CPU, and then switch
    % back. For anything but ridiculously short simulations it is more than worth the overhead.
    Policy=gather(Policy);
    MoveOutputtoGPU=1;
end

simperiods=gather(simoptions.simperiods);

%% First do the case without e variables, otherwise do with e variables
if ~isfield(simoptions,'n_e')
    
    Policy=KronPolicyIndexes_FHorz_Case1_noz_semiz(Policy, n_d1, n_d2, n_a, n_semiz, N_j);
    
    InitialDist=gather(InitialDist); % Make sure it is not on gpu
    numbersims=gather(simoptions.numbersims); % This is just to deal with weird error that matlab decided simoptions.numbersims was on gpu and so couldn't be an input to rand()
    % Get seedpoints from InitialDist
    seedpoints=nan(simoptions.numbersims,3); % 3 as a,semiz,j (vectorized)
    if numel(InitialDist)==N_a*N_semiz % Has just been given for age j=1
        cumsumInitialDistVec=cumsum(reshape(InitialDist,[N_a*N_semiz,1]));
        parfor ii=1:simoptions.numbersims
            seedpointvec=max(cumsumInitialDistVec>rand(1));
            seedpoints(ii,:)=[ind2sub_homemade([N_a,N_semiz],seedpointvec),1];
        end
    else % Distribution across ages as well
        cumsumInitialDistVec=cumsum(reshape(InitialDist,[N_a*N_semiz*N_j,1]));
        parfor ii=1:simoptions.numbersims
            seedpointvec=max(cumsumInitialDistVec>rand(1));
            seedpoints(ii,:)=ind2sub_homemade([N_a,N_semiz,N_j],seedpointvec);
        end
    end
    seedpoints=floor(seedpoints); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.
    
    
    if simoptions.simpanelindexkron==0 % For some VFI Toolkit commands the kron is faster to use
        SimPanel=nan(l_a+l_semiz+1,simperiods,simoptions.numbersims); % (a,semiz,j)
        if simoptions.parallel==0
            for ii=1:simoptions.numbersims
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_noz_semiz_raw(Policy,N_d1,N_d2,N_a,N_semiz,N_j,cumsumpi_semiz_J, seedpoint, simperiods);
                
                SimPanel_ii=nan(l_a+l_semiz+1,simperiods);
                
                j1=seedpoint(3); % 3 as j in (a,semiz,j)
                j2=min(N_j,j1+simperiods);
                for t=1:(j2-j1+1)
                    jj=t+j1-1;
                    temp=SimLifeCycleKron(:,jj);
                    if ~isnan(temp)
                        a_c_vec=ind2sub_homemade(n_a,temp(1));
                        semiz_c_vec=ind2sub_homemade(n_semiz,temp(2));
                        for kk=1:l_a
                            SimPanel_ii(kk,t)=a_c_vec(kk);
                        end
                        for kk=1:l_semiz
                            SimPanel_ii(l_a+kk,t)=semiz_c_vec(kk);
                        end
                    end
                    SimPanel_ii(l_a+l_semiz+1,t)=jj;
                end
                SimPanel(:,:,ii)=SimPanel_ii;
            end
        else
            parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_noz_semiz_raw(Policy,N_d1,N_d2,N_a,N_semiz,N_j,cumsumpi_semiz_J, seedpoint, simperiods);
                
                SimPanel_ii=nan(l_a+l_semiz+1,simperiods);
                
                j1=seedpoint(3); % 3 as j in (a,semiz,j)
                j2=min(N_j,j1+simperiods);
                for t=1:(j2-j1+1)
                    jj=t+j1-1;
                    temp=SimLifeCycleKron(:,jj);
                    if ~isnan(temp)
                        a_c_vec=ind2sub_homemade(n_a,temp(1));
                        semiz_c_vec=ind2sub_homemade(n_semiz,temp(2));
                        for kk=1:l_a
                            SimPanel_ii(kk,t)=a_c_vec(kk);
                        end
                        for kk=1:l_semiz
                            SimPanel_ii(l_a+kk,t)=semiz_c_vec(kk);
                        end
                    end
                    SimPanel_ii(l_a+l_semiz+1,t)=jj;
                end
                SimPanel(:,:,ii)=SimPanel_ii;
            end
        end
        
        
    else % simoptions.simpanelindexkron==1 % For some VFI Toolkit commands the kron is faster to use
        SimPanel=nan(4,simperiods,simoptions.numbersims); % (a,z,semiz,j)
        if simoptions.parallel==0
            for ii=1:simoptions.numbersims
                seedpoint=seedpoints(ii,:);
                SimPanel(:,:,ii)=SimLifeCycleIndexes_FHorz_Case1_noz_semiz_raw(Policy,N_d1,N_d2,N_a,N_semiz,N_j,cumsumpi_semiz_J, seedpoint, simperiods);
            end
        else
            parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_noz_semiz_raw(Policy,N_d1,N_d2,N_a,N_semiz,N_j,cumsumpi_semiz_J, seedpoint, simperiods);
                SimPanel(:,:,ii)=SimLifeCycleKron;
            end
        end
        
    end
    
else %if isfield(simoptions,'n_e')
    %% this time with e variables
    % If e variables are used they are treated seperately as this is faster/better
    eval('fieldexists_EiidShockFn=1;simoptions.EiidShockFn;','fieldexists_EiidShockFn=0;')
    eval('fieldexists_EiidShockFnParamNames=1;simoptions.EiidShockFnParamNames;','fieldexists_EiidShockFnParamNames=0;')
    eval('fieldexists_pi_e_J=1;simoptions.pi_e_J;','fieldexists_pi_e_J=0;')
    
    N_e=prod(simoptions.n_e);
    l_e=length(simoptions.n_e);
    
    if fieldexists_pi_e_J==1
        cumsumpi_e_J=gather(cumsum(simoptions.pi_e_J,1));
    elseif fieldexists_EiidShockFn==1
        cumsumpi_e_J=nan(N_e,N_j);
        for jj=1:N_j
            if fieldexists_EiidShockFnParamNames==1
                EiidShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.EiidShockFnParamNames,jj);
                EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                for kk=1:length(EiidShockFnParamsVec)
                    EiidShockFnParamsCell(kk,1)={EiidShockFnParamsVec(kk)};
                end
                [~,pi_e_jj]=simoptions.EiidShockFn(EiidShockFnParamsCell{:});
            else
                [~,pi_e_jj]=simoptions.EiidShockFn(jj);
            end
            cumsumpi_e_J(:,jj)=gather(cumsum(pi_e_jj));
        end
    else
        cumsumpi_e_J=gather(cumsum(simoptions.pi_e)).*ones(1,N_j);
    end
    
    Policy=KronPolicyIndexes_FHorz_Case1_noz_semiz(Policy, n_d1, n_d2, n_a, n_semiz, N_j, simoptions.n_e);
    
    InitialDist=gather(InitialDist); % Make sure it is not on gpu
    numbersims=gather(simoptions.numbersims); % This is just to deal with weird error that matlab decided simoptions.numbersims was on gpu and so couldn't be an input to rand()
    % Get seedpoints from InitialDist
    seedpoints=nan(simoptions.numbersims,4); % 4 as a,semiz,e,j (vectorized)
    if numel(InitialDist)==N_a*N_semiz*N_e % Has just been given for age j=1
        cumsumInitialDistVec=cumsum(reshape(InitialDist,[N_a*N_semiz*N_e,1]));
        parfor ii=1:simoptions.numbersims
            [~,seedpointvec]=max(cumsumInitialDistVec>rand(1));
            seedpoints(ii,:)=[ind2sub_homemade([N_a,N_semiz,N_e],seedpointvec),1];
        end
    else % Distribution across ages as well
        cumsumInitialDistVec=cumsum(reshape(InitialDist,[N_a*N_semiz*N_e*N_j,1]));
        parfor ii=1:simoptions.numbersims
            seedpointvec=max(cumsumInitialDistVec>rand(1));
            seedpoints(ii,:)=ind2sub_homemade([N_a,N_semiz,N_e,N_j],seedpointvec);
        end
    end
    seedpoints=floor(seedpoints); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.
    
    if simoptions.simpanelindexkron==0 % For some VFI Toolkit commands the kron is faster to use
        SimPanel=nan(l_a+l_semiz+l_e+1,simperiods,simoptions.numbersims); % (a,semiz,e,j)
        if simoptions.parallel==0
            for ii=1:simoptions.numbersims
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_noz_semiz_e_raw(Policy,N_d1,N_d2,N_a,N_semiz,N_e,N_j,cumsumpi_semiz_J,cumsumpi_e_J, seedpoint, simperiods);
                
                SimPanel_ii=nan(l_a+l_semiz+l_e+1,simperiods);
                
                j1=seedpoint(4); % 4 as j in (a,semiz,e,j)
                j2=min(N_j,j1+simperiods);
                for t=1:(j2-j1+1)
                    jj=t+j1-1;
                    temp=SimLifeCycleKron(:,jj);
                    if ~isnan(temp)
                        a_c_vec=ind2sub_homemade(n_a,temp(1));
                        semiz_c_vec=ind2sub_homemade(n_semiz,temp(2));
                        e_c_vec=ind2sub_homemade(n_e,temp(3));
                        for kk=1:l_a
                            SimPanel_ii(kk,t)=a_c_vec(kk);
                        end
                        for kk=1:l_semiz
                            SimPanel_ii(l_a+kk,t)=semiz_c_vec(kk);
                        end
                        for kk=1:l_e
                            SimPanel_ii(l_a+l_semiz+kk,t)=e_c_vec(kk);
                        end
                        SimPanel_ii(l_a+l_semiz+l_e+1,t)=jj; % Note: temp(5) is jj, but no need to actually access it
                    end
                end
                SimPanel(:,:,ii)=SimPanel_ii;
            end
        else
            parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0                
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_noz_semiz_e_raw(Policy,N_d1,N_d2,N_a,N_semiz,N_e,N_j,cumsumpi_semiz_J,cumsumpi_e_J, seedpoint, simperiods);
                
                SimPanel_ii=nan(l_a+l_semiz+l_e+1,simperiods);
                
                j1=seedpoint(4); % 4 as j in (a,semiz,e,j)
                j2=min(N_j,j1+simperiods);
                for t=1:(j2-j1+1)
                    jj=t+j1-1;
                    temp=SimLifeCycleKron(:,jj);
                    if ~isnan(temp)
                        a_c_vec=ind2sub_homemade(n_a,temp(1));
                        semiz_c_vec=ind2sub_homemade(n_semiz,temp(2));
                        e_c_vec=ind2sub_homemade(n_e,temp(3));
                        for kk=1:l_a
                            SimPanel_ii(kk,t)=a_c_vec(kk);
                        end
                        for kk=1:l_semiz
                            SimPanel_ii(l_a+kk,t)=semiz_c_vec(kk);
                        end
                        for kk=1:l_e
                            SimPanel_ii(l_a+l_semiz+kk,t)=e_c_vec(kk);
                        end
                        SimPanel_ii(l_a+l_semiz+l_e+1,t)=jj;  % Note: temp(5) is jj, but no need to actually access it
                    end
                end
                SimPanel(:,:,ii)=SimPanel_ii;
            end
        end
        
        
    else % simoptions.simpanelindexkron==1 % For some VFI Toolkit commands the kron is faster to use
        
        SimPanel=nan(4,simperiods,simoptions.numbersims); % (a,semiz,e,j)
        if simoptions.parallel==0
            for ii=1:simoptions.numbersims
                seedpoint=seedpoints(ii,:);
                SimPanel(:,:,ii)=SimLifeCycleIndexes_FHorz_Case1_noz_semiz_e_raw(Policy,N_d1,N_d2,N_a,N_semiz,N_e,N_j,cumsumpi_semiz_J,cumsumpi_e_J, seedpoint, simperiods);
            end
        else
            parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_noz_semiz_e_raw(Policy,N_d1,N_d2,N_a,N_semiz,N_e,N_j,cumsumpi_semiz_J,cumsumpi_e_J, seedpoint, simperiods);
                SimPanel(:,:,ii)=SimLifeCycleKron; 
            end
        end
                
    end
end

if MoveOutputtoGPU==1
    SimPanel=gpuArray(SimPanel);
end

end



