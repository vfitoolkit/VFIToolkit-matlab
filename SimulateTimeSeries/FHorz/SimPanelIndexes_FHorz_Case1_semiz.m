function SimPanel=SimPanelIndexes_FHorz_Case1_semiz(InitialDist,PolicyKron,n_d,n_a,n_z,N_j,cumsumpi_z_J, simoptions)
% Intended to be called from SimPanel=SimPanelIndexes_FHorz_Case1(InitialDist,PolicyKron,n_d,n_a,n_z,N_j,pi_z_J, simoptions)

N_a=prod(n_a);
N_z=prod(n_z);

l_a=length(n_a);
l_z=length(n_z);

simperiods=gather(simoptions.simperiods);

%% Setup related to semi-exogenous state (an exogenous state whose transition probabilities depend on a decision variable)
Parameters=simoptions.Parameters; % Hid a copy here for this purpose :)

if ~isfield(simoptions,'n_semiz')
    error('When using simoptions.SemiExoShockFn you must declare simoptions.n_semiz')
end
if ~isfield(simoptions,'semiz_grid')
    error('When using simoptions.SemiExoShockFn you must declare simoptions.semiz_grid')
end
if ~isfield(simoptions,'d_grid')
    error('When using simoptions.SemiExoShockFn you must declare simoptions.d_grid')
else
    simoptions.d_grid=gpuArray(simoptions.d_grid);
end
if ~isfield(simoptions,'numd_semiz')
    simoptions.numd_semiz=1; % by default, only one decision variable influences the semi-exogenous state
end
if length(n_d)>simoptions.numd_semiz
    n_d1=n_d(1:end-simoptions.numd_semiz);
    d1_grid=simoptions.d_grid(1:sum(n_d1));
else
    n_d1=0; d1_grid=[];
end
N_d1=prod(n_d1);
n_d2=n_d(end-simoptions.numd_semiz+1:end); % n_d2 is the decision variable that influences the transition probabilities of the semi-exogenous state
l_d2=length(n_d2);
d2_grid=simoptions.d_grid(sum(n_d1)+1:end);
% Create the transition matrix in terms of (d,zprime,z) for the semi-exogenous states for each age
N_semiz=prod(simoptions.n_semiz);
l_semiz=length(simoptions.n_semiz);
temp=getAnonymousFnInputNames(simoptions.SemiExoStateFn);
if length(temp)>(l_semiz+l_semiz+l_d2) % This is largely pointless, the SemiExoShockFn is always going to have some parameters
    SemiExoStateFnParamNames={temp{l_semiz+l_semiz+l_d2+1:end}}; % the first inputs will always be (semiz,semizprime,d)
else
    SemiExoStateFnParamNames={};
end
pi_semiz_J=zeros(N_semiz,N_semiz,prod(n_d2),N_j);
for jj=1:N_j
    SemiExoStateFnParamValues=CreateVectorFromParams(Parameters,SemiExoStateFnParamNames,jj);
    pi_semiz_J(:,:,:,jj)=CreatePiSemiZ(n_d2,simoptions.n_semiz,d2_grid,simoptions.semiz_grid,simoptions.SemiExoStateFn,SemiExoStateFnParamValues);
end
cumsum_pi_semiz_J=gather(cumsum(pi_semiz_J,2));

%%
simperiods=gather(simoptions.simperiods);
numbersims=gather(simoptions.numbersims); % This is just to deal with weird error that matlab decided simoptions.numbersims was on gpu and so couldn't be an input to rand()

cumsumInitialDistVec=cumsum(InitialDist(:))/sum(InitialDist(:)); % Note: by using (:) I can ignore what the original dimensions were

%% First do the case without e variables, otherwise do with e variables
if ~isfield(simoptions,'n_e')
    
    % Get seedpoints from InitialDist
    [~,seedpointind]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims)); % will end up with simoptions.numbersims random draws from cumsumInitialDistVec
    if numel(InitialDist)==N_a*N_semiz*N_z % Has just been given for age j=1
        seedpoints=[ind2sub_vec_homemade([N_a,N_semiz,N_z],seedpointind'),ones(simoptions.numbersims,1)];
    else  % Distribution across ages as well
        seedpoints=[ind2sub_vec_homemade([N_a,N_semiz,N_z,N_j],seedpointind'),ones(simoptions.numbersims,1)];
    end
    seedpoints=gather(floor(seedpoints)); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.
    
    SimPanel=nan(4,simperiods,simoptions.numbersims); % (a,semiz,z,j)
    if simoptions.parallel==0
        for ii=1:simoptions.numbersims
            seedpoint=seedpoints(ii,:);
            SimPanel(:,:,ii)=SimLifeCycleIndexes_FHorz_Case1_semiz_raw(PolicyKron,N_d1,N_j,cumsumpi_z_J,cumsumpi_semiz_J, seedpoint, simperiods);
        end
    else
        parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
            seedpoint=seedpoints(ii,:);
            SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_semiz_raw(PolicyKron,N_d1,N_j,cumsumpi_z_J,cumsumpi_semiz_J, seedpoint, simperiods);
            SimPanel(:,:,ii)=SimLifeCycleKron;
        end
    end

    if simoptions.simpanelindexkron==0 % Convert results out of kron
        SimPanelKron=reshape(SimPanel,[4,N_j*simoptions.numbersims]);
        SimPanel=nan(l_a+l_semiz+l_z+1,N_j*simoptions.numbersims); % (a,semiz,z,j)
        
        SimPanel(1:l_a,:)=ind2sub_homemade(n_a,SimPanelKron(1,:)); % a
        SimPanel(l_a+1:l_a+l_semiz,:)=ind2sub_homemade(n_semiz,SimPanelKron(2,:)); % semiz
        SimPanel(l_a+l_semiz+1:l_a+l_semiz+l_z,:)=ind2sub_homemade(n_z,SimPanelKron(3,:)); % z
        SimPanel(end,:)=SimPanelKron(4,:); % j
    end

    
else %if isfield(simoptions,'n_e')
    %% this time with e variables
    % If e variables are used they are treated seperately as this is faster/better
    N_e=prod(simoptions.n_e);
    l_e=length(simoptions.n_e);
    
    cumsumpi_e_J=gather(cumsum(simoptions.pi_e_J,1));
    
    % Get seedpoints from InitialDist
    [~,seedpointind]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims)); % will end up with simoptions.numbersims random draws from cumsumInitialDistVec
    if numel(InitialDist)==N_a*N_semiz*N_z*N_e % Has just been given for age j=1
        seedpoints=[ind2sub_vec_homemade([N_a,N_semiz,N_z,N_e],seedpointind'),ones(simoptions.numbersims,1)];
    else  % Distribution across ages as well
        seedpoints=[ind2sub_vec_homemade([N_a,N_semiz,N_z,N_e,N_j],seedpointind'),ones(simoptions.numbersims,1)];
    end
    seedpoints=gather(floor(seedpoints)); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.
    
    SimPanel=nan(5,simperiods,simoptions.numbersims); % (a,semiz,z,e,j)
    if simoptions.parallel==0
        for ii=1:simoptions.numbersims
            seedpoint=seedpoints(ii,:);
            SimPanel(:,:,ii)=SimLifeCycleIndexes_FHorz_Case1_semiz_e_raw(PolicyKron,N_d1,N_j,cumsumpi_z_J,cumsumpi_semiz_J,cumsumpi_e_J, seedpoint, simperiods);
        end
    else
        parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
            seedpoint=seedpoints(ii,:);
            SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_semiz_e_raw(PolicyKron,N_d1,N_j,cumsumpi_z_J,cumsumpi_semiz_J,cumsumpi_e_J, seedpoint, simperiods);
            SimPanel(:,:,ii)=SimLifeCycleKron;
        end
    end
    

    if simoptions.simpanelindexkron==0 % Convert results out of kron
        SimPanelKron=reshape(SimPanel,[5,N_j*simoptions.numbersims]);
        SimPanel=nan(l_a+l_semiz+l_z+l_e+1,N_j*simoptions.numbersims); % (a,semiz,z,e,j)
        
        SimPanel(1:l_a,:)=ind2sub_homemade(n_a,SimPanelKron(1,:)); % a
        SimPanel(l_a+1:l_a+l_semiz,:)=ind2sub_homemade(n_semiz,SimPanelKron(2,:)); % semiz
        SimPanel(l_a+l_semiz+1:l_a+l_semiz+l_z,:)=ind2sub_homemade(n_z,SimPanelKron(3,:)); % z
        SimPanel(l_a+l_semiz+l_z+1:l_a+l_semiz+l_z+l_e,:)=ind2sub_homemade(simoptions.n_e,SimPanelKron(4,:)); % e
        SimPanel(end,:)=SimPanelKron(5,:); % j
    end

end

if MoveOutputtoGPU==1
    SimPanel=gpuArray(SimPanel);
end

end



