function SimPanel=SimPanelIndexes_FHorz_Case1_noz_semiz_aggshock(InitialDist,PolicyKron,n_d,n_a,N_j, simoptions)
% Intended to be called from SimPanel=SimPanelIndexes_FHorz_Case1(InitialDist,PolicyKron,n_d,n_a,n_z,N_j,pi_z_J, simoptions)

N_a=prod(n_a);

l_a=length(n_a);

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
% cumsumpi_semiz_J=gather(cumsum(pi_semiz_J,2));

%
n_semiz=simoptions.n_semiz;
N_d2=prod(n_d2);

%% Aggregate shock is assumed to be the last semiz
n_aggz=n_semiz(end);
N_aggz=prod(n_aggz);
n_semiz_idio=n_semiz(1:end-1);
N_semiz_idio=prod(n_semiz_idio);



%% Aggregate shock sequence (indices)
pi_aggz=pi_semiz_J(:,:,1,1); % Which (d,j) we use should be irrelevant to agg shocks, so just use (1,1)
pi_aggz=reshape(pi_aggz,[N_semiz_idio,N_aggz,N_semiz_idio,N_aggz]);
pi_aggz=sum(pi_aggz(1,:,:,:),3); % sum across next period idiosyncratic shocks; use an arbitrary one of the idiosyncratic shocks for this period (the first)
pi_aggz=reshape(pi_aggz,[N_aggz,N_aggz]);
cumsum_pi_aggz=cumsum(pi_aggz,2);
% Remark: Might be a good idea to add a test here that we get the same
% thing with a different J (or more generally that the aggregate shocks
% actually are independent of the idiosyncratic shocks)

burnin=100;
if ~isfield(simoptions,'AggShockIndex')
    % Need to simulate an aggregate shock sequence
    aggz_c=1+floor(n_aggz*rand(1,1));
    for bb=1:burnin
        [~,aggz_c]=max(cumsum_pi_aggz(aggz_c,:)>rand(1,1));
    end
    AggShockIndex=zeros(1,simperiods,'gpuArray');
    for tt=1:simperiods
        [~,aggz_c]=max(cumsum_pi_aggz(aggz_c,:)>rand(1,1));
        AggShockIndex(tt)=aggz_c;
    end
else
    AggShockIndex=simoptions.AggShockIndex;
end

%% Set up Policy and pi_semiz_J that have the aggregate shock in last dimension so as it is easy so get the right one for each time period
pi_semiz_idio_J=reshape(pi_semiz_J,[N_semiz_idio,N_aggz,N_semiz_idio,N_aggz,N_d2,N_j]);
pi_semiz_idio_J=permute(pi_semiz_idio_J,[1,3,2,4,5,6]); % (semiz_idio, semiz_idioprime,aggz,aggzprime,d2,j)
% Need to normalize so that transition probabilies are one conditional on aggzprime
pi_semiz_idio_J=pi_semiz_idio_J./sum(pi_semiz_idio_J,2);

PolicyKron=reshape(PolicyKron,[size(PolicyKron,1),N_a,N_semiz_idio,N_aggz,N_j]);
PolicyKron=permute(PolicyKron,[1,2,3,5,4]);

cumsumpi_semiz_idio_J=gather(cumsum(pi_semiz_idio_J,2));

%%
% simperiods=gather(simoptions.simperiods);
% numbersims=gather(simoptions.numbersims); % This is just to deal with weird error that matlab decided simoptions.numbersims was on gpu and so couldn't be an input to rand()

cumsumInitialDistVec=cumsum(InitialDist(:))/sum(InitialDist(:)); % Note: by using (:) I can ignore what the original dimensions were

if numel(InitialDist)==N_a*N_semiz_idio % Has just been given for age j=1
elseif numel(InitialDist)==N_a*N_semiz_idio*N_j % Distribution across ages as well
else
    error('InitialDist is not of a recognized size (should be (a,z,j) or (a,z), should not depend on aggregate shock)')
end

%% First do the case without e variables, otherwise do with e variables
if ~isfield(simoptions,'n_e')
    
    SimPanel=nan(3,N_j,simoptions.numbersims); % (a,semiz,j)
    if simoptions.parallel==0
        for ii=1:simoptions.numbersims
            % Get seedpoint from InitialDist
            [~,seedpointind]=max(cumsumInitialDistVec>rand(1,1)); % will end up with simoptions.numbersims random draws from cumsumInitialDistVec
            if numel(InitialDist)==N_a*N_semiz_idio % Has just been given for age j=1
                seedpoint=[ind2sub_homemade([N_a,N_semiz_idio],seedpointind),1];
            elseif numel(InitialDist)==N_a*N_semiz_idio*N_j % Distribution across ages as well
                seedpoint=ind2sub_homemade([N_a,N_semiz_idio,N_j],seedpointind);
            end
            seedpoint=gather(floor(seedpoint)); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.

            SimPanel(:,:,ii)=SimLifeCycleIndexes_FHorz_Case1_noz_semiz_aggshock_raw(PolicyKron,AggShockIndex,N_d1,N_semiz_idio,N_j,cumsumpi_semiz_idio_J, seedpoint, simperiods);
        end
    else
        parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
            % Get seedpoint from InitialDist
            [~,seedpointind]=max(cumsumInitialDistVec>rand(1,1)); % will end up with simoptions.numbersims random draws from cumsumInitialDistVec
            if numel(InitialDist)==N_a*N_semiz_idio % Has just been given for age j=1
                seedpoint=[ind2sub_homemade([N_a,N_semiz_idio],seedpointind),1];
            elseif numel(InitialDist)==N_a*N_semiz_idio*N_j % Distribution across ages as well
                seedpoint=ind2sub_homemade([N_a,N_semiz_idio,N_j],seedpointind);
            end
            seedpoint=gather(floor(seedpoint)); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.

            SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_noz_semiz_aggshock_raw(PolicyKron,AggShockIndex,N_d1,N_semiz_idio,N_j,cumsumpi_semiz_idio_J, seedpoint, simperiods);

            SimPanel(:,:,ii)=SimLifeCycleKron;
        end
    end


    if simoptions.simpanelindexkron==0 % Convert results out of kron
        SimPanelKron=reshape(SimPanel,[3,N_j*simoptions.numbersims]);
        SimPanel=nan(l_a+l_semiz+1,N_j*simoptions.numbersims); % (a,semiz,j)
        
        SimPanel(1:l_a,:)=ind2sub_homemade(n_a,SimPanelKron(1,:)); % a
        SimPanel(l_a+1:l_a+l_semiz,:)=ind2sub_homemade(n_semiz,SimPanelKron(2,:)); % semiz
        SimPanel(end,:)=SimPanelKron(3,:); % j
    end
        
    
else %if isfield(simoptions,'n_e')
    error('Not yet implemented')
    % %% this time with e variables
    % % If e variables are used they are treated seperately as this is faster/better
    % N_e=prod(simoptions.n_e);
    % l_e=length(simoptions.n_e);
    % 
    % cumsumpi_e_J=gather(cumsum(simoptions.pi_e_J,1));
    % 
    % % Get seedpoints from InitialDist
    % [~,seedpointind]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims)); % will end up with simoptions.numbersims random draws from cumsumInitialDistVec
    % if numel(InitialDist)==N_a*N_semiz*N_e % Has just been given for age j=1
    %     seedpoints=[ind2sub_vec_homemade([N_a,N_semiz,N_e],seedpointind'),ones(simoptions.numbersims,1)];
    % else  % Distribution across ages as well
    %     seedpoints=[ind2sub_vec_homemade([N_a,N_semiz,N_e,N_j],seedpointind'),ones(simoptions.numbersims,1)];
    % end
    % seedpoints=gather(floor(seedpoints)); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.
    % 
    % SimPanel=nan(4,simperiods,simoptions.numbersims); % (a,semiz,e,j)
    % if simoptions.parallel==0
    %     for ii=1:simoptions.numbersims
    %         seedpoint=seedpoints(ii,:);
    %         SimPanel(:,:,ii)=SimLifeCycleIndexes_FHorz_Case1_noz_semiz_e_raw(PolicyKron,N_d1,N_j,cumsumpi_semiz_J,cumsumpi_e_J, seedpoint, simperiods);
    %     end
    % else
    %     parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
    %         seedpoint=seedpoints(ii,:);
    %         SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_noz_semiz_e_raw(PolicyKron,N_d1,N_j,cumsumpi_semiz_J,cumsumpi_e_J, seedpoint, simperiods);
    %         SimPanel(:,:,ii)=SimLifeCycleKron;
    %     end
    % end
    % 
    % if simoptions.simpanelindexkron==0 % Convert results out of kron
    %     SimPanelKron=reshape(SimPanel,[4,N_j*simoptions.numbersims]);
    %     SimPanel=nan(l_a+l_semiz+l_e+1,N_j*simoptions.numbersims); % (a,semiz,e,j)
    % 
    %     SimPanel(1:l_a,:)=ind2sub_homemade(n_a,SimPanelKron(1,:)); % a
    %     SimPanel(l_a+1:l_a+l_semiz,:)=ind2sub_homemade(n_semiz,SimPanelKron(2,:)); % semiz
    %     SimPanel(l_a+l_semiz+1:l_a+l_semiz+l_e,:)=ind2sub_homemade(simoptions.n_e,SimPanelKron(3,:)); % e
    %     SimPanel(end,:)=SimPanelKron(4,:); % j
    % end

end


end



