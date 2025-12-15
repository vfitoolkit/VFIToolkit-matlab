function SimPanel=SimPanelIndexes_FHorz_ExpAsset_semiz(InitialDist,Policy,n_d,n_a,n_z,N_j,pi_z_J, Parameters, simoptions)
% Inputs should already be on cpu, output is on cpu
% 
% Intended to be called from SimPanelValues_FHorz_Case1()

%%
if ~isfield(simoptions,'l_dsemiz')
    simoptions.l_dsemiz=1;
end

%% Experience asset and semi-exogenous state
n_d3=n_d(end-simoptions.l_dsemiz+1:end); % decision variable that controls semi-exogenous state
n_d2=n_d(end-simoptions.l_dsemiz); % decision variables that controls experience asset
if length(n_d)>2
    n_d1=n_d(1:end-2);
    l_d1=length(n_d1);
else
    % n_d1=0;
    l_d1=0;
end
l_d2=length(n_d2); % wouldn't be here if no d2
l_d3=length(n_d3); % wouldn't be here if no d3

l_d12=l_d1+l_d2;

%% Setup related to experience asset
% Split endogenous assets into the standard ones and the experience asset
if isscalar(n_a)
    n_a1=0;
else
    n_a1=n_a(1:end-1);
end
n_a2=n_a(end); % n_a2 is the experience asset

if ~isfield(simoptions,'aprimeFn')
    error('To use an experience asset you must define simoptions.aprimeFn')
end
if isfield(simoptions,'a_grid')
    a2_grid=simoptions.a_grid(sum(n_a1)+1:end);
else
    error('To use an experience asset you must define simoptions.a_grid')
end
if isfield(simoptions,'d_grid')
    d_grid=simoptions.d_grid;
else
    error('To use an experience asset you must define simoptions.d_grid')
end


% aprimeFnParamNames in same fashion
% l_d2=length(n_d2);
l_a2=length(n_a2);
temp=getAnonymousFnInputNames(simoptions.aprimeFn);
if length(temp)>(l_d2+l_a2)
    aprimeFnParamNames={temp{l_d2+l_a2+1:end}}; % the first inputs will always be (d2,a2)
else
    aprimeFnParamNames={};
end

% Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
simoptions=SemiExogShockSetup_FHorz(n_d,N_j,simoptions.d_grid,Parameters,simoptions,2);
% output: vfoptions.semiz_gridvals_J, vfoptions.pi_semiz_J
% size(semiz_gridvals_J)=[prod(n_z),length(n_z),N_j]
% size(pi_semiz_J)=[prod(n_semiz),prod(n_semiz),prod(n_dsemiz),N_j]
% If no semiz, then vfoptions just does not contain these field


%%
l_d=length(n_d);
l_a=length(n_a);

N_a=prod(n_a);
N_semiz=prod(simoptions.n_semiz);
N_z=prod(n_z);

N_e=prod(simoptions.n_e);
if N_e>0
    l_e=length(simoptions.n_e);
    cumsumpi_e_J=gather(cumsum(simoptions.pi_e_J,1));
end

%%
if N_z==0
    if N_e==0
        N_bothze=N_semiz;
    else
        N_bothze=N_semiz*N_e;
    end
else
    if N_e==0
        N_bothze=N_semiz*N_z;
    else
        N_bothze=N_semiz*N_z*N_e;
    end
end

InitialDist=reshape(InitialDist,[N_a*N_bothze,1]);
Policy=reshape(Policy,[size(Policy,1),N_a,N_bothze,N_j]);


%% expasset transitions
% Policy is currently about d and a1prime. Convert it to being about aprime
% as that is what we need for simulation, and we can then just send it to standard Case1 commands.
Policy_aprime=zeros(N_a,N_bothze,2,N_j,'gpuArray'); % the lower grid point
PolicyProbs=zeros(N_a,N_bothze,2,N_j,'gpuArray'); % The fourth dimension is lower/upper grid point
whichisdforexpasset=length(n_d)-1;  % is just saying which is the decision variable that influences the experience asset (it is the 'second last' decision variable)
for jj=1:N_j
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [aprimeIndexes, aprimeProbs]=CreateaprimePolicyExperienceAsset_Case1(Policy(:,:,:,jj),simoptions.aprimeFn, whichisdforexpasset, n_d, n_a1,n_a2, N_bothze, d_grid, a2_grid, aprimeFnParamsVec);
    % Note: aprimeIndexes and aprimeProbs are both [N_a,N_bothze]
    % Note: aprimeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the aprimeProbs are the probability of this lower point (prob of upper point is just 1 minus this).

    if l_a==1
        Policy_aprime(:,:,1,jj)=aprimeIndexes;
        Policy_aprime(:,:,2,jj)=aprimeIndexes+1;
    elseif l_a==2 % experience asset and one other asset
        Policy_aprime(:,:,1,jj)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(aprimeIndexes-1);
        Policy_aprime(:,:,2,jj)=Policy_aprime(:,:,1,jj)+n_a(1);
    elseif l_a==3 % experience asset and two other assets
        Policy_aprime(:,:,1,jj)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(shiftdim(Policy(l_d+2,:,:,jj),1)-1)+prod(n_a(1:2))*(aprimeIndexes-1);
        Policy_aprime(:,:,2,jj)=Policy_aprime(:,:,1,jj)+prod(n_a(1:2));
    else
        error('Not yet implemented experience asset with length(n_a)>3')
    end
    PolicyProbs(:,:,1,jj)=aprimeProbs;
    PolicyProbs(:,:,2,jj)=1-aprimeProbs;
end

N_probs=2;
if simoptions.gridinterplayer==1
    N_probs=4;
    % (a,z,2,j)
    Policy_aprime=repmat(Policy_aprime,1,1,2,1);
    PolicyProbs=repmat(PolicyProbs,1,1,2,1);
    % Policy_aprime(:,:,1:2,:) lower grid point for a1 is unchanged 
    Policy_aprime(:,:,3:4,:)=Policy_aprime(:,:,3:4,:)+1; % add one to a1, to get upper grid point

    aprimeProbs_upper=reshape(shiftdim((Policy(end,:,:,:)-1)/(simoptions.ngridinterp+1),1),[N_a,N_bothze,1,N_j]); % probability of upper grid point (from L2 index)
    PolicyProbs(:,:,1:2,:)=PolicyProbs(:,:,1:2,:).*(1-aprimeProbs_upper); % lower a1
    PolicyProbs(:,:,3:4,:)=PolicyProbs(:,:,3:4,:).*aprimeProbs_upper; % upper a1
end
CumPolicyProbs=cumsum(PolicyProbs,3);

%% Policy_dsemiexo

% d3 is the variable relevant for the semi-exogenous asset. 
if l_d3==1
    Policy_dsemiexo=Policy(l_d12+1,:,:,:);
elseif l_d3==2
    Policy_dsemiexo=Policy(l_d12+1,:,:,:)+n_d(l_d12+1)*(Policy(l_d12+2,:,:,:)-1);
elseif l_d3==3
    Policy_dsemiexo=Policy(l_d12+1,:,:,:)+n_d(l_d12+1)*(Policy(l_d12+2,:,:,:)-1)+n_d(l_d12+1)*n_d(l_d12+2)*(Policy(l_d12+3,:,:,:)-1); 
elseif l_d3==4
    Policy_dsemiexo=Policy(l_d12+1,:,:,:)+n_d(l_d12+1)*(Policy(l_d12+2,:,:,:)-1)+n_d(l_d12+1)*n_d(l_d12+2)*(Policy(l_d12+3,:,:,:)-1)+n_d(l_d12+1)*n_d(l_d12+2)*n_d(l_d12+3)*(Policy(l_d12+4,:,:,:)-1);
end
Policy_dsemiexo=shiftdim(Policy_dsemiexo,1);


%%
cumsumpi_semiz_J=gather(cumsum(simoptions.pi_semiz_J,2));

if N_z>0
    cumsumpi_z_J=cumsum(pi_z_J,2);
end

%%
simperiods=gather(simoptions.simperiods);

cumsumInitialDistVec=cumsum(InitialDist(:))/sum(InitialDist(:)); % Note: by using (:) I can ignore what the original dimensions were


%% Do the simulations themselves

if N_z==0
    if N_e==0  % No z, No e
        Policy_aprime=reshape(Policy_aprime,[N_a,N_semiz,N_probs,N_j]);
        CumPolicyProbs=reshape(CumPolicyProbs,[N_a,N_semiz,N_probs,N_j]);
        Policy_dsemiexo=reshape(Policy_dsemiexo,[N_a,N_semiz,N_j]);


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
                SimPanel(:,:,ii)=SimLifeCycleIndexes_FHorz_ExpAsset_semiz_noz_raw(Policy_aprime,CumPolicyProbs,Policy_dsemiexo,N_j,cumsumpi_semiz_J, simoptions, seedpoint);
            end
        else
            parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_ExpAsset_semiz_noz_raw(Policy_aprime,CumPolicyProbs,Policy_dsemiexo,N_j,cumsumpi_semiz_J, simoptions, seedpoint);
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

    else % No z, e
        Policy_aprime=reshape(Policy_aprime,[N_a,N_semiz,N_e,N_probs,N_j]);
        CumPolicyProbs=reshape(CumPolicyProbs,[N_a,N_semiz,N_e,N_probs,N_j]);
        Policy_dsemiexo=reshape(Policy_dsemiexo,[N_a,N_semiz,N_e,N_j]);

        % Get seedpoints from InitialDist
        [~,seedpointind]=max(cumsumInitialDistVec>rand(1,simoptions.numbersims)); % will end up with simoptions.numbersims random draws from cumsumInitialDistVec
        if numel(InitialDist)==N_a*N_semiz*N_e % Has just been given for age j=1
            seedpoints=[ind2sub_vec_homemade([N_a,N_semiz,N_e],seedpointind'),ones(simoptions.numbersims,1)];
        else  % Distribution across ages as well
            seedpoints=[ind2sub_vec_homemade([N_a,N_semiz,N_e,N_j],seedpointind'),ones(simoptions.numbersims,1)];
        end
        seedpoints=gather(floor(seedpoints)); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.

        SimPanel=nan(5,simperiods,simoptions.numbersims); % (a,semiz,z,e,j)
        if simoptions.parallel==0
            for ii=1:simoptions.numbersims
                seedpoint=seedpoints(ii,:);
                SimPanel(:,:,ii)=SimLifeCycleIndexes_FHorz_ExpAsset_semiz_noz_e_raw(Policy_aprime,CumPolicyProbs,Policy_dsemiexo,N_j,cumsumpi_semiz_J,cumsumpi_e_J, simoptions, seedpoint);
            end
        else
            parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_ExpAsset_semiz_noz_e_raw(Policy_aprime,CumPolicyProbs,Policy_dsemiexo,N_j,cumsumpi_semiz_J,cumsumpi_e_J, simoptions, seedpoint);
                SimPanel(:,:,ii)=SimLifeCycleKron;
            end
        end

        
        if simoptions.simpanelindexkron==0 % Convert results out of kron
            SimPanelKron=reshape(SimPanel,[4,N_j*simoptions.numbersims]);
            SimPanel=nan(l_a+l_semiz+l_e+1,N_j*simoptions.numbersims); % (a,semiz,e,j)

            SimPanel(1:l_a,:)=ind2sub_homemade(n_a,SimPanelKron(1,:)); % a
            SimPanel(l_a+1:l_a+l_semiz,:)=ind2sub_homemade(n_semiz,SimPanelKron(2,:)); % semiz
            SimPanel(l_a+l_semiz+l_z+1:l_a+l_semiz+l_e,:)=ind2sub_homemade(simoptions.n_e,SimPanelKron(3,:)); % e
            SimPanel(end,:)=SimPanelKron(4,:); % j

            SimPanel=reshape(SimPanel,[4,N_j,simoptions.numbersims]);
        end
    end

else % N_z>0
    if N_e==0  % z, No e
        Policy_aprime=reshape(Policy_aprime,[N_a,N_semiz,N_z,N_probs,N_j]);
        CumPolicyProbs=reshape(CumPolicyProbs,[N_a,N_semiz,N_z,N_probs,N_j]);
        Policy_dsemiexo=reshape(Policy_dsemiexo,[N_a,N_semiz,N_z,N_j]);

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
                SimPanel(:,:,ii)=SimLifeCycleIndexes_FHorz_ExpAsset_semiz_raw(Policy_aprime,CumPolicyProbs,Policy_dsemiexo,N_j,cumsumpi_z_J,cumsumpi_semiz_J, simoptions, seedpoint);
            end
        else
            parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_ExpAsset_semiz_raw(Policy_aprime,CumPolicyProbs,Policy_dsemiexo,N_j,cumsumpi_z_J,cumsumpi_semiz_J, simoptions, seedpoint);
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

            SimPanel=reshape(SimPanel,[4,N_j,simoptions.numbersims]);
        end
        
    else % z, e
        Policy_aprime=reshape(Policy_aprime,[N_a,N_semiz,N_z,N_e,N_probs,N_j]);
        CumPolicyProbs=reshape(CumPolicyProbs,[N_a,N_semiz,N_z,N_e,N_probs,N_j]);
        Policy_dsemiexo=reshape(Policy_dsemiexo,[N_a,N_semiz,N_z,N_e,N_j]);

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
                SimPanel(:,:,ii)=SimLifeCycleIndexes_FHorz_ExpAsset_semiz_e_raw(Policy_aprime,CumPolicyProbs,Policy_dsemiexo,N_j,cumsumpi_z_J,cumsumpi_semiz_J,cumsumpi_e_J, simoptions, seedpoint);
            end
        else
            parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
                seedpoint=seedpoints(ii,:);
                SimLifeCycleKron=SimLifeCycleIndexes_FHorz_ExpAsset_semiz_e_raw(Policy_aprime,CumPolicyProbs,Policy_dsemiexo,N_j,cumsumpi_z_J,cumsumpi_semiz_J,cumsumpi_e_J, simoptions, seedpoint);
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

            SimPanel=reshape(SimPanel,[5,N_j,simoptions.numbersims]);
        end
    end
end




end



