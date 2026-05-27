function SimPanel=SimPanelIndexes_FHorz_RiskyAsset(InitialDist,Policy,n_d,n_a,n_z,N_j,pi_z_J, Parameters, simoptions)
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
    cumsumpi_z_J=gather(cumsum(pi_z_J,2));
else
    l_z=0;
end
N_e=prod(simoptions.n_e);
if N_e==0
    l_e=0;
else
    l_e=length(simoptions.n_e);
    cumsumpi_e_J=gather(cumsum(simoptions.pi_e_J,1));
end

cumsumInitialDistVec=cumsum(InitialDist(:))/sum(InitialDist(:)); % Note: by using (:) I can ignore what the original dimensions were

%% Setup related to risky asset
if ~isfield(simoptions,'aprimeFn')
    error('To use a risky asset you must define simoptions.aprimeFn')
end
if ~isfield(simoptions,'a_grid')
    error('To use a risky asset you must define simoptions.a_grid')
end
if ~isfield(simoptions,'d_grid')
    error('To use a risky asset you must define simoptions.d_grid')
end

% Sort out decision variables, need to get those for riskyasset
if ~isfield(simoptions,'refine_d')
    % When not using refine_d, everything is implicitly a d3 [in both aprimeFn and ReturnFn; note that here only aprimeFn matters anyway]
    simoptions.refine_d=[0,0,length(n_d)];
end
n_d23=n_d(simoptions.refine_d(1)+1:sum(simoptions.refine_d(1:3))); % decision variables for riskyasset

% Split endogenous assets into the standard ones and the risky asset
if isscalar(n_a)
    n_a1=0;
else
    n_a1=n_a(1:end-1);
end
n_a2=n_a(end); % n_a2 is the risky asset
a2_grid=simoptions.a_grid(sum(n_a1)+1:end);
d_grid=simoptions.d_grid;

%%
if ~isfield(simoptions,'n_u')
    error('To use a risky asset you must define simoptions.n_u')
end
if ~isfield(simoptions,'u_grid')
    error('To use a risky asset you must define simoptions.u_grid')
end
if ~isfield(simoptions,'pi_u')
    error('To use a risky asset you must define simoptions.pi_u')
end
% to evaluate the aprimeFn we need the grids on gpu
n_u=simoptions.n_u;
u_grid=gpuArray(simoptions.u_grid);
pi_u=gpuArray(simoptions.pi_u);
N_u=prod(n_u);

%% aprimeFnParamNames: aprimeFn takes (d, u, ...)
l_u=length(n_u);
l_d23=length(n_d23);
temp=getAnonymousFnInputNames(simoptions.aprimeFn);
if length(temp)>(l_d23+l_u)
    aprimeFnParamNames={temp{l_d23+l_u+1:end}}; % the first inputs will always be (d,u)
else
    aprimeFnParamNames={};
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
    Policy=reshape(Policy,[size(Policy,1),N_a,1,N_j]);
else
    Policy=reshape(Policy,[size(Policy,1),N_a,N_bothze,N_j]);
end

%% riskyasset transitions
Policy_aprime=zeros(N_a,max(1,N_bothze),N_u,2,N_j,'gpuArray'); % the lower grid point
PolicyProbs=zeros(N_a,max(1,N_bothze),N_u,2,N_j,'gpuArray'); % probabilities of grid points
whichisdforriskyasset=(simoptions.refine_d(1)+1):1:length(n_d);  % is just saying which is the decision variable that influences the risky asset (namely, d2 and d3 both do)
for jj=1:N_j
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [aprimeIndexes, aprimeProbs]=CreateaprimePolicyRiskyAsset(Policy(1:l_d,:,:,jj),simoptions.aprimeFn, whichisdforriskyasset, n_d, n_a1,n_a2, N_bothze, n_u, d_grid, a2_grid, u_grid, aprimeFnParamsVec);
    % Note: aprimeIndexes and aprimeProbs are both [N_a,max(1,N_bothze),N_u]
    % Note: aprimeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the aprimeProbs are the probability of this lower point (prob of upper point is just 1 minus this).

    if l_a==1 % just riskyasset
        Policy_aprime(:,:,:,1,jj)=aprimeIndexes;
        Policy_aprime(:,:,:,2,jj)=aprimeIndexes+1;
    elseif l_a==2 % one other asset, then riskyasset
        Policy_aprime(:,:,:,1,jj)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(aprimeIndexes-1);
        Policy_aprime(:,:,:,2,jj)=Policy_aprime(:,:,:,1,jj)+n_a(1);
    elseif l_a==3 % two other assets, then riskyasset
        Policy_aprime(:,:,:,1,jj)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(shiftdim(Policy(l_d+2,:,:,jj),1)-1)+n_a(1)*n_a(2)*(aprimeIndexes-1);
        Policy_aprime(:,:,:,2,jj)=Policy_aprime(:,:,:,1,jj)+n_a(1)*n_a(2);
    else
        error('Not yet implemented riskyasset with length(n_a)>3')
    end

    % Encode the u probabilities (pi_u) into the PolicyProbs
    PolicyProbs(:,:,:,1,jj)=aprimeProbs.*shiftdim(pi_u,-2); % lower grid point probability (and probability of u)
    PolicyProbs(:,:,:,2,jj)=(1-aprimeProbs).*shiftdim(pi_u,-2); % upper grid point probability (and probability of u)
end

Policy_aprime=reshape(Policy_aprime,[N_a,max(1,N_bothze),N_u*2,N_j]);
PolicyProbs=reshape(PolicyProbs,[N_a,max(1,N_bothze),N_u*2,N_j]);

N_probs=N_u*2;
if simoptions.gridinterplayer==1
    N_probs=2*N_u*2;
    % (a,z,N_u*2,j)
    Policy_aprime=repmat(Policy_aprime,1,1,2,1);
    PolicyProbs=repmat(PolicyProbs,1,1,2,1);
    % Policy_aprime(:,:,1:N_u*2,:) lower grid point for a1 is unchanged
    Policy_aprime(:,:,N_u*2+1:end,:)=Policy_aprime(:,:,N_u*2+1:end,:)+1; % add one to a1, to get upper grid point

    aprimeProbs_upper=reshape(shiftdim((Policy(end,:,:,:)-1)/(simoptions.ngridinterp+1),1),[N_a,max(1,N_bothze),1,N_j]); % probability of upper grid point (from L2 index)
    PolicyProbs(:,:,1:N_u*2,:)=PolicyProbs(:,:,1:N_u*2,:).*(1-aprimeProbs_upper); % lower a1
    PolicyProbs(:,:,N_u*2+1:end,:)=PolicyProbs(:,:,N_u*2+1:end,:).*aprimeProbs_upper; % upper a1
end
CumPolicyProbs=cumsum(PolicyProbs,3);

%% Simulations are done on cpu
Policy_aprime=gather(Policy_aprime);
CumPolicyProbs=gather(CumPolicyProbs);

%% Dispatch on (z, e) presence
if N_z==0
    if N_e==0 % No z, No e
        Policy_aprime=reshape(Policy_aprime,[N_a,N_probs,N_j]);
        CumPolicyProbs=reshape(CumPolicyProbs,[N_a,N_probs,N_j]);

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
        parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
            seedpoint=seedpoints(ii,:);
            SimLifeCycleKron=SimLifeCycleIndexes_FHorz_PolicyProbs_noz_raw(Policy_aprime,CumPolicyProbs,N_j, simoptions, seedpoint);
            SimPanel(:,:,ii)=SimLifeCycleKron;
        end

        if simoptions.simpanelindexkron==0 % Convert results out of kron
            SimPanelKron=reshape(SimPanel,[2,N_j*simoptions.numbersims]);
            SimPanel=nan(l_a+1,N_j*simoptions.numbersims); % (a,j)

            SimPanel(1:l_a,:)=ind2sub_vec_homemade(n_a,SimPanelKron(1,:)')'; % a
            SimPanel(end,:)=SimPanelKron(2,:); % j

            SimPanel=reshape(SimPanel,[2,N_j,simoptions.numbersims]);
        end

    else  % No z, with e
        Policy_aprime=reshape(Policy_aprime,[N_a,N_e,N_probs,N_j]);
        CumPolicyProbs=reshape(CumPolicyProbs,[N_a,N_e,N_probs,N_j]);

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
        parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
            seedpoint=seedpoints(ii,:);
            SimLifeCycleKron=SimLifeCycleIndexes_FHorz_PolicyProbs_noz_e_raw(Policy_aprime,CumPolicyProbs,N_j,cumsumpi_e_J, simoptions, seedpoint);
            SimPanel(:,:,ii)=SimLifeCycleKron;
        end

        if simoptions.simpanelindexkron==0 % Convert results out of kron
            SimPanelKron=reshape(SimPanel,[3,N_j*simoptions.numbersims]);
            SimPanel=nan(l_a+l_e+1,N_j*simoptions.numbersims); % (a,e,j)

            SimPanel(1:l_a,:)=ind2sub_homemade(n_a,SimPanelKron(1,:)); % a
            SimPanel(l_a+1:l_a+l_e,:)=ind2sub_homemade(simoptions.n_e,SimPanelKron(2,:)); % e
            SimPanel(end,:)=SimPanelKron(3,:); % j

            SimPanel=reshape(SimPanel,[3,N_j,simoptions.numbersims]);
        else
            % All exogenous states together
            % Only e, so already is
        end
    end
else % N_z>0
    if N_e==0 % z, no e
        Policy_aprime=reshape(Policy_aprime,[N_a,N_z,N_probs,N_j]);
        CumPolicyProbs=reshape(CumPolicyProbs,[N_a,N_z,N_probs,N_j]);

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
        parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
            seedpoint=seedpoints(ii,:);
            SimLifeCycleKron=SimLifeCycleIndexes_FHorz_PolicyProbs_raw(Policy_aprime,CumPolicyProbs,N_j,cumsumpi_z_J, simoptions, seedpoint);
            SimPanel(:,:,ii)=SimLifeCycleKron;
        end

        if simoptions.simpanelindexkron==0 % Convert results out of kron
            SimPanelKron=reshape(SimPanel,[3,N_j*simoptions.numbersims]);
            SimPanel=nan(l_a+l_z+1,N_j*simoptions.numbersims); % (a,z,j)

            SimPanel(1:l_a,:)=ind2sub_vec_homemade(n_a,SimPanelKron(1,:)')'; % a
            SimPanel(l_a+1:l_a+l_z,:)=ind2sub_vec_homemade(n_z,SimPanelKron(2,:)')'; % z
            SimPanel(end,:)=SimPanelKron(3,:); % j

            SimPanel=reshape(SimPanel,[3,N_j,simoptions.numbersims]);
        else
            % All exogenous states together
            % Only z, so already is
        end


    else % z and e
        Policy_aprime=reshape(Policy_aprime,[N_a,N_z,N_e,N_probs,N_j]);
        CumPolicyProbs=reshape(CumPolicyProbs,[N_a,N_z,N_e,N_probs,N_j]);

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
        parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
            seedpoint=seedpoints(ii,:);
            SimLifeCycleKron=SimLifeCycleIndexes_FHorz_PolicyProbs_e_raw(Policy_aprime,CumPolicyProbs,N_j,cumsumpi_z_J,cumsumpi_e_J, simoptions, seedpoint);
            SimPanel(:,:,ii)=SimLifeCycleKron;
        end

        if simoptions.simpanelindexkron==0 % Convert results out of kron
            SimPanelKron=reshape(SimPanel,[4,N_j*simoptions.numbersims]);
            SimPanel=nan(l_a+l_z+l_e+1,N_j*simoptions.numbersims); % (a,z,e,j)

            SimPanel(1:l_a,:)=ind2sub_homemade(n_a,SimPanelKron(1,:)); % a
            SimPanel(l_a+1:l_a+l_z,:)=ind2sub_homemade(n_z,SimPanelKron(2,:)); % z
            SimPanel(l_a+l_z+1:l_a+l_z+l_e,:)=ind2sub_homemade(simoptions.n_e,SimPanelKron(3,:)); % e
            SimPanel(end,:)=SimPanelKron(4,:); % j

            SimPanel=reshape(SimPanel,[4,N_j,simoptions.numbersims]);
        else
            % All exogenous states together
            SimPanel(2,:,:)=SimPanel(2,:,:)+N_z*(SimPanel(3,:,:)-1); % put z and e together
            SimPanel(3,:,:)=SimPanel(4,:,:); % move j forward
            SimPanel=SimPanel(1:3,:,:);
        end
    end
end


end
