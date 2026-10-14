function SimPanel=SimPanelIndexes_FHorz_ExpAssetU_semiz(InitialDist,Policy,n_d,n_a,n_z,N_j,pi_z_J, Parameters, simoptions)
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
if N_e==0
    l_e=0;
else
    l_e=length(simoptions.n_e);
    cumsumpi_e_J=gather(cumsum(simoptions.pi_e_J,1));
end

cumsumInitialDistVec=cumsum(InitialDist(:))/sum(InitialDist(:)); % Note: by using (:) I can ignore what the original dimensions were

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

%% Setup related to experience asset u
% Split endogenous assets into the standard ones and the experience asset
% simoptions.experienceassetu is the *count* of EAU dims (1 or 2).
l_a2=simoptions.experienceassetu;
if length(n_a)<=l_a2
    n_a1=0;
else
    n_a1=n_a(1:end-l_a2);
end
n_a2=n_a(end-l_a2+1:end); % last l_a2 dims are the experience asset(s)

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

if isfield(simoptions,'n_u')
    n_u=simoptions.n_u;
else
    error('To use an experience assetu you must define vfoptions.n_u')
end
N_u=prod(n_u);
if isfield(simoptions,'u_grid')
    u_grid=simoptions.u_grid(:); % note, column
else
    error('To use an experience assetu you must define vfoptions.u_grid')
end
if isfield(simoptions,'pi_u')
    pi_u=simoptions.pi_u(:); % note, column
else
    error('To use an experience assetu you must define vfoptions.pi_u')
end

% Make sure u_grid and pi_u are on gpu
u_grid=gpuArray(u_grid);
pi_u=gpuArray(pi_u);
simoptions.u_grid=gpuArray(simoptions.u_grid); % needed to by some subfns

% aprimeFnParamNames in same fashion
% l_d2=length(n_d2);
% l_a2 already set from simoptions.experienceassetu above
l_u=length(n_u);
temp=getAnonymousFnInputNames(simoptions.aprimeFn);
if length(temp)>(l_d2+l_a2+l_u+(l_a2>=2))  % the (l_a2>=2) term is the 'whicha' selector slot, which aprimeFn only takes when there are two experience assets
    aprimeFnParamNames={temp{l_d2+l_a2+l_u+(l_a2>=2)+1:end}}; % the first inputs will always be (d2,a2,u)
else
    aprimeFnParamNames={};
end

%% Setup related to semi-exogenous state
% Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
simoptions=SemiExogShockSetup_FHorz(n_d,N_j,simoptions.d_grid,Parameters,simoptions,3);
% output: vfoptions.semiz_gridvals_J, vfoptions.pi_semiz_J
% size(semiz_gridvals_J)=[prod(n_z),length(n_z),N_j]
% size(pi_semiz_J)=[prod(n_semiz),prod(n_semiz),prod(n_dsemiz),N_j]
% If no semiz, then vfoptions just does not contain these field

% Create the transition matrix in terms of (d,zprime,z) for the semi-exogenous states for each age
N_semiz=prod(simoptions.n_semiz);
l_semiz=length(simoptions.n_semiz);
cumsumpi_semiz_J=gather(cumsum(simoptions.pi_semiz_J,2));


%%
if N_z==0
    if N_e==0
        N_semizze=N_semiz;
    else
        N_semizze=N_semiz*N_e;
    end
else
    if N_e==0
        N_semizze=N_semiz*N_z;
    else
        N_semizze=N_semiz*N_z*N_e;
    end
end

InitialDist=reshape(InitialDist,[N_a*N_semizze,1]);
Policy=reshape(Policy,[size(Policy,1),N_a,N_semizze,N_j]);


%% expassetu transitions
% Policy is currently about d and a2prime. Convert it to being about aprime
% as that is what we need for simulation, and we can then just send it to standard Case1 commands.
% For l_a2==1: 2 corners (lower/upper) per u. For l_a2==2: 4 corners (bilinear lattice) per u.
N_corners=2^l_a2;
l_a1=length(n_a)-l_a2;
N_a1kron=max(prod(n_a1),1); % so the Kron offset N_a1kron*(a2Kron-1) is well-defined in the noa1 case
Policy_aprime=zeros(N_a,N_semizze,N_u,N_corners,N_j,'gpuArray'); % Kron'd a-index per corner
PolicyProbs=zeros(N_a,N_semizze,N_u,N_corners,N_j,'gpuArray'); % corner probability times the probability of u
whichisdforexpasset=length(n_d)-l_d3;  % is just saying which is the decision variable that influences the experience asset (it is the 'second last' decision variable)
for jj=1:N_j
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [aprimeIndexes, aprimeProbs]=CreateaprimePolicyExperienceAssetu(Policy(:,:,:,jj),simoptions.aprimeFn, whichisdforexpasset, n_d, n_a1,n_a2, N_semizze,n_u, d_grid, a2_grid,u_grid, aprimeFnParamsVec);
    % Note: aprimeIndexes and aprimeProbs are both [N_a,N_z,N_u]
    % Note: aprimeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the aprimeProbs are the probability of this lower point (prob of upper point is just 1 minus this).

    % Build the a1-Kron'd index, same shape for all corners.
    if l_a1==0
        a1primeKron=zeros(N_a,N_semizze,'gpuArray'); % no a1; offset is 0 (a2Kron itself is the index)
    else
        a1primeKron=shiftdim(Policy(l_d+1,:,:,jj),1);
        if l_a1>=2
            a1primeKron=a1primeKron+n_a(1)*(shiftdim(Policy(l_d+2,:,:,jj),1)-1);
        end
        if l_a1>=3
            error('Not yet implemented experienceassetu with length(n_a1)>2')
        end
        a1primeKron=a1primeKron-1; % zero-based offset so the final +1 comes from a2Kron
    end

    if l_a2==1
        for c=1:N_corners
            if c==1
                a2Kron=aprimeIndexes;
                pcorner=aprimeProbs;
            else
                a2Kron=aprimeIndexes+1;
                pcorner=1-aprimeProbs;
            end
            if l_a1==0
                Policy_aprime(:,:,:,c,jj)=a2Kron;
            else
                Policy_aprime(:,:,:,c,jj)=a1primeKron+1+N_a1kron*(a2Kron-1);
            end
            % Encode the u probabilities (pi_u) into the PolicyProbs
            PolicyProbs(:,:,:,c,jj)=pcorner.*shiftdim(pi_u,-2);
        end
    else % l_a2==2
        % aprimeIndexes/aprimeProbs are [N_a,l_a2,N_semizze,N_u] per-dim factored; fold the two per-dim
        % lower indexes into the four corners of the a2 lattice.
        n_a2_1=n_a2(1);
        loIdx_1=reshape(aprimeIndexes(:,1,:,:),[N_a,N_semizze,N_u]);
        loIdx_2=reshape(aprimeIndexes(:,2,:,:),[N_a,N_semizze,N_u]);
        prob_1=reshape(aprimeProbs(:,1,:,:),[N_a,N_semizze,N_u]);
        prob_2=reshape(aprimeProbs(:,2,:,:),[N_a,N_semizze,N_u]);
        bits=[0 0; 1 0; 0 1; 1 1];
        for c=1:N_corners
            b1=bits(c,1); b2=bits(c,2);
            a2Kron=(loIdx_1+b1)+n_a2_1*((loIdx_2+b2)-1);
            if l_a1==0
                Policy_aprime(:,:,:,c,jj)=a2Kron;
            else
                Policy_aprime(:,:,:,c,jj)=a1primeKron+1+N_a1kron*(a2Kron-1);
            end
            p1=prob_1; if b1==1, p1=1-p1; end
            p2=prob_2; if b2==1, p2=1-p2; end
            % Encode the u probabilities (pi_u) into the PolicyProbs
            PolicyProbs(:,:,:,c,jj)=p1.*p2.*shiftdim(pi_u,-2);
        end
    end
end
Policy_aprime=reshape(Policy_aprime,[N_a,N_semizze,N_u*N_corners,N_j]);
PolicyProbs=reshape(PolicyProbs,[N_a,N_semizze,N_u*N_corners,N_j]);

N_probs=N_u*N_corners;
if simoptions.gridinterplayer==1
    N_probs=2*N_u*N_corners;
    % (a,z,N_u*N_corners,j)
    Policy_aprime=repmat(Policy_aprime,1,1,2,1);
    PolicyProbs=repmat(PolicyProbs,1,1,2,1);
    % Policy_aprime(:,:,1:2,:) lower grid point for a1 is unchanged
    Policy_aprime(:,:,N_u*N_corners+1:end,:)=Policy_aprime(:,:,N_u*N_corners+1:end,:)+1; % add one to a1, to get upper grid point

    % L2flag override (1=force all weight to lower, 2=usual, 3=force all weight to upper)
    L2index=Policy(end-1,:,:,:); % L2 index (end-1 because end is L2flag)
    L2flag=Policy(end,:,:,:);
    L2index(L2flag==1)=1;                        % force all weight to lower grid point
    L2index(L2flag==3)=simoptions.ngridinterp+2; % force all weight to upper grid point
    aprimeProbs_upper=reshape(shiftdim((L2index-1)/(simoptions.ngridinterp+1),1),[N_a,N_semizze,1,N_j]); % probability of upper grid point
    PolicyProbs(:,:,1:N_u*N_corners,:)=PolicyProbs(:,:,1:N_u*N_corners,:).*(1-aprimeProbs_upper); % lower a1
    PolicyProbs(:,:,N_u*N_corners+1:end,:)=PolicyProbs(:,:,N_u*N_corners+1:end,:).*aprimeProbs_upper; % upper a1
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

%% Simulations are done on cpu
Policy_aprime=gather(Policy_aprime);
CumPolicyProbs=gather(CumPolicyProbs);
Policy_dsemiexo=gather(Policy_dsemiexo);


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

        SimPanel=nan(3,N_j,simoptions.numbersims); % (a,semiz,j)
        parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
            seedpoint=seedpoints(ii,:);
            SimLifeCycleKron=SimLifeCycleIndexes_FHorz_PolicyProbs_semiz_noz_raw(Policy_aprime,CumPolicyProbs,Policy_dsemiexo,N_j,cumsumpi_semiz_J, simoptions, seedpoint);
            SimPanel(:,:,ii)=SimLifeCycleKron;
        end

        if simoptions.simpanelindexkron==0 % Convert results out of kron
            SimPanelKron=reshape(SimPanel,[3,N_j*simoptions.numbersims]);
            SimPanel=nan(l_a+l_semiz+1,N_j*simoptions.numbersims); % (a,semiz,j)

            SimPanel(1:l_a,:)=ind2sub_vec_homemade(n_a,SimPanelKron(1,:)')'; % a
            SimPanel(l_a+1:l_a+l_semiz,:)=ind2sub_vec_homemade(n_semiz,SimPanelKron(2,:)')'; % semiz
            SimPanel(end,:)=SimPanelKron(3,:); % j

            SimPanel=reshape(SimPanel,[l_a+l_semiz+1,N_j,simoptions.numbersims]);
        else
            % All exogenous states together
            % Only semiz, so nothing to do
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

        SimPanel=nan(4,N_j,simoptions.numbersims); % (a,semiz,e,j)
        parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
            seedpoint=seedpoints(ii,:);
            SimLifeCycleKron=SimLifeCycleIndexes_FHorz_PolicyProbs_semiz_noz_e_raw(Policy_aprime,CumPolicyProbs,Policy_dsemiexo,N_j,cumsumpi_semiz_J,cumsumpi_e_J, simoptions, seedpoint);
            SimPanel(:,:,ii)=SimLifeCycleKron;
        end

        if simoptions.simpanelindexkron==0 % Convert results out of kron
            SimPanelKron=reshape(SimPanel,[4,N_j*simoptions.numbersims]);
            SimPanel=nan(l_a+l_semiz+l_e+1,N_j*simoptions.numbersims); % (a,semiz,e,j)

            SimPanel(1:l_a,:)=ind2sub_vec_homemade(n_a,SimPanelKron(1,:)')'; % a
            SimPanel(l_a+1:l_a+l_semiz,:)=ind2sub_vec_homemade(n_semiz,SimPanelKron(2,:)')'; % semiz
            SimPanel(l_a+l_semiz+l_z+1:l_a+l_semiz+l_e,:)=ind2sub_vec_homemade(simoptions.n_e,SimPanelKron(3,:)')'; % e
            SimPanel(end,:)=SimPanelKron(4,:); % j

            SimPanel=reshape(SimPanel,[l_a+l_semiz+l_e+1,N_j,simoptions.numbersims]);
        else
            % All exogenous states together
            SimPanel(2,:,:)=SimPanel(2,:,:)+N_semiz*(SimPanel(3,:,:)-1); % put semiz and e together
            SimPanel(3,:,:)=SimPanel(4,:,:); % move j forward
            SimPanel=SimPanel(1:3,:,:);
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

        SimPanel=nan(4,N_j,simoptions.numbersims); % (a,semiz,z,j)
        parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
            seedpoint=seedpoints(ii,:);
            SimLifeCycleKron=SimLifeCycleIndexes_FHorz_PolicyProbs_semiz_raw(Policy_aprime,CumPolicyProbs,Policy_dsemiexo,N_j,cumsumpi_z_J,cumsumpi_semiz_J, simoptions, seedpoint);
            SimPanel(:,:,ii)=SimLifeCycleKron;
        end

        if simoptions.simpanelindexkron==0 % Convert results out of kron
            SimPanelKron=reshape(SimPanel,[4,N_j*simoptions.numbersims]);
            SimPanel=nan(l_a+l_semiz+l_z+1,N_j*simoptions.numbersims); % (a,semiz,z,j)

            SimPanel(1:l_a,:)=ind2sub_vec_homemade(n_a,SimPanelKron(1,:)')'; % a
            SimPanel(l_a+1:l_a+l_semiz,:)=ind2sub_vec_homemade(n_semiz,SimPanelKron(2,:)')'; % semiz
            SimPanel(l_a+l_semiz+1:l_a+l_semiz+l_z,:)=ind2sub_vec_homemade(n_z,SimPanelKron(3,:)')'; % z
            SimPanel(end,:)=SimPanelKron(4,:); % j

            SimPanel=reshape(SimPanel,[l_a+l_semiz+l_z+1,N_j,simoptions.numbersims]);
        else
            % All exogenous states together
            SimPanel(2,:,:)=SimPanel(2,:,:)+N_semiz*(SimPanel(3,:,:)-1); % put semiz and z together
            SimPanel(3,:,:)=SimPanel(4,:,:); % move j forward
            SimPanel=SimPanel(1:3,:,:);
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

        SimPanel=nan(5,N_j,simoptions.numbersims); % (a,semiz,z,e,j)
        parfor ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
            seedpoint=seedpoints(ii,:);
            SimLifeCycleKron=SimLifeCycleIndexes_FHorz_PolicyProbs_semiz_e_raw(Policy_aprime,CumPolicyProbs,Policy_dsemiexo,N_j,cumsumpi_z_J,cumsumpi_semiz_J,cumsumpi_e_J, simoptions, seedpoint);
            SimPanel(:,:,ii)=SimLifeCycleKron;
        end

        if simoptions.simpanelindexkron==0 % Convert results out of kron
            SimPanelKron=reshape(SimPanel,[5,N_j*simoptions.numbersims]);
            SimPanel=nan(l_a+l_semiz+l_z+l_e+1,N_j*simoptions.numbersims); % (a,semiz,z,e,j)

            SimPanel(1:l_a,:)=ind2sub_vec_homemade(n_a,SimPanelKron(1,:)')'; % a
            SimPanel(l_a+1:l_a+l_semiz,:)=ind2sub_vec_homemade(n_semiz,SimPanelKron(2,:)')'; % semiz
            SimPanel(l_a+l_semiz+1:l_a+l_semiz+l_z,:)=ind2sub_vec_homemade(n_z,SimPanelKron(3,:)')'; % z
            SimPanel(l_a+l_semiz+l_z+1:l_a+l_semiz+l_z+l_e,:)=ind2sub_vec_homemade(simoptions.n_e,SimPanelKron(4,:)')'; % e
            SimPanel(end,:)=SimPanelKron(5,:); % j

            SimPanel=reshape(SimPanel,[l_a+l_semiz+l_z+l_e+1,N_j,simoptions.numbersims]);
        else
            % All exogenous states together
            SimPanel(2,:,:)=SimPanel(2,:,:)+N_semiz*(SimPanel(3,:,:)-1)+N_semiz*N_z*(SimPanel(4,:,:)-1); % put semiz and z and e together
            SimPanel(3,:,:)=SimPanel(5,:,:); % move j forward
            SimPanel=SimPanel(1:3,:,:);
        end
    end
end




end



