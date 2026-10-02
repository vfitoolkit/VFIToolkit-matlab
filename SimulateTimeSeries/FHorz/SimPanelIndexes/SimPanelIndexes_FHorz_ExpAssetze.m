function SimPanel=SimPanelIndexes_FHorz_ExpAssetze(InitialDist,Policy,n_d,n_a,n_z,N_j,z_gridvals_J,pi_z_J, Parameters, simoptions)
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
if N_z==0
    error('Cannot use simoptions.experienceassetze=1 with no z variables (z is required)')
end
l_z=length(n_z);
cumsumpi_z_J=gather(cumsum(pi_z_J,2));

N_e=prod(simoptions.n_e);
if N_e==0
    error('Cannot use simoptions.experienceassetze=1 with no e variables (e is required)')
end
l_e=length(simoptions.n_e);
cumsumpi_e_J=gather(cumsum(simoptions.pi_e_J,1));

cumsumInitialDistVec=cumsum(InitialDist(:))/sum(InitialDist(:)); % Note: by using (:) I can ignore what the original dimensions were

%% Experience asset
n_d2=n_d(end); % decision variable that controls experience asset
l_d2=length(n_d2); % wouldn't be here if no d2

%% Setup related to experience asset
% Split endogenous assets into the standard ones and the experience asset
l_a2=simoptions.experienceassetze; % integer COUNT of experience-asset dims (1 or 2), not a flag
if length(n_a)<=l_a2
    n_a1=0;
    l_a1=0;
    N_a1=0;
else
    n_a1=n_a(1:end-l_a2);
    l_a1=length(n_a1);
    N_a1=prod(n_a1);
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


% aprimeFnParamNames in same fashion: (d2, a2, z, e) are the leading inputs
% l_a2 already set from simoptions.experienceassetze above
temp=getAnonymousFnInputNames(simoptions.aprimeFn);
if length(temp)>(l_d2+l_a2+l_z+l_e+(l_a2>=2))  % the (l_a2>=2) term is the 'whicha' selector slot, which aprimeFn only takes when there are two experience assets
    aprimeFnParamNames={temp{l_d2+l_a2+l_z+l_e+(l_a2>=2)+1:end}}; % the first inputs are (d2,a2,z,e), plus the 'whicha' selector when l_a2>=2
else
    aprimeFnParamNames={};
end

%%
N_bothze=N_z*N_e;

Policy=reshape(Policy,[size(Policy,1),N_a,N_bothze,N_j]);

%% expassetze transitions
% Policy is currently about d and a1prime. Convert it to being about aprime
% as that is what we need for simulation, and we can then just send it to standard Case1 commands.
N_probs=2^l_a2; % 2 points (lower and upper index) per dimension of a2
Policy_aprime=zeros(N_a,N_bothze,N_probs,N_j,'gpuArray');
PolicyProbs=zeros(N_a,N_bothze,N_probs,N_j,'gpuArray'); % third dimension indexes the interpolation corners
whichisdforexpassetze=length(n_d);  % is just saying which is the decision variable that influences the experience asset (it is the 'last' decision variable)
for jj=1:N_j
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [aprimeIndexes, aprimeProbs]=CreateaprimePolicyExperienceAssetze(Policy(:,:,:,jj),simoptions.aprimeFn, whichisdforexpassetze, n_d, n_a1,n_a2, n_z, simoptions.n_e, 0,N_z,N_e, d_grid, a2_grid, z_gridvals_J(:,:,jj), simoptions.e_gridvals_J(:,:,jj), aprimeFnParamsVec);
    % Note: aprimeIndexes and aprimeProbs are both [N_a,N_bothze] with z varying fastest -- matches N_bothze=[n_z,n_e] ordering.
    % Note: aprimeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the aprimeProbs are the probability of this lower point (prob of upper point is just 1 minus this).

    if l_a2==1
        if l_a1==0
            Policy_aprime(:,:,1,jj)=aprimeIndexes;
            Policy_aprime(:,:,2,jj)=aprimeIndexes+1;
        elseif l_a1==1 % experience asset and one other asset
            Policy_aprime(:,:,1,jj)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(aprimeIndexes-1);
            Policy_aprime(:,:,2,jj)=Policy_aprime(:,:,1,jj)+n_a(1);
        elseif l_a1==2 % experience asset and two other assets
            Policy_aprime(:,:,1,jj)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(shiftdim(Policy(l_d+2,:,:,jj),1)-1)+prod(n_a(1:2))*(aprimeIndexes-1);
            Policy_aprime(:,:,2,jj)=Policy_aprime(:,:,1,jj)+prod(n_a(1:2));
        else
            error('Not yet implemented experienceassetze with more than two standard assets')
        end
        PolicyProbs(:,:,1,jj)=aprimeProbs;
        PolicyProbs(:,:,2,jj)=1-aprimeProbs;
    else
        % l_a2==2: aprimeIndexes/aprimeProbs are [N_a,l_a2,N_bothze] per-dim factored.
        % Kron-fold to N_probs=4 corners (mirrors SimPanelIndexes_FHorz_ExpAsset).
        n_a2_1=n_a2(1);
        loIdx_1=reshape(aprimeIndexes(:,1,:),[N_a,N_bothze]);
        loIdx_2=reshape(aprimeIndexes(:,2,:),[N_a,N_bothze]);
        prob_1=reshape(aprimeProbs(:,1,:),[N_a,N_bothze]);
        prob_2=reshape(aprimeProbs(:,2,:),[N_a,N_bothze]);
        if l_a1==0
            a1primeIndexes=[];
        elseif l_a1==1
            a1primeIndexes=shiftdim(Policy(l_d+1,:,:,jj),1);
        elseif l_a1==2
            a1primeIndexes=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(shiftdim(Policy(l_d+2,:,:,jj),1)-1);
        else
            error('Not yet implemented experienceassetze with more than two standard assets')
        end
        bits=[0 0; 1 0; 0 1; 1 1];
        for c=1:N_probs
            b1=bits(c,1); b2=bits(c,2);
            a2_kron=(loIdx_1+b1)+n_a2_1*((loIdx_2+b2)-1);
            if l_a1==0
                Policy_aprime(:,:,c,jj)=a2_kron;
            else
                Policy_aprime(:,:,c,jj)=a1primeIndexes+N_a1*(a2_kron-1);
            end
            p1=prob_1; if b1==1, p1=1-p1; end
            p2=prob_2; if b2==1, p2=1-p2; end
            PolicyProbs(:,:,c,jj)=p1.*p2;
        end
    end
end

if simoptions.gridinterplayer==1
    % The interpolation layer adds the two a1prime grid points: duplicate the a2 corners, the
    % first N_probs keeping the lower a1 point and the next N_probs taking the upper one.
    Policy_aprime=repmat(Policy_aprime,1,1,2,1);
    PolicyProbs=repmat(PolicyProbs,1,1,2,1);
    % Policy_aprime(:,:,1:N_probs,:) lower grid point for a1 is unchanged
    Policy_aprime(:,:,N_probs+1:2*N_probs,:)=Policy_aprime(:,:,N_probs+1:2*N_probs,:)+1; % add one to a1, to get upper grid point (a1 is the fastest-varying dim of the aprime index)

    % L2flag override (1=force all weight to lower, 2=usual, 3=force all weight to upper)
    L2index=Policy(end-1,:,:,:); % L2 index (end-1 because end is L2flag)
    L2flag=Policy(end,:,:,:);
    L2index(L2flag==1)=1;                        % force all weight to lower grid point
    L2index(L2flag==3)=simoptions.ngridinterp+2; % force all weight to upper grid point
    aprimeProbs_upper=reshape(shiftdim((L2index-1)/(simoptions.ngridinterp+1),1),[N_a,N_bothze,1,N_j]); % probability of upper grid point
    PolicyProbs(:,:,1:N_probs,:)=PolicyProbs(:,:,1:N_probs,:).*(1-aprimeProbs_upper); % lower a1
    PolicyProbs(:,:,N_probs+1:2*N_probs,:)=PolicyProbs(:,:,N_probs+1:2*N_probs,:).*aprimeProbs_upper; % upper a1
    N_probs=2*N_probs;
end
CumPolicyProbs=cumsum(PolicyProbs,3);

%% Simulations are done on cpu
Policy_aprime=gather(Policy_aprime);
CumPolicyProbs=gather(CumPolicyProbs);

%% z and e both required: only the (z, e) branch matters
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

    SimPanel(1:l_a,:)=ind2sub_vec_homemade(n_a,SimPanelKron(1,:)')'; % a
    SimPanel(l_a+1:l_a+l_z,:)=ind2sub_vec_homemade(n_z,SimPanelKron(2,:)')'; % z
    SimPanel(l_a+l_z+1:l_a+l_z+l_e,:)=ind2sub_vec_homemade(simoptions.n_e,SimPanelKron(3,:)')'; % e
    SimPanel(end,:)=SimPanelKron(4,:); % j

    SimPanel=reshape(SimPanel,[l_a+l_z+l_e+1,N_j,simoptions.numbersims]);
else
    % All exogenous states together
    SimPanel(2,:,:)=SimPanel(2,:,:)+N_z*(SimPanel(3,:,:)-1); % put z and e together
    SimPanel(3,:,:)=SimPanel(4,:,:); % move j forward
    SimPanel=SimPanel(1:3,:,:);
end


end
