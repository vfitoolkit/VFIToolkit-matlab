function SimPanel=SimPanelIndexes_FHorz_RiskyAsset_semiz(InitialDist,Policy,n_d,n_a,n_z,N_j,pi_z_J, Parameters, simoptions)
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

% Sort out decision variables, need to get those for riskyasset, and those for semiz
if ~isfield(simoptions,'refine_d')
    error('Cannot use riskyasset+semiz without setting simoptions.refine_d')
end
if length(simoptions.refine_d)>=4 && simoptions.refine_d(end)==simoptions.l_dsemiz
    if sum(simoptions.refine_d)~=length(n_d)
        error('simoptions.refine_d (and agreeing simoptions.l_dsemiz) should sum together to length(n_d)')
    end
elseif (sum(simoptions.refine_d)+simoptions.l_dsemiz)~=length(n_d)
    error('simoptions.refine_d and simoptions.l_dsemiz should all sum together to length(n_d)')
end
l_d4=simoptions.l_dsemiz;
n_d23=n_d(simoptions.refine_d(1)+1:sum(simoptions.refine_d(1:3))); % decision variables for riskyasset
l_d123=sum(simoptions.refine_d(1:3)); % everything except the d_semiz

% Split endogenous assets into the standard ones and the risky asset
if isscalar(n_a)
    n_a1=0;
else
    n_a1=n_a(1:end-1);
end
n_a2=n_a(end); % n_a2 is the risky asset
a2_grid=simoptions.a_grid(sum(n_a1)+1:end);

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

%% Setup related to semi-exogenous state
% Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
simoptions=SemiExogShockSetup_FHorz(n_d,N_j,simoptions.d_grid,Parameters,simoptions,3);
% output: simoptions.semiz_gridvals_J, simoptions.pi_semiz_J

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


%% riskyasset transitions
Policy_aprime=zeros(N_a,N_semizze,N_u,2,N_j,'gpuArray'); % the lower grid point
PolicyProbs=zeros(N_a,N_semizze,N_u,2,N_j,'gpuArray'); % probabilities of grid points
whichisdforriskyasset=(simoptions.refine_d(1)+1):1:sum(simoptions.refine_d(1:3));  % is just saying which is the decision variable that influences the risky asset
for jj=1:N_j
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [aprimeIndexes, aprimeProbs]=CreateaprimePolicyRiskyAsset(Policy(1:l_d,:,:,jj),simoptions.aprimeFn, whichisdforriskyasset, n_d, n_a1,n_a2, N_semizze, n_u, simoptions.d_grid, a2_grid, u_grid, aprimeFnParamsVec);
    % Note: aprimeIndexes and aprimeProbs are both [N_a,N_semizze,N_u]
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

Policy_aprime=reshape(Policy_aprime,[N_a,N_semizze,N_u*2,N_j]);
PolicyProbs=reshape(PolicyProbs,[N_a,N_semizze,N_u*2,N_j]);

N_probs=N_u*2;
if simoptions.gridinterplayer==1
    N_probs=2*N_u*2;
    % (a,z,N_u*2,j)
    Policy_aprime=repmat(Policy_aprime,1,1,2,1);
    PolicyProbs=repmat(PolicyProbs,1,1,2,1);
    % Policy_aprime(:,:,1:N_u*2,:) lower grid point for a1 is unchanged
    Policy_aprime(:,:,N_u*2+1:end,:)=Policy_aprime(:,:,N_u*2+1:end,:)+1; % add one to a1, to get upper grid point

    aprimeProbs_upper=reshape(shiftdim((Policy(end,:,:,:)-1)/(simoptions.ngridinterp+1),1),[N_a,N_semizze,1,N_j]); % probability of upper grid point (from L2 index)
    PolicyProbs(:,:,1:N_u*2,:)=PolicyProbs(:,:,1:N_u*2,:).*(1-aprimeProbs_upper); % lower a1
    PolicyProbs(:,:,N_u*2+1:end,:)=PolicyProbs(:,:,N_u*2+1:end,:).*aprimeProbs_upper; % upper a1
end
CumPolicyProbs=cumsum(PolicyProbs,3);

%% Policy_dsemiexo

% d4 is the variable relevant for the semi-exogenous asset.
if l_d4==1
    Policy_dsemiexo=Policy(l_d123+1,:,:,:);
elseif l_d4==2
    Policy_dsemiexo=Policy(l_d123+1,:,:,:)+n_d(l_d123+1)*(Policy(l_d123+2,:,:,:)-1);
elseif l_d4==3
    Policy_dsemiexo=Policy(l_d123+1,:,:,:)+n_d(l_d123+1)*(Policy(l_d123+2,:,:,:)-1)+n_d(l_d123+1)*n_d(l_d123+2)*(Policy(l_d123+3,:,:,:)-1);
elseif l_d4==4
    Policy_dsemiexo=Policy(l_d123+1,:,:,:)+n_d(l_d123+1)*(Policy(l_d123+2,:,:,:)-1)+n_d(l_d123+1)*n_d(l_d123+2)*(Policy(l_d123+3,:,:,:)-1)+n_d(l_d123+1)*n_d(l_d123+2)*n_d(l_d123+3)*(Policy(l_d123+4,:,:,:)-1);
end
Policy_dsemiexo=shiftdim(Policy_dsemiexo,1);

%% Simulations are done on cpu
Policy_aprime=gather(Policy_aprime);
CumPolicyProbs=gather(CumPolicyProbs);
Policy_dsemiexo=gather(Policy_dsemiexo);

%% Dispatch on (z, e) presence
if N_z==0
    if N_e==0 % No z, No e
        Policy_aprime=reshape(Policy_aprime,[N_a,N_semiz,N_probs,N_j]);
        CumPolicyProbs=reshape(CumPolicyProbs,[N_a,N_semiz,N_probs,N_j]);
        Policy_dsemiexo=reshape(Policy_dsemiexo,[N_a,N_semiz,N_j]);

        seedpointdim=[N_a,N_semiz];
        if numel(InitialDist)==N_a*N_semiz*N_j
            seedpointdim=[N_a,N_semiz,N_j]; % Initial dist depends on j
        end

        SimPanel=nan(3,N_j,simoptions.numbersims); % (a,semiz,j)
        parfor ii=1:simoptions.numbersims
            [~,seedpointind]=max(cumsumInitialDistVec>rand(1,1)); % Get seedpoint from InitialDist
            seedpoint=[ind2sub_homemade(seedpointdim,seedpointind'),1];
            seedpoint=gather(floor(seedpoint)); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.
            SimLifeCycleKron=SimLifeCycleIndexes_FHorz_PolicyProbs_semiz_noz_raw(Policy_aprime,CumPolicyProbs,Policy_dsemiexo,N_j,cumsumpi_semiz_J, simoptions, seedpoint);
            SimPanel(:,:,ii)=SimLifeCycleKron;
        end

        if simoptions.simpanelindexkron==0 % Convert results out of kron
            SimPanelKron=reshape(SimPanel,[3,N_j*simoptions.numbersims]);
            SimPanel=nan(l_a+l_semiz+1,N_j*simoptions.numbersims); % (a,semiz,j)

            SimPanel(1:l_a,:)=ind2sub_homemade(n_a,SimPanelKron(1,:)); % a
            SimPanel(l_a+1:l_a+l_semiz,:)=ind2sub_homemade(simoptions.n_semiz,SimPanelKron(2,:)); % semiz
            SimPanel(end,:)=SimPanelKron(3,:); % j

            SimPanel=reshape(SimPanel,[3,N_j,simoptions.numbersims]);
        else
            % All exogenous states together
            % Only semiz, so nothing to do
        end

    else % No z, e
        Policy_aprime=reshape(Policy_aprime,[N_a,N_semiz,N_e,N_probs,N_j]);
        CumPolicyProbs=reshape(CumPolicyProbs,[N_a,N_semiz,N_e,N_probs,N_j]);
        Policy_dsemiexo=reshape(Policy_dsemiexo,[N_a,N_semiz,N_e,N_j]);

        seedpointdim=[N_a,N_semiz,N_e];
        if numel(InitialDist)==N_a*N_semiz*N_e*N_j
            seedpointdim=[N_a,N_semiz,N_e,N_j]; % Initial dist depends on j
        end

        SimPanel=nan(4,N_j,simoptions.numbersims); % (a,semiz,e,j)
        parfor ii=1:simoptions.numbersims
            [~,seedpointind]=max(cumsumInitialDistVec>rand(1,1)); % Get seedpoint from InitialDist
            seedpoint=[ind2sub_homemade(seedpointdim,seedpointind'),1];
            seedpoint=gather(floor(seedpoint)); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.
            SimLifeCycleKron=SimLifeCycleIndexes_FHorz_PolicyProbs_semiz_noz_e_raw(Policy_aprime,CumPolicyProbs,Policy_dsemiexo,N_j,cumsumpi_semiz_J,cumsumpi_e_J, simoptions, seedpoint);
            SimPanel(:,:,ii)=SimLifeCycleKron;
        end

        if simoptions.simpanelindexkron==0 % Convert results out of kron
            SimPanelKron=reshape(SimPanel,[4,N_j*simoptions.numbersims]);
            SimPanel=nan(l_a+l_semiz+l_e+1,N_j*simoptions.numbersims); % (a,semiz,e,j)

            SimPanel(1:l_a,:)=ind2sub_homemade(n_a,SimPanelKron(1,:)); % a
            SimPanel(l_a+1:l_a+l_semiz,:)=ind2sub_homemade(simoptions.n_semiz,SimPanelKron(2,:)); % semiz
            SimPanel(l_a+l_semiz+1:l_a+l_semiz+l_e,:)=ind2sub_homemade(simoptions.n_e,SimPanelKron(3,:)); % e
            SimPanel(end,:)=SimPanelKron(4,:); % j

            SimPanel=reshape(SimPanel,[4,N_j,simoptions.numbersims]);
        else
            % All exogenous states together
            SimPanel(2,:,:)=SimPanel(2,:,:)+N_semiz*(SimPanel(3,:,:)-1); % put semiz and e together
            SimPanel(3,:,:)=SimPanel(4,:,:); % move j forward
            SimPanel=SimPanel(1:3,:,:);
        end
    end

else % N_z>0
    if N_e==0 % z, no e
        Policy_aprime=reshape(Policy_aprime,[N_a,N_semiz,N_z,N_probs,N_j]);
        CumPolicyProbs=reshape(CumPolicyProbs,[N_a,N_semiz,N_z,N_probs,N_j]);
        Policy_dsemiexo=reshape(Policy_dsemiexo,[N_a,N_semiz,N_z,N_j]);

        seedpointdim=[N_a,N_semiz,N_z];
        if numel(InitialDist)==N_a*N_semiz*N_z*N_j
            seedpointdim=[N_a,N_semiz,N_z,N_j]; % Initial dist depends on j
        end

        SimPanel=nan(4,N_j,simoptions.numbersims); % (a,semiz,z,j)
        parfor ii=1:simoptions.numbersims
            [~,seedpointind]=max(cumsumInitialDistVec>rand(1,1)); % Get seedpoint from InitialDist
            seedpoint=[ind2sub_homemade(seedpointdim,seedpointind'),1];
            seedpoint=gather(floor(seedpoint)); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.
            SimLifeCycleKron=SimLifeCycleIndexes_FHorz_PolicyProbs_semiz_raw(Policy_aprime,CumPolicyProbs,Policy_dsemiexo,N_j,cumsumpi_z_J,cumsumpi_semiz_J, simoptions, seedpoint);
            SimPanel(:,:,ii)=SimLifeCycleKron;
        end

        if simoptions.simpanelindexkron==0 % Convert results out of kron
            SimPanelKron=reshape(SimPanel,[4,N_j*simoptions.numbersims]);
            SimPanel=nan(l_a+l_semiz+l_z+1,N_j*simoptions.numbersims); % (a,semiz,z,j)

            SimPanel(1:l_a,:)=ind2sub_homemade(n_a,SimPanelKron(1,:)); % a
            SimPanel(l_a+1:l_a+l_semiz,:)=ind2sub_homemade(simoptions.n_semiz,SimPanelKron(2,:)); % semiz
            SimPanel(l_a+l_semiz+1:l_a+l_semiz+l_z,:)=ind2sub_homemade(n_z,SimPanelKron(3,:)); % z
            SimPanel(end,:)=SimPanelKron(4,:); % j

            SimPanel=reshape(SimPanel,[4,N_j,simoptions.numbersims]);
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

        seedpointdim=[N_a,N_semiz,N_z,N_e];
        if numel(InitialDist)==N_a*N_semiz*N_z*N_e*N_j
            seedpointdim=[N_a,N_semiz,N_z,N_e,N_j]; % Initial dist depends on j
        end

        SimPanel=nan(5,N_j,simoptions.numbersims); % (a,semiz,z,e,j)
        parfor ii=1:simoptions.numbersims
            [~,seedpointind]=max(cumsumInitialDistVec>rand(1,1)); % Get seedpoint from InitialDist
            seedpoint=[ind2sub_homemade(seedpointdim,seedpointind'),1];
            seedpoint=gather(floor(seedpoint)); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.
            SimLifeCycleKron=SimLifeCycleIndexes_FHorz_PolicyProbs_semiz_e_raw(Policy_aprime,CumPolicyProbs,Policy_dsemiexo,N_j,cumsumpi_z_J,cumsumpi_semiz_J,cumsumpi_e_J, simoptions, seedpoint);
            SimPanel(:,:,ii)=SimLifeCycleKron;
        end

        if simoptions.simpanelindexkron==0 % Convert results out of kron
            SimPanelKron=reshape(SimPanel,[5,N_j*simoptions.numbersims]);
            SimPanel=nan(l_a+l_semiz+l_z+l_e+1,N_j*simoptions.numbersims); % (a,semiz,z,e,j)

            SimPanel(1:l_a,:)=ind2sub_homemade(n_a,SimPanelKron(1,:)); % a
            SimPanel(l_a+1:l_a+l_semiz,:)=ind2sub_homemade(simoptions.n_semiz,SimPanelKron(2,:)); % semiz
            SimPanel(l_a+l_semiz+1:l_a+l_semiz+l_z,:)=ind2sub_homemade(n_z,SimPanelKron(3,:)); % z
            SimPanel(l_a+l_semiz+l_z+1:l_a+l_semiz+l_z+l_e,:)=ind2sub_homemade(simoptions.n_e,SimPanelKron(4,:)); % e
            SimPanel(end,:)=SimPanelKron(5,:); % j

            SimPanel=reshape(SimPanel,[5,N_j,simoptions.numbersims]);
        else
            % All exogenous states together
            SimPanel(2,:,:)=SimPanel(2,:,:)+N_semiz*(SimPanel(3,:,:)-1)+N_semiz*N_z*(SimPanel(4,:,:)-1); % put semiz, z and e together
            SimPanel(3,:,:)=SimPanel(5,:,:); % move j forward
            SimPanel=SimPanel(1:3,:,:);
        end
    end
end


end
