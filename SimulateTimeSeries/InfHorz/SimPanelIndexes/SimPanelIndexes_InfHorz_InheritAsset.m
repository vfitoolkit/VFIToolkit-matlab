function SimPanel=SimPanelIndexes_InfHorz_InheritAsset(InitialDist,Policy,n_d,n_a,n_z,pi_z, Parameters, simoptions, CondlProbOfSurvival)
% Input must already be on CPU
% Simulates a panel based on PolicyIndexes of 'numbersims' agents of length 'simperiods' beginning from randomly drawn InitialDist. 
%
% CondlProbOfSurvival is an optional input. Only needed when using: simoptions.exitinpanel=1, there there is exit, either exog, endog or mix of both.
% Parameters is an optional input. Only needed when you have mixed (endogenous and exogenous) exit.
%
% simoptions are already set, this is an internal use only command
% (SimPanelIndexes is called by SimPanelValues)

%%
simoptions.numbersims=gather(simoptions.numbersims); % This is just to deal with weird error that matlab decided simoptions.numbersims was on gpu and so couldn't be an input to rand()
simoptions.simperiods=gather(simoptions.simperiods);
simoptions.burnin=gather(simoptions.burnin);

%%
N_d=prod(n_d);
if N_d>0
    l_d=length(n_d);
else
    l_d=0;
end

N_a=prod(n_a);
l_a=length(n_a);

N_z=prod(n_z);
l_z=length(n_z); % Inheritanceasset, so there must be a z
cumsumpi_z=cumsum(pi_z,2);

N_e=prod(simoptions.n_e);
if N_e>0
    l_e=length(simoptions.n_e);
    cumsumpi_e=gather(cumsum(simoptions.pi_e,1));
else
    l_e=0;
end

cumsumInitialDistVec=cumsum(InitialDist(:))/sum(InitialDist(:)); % Note: by using (:) I can ignore what the original dimensions were

%%
if exist('CondlProbOfSurvival','var')
    simoptions.exitinpanel=1;
    CondlProbOfSurvival=reshape(CondlProbOfSurvival,[N_a,N_z]);
    if ~isfield(simoptions, 'endogenousexit')
        simoptions.endogenousexit=0;  % Note: this will only be relevant if exitinpanel=1
    end
else
    CondlProbOfSurvival=0; % will be unused, but otherwise there was an error that it wasnt recognized
end


%%
% Inheritanceasset, so there must be a z
if N_e==0
    N_bothze=N_z;
else
    N_bothze=N_z*N_e;
end

N_zprime=N_z;

Policy=reshape(Policy,[size(Policy,1),N_a,N_bothze]);
Policy_aprime=Policy(l_d+1:end,:,:,:);

%% Setup related to inheritance asset
n_d2=n_d(end);
% Split endogenous assets into the standard ones and the inheritance asset
if isscalar(n_a)
    n_a1=0;
else
    n_a1=n_a(1:end-1);
end
n_a2=n_a(end); % n_a2 is the inheritance asset

if ~isfield(simoptions,'aprimeFn')
    error('To use an inheritance asset you must define simoptions.aprimeFn')
end
if isfield(simoptions,'a_grid')
    % a_grid=simoptions.a_grid;
    % a1_grid=simoptions.a_grid(1:sum(n_a1));
    a2_grid=simoptions.a_grid(sum(n_a1)+1:end);
else
    error('To use an inheritance asset you must define simoptions.a_grid')
end
if isfield(simoptions,'d_grid')
    d_grid=simoptions.d_grid;
else
    error('To use an inheritance asset you must define simoptions.d_grid')
end
if isfield(simoptions,'z_grid')
    z_grid=simoptions.z_grid;
else
    error('To use an inheritance asset you must define simoptions.z_grid')
end

% aprimeFnParamNames in same fashion
l_d2=length(n_d2);
l_z=length(n_z);
temp=getAnonymousFnInputNames(simoptions.aprimeFn);
if length(temp)>(l_d2+2*l_z)
    aprimeFnParamNames={temp{l_d2+2*l_z+1:end}}; % the first inputs will always be (d2,a2)
else
    aprimeFnParamNames={};
end

z_gridvals=CreateGridvals(n_z,z_grid,1);

%% Policy is currently about d and a1prime. Convert it to being about aprime as that is what we need for simulation.
Policy_a2prime=zeros(N_a,N_z,N_zprime,2,'gpuArray'); % the lower grid point
PolicyProbs=zeros(N_a,N_z,N_zprime,2,'gpuArray'); % The fourth dimension is lower/upper grid point
whichisdforinheritasset=length(n_d);  % is just saying which is the decision variable that influences the experience asset (it is the 'last' decision variable)
aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames);
[a2primeIndexes, a2primeProbs]=CreateaprimePolicyInheritanceAsset(Policy,simoptions.aprimeFn, whichisdforinheritasset, n_d, n_a1,n_a2, n_z, n_z, gpuArray(d_grid), a2_grid, gpuArray(z_gridvals), gpuArray(z_gridvals), aprimeFnParamsVec);
% Note: aprimeIndexes and aprimeProbs are both [N_a,N_z,N_zprime]
% Note: aprimeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the aprimeProbs are the probability of this lower point (prob of upper point is just 1 minus this).
Policy_a2prime(:,:,:,1)=a2primeIndexes; % lower grid point
Policy_a2prime(:,:,:,2)=a2primeIndexes+1; % upper grid point
PolicyProbs(:,:,:,1)=a2primeProbs; % probability of lower grid point
PolicyProbs(:,:,:,2)=1-a2primeProbs; % probability of upper grid point

if l_a==1 % just inheritanceasset
    Policy_aprime=Policy_a2prime;
elseif l_a==2 % one other asset, then inheritance asset
    Policy_aprime(:,:,:,1)=reshape(Policy(l_d+1,:,:),[N_a,N_z,1])+n_a(1)*(Policy_a2prime(:,:,:,1)-1);
    Policy_aprime(:,:,:,2)=reshape(Policy(l_d+1,:,:),[N_a,N_z,1])+n_a(1)*Policy_a2prime(:,:,:,1); % Note: upper grid point minus 1 is anyway just lower grid point
elseif l_a==3 % two other assets, then inheritance asset
    Policy_aprime(:,:,:,1)=reshape(Policy(l_d+1,:,:),[N_a,N_z,1])+n_a(1)*reshape(Policy(l_d+2,:,:),[N_a,N_z,1])+n_a(1)*n_a(2)*(Policy_a2prime(:,:,:,1)-1);
    Policy_aprime(:,:,:,2)=reshape(Policy(l_d+1,:,:),[N_a,N_z,1])+n_a(1)*reshape(Policy(l_d+2,:,:),[N_a,N_z,1])+n_a(1)*n_a(2)*Policy_a2prime(:,:,:,1); % Note: upper grid point minus 1 is anyway just lower grid point
elseif l_a>3
    error('Not yet implemented inheritanceasset with length(n_a)>3')
end

N_probs=2;
if simoptions.gridinterplayer==1
    N_probs=4;
    % (a,z,2,j)
    Policy_aprime=repmat(Policy_aprime,1,1,1,2);
    PolicyProbs=repmat(PolicyProbs,1,1,1,2);
    % Policy_aprime(:,:,:,1:2) lower grid point for a1 is unchanged
    Policy_aprime(:,:,:,3:4)=Policy_aprime(:,:,:,3:4)+1; % add one to a1, to get upper grid point

    aprimeProbs_upper=reshape(shiftdim((Policy(end,:)-1)/(simoptions.ngridinterp+1),1),[N_a,max(1,N_bothze)]); % probability of upper grid point (from L2 index)
    PolicyProbs(:,:,:,1:2)=PolicyProbs(:,:,:,1:2).*(1-aprimeProbs_upper); % lower a1
    PolicyProbs(:,:,:,3:4)=PolicyProbs(:,:,:,3:4).*aprimeProbs_upper; % upper a1
end
CumPolicyProbs=cumsum(PolicyProbs,3);


%%

SimPanel=nan(l_a+l_z,simoptions.simperiods,simoptions.numbersims); % preallocate

if N_z>0
    if N_e==0 % z, no e
        Policy_aprime=reshape(Policy_aprime,[N_a,N_z,N_zprime,N_probs]);
        CumPolicyProbs=reshape(CumPolicyProbs,[N_a,N_z,N_zprime,N_probs]);

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
        seedpoints=[ind2sub_vec_homemade([N_a,N_z],seedpointind'),ones(simoptions.numbersims,1)];
        seedpoints=gather(floor(seedpoints)); % For some reason seedpoints had heaps of '.0000' decimal places and were not being treated as integers, this solves that.

        % simoptions.simpanelindexkron==1 % Create the simulated data in kron form
        SimPanel=nan(2,simoptions.simperiods,simoptions.numbersims); % (a,z)
        % par
        for ii=1:simoptions.numbersims % This is only change from the simoptions.parallel==0
            seedpoint=seedpoints(ii,:);
            SimLifeCycleKron=SimTimeSeriesIndexes_InfHorz_zprime_PolicyProbs_raw(Policy_aprime,CumPolicyProbs,cumsumpi_z, simoptions, seedpoint);
            SimPanel(:,:,ii)=SimLifeCycleKron;
        end

        if simoptions.simpanelindexkron==0 % Convert results out of kron
            SimPanelKron=reshape(SimPanel,[3,simoptions.simperiods*simoptions.numbersims]);
            SimPanel=nan(l_a+l_z+1,simoptions.simperiods*simoptions.numbersims); % (a,z)

            SimPanel(1:l_a,:)=ind2sub_vec_homemade(n_a,SimPanelKron(1,:)')'; % a
            SimPanel(l_a+1:l_a+l_z,:)=ind2sub_vec_homemade(n_z,SimPanelKron(2,:)')'; % z
            SimPanel=reshape(SimPanel,[3,simoptions.simperiods,simoptions.numbersims]);
        else
            % All exogenous states together
            % Only z, so already is
        end
    end
end



end



