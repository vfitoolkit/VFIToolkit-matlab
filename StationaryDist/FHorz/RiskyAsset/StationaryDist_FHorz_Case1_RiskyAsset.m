function StationaryDist=StationaryDist_FHorz_Case1_RiskyAsset(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions)


%% Setup related to risky asset
if ~isfield(simoptions,'aprimeFn')
    error('To use an risky asset you must define simoptions.aprimeFn')
end
if ~isfield(simoptions,'a_grid')
    error('To use an risky asset you must define simoptions.a_grid')
end
if ~isfield(simoptions,'d_grid')
    error('To use an risky asset you must define simoptions.d_grid')
end

% Split endogenous assets into the standard ones and the experience asset
if length(n_a)==1
    n_a1=0;
else
    n_a1=n_a(1:end-1);
end
n_a2=n_a(end); % n_a2 is the experience asset
a1_grid=simoptions.a_grid(1:sum(n_a1));
a2_grid=simoptions.a_grid(sum(n_a1)+1:end);

%%
if ~isfield(simoptions,'n_u')
    error('To use an risky asset you must define simoptions.n_u')
end
if ~isfield(simoptions,'u_grid')
    error('To use an risky asset you must define simoptions.u_grid')
end
if ~isfield(simoptions,'pi_u')
    error('To use an risky asset you must define simoptions.pi_u')
end
% to evaluate the aprimeFn we need the grids on gpu
u_grid=gpuArray(simoptions.u_grid);
pi_u=gpuArray(simoptions.pi_u);

%%
if ~isfield(simoptions,'aprimedependsonage')
    simoptions.aprimedependsonage=0;
end

% N_d=prod(n_d);
N_a=prod(n_a);
N_a1=prod(n_a1);
% N_a2=prod(n_a2);
N_z=prod(n_z);
N_u=prod(simoptions.n_u);

if N_a1==0
    n_a=n_a2;
else
    n_a=[n_a1,n_a2];
end

if isfield(simoptions,'n_e')
    N_e=prod(simoptions.n_e);
    jequaloneDistKron=reshape(jequaloneDist,[N_a*N_z*N_e,1]);
    Policy=reshape(Policy,[size(Policy,1),N_a,N_z*N_e,N_j]);
    N_ze=N_z*N_e;
else
    jequaloneDistKron=reshape(jequaloneDist,[N_a*N_z,1]);
    Policy=reshape(Policy,[size(Policy,1),N_a,N_z,N_j]);
    N_ze=N_z;
end
% NOTE: have rolled e into z

%%
l_d=length(n_d);
l_u=length(n_a);

% aprimeFnParamNames in same fashion
l_u=length(simoptions.n_u);
temp=getAnonymousFnInputNames(simoptions.aprimeFn);
if length(temp)>(l_d+l_u)
    aprimeFnParamNames={temp{l_d+l_u+1:end}}; % the first inputs will always be (d,u)
else
    aprimeFnParamNames={};
end

%%
Policy_a2prime=zeros(N_a,N_ze,N_u,2,N_j,'gpuArray'); % the lower grid point
PolicyProbs=zeros(N_a,N_ze,N_u,2,N_j,'gpuArray'); % probabilities of grid points
whichisdforriskyasset=N_a1+1:1:length(n_d);  % is just saying which is the decision variable that influences the risky asset (it is all the decision variables)
for jj=1:N_j
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndexes,a2primeProbs]=CreateaprimePolicyRiskyAsset_Case1(Policy(:,:,:,jj),simoptions.aprimeFn, whichisdforriskyasset, n_d, n_a, N_ze, simoptions.n_u, simoptions.d_grid, a2_grid, u_grid, aprimeFnParamsVec);
    % Note: aprimeIndexes and aprimeProbs are both [N_a,N_z,N_u]
    % Note: aprimeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the aprimeProbs are the probability of this lower point (prob of upper point is just 1 minus this).
    Policy_a2prime(:,:,:,1,jj)=a2primeIndexes; % lower grid point
    Policy_a2prime(:,:,:,2,jj)=a2primeIndexes+1; % upper grid point
    % Encode the u probabilities (pi_u) into the PolicyProbs
    PolicyProbs(:,:,:,1,jj)=a2primeProbs.*shiftdim(pi_u,-2); % lower grid point probability (and probability of u)
    PolicyProbs(:,:,:,2,jj)=(1-a2primeProbs).*shiftdim(pi_u,-2); % upper grid point probability (and probability of u)
end

if N_a1>0
    Policy_aprime(:,:,:,1,:)=reshape(Policy(1,:,:,:),[size(Policy,1),N_a,N_ze,1,N_j])+N_a1*(Policy_a2prime(:,:,:,1,jj)-1);
    Policy_aprime(:,:,:,2,:)=reshape(Policy(1,:,:,:),[size(Policy,1),N_a,N_ze,1,N_j])+N_a1*Policy_a2prime(:,:,:,1,jj); % Note: upper grid point minus 1 is anyway just lower grid point
    if length(n_a1)>1
        error('Only one asset other than the risky asset is allowed (email if you need this)')
    end
end


%%
% Note that PolicyProbs contains pi_u already.
if N_a1==0
    if simoptions.iterate==0
        error('simulation of agent distribution is not yet supported with riskyasset')
        % if isfield(simoptions,'n_e')
        %     StationaryDistKron=StationaryDist_FHorz_Case1_RiskyAsset_Simulation_e_raw(jequaloneDistKron,AgeWeightParamNames,Policy_a2prime,PolicyProbs,n_d,n_a2,n_z,simoptions.n_e,simoptions.n_u,N_j,simoptions.d_grid,a2_grid,u_grid,pi_z_J,pi_e_J,pi_u,simoptions.aprimeFn,Parameters,aprimeFnParamNames, simoptions);
        % else
        %     StationaryDistKron=StationaryDist_FHorz_Case1_RiskyAsset_Simulation_raw(jequaloneDistKron,AgeWeightParamNames,Policy_a2prime,PolicyProbs,n_d,n_a2,n_z,simoptions.n_u,N_j,simoptions.d_grid,a2_grid,u_grid,pi_z_J,pi_u,simoptions.aprimeFn,Parameters,aprimeFnParamNames,simoptions);
        % end
    elseif simoptions.iterate==1
        if isfield(simoptions,'n_e')
            StationaryDistKron=StationaryDist_FHorz_Case1_Iteration_uProbs_e_raw(jequaloneDistKron,AgeWeightParamNames,Policy_a2prime,PolicyProbs,N_a,N_z,N_e,N_u,N_j,pi_z_J,simoptions.pi_e_J,Parameters);
        else
            StationaryDistKron=StationaryDist_FHorz_Case1_Iteration_uProbs_raw(jequaloneDistKron,AgeWeightParamNames,Policy_a2prime,PolicyProbs,N_a,N_z,N_u,N_j,pi_z_J,Parameters);
        end
    end
else
    if simoptions.iterate==1
        if isfield(simoptions,'n_e')
            StationaryDistKron=StationaryDist_FHorz_Case1_Iteration_uProbs_e_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_a,N_z,N_e,N_u,N_j,pi_z_J,simoptions.pi_e_J,Parameters);
        else
            StationaryDistKron=StationaryDist_FHorz_Case1_Iteration_uProbs_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_a,N_z,N_u,N_j,pi_z_J,Parameters);
        end
    end
end

if isfield(simoptions,'n_e')
    StationaryDist=reshape(StationaryDistKron,[n_a,n_z,simoptions.n_e,N_j]);    
else
    StationaryDist=reshape(StationaryDistKron,[n_a,n_z,N_j]);
end

end
