function StationaryDist=StationaryDist_FHorz_Case1_RiskyAssetSemiExo(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_semiz,n_z,N_j,pi_semiz_J,pi_z_J,Parameters,simoptions)


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

% Sort out decision variables, need to get those for riskyasset, and those for semiz
if ~isfield(simoptions,'refine_d')
    error('Cannot use riskyasset+semiz without setting simoptions.refine_d')
end
n_d23=n_d(simoptions.refine_d(1)+1:sum(simoptions.refine_d(1:3))); % decision variables for riskyasset
n_d4=n_d(sum(simoptions.refine_d(1:3))+1:sum(simoptions.refine_d(1:4))); % decision variables for semiz
d4_grid=simoptions.d_grid(sum(n_d(1:sum(simoptions.refine_d(1:3))))+1:end);

% Split endogenous assets into the standard ones and the risky asset
if isscalar(n_a)
    n_a1=0;
else
    n_a1=n_a(1:end-1);
end
n_a2=n_a(end); % n_a2 is the experience asset
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
l_d=length(n_d);

% aprimeFnParamNames in same fashion
l_u=length(simoptions.n_u);
l_d23=length(n_d23);
temp=getAnonymousFnInputNames(simoptions.aprimeFn);
if length(temp)>(l_d23+l_u)
    aprimeFnParamNames={temp{l_d23+l_u+1:end}}; % the first inputs will always be (d,u)
else
    aprimeFnParamNames={};
end


%%
if n_z(1)==0
    error('Have not yet impelmented N_z=0 in StationaryDist_FHorz_Case1_RiskyAssetSemiExo (contact me)')
end

%%
if ~isfield(simoptions,'aprimedependsonage')
    simoptions.aprimedependsonage=0;
end

l_a=length(n_a);

N_a=prod(n_a);
N_a1=prod(n_a1);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_u=prod(simoptions.n_u);

if N_a1==0
    n_a=n_a2;
else
    n_a=[n_a1,n_a2];
end

% n_bothz=[n_semiz,n_z];
N_bothz=N_semiz*N_z;

%%
if isfield(simoptions,'n_e')
    N_e=prod(simoptions.n_e);
    jequaloneDistKron=reshape(jequaloneDist,[N_a*N_bothz*N_e,1]);
    Policy=reshape(Policy,[size(Policy,1),N_a,N_bothz*N_e,N_j]);
    n_bothze=[n_semiz,n_z,simoptions.n_e];
    N_bothze=N_bothz*N_e;
else
    jequaloneDistKron=reshape(jequaloneDist,[N_a*N_bothz,1]);
    Policy=reshape(Policy,[size(Policy,1),N_a,N_bothz,N_j]);
    n_bothze=[n_semiz,n_z];
    N_bothze=N_bothz;
end
% NOTE: have rolled e into z


%% riskyasset transitions
Policy_a2prime=zeros(N_a,N_bothze,N_u,2,N_j,'gpuArray'); % the lower grid point
PolicyProbs=zeros(N_a,N_bothze,N_u,2,N_j,'gpuArray'); % probabilities of grid points
whichisdforriskyasset=(simoptions.refine_d(1)+1):1:sum(simoptions.refine_d(1:3));  % is just saying which is the decision variable that influences the risky asset (it is all the decision variables)
for jj=1:N_j
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndexes,a2primeProbs]=CreateaprimePolicyRiskyAsset_Case1(Policy(1:l_d,:,:,jj),simoptions.aprimeFn, whichisdforriskyasset, n_d, n_a1,n_a2, N_bothze, simoptions.n_u, simoptions.d_grid, a2_grid, u_grid, aprimeFnParamsVec);
    % Note: aprimeIndexes and aprimeProbs are both [N_a,N_z,N_u]
    % Note: aprimeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the aprimeProbs are the probability of this lower point (prob of upper point is just 1 minus this).
    Policy_a2prime(:,:,:,1,jj)=a2primeIndexes; % lower grid point
    Policy_a2prime(:,:,:,2,jj)=a2primeIndexes+1; % upper grid point
    % Encode the u probabilities (pi_u) into the PolicyProbs
    PolicyProbs(:,:,:,1,jj)=a2primeProbs.*shiftdim(pi_u,-2); % lower grid point probability (and probability of u)
    PolicyProbs(:,:,:,2,jj)=(1-a2primeProbs).*shiftdim(pi_u,-2); % upper grid point probability (and probability of u)
end

if l_a==1 % just riskyasset
    Policy_aprime=Policy_a2prime;
elseif l_a==2 % one other asset, then riskyasset
    Policy_aprime(:,:,:,1,:)=reshape(Policy(l_d+1,:,:,:),[N_a,N_bothze,1,1,N_j])+n_a(1)*(Policy_a2prime(:,:,:,1,:)-1);
    Policy_aprime(:,:,:,2,:)=reshape(Policy(l_d+1,:,:,:),[N_a,N_bothze,1,1,N_j])+n_a(1)*Policy_a2prime(:,:,:,1,:); % Note: upper grid point minus 1 is anyway just lower grid point
elseif l_a==3 % two other assets, then riskyasset
    Policy_aprime(:,:,:,1,:)=reshape(Policy(l_d+1,:,:,:),[N_a,N_bothze,1,1,N_j])+n_a(1)*reshape(Policy(l_d+1,:,:,:),[N_a,N_bothze,1,1,N_j])+n_a(1)*n_a(2)*(Policy_a2prime(:,:,:,1,:)-1);
    Policy_aprime(:,:,:,2,:)=reshape(Policy(l_d+1,:,:,:),[N_a,N_bothze,1,1,N_j])+n_a(1)*reshape(Policy(l_d+1,:,:,:),[N_a,N_bothze,1,1,N_j])+n_a(1)*n_a(2)*Policy_a2prime(:,:,:,1,:); % Note: upper grid point minus 1 is anyway just lower grid point
elseif l_a>3
    error('Only two assets other than the risky asset is allowed (email if you need this)')
end

%%
% d variables relevant for the semi-exogenous asset. 
l_d123=sum(simoptions.refine_d(1:3));
if simoptions.refine_d(4)==1
    Policy_dsemiexo=shiftdim(Policy(l_d123+1,:,:,:),1);
elseif simoptions.refine_d(4)==2
    Policy_dsemiexo=shiftdim(Policy(l_d123+1,:,:,:)+n_d(l_d123+1)*Policy(l_d123+2,:,:,:),1);
elseif simoptions.refine_d(4)==3
    Policy_dsemiexo=shiftdim(Policy(l_d123+1,:,:,:)+n_d(l_d123+1)*Policy(l_d123+2,:,:,:)+n_d(l_d123+1)*n_d(l_d123+2)*Policy(l_d123+3,:,:,:),1); 
elseif simoptions.refine_d(4)==4
    Policy_dsemiexo=shiftdim(Policy(l_d123+1,:,:,:)+n_d(l_d123+1)*Policy(l_d123+2,:,:,:)+n_d(l_d123+1)*n_d(l_d123+2)*Policy(l_d123+3,:,:,:)+n_d(l_d123+1)*n_d(l_d123+2)*n_d(l_d123+3)*Policy(l_d123+4,:,:,:),1);
end

% Note that PolicyProbs contains pi_u already.

% Note: N_z=0 is a different code
if isfield(simoptions,'n_e')
    error('Have not yet impelmented N_e in StationaryDist_FHorz_Case1_RiskyAssetSemiExo (contact me)')
else
    StationaryDist=StationaryDist_FHorz_Case1_Iteration_SemiExo_uProbs_raw(jequaloneDistKron,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,N_a,N_semiz,N_z,N_u,N_j,pi_semiz_J,pi_z_J,Parameters);
end


if simoptions.parallel==2
    StationaryDist=gpuArray(StationaryDist); % move output to gpu
end
if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,n_bothze,N_j]);
else
    % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDist,[N_a,N_bothze,N_j]);
end



end
