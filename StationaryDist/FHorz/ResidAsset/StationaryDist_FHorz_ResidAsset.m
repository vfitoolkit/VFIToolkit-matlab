function StationaryDist=StationaryDist_FHorz_ResidAsset(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions)


%% Setup related to experience asset
if isfield(simoptions,'rprimeFn')
    rprimeFn=simoptions.rprimeFn;
else
    error('To use an residual asset you must define simoptions.rprimeFn')
end
if ~isfield(simoptions,'a_grid')
    error('To use an residual asset you must define simoptions.a_grid')
end

n_r=n_a(end);
n_a=n_a(1:end-1);

a_grid=gpuArray(simoptions.a_grid(1:sum(n_a)));
r_grid=gpuArray(simoptions.a_grid(sum(n_a)+1:end));

% rprimeFnParamNames in same fashion
if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);
temp=getAnonymousFnInputNames(rprimeFn);
if length(temp)>(l_d+l_a+l_a+l_z)
    rprimeFnParamNames={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d2,a2)
else
    rprimeFnParamNames={};
end

%%
if n_z(1)==0
    error('Not yet implemented n_z=0 with residualasset, (you can just pretend by using n_z=1 and pi_z=1, not using the value of z anywhere)')
end

%%
N_a=prod(n_a);
N_r=prod(n_r);
N_z=prod(n_z);

%%
if isfield(simoptions,'n_e')
    N_e=prod(simoptions.n_e);
    jequaloneDistKron=reshape(jequaloneDist,[N_a*N_r*N_z*N_e,1]);
    Policy=reshape(Policy,[size(Policy,1),N_a*N_r,N_z*N_e,N_j]);
    n_ze=[n_z,simoptions.n_e];
    N_ze=N_z*N_e;
else
    jequaloneDistKron=reshape(jequaloneDist,[N_a*N_r*N_z,1]);
    Policy=reshape(Policy,[size(Policy,1),N_a*N_r,N_z,N_j]);
    n_ze=n_z;
    N_ze=N_z;
end
% NOTE: have rolled e into z


%% Residual asset
% Get policy for aprime, then get policy for rprime, then combine (all just in terms of current state)
if l_a==1
    Policy_aprime=shiftdim(Policy(l_d+1,:,:,:),1);
elseif l_a==2
    Policy_aprime=shiftdim(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1),1);
elseif l_a==3
    Policy_aprime=shiftdim(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:,:)-1),1);
elseif l_a==4
    Policy_aprime=shiftdim(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(Policy(l_d+4,:,:,:)-1),1);
end
Policy_aprime=reshape(Policy_aprime,[N_a*N_r,N_ze,1,N_j]);

Policy_rprime=zeros(N_a*N_r,N_ze,1,N_j,'gpuArray'); % the lower grid points
PolicyProbs=zeros(N_a*N_r,N_ze,2,N_j,'gpuArray'); % The fourth dimension is lower/upper grid point
for jj=1:N_j
    rprimeFnParamsVec=CreateVectorFromParams(Parameters, rprimeFnParamNames,jj);
    [rprimeIndexes, rprimeProbs]=CreaterprimePolicyResidualAsset_Case1(Policy(:,:,:,jj),rprimeFn, n_d, n_a, n_r, n_ze, gpuArray(simoptions.d_grid), a_grid, r_grid, gpuArray(simoptions.z_grid), rprimeFnParamsVec);
    % rprimeIndexes is [N_a*N_r,N_ze], rprimeProbs is [N_a*N_r,N_ze]
    Policy_rprime(:,:,1,jj)=rprimeIndexes;
    PolicyProbs(:,:,1,jj)=rprimeProbs;
    PolicyProbs(:,:,2,jj)=1-rprimeProbs;
end

Policy_arprime=zeros(N_a*N_r,N_ze,2,N_j,'gpuArray');
Policy_arprime(:,:,1,:)=Policy_aprime+N_a*(Policy_rprime-1);
Policy_arprime(:,:,2,:)=Policy_aprime+N_a*(Policy_rprime+1-1);


%%   
% Note: N_z=0 is a different code
if isfield(simoptions,'n_e')
    error('Have not yet impelmented N_e in StationaryDist_FHorz_Case1_ResidAsset (contact me)')
else % no e
    StationaryDist=StationaryDist_FHorz_Iteration_TwoProbs_raw(jequaloneDistKron,AgeWeightParamNames,Policy_arprime,PolicyProbs,N_a*N_r,N_ze,N_j,pi_z_J,Parameters); % zero is n_d, because we already converted Policy to only contain aprime
end

if simoptions.parallel==2
    StationaryDist=gpuArray(StationaryDist); % move output to gpu
end
if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,n_r,n_ze,N_j]);
else
    % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDist,[N_a,N_r,N_ze,N_j]);
end

end
