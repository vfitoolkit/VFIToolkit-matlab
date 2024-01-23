function StationaryDist=StationaryDist_FHorz_Case1_ResidAsset(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions)

%% Check for the age weights parameter, and make sure it is a row vector
if size(Parameters.(AgeWeightParamNames{1}),2)==1 % Seems like column vector
    Parameters.(AgeWeightParamNames{1})=Parameters.(AgeWeightParamNames{1})'; 
    % Note: assumed there is only one AgeWeightParamNames
end

%% Check that the age one distribution is of mass one
if abs(sum(jequaloneDist(:))-1)>10^(-9)
    error('The jequaloneDist must be of mass one')
end

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
if isfield(simoptions,'n_e')
    if n_z(1)==0
        error('Not yet implemented n_z=0 with n_e and residualasset, email me and I will do it (or you can just pretend by using n_z=1 and pi_z=1, not using the value of z anywhere)')
    else
%         StationaryDist=StationaryDist_FHorz_Case1_ResidAsset_e(jequaloneDist,AgeWeightParamNames,Policy,n_d1,n_d2,n_a,n_z,N_j,pi_z,rprimeFn,Parameters,simoptions);
    end
    return
end

if n_z(1)==0
    error('Not yet implemented n_z=0 with residualasset, (you can just pretend by using n_z=1 and pi_z=1, not using the value of z anywhere)')
end

N_a=prod(n_a);
N_r=prod(n_r);
N_z=prod(n_z);

jequaloneDistKron=reshape(jequaloneDist,[N_a*N_r*N_z,1]);
if simoptions.parallel~=2 && simoptions.parallel~=4
    Policy=gather(Policy);
    jequaloneDistKron=gather(jequaloneDistKron);    
    pi_z_J=gather(pi_z_J);
end

% Get policy for aprime, then get policy for rprime, then combine (all just in terms of current state)
Policy=reshape(Policy,[size(Policy,1),N_a*N_r,N_z,N_j]);

if l_a==1
    Policy_aprime=shiftdim(Policy(l_d+1,:,:,:),1);
elseif l_a==2
    Policy_aprime=shiftdim(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1),1);
elseif l_a==3
    Policy_aprime=shiftdim(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:,:)-1),1);
elseif l_a==4
    Policy_aprime=shiftdim(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(Policy(l_d+4,:,:,:)-1),1);
end
Policy_aprime=reshape(Policy_aprime,[N_a*N_r,N_z,1,N_j]);

Policy_rprime=zeros(N_a*N_r,N_z,1,N_j,'gpuArray'); % the lower grid points
PolicyProbs=zeros(N_a*N_r,N_z,2,N_j,'gpuArray'); % The fourth dimension is lower/upper grid point
for jj=1:N_j
    rprimeFnParamsVec=CreateVectorFromParams(Parameters, rprimeFnParamNames,jj);
    [rprimeIndexes, rprimeProbs]=CreaterprimePolicyResidualAsset_Case1(Policy(:,:,:,jj),rprimeFn, n_d, n_a, n_r, n_z, gpuArray(simoptions.d_grid), a_grid, r_grid, gpuArray(simoptions.z_grid), rprimeFnParamsVec);
    % rprimeIndexes is [N_a*N_r,N_z], rprimeProbs is [N_a*N_r,N_z]
    Policy_rprime(:,:,1,jj)=rprimeIndexes;
    PolicyProbs(:,:,1,jj)=rprimeProbs;
    PolicyProbs(:,:,2,jj)=1-rprimeProbs;
end

Policy_arprime=zeros(N_a*N_r,N_z,2,N_j,'gpuArray');
Policy_arprime(:,:,1,:)=Policy_aprime+N_a*(Policy_rprime-1);
Policy_arprime(:,:,2,:)=Policy_aprime+N_a*(Policy_rprime+1-1);

if simoptions.iterate==0
    PolicyProbs=gather(PolicyProbs); % simulation is always with cpu
    Policy_arprime=gather(Policy_arprime);
    if simoptions.parallel>=3
        % Sparse matrix is not relevant for the simulation methods, only for iteration method
        simoptions.parallel=2; % will simulate on parallel cpu, then transfer solution to gpu
    end
    StationaryDist=StationaryDist_FHorz_Case1_Simulation_TwoProbs_raw(jequaloneDistKron,AgeWeightParamNames,Policy_arprime,PolicyProbs,N_a*N_r,N_z,N_j,pi_z_J, Parameters, simoptions);
elseif simoptions.iterate==1
    StationaryDist=StationaryDist_FHorz_Case1_Iteration_TwoProbs_raw(jequaloneDistKron,AgeWeightParamNames,Policy_arprime,PolicyProbs,N_a*N_r,N_z,N_j,pi_z_J,Parameters); % zero is n_d, because we already converted Policy to only contain aprime
end

if simoptions.parallel==2
    StationaryDist=gpuArray(StationaryDist); % move output to gpu
end
if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,n_r,n_z,N_j]);
else
    % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDist,[N_a,N_r,N_z,N_j]);
end

end
