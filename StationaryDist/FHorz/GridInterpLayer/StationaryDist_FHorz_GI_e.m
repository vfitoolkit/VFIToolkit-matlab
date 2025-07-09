function StationaryDist=StationaryDist_FHorz_GI_e(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,n_e,N_j,pi_z_J,pi_e_J,Parameters,simoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

l_d=length(n_d);
if N_d==0
    l_d=0;
end

%%
jequaloneDist=reshape(jequaloneDist,[N_a*N_z*N_e,1]);

Policy=reshape(Policy,[size(Policy,1),N_a,N_z*N_e,1,N_j]);

pi_z_J=gather(pi_z_J);

%% Switch Policy to being on the grid (rather than gridded interpolation)
% and throw out the decion variable policies while we are at it
Policy_aprime=zeros(N_a,N_z*N_e,2,N_j,'gpuArray'); % the lower grid point
PolicyProbs=zeros(N_a,N_z*N_e,2,N_j,'gpuArray'); % The fourth dimension is lower/upper grid point
if isscalar(n_a)
    Policy_aprime(:,:,1,:)=shiftdim(Policy(l_d+1,:,:,1,:),1); % lower grid point
    Policy_aprime(:,:,2,:)=shiftdim(Policy(l_d+1,:,:,1,:),1)+1; % upper grid point
    aprimeProbs=shiftdim((simoptions.ngridinterp+1-Policy(l_d+2,:,:,1,:)+1)/(simoptions.ngridinterp+1),1); % probability of lower grid point
    PolicyProbs(:,:,1,:)=aprimeProbs;
    PolicyProbs(:,:,2,:)=1-aprimeProbs;
    % aprimeProbs_upper=shiftdim((Policy(l_d+2,:,:,1,:)-1)/(simoptions.ngridinterp+1),1); % probability of upper grid point
    % PolicyProbs(:,:,1,:)=1-aprimeProbs_upper;
    % PolicyProbs(:,:,2,:)=aprimeProbs_upper;
else
    error('length(n_a)>1 not yet supported here, let me know if you want it')
end

%%
StationaryDist=StationaryDist_FHorz_Iteration_TwoProbs_e_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_a,N_z,N_e,N_j,pi_z_J,pi_e_J,Parameters);

if simoptions.parallel==2
    StationaryDist=gpuArray(StationaryDist); % move output to gpu
end
if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,n_z,n_e,N_j]);
else
    % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDist,[N_a,N_z,N_e,N_j]);
end

end
