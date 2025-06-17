function StationaryDist=StationaryDist_FHorz_SemiExo_GI_e(jequaloneDistKron,AgeWeightParamNames,Policy,n_d,n_a,n_semiz,n_z,n_e,N_j,pi_semiz_J,pi_z_J,pi_e_J,Parameters,simoptions)

%% Decision variables that determine semi-exogenous state
if length(n_d)>simoptions.l_dsemiz
    n_d1=n_d(1:end-simoptions.l_dsemiz);
    l_d1=length(n_d1);
    n_d2=n_d(end-simoptions.l_dsemiz+1:end);  % decision variable that controls semi-exogenous state
else
    % n_d1=0;
    l_d1=0;
    n_d2=n_d;  % decision variable that controls semi-exogenous state
end
l_d2=length(n_d2); % wouldn't be here if no d2

%%
l_d=length(n_d);

N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_e=prod(n_e);

N_bothze=N_semiz*N_z*N_e;

if ~isfield(simoptions,'loopovere')
    simoptions.loopovere=0; % default is parallel over e, 1 will loop over e, 2 will parfor loop over e
end

%%
jequaloneDistKron=reshape(jequaloneDistKron,[N_a*N_semiz*N_z*N_e,1]);
jequaloneDistKron=gather(jequaloneDistKron);

Policy=reshape(Policy,[size(Policy,1),N_a,N_bothze,1,N_j]);

pi_z_J=gather(pi_z_J);
pi_e_J=gather(pi_e_J);


%% Switch Policy to being on the grid (rather than gridded interpolation)
% and throw out the decion variable policies while we are at it
Policy_aprime=zeros(N_a,N_bothze,2,N_j,'gpuArray'); % the lower grid point
PolicyProbs=zeros(N_a,N_bothze,2,N_j,'gpuArray'); % The fourth dimension is lower/upper grid point
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

%% Policy for semi-exogenous shocks
% d2 is the variable relevant for the semi-exogenous asset. 
if l_d2==1
    Policy_dsemiexo=shiftdim(Policy(l_d1+1,:,:,:),1);
elseif l_d2==2
    Policy_dsemiexo=shiftdim(Policy(l_d1+1,:,:,:)+n_d(l_d1+1)*Policy(l_d1+2,:,:,:),1);
elseif l_d2==3
    Policy_dsemiexo=shiftdim(Policy(l_d1+1,:,:,:)+n_d(l_d1+1)*Policy(l_d1+2,:,:,:)+n_d(l_d1+1)*n_d(l_d1+2)*Policy(l_d1+3,:,:,:),1); 
elseif l_d2==4
    Policy_dsemiexo=shiftdim(Policy(l_d1+1,:,:,:)+n_d(l_d1+1)*Policy(l_d1+2,:,:,:)+n_d(l_d1+1)*n_d(l_d1+2)*Policy(l_d1+3,:,:,:)+n_d(l_d1+1)*n_d(l_d1+2)*n_d(l_d1+3)*Policy(l_d1+4,:,:,:),1);
end


%%
StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_TwoProbs_e_raw(jequaloneDistKron,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,N_a,N_semiz,N_z,N_e,N_j,pi_semiz_J,pi_z_J,pi_e_J,Parameters);

if simoptions.parallel==2
    StationaryDist=gpuArray(StationaryDist); % move output to gpu
end
if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,simoptions.n_semiz,n_z,n_e,N_j]);
else
    % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDist,[N_a,N_semiz,N_z,N_e,N_j]);
end

end
