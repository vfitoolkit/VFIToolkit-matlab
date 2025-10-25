function StationaryDist=StationaryDist_FHorz_RiskyAsset(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions)


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

% Sort out decision variables, need to get those for riskyasset
if ~isfield(simoptions,'refine_d')
    % When not using refine_d, everything is implicitly a d3
    simoptions.refine_d=[0,0,length(n_d)];
end
n_d23=n_d(simoptions.refine_d(1)+1:sum(simoptions.refine_d(1:3))); % decision variables for riskyasset

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
n_u=simoptions.n_u;
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


if isfield(simoptions,'n_e')
    N_e=prod(simoptions.n_e);
else
    N_e=0;
end

%%
if n_z(1)==0 && N_e==0
    error('Have not yet impelmented N_z=0 and N_e=0 in StationaryDist_FHorz_RiskyAsset (contact me)')
end
%%
if ~isfield(simoptions,'aprimedependsonage')
    simoptions.aprimedependsonage=0;
end

N_a1=prod(n_a1);

if N_a1==0
    n_a=n_a2;
else
    n_a=[n_a1,n_a2];
end



%%
l_d=length(n_d);
l_a=length(n_a);

N_a=prod(n_a);
N_z=prod(n_z);
N_u=prod(n_u);

%%
if N_z==0
    % Note: n_z(1)==0 && N_e==0 already got sent elsewhere
    n_ze=simoptions.n_e;
    N_ze=N_e;
else
    if N_e==0
        n_ze=n_z;
        N_ze=N_z;
    else
        n_ze=[n_z,simoptions.n_e];
        N_ze=N_z*N_e;
    end
end

jequaloneDist=gpuArray(jequaloneDist); % make sure it is on gpu
jequaloneDist=reshape(jequaloneDist,[N_a*N_ze,1]);
Policy=reshape(Policy,[size(Policy,1),N_a,N_ze,N_j]);

%% riskyasset transitions
Policy_a2prime=zeros(N_a,N_ze,N_u,2,N_j,'gpuArray'); % the lower grid point
PolicyProbs=zeros(N_a,N_ze,N_u,2,N_j,'gpuArray'); % probabilities of grid points
whichisdforriskyasset=(simoptions.refine_d(1)+1):1:length(n_d);  % is just saying which is the decision variable that influences the risky asset (it is all the decision variables)
for jj=1:N_j
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndexes,a2primeProbs]=CreateaprimePolicyRiskyAsset_Case1(Policy(1:l_d,:,:,jj),simoptions.aprimeFn, whichisdforriskyasset, n_d, n_a1,n_a2, N_ze, simoptions.n_u, simoptions.d_grid, a2_grid, u_grid, aprimeFnParamsVec);
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
    Policy_aprime(:,:,:,1,:)=reshape(Policy(l_d+1,:,:,:),[N_a,N_ze,1,1,N_j])+n_a(1)*(Policy_a2prime(:,:,:,1,:)-1);
    Policy_aprime(:,:,:,2,:)=reshape(Policy(l_d+1,:,:,:),[N_a,N_ze,1,1,N_j])+n_a(1)*Policy_a2prime(:,:,:,1,:); % Note: upper grid point minus 1 is anyway just lower grid point
elseif l_a==3 % two other assets, then riskyasset
    Policy_aprime(:,:,:,1,:)=reshape(Policy(l_d+1,:,:,:),[N_a,N_ze,1,1,N_j])+n_a(1)*reshape(Policy(l_d+1,:,:,:),[N_a,N_ze,1,1,N_j])+n_a(1)*n_a(2)*(Policy_a2prime(:,:,:,1,:)-1);
    Policy_aprime(:,:,:,2,:)=reshape(Policy(l_d+1,:,:,:),[N_a,N_ze,1,1,N_j])+n_a(1)*reshape(Policy(l_d+1,:,:,:),[N_a,N_ze,1,1,N_j])+n_a(1)*n_a(2)*Policy_a2prime(:,:,:,1,:); % Note: upper grid point minus 1 is anyway just lower grid point
elseif l_a>3
    error('Only two assets other than the risky asset is allowed (email if you need this)')
end

Policy_aprime=reshape(Policy_aprime,[N_a,N_ze,N_u*2,N_j]);
PolicyProbs=reshape(PolicyProbs,[N_a,N_ze,N_u*2,N_j]);


%%
if simoptions.gridinterplayer==0
    % Note: N_z=0 && N_e=0 is a different code
    if N_e==0 % just z
        StationaryDist=StationaryDist_FHorz_Iteration_nProbs_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_u*2,N_a,N_z,N_j,pi_z_J,Parameters); % zero is n_d, because we already converted Policy to only contain aprime
        StationaryDist=gpuArray(StationaryDist);
    elseif N_z==0 % just e
        StationaryDist=StationaryDist_FHorz_Iteration_nProbs_noz_e_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_u*2,N_a,N_e,N_j,simoptions.pi_e_J,Parameters); % zero is n_d, because we already converted Policy to only contain aprime
        StationaryDist=gpuArray(StationaryDist);
    else % both z and e
        StationaryDist=StationaryDist_FHorz_Iteration_nProbs_e_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_u*2,N_a,N_z,N_e,N_j,pi_z_J,simoptions.pi_e_J,Parameters); % zero is n_d, because we already converted Policy to only contain aprime
        StationaryDist=gpuArray(StationaryDist);
    end
elseif simoptions.gridinterplayer==1
    % (a,z,2,j)
    Policy_aprime=repmat(Policy_aprime,1,1,2,1);
    PolicyProbs=repmat(PolicyProbs,1,1,2,1);
    % Policy_aprime(:,:,:,1:2*N_u,:) lower grid point for a1 is unchanged 
    Policy_aprime(:,:,2*N_u+1:end,:)=Policy_aprime(:,:,2*N_u+1:end,:)+1; % add one to a1, to get upper grid point

    aprimeProbs_upper=reshape(shiftdim((Policy(end,:,:,:)-1)/(simoptions.ngridinterp+1),1),[N_a,N_ze,1,N_j]); % probability of upper grid point (from L2 index)
    PolicyProbs(:,:,1:2*N_u,:)=PolicyProbs(:,:,1:2*N_u,:).*(1-aprimeProbs_upper); % lower a1
    PolicyProbs(:,:,2*N_u+1:end,:)=PolicyProbs(:,:,2*N_u+1:end,:).*aprimeProbs_upper; % upper a1

    % Note: N_z=0 && N_e=0 is a different code
    if N_e==0 % just z
        StationaryDist=StationaryDist_FHorz_Iteration_nProbs_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,PolicyProbs,2*N_u*2,N_a,N_z,N_j,pi_z_J,Parameters);
    elseif N_z==0 % just e
        StationaryDist=StationaryDist_FHorz_Iteration_nProbs_noz_e_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,PolicyProbs,2*N_u*2,N_a,N_e,N_j,simoptions.pi_e_J,Parameters);
    else % both z and e
        StationaryDist=StationaryDist_FHorz_Iteration_nProbs_e_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,PolicyProbs,2*N_u*2,N_a,N_z,N_e,N_j,pi_z_J,simoptions.pi_e_J,Parameters);
    end
end


if simoptions.parallel==2
    StationaryDist=gpuArray(StationaryDist); % move output to gpu
end
if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,n_ze,N_j]);
% else
%     % If 1 then leave output in Kron form
%     StationaryDist=reshape(StationaryDist,[N_a,N_ze,N_j]);
end

end
