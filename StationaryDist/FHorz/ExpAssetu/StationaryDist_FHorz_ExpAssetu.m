function StationaryDist=StationaryDist_FHorz_ExpAssetu(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions)

%% Setup related to experience asset
n_d2=n_d(end);
% Split endogenous assets into the standard ones and the experience asset
if isscalar(n_a)
    n_a1=0;
else
    n_a1=n_a(1:end-1);
end
n_a2=n_a(end); % n_a2 is the experience asset

if ~isfield(simoptions,'aprimeFn')
    error('To use an experience asset you must define simoptions.aprimeFn')
end
if isfield(simoptions,'a_grid')
    % a_grid=simoptions.a_grid;
    % a1_grid=simoptions.a_grid(1:sum(n_a1));
    a2_grid=simoptions.a_grid(sum(n_a1)+1:end);
else
    error('To use an experience asset you must define simoptions.a_grid')
end
if isfield(simoptions,'d_grid')
    d_grid=simoptions.d_grid;
else
    error('To use an experience asset you must define simoptions.d_grid')
end


if isfield(simoptions,'n_u')
    n_u=simoptions.n_u;
else
    error('To use an experience assetu you must define vfoptions.n_u')
end
if isfield(simoptions,'u_grid')
    u_grid=simoptions.u_grid(:); % note, column
else
    error('To use an experience assetu you must define vfoptions.u_grid')
end
if isfield(simoptions,'pi_u')
    pi_u=simoptions.pi_u(:); % note, column
else
    error('To use an experience assetu you must define vfoptions.pi_u')
end


% Make sure u_grid and pi_u are on gpu
u_grid=gpuArray(u_grid);
pi_u=gpuArray(pi_u);
simoptions.u_grid=gpuArray(simoptions.u_grid); % needed to by some subfns

% aprimeFnParamNames in same fashion
l_d2=length(n_d2);
l_a2=length(n_a2);
l_u=length(n_u);
temp=getAnonymousFnInputNames(simoptions.aprimeFn);
if length(temp)>(l_d2+l_a2+l_u)
    aprimeFnParamNames={temp{l_d2+l_a2+l_u+1:end}}; % the first inputs will always be (d2,a2,u)
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
    StationaryDist=StationaryDist_FHorz_ExpAssetu_noz(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_u,N_j,d_grid,a2_grid,u_grid,pi_u,Parameters,simoptions);
    return
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


%% expassetu transitions
% Policy is currently about d and a2prime. Convert it to being about aprime
% as that is what we need for simulation, and we can then just send it to standard Case1 commands.
Policy_aprime=zeros(N_a,N_ze,N_u,2,N_j,'gpuArray'); % the lower grid point
PolicyProbs=zeros(N_a,N_ze,N_u,2,N_j,'gpuArray'); % probabilities of grid points
whichisdforexpasset=length(n_d);  % is just saying which is the decision variable that influences the experience asset (it is the 'last' decision variable)
for jj=1:N_j
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [aprimeIndexes, aprimeProbs]=CreateaprimePolicyExperienceAssetu_Case1(Policy(:,:,:,jj),simoptions.aprimeFn, whichisdforexpasset, n_d, n_a1,n_a2, N_ze,n_u, d_grid, a2_grid,u_grid, aprimeFnParamsVec);
    % Note: aprimeIndexes and aprimeProbs are both [N_a,N_z,N_u]
    % Note: aprimeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the aprimeProbs are the probability of this lower point (prob of upper point is just 1 minus this).

    if l_a==1 % just experienceassetu
        Policy_aprime(:,:,:,1,jj)=aprimeIndexes;
        Policy_aprime(:,:,:,2,jj)=aprimeIndexes+1;
    elseif l_a==2 % one other asset, then experience assetu
        Policy_aprime(:,:,:,1,jj)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(aprimeIndexes-1);
        Policy_aprime(:,:,:,2,jj)=Policy_aprime(:,:,:,1,jj)+n_a(1);
    elseif l_a==3 % two other assets, then experience assetu
        Policy_aprime(:,:,:,1,jj)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(shiftdim(Policy(l_d+2,:,:,jj),1)-1)+n_a(1)*n_a(2)*(aprimeIndexes-1);
        Policy_aprime(:,:,:,2,jj)=Policy_aprime(:,:,:,1,jj)+n_a(1)*n_a(2);
    else
        error('Not yet implemented experience asset with length(n_a)>3')
    end

    % Encode the u probabilities (pi_u) into the PolicyProbs
    PolicyProbs(:,:,:,1,jj)=aprimeProbs.*shiftdim(pi_u,-2); % lower grid point probability (and probability of u)
    PolicyProbs(:,:,:,2,jj)=(1-aprimeProbs).*shiftdim(pi_u,-2); % upper grid point probability (and probability of u)
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
    % If 1 then leave output in Kron form
    % StationaryDist=reshape(StationaryDist,[N_a,N_ze,N_j]);
end

end
