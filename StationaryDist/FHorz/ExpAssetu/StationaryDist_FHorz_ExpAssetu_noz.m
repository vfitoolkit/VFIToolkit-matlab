function StationaryDist=StationaryDist_FHorz_ExpAssetu_noz(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_u,N_j,d_grid,a2_grid,u_grid,pi_u,Parameters,simoptions)

%% Setup related to experience asset
n_d2=n_d(end);
% Split endogenous assets into the standard ones and the experience asset
if isscalar(n_a)
    n_a1=0;
else
    n_a1=n_a(1:end-1);
end
n_a2=n_a(end); % n_a2 is the experience asset

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

%%
l_d=length(n_d);
l_a=length(n_a);

N_a=prod(n_a);
N_u=prod(n_u);

jequaloneDistKron=reshape(jequaloneDist,[N_a,1]);
Policy=reshape(Policy,[size(Policy,1),N_a,N_j]);


%%
% Policy is currently about d and a2prime. Convert it to being about aprime
% as that is what we need for simulation, and we can then just send it to standard Case1 commands.
Policy=reshape(Policy,[size(Policy,1),N_a,N_j]);
Policy_aprime=zeros(N_a,N_u,2,N_j,'gpuArray'); % the lower grid point
PolicyProbs=zeros(N_a,N_u,2,N_j,'gpuArray'); % probabilities of grid points
whichisdforexpasset=length(n_d);  % is just saying which is the decision variable that influences the experience asset (it is the 'last' decision variable)
for jj=1:N_j
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [aprimeIndexes, aprimeProbs]=CreateaprimePolicyExperienceAssetu_Case1(Policy(:,:,jj),simoptions.aprimeFn, whichisdforexpasset, n_d, n_a1,n_a2, 0,n_u, d_grid, a2_grid,u_grid, aprimeFnParamsVec);
    % Note: aprimeIndexes and aprimeProbs are both [N_a,N_z,N_u]
    % Note: aprimeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the aprimeProbs are the probability of this lower point (prob of upper point is just 1 minus this).

    if l_a==1 % just experienceassetu
        Policy_aprime(:,:,1,jj)=aprimeIndexes;
        Policy_aprime(:,:,2,jj)=aprimeIndexes+1;
    elseif l_a==2 % one other asset, then experience assetu
        Policy_aprime(:,:,1,jj)=shiftdim(Policy(l_d+1,:,jj),1)+n_a(1)*(aprimeIndexes-1);
        Policy_aprime(:,:,2,jj)=Policy_aprime(:,:,1,jj)+n_a(1);
    elseif l_a==3 % two other assets, then experience assetu
        Policy_aprime(:,:,1,jj)=shiftdim(Policy(l_d+1,:,jj),1)+n_a(1)*(shiftdim(Policy(l_d+2,:,jj),1)-1)+n_a(1)*n_a(2)*(aprimeIndexes-1);
        Policy_aprime(:,:,2,jj)=Policy_aprime(:,:,1,jj)+n_a(1)*n_a(2);
    else
        error('Not yet implemented experience asset with length(n_a)>3')
    end

    % Encode the u probabilities (pi_u) into the PolicyProbs
    PolicyProbs(:,:,1,jj)=aprimeProbs.*shiftdim(pi_u,-1); % lower grid point probability (and probability of u)
    PolicyProbs(:,:,2,jj)=(1-aprimeProbs).*shiftdim(pi_u,-1); % upper grid point probability (and probability of u)
end

Policy_aprime=reshape(Policy_aprime,[N_a,N_u*2,N_j]);
PolicyProbs=reshape(PolicyProbs,[N_a,N_u*2,N_j]);


%%

if simoptions.gridinterplayer==0
    StationaryDist=StationaryDist_FHorz_Iteration_nProbs_noz_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_u*2,N_a,N_j,Parameters);
    StationaryDist=gpuArray(StationaryDist); % NEED TO MOVE TAN IMPROVEMENT TO GPU

elseif simoptions.gridinterplayer==1
    % (a,u,2,j)
    Policy_aprime=repmat(Policy_aprime,1,2,1);
    PolicyProbs=repmat(PolicyProbs,1,2,1);
    % Policy_aprime(:,1:2*N_u,:) lower grid point for a1 is unchanged 
    Policy_aprime(:,2*N_u+1:end,:)=Policy_aprime(:,2*N_u+1:end,:)+1; % add one to a1, to get upper grid point

    aprimeProbs_upper=reshape(shiftdim((Policy(end,:,:)-1)/(simoptions.ngridinterp+1),1),[N_a,1,N_j]); % probability of upper grid point
    PolicyProbs(:,1:2*N_u,:)=PolicyProbs(:,1:2*N_u,:).*(1-aprimeProbs_upper); % lower a1
    PolicyProbs(:,2*N_u+1:end,:)=PolicyProbs(:,2*N_u+1:end,:).*aprimeProbs_upper; % upper a1

    StationaryDist=StationaryDist_FHorz_Iteration_nProbs_noz_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,2*N_u*2,N_a,N_j,Parameters);
end


if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,N_j]);
else
    % If 1 then leave output in Kron form
    % StationaryDist=reshape(StationaryDist,[N_a,N_j]);
end

end
