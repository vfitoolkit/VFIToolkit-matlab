function StationaryDist=StationaryDist_FHorz_ExpAsset(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions)

%% Setup related to experience asset
n_d2=n_d(end-simoptions.l_dexperienceasset+1:end);
% Split endogenous assets into the standard ones and the experience asset
if length(n_a)<=simoptions.experienceasset
    n_a1=0;
else
    n_a1=n_a(1:end-simoptions.experienceasset);
end
n_a2=n_a(end-simoptions.experienceasset+1:end); % last simoptions.experienceasset dims are the experience asset

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

% aprimeFnParamNames in same fashion
l_d2=length(n_d2);
l_a2=length(n_a2);
temp=getAnonymousFnInputNames(simoptions.aprimeFn);
if length(temp)>(l_d2+l_a2+(l_a2>=2))
    aprimeFnParamNames={temp{l_d2+l_a2+(l_a2>=2)+1:end}}; % the first inputs will always be (d2,a2)
else
    aprimeFnParamNames={};
end

N_e=prod(simoptions.n_e);
N_z=prod(n_z);

%%
if N_z==0 && N_e==0
    StationaryDist=StationaryDist_FHorz_ExpAsset_noz(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,N_j,d_grid,a2_grid,Parameters,simoptions);
    return
end

%%
l_d=length(n_d);
l_a=length(n_a);
l_a2=simoptions.experienceasset; % number of a2 (experience-asset) dims
if isscalar(n_a1) && n_a1==0
    l_a1=0; N_a1=0;
else
    l_a1=length(n_a1); N_a1=prod(n_a1);
end

N_a=prod(n_a);

%%
if N_z==0
    % Note: N_z==0 && N_e==0 already got sent elsewhere
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

%% expasset transitions
% Policy is currently about d and a1prime. Convert it to being about aprime
% as that is what we need for simulation, and we can then just send it to standard Case1 commands.
N_probs=2^l_a2; % 2 points (lower and upper index) per dimension of a2
Policy_aprime=zeros(N_a,N_ze,N_probs,N_j,'gpuArray');
PolicyProbs=zeros(N_a,N_ze,N_probs,N_j,'gpuArray'); % third dimension indexes the interpolation corners
whichisdforexpasset=length(n_d)-simoptions.l_dexperienceasset+1:length(n_d);  % is just saying which is the decision variable that influences the experience asset (it is the 'last' decision variable)
for jj=1:N_j
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [aprimeIndexes, aprimeProbs]=CreateaprimePolicyExperienceAsset(Policy(:,:,:,jj),simoptions.aprimeFn, whichisdforexpasset, n_d, n_a1,n_a2, N_ze, d_grid, a2_grid, aprimeFnParamsVec);
    % Note: aprimeIndexes and aprimeProbs are both [N_a,N_z]
    % Note: aprimeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the aprimeProbs are the probability of this lower point (prob of upper point is just 1 minus this).

    if l_a2==1
        if l_a1==0 % just experience asset
            Policy_aprime(:,:,1,jj)=aprimeIndexes;
            Policy_aprime(:,:,2,jj)=aprimeIndexes+1;
        elseif l_a1==1 % one other asset, then experience asset
            Policy_aprime(:,:,1,jj)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(aprimeIndexes-1);
            Policy_aprime(:,:,2,jj)=Policy_aprime(:,:,1,jj)+n_a(1);
        elseif l_a1==2 % two other assets, then experience asset
            Policy_aprime(:,:,1,jj)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(shiftdim(Policy(l_d+2,:,:,jj),1)-1)+prod(n_a(1:2))*(aprimeIndexes-1);
            Policy_aprime(:,:,2,jj)=Policy_aprime(:,:,1,jj)+prod(n_a(1:2));
        elseif l_a1==3 % three other assets, then experience asset
            Policy_aprime(:,:,1,jj)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(shiftdim(Policy(l_d+2,:,:,jj),1)-1)+prod(n_a(1:2))*(shiftdim(Policy(l_d+3,:,:,jj),1)-1)+prod(n_a(1:3))*(aprimeIndexes-1);
            Policy_aprime(:,:,2,jj)=Policy_aprime(:,:,1,jj)+prod(n_a(1:3));
        elseif l_a1==4 % four other assets, then experience asset
            Policy_aprime(:,:,1,jj)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(shiftdim(Policy(l_d+2,:,:,jj),1)-1)+prod(n_a(1:2))*(shiftdim(Policy(l_d+3,:,:,jj),1)-1)+prod(n_a(1:3))*(shiftdim(Policy(l_d+4,:,:,jj),1)-1)+prod(n_a(1:4))*(aprimeIndexes-1);
            Policy_aprime(:,:,2,jj)=Policy_aprime(:,:,1,jj)+prod(n_a(1:4));
        else
            error('Not yet implemented experience asset with more than four standard assets')
        end

        PolicyProbs(:,:,1,jj)=aprimeProbs;
        PolicyProbs(:,:,2,jj)=1-aprimeProbs;
    else
        % l_a2==2: aprimeIndexes/aprimeProbs are [N_a,l_a2,N_ze] per-dim factored.
        % Kron-fold to N_probs=4 corners (mirrors the _noz sibling and SimPanelIndexes).
        n_a2_1=n_a2(1);
        loIdx_1=reshape(aprimeIndexes(:,1,:),[N_a,N_ze]);
        loIdx_2=reshape(aprimeIndexes(:,2,:),[N_a,N_ze]);
        prob_1=reshape(aprimeProbs(:,1,:),[N_a,N_ze]);
        prob_2=reshape(aprimeProbs(:,2,:),[N_a,N_ze]);
        if l_a1==0
            a1primeIndexes=[];
        elseif l_a1==1
            a1primeIndexes=shiftdim(Policy(l_d+1,:,:,jj),1);
        elseif l_a1==2
            a1primeIndexes=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(shiftdim(Policy(l_d+2,:,:,jj),1)-1);
        elseif l_a1==3
            a1primeIndexes=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(shiftdim(Policy(l_d+2,:,:,jj),1)-1)+prod(n_a(1:2))*(shiftdim(Policy(l_d+3,:,:,jj),1)-1);
        elseif l_a1==4
            a1primeIndexes=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(shiftdim(Policy(l_d+2,:,:,jj),1)-1)+prod(n_a(1:2))*(shiftdim(Policy(l_d+3,:,:,jj),1)-1)+prod(n_a(1:3))*(shiftdim(Policy(l_d+4,:,:,jj),1)-1);
        else
            error('Not yet implemented experience asset with more than four standard assets')
        end
        bits=[0 0; 1 0; 0 1; 1 1];
        for c=1:N_probs
            b1=bits(c,1); b2=bits(c,2);
            a2_kron=(loIdx_1+b1)+n_a2_1*((loIdx_2+b2)-1);
            if l_a1==0
                Policy_aprime(:,:,c,jj)=a2_kron;
            else
                Policy_aprime(:,:,c,jj)=a1primeIndexes+N_a1*(a2_kron-1);
            end
            p1=prob_1; if b1==1, p1=1-p1; end
            p2=prob_2; if b2==1, p2=1-p2; end
            PolicyProbs(:,:,c,jj)=p1.*p2;
        end
    end
end



%%
if simoptions.gridinterplayer==0
    % Note: N_z=0 && N_e=0 is a different code
    if N_e==0 % just z
        StationaryDist=StationaryDist_FHorz_Iteration_nProbs_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_probs,N_a,N_z,N_j,pi_z_J,Parameters);
    elseif N_z==0 % just e
        StationaryDist=StationaryDist_FHorz_Iteration_nProbs_noz_e_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_probs,N_a,N_e,N_j,simoptions.pi_e_J,Parameters);
    else % both z and e
        StationaryDist=StationaryDist_FHorz_Iteration_nProbs_e_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_probs,N_a,N_z,N_e,N_j,pi_z_J,simoptions.pi_e_J,Parameters);
    end
elseif simoptions.gridinterplayer==1
    % The interpolation layer adds the two a1prime grid points: duplicate the a2 corners, the
    % first N_probs keeping the lower a1 point and the next N_probs taking the upper one.
    Policy_aprime=repmat(Policy_aprime,1,1,2,1);
    PolicyProbs=repmat(PolicyProbs,1,1,2,1);
    % Policy_aprime(:,:,1:N_probs,:) lower grid point for a1 is unchanged
    Policy_aprime(:,:,N_probs+1:2*N_probs,:)=Policy_aprime(:,:,N_probs+1:2*N_probs,:)+1; % add one to a1, to get upper grid point (a1 is the fastest-varying dim of the aprime index)

    % L2flag override (1=force all weight to lower, 2=usual, 3=force all weight to upper)
    L2index=Policy(end-1,:,:,:); % L2 index (end-1 because end is L2flag)
    L2flag=Policy(end,:,:,:);
    L2index(L2flag==1)=1;                        % force all weight to lower grid point
    L2index(L2flag==3)=simoptions.ngridinterp+2; % force all weight to upper grid point
    aprimeProbs_upper=reshape(shiftdim((L2index-1)/(simoptions.ngridinterp+1),1),[N_a,N_ze,1,N_j]); % probability of upper grid point
    PolicyProbs(:,:,1:N_probs,:)=PolicyProbs(:,:,1:N_probs,:).*(1-aprimeProbs_upper); % lower a1
    PolicyProbs(:,:,N_probs+1:2*N_probs,:)=PolicyProbs(:,:,N_probs+1:2*N_probs,:).*aprimeProbs_upper; % upper a1
    N_probs=2*N_probs;

    % Note: N_z=0 && N_e=0 is a different code
    if N_e==0 % just z
        StationaryDist=StationaryDist_FHorz_Iteration_nProbs_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_probs,N_a,N_z,N_j,pi_z_J,Parameters);
    elseif N_z==0 % just e
        StationaryDist=StationaryDist_FHorz_Iteration_nProbs_noz_e_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_probs,N_a,N_e,N_j,simoptions.pi_e_J,Parameters);
    else % both z and e
        StationaryDist=StationaryDist_FHorz_Iteration_nProbs_e_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_probs,N_a,N_z,N_e,N_j,pi_z_J,simoptions.pi_e_J,Parameters);
    end
end



if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,n_ze,N_j]);
% else
    % If 1 then leave output in Kron form
    % StationaryDist=reshape(StationaryDist,[N_a,N_ze,N_j]);
end

end
