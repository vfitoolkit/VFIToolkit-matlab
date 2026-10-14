function StationaryDist=StationaryDist_FHorz_ExpAssetu_noz(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_u,N_j,d_grid,a2_grid,u_grid,pi_u,Parameters,simoptions)

%% Setup related to experience asset
n_d2=n_d(end-simoptions.l_dexperienceassetu+1:end); % the last l_dexperienceassetu decision variables control the experience asset
% Split endogenous assets into the standard ones and the experience asset
% simoptions.experienceassetu is the *count* of EAU dims (1 or 2).
l_a2=simoptions.experienceassetu;
if length(n_a)<=l_a2
    n_a1=0;
else
    n_a1=n_a(1:end-l_a2);
end
n_a2=n_a(end-l_a2+1:end); % last l_a2 dims are the experience asset(s)

% aprimeFnParamNames in same fashion
l_d2=length(n_d2);
% l_a2 already set from simoptions.experienceassetu above
l_u=length(n_u);
temp=getAnonymousFnInputNames(simoptions.aprimeFn);
if length(temp)>(l_d2+l_a2+l_u+(l_a2>=2))  % the (l_a2>=2) term is the 'whicha' selector slot, which aprimeFn only takes when there are two experience assets
    aprimeFnParamNames={temp{l_d2+l_a2+l_u+(l_a2>=2)+1:end}}; % the first inputs will always be (d2,a2,u)
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


%% expassetu transitions
% Policy is currently about d and a2prime. Convert it to being about aprime
% as that is what we need for simulation, and we can then just send it to standard Case1 commands.
Policy=reshape(Policy,[size(Policy,1),N_a,N_j]);
% For l_a2==1: 2 corners (lower/upper) per u. For l_a2==2: 4 corners (bilinear lattice) per u.
N_corners=2^l_a2;
Policy_aprime=zeros(N_a,N_u,N_corners,N_j,'gpuArray'); % Kron'd a-index per corner
PolicyProbs=zeros(N_a,N_u,N_corners,N_j,'gpuArray'); % corner probability times the probability of u
whichisdforexpasset=length(n_d)-simoptions.l_dexperienceassetu+1:length(n_d);  % which decision variables influence the experience asset (the last l_dexperienceassetu decision variables)
l_a1=length(n_a)-l_a2;
N_a1=prod(n_a1);
if N_a1==0
    N_a1=1; % so the Kron offset N_a1*(a2Kron-1) is well-defined; a2Kron alone is the index
end

for jj=1:N_j
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [aprimeIndexes, aprimeProbs]=CreateaprimePolicyExperienceAssetu(Policy(:,:,jj),simoptions.aprimeFn, whichisdforexpasset, n_d, n_a1,n_a2, 0,n_u, d_grid, a2_grid,u_grid, aprimeFnParamsVec);
    % Note: aprimeIndexes and aprimeProbs are both [N_a,N_z,N_u]
    % Note: aprimeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the aprimeProbs are the probability of this lower point (prob of upper point is just 1 minus this).

    % Build the a1-Kron'd index, same shape for all corners ([N_a,N_u]).
    if l_a1==0
        a1primeKron=zeros(N_a,N_u,'gpuArray'); % no a1; offset is 0 (a2Kron itself is the index)
    else
        a1primeKron=shiftdim(Policy(l_d+1,:,jj),1);
        if l_a1>=2
            a1primeKron=a1primeKron+n_a(1)*(shiftdim(Policy(l_d+2,:,jj),1)-1);
        end
        if l_a1>=3
            error('Not yet implemented experienceassetu with length(n_a1)>2')
        end
        a1primeKron=a1primeKron-1; % zero-based offset so the final +1 comes from a2Kron
    end

    if l_a2==1
        for c=1:N_corners
            if c==1
                a2Kron=aprimeIndexes;
                pcorner=aprimeProbs;
            else
                a2Kron=aprimeIndexes+1;
                pcorner=1-aprimeProbs;
            end
            if l_a1==0
                Policy_aprime(:,:,c,jj)=a2Kron;
            else
                Policy_aprime(:,:,c,jj)=a1primeKron+1+N_a1*(a2Kron-1);
            end
            % Encode the u probabilities (pi_u) into the PolicyProbs
            PolicyProbs(:,:,c,jj)=pcorner.*shiftdim(pi_u,-1);
        end
    else % l_a2==2
        % aprimeIndexes/aprimeProbs are [N_a,l_a2,N_u] per-dim factored; fold the two per-dim
        % lower indexes into the four corners of the a2 lattice.
        n_a2_1=n_a2(1);
        loIdx_1=reshape(aprimeIndexes(:,1,:),[N_a,N_u]);
        loIdx_2=reshape(aprimeIndexes(:,2,:),[N_a,N_u]);
        prob_1=reshape(aprimeProbs(:,1,:),[N_a,N_u]);
        prob_2=reshape(aprimeProbs(:,2,:),[N_a,N_u]);
        bits=[0 0; 1 0; 0 1; 1 1];
        for c=1:N_corners
            b1=bits(c,1); b2=bits(c,2);
            a2Kron=(loIdx_1+b1)+n_a2_1*((loIdx_2+b2)-1);
            if l_a1==0
                Policy_aprime(:,:,c,jj)=a2Kron;
            else
                Policy_aprime(:,:,c,jj)=a1primeKron+1+N_a1*(a2Kron-1);
            end
            p1=prob_1; if b1==1, p1=1-p1; end
            p2=prob_2; if b2==1, p2=1-p2; end
            % Encode the u probabilities (pi_u) into the PolicyProbs
            PolicyProbs(:,:,c,jj)=p1.*p2.*shiftdim(pi_u,-1);
        end
    end
end

Policy_aprime=reshape(Policy_aprime,[N_a,N_u*N_corners,N_j]);
PolicyProbs=reshape(PolicyProbs,[N_a,N_u*N_corners,N_j]);


%%

if simoptions.gridinterplayer==0
    StationaryDist=StationaryDist_FHorz_Iteration_nProbs_noz_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_u*N_corners,N_a,N_j,Parameters);

elseif simoptions.gridinterplayer==1
    % GI doubles the corner count again: each (u,EAU corner) pair -> (lower a1, upper a1).
    Policy_aprime=repmat(Policy_aprime,1,2,1);
    PolicyProbs=repmat(PolicyProbs,1,2,1);
    % Policy_aprime(:,1:N_u*N_corners,:) lower grid point for a1 is unchanged
    Policy_aprime(:,N_u*N_corners+1:end,:)=Policy_aprime(:,N_u*N_corners+1:end,:)+1; % add one to a1, to get upper grid point

    % L2flag override (1=force all weight to lower, 2=usual, 3=force all weight to upper)
    L2index=Policy(end-1,:,:); % L2 index (end-1 because end is L2flag)
    L2flag=Policy(end,:,:);
    L2index(L2flag==1)=1;                        % force all weight to lower grid point
    L2index(L2flag==3)=simoptions.ngridinterp+2; % force all weight to upper grid point
    aprimeProbs_upper=reshape(shiftdim((L2index-1)/(simoptions.ngridinterp+1),1),[N_a,1,N_j]); % probability of upper grid point
    PolicyProbs(:,1:N_u*N_corners,:)=PolicyProbs(:,1:N_u*N_corners,:).*(1-aprimeProbs_upper); % lower a1
    PolicyProbs(:,N_u*N_corners+1:end,:)=PolicyProbs(:,N_u*N_corners+1:end,:).*aprimeProbs_upper; % upper a1

    StationaryDist=StationaryDist_FHorz_Iteration_nProbs_noz_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,2*N_u*N_corners,N_a,N_j,Parameters);
end


if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,N_j]);
else
    % If 1 then leave output in Kron form
    % StationaryDist=reshape(StationaryDist,[N_a,N_j]);
end

end
