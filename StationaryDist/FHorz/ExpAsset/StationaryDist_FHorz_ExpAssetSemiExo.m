function StationaryDist=StationaryDist_FHorz_ExpAssetSemiExo(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_semiz,n_z,N_j,pi_semiz_J,pi_z_J,Parameters,simoptions)

%% Experience asset and semi-exogenous state
n_d3=n_d(end-simoptions.l_dsemiz+1:end); % decision variable that controls semi-exogenous state
n_d2=n_d(end-simoptions.l_dexperienceasset-simoptions.l_dsemiz+1:end-simoptions.l_dsemiz); % decision variables that controls experience asset
if length(n_d)>2
    n_d1=n_d(1:end-2);
    l_d1=length(n_d1);
else
    % n_d1=0;
    l_d1=0;
end
l_d2=length(n_d2); % wouldn't be here if no d2
l_d3=length(n_d3); % wouldn't be here if no d3

l_d12=l_d1+l_d2;

N_dsemiz=prod(n_d3);

%% Setup related to experience asset
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
% l_d2=length(n_d2);
l_a2=length(n_a2);
temp=getAnonymousFnInputNames(simoptions.aprimeFn);
if length(temp)>(l_d2+l_a2+(l_a2>=2))  % the (l_a2>=2) term is the 'whicha' selector slot, which aprimeFn only takes when there are two experience assets
    aprimeFnParamNames={temp{l_d2+l_a2+(l_a2>=2)+1:end}}; % the first inputs are (d2,a2), plus the 'whicha' selector when l_a2>=2
else
    aprimeFnParamNames={};
end


%%
l_d=length(n_d);
l_a=length(n_a);
if isscalar(n_a1) && n_a1==0
    l_a1=0; N_a1=0;
else
    l_a1=length(n_a1); N_a1=prod(n_a1);
end

N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);

N_e=prod(simoptions.n_e);

%%
if N_z==0
    if N_e==0
        n_bothze=simoptions.n_semiz;
        N_bothze=N_semiz;
    else
        n_bothze=[simoptions.n_semiz,simoptions.n_e];
        N_bothze=N_semiz*N_e;
    end
else
    if N_e==0
        n_bothze=[simoptions.n_semiz,n_z];
        N_bothze=N_semiz*N_z;
    else
        n_bothze=[simoptions.n_semiz,n_z,simoptions.n_e];
        N_bothze=N_semiz*N_z*N_e;
    end
end

jequaloneDist=gpuArray(jequaloneDist); % make sure it is on gpu
jequaloneDist=reshape(jequaloneDist,[N_a*N_bothze,1]);
Policy=reshape(Policy,[size(Policy,1),N_a,N_bothze,N_j]);


%% expasset transitions
% Policy is currently about d and a1prime. Convert it to being about aprime
% as that is what we need for simulation, and we can then just send it to standard Case1 commands.
N_probs=2^l_a2; % 2 points (lower and upper index) per dimension of a2
Policy_aprime=zeros(N_a,N_bothze,N_probs,N_j,'gpuArray');
PolicyProbs=zeros(N_a,N_bothze,N_probs,N_j,'gpuArray'); % third dimension indexes the interpolation corners
whichisdforexpasset=length(n_d)-simoptions.l_dexperienceasset-simoptions.l_dsemiz+1:length(n_d)-simoptions.l_dsemiz;  % is just saying which is the decision variable that influences the experience asset (it is the 'second last' decision variable)
for jj=1:N_j
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [aprimeIndexes, aprimeProbs]=CreateaprimePolicyExperienceAsset(Policy(:,:,:,jj),simoptions.aprimeFn, whichisdforexpasset, n_d, n_a1,n_a2, N_bothze, d_grid, a2_grid, aprimeFnParamsVec);
    % Note: aprimeIndexes and aprimeProbs are both [N_a,N_bothze]
    % Note: aprimeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the aprimeProbs are the probability of this lower point (prob of upper point is just 1 minus this).

    if l_a2==1
        if l_a1==0
            Policy_aprime(:,:,1,jj)=aprimeIndexes;
            Policy_aprime(:,:,2,jj)=aprimeIndexes+1;
        elseif l_a1==1 % experience asset and one other asset
            Policy_aprime(:,:,1,jj)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(aprimeIndexes-1);
            Policy_aprime(:,:,2,jj)=Policy_aprime(:,:,1,jj)+n_a(1);
        elseif l_a1==2 % experience asset and two other assets
            Policy_aprime(:,:,1,jj)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(shiftdim(Policy(l_d+2,:,:,jj),1)-1)+prod(n_a(1:2))*(aprimeIndexes-1);
            Policy_aprime(:,:,2,jj)=Policy_aprime(:,:,1,jj)+prod(n_a(1:2));
        else
            error('Not yet implemented experience asset with length(n_a)>3')
        end
        PolicyProbs(:,:,1,jj)=aprimeProbs;
        PolicyProbs(:,:,2,jj)=1-aprimeProbs;
    else
        % l_a2==2: aprimeIndexes/aprimeProbs are [N_a,l_a2,N_bothze] per-dim factored.
        % Kron-fold to N_probs=4 corners (mirrors StationaryDist_FHorz_ExpAsset).
        n_a2_1=n_a2(1);
        loIdx_1=reshape(aprimeIndexes(:,1,:),[N_a,N_bothze]);
        loIdx_2=reshape(aprimeIndexes(:,2,:),[N_a,N_bothze]);
        prob_1=reshape(aprimeProbs(:,1,:),[N_a,N_bothze]);
        prob_2=reshape(aprimeProbs(:,2,:),[N_a,N_bothze]);
        if l_a1==0
            a1primeIndexes=[];
        elseif l_a1==1
            a1primeIndexes=shiftdim(Policy(l_d+1,:,:,jj),1);
        elseif l_a1==2
            a1primeIndexes=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(shiftdim(Policy(l_d+2,:,:,jj),1)-1);
        else
            error('Not yet implemented experience asset with length(n_a)>3')
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


%% Policy_dsemiexo

% d3 is the variable relevant for the semi-exogenous asset.
if l_d3==1
    Policy_dsemiexo=Policy(l_d12+1,:,:,:);
elseif l_d3==2
    Policy_dsemiexo=Policy(l_d12+1,:,:,:)+n_d(l_d12+1)*(Policy(l_d12+2,:,:,:)-1);
elseif l_d3==3
    Policy_dsemiexo=Policy(l_d12+1,:,:,:)+n_d(l_d12+1)*(Policy(l_d12+2,:,:,:)-1)+n_d(l_d12+1)*n_d(l_d12+2)*(Policy(l_d12+3,:,:,:)-1);
elseif l_d3==4
    Policy_dsemiexo=Policy(l_d12+1,:,:,:)+n_d(l_d12+1)*(Policy(l_d12+2,:,:,:)-1)+n_d(l_d12+1)*n_d(l_d12+2)*(Policy(l_d12+3,:,:,:)-1)+n_d(l_d12+1)*n_d(l_d12+2)*n_d(l_d12+3)*(Policy(l_d12+4,:,:,:)-1);
end
Policy_dsemiexo=shiftdim(Policy_dsemiexo,1);


%% Do the Tan improvement on the CPU (is possible on GPU, but gives out of memory errors to easily)

%%
if simoptions.gridinterplayer==0
    if N_z==0 && N_e==0
        StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_nProbs_noz_raw(jequaloneDist,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,N_probs,N_dsemiz,N_a,N_semiz,N_j,pi_semiz_J,Parameters);
    elseif N_e==0 % just z
        StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_nProbs_raw(jequaloneDist,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,N_probs,N_dsemiz,N_a,N_semiz,N_z,N_j,pi_semiz_J,pi_z_J,Parameters);
    elseif N_z==0 % just e
        StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_nProbs_noz_e_raw(jequaloneDist,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,N_probs,N_dsemiz,N_a,N_semiz,N_e,N_j,pi_semiz_J,simoptions.pi_e_J,Parameters);
    else % both z and e
        StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_nProbs_e_raw(jequaloneDist,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,N_probs,N_dsemiz,N_a,N_semiz,N_z,N_e,N_j,pi_semiz_J,pi_z_J,simoptions.pi_e_J,Parameters);
    end
elseif simoptions.gridinterplayer==1
    Policy_aprime=repmat(Policy_aprime,1,1,2,1);
    PolicyProbs=repmat(PolicyProbs,1,1,2,1);
    % Policy_aprime(:,:,1:N_probs,:) lower grid point for a1 is unchanged
    Policy_aprime(:,:,N_probs+1:2*N_probs,:)=Policy_aprime(:,:,N_probs+1:2*N_probs,:)+1; % add one to a1, to get upper grid point (a1 is the fastest-varying dim of the aprime index)

    % L2flag override (1=force all weight to lower, 2=usual, 3=force all weight to upper)
    L2index=Policy(end-1,:,:,:); % L2 index (end-1 because end is L2flag)
    L2flag=Policy(end,:,:,:);
    L2index(L2flag==1)=1;                        % force all weight to lower grid point
    L2index(L2flag==3)=simoptions.ngridinterp+2; % force all weight to upper grid point
    aprimeProbs_upper=reshape(shiftdim((L2index-1)/(simoptions.ngridinterp+1),1),[N_a,N_bothze,1,N_j]); % probability of upper grid point
    PolicyProbs(:,:,1:N_probs,:)=PolicyProbs(:,:,1:N_probs,:).*(1-aprimeProbs_upper); % lower a1
    PolicyProbs(:,:,N_probs+1:2*N_probs,:)=PolicyProbs(:,:,N_probs+1:2*N_probs,:).*aprimeProbs_upper; % upper a1
    N_probs=2*N_probs;

    if N_z==0 && N_e==0
        StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_nProbs_noz_raw(jequaloneDist,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,N_probs,N_dsemiz,N_a,N_semiz,N_j,pi_semiz_J,Parameters);
    elseif N_e==0 % just z
        StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_nProbs_raw(jequaloneDist,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,N_probs,N_dsemiz,N_a,N_semiz,N_z,N_j,pi_semiz_J,pi_z_J,Parameters);
    elseif N_z==0 % just e
        StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_nProbs_noz_e_raw(jequaloneDist,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,N_probs,N_dsemiz,N_a,N_semiz,N_e,N_j,pi_semiz_J,simoptions.pi_e_J,Parameters);
    else % both z and e
        StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_nProbs_e_raw(jequaloneDist,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,N_probs,N_dsemiz,N_a,N_semiz,N_z,N_e,N_j,pi_semiz_J,pi_z_J,simoptions.pi_e_J,Parameters);
    end
end



if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,n_bothze,N_j]);
else
    % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDist,[N_a,N_bothze,N_j]);
end

end
