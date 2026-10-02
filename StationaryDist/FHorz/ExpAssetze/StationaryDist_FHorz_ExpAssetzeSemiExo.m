function StationaryDist=StationaryDist_FHorz_ExpAssetzeSemiExo(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_semiz,n_z,N_j,pi_semiz_J,z_gridvals_J,pi_z_J,Parameters,simoptions)

%% Experience asset and semi-exogenous state
n_d3=n_d(end-simoptions.l_dsemiz+1:end); % decision variable that controls semi-exogenous state
n_d2=n_d(end-simoptions.l_dexperienceassetze-simoptions.l_dsemiz+1:end-simoptions.l_dsemiz); % decision variables that controls experience asset
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
% simoptions.experienceassetze is the *count* of EAZE dims (1 or 2).
l_a2=simoptions.experienceassetze;
if length(n_a)<=l_a2
    n_a1=0;
else
    n_a1=n_a(1:end-l_a2);
end
n_a2=n_a(end-l_a2+1:end); % last l_a2 dims are the experience asset(s)

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
l_z=length(n_z);
l_e=length(simoptions.n_e);
temp=getAnonymousFnInputNames(simoptions.aprimeFn);
if length(temp)>(l_d2+l_a2+l_z+l_e+(l_a2>=2))
    aprimeFnParamNames={temp{l_d2+l_a2+l_z+l_e+(l_a2>=2)+1:end}}; % the first inputs will always be (d2,a2,z,e), plus a 'whicha' slot when l_a2>=2
else
    aprimeFnParamNames={};
end


%%
l_d=length(n_d);
l_a=length(n_a);

N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);

N_e=prod(simoptions.n_e);

%%
% Both z and e are required for experienceassetze
n_bothze=[simoptions.n_semiz,n_z,simoptions.n_e];
N_bothze=N_semiz*N_z*N_e;

jequaloneDist=gpuArray(jequaloneDist); % make sure it is on gpu
jequaloneDist=reshape(jequaloneDist,[N_a*N_bothze,1]);
Policy=reshape(Policy,[size(Policy,1),N_a,N_bothze,N_j]);


%% expassetze transitions
% Policy is currently about d and a1prime. Convert it to being about aprime
% as that is what we need for simulation, and we can then just send it to standard Case1 commands.
% For l_a2==1: 2 corners (lower/upper). For l_a2==2: 4 corners (bilinear lattice).
N_probs=2^l_a2;
Policy_aprime=zeros(N_a,N_bothze,N_probs,N_j,'gpuArray'); % Kron'd a-index per corner
PolicyProbs=zeros(N_a,N_bothze,N_probs,N_j,'gpuArray'); % corner probabilities
whichisdforexpassetze=length(n_d)-simoptions.l_dexperienceassetze-simoptions.l_dsemiz+1:length(n_d)-simoptions.l_dsemiz;  % is just saying which is the decision variable that influences the experience asset (it is the 'second last' decision variable)
l_a1=length(n_a)-l_a2;
N_a1=prod(n_a1);
if N_a1==0
    N_a1=1; % so the Kron offset N_a1*(a2Kron-1) is well-defined; a2Kron alone is the index
end

for jj=1:N_j
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [aprimeIndexes, aprimeProbs]=CreateaprimePolicyExperienceAssetze(Policy(:,:,:,jj),simoptions.aprimeFn, whichisdforexpassetze, n_d, n_a1,n_a2, n_z, simoptions.n_e, N_semiz,N_z,N_e, d_grid, a2_grid, z_gridvals_J(:,:,jj), simoptions.e_gridvals_J(:,:,jj), aprimeFnParamsVec);
    % Note: aprimeIndexes and aprimeProbs are both [N_a,N_bothze] with semiz fastest, then z, then e -- matches n_bothze=[n_semiz,n_z,n_e] ordering.
    % Note: aprimeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the aprimeProbs are the probability of this lower point (prob of upper point is just 1 minus this).

    % Build the a1-Kron'd index, same shape for all corners ([N_a,N_bothze]).
    if l_a1==0
        a1primeKron=zeros(N_a,N_bothze,'gpuArray'); % no a1; offset is 0 (a2Kron itself is the index)
    else
        a1primeKron=shiftdim(Policy(l_d+1,:,:,jj),1);
        if l_a1>=2
            a1primeKron=a1primeKron+n_a(1)*(shiftdim(Policy(l_d+2,:,:,jj),1)-1);
        end
        if l_a1>=3
            error('Not yet implemented experienceassetze with length(n_a1)>2')
        end
        a1primeKron=a1primeKron-1; % zero-based offset so the final +1 comes from a2Kron
    end

    if l_a2==1
        for c=1:N_probs
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
            PolicyProbs(:,:,c,jj)=pcorner;
        end
    else % l_a2==2
        % aprimeIndexes/aprimeProbs are [N_a,l_a2,N_bothze] per-dim factored; fold the two
        % per-dim lower indexes into the four corners of the a2 lattice.
        n_a2_1=n_a2(1);
        loIdx_1=reshape(aprimeIndexes(:,1,:),[N_a,N_bothze]);
        loIdx_2=reshape(aprimeIndexes(:,2,:),[N_a,N_bothze]);
        prob_1=reshape(aprimeProbs(:,1,:),[N_a,N_bothze]);
        prob_2=reshape(aprimeProbs(:,2,:),[N_a,N_bothze]);
        bits=[0 0; 1 0; 0 1; 1 1];
        for c=1:N_probs
            b1=bits(c,1); b2=bits(c,2);
            a2Kron=(loIdx_1+b1)+n_a2_1*((loIdx_2+b2)-1);
            if l_a1==0
                Policy_aprime(:,:,c,jj)=a2Kron;
            else
                Policy_aprime(:,:,c,jj)=a1primeKron+1+N_a1*(a2Kron-1);
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
    % Both z and e required for experienceassetze
    StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_nProbs_e_raw(jequaloneDist,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,N_probs,N_dsemiz,N_a,N_semiz,N_z,N_e,N_j,pi_semiz_J,pi_z_J,simoptions.pi_e_J,Parameters);
elseif simoptions.gridinterplayer==1
    % GI doubles the corner count: each EAZE corner -> (lower a1, upper a1) pair.
    % l_a2==1: N_probs=2 -> Kaprimepts_GI=4 (legacy)
    % l_a2==2: N_probs=4 -> Kaprimepts_GI=8
    Kaprimepts_GI=2*N_probs;
    Policy_aprime=repmat(Policy_aprime,1,1,2,1); % (N_a,N_bothze,Kaprimepts_GI,N_j)
    PolicyProbs=repmat(PolicyProbs,1,1,2,1);
    % Corners 1..N_probs are EAZE corners at lower a1; N_probs+1..Kaprimepts_GI at upper a1.
    Policy_aprime(:,:,N_probs+1:Kaprimepts_GI,:)=Policy_aprime(:,:,N_probs+1:Kaprimepts_GI,:)+1; % add one to a1, to get upper grid point

    % L2flag override (1=force all weight to lower, 2=usual, 3=force all weight to upper)
    L2index=Policy(end-1,:,:,:); % L2 index (end-1 because end is L2flag)
    L2flag=Policy(end,:,:,:);
    L2index(L2flag==1)=1;                        % force all weight to lower grid point
    L2index(L2flag==3)=simoptions.ngridinterp+2; % force all weight to upper grid point
    aprimeProbs_upper=reshape(shiftdim((L2index-1)/(simoptions.ngridinterp+1),1),[N_a,N_bothze,1,N_j]); % probability of upper grid point
    PolicyProbs(:,:,1:N_probs,:)=PolicyProbs(:,:,1:N_probs,:).*(1-aprimeProbs_upper); % lower a1
    PolicyProbs(:,:,N_probs+1:Kaprimepts_GI,:)=PolicyProbs(:,:,N_probs+1:Kaprimepts_GI,:).*aprimeProbs_upper; % upper a1

    StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_nProbs_e_raw(jequaloneDist,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,Kaprimepts_GI,N_dsemiz,N_a,N_semiz,N_z,N_e,N_j,pi_semiz_J,pi_z_J,simoptions.pi_e_J,Parameters);
end



if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,n_bothze,N_j]);
else
    % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDist,[N_a,N_bothze,N_j]);
end

end
