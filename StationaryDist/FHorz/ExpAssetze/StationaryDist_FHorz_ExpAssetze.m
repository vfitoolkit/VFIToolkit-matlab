function StationaryDist=StationaryDist_FHorz_ExpAssetze(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,z_gridvals_J,pi_z_J,Parameters,simoptions)

%% Setup related to experience asset
n_d2=n_d(end-simoptions.l_dexperienceassetze+1:end);
% Split endogenous assets into the standard ones and the experience asset(s).
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
l_d2=length(n_d2);
l_a2=length(n_a2);
l_z=length(n_z);
l_e=length(simoptions.n_e);
temp=getAnonymousFnInputNames(simoptions.aprimeFn);
if length(temp)>(l_d2+l_a2+l_z+l_e+(l_a2>=2))
    aprimeFnParamNames={temp{l_d2+l_a2+l_z+l_e+(l_a2>=2)+1:end}}; % the first inputs will always be (d2,a2,z,e), plus a 'whicha' slot when l_a2>=2
else
    aprimeFnParamNames={};
end

N_e=prod(simoptions.n_e);
N_z=prod(n_z);

%%
l_d=length(n_d);
l_a=length(n_a);

N_a=prod(n_a);

%%
n_ze=[n_z,simoptions.n_e];
N_ze=N_z*N_e;

jequaloneDist=reshape(jequaloneDist,[N_a*N_ze,1]);
Policy=reshape(Policy,[size(Policy,1),N_a,N_ze,N_j]);

%% expassetze transitions
% Policy is currently about d and a1prime. Convert it to being about aprime
% (Kron'd linear index in N_a=N_a1*N_a2 space), per corner, with probs.
% For l_a2==1: 2 corners (lower/upper). For l_a2==2: 4 corners (bilinear lattice).
Kaprimepts=2^l_a2;
Policy_aprime=zeros(N_a,N_ze,Kaprimepts,N_j,'gpuArray'); % Kron'd a-index per corner
PolicyProbs  =zeros(N_a,N_ze,Kaprimepts,N_j,'gpuArray'); % corner probabilities
whichisdforexpassetze=length(n_d)-simoptions.l_dexperienceassetze+1:length(n_d);

l_a1=length(n_a)-l_a2;
N_a1=prod(n_a1);
if N_a1==0
    N_a1=1; % so the Kron offset N_a1*(a2Kron-1) is well-defined; a2Kron alone is the index
end
N_a2=prod(n_a2);

for jj=1:N_j
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [aprimeIndexes, aprimeProbs]=CreateaprimePolicyExperienceAssetze(Policy(:,:,:,jj),simoptions.aprimeFn, whichisdforexpassetze, n_d, n_a1,n_a2, n_z, simoptions.n_e, 0,N_z,N_e, d_grid, a2_grid, z_gridvals_J(:,:,jj), simoptions.e_gridvals_J(:,:,jj), aprimeFnParamsVec);
    % l_a2==1: aprimeIndexes/aprimeProbs are [N_a, N_ze] (lower-corner; upper = lower+1)
    % l_a2==2: aprimeIndexes/aprimeProbs are [N_a, l_a2, N_ze] (per-dim factored)

    % Build a1-Kron'd index, same shape for all corners ([N_a, N_ze]).
    if l_a1==0
        a1primeKron=zeros(N_a,N_ze,'gpuArray'); % no a1; offset is 0 (a2Kron itself is the index)
    else
        a1primeKron=shiftdim(Policy(l_d+1,:,:,jj),1);
        if l_a1>=2
            a1primeKron=a1primeKron+n_a(1)*(shiftdim(Policy(l_d+2,:,:,jj),1)-1);
        end
        if l_a1>=3
            a1primeKron=a1primeKron+prod(n_a(1:2))*(shiftdim(Policy(l_d+3,:,:,jj),1)-1);
        end
        if l_a1>=4
            a1primeKron=a1primeKron+prod(n_a(1:3))*(shiftdim(Policy(l_d+4,:,:,jj),1)-1);
        end
        if l_a1>=5
            error('Not yet implemented experience asset with length(n_a1)>4')
        end
        a1primeKron=a1primeKron-1; % zero-based offset so the final +1 comes from a2Kron
    end

    if l_a2==1
        for c=1:Kaprimepts
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
        n_a2_1=n_a2(1);
        loIdx_1=reshape(aprimeIndexes(:,1,:),[N_a,N_ze]);
        loIdx_2=reshape(aprimeIndexes(:,2,:),[N_a,N_ze]);
        prob_1=reshape(aprimeProbs(:,1,:),[N_a,N_ze]);
        prob_2=reshape(aprimeProbs(:,2,:),[N_a,N_ze]);
        bits=[0 0; 1 0; 0 1; 1 1];
        for c=1:Kaprimepts
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


jequaloneDist=gather(jequaloneDist); % Tan improvement is done on cpu
clear aprimeIndexes aprimeProbs

%%
if simoptions.gridinterplayer==0
    % Both z and e required for experienceassetze
    StationaryDist=StationaryDist_FHorz_Iteration_nProbs_e_raw(jequaloneDist,AgeWeightParamNames,gather(Policy_aprime),gather(PolicyProbs),Kaprimepts,N_a,N_z,N_e,N_j,pi_z_J,simoptions.pi_e_J,Parameters);
elseif simoptions.gridinterplayer==1
    if l_a2>1
        error('gridinterplayer=1 not yet supported with multi-dim experienceassetze (l_a2>1)')
    end
    % (a,z,2,j)
    Policy_aprime=repmat(Policy_aprime,1,1,2,1);
    % Policy_aprime(:,:,1:2,:) lower grid point for a1 is unchanged
    Policy_aprime(:,:,3:4,:)=Policy_aprime(:,:,3:4,:)+1; % add one to a1, to get upper grid point
    Policy_aprime=gather(Policy_aprime);

    PolicyProbs=repmat(PolicyProbs,1,1,2,1);
    aprimeProbs_upper=reshape(shiftdim((Policy(end-1,:,:,:)-1)/(simoptions.ngridinterp+1),1),[N_a,N_ze,1,N_j]); % probability of upper grid point (from L2 index; end-1 because end is now L2flag)
    PolicyProbs(:,:,1:2,:)=PolicyProbs(:,:,1:2,:).*(1-aprimeProbs_upper); % lower a1
    PolicyProbs(:,:,3:4,:)=PolicyProbs(:,:,3:4,:).*aprimeProbs_upper; % upper a1

    StationaryDist=StationaryDist_FHorz_Iteration_nProbs_e_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,gather(PolicyProbs),4,N_a,N_z,N_e,N_j,pi_z_J,simoptions.pi_e_J,Parameters);
end



if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,n_ze,N_j]);
% else
    % If 1 then leave output in Kron form
    % StationaryDist=reshape(StationaryDist,[N_a,N_ze,N_j]);
end

end
