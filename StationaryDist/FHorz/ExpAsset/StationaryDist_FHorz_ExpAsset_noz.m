function StationaryDist=StationaryDist_FHorz_ExpAsset_noz(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,N_j,d_grid,a2_grid,Parameters,simoptions)

%% Setup related to experience asset
n_d2=n_d(end);
% Split endogenous assets into the standard ones and the experience asset
l_a2=simoptions.experienceasset; % l_a2 = number of a2 (experience-asset) dims
if length(n_a)<=l_a2
    n_a1=0;
    l_a1=0;
    N_a1=0;
else
    n_a1=n_a(1:end-l_a2);
    l_a1=length(n_a1);
    N_a1=prod(n_a1);
end
n_a2=n_a(end-l_a2+1:end); % last l_a2 dims are the experience asset

% aprimeFnParamNames in same fashion
l_d2=length(n_d2);
temp=getAnonymousFnInputNames(simoptions.aprimeFn);
if length(temp)>(l_d2+l_a2+(l_a2>=2))
    aprimeFnParamNames={temp{l_d2+l_a2+(l_a2>=2)+1:end}}; % the first inputs will always be (d2,a2)
else
    aprimeFnParamNames={};
end

%%
l_d=length(n_d);

N_a=prod(n_a);

jequaloneDistKron=reshape(jequaloneDist,[N_a,1]);
jequaloneDistKron=gpuArray(jequaloneDistKron); % make sure it is on gpu

%%
% Policy is currently about d and a1prime. Convert to aprime (= a1prime kron a2prime corners).
Policy=reshape(Policy,[size(Policy,1),N_a,N_j]);
Kaprimepts=2^l_a2; % 2 points (upper and lower indexes) per dimension of a2
Policy_aprime=zeros(N_a,Kaprimepts,N_j,'gpuArray');
PolicyProbs=zeros(N_a,Kaprimepts,N_j,'gpuArray');
whichisdforexpasset=length(n_d);  % the d variable that influences the experience asset (last d)
for jj=1:N_j
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [aprimeIndexes, aprimeProbs]=CreateaprimePolicyExperienceAsset(Policy(:,:,jj),simoptions.aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, 0, d_grid, a2_grid, aprimeFnParamsVec);
    % l_a2==1: aprimeIndexes/aprimeProbs are [N_a, 1] (legacy lower-corner; upper = lower+1)
    % l_a2>1 : aprimeIndexes/aprimeProbs are [N_a, Kaprimepts] (Kron fold; index in N_a2 product space)

    % Build a1prime joint Kron index in N_a1 space ([N_a, 1])
    if l_a1==0
        a1primeIndexes=[];
    else
        a1primeIndexes=reshape(Policy(l_d+1:l_d+l_a1,:,jj),[l_a1,N_a]);
        strides_a1=cumprod([1,n_a1(1:end-1)]); % [1, l_a1]
        a1primeIndexes=(strides_a1*(a1primeIndexes-1))'+1; % [N_a, 1]
    end

    if l_a2==1
        % upper and lower grid points for a2
        if l_a1==0
            Policy_aprime(:,1,jj)=aprimeIndexes;
            Policy_aprime(:,2,jj)=aprimeIndexes+1;
        else
            Policy_aprime(:,1,jj)=a1primeIndexes+N_a1*(aprimeIndexes-1);
            Policy_aprime(:,2,jj)=Policy_aprime(:,1,jj)+N_a1;
        end
        PolicyProbs(:,1,jj)=aprimeProbs;
        PolicyProbs(:,2,jj)=1-aprimeProbs;
    else
        % aprimeIndexes/aprimeProbs shape [N_a, l_a2=2] per-dim. Kron-fold to Kaprimepts=4 corners.
        n_a2_1=n_a2(1);
        loIdx_1=aprimeIndexes(:,1);
        loIdx_2=aprimeIndexes(:,2);
        prob_1=aprimeProbs(:,1);
        prob_2=aprimeProbs(:,2);
        bits=[0 0; 1 0; 0 1; 1 1];
        for c=1:Kaprimepts
            b1=bits(c,1); b2=bits(c,2);
            a2_kron=(loIdx_1+b1)+n_a2_1*((loIdx_2+b2)-1);
            if l_a1==0
                Policy_aprime(:,c,jj)=a2_kron;
            else
                Policy_aprime(:,c,jj)=a1primeIndexes+N_a1*(a2_kron-1);
            end
            p1=prob_1; if b1==1, p1=1-p1; end
            p2=prob_2; if b2==1, p2=1-p2; end
            PolicyProbs(:,c,jj)=p1.*p2;
        end
    end
end


if simoptions.gridinterplayer==0

    StationaryDist=StationaryDist_FHorz_Iteration_nProbs_noz_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,Kaprimepts,n_a1,n_a2,N_j,Parameters,simoptions);

elseif simoptions.gridinterplayer==1
    if l_a2>1
        error('gridinterplayer=1 not yet supported with multi-dim experience asset (l_a2>1)')
    end
    % (a,u,2,j)
    Policy_aprime=repmat(Policy_aprime,1,2,1);
    PolicyProbs=repmat(PolicyProbs,1,2,1);
    % Policy_aprime(:,1:2,:) lower grid point for a1 is unchanged
    Policy_aprime(:,3:4,:)=Policy_aprime(:,3:4,:)+1; % add one to a1, to get upper grid point

    aprimeProbs_upper=reshape(shiftdim((Policy(end-1,:,:)-1)/(simoptions.ngridinterp+1),1),[N_a,1,N_j]); % probability of upper grid point (end-1 because end is now L2flag)
    PolicyProbs(:,1:2,:)=PolicyProbs(:,1:2,:).*(1-aprimeProbs_upper); % lower a1
    PolicyProbs(:,3:4,:)=PolicyProbs(:,3:4,:).*aprimeProbs_upper; % upper a1

    StationaryDist=StationaryDist_FHorz_Iteration_nProbs_noz_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,4,n_a1,n_a2,N_j,Parameters,simoptions);
end


if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,N_j]);
else
    % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDist,[N_a,N_j]);
end

end
