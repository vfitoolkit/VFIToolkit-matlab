function StationaryDist=StationaryDist_FHorz_ExpAsset_noz(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,N_j,d_grid,a2_grid,Parameters,simoptions)

n_d2=n_d(end);
% Split endogenous assets into the standard ones and the experience asset
if length(n_a)==1
    n_a1=0;
else
    n_a1=n_a(1:end-1);
end
n_a2=n_a(end); % n_a2 is the experience asset

% aprimeFnParamNames in same fashion
l_d2=length(n_d2);
l_a2=length(n_a2);
temp=getAnonymousFnInputNames(simoptions.aprimeFn);
if length(temp)>(l_d2+l_a2)
    aprimeFnParamNames={temp{l_d2+l_a2+1:end}}; % the first inputs will always be (d2,a2)
else
    aprimeFnParamNames={};
end

l_d=length(n_d);
l_a=length(n_a);

N_a=prod(n_a);

%%
jequaloneDistKron=reshape(jequaloneDist,[N_a,1]);
if simoptions.parallel~=2 && simoptions.parallel~=4
    Policy=gather(Policy);
    jequaloneDistKron=gather(jequaloneDistKron);    
end

% Policy is currently about d and a2prime. Convert it to being about aprime
% as that is what we need for simulation, and we can then just send it to standard Case1 commands.
Policy=reshape(Policy,[size(Policy,1),N_a,N_j]);
Policy_aprime=zeros(N_a,2,N_j,'gpuArray'); % The fourth dimension is lower/upper grid point
PolicyProbs=zeros(N_a,2,N_j,'gpuArray'); % The fourth dimension is lower/upper grid point
whichisdforexpasset=length(n_d);  % is just saying which is the decision variable that influences the experience asset (it is the 'last' decision variable)
for jj=1:N_j
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [aprimeIndexes, aprimeProbs]=CreateaprimePolicyExperienceAsset_Case1(Policy(:,:,jj),simoptions.aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, 0, d_grid, a2_grid, aprimeFnParamsVec); % Note: 1 is pretending N_z=1, which does what we want for no z variable
    if l_a==1
        Policy_aprime(:,1,jj)=aprimeIndexes;
        Policy_aprime(:,2,jj)=aprimeIndexes+1;
        PolicyProbs(:,1,jj)=aprimeProbs;
        PolicyProbs(:,2,jj)=1-aprimeProbs;
    elseif l_a==2 % experience asset and one other asset
        Policy_aprime(:,1,jj)=shiftdim(Policy(l_d+1,:,jj),1)+n_a(1)*(aprimeIndexes-1);
        Policy_aprime(:,2,jj)=shiftdim(Policy(l_d+1,:,jj),1)+n_a(1)*(aprimeIndexes-1+1);
        PolicyProbs(:,1,jj)=aprimeProbs;
        PolicyProbs(:,2,jj)=1-aprimeProbs;
    elseif l_a==3 % experience asset and two other assets
        Policy_aprime(:,1,jj)=shiftdim(Policy(l_d+1,:,jj),1)+n_a(1)*(shiftdim(Policy(l_d+2,:,jj),1)-1)+prod(n_a(1:2))*(aprimeIndexes-1);
        Policy_aprime(:,2,jj)=shiftdim(Policy(l_d+1,:,jj),1)+n_a(1)*(shiftdim(Policy(l_d+2,:,jj),1)-1)+prod(n_a(1:2))*(aprimeIndexes-1+1);
        PolicyProbs(:,1,jj)=aprimeProbs;
        PolicyProbs(:,2,jj)=1-aprimeProbs;       
    else
        error('Not yet implemented experience asset with length(n_a)>3')
    end
end

if simoptions.iterate==0
    dbstack
    error('This combo only supports iteration')
elseif simoptions.iterate==1
    StationaryDist=StationaryDist_FHorz_Iteration_TwoProbs_noz_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_a,N_j,Parameters); % zero is n_d, because we already converted Policy to only contain aprime
end

if simoptions.parallel==2
    StationaryDist=gpuArray(StationaryDist);
end
if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,N_j]);
else
    % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDist,[N_a,N_j]);
end

end
