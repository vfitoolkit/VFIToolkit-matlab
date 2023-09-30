function StationaryDist=StationaryDist_FHorz_Case1_ExpAsset_noz(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,N_j,Parameters,simoptions)

n_d2=n_d(end);
n_a2=n_a(end);

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
Policy_aprime=zeros(N_a,N_j,2,'gpuArray'); % The fourth dimension is lower/upper grid point
PolicyProbs=zeros(N_a,N_j,2,'gpuArray'); % The fourth dimension is lower/upper grid point
whichisdforexpasset=length(n_d);  % is just saying which is the decision variable that influences the experience asset (it is the 'last' decision variable)
for jj=1:N_j
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [aprimeIndexes, aprimeProbs]=CreateaprimePolicyExperienceAsset_Case1(Policy(:,:,jj),simoptions.aprimeFn, whichisdforexpasset, n_a, 1, gpuArray(simoptions.a_grid), aprimeFnParamsVec); % Note: 1 is pretending N_z=1, which does what we want for no z variable
    if l_a==1
        Policy_aprime(:,jj,1)=aprimeIndexes;
        Policy_aprime(:,jj,2)=aprimeIndexes+1;
        PolicyProbs(:,jj,1)=aprimeProbs;
        PolicyProbs(:,jj,2)=1-aprimeProbs;
    elseif l_a==2 % experience asset and one other asset
        Policy_aprime(:,jj,1)=shiftdim(Policy(l_d+1,:,jj),1)+n_a(1)*(aprimeIndexes-1);
        Policy_aprime(:,jj,2)=shiftdim(Policy(l_d+1,:,jj),1)+n_a(1)*(aprimeIndexes-1+1);
        PolicyProbs(:,jj,1)=aprimeProbs;
        PolicyProbs(:,jj,2)=1-aprimeProbs;
    elseif l_a==3 % experience asset and two other assets
        Policy_aprime(:,jj,1)=shiftdim(Policy(l_d+1,:,jj),1)+n_a(1)*(shiftdim(Policy(l_d+2,:,jj),1)-1)+prod(n_a(1:2))*(aprimeIndexes-1);
        Policy_aprime(:,jj,2)=shiftdim(Policy(l_d+1,:,jj),1)+n_a(1)*(shiftdim(Policy(l_d+2,:,jj),1)-1)+prod(n_a(1:2))*(aprimeIndexes-1+1);
        PolicyProbs(:,jj,1)=aprimeProbs;
        PolicyProbs(:,jj,2)=1-aprimeProbs;       
    else
        error('Not yet implemented experience asset with length(n_a)>3')
    end
end

if simoptions.iterate==0
    error('This combo only supports iteration')
    % PolicyProbs=gather(PolicyProbs); % simulation is always with cpu
    % Policy_aprime=gather(Policy_aprime);
    % if simoptions.parallel>=3
    %     % Sparse matrix is not relevant for the simulation methods, only for iteration method
    %     simoptions.parallel=2; % will simulate on parallel cpu, then transfer solution to gpu
    % end
    % StationaryDistKron=StationaryDist_FHorz_Case1_Simulation_TwoProbs_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_a,N_z,N_j,pi_z, Parameters, simoptions);
elseif simoptions.iterate==1
    % Temporary hack:
    N_z=1;
    pi_z=1;
    Policy_aprime=reshape(Policy_aprime,[N_a,1,N_j,2]);
    PolicyProbs=reshape(PolicyProbs,[N_a,1,N_j,2]);
    StationaryDistKron=StationaryDist_FHorz_Case1_Iteration_TwoProbs_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_a,N_z,N_j,pi_z,Parameters,simoptions); % zero is n_d, because we already converted Policy to only contain aprime
end

if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDistKron,[n_a,N_j]);
else
    % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDistKron,[N_a,N_j]);
end

end
