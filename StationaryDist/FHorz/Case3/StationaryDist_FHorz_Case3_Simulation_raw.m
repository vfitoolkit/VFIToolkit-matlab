function StationaryDistKron=StationaryDist_FHorz_Case3_Simulation_raw(jequaloneDistKron,AgeWeightParamNames,PolicyIndexesKron,n_d,n_a,n_z,n_u,N_j,d_grid, a_grid, u_grid,pi_z,pi_u,aprimeFn,Parameters,aprimeParamNames, simoptions)
% Case3: aprime(d,u)
%
% Options needed
%    simoptions.nsims
%    simoptions.parallel
%    simoptions.verbose
%    simoptions.ncores


N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_u=prod(n_u);

MoveSSDKtoGPU=0;
if simoptions.parallel==2
    % Simulation on GPU is really slow.
    % So instead, switch to CPU.
    % For anything but ridiculously short simulations it is more than worth the overhead to switch to CPU and back.
    PolicyIndexesKron=gather(PolicyIndexesKron);
    pi_z=gather(pi_z);
    % Use parallel cpu for these simulations
    simoptions.parallel=1;
    
    MoveSSDKtoGPU=1;
end

% This implementation is slightly inefficient when shocks are not age dependent, but speed loss is fairly trivial
eval('fieldexists_ExogShockFn=1;simoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;simoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')
eval('fieldexists_pi_z_J=1;simoptions.pi_z_J;','fieldexists_pi_z_J=0;')

if fieldexists_pi_z_J==1
    pi_z_J=simoptions.pi_z_J;
elseif fieldexists_ExogShockFn==1
    pi_z_J=zeros(N_z,N_z,N_j);
    for jj=1:N_j
        if fieldexists_ExogShockFnParamNames==1
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
            [~,pi_z]=simoptions.ExogShockFn(ExogShockFnParamsVec);
        else
            [~,pi_z]=simoptions.ExogShockFn(jj);
        end
        pi_z_J(:,:,jj)=pi_z;
    end
else
    pi_z_J=repmat(pi_z,1,1,N_j);
end

pi_u_J=zeros(N_u,N_j); % Not yet meaningfully implemented
for jj=1:N_j
    pi_u_J(:,jj)=pi_u;
end

if simoptions.aprimedependsonage==0
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,1);
    [aprimeIndex,aprimeProbs]=CreateaprimeFnMatrix_Case3(aprimeFn, n_d, n_a, n_u, d_grid, a_grid, u_grid, aprimeFnParamsVec,0); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)

    StationaryDistKron=zeros(N_a,N_z,N_j,simoptions.ncores);
    cumsum_pi_z_J=cumsum(pi_z_J,2);
    cumsum_pi_u_J=cumsum(pi_u_J,1);
    jequaloneDistKroncumsum=cumsum(jequaloneDistKron);
    %Create simoptions.ncores different steady state distn's, then combine them.
    if simoptions.parallel==1
        parfor ncore_c=1:simoptions.ncores
            SteadyStateDistKron_ncore_c=zeros(N_a,N_z,N_j);
            % Pull a random start point from jequaloneDistKron
            currstate=find(jequaloneDistKroncumsum>rand(1,1),'first'); %Pick a random start point on the (vectorized) (a,z) grid for j=1
            currstate=ind2sub_homemade([N_a,N_z],currstate); % (a,z)
            SteadyStateDistKron_ncore_c(currstate(1),currstate(2),1)=SteadyStateDistKron_ncore_c(currstate(1),currstate(2),1)+1;
            for jj=1:(N_j-1)
                dindex=PolicyIndexesKron(currstate(1),currstate(2),jj);
                currstate(2)=find(cumsum_pi_z_J(currstate(2),:,jj)>rand(1,1),'first'); % zprime
                u_ind=find(cumsum_pi_u_J(:,jj)>rand(1,1),'first'); % u
                u_lowerupper=max(aprimeProbs(u_ind)>rand(1,1)); % value not index (hence max not find)
                currstate(1)=aprimeIndex(dindex,u_ind+u_lowerupper); % Case3: aprime(d,u)
                SteadyStateDistKron_ncore_c(currstate(1),currstate(2),jj+1)=SteadyStateDistKron_ncore_c(currstate(1),currstate(2),jj+1)+1;
            end
            StationaryDistKron(:,:,:,ncore_c)=SteadyStateDistKron_ncore_c;
        end
    elseif simoptions.parallel==0 % Only different is for instead of parfor
        for ncore_c=1:simoptions.ncores
            SteadyStateDistKron_ncore_c=zeros(N_a,N_z,N_j);
            % Pull a random start point from jequaloneDistKron
            currstate=find(jequaloneDistKroncumsum>rand(1,1),'first'); %Pick a random start point on the (vectorized) (a,z) grid for j=1
            currstate=ind2sub_homemade([N_a,N_z],currstate); % (a,z)
            SteadyStateDistKron_ncore_c(currstate(1),currstate(2),1)=SteadyStateDistKron_ncore_c(currstate(1),currstate(2),1)+1;
            for jj=1:(N_j-1)
                dindex=PolicyIndexesKron(currstate(1),currstate(2),jj);
                currstate(2)=find(cumsum_pi_z_J(currstate(2),:,jj)>rand(1,1),'first'); % zprime
                u_ind=find(cumsum_pi_u_J(:,jj)>rand(1,1),'first'); % u
                u_lowerupper=max(aprimeProbs(u_ind)>rand(1,1)); % value not index (hence max not find)
                currstate(1)=aprimeIndex(dindex,u_ind+u_lowerupper); % Case3: aprime(d,u)
                SteadyStateDistKron_ncore_c(currstate(1),currstate(2),jj+1)=SteadyStateDistKron_ncore_c(currstate(1),currstate(2),jj+1)+1;
            end
            StationaryDistKron(:,:,:,ncore_c)=SteadyStateDistKron_ncore_c;
        end
    end
    StationaryDistKron=sum(StationaryDistKron,4);
    StationaryDistKron=reshape(StationaryDistKron,[N_a*N_z,1]);
    StationaryDistKron=StationaryDistKron./sum(StationaryDistKron,1); % Normalize conditional on age
else % simoptions.aprimedependsonage==1
    aprimeIndex_J=zeros(N_d,N_u,N_j);
    aprimeProbs_J=zeros(N_d,N_u,N_j);
    for jj=1:N_j
        aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
        [aprimeIndex,aprimeProbs]=CreateaprimeFnMatrix_Case3(aprimeFn, n_d, n_a, n_u, d_grid, a_grid, u_grid, aprimeFnParamsVec,0); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
        aprimeIndex_J(:,:,jj)=aprimeIndex;
        aprimeProbs_J(:,:,jj)=aprimeProbs;
    end

    StationaryDistKron=zeros(N_a,N_z,N_j,simoptions.ncores);
    cumsum_pi_z_J=cumsum(pi_z_J,2);
    cumsum_pi_u_J=cumsum(pi_u_J,1);
    jequaloneDistKroncumsum=cumsum(jequaloneDistKron);
    %Create simoptions.ncores different steady state distn's, then combine them.
    if simoptions.parallel==1
        parfor ncore_c=1:simoptions.ncores
            SteadyStateDistKron_ncore_c=zeros(N_a,N_z,N_j);
            % Pull a random start point from jequaloneDistKron
            currstate=find(jequaloneDistKroncumsum>rand(1,1),'first'); %Pick a random start point on the (vectorized) (a,z) grid for j=1
            currstate=ind2sub_homemade([N_a,N_z],currstate); % (a,z)
            SteadyStateDistKron_ncore_c(currstate(1),currstate(2),1)=SteadyStateDistKron_ncore_c(currstate(1),currstate(2),1)+1;
            for jj=1:(N_j-1)
                dindex=PolicyIndexesKron(currstate(1),currstate(2),jj);
                currstate(2)=find(cumsum_pi_z_J(currstate(2),:,jj)>rand(1,1),'first'); % zprime
                u_ind=find(cumsum_pi_u_J(:,jj)>rand(1,1),'first'); % u
                u_lowerupper=max(aprimeProbs(u_ind)>rand(1,1)); % value not index (hence max not find)
                currstate(1)=aprimeIndex_J(dindex,u_ind+u_lowerupper,jj); % Case3: aprime(d,u)
                SteadyStateDistKron_ncore_c(currstate(1),currstate(2),jj+1)=SteadyStateDistKron_ncore_c(currstate(1),currstate(2),jj+1)+1;
            end
            StationaryDistKron(:,:,:,ncore_c)=SteadyStateDistKron_ncore_c;
        end
    elseif simoptions.parallel==0
        for ncore_c=1:simoptions.ncores
            SteadyStateDistKron_ncore_c=zeros(N_a,N_z,N_j);
            % Pull a random start point from jequaloneDistKron
            currstate=find(jequaloneDistKroncumsum>rand(1,1),'first'); %Pick a random start point on the (vectorized) (a,z) grid for j=1
            currstate=ind2sub_homemade([N_a,N_z],currstate); % (a,z)
            SteadyStateDistKron_ncore_c(currstate(1),currstate(2),1)=SteadyStateDistKron_ncore_c(currstate(1),currstate(2),1)+1;
            for jj=1:(N_j-1)
                dindex=PolicyIndexesKron(currstate(1),currstate(2),jj);
                currstate(2)=find(cumsum_pi_z_J(currstate(2),:,jj)>rand(1,1),'first'); % zprime
                u_ind=find(cumsum_pi_u_J(:,jj)>rand(1,1),'first'); % u
                u_lowerupper=max(aprimeProbs(u_ind)>rand(1,1)); % value not index (hence max not find)
                currstate(1)=aprimeIndex_J(dindex,u_ind+u_lowerupper,jj); % Case3: aprime(d,u)
                SteadyStateDistKron_ncore_c(currstate(1),currstate(2),jj+1)=SteadyStateDistKron_ncore_c(currstate(1),currstate(2),jj+1)+1;
            end
            StationaryDistKron(:,:,:,ncore_c)=SteadyStateDistKron_ncore_c;
        end
    end
    StationaryDistKron=sum(StationaryDistKron,4);
    StationaryDistKron=reshape(StationaryDistKron,[N_a*N_z,1]);
    StationaryDistKron=StationaryDistKron./sum(StationaryDistKron,1); % Normalize conditional on age
end

% Reweight the different ages based on 'AgeWeightParamNames'. (it is
% assumed there is only one Age Weight Parameter (name))
FullParamNames=fieldnames(Parameters);
nFields=length(FullParamNames);
found=0;
for iField=1:nFields
    if strcmp(AgeWeightParamNames{1},FullParamNames{iField})
        AgeWeights=Parameters.(FullParamNames{iField});
        found=1;
    end
end
if found==0 % Have added this check so that user can see if they are missing a parameter
    error(['FAILED TO FIND PARAMETER ',AgeWeightParamNames{1}])
end
% I assume AgeWeights is a row vector
StationaryDistKron=StationaryDistKron.*(ones(N_a*N_z,1)*AgeWeights);

if MoveSSDKtoGPU==1
    StationaryDistKron=gpuArray(StationaryDistKron);
end



end
