function StationaryDistKron=StationaryDist_FHorz_Case1_RiskyAsset_Simulation_e_raw(jequaloneDistKron,AgeWeightParamNames,PolicyIndexesKron,n_d,n_a,n_z,n_e,n_u,N_j,d_grid, a_grid,u_grid,pi_z_J,pi_e_J,pi_u,aprimeFn,Parameters,aprimeFnParamNames, simoptions)
% Case3: aprime(d,u)
% Options needed
%    simoptions.nsims
%    simoptions.parallel
%    simoptions.verbose
%    simoptions.ncores

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);
N_u=prod(n_u);

MoveSSDKtoGPU=0;
if simoptions.parallel==2
    % Simulation on GPU is really slow.
    % So instead, switch to CPU.
    % For anything but ridiculously short simulations it is more than worth the overhead to switch to CPU and back.
    PolicyIndexesKron=gather(PolicyIndexesKron);
    pi_z_J=gather(pi_z_J);
    % Use parallel cpu for these simulations
    simoptions.parallel=1;
    
    MoveSSDKtoGPU=1;
end

pi_u_J=zeros(N_u,N_j); % Not yet meaningfully implemented
for jj=1:N_j
    pi_u_J(:,jj)=pi_u;
end

if simoptions.aprimedependsonage==0
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,1);
    [aprimeIndex,aprimeProbs]=CreateaprimeFnMatrix_Case3(aprimeFn, n_d, n_a, n_u, d_grid, a_grid, u_grid, aprimeFnParamsVec,0); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    
    cumsum_pi_z_J=cumsum(pi_z_J,2);
    cumsum_pi_e_J=cumsum(pi_e_J,1);
    cumsum_pi_u_J=cumsum(pi_u_J,1);
    jequaloneDistKroncumsum=cumsum(jequaloneDistKron);
    
    if simoptions.parallel==1
        %Create simoptions.ncores different steady state distn's, then combine them.
        StationaryDistKron=zeros(N_a,N_z,N_e,N_j,simoptions.ncores);
        nsimspercore=ceil(simoptions.nsims/simoptions.ncores); % Create simoptions.ncores different steady state distns, then combine them
        parfor ncore_c=1:simoptions.ncores
            StationaryDistKron_ncore_c=zeros(N_a,N_z,N_e,N_j);
            for ii=1:nsimspercore
                % Pull a random start point from jequaloneDistKron
                currstate=find(jequaloneDistKroncumsum>rand(1,1),1,'first'); %Pick a random start point on the (vectorized) (a,z) grid for j=1
                currstate=ind2sub_homemade([N_a,N_z,N_e],currstate); % (a,z,e)
                StationaryDistKron_ncore_c(currstate(1),currstate(2),currstate(3),1)=StationaryDistKron_ncore_c(currstate(1),currstate(2),currstate(3),1)+1;
                for jj=1:(N_j-1)
                    dindex=PolicyIndexesKron(currstate(1),currstate(2),currstate(3),jj);
                    currstate(2)=find(cumsum_pi_z_J(currstate(2),:,jj)>rand(1,1),1,'first'); % zprime
                    currstate(3)=find(cumsum_pi_e_J(:,jj)>rand(1,1),1,'first'); % eprime
                    u_ind=find(cumsum_pi_u_J(:,jj)>rand(1,1),1,'first'); % u
                    u_lowerupper=max(aprimeProbs(u_ind)>rand(1,1)); % value not index (hence max not find)
                    currstate(1)=aprimeIndex(dindex,u_ind)+u_lowerupper; % Case3: aprime(d,u)
                    StationaryDistKron_ncore_c(currstate(1),currstate(2),currstate(3),jj+1)=StationaryDistKron_ncore_c(currstate(1),currstate(2),currstate(3),jj+1)+1;
                end
            end
            StationaryDistKron(:,:,:,:,ncore_c)=StationaryDistKron_ncore_c;
        end
        StationaryDistKron=sum(StationaryDistKron,5);
    elseif simoptions.parallel==0 % Only different is for instead of parfor
        StationaryDistKron=zeros(N_a,N_z,N_e,N_j);
        for ii=1:simoptions.nsims
            % Pull a random start point from jequaloneDistKron
            currstate=find(jequaloneDistKroncumsum>rand(1,1),1,'first'); %Pick a random start point on the (vectorized) (a,z) grid for j=1
            currstate=ind2sub_homemade([N_a,N_z,N_e],currstate); % (a,z,e)
            StationaryDistKron(currstate(1),currstate(2),currstate(3),1)=StationaryDistKron(currstate(1),currstate(2),currstate(3),1)+1;
            for jj=1:(N_j-1)
                dindex=PolicyIndexesKron(currstate(1),currstate(2),currstate(3),jj);
                currstate(2)=find(cumsum_pi_z_J(currstate(2),:,jj)>rand(1,1),1,'first'); % zprime
                currstate(3)=find(cumsum_pi_e_J(:,jj)>rand(1,1),1,'first'); % eprime
                u_ind=find(cumsum_pi_u_J(:,jj)>rand(1,1),1,'first'); % u
                u_lowerupper=max(aprimeProbs(u_ind)>rand(1,1)); % value not index (hence max not find)
                currstate(1)=aprimeIndex(dindex,u_ind)+u_lowerupper; % Case3: aprime(d,u)
                StationaryDistKron(currstate(1),currstate(2),currstate(3),jj+1)=StationaryDistKron(currstate(1),currstate(2),currstate(3),jj+1)+1;
            end
        end
    end
    StationaryDistKron=reshape(StationaryDistKron,[N_a*N_z*N_e,N_j]);
    StationaryDistKron=StationaryDistKron./sum(StationaryDistKron,1); % Normalize conditional on age
else
    error('aprimeFn depending on age is not presently supported when simulating agent dist, please contact me if you want/need it')
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
if size(AgeWeights,2)==1 % If it seems to be a column vector, then transpose it
    AgeWeights=AgeWeights';
end
StationaryDistKron=StationaryDistKron.*(ones(N_a*N_z*N_e,1)*AgeWeights);

if MoveSSDKtoGPU==1
    StationaryDistKron=gpuArray(StationaryDistKron);
end



end
