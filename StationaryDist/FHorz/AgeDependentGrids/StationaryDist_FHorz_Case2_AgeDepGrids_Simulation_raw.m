function StationaryDistKron=StationaryDist_FHorz_Case2_AgeDepGrids_Simulation_raw(jequaloneDistKron,AgeWeightParamNames,PolicyIndexesKron,daz_gridstructure,N_j,Phi_aprimeFn,Case2_Type,Parameters,PhiaprimeParamNames, simoptions)
%Simulates path based on PolicyIndexes of length 'periods' after a burn
%in of length 'burnin' (burn-in are the initial run of points that are then
%dropped)

% Options needed
%    simoptions.nsims
%    simoptions.parallel
%    simoptions.verbose
%    simoptions.ncores

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

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

if simoptions.parallel==1
    nsimspercore=ceil(simoptions.nsims/simoptions.ncores);
    %     disp('Create simoptions.ncores different steady state distns, then combine them')
    StationaryDistKron=zeros(N_a*N_z,N_j,simoptions.ncores);
    cumsum_pi_z_J=cumsum(pi_z_J,2);
    jequaloneDistKroncumsum=cumsum(jequaloneDistKron);
    %Create simoptions.ncores different steady state distn's, then combine them.
    if N_d==0
        parfor ncore_c=1:simoptions.ncores
            SteadyStateDistKron_ncore_c=zeros(N_a,N_z,N_j);
            % Pull a random start point from jequaloneDistKron
            currstate=max(jequaloneDistKroncumsum>rand(1,1)); %Pick a random start point on the (vectorized) (a,z) grid for j=1
            currstate=ind2sub_homemade([N_a,N_z],currstate);
            SteadyStateDistKron_ncore_c(currstate(1),currstate(2),1)=SteadyStateDistKron_ncore_c(currstate(1),currstate(2),1)+1;
            for jj=1:(N_j-1)
                currstate(1)=PolicyIndexesKron(currstate(1),currstate(2),jj);
                currstate(2)=max(cumsum_pi_z_J(currstate(2),:,jj)>rand(1,1)); %NEED TO IMPLEMENT OPTION OF AGE DEPENDENT EXOGSHOCKFN
                SteadyStateDistKron_ncore_c(currstate(1),currstate(2),jj+1)=SteadyStateDistKron_ncore_c(currstate(1),currstate(2),jj+1)+1;
            end
            StationaryDistKron(:,:,:,ncore_c)=SteadyStateDistKron_ncore_c;
        end
        StationaryDistKron=sum(StationaryDistKron,4);
        StationaryDistKron=StationaryDistKron./sum(sum(StationaryDistKron,1),2);
    else
        optaprime=2;
        parfor ncore_c=1:simoptions.ncores
            SteadyStateDistKron_ncore_c=zeros(N_a,N_z,N_j);
            % Pull a random start point from jequaloneDistKron
            currstate=max(jequaloneDistKroncumsum>rand(1,1)); %Pick a random start point on the (vectorized) (a,z) grid for j=1
            currstate=ind2sub_homemade([N_a,N_z],currstate);
            SteadyStateDistKron_ncore_c(currstate(1),currstate(2),1)=SteadyStateDistKron_ncore_c(currstate(1),currstate(2),1)+1;
            for jj=1:(N_j-1)
                currstate(1)=PolicyIndexesKron(optaprime,currstate(1),currstate(2),jj);
                currstate(2)=max(cumsum_pi_z_J(currstate(2),:,jj)>rand(1,1)); %NEED TO IMPLEMENT OPTION OF AGE DEPENDENT EXOGSHOCKFN
                SteadyStateDistKron_ncore_c(currstate(1),currstate(2),jj+1)=SteadyStateDistKron_ncore_c(currstate(1),currstate(2),jj+1)+1;
            end
            StationaryDistKron(:,:,:,ncore_c)=SteadyStateDistKron_ncore_c;
        end
        StationaryDistKron=sum(StationaryDistKron,4);
        StationaryDistKron=StationaryDistKron./sum(sum(StationaryDistKron,1),2);
    end
elseif simoptions.parallel==0
    nsimspercore=ceil(simoptions.nsims/simoptions.ncores);
    %     disp('Create simoptions.ncores different steady state distns, then combine them')
    StationaryDistKron=zeros(N_a*N_z,N_j,simoptions.ncores);
    cumsum_pi_z_J=cumsum(pi_z_J,2);
    jequaloneDistKroncumsum=cumsum(jequaloneDistKron);
    %Create simoptions.ncores different steady state distn's, then combine them.
    if N_d==0
        SteadyStateDistKron=zeros(N_a,N_z,N_j);
        % Pull a random start point from jequaloneDistKron
        currstate=max(jequaloneDistKroncumsum>rand(1,1)); %Pick a random start point on the (vectorized) (a,z) grid for j=1
        currstate=ind2sub_homemade([N_a,N_z],currstate);
        SteadyStateDistKron(currstate(1),currstate(2),1)=SteadyStateDistKron(currstate(1),currstate(2),1)+1;
        for jj=1:(N_j-1)
            currstate(1)=PolicyIndexesKron(currstate(1),currstate(2),jj);
            currstate(2)=max(cumsum_pi_z_J(currstate(2),:,jj)>rand(1,1)); %NEED TO IMPLEMENT OPTION OF AGE DEPENDENT EXOGSHOCKFN
            SteadyStateDistKron(currstate(1),currstate(2),jj+1)=SteadyStateDistKron(currstate(1),currstate(2),jj+1)+1;
        end
        StationaryDistKron=sum(StationaryDistKron,4);
        StationaryDistKron=StationaryDistKron./sum(sum(StationaryDistKron,1),2);
    else
        optaprime=2;
        SteadyStateDistKron=zeros(N_a,N_z,N_j);
        % Pull a random start point from jequaloneDistKron
        currstate=max(jequaloneDistKroncumsum>rand(1,1)); %Pick a random start point on the (vectorized) (a,z) grid for j=1
        currstate=ind2sub_homemade([N_a,N_z],currstate);
        SteadyStateDistKron(currstate(1),currstate(2),1)=SteadyStateDistKron(currstate(1),currstate(2),1)+1;
        for jj=1:(N_j-1)
            currstate(1)=PolicyIndexesKron(optaprime,currstate(1),currstate(2),jj);
            currstate(2)=max(cumsum_pi_z_J(currstate(2),:,jj)>rand(1,1)); %NEED TO IMPLEMENT OPTION OF AGE DEPENDENT EXOGSHOCKFN
            SteadyStateDistKron(currstate(1),currstate(2),jj+1)=SteadyStateDistKron(currstate(1),currstate(2),jj+1)+1;
        end
        StationaryDistKron=sum(StationaryDistKron,4);
        StationaryDistKron=StationaryDistKron./sum(sum(StationaryDistKron,1),2);
    end
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
    fprintf(['FAILED TO FIND PARAMETER ',AgeWeightParamNames{1}])
end
% I assume AgeWeights is a row vector
StationaryDistKron=StationaryDistKron.*(ones(N_a*N_z,1)*AgeWeights);

if MoveSSDKtoGPU==1
    StationaryDistKron=gpuArray(StationaryDistKron);
end



end
