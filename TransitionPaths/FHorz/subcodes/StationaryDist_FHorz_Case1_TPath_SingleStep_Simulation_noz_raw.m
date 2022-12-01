function AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_Simulation_noz_raw(AgentDist,AgeWeightParamNames,PolicyIndexesKron,N_d,N_a,N_z,N_j,pi_z, Parameters, simoptions)

% Options needed
%    simoptions.nsims (will do nsims per age)
%    simoptions.parallel
%    simoptions.verbose
%    simoptions.ncores
% Extra options you might want
%    simoptions.ExogShockFn
%    simoptions.ExogShockFnParamNames

MoveSSDKtoGPU=0;
if simoptions.parallel==2
    % Simulation on GPU is really slow.
    % So instead, switch to CPU.
    % For anything but ridiculously short simulations it is more than worth the overhead to switch to CPU and back.
    PolicyIndexesKron=gather(PolicyIndexesKron);
    % Use parallel cpu for these simulations
    simoptions.parallel=1;
    
    MoveSSDKtoGPU=1;
end

% Remove the existing age weights, then impose the new age weights at the end
% (note, this is unnecessary overhead when the age weights are unchanged, but can't be bothered doing a clearer version)
AgentDist=AgentDist./(ones(N_a,1)*sum(AgentDist,1)); % Note: sum(AgentDist,1) are the current age weights
jequaloneDistKroncumsum=cumsum(AgentDist); % This period. Will then wipe AgentDist so as to use it for next period

if simoptions.parallel==1
    nsimspercore=ceil(simoptions.nsims/simoptions.ncores);
    %     disp('Create simoptions.ncores different steady state distns, then combine them')
    AgentDist=zeros(N_a,N_j,simoptions.ncores);
    %Create simoptions.ncores different steady state distn's, then combine them.
    if N_d==0
        parfor ncore_c=1:simoptions.ncores
            StationaryDistKron_ncore_c=zeros(N_a,N_j);
            for ii=1:nsimspercore
                % Pull a random start point from jequaloneDistKron
                currstate=find(jequaloneDistKroncumsum>rand(1,1),1,'first'); %Pick a random start point on the (vectorized) (a,z) grid for j=1
                StationaryDistKron_ncore_c(currstate,1)=StationaryDistKron_ncore_c(currstate,1)+1;
                for jj=1:(N_j-1)
                    currstate=PolicyIndexesKron(currstate,jj);
                    StationaryDistKron_ncore_c(currstate,jj+1)=StationaryDistKron_ncore_c(currstate,jj+1)+1;
                end
            end
            AgentDist(:,:,ncore_c)=StationaryDistKron_ncore_c;
        end
        AgentDist=sum(AgentDist,3);
        AgentDist=AgentDist./sum(AgentDist,1);
    else
        parfor ncore_c=1:simoptions.ncores
            StationaryDistKron_ncore_c=zeros(N_a,N_j);
            for ii=1:nsimspercore
                % Pull a random start point from jequaloneDistKron
                currstate=find(jequaloneDistKroncumsum>rand(1,1),1,'first'); %Pick a random start point on the (vectorized) (a,z) grid for j=1
                StationaryDistKron_ncore_c(currstate,1)=StationaryDistKron_ncore_c(currstate,1)+1;
                for jj=1:(N_j-1)
                    currstate=PolicyIndexesKron(2,currstate,jj);
                    StationaryDistKron_ncore_c(currstate,jj+1)=StationaryDistKron_ncore_c(currstate,jj+1)+1;
                end
            end
            AgentDist(:,:,ncore_c)=StationaryDistKron_ncore_c;
        end
        AgentDist=sum(AgentDist,3);
        AgentDist=AgentDist./sum(AgentDist,1);
    end
elseif simoptions.parallel==0
    AgentDist=zeros(N_a,N_j);
    if N_d==0
        for ii=1:simoptions.nsims
            % Pull a random start point from jequaloneDistKron
            currstate=find(jequaloneDistKroncumsum>rand(1,1),1,'first'); %Pick a random start point on the (vectorized) (a,z) grid for j=1
            AgentDist(currstate,1)=AgentDist(currstate,1)+1;
            for jj=1:(N_j-1)
                currstate=PolicyIndexesKron(currstate,jj);
                AgentDist(currstate,jj+1)=AgentDist(currstate,jj+1)+1;
            end
        end
        AgentDist=AgentDist./sum(sum(AgentDist,1),2);
    else
        for ii=1:simoptions.nsims
            % Pull a random start point from jequaloneDistKron
            currstate=find(jequaloneDistKroncumsum>rand(1,1),1,'first'); %Pick a random start point on the (vectorized) (a,z) grid for j=1
            AgentDist(currstate,1)=AgentDist(currstate,1)+1;
            for jj=1:(N_j-1)
                currstate=PolicyIndexesKron(2,currstate,jj);
                AgentDist(currstate,jj+1)=AgentDist(currstate,jj+1)+1;
            end
        end
        AgentDist=AgentDist./sum(AgentDist,1);
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
% I assume AgeWeights is a row vector, if it has been given as column then
% transpose it.
if length(AgeWeights)~=size(AgeWeights,2)
    AgeWeights=AgeWeights';
end

% Need to remove the old age weights, and impose the new ones
% Already removed the old age weights earlier, so now just impose the new ones.
% I assume AgeWeights is a row vector
AgentDist=AgentDist.*(ones(N_a,1)*AgeWeights);

if MoveSSDKtoGPU==1
    AgentDist=gpuArray(AgentDist);
end

end
