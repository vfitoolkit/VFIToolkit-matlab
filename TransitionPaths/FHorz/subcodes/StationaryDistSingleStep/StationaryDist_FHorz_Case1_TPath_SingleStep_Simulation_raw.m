function AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_Simulation_raw(AgentDist,AgeWeights,PolicyIndexesKron,N_d,N_a,N_z,N_j,pi_z_J, simoptions)

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
    pi_z_J=gather(pi_z_J);
    % Use parallel cpu for these simulations
    simoptions.parallel=1;
    
    MoveSSDKtoGPU=1;
end

% Remove the existing age weights, then impose the new age weights at the end
% (note, this is unnecessary overhead when the age weights are unchanged, but can't be bothered doing a clearer version)
AgentDist=AgentDist./(ones(N_a*N_z,1)*sum(AgentDist,1)); % Note: sum(AgentDist,1) are the current age weights
jequaloneDistKroncumsum=cumsum(AgentDist); % This period. Will then wipe AgentDist so as to use it for next period
cumsum_pi_z_J=cumsum(pi_z_J,2);

if simoptions.parallel==1
    nsimspercore=ceil(simoptions.nsims/simoptions.ncores);
    %     disp('Create simoptions.ncores different steady state distns, then combine them')
    AgentDist=zeros(N_a,N_z,N_j,simoptions.ncores);
    %Create simoptions.ncores different steady state distn's, then combine them.
    if N_d==0
        parfor ncore_c=1:simoptions.ncores
            StationaryDistKron_ncore_c=zeros(N_a,N_z,N_j);
            for ii=1:nsimspercore
                % Pull a random start point from jequaloneDistKron
                currstate=find(jequaloneDistKroncumsum>rand(1,1),1,'first'); %Pick a random start point on the (vectorized) (a,z) grid for j=1
                currstate=ind2sub_homemade([N_a,N_z],currstate);
                StationaryDistKron_ncore_c(currstate(1),currstate(2),1)=StationaryDistKron_ncore_c(currstate(1),currstate(2),1)+1;
                for jj=1:(N_j-1)
                    currstate(1)=PolicyIndexesKron(currstate(1),currstate(2),jj);
                    currstate(2)=find(cumsum_pi_z_J(currstate(2),:,jj)>rand(1,1),1,'first');
                    StationaryDistKron_ncore_c(currstate(1),currstate(2),jj+1)=StationaryDistKron_ncore_c(currstate(1),currstate(2),jj+1)+1;
                end
            end
            AgentDist(:,:,:,ncore_c)=StationaryDistKron_ncore_c;
        end
        AgentDist=sum(AgentDist,4);
        AgentDist=AgentDist./sum(sum(AgentDist,1),2);
    else
        optaprime=2;
        parfor ncore_c=1:simoptions.ncores
            StationaryDistKron_ncore_c=zeros(N_a,N_z,N_j);
            for ii=1:nsimspercore
                % Pull a random start point from jequaloneDistKron
                currstate=find(jequaloneDistKroncumsum>rand(1,1),1,'first'); %Pick a random start point on the (vectorized) (a,z) grid for j=1
                currstate=ind2sub_homemade([N_a,N_z],currstate);
                StationaryDistKron_ncore_c(currstate(1),currstate(2),1)=StationaryDistKron_ncore_c(currstate(1),currstate(2),1)+1;
                for jj=1:(N_j-1)
                    currstate(1)=PolicyIndexesKron(optaprime,currstate(1),currstate(2),jj);
                    currstate(2)=find(cumsum_pi_z_J(currstate(2),:,jj)>rand(1,1),1,'first');
                    StationaryDistKron_ncore_c(currstate(1),currstate(2),jj+1)=StationaryDistKron_ncore_c(currstate(1),currstate(2),jj+1)+1;
                end
            end
            AgentDist(:,:,:,ncore_c)=StationaryDistKron_ncore_c;
        end
        AgentDist=sum(AgentDist,4);
        AgentDist=AgentDist./sum(sum(AgentDist,1),2);
    end
elseif simoptions.parallel==0
    AgentDist=zeros(N_a,N_z,N_j);
    if N_d==0
        for ii=1:simoptions.nsims
            % Pull a random start point from jequaloneDistKron
            currstate=find(jequaloneDistKroncumsum>rand(1,1),1,'first'); %Pick a random start point on the (vectorized) (a,z) grid for j=1
            currstate=ind2sub_homemade([N_a,N_z],currstate);
            AgentDist(currstate(1),currstate(2),1)=AgentDist(currstate(1),currstate(2),1)+1;
            for jj=1:(N_j-1)
                currstate(1)=PolicyIndexesKron(currstate(1),currstate(2),jj);
                currstate(2)=find(cumsum_pi_z_J(currstate(2),:,jj)>rand(1,1),1,'first');
                AgentDist(currstate(1),currstate(2),jj+1)=AgentDist(currstate(1),currstate(2),jj+1)+1;
            end
        end
        AgentDist=AgentDist./sum(sum(AgentDist,1),2);
    else
        optaprime=2;
        for ii=1:simoptions.nsims
            % Pull a random start point from jequaloneDistKron
            currstate=find(jequaloneDistKroncumsum>rand(1,1),1,'first'); %Pick a random start point on the (vectorized) (a,z) grid for j=1
            currstate=ind2sub_homemade([N_a,N_z],currstate);
            AgentDist(currstate(1),currstate(2),1)=AgentDist(currstate(1),currstate(2),1)+1;
            for jj=1:(N_j-1)
                currstate(1)=PolicyIndexesKron(optaprime,currstate(1),currstate(2),jj);
                currstate(2)=find(cumsum_pi_z_J(currstate(2),:,jj)>rand(1,1),1,'first');
                AgentDist(currstate(1),currstate(2),jj+1)=AgentDist(currstate(1),currstate(2),jj+1)+1;
            end
        end
        AgentDist=AgentDist./sum(sum(AgentDist,1),2);
    end
end


% Need to remove the old age weights, and impose the new ones
% Already removed the old age weights earlier, so now just impose the new ones.
% AgeWeights is a column vector
AgentDist=AgentDist.*shiftdim(AgeWeights,-1);

if MoveSSDKtoGPU==1
    AgentDist=gpuArray(AgentDist);
end

end
