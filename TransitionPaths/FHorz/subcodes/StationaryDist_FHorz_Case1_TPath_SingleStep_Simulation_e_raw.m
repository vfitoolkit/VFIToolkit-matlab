function AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_Simulation_e_raw(AgentDist,AgeWeightParamNames,PolicyIndexesKron,N_d,N_a,N_z,N_e,N_j,pi_z,pi_e, Parameters, simoptions)

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
    pi_z=gather(pi_z);
    pi_e=gather(pi_e);
    % Use parallel cpu for these simulations
    simoptions.parallel=1;
    
    MoveSSDKtoGPU=1;
end

% This implementation is slightly inefficient when shocks are not age dependent, but speed loss is fairly trivial
eval('fieldexists_pi_z_J=1;vfoptions.pi_z_J;','fieldexists_pi_z_J=0;')
eval('fieldexists_ExogShockFn=1;vfoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;vfoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')
if fieldexists_pi_z_J==0 && fieldexists_ExogShockFn==0
    pi_z_J=pi_z.*ones(1,1,N_j);
elseif fieldexists_pi_z_J==1
    pi_z_J=vfoptions.pi_z_J;
elseif fieldexists_ExogShockFn==1
    pi_z_J=zeros(N_z,N_z,N_j,'gpuArray');
    for jj=1:N_j
        ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
        ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
        for ii=1:length(ExogShockFnParamsVec)
            ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
        end
        [~,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
        pi_z_J(:,:,jj)=gpuArray(pi_z);
    end
end
eval('fieldexists_pi_e_J=1;vfoptions.pi_e_J;','fieldexists_pi_e_J=0;')
eval('fieldexists_EiidShockFn=1;vfoptions.EiidShockFn;','fieldexists_EiidShockFn=0;')
eval('fieldexists_EiidShockFnParamNames=1;vfoptions.EiidShockFnParamNames;','fieldexists_EiidShockFnParamNames=0;')
if fieldexists_pi_e_J==0 && fieldexists_EiidShockFn==0
    pi_e_J=pi_e.*ones(1,N_j);
elseif fieldexists_pi_e_J==1
    pi_e_J=vfoptions.pi_e_J;
elseif fieldexists_EiidShockFn==1
    pi_e_J=zeros(N_e,N_e,N_j,'gpuArray');
    for jj=1:N_j
        EiidShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.EiidShockFnParamNames,jj);
        EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
        for ii=1:length(EiidShockFnParamsVec)
            EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
        end
        [~,pi_e]=vfoptions.EiidShockFn(EiidShockFnParamsCell{:});
        pi_e_J(:,jj)=gpuArray(pi_e);
    end
end


% Remove the existing age weights, then impose the new age weights at the end
% (note, this is unnecessary overhead when the age weights are unchanged, but can't be bothered doing a clearer version)
AgentDist=AgentDist./(ones(N_a*N_z*N_e,1)*sum(AgentDist,1)); % Note: sum(AgentDist,1) are the current age weights
jequaloneDistKroncumsum=cumsum(AgentDist); % This period. Will then wipe AgentDist so as to use it for next period
cumsum_pi_z_J=cumsum(pi_z_J,2);
cumsum_pi_e_J=cumsum(pi_e_J,1);

if simoptions.parallel==1
    nsimspercore=ceil(simoptions.nsims/simoptions.ncores);
    %     disp('Create simoptions.ncores different steady state distns, then combine them')
    AgentDist=zeros(N_a,N_z,N_e,N_j,simoptions.ncores);
    %Create simoptions.ncores different steady state distn's, then combine them.
    if N_d==0
        parfor ncore_c=1:simoptions.ncores
            StationaryDistKron_ncore_c=zeros(N_a,N_z,N_e,N_j);
            for ii=1:nsimspercore
                % Pull a random start point from jequaloneDistKron
                currstate=find(jequaloneDistKroncumsum>rand(1,1),1,'first'); %Pick a random start point on the (vectorized) (a,z) grid for j=1
                currstate=ind2sub_homemade([N_a,N_z,N_e],currstate);
                StationaryDistKron_ncore_c(currstate(1),currstate(2),currstate(3),1)=StationaryDistKron_ncore_c(currstate(1),currstate(2),currstate(3),1)+1;
                for jj=1:(N_j-1)
                    currstate(1)=PolicyIndexesKron(currstate(1),currstate(2),currstate(3),jj);
                    currstate(2)=find(cumsum_pi_z_J(currstate(2),:,jj)>rand(1,1),1,'first');
                    currstate(3)=find(cumsum_pi_e_J(:,jj)>rand(1,1),1,'first');
                    StationaryDistKron_ncore_c(currstate(1),currstate(2),currstate(3),jj+1)=StationaryDistKron_ncore_c(currstate(1),currstate(2),currstate(3),jj+1)+1;
                end
            end
            AgentDist(:,:,:,:,ncore_c)=StationaryDistKron_ncore_c;
        end
    else
        optaprime=2;
        parfor ncore_c=1:simoptions.ncores
            StationaryDistKron_ncore_c=zeros(N_a,N_z,N_e,N_j);
            for ii=1:nsimspercore
                % Pull a random start point from jequaloneDistKron
                currstate=find(jequaloneDistKroncumsum>rand(1,1),1,'first'); %Pick a random start point on the (vectorized) (a,z) grid for j=1
                currstate=ind2sub_homemade([N_a,N_z,N_e],currstate);
                StationaryDistKron_ncore_c(currstate(1),currstate(2),currstate(3),1)=StationaryDistKron_ncore_c(currstate(1),currstate(2),currstate(3),1)+1;
                for jj=1:(N_j-1)
                    currstate(1)=PolicyIndexesKron(optaprime,currstate(1),currstate(2),currstate(3),jj);
                    currstate(2)=find(cumsum_pi_z_J(currstate(2),:,jj)>rand(1,1),1,'first');
                    currstate(3)=find(cumsum_pi_e_J(,:,jj)>rand(1,1),1,'first');
                    StationaryDistKron_ncore_c(currstate(1),currstate(2),currstate(3),jj+1)=StationaryDistKron_ncore_c(currstate(1),currstate(2),currstate(3),jj+1)+1;
                end
            end
            AgentDist(:,:,:,:,ncore_c)=StationaryDistKron_ncore_c;
        end
    end
    AgentDist=sum(AgentDist,5);
    AgentDist=AgentDist./sum(sum(sum(AgentDist,1),2),3);
elseif simoptions.parallel==0
    AgentDist=zeros(N_a,N_z,N_e,N_j);
    if N_d==0
        for ii=1:simoptions.nsims
            % Pull a random start point from jequaloneDistKron
            currstate=find(jequaloneDistKroncumsum>rand(1,1),1,'first'); %Pick a random start point on the (vectorized) (a,z) grid for j=1
            currstate=ind2sub_homemade([N_a,N_z,N_e],currstate);
            AgentDist(currstate(1),currstate(2),currstate(3),1)=AgentDist(currstate(1),currstate(2),currstate(3),1)+1;
            for jj=1:(N_j-1)
                currstate(1)=PolicyIndexesKron(currstate(1),currstate(2),currstate(3),jj);
                currstate(2)=find(cumsum_pi_z_J(currstate(2),:,jj)>rand(1,1),1,'first');
                currstate(3)=find(cumsum_pi_e_J(:,jj)>rand(1,1),1,'first');
                AgentDist(currstate(1),currstate(2),currstate(3),jj+1)=AgentDist(currstate(1),currstate(2),currstate(3),jj+1)+1;
            end
    else
        optaprime=2;
        for ii=1:simoptions.nsims
            % Pull a random start point from jequaloneDistKron
            currstate=find(jequaloneDistKroncumsum>rand(1,1),1,'first'); %Pick a random start point on the (vectorized) (a,z) grid for j=1
            currstate=ind2sub_homemade([N_a,N_z,N_e],currstate);
            AgentDist(currstate(1),currstate(2),currstate(3),1)=AgentDist(currstate(1),currstate(2),currstate(3),1)+1;
            for jj=1:(N_j-1)
                currstate(1)=PolicyIndexesKron(optaprime,currstate(1),currstate(2),currstate(3),jj);
                currstate(2)=find(cumsum_pi_z_J(currstate(2),:,jj)>rand(1,1),1,'first');
                currstate(3)=find(cumsum_pi_e_J(,:,jj)>rand(1,1),1,'first');
                AgentDist(currstate(1),currstate(2),currstate(3),jj+1)=AgentDist(currstate(1),currstate(2),currstate(3),jj+1)+1;
            end
        end
    end
    AgentDist=AgentDist./sum(sum(sum(AgentDist,1),2),3);
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
AgentDist=AgentDist.*(ones(N_a*N_z*N_e,1)*AgeWeights);

if MoveSSDKtoGPU==1
    AgentDist=gpuArray(AgentDist);
end

end
