function StationaryDistKron=StationaryDist_FHorz_Case1_SemiExo_Simulation_e_raw(jequaloneDistKron,AgeWeightParamNames,PolicyIndexesKron,n_d1,n_d2,N_a,N_z,N_semiz,N_e,N_j,pi_z_J,pi_semiz_J,pi_e_J, Parameters, simoptions)
% Simulates path based on PolicyIndexes of length 'periods' after a burn
% in of length 'burnin' (burn-in are the initial run of points that are then dropped)
%
% Options needed
%    simoptions.nsims
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
    pi_e_J=gather(pi_e_J);
    % Use parallel cpu for these simulations
    simoptions.parallel=1;
    
    MoveSSDKtoGPU=1;
end


%%
if simoptions.parallel==1
    nsimspercore=ceil(simoptions.nsims/simoptions.ncores);
    StationaryDistKron=zeros(N_a,N_semiz,N_z,N_e,N_j,simoptions.ncores);
    cumsum_pi_z_J=cumsum(pi_z_J,2);
    cumsum_pi_semiz_J=cumsum(pi_semiz_J,2);
    cumsum_pi_e_J=cumsum(pi_e_J,1);
    jequaloneDistKroncumsum=cumsum(jequaloneDistKron);
    %Create simoptions.ncores different steady state distn's, then combine them.
    for ncore_c=1:simoptions.ncores
        StationaryDistKron_ncore_c=zeros(N_a,N_semiz,N_z,N_e,N_j);
        for ii=1:nsimspercore
            % Pull a random start point from jequaloneDistKron
            currstate=find(jequaloneDistKroncumsum>rand(1,1),1,'first'); %Pick a random start point on the (vectorized) (a,z) grid for j=1
            currstate=ind2sub_homemade([N_a,N_semiz,N_z,N_e],currstate);
            StationaryDistKron_ncore_c(currstate(1),currstate(2),currstate(3),currstate(4),1)=StationaryDistKron_ncore_c(currstate(1),currstate(2),currstate(3),currstate(4),1)+1;
            for jj=1:(N_j-1)
                dsub=ind2sub_homemade([n_d1,n_d2],PolicyIndexesKron(1,currstate(1),currstate(2)+N_semiz*(currstate(3)-1),currstate(4),jj));
                currstate(1)=PolicyIndexesKron(2,currstate(1),currstate(2)+N_semiz*(currstate(3)-1),currstate(4),jj);
                currstate(2)=find(cumsum_pi_semiz_J(currstate(2),:,dsub(end),jj)>rand(1,1),1,'first');
                currstate(3)=find(cumsum_pi_z_J(currstate(3),:,jj)>rand(1,1),1,'first');
                currstate(4)=find(cumsum_pi_e_J(:,jj)>rand(1,1),1,'first');
                StationaryDistKron_ncore_c(currstate(1),currstate(2),currstate(3),currstate(4),jj+1)=StationaryDistKron_ncore_c(currstate(1),currstate(2),currstate(3),currstate(4),jj+1)+1;
            end
        end
        StationaryDistKron(:,:,:,:,:,ncore_c)=StationaryDistKron_ncore_c;
    end
    StationaryDistKron=sum(StationaryDistKron,6);
    StationaryDistKron=StationaryDistKron./sum(sum(sum(sum(StationaryDistKron,1),2),3),4);
elseif simoptions.parallel==0 % Note: You probably never want to actually use this.
    StationaryDistKron=zeros(N_a,N_semiz,N_z,N_e,N_j);
    cumsum_pi_z_J=cumsum(pi_z_J,2);
    cumsum_pi_semiz_J=cumsum(pi_semiz_J,2);
    cumsum_pi_e_J=cumsum(pi_e_J,1);
    jequaloneDistKroncumsum=cumsum(jequaloneDistKron);
    for ii=1:simoptions.nsims
        % Pull a random start point from jequaloneDistKron
        currstate=find(jequaloneDistKroncumsum>rand(1,1),1,'first'); %Pick a random start point on the (vectorized) (a,z) grid for j=1
        currstate=ind2sub_homemade([N_a,N_semiz,N_z,N_e],currstate);
        StationaryDistKron(currstate(1),currstate(2),currstate(3),currstate(4),1)=StationaryDistKron(currstate(1),currstate(2),currstate(3),currstate(4),1)+1;
        for jj=1:(N_j-1)
            dsub=ind2sub_homemade([n_d1,n_d2],PolicyIndexesKron(1,currstate(1),currstate(2)+N_semiz*(currstate(3)-1),currstate(4),jj));
            currstate(1)=PolicyIndexesKron(2,currstate(1),currstate(2)+N_semiz*(currstate(3)-1),currstate(4),jj);
            currstate(2)=find(cumsum_pi_semiz_J(currstate(2),:,dsub(end),jj)>rand(1,1),1,'first');
            currstate(3)=find(cumsum_pi_z_J(currstate(3),:,jj)>rand(1,1),1,'first');
            currstate(4)=find(cumsum_pi_e_J(:,jj)>rand(1,1),1,'first');
            StationaryDistKron(currstate(1),currstate(2),currstate(3),currstate(4),jj+1)=StationaryDistKron(currstate(1),currstate(2),currstate(3),currstate(4),jj+1)+1;
        end
    end
    StationaryDistKron=StationaryDistKron./sum(sum(sum(sum(StationaryDistKron,1),2),3),4);
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
% I assume AgeWeights is a row vector, if it has been given as column then transpose it.
if length(AgeWeights)~=size(AgeWeights,2)
    AgeWeights=AgeWeights';
end
StationaryDistKron=StationaryDistKron.*shiftdim(AgeWeights,-3);  % -3 because of the semi-exogenous shocks and e vars

if MoveSSDKtoGPU==1
    StationaryDistKron=gpuArray(StationaryDistKron);
end

end
