function StationaryDistKron=StationaryDist_FHorz_Case1_Simulation_SemiExo_TwoProbs_raw(jequaloneDistKron,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,N_a,N_z,N_semiz,N_j,pi_z,pi_semiz_J, Parameters, simoptions)
% 'TwoProbs' refers to two probabilities.
% Policy_aprime has an additional final dimension of length 2 which is
% the two points (and contains only the aprime indexes, no d indexes as would usually be the case). 
% PolicyProbs are the corresponding probabilities of each of these two.

MoveSSDKtoGPU=0;
if simoptions.parallel==2
    % Simulation on GPU is really slow.
    % So instead, switch to CPU.
    % For anything but ridiculously short simulations it is more than worth the overhead to switch to CPU and back.
    Policy_aprime=gather(Policy_aprime);
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
            ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
            for ii=1:length(ExogShockFnParamsVec)
                ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
            end
            [~,pi_z]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
        else
            [~,pi_z]=simoptions.ExogShockFn(jj);
        end
        pi_z_J(:,:,jj)=gather(pi_z);
    end
else
    pi_z_J=repmat(pi_z,1,1,N_j);
end


if simoptions.parallel==1
    nsimspercore=ceil(simoptions.nsims/simoptions.ncores);
    %     disp('Create simoptions.ncores different steady state distns, then combine them')
    StationaryDistKron=zeros(N_a,N_z,N_semiz,N_j,simoptions.ncores);
    cumsum_pi_z_J=cumsum(pi_z_J,2);
    cumsum_pi_semiz_J=cumsum(pi_semiz_J,2);
    jequaloneDistKroncumsum=cumsum(jequaloneDistKron);
    %Create simoptions.ncores different steady state distn's, then combine them.
    parfor ncore_c=1:simoptions.ncores
        StationaryDistKron_ncore_c=zeros(N_a,N_z,N_semiz,N_j);
        for ii=1:nsimspercore
            % Pull a random start point from jequaloneDistKron
            currstate=find(jequaloneDistKroncumsum>rand(1,1),1,'first'); %Pick a random start point on the (vectorized) (a,z) grid for j=1
            currstate=ind2sub_homemade([N_a,N_z,N_semiz],currstate);
            StationaryDistKron_ncore_c(currstate(1),currstate(2),currstate(3),1)=StationaryDistKron_ncore_c(currstate(1),currstate(2),currstate(3),1)+1;
            for jj=1:(N_j-1)
                whichoftwo=2-(PolicyProbs(currstate(1),currstate(2),jj,1)>rand(1)); % if the random number is less than the probability, then first, otherwise second
                d_semiexo=Policy_dsemiexo(currstate(1),currstate(2),currstate(3));
                currstate(1)=Policy_aprime(currstate(1),currstate(2),currstate(3),jj,whichoftwo);
                currstate(2)=find(cumsum_pi_z_J(currstate(2),:,jj)>rand(1,1),1,'first');
                currstate(3)=find(cumsum_pi_semiz_J(currstate(3),:,d_semiexo,jj)>rand(1,1),1,'first');
                StationaryDistKron_ncore_c(currstate(1),currstate(2),currstate(3),jj+1)=StationaryDistKron_ncore_c(currstate(1),currstate(2),currstate(3),jj+1)+1;
            end
        end
        StationaryDistKron(:,:,:,:,ncore_c)=StationaryDistKron_ncore_c;
    end
    StationaryDistKron=sum(StationaryDistKron,5);
    StationaryDistKron=StationaryDistKron./sum(sum(sum(StationaryDistKron,1),2),3);
elseif simoptions.parallel==0
    StationaryDistKron=zeros(N_a,N_z,N_semiz,N_j);
    cumsum_pi_z_J=cumsum(pi_z_J,2);
    cumsum_pi_semiz_J=cumsum(pi_semiz_J,2);
    jequaloneDistKroncumsum=cumsum(jequaloneDistKron);
    for ii=1:simoptions.nsims
        % Pull a random start point from jequaloneDistKron
        currstate=find(jequaloneDistKroncumsum>rand(1,1),1,'first'); %Pick a random start point on the (vectorized) (a,z) grid for j=1
        currstate=ind2sub_homemade([N_a,N_z,N_semiz],currstate);
        StationaryDistKron(currstate(1),currstate(2),1)=StationaryDistKron(currstate(1),currstate(2),1)+1;
        for jj=1:(N_j-1)
            whichoftwo=2-(PolicyProbs(currstate(1),currstate(2),jj,1)>rand(1)); % if the random number is less than the probability, then first, otherwise second
            d_semiexo=Policy_dsemiexo(currstate(1),currstate(2),currstate(3));
            currstate(1)=Policy_aprime(currstate(1),currstate(2),currstate(3),jj,whichoftwo);
            currstate(2)=find(cumsum_pi_z_J(currstate(2),:,jj)>rand(1,1),1,'first');
            currstate(3)=find(cumsum_pi_semiz_J(currstate(3),:,d_semiexo,jj)>rand(1,1),1,'first');
            StationaryDistKron(currstate(1),currstate(2),currstate(3),jj+1)=StationaryDistKron(currstate(1),currstate(2),currstate(3),jj+1)+1;
        end
    end
    StationaryDistKron=StationaryDistKron./sum(sum(sum(StationaryDistKron,1),2),3);
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
StationaryDistKron=StationaryDistKron.*shiftdim(AgeWeights,-2); % -2 because of the semi-exogenous shocks

if MoveSSDKtoGPU==1
    StationaryDistKron=gpuArray(StationaryDistKron);
end

end
