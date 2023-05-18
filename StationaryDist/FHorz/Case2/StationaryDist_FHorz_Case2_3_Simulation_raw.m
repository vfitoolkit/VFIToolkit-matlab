function StationaryDistKron=StationaryDist_FHorz_Case2_3_Simulation_raw(jequaloneDistKron,AgeWeightParamNames,PolicyIndexesKron,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid,pi_z,Phi_aprime,Case2_Type,Parameters,PhiaprimeParamNames, simoptions)
% Case2_Type=3: aprime(d,z')
%
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

if simoptions.phiaprimedependsonage==0
    PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
    if simoptions.lowmemory==0
        Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid, PhiaprimeParamsVec);
    end
    
    if simoptions.parallel==1
        StationaryDistKron=zeros(N_a,N_z,N_j,simoptions.ncores);
        cumsum_pi_z_J=cumsum(pi_z_J,2);
        jequaloneDistKroncumsum=cumsum(jequaloneDistKron);
        %Create simoptions.ncores different steady state distn's, then combine them.
        parfor ncore_c=1:simoptions.ncores
            StationaryDistKron_ncore_c=zeros(N_a,N_z,N_j);
            % Pull a random start point from jequaloneDistKron
            currstate=find(jequaloneDistKroncumsum>rand(1,1),1,'first'); %Pick a random start point on the (vectorized) (a,z) grid for j=1
            currstate=ind2sub_homemade([N_a,N_z],currstate); % (a,z)
            StationaryDistKron_ncore_c(currstate(1),currstate(2),1)=StationaryDistKron_ncore_c(currstate(1),currstate(2),1)+1;
            for jj=1:(N_j-1)
                dindex=PolicyIndexesKron(currstate(1),currstate(2),jj);
                currstate(2)=find(cumsum_pi_z_J(currstate(2),:,jj)>rand(1,1),1,'first'); % zprime
                currstate(1)=Phi_aprimeMatrix(dindex,currstate(2)); % Case2_Type=3: aprime(d,z')
                StationaryDistKron_ncore_c(currstate(1),currstate(2),jj+1)=StationaryDistKron_ncore_c(currstate(1),currstate(2),jj+1)+1;
            end
            StationaryDistKron(:,:,:,ncore_c)=StationaryDistKron_ncore_c;
        end
        StationaryDistKron=sum(StationaryDistKron,4);
    elseif simoptions.parallel==0 % Only different is for instead of parfor
        StationaryDistKron=zeros(N_a,N_z,N_j);
        cumsum_pi_z_J=cumsum(pi_z_J,2);
        jequaloneDistKroncumsum=cumsum(jequaloneDistKron);
        %Create simoptions.ncores different steady state distn's, then combine them.
        for ii=1:simoptions.ncores
            % Pull a random start point from jequaloneDistKron
            currstate=find(jequaloneDistKroncumsum>rand(1,1),1,'first'); %Pick a random start point on the (vectorized) (a,z) grid for j=1
            currstate=ind2sub_homemade([N_a,N_z],currstate); % (a,z)
            StationaryDistKron(currstate(1),currstate(2),1)=StationaryDistKron(currstate(1),currstate(2),1)+1;
            for jj=1:(N_j-1)
                dindex=PolicyIndexesKron(currstate(1),currstate(2),jj);
                currstate(2)=find(cumsum_pi_z_J(currstate(2),:,jj)>rand(1,1),1,'first'); % zprime
                currstate(1)=Phi_aprimeMatrix(dindex,currstate(2)); % Case2_Type=3: aprime(d,z')
                StationaryDistKron(currstate(1),currstate(2),jj+1)=StationaryDistKron(currstate(1),currstate(2),jj+1)+1;
            end
        end
    end
    StationaryDistKron=reshape(StationaryDistKron,[N_a*N_z,N_j]);
    StationaryDistKron=StationaryDistKron./sum(StationaryDistKron,1); % Normalize conditional on age
else
    error('Phi_aprime depending on age is not presently supported when simulating agent dist, please contact me if you want/need it')
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
StationaryDistKron=StationaryDistKron.*(ones(N_a*N_z,1)*AgeWeights);

if MoveSSDKtoGPU==1
    StationaryDistKron=gpuArray(StationaryDistKron);
end



end
