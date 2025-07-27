function AgentDistPath=AgentDistOnTransPath_Case1(AgentDist_initial, PolicyPath,n_d,n_a,n_z,pi_z,T,simoptions)
n_e=0; % NOT YET IMPLEMENTED FOR TRANSITION PATHS

N_d=prod(n_d);
N_z=prod(n_z);
N_a=prod(n_a);

%% Check which simoptions have been used, set all others to defaults 
if exist('simoptions','var')==0
    simoptions.nsims=10^4;
    simoptions.parallel=1+(gpuDeviceCount>0);
    simoptions.verbose=0;
    simoptions.ncores=feature('numcores'); % Number of CPU cores
    simoptions.iterate=1;
    simoptions.tolerance=10^(-9);
else
    %Check vfoptions for missing fields, if there are some fill them with
    %the defaults
    if isfield(simoptions,'tolerance')==0
        simoptions.tolerance=10^(-9);
    end
    if isfield(simoptions,'nsims')==0
        simoptions.nsims=10^4;
    end
    if isfield(simoptions,'parallel')==0
        simoptions.parallel=1+(gpuDeviceCount>0);
    end
    if isfield(simoptions,'verbose')==0
        simoptions.verbose=0;
    end
    if isfield(simoptions,'ncores')==0
        simoptions.ncores=feature('numcores'); % Number of CPU cores
    end
    if isfield(simoptions,'iterate')==0
        simoptions.iterate=1;
    end
end

PolicyPath=KronPolicyIndexes_TransPath_Case1(PolicyPath, n_d, n_a, n_z,T);

AgentDistPath=zeros(N_a*N_z,T,'gpuArray');
% Call AgentDist the current periods distn
AgentDist=reshape(AgentDist_initial,[N_a*N_z,1]);
AgentDistPath(:,1)=AgentDist;
if N_d==0 && simoptions.gridinterplayer==0
    for tt=1:T-1
        Policy=PolicyPath(:,:,tt);
        AgentDist=StationaryDist_InfHorz_TPath_SingleStep(AgentDist,Policy,N_d,N_a,N_z,sparse(pi_z));
        AgentDistPath(:,tt+1)=AgentDist;
    end
elseif N_d==0 && simoptions.gridinterplayer==1

elseif N_d>0 && simoptions.gridinterplayer==0
    for tt=1:T-1
        Policy=PolicyPath(:,:,:,tt);
        AgentDist=StationaryDist_InfHorz_TPath_SingleStep(AgentDist,Policy,N_d,N_a,N_z,sparse(pi_z));
        AgentDistPath(:,tt+1)=AgentDist;
    end
elseif N_d>0 && simoptions.gridinterplayer==1

end


AgentDistPath=reshape(AgentDistPath,[n_a,n_z,T]);


end