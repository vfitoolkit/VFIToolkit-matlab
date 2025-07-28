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
    simoptions.iterate=1;
    simoptions.tolerance=10^(-9);
    simoptions.gridinterplayer=0;
else
    %Check vfoptions for missing fields, if there are some fill them with
    %the defaults
    if ~isfield(simoptions,'tolerance')
        simoptions.tolerance=10^(-9);
    end
    if ~isfield(simoptions,'nsims')
        simoptions.nsims=10^4;
    end
    if ~isfield(simoptions,'parallel')
        simoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
    end
    if ~isfield(simoptions,'iterate')
        simoptions.iterate=1;
    end
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0;
    end
end

PolicyPath=KronPolicyIndexes_InfHorz_TransPath(PolicyPath, n_d, n_a, n_z,T,simoptions);


AgentDistPath=zeros(N_a*N_z,T,'gpuArray');
% Call AgentDist the current periods distn
AgentDist=reshape(AgentDist_initial,[N_a*N_z,1]);
AgentDistPath(:,1)=AgentDist;
if N_d==0 && simoptions.gridinterplayer==0
    Policy_aprime=reshape(PolicyPath(:,:,:),[N_a*N_z,T]);
    Policy_aprimez=Policy_aprime+repmat(N_a*gpuArray(0:1:N_z-1)',N_a,1);
    for tt=1:T-1
        AgentDist=StationaryDist_InfHorz_TPath_SingleStep(AgentDist,gather(Policy_aprimez(:,tt)),N_a,N_z,sparse(pi_z));
        AgentDistPath(:,tt+1)=AgentDist;
    end
elseif N_d>0 && simoptions.gridinterplayer==0
    Policy_aprime=reshape(PolicyPath(2,:,:,:),[N_a*N_z,T]);
    Policy_aprimez=Policy_aprime+repmat(N_a*gpuArray(0:1:N_z-1)',N_a,1);
    for tt=1:T-1
        AgentDist=StationaryDist_InfHorz_TPath_SingleStep(AgentDist,gather(Policy_aprimez(:,tt)),N_a,N_z,sparse(pi_z));
        AgentDistPath(:,tt+1)=AgentDist;
    end
elseif simoptions.gridinterplayer==1
    if N_d==0
        l_d=0;
    else
        l_d=length(n_d);
    end
    Policy_aprime=zeros(N_a*N_z,2,T,'gpuArray'); % preallocate
    PolicyProbs=zeros(N_a*N_z,2,T,'gpuArray'); % preallocate
    II2=gpuArray([1:1:N_a*N_z; 1:1:N_a*N_z]'); % Index for this period (a,z), note the 2 copies
    Policy_aprime(:,1,:)=reshape(PolicyPath(l_d+1,:,:,:),[N_a*N_z,1,T]); % lower grid point
    Policy_aprime(:,2,:)=Policy_aprime(:,1,:)+1; % upper grid point
    Policy_aprimez=Policy_aprime+repmat(N_a*gpuArray(0:1:N_z-1)',N_a,1);
    PolicyProbs(:,1,:)=reshape(PolicyPath(l_d+2,:,:,:),[N_a*N_z,1,T]); % L2 index
    PolicyProbs(:,1,:)=1-(PolicyProbs(:,1,:)-1)/(1+simoptions.ngridinterp); % probability of lower grid point
    PolicyProbs(:,2,:)=1-PolicyProbs(:,1,:); % probability of upper grid point
    for tt=1:T-1
        AgentDist=StationaryDist_InfHorz_TPath_SingleStep_TwoProbs(AgentDist,gather(Policy_aprimez(:,:,tt)),gather(II2),gather(PolicyProbs(:,:,tt)),N_a,N_z,sparse(pi_z));
        AgentDistPath(:,tt+1)=AgentDist;
    end
end


AgentDistPath=reshape(AgentDistPath,[n_a,n_z,T]);


end