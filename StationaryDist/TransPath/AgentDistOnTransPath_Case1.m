function AgentDistPath=AgentDistOnTransPath_Case1(AgentDist_initial, PolicyPath,n_d,n_a,n_z,pi_z,T,simoptions)
n_e=0; % NOT YET IMPLEMENTED FOR TRANSITION PATHS

N_d=prod(n_d);
N_z=prod(n_z);
N_a=prod(n_a);


%% Check which simoptions have been used, set all others to defaults 
if exist('simoptions','var')==0
    simoptions.verbose=0;
    simoptions.gridinterplayer=0;
else
    % Check simoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
    end
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0;
    end
end

PolicyPath=KronPolicyIndexes_InfHorz_TransPath(PolicyPath, n_d, n_a, n_z,T,simoptions);

AgentDistPath=zeros(N_a*N_z,T,'gpuArray');
% Call AgentDist the current periods distn
AgentDist=reshape(AgentDist_initial,[N_a*N_z,1]);
pi_z_sparse=sparse(pi_z);
AgentDistPath(:,1)=AgentDist;
if simoptions.gridinterplayer==0
    II1=gpuArray(1:1:N_a*N_z); % Index for this period (a,z)
    IIones=ones(N_a*N_z,1,'gpuArray'); % Next period 'probabilities'
    if N_d==0
        Policy_aprime=reshape(PolicyPath(:,:,:),[N_a*N_z,T]);
    else
        Policy_aprime=reshape(PolicyPath(2,:,:,:),[N_a*N_z,T]);
    end
    Policy_aprimez=Policy_aprime+repelem(N_a*gpuArray(0:1:N_z-1)',N_a,1);
    for tt=1:T-1
        AgentDist=StationaryDist_InfHorz_TPath_SingleStep(AgentDist,gather(Policy_aprimez(:,tt)),II1,IIones,N_a,N_z,pi_z_sparse);
        AgentDistPath(:,tt+1)=AgentDist;
    end
elseif simoptions.gridinterplayer==1

    l_a=length(n_a);
    Policy_aprime=zeros(N_a*N_z,2,T,'gpuArray'); % preallocate
    PolicyProbs=zeros(N_a*N_z,2,T,'gpuArray'); % preallocate
    II2=gpuArray([1:1:N_a*N_z; 1:1:N_a*N_z]'); % Index for this period (a,z), note the 2 copies
    if N_d==0
        Policy_aprime(:,1,:)=reshape(PolicyPath(1,:,:,:),[N_a*N_z,1,T]); % lower grid point
        if l_a>1
            Policy_aprime(:,1,:)=+Policy_aprime(:,1,:)+n_a(1)*(reshape(PolicyPath(2,:,:,:),[N_a*N_z,1,T])-1);
        end
    else
        Policy_aprime(:,1,:)=reshape(PolicyPath(2,:,:,:),[N_a*N_z,1,T]); % lower grid point
        if l_a>1
            Policy_aprime(:,1,:)=+Policy_aprime(:,1,:)+n_a(1)*(reshape(PolicyPath(3,:,:,:),[N_a*N_z,1,T])-1);
        end
    end
    Policy_aprime(:,2,:)=Policy_aprime(:,1,:)+1; % upper grid point
    
    Policy_aprimez=Policy_aprime+repelem(N_a*gpuArray(0:1:N_z-1)',N_a,1);

    PolicyProbs(:,2,:)=(reshape(PolicyPath(end,:,:,:),[N_a*N_z,1,T])-1)/(1+simoptions.ngridinterp); % probability of upper grid point
    PolicyProbs(:,1,:)=1-PolicyProbs(:,2,:); % probability of lower grid point

    for tt=1:T-1
        AgentDist=StationaryDist_InfHorz_TPath_SingleStep_TwoProbs(AgentDist,Policy_aprimez(:,:,tt),II2,PolicyProbs(:,:,tt),N_a,N_z,pi_z_sparse);
        AgentDistPath(:,tt+1)=AgentDist;
    end
end


AgentDistPath=reshape(AgentDistPath,[n_a,n_z,T]);


end