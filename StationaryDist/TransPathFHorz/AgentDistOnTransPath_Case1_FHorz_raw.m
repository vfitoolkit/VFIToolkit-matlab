function AgentDistPath=AgentDistOnTransPath_Case1_FHorz_raw(AgentDist_initial, PolicyPath, AgeWeights,n_d,n_a,n_z,N_j, T, pi_z_J, transpathoptions, simoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

AgentDist_initial=reshape(AgentDist_initial,[N_a*N_z,N_j]); % if simoptions.fastOLG==0
AgeWeights_initial=sum(AgentDist_initial,1); % [1,N_j]
if transpathoptions.ageweightstrivial==0
    AgeWeights_T=AgeWeights;
elseif transpathoptions.ageweightstrivial==1
    AgeWeights=AgeWeights_initial;
    AgeWeightsOld=AgeWeights;
end
if simoptions.fastOLG==1
    AgentDist_initial=reshape(AgentDist_initial,[N_a*N_z*N_j,1]);
    pi_z_J=permute(pi_z_J,[4,2,1,3]); % note, 4th dimension is singular: so get 1-zprime-z-j
end

%%
if simoptions.fastOLG==0
    AgentDistPath=zeros(N_a*N_z,N_j,T);
    AgentDistPath(:,:,1)=AgentDist_initial;
else % simoptions.fastOLG==1
    AgentDistPath=zeros(N_a*N_j*N_z,T);
    AgentDistPath(:,1)=AgentDist_initial;
end
PolicyPath=KronPolicyIndexes_TransPathFHorz_Case1(PolicyPath, n_d, n_a, n_z, N_j,T);

% Now we have the full PolicyIndexesPath, we go forward in time from 1
% to T using the policies to update the agents distribution generating anew price path

% Call AgentDist the current periods distn
AgentDist=AgentDist_initial;
for tt=1:T-1
    
    %Get the current optimal policy
    if N_d>0
        Policy=PolicyPath(:,:,:,:,tt);
    else
        Policy=PolicyPath(:,:,:,tt);
    end

    if transpathoptions.zpathtrivial==1
        pi_z_J=transpathoptions.pi_z_J_T(:,:,:,tt);
        if simoptions.fastOLG==1
            pi_z_J=transpathoptions.pi_z_J_T(:,:,:,:,tt);
        end
    end
    
    if transpathoptions.ageweightstrivial==0
        AgeWeightsOld=AgeWeights;
        AgeWeights=AgeWeights_T(tt,:);
    end
    if simoptions.fastOLG==0
        AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_Iteration_raw(AgentDist,AgeWeights,AgeWeightsOld,PolicyIndexesKron,N_d,N_a,N_z,N_j,pi_z_J);
        AgentDistPath(:,:,tt+1)=AgentDist; % [N_a*N_z,N_j,T]
    elseif simoptions.fastOLG==1
        AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_IterFast_raw(AgentDist,AgeWeights,AgeWeightsOld,Policy,N_d,N_a,N_z,N_j,pi_z_J);
        AgentDistPath(:,tt+1)=AgentDist; % [N_a*N_z*N_j,T]
    end
end

if simoptions.fastOLG==1
    AgentDistPath=reshape(AgentDistPath,[N_a,N_j,N_z,T]);
    AgentDistPath=permute(AgentDistPath,[1,3,2,4]);
end

AgentDistPath=reshape(AgentDistPath,[n_a,n_z,N_j,T]);


end