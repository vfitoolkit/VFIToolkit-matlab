function AgentDistPath=AgentDistOnTransPath_Case1_FHorz_noz(AgentDist_initial, PolicyPath, AgeWeights,n_d,n_a,N_j, T, simoptions)

N_d=prod(n_d);
N_a=prod(n_a);

AgentDist_initial=reshape(AgentDist_initial,[N_a,N_j]); % if simoptions.fastOLG==0
AgeWeights_initial=sum(AgentDist_initial,1); % [1,N_j]
if transpathoptions.ageweightstrivial==0
    AgeWeights_T=AgeWeights;
elseif transpathoptions.ageweightstrivial==1
    AgeWeights=AgeWeights_initial;
    AgeWeightsOld=AgeWeights;
end
if simoptions.fastOLG==1
    AgentDist_initial=reshape(AgentDist_initial,[N_a*N_j,1]); % if simoptions.fastOLG==0
end


%%
if simoptions.fastOLG==0
    AgentDistPath=zeros(N_a,N_j,T);
    AgentDistPath(:,:,1)=AgentDist_initial;
else % simoptions.fastOLG==1
    AgentDistPath=zeros(N_a*N_j,T);
    AgentDistPath(:,1)=AgentDist_initial;
end
PolicyPath=KronPolicyIndexes_TransPathFHorz_Case1_noz(PolicyPath, n_d, n_a, N_j,T);

% Now we have the full PolicyIndexesPath, we go forward in time from 1
% to T using the policies to update the agents distribution generating anew price path

% Call AgentDist the current periods distn
AgentDist=AgentDist_initial;
for tt=1:T-1

    %Get the current optimal policy
    if N_d>0
        Policy=PolicyPath(:,:,:,tt);
    else
        Policy=PolicyPath(:,:,tt);
    end

    if transpathoptions.ageweightstrivial==0
        AgeWeightsOld=AgeWeights;
        AgeWeights=AgeWeights_T(tt,:);
    end
    if simoptions.fastOLG==0
        AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_Iteration_noz_raw(AgentDist,AgeWeights,AgeWeightsOld,Policy,N_d,N_a,N_j);
        AgentDistPath(:,:,tt+1)=AgentDist;
    else % simoptions.fastOLG==1
        AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_IterFast_noz_raw(AgentDist,AgeWeights,AgeWeightsOld,Policy,N_d,N_a,N_j);
        AgentDistPath(:,tt+1)=AgentDist;
    end

end

AgentDistPath=reshape(AgentDistPath,[n_a,N_j,T]);





end