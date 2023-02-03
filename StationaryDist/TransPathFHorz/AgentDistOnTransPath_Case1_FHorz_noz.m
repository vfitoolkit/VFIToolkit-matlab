function AgentDistPath=AgentDistOnTransPath_Case1_FHorz_noz(AgentDist_initial,PricePath, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, PolicyPath, AgeWeightsParamNames,n_d,n_a,N_j, T,Parameters, transpathoptions, simoptions)

N_d=prod(n_d);
N_a=prod(n_a);

%%
PolicyPath=KronPolicyIndexes_TransPathFHorz_Case1_noz(PolicyPath, n_d, n_a, N_j,T);
AgentDistPath=zeros(N_a,N_j,T);

% Now we have the full PolicyIndexesPath, we go forward in time from 1
% to T using the policies to update the agents distribution generating anew price path

% Call AgentDist the current periods distn
AgentDist_initial=reshape(AgentDist_initial,[N_a,N_j]);
AgentDistPath(:,:,1)=AgentDist_initial;
AgentDist=AgentDist_initial;
for tt=1:T-1
    
    %Get the current optimal policy
    if N_d>0
        Policy=PolicyPath(:,:,:,tt);
    else
        Policy=PolicyPath(:,:,tt);
    end
    
    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
    end
    
    AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_noz(AgentDist,AgeWeightsParamNames,Policy,n_d,n_a,N_j,Parameters,simoptions);
    
    AgentDistPath(:,:,tt+1)=AgentDist;

end

AgentDistPath=reshape(AgentDistPath,[n_a,N_j,T]);





end