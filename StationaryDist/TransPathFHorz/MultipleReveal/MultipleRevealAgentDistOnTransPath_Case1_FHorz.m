function [RealizedAgentDistPath, AgentDistPath]=MultipleRevealAgentDistOnTransPath_Case1_FHorz(AgentDist_initial, jequalOneDist, PricePath, ParamPath, PolicyPath, AgeWeightsParamNames,n_d,n_a,n_z,N_j,pi_z, T,Parameters, transpathoptions, simoptions)

revealperiodnames=fieldnames(ParamPath);
nReveals=length(revealperiodnames);
% Just assume that the reveals are all set up correctly, as they already solved PricePath
revealperiods=zeros(nReveals,1);
for rr=1:nReveals
    currentrevealname=revealperiodnames{rr};
    try
        revealperiods(rr)=str2double(currentrevealname(2:end));
    catch
        error('Multiple reveal transition paths: a field in ParamPath is misnamed (it must be tXXXX, where the X are numbers; you have one of the XXXX not being a number)')
    end
end
historylength=revealperiods(nReveals)+T-1; % length of realized path

for rr=1:nReveals
    PricePath_rr=PricePath.(revealperiodnames{rr});
    ParamPath_rr=ParamPath.(revealperiodnames{rr});
    PolicyPath_rr=PolicyPath.(revealperiodnames{rr});

    if rr<nReveals
        durationofreveal_rr=revealperiods(rr+1)-revealperiods(rr);
    else
        durationofreveal_rr=T;
    end

    % We already have AgentDist_initial, but if one of our ParamPaths is on the age weights, we need to impose this onto AgentDist_initial
    ParamsOnPathNames=fieldnames(ParamPath_rr);
    overlap=0;
    for ii=1:length(ParamsOnPathNames)
        if strcmp(ParamsOnPathNames{ii},AgeWeightsParamNames)
            overlap=1;
        end
    end
    if overlap==1
        tempsize=size(AgentDist_initial);
        AgentDist_initial=reshape(AgentDist_initial,[numel(AgentDist_initial)/N_j,N_j]);
        AgentDist_initial=AgentDist_initial./sum(AgentDist_initial,1); % remove current age weights
        temp=ParamPath.(revealperiodnames{rr}).(AgeWeightsParamNames{1});
        AgentDist_initial=AgentDist_initial.*temp(:,1)'; % Note: we already put current ParamPath reveal into Parameters
        AgentDist_initial=reshape(AgentDist_initial,tempsize);
    end

    AgentDistPath_rr=AgentDistOnTransPath_Case1_FHorz(AgentDist_initial, jequalOneDist, PricePath_rr, ParamPath_rr, PolicyPath_rr, AgeWeightsParamNames,n_d,n_a,n_z,N_j,pi_z, T,Parameters, transpathoptions, simoptions);
    AgentDistPath.(revealperiodnames{rr})=AgentDistPath_rr;

    if rr==1
        temp_agentdistsize=size(AgentDistPath_rr);
        RealizedAgentDistPath=zeros([prod(temp_agentdistsize(1:end-1)),historylength]);
    end

    AgentDistPath_rr=reshape(AgentDistPath_rr,[prod(temp_agentdistsize(1:end-1)),T]);
    if rr<nReveals
        RealizedAgentDistPath(:,revealperiods(rr):revealperiods(rr+1)-1)=AgentDistPath_rr(:,1:durationofreveal_rr);
    else
        RealizedAgentDistPath(:,revealperiods(rr):end)=AgentDistPath_rr(:,1:durationofreveal_rr);
    end

    %% Update the AgentDist_initial for use in the next transition
    if rr<nReveals
        % Get agent dist in period durationofreveal(rr) of the current path
        AgentDistPath_rr=reshape(AgentDistPath_rr,[prod(temp_agentdistsize(1:end-1)),T]);
        AgentDist_initial=AgentDistPath_rr(:,durationofreveal_rr);
        AgentDist_initial=reshape(AgentDist_initial,temp_agentdistsize(1:end-1));
    end
end
% Reshape for output (get them out of kron from)
RealizedAgentDistPath=reshape(RealizedAgentDistPath,[temp_agentdistsize(1:end-1),historylength]);



end