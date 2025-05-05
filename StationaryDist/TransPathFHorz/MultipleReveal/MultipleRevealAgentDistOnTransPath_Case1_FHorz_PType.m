function [RealizedAgentDistPath, AgentDistPath]=MultipleRevealAgentDistOnTransPath_Case1_FHorz_PType(AgentDist_initial, jequalOneDist, PricePath, ParamPath, PolicyPath, AgeWeightsParamNames,n_d,n_a,n_z,N_j,Names_i,pi_z, T,Parameters, transpathoptions, simoptions)

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

% Setup N_i and Names_i
if iscell(Names_i)
    N_i=length(Names_i);
else
    N_i=Names_i;
    Names_i={'ptype001'};
    for ii=2:N_i
        if ii<10
            Names_i{ii}=['ptype00',num2str(ii)];
        elseif ii<100
            Names_i{ii}=['ptype0',num2str(ii)];
        elseif ii<1000
            Names_i{ii}=['ptype',num2str(ii)];
        end
    end
end

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
        for ii=1:N_i
            tempsize=size(AgentDist_initial.(Names_i{ii}));
            AgentDist_initial.(Names_i{ii})=reshape(AgentDist_initial.(Names_i{ii}),[numel(AgentDist_initial.(Names_i{ii}))/N_j,N_j]);
            AgentDist_initial.(Names_i{ii})=AgentDist_initial.(Names_i{ii})./sum(AgentDist_initial.(Names_i{ii}),1); % remove current age weights
            temp=ParamPath.(revealperiodnames{rr}).(AgeWeightsParamNames{1});
            AgentDist_initial.(Names_i{ii})=AgentDist_initial.(Names_i{ii}).*temp(:,1)'; % Note: we already put current ParamPath reveal into Parameters
            AgentDist_initial.(Names_i{ii})=reshape(AgentDist_initial.(Names_i{ii}),tempsize);
        end
    end

    AgentDistPath_rr=AgentDistOnTransPath_Case1_FHorz_PType(AgentDist_initial, jequalOneDist, PricePath_rr, ParamPath_rr, PolicyPath_rr, AgeWeightsParamNames,n_d,n_a,n_z,N_j,Names_i,pi_z, T,Parameters, transpathoptions, simoptions);
    AgentDistPath.(revealperiodnames{rr})=AgentDistPath_rr;

    for ii=1:N_i
        if rr==1
            temp_agentdistsize=size(VPath_rr.(Names_i{ii}));
            RealizedAgentDistPath.(Names_i{ii})=zeros([prod(temp_agentdistsize(1:end-1)),historylength]);
        end
        
        temp=RealizedAgentDistPath.(Names_i{ii});
        temp2=reshape(AgentDistPath_rr.(Names_i{ii}),[prod(temp_agentdistsize(1:end-1)),T]);
        if rr<nReveals
            temp(:,revealperiods(rr):revealperiods(rr+1)-1)=temp2(:,1:durationofreveal2(rr));
        else
            temp(:,revealperiods(rr):end)=temp2(:,1:durationofreveal2(rr));
        end
        RealizedAgentDistPath.(Names_i{ii})=temp;
    end

    %% Update the AgentDist_initial for use in the next transition
    if rr<nReveals
        AgentDist_initial=struct();
        AgentDist_initial.ptweights=AgentDistPath_rr.ptweights;
        sizesbyptype=struct(); % this is same for every reveal, buy just going to overwrite each time anyway
        for ii=1:N_i
            % Get agent dist in period durationofreveal(rr) of the current path
            sizesbyptype(ii).temp_agentdistsize=size(AgentDistPath_rr.(Names_i{ii}));
            AgentDistPath_rr_ii=reshape(AgentDistPath_rr.(Names_i{ii}),[prod(sizesbyptype(ii).temp_agentdistsize(1:end-1)),T]);
            AgentDist_initial.(Names_i{ii})=AgentDistPath_rr_ii(:,durationofreveal(rr));
            AgentDist_initial.(Names_i{ii})=reshape(AgentDist_initial.(Names_i{ii}),sizesbyptype(ii).temp_agentdistsize(1:end-1));
        end
    end
end

% Reshape for output (get them out of kron from)
for ii=1:N_i
    RealizedAgentDistPath.(Names_i{ii})=reshape(RealizedAgentDistPath.(Names_i{ii}),[sizesbyptype(ii).temp_agentdistsize(1:end-1),historylength]);
end

end