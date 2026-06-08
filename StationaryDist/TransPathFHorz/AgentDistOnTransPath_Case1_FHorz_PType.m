function AgentDistPath=AgentDistOnTransPath_Case1_FHorz_PType(AgentDist_initial, jequalOneDist, PricePath, ParamPath, PolicyPath, AgeWeightsParamNames,n_d,n_a,n_z,N_j,Names_i,pi_z, T,Parameters, transpathoptions, simoptions)
% Remark to self: No real need for T as input, as this is anyway the length of PricePath

AgentDistPath=struct();

%%
if iscell(Names_i)
    N_i=length(Names_i);
else
    N_i=Names_i; % It is the number of PTypes (which have not been given names)
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


%% Loop over permanent types
for ii=1:N_i
    iistr=Names_i{ii};

    % First set up transpathoptions
    if exist('transpathoptions','var')
        transpathoptions_temp=PType_Options(transpathoptions,iistr);
        if ~isfield(transpathoptions_temp,'verbose')
            transpathoptions_temp.verbose=0;
        end
        if ~isfield(transpathoptions_temp,'verboseparams')
            transpathoptions_temp.verboseparams=0;
        end
    else
        transpathoptions_temp.verbose=0;
        transpathoptions_temp.verboseparams=0;
    end

    % First set up simoptions
    if exist('simoptions','var')
        simoptions_temp=PType_Options(simoptions,iistr);
        if ~isfield(simoptions_temp,'verbose')
            simoptions_temp.verbose=0;
        end
        if ~isfield(simoptions_temp,'verboseparams')
            simoptions_temp.verboseparams=0;
        end
        if ~isfield(simoptions_temp,'ptypestorecpu')
            simoptions_temp.ptypestorecpu=1; % GPU memory is limited, so switch solutions to the cpu
        end
    else
        simoptions_temp.verbose=0;
        simoptions_temp.verboseparams=0;
        simoptions_temp.ptypestorecpu=1; % GPU memory is limited, so switch solutions to the cpu
    end

    if simoptions_temp.verbose==1
        fprintf('Permanent type: %i of %i \n',ii, N_i)
    end

    PolicyPath_temp=PolicyPath.(iistr);

    %% Go through everything which might be dependent on fixed type (PType)
    if isstruct(n_d)
        n_d_temp=n_d.(iistr);
    else
        n_d_temp=n_d;
    end
    if isstruct(n_a)
        n_a_temp=n_a.(iistr);
    else
        n_a_temp=n_a;
    end

    if isstruct(N_j)
        N_j_temp=N_j.(iistr);
    else
        N_j_temp=N_j;
    end

    % Exogenous shocks
    [n_z_temp,~,pi_z_temp,simoptions_temp]=PType_setup_ExogShocks(ii,iistr,N_i,n_z,[],pi_z,simoptions_temp,3);

    % Parameters
    Parameters_temp=PType_setup_Parameters(ii,iistr,N_i,Parameters,3);

    if simoptions_temp.verboseparams==1
        sprintf('Parameter values for the current permanent type')
        Parameters_temp
    end

    if isstruct(AgentDist_initial)
        AgentDist_initial_temp=AgentDist_initial.(iistr);
    else
        AgentDist_initial_temp=AgentDist_initial; % NEED TO DEAL WITH THIS PROPERLY
    end
    if isstruct(AgeWeightsParamNames)
        AgeWeightsParamNames_temp=AgeWeightsParamNames.(iistr);
    else
        AgeWeightsParamNames_temp=AgeWeightsParamNames;
    end
    if isstruct(jequalOneDist)
        jequalOneDist_temp=jequalOneDist.(iistr);
    else
        jequalOneDist_temp=jequalOneDist;
    end


    % PricePath can include parameters that differ by ptype
    PricePath_temp=PricePath;
    PricePathNames=fieldnames(PricePath);
    for nn=1:length(PricePathNames)
        if isstruct(PricePath_temp.(PricePathNames{nn}))
            PricePath_temp.(PricePathNames{nn})=PricePath.(PricePathNames{nn}).(iistr);
        elseif any(size(PricePath_temp.(PricePathNames{nn}))==N_i)
            if size(PricePath_temp.(PricePathNames{nn}),1)==N_i
                temp=PricePath_temp.(PricePathNames{nn});
                PricePath_temp.(PricePathNames{nn})=temp(ii,:);
            elseif size(PricePath_temp.(PricePathNames{nn}),2)==N_i
                temp=PricePath_temp.(PricePathNames{nn});
                PricePath_temp.(PricePathNames{nn})=temp(:,ii);
            end
        end
    end

    % ParamPath can include parameters that differ by ptype
    ParamPath_temp=ParamPath;
    ParamPathNames=fieldnames(ParamPath);
    for nn=1:length(ParamPathNames)
        if isstruct(ParamPath_temp.(ParamPathNames{nn}))
            ParamPath_temp.(ParamPathNames{nn})=ParamPath.(ParamPathNames{nn}).(iistr);
        elseif any(size(ParamPath_temp.(ParamPathNames{nn}))==N_i)
            if size(ParamPath_temp.(ParamPathNames{nn}),1)==N_i
                temp=ParamPath_temp.(ParamPathNames{nn});
                ParamPath_temp.(ParamPathNames{nn})=temp(ii,:);
            elseif size(ParamPath_temp.(ParamPathNames{nn}),2)==N_i
                temp=ParamPath_temp.(ParamPathNames{nn});
                ParamPath_temp.(ParamPathNames{nn})=temp(:,ii);
            end
        end
    end


    % Compute the agent distribution path for permanent type ii
    AgentDistPath_ii=AgentDistOnTransPath_Case1_FHorz(AgentDist_initial_temp, jequalOneDist_temp, PricePath_temp, ParamPath_temp, PolicyPath_temp, AgeWeightsParamNames_temp,n_d_temp,n_a_temp,n_z_temp,N_j_temp,pi_z_temp, T,Parameters_temp, transpathoptions_temp, simoptions_temp);
    % Note: T cannot depend on ptype, nor can PricePath depend on ptype

    AgentDistPath.(iistr)=AgentDistPath_ii;

end


AgentDistPath.ptweights=AgentDist_initial.ptweights;


end