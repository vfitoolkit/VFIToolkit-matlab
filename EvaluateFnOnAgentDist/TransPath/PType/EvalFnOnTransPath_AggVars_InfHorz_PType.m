function AggVarsPath=EvalFnOnTransPath_AggVars_InfHorz_PType(FnsToEvaluate, PricePath, ParamPath, Parameters, T, PolicyPath, AgentDistPath, n_d,n_a,n_z, Names_i, d_grid,a_grid,z_grid, simoptions)
% AggVars is simple in the sense we can just solve to get AggVars for each ptype and then take the weighted sum over them
% This only works because we are just after the mean


AggVarsPath=struct();

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

% Need to initialize the aggregates as zeros, these will then be updated adding on each ptype
FnNames=fieldnames(FnsToEvaluate);
for ff=1:length(FnNames)
    AggVarsPath.(FnNames{ff}).Mean=0;
end


%% Loop over permanent types
for ii=1:N_i
    iistr=Names_i{ii};

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
    AgentDistPath_temp=AgentDistPath.(iistr);


    %% Go through everything which might be dependent on fixed type (PType)
    [n_d_temp,n_a_temp,d_grid_temp,a_grid_temp]=PType_setup_da(iistr,n_d,n_a,d_grid,a_grid);

    % Exogenous shocks
    [n_z_temp,z_grid_temp,~,simoptions_temp]=PType_setup_ExogShocks(ii,iistr,N_i,n_z,z_grid,[],simoptions_temp,3);

    % Parameters
    Parameters_temp=PType_setup_Parameters(ii,iistr,N_i,Parameters,3);

    if simoptions_temp.verboseparams==1
        sprintf('Parameter values for the current permanent type')
        Parameters_temp
    end

    % ParamPath can include parameters that differ by ptype
    ParamPath_temp=ParamPath;
    ParamPathNames=fieldnames(ParamPath);
    for nn=1:length(ParamPathNames)
        if isstruct(ParamPath_temp.(ParamPathNames{nn}))
            ParamPath_temp.(ParamPathNames{nn})=ParamPath.(ParamPathNames{nn}).(iistr);
        end
    end

    l_d_temp=length(n_d_temp);
    if prod(n_d_temp)==0
        l_d_temp=0;
    end
    l_a_temp=length(n_a_temp);
    l_z_temp=length(n_z_temp);
    if prod(n_z_temp)==0
        l_z_temp=0;
    end
    [FnsToEvaluate_temp,~, ~,~]=PType_FnsToEvaluate(FnsToEvaluate,Names_i,ii,l_d_temp,l_a_temp,l_z_temp,0);

    AggVarsPath_ii=EvalFnOnTransPath_AggVars_InfHorz(FnsToEvaluate_temp, AgentDistPath_temp, PolicyPath_temp, PricePath, ParamPath_temp, Parameters_temp, T, n_d_temp, n_a_temp, n_z_temp, d_grid_temp, a_grid_temp,z_grid_temp, simoptions_temp);

    % Keep the ptype-conditional values
    AggVarsPath.(iistr)=AggVarsPath_ii;
    % And also create the actual aggregate values
    FnNames_temp=fieldnames(FnsToEvaluate_temp);
    for ff=1:length(FnNames_temp)
        AggVarsPath.(FnNames_temp{ff}).Mean=AggVarsPath.(FnNames_temp{ff}).Mean+AgentDistPath.ptweights(ii)*AggVarsPath_ii.(FnNames_temp{ff}).Mean;
    end
end


end
