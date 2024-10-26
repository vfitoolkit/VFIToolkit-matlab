function AggVarsPath=EvalFnOnTransPath_AggVars_Case1_FHorz_PType(FnsToEvaluate, AgentDistPath, PolicyPath, PricePath, ParamPath, Parameters, T, n_d, n_a, n_z, N_j, Names_i, d_grid, a_grid,z_grid, transpathoptions, simoptions)
% AggVars is simple in the sense we can just solve to get AggVars for each ptype and then take the weigthed sum over them
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
    AggVarsPath.(FnNames{ff}).Mean=zeros(1,T);
end




%% Loop over permanent types
for ii=1:N_i

    % First set up transpathoptions
    if exist('transpathoptions','var')
        transpathoptions_temp=PType_Options(simoptions,Names_i,ii);
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
        simoptions_temp=PType_Options(simoptions,Names_i,ii);
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
           
    PolicyPath_temp=PolicyPath.(Names_i{ii});
    AgentDistPath_temp=AgentDistPath.(Names_i{ii});

    
    % Go through everything which might be dependent on permanent type (PType)
    % Notice that the way this is coded the grids (etc.) could be either
    % fixed, or a function (that depends on age, and possibly on permanent
    % type), or they could be a structure. Only in the case where they are
    % a structure is there a need to take just a specific part and send
    % only that to the 'non-PType' version of the command.
    if isa(n_d,'struct')
        n_d_temp=n_d.(Names_i{ii});
    else
        n_d_temp=n_d;
    end
    l_d_temp=length(n_d_temp);
    if isa(n_a,'struct')
        n_a_temp=n_a.(Names_i{ii});
    else
        n_a_temp=n_a;
    end
    l_a_temp=length(n_a_temp);
    if isa(n_z,'struct')
        n_z_temp=n_z.(Names_i{ii});
    else
        n_z_temp=n_z;
    end
    l_z_temp=length(n_z_temp);
    if isa(N_j,'struct')
        N_j_temp=N_j.(Names_i{ii});
    else
        N_j_temp=N_j;
    end
    if isa(d_grid,'struct')
        d_grid_temp=d_grid.(Names_i{ii});
    else
        d_grid_temp=d_grid;
    end
    if isa(a_grid,'struct')
        a_grid_temp=a_grid.(Names_i{ii});
    else
        a_grid_temp=a_grid;
    end
    if isa(z_grid,'struct')
        z_grid_temp=z_grid.(Names_i{ii});
    else
        z_grid_temp=z_grid;
    end
    
    % Parameters are allowed to be given as structure, or as vector/matrix
    % (in terms of their dependence on fixed type). So go through each of
    % these in term.
    Parameters_temp=Parameters;
    FullParamNames=fieldnames(Parameters);
    nFields=length(FullParamNames);
    for kField=1:nFields
        if isa(Parameters.(FullParamNames{kField}), 'struct') % Check for permanent type in structure form
            names=fieldnames(Parameters.(FullParamNames{kField}));
            for jj=1:length(names)
                if strcmp(names{jj},Names_i{ii})
                    Parameters_temp.(FullParamNames{kField})=Parameters.(FullParamNames{kField}).(names{jj});
                end
            end
        elseif any(size(Parameters.(FullParamNames{kField}))==N_i) % Check for permanent type in vector/matrix form.
            temp=Parameters.(FullParamNames{kField});
            [~,ptypedim]=max(size(Parameters.(FullParamNames{kField}))==N_i); % Parameters as vector/matrix can be at most two dimensional, figure out which relates to PType.
            if ptypedim==1
                Parameters_temp.(FullParamNames{kField})=temp(ii,:);
            elseif ptypedim==2
                Parameters_temp.(FullParamNames{kField})=temp(:,ii);
            end
        end
    end
    
    if simoptions_temp.verboseparams==1
        sprintf('Parameter values for the current permanent type')
        Parameters_temp
    end
    
    % ParamPath can include parameters that differ by ptype
    ParamPath_temp=ParamPath;
    ParamPathNames=fieldnames(ParamPath);
    for nn=1:length(ParamPathNames)
        if isstruct(ParamPath_temp.(ParamPathNames{nn}))
            ParamPath_temp.(ParamPathNames{nn})=ParamPath.(ParamPathNames{nn}).(Names_i{ii});
        end
    end

    [FnsToEvaluate_temp,~, ~,~]=PType_FnsToEvaluate(FnsToEvaluate,Names_i,ii,l_d_temp,l_a_temp,l_z_temp,0);

    AggVarsPath_ii=EvalFnOnTransPath_AggVars_Case1_FHorz(FnsToEvaluate_temp, AgentDistPath_temp, PolicyPath_temp, PricePath, ParamPath_temp, Parameters_temp, T, n_d_temp, n_a_temp, n_z_temp, N_j_temp, d_grid_temp, a_grid_temp,z_grid_temp, transpathoptions_temp, simoptions_temp);

    % Keep the ptype-conditional values
    AggVarsPath.(Names_i{ii})=AggVarsPath_ii;
    % And also create the actual aggregate values
    FnNames_temp=fieldnames(FnsToEvaluate_temp);
    for ff=1:length(FnNames_temp)
        AggVarsPath.(FnNames_temp{ff}).Mean=AggVarsPath.(FnNames_temp{ff}).Mean+AgentDistPath.ptweights(ii)*AggVarsPath_ii.(FnNames_temp{ff}).Mean;
    end
end


end