function AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1_FieldExp_PType(StationaryDist, Policy, FnsToEvaluate, Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_grid, TreatmentParams, TreatmentAgeRange, TreatmentDuration, simoptions)
% Allows for different permanent (fixed) types of agent.
% See ValueFnIter_PType for general idea.
%
% simoptions.verbose=1 will give feedback
% simoptions.verboseparams=1 will give further feedback on the param values of each permanent type
%
% Rest of this description describes how those inputs not already used for
% ValueFnIter_PType or StationaryDist_PType should be set up.
%
% jequaloneDist can either be same for all permanent types, or must be passed as a structure.
% AgeWeightParamNames is either same for all permanent types, or must be passed as a structure.
%
% The stationary distribution be a structure and will contain both the
% weights/distribution across the permenant types, as well as a pdf for the
% stationary distribution of each specific permanent type.
%
% How exactly to handle these differences between permanent (fixed) types
% is to some extent left to the user. You can, for example, input
% parameters that differ by permanent type as a vector with different rows f
% for each type, or as a structure with different fields for each type.
%
% Any input that does not depend on the permanent type is just passed in
% exactly the same form as normal.

% Names_i can either be a cell containing the 'names' of the different
% permanent types, or if there are no structures used (just parameters that
% depend on permanent type and inputted as vectors or matrices as appropriate)
% then Names_i can just be the number of permanent types (but does not have to be, can still be names).
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


if isstruct(FnsToEvaluate)
    numFnsToEvaluate=length(fieldnames(FnsToEvaluate));
else
    numFnsToEvaluate=length(FnsToEvaluate);
end

% Set default of grouping all the PTypes together when reporting statistics
if ~exist('simoptions','var')
    simoptions.groupptypesforstats=1;
    simoptions.ptypestorecpu=1; % GPU memory is limited, so switch solutions to the cpu
    simoptions.verbose=0;
    simoptions.verboseparams=0;
else
    if ~isfield(simoptions,'groupptypesforstats')
        simoptions.groupptypesforstats=1;
    end
    if ~isfield(simoptions,'ptypestorecpu')
        if simoptions.groupptypesforstats==1
            simoptions.ptypestorecpu=1; % GPU memory is limited, so switch solutions to the cpu
        elseif simoptions.groupptypesforstats==0
            simoptions.ptypestorecpu=0;
        end
    end
    if ~isfield(simoptions,'verboseparams')
        simoptions.verboseparams=100;
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=100;
    end
end

if simoptions.groupptypesforstats==1 
    if isa(StationaryDist.control.(Names_i{1}), 'gpuArray')
        AggVars_control=zeros(numFnsToEvaluate,1,'gpuArray');
        AggVars_treatment=zeros(numFnsToEvaluate,1,'gpuArray');
    else
        AggVars_control=zeros(numFnsToEvaluate,1);
        AggVars_treatment=zeros(numFnsToEvaluate,1);
    end
else % simoptions.groupptypesforstats==0
    AggVars_control=struct();
    AggVars_treatment=struct();
end

%%
for ii=1:N_i
    
    % First set up simoptions
    simoptions_temp=PType_Options(simoptions,Names_i,ii); % Note: already check for existence of simoptions and created it if it was not inputted
    
    if simoptions_temp.verbose==1
        fprintf('Permanent type: %i of %i \n',ii, N_i)
    end    
    if simoptions_temp.ptypestorecpu==1 % Things are being stored on cpu but solved on gpu
        PolicyIndexes_control_temp=gpuArray(Policy.control.(Names_i{ii})); % Essentially just assuming vfoptions.ptypestorecpu=1 as well
        PolicyIndexes_treat_temp=struct();
        for j_p=TreatmentAgeRange(1):TreatmentAgeRange(2)
            PolicyIndexes_treat_temp.(['treatmentage',num2str(j_p)])=gpuArray(Policy.(['treatmentage',num2str(j_p)]).(Names_i{ii}));
        end
        StationaryDist_control_temp=gpuArray(StationaryDist.control.(Names_i{ii}));
        StationaryDist_treat_temp=gpuArray(StationaryDist.treatment.(Names_i{ii}));
    else
        PolicyIndexes_control_temp=Policy.control.(Names_i{ii});
        PolicyIndexes_treat_temp=struct();
        for j_p=TreatmentAgeRange(1):TreatmentAgeRange(2)
            PolicyIndexes_treat_temp.(['treatmentage',num2str(j_p)])=Policy.(['treatmentage',num2str(j_p)]).(Names_i{ii});
        end
        StationaryDist_control_temp=StationaryDist.control.(Names_i{ii});
        StationaryDist_treat_temp=StationaryDist.treatment.(Names_i{ii});
    end
    if isa(StationaryDist_control_temp, 'gpuArray')
        Parallel_temp=2;
    else
        Parallel_temp=1;
    end
    
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
    if isa(n_a,'struct')
        n_a_temp=n_a.(Names_i{ii});
    else
        n_a_temp=n_a;
    end
    if isa(n_z,'struct')
        n_z_temp=n_z.(Names_i{ii});
    else
        n_z_temp=n_z;
    end
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
    % (in terms of their dependence on permanent type). So go through each of
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

    % Note: TreatmentParams are not allowed to depend on permanent type, hence why we do not do the same for them.
    
    if simoptions_temp.verboseparams==1
        fprintf('Parameter values for the current permanent type \n')
        Parameters_temp
    end    
    
    % Figure out which functions are actually relevant to the present PType. Only the relevant ones need to be evaluated.
    % The dependence of FnsToEvaluate and FnsToEvaluateFnParamNames are necessarily the same.
    % Allows for FnsToEvaluate as structure.
    if n_d_temp(1)==0
        l_d_temp=0;
    else
        l_d_temp=1;
    end
    l_a_temp=length(n_a_temp);
    l_z_temp=length(n_z_temp);
    l_ze_temp=l_z_temp;
    if isfield(simoptions_temp,'n_e')
        if simoptions_temp.n_e(1)>0
            l_ze_temp=l_z_temp+length(simoptions.n_e);
        end
    end
    [FnsToEvaluate_temp,FnsToEvaluateParamNames_temp, WhichFnsForCurrentPType,~]=PType_FnsToEvaluate(FnsToEvaluate,Names_i,ii,l_d_temp,l_a_temp,l_ze_temp,0);
    
    %% Some setup to get just the relevant parts of the parameters and policy for the control group
    % Want to modify just the control parameters
    Parameters_control_temp=Parameters_temp;
    % For the age-dependent parameters, get the values that correspond to the ages that are in the field experiment
    allparamnames=fieldnames(Parameters_control_temp);
    for nn=1:length(allparamnames)
        if length(Parameters_control_temp.(allparamnames{nn}))==N_j_temp
            temp=Parameters_control_temp.(allparamnames{nn});
            Parameters_control_temp.(allparamnames{nn})=temp(TreatmentAgeRange(1):TreatmentAgeRange(2)+TreatmentDuration-1);
        end
    end    

    N_j_temp_FieldExp=TreatmentAgeRange(2)-TreatmentAgeRange(1)+TreatmentDuration;
    % Need to restrict the policy for control-group to just the relevant periods
    lengthPolicy=length(size(PolicyIndexes_control_temp));
    if lengthPolicy==2
        PolicyIndexes_control_temp=PolicyIndexes_control_temp(:,TreatmentAgeRange(1):TreatmentAgeRange(2)+TreatmentDuration-1);
    elseif lengthPolicy==3
        PolicyIndexes_control_temp=PolicyIndexes_control_temp(:,:,TreatmentAgeRange(1):TreatmentAgeRange(2)+TreatmentDuration-1);
    elseif lengthPolicy==4
        PolicyIndexes_control_temp=PolicyIndexes_control_temp(:,:,:,TreatmentAgeRange(1):TreatmentAgeRange(2)+TreatmentDuration-1);
    elseif lengthPolicy==5
        PolicyIndexes_control_temp=PolicyIndexes_control_temp(:,:,:,:,TreatmentAgeRange(1):TreatmentAgeRange(2)+TreatmentDuration-1);
    elseif lengthPolicy==6
        PolicyIndexes_control_temp=PolicyIndexes_control_temp(:,:,:,:,:,TreatmentAgeRange(1):TreatmentAgeRange(2)+TreatmentDuration-1);
    elseif lengthPolicy==7
        PolicyIndexes_control_temp=PolicyIndexes_control_temp(:,:,:,:,:,:,TreatmentAgeRange(1):TreatmentAgeRange(2)+TreatmentDuration-1);
    elseif lengthPolicy==8
        PolicyIndexes_control_temp=PolicyIndexes_control_temp(:,:,:,:,:,:,:,TreatmentAgeRange(1):TreatmentAgeRange(2)+TreatmentDuration-1);
    elseif lengthPolicy==9
        PolicyIndexes_control_temp=PolicyIndexes_control_temp(:,:,:,:,:,:,:,:,TreatmentAgeRange(1):TreatmentAgeRange(2)+TreatmentDuration-1);
    elseif lengthPolicy==10
        PolicyIndexes_control_temp=PolicyIndexes_control_temp(:,:,:,:,:,:,:,:,:,TreatmentAgeRange(1):TreatmentAgeRange(2)+TreatmentDuration-1);
    elseif lengthPolicy==11
        PolicyIndexes_control_temp=PolicyIndexes_control_temp(:,:,:,:,:,:,:,:,:,:,TreatmentAgeRange(1):TreatmentAgeRange(2)+TreatmentDuration-1);
    elseif lengthPolicy==12
        PolicyIndexes_control_temp=PolicyIndexes_control_temp(:,:,:,:,:,:,:,:,:,:,:,TreatmentAgeRange(1):TreatmentAgeRange(2)+TreatmentDuration-1);
    elseif lengthPolicy==13
        PolicyIndexes_control_temp=PolicyIndexes_control_temp(:,:,:,:,:,:,:,:,:,:,:,:,TreatmentAgeRange(1):TreatmentAgeRange(2)+TreatmentDuration-1);
    end
    
    %% Relevant parts of z_grid (and e_grid when relevant)
    if isfield(simoptions_temp,'ExogShockFn') % If using ExogShockFn then figure out the parameter names
        simoptions_temp.ExogShockFnParamNames=getAnonymousFnInputNames(simoptions_temp.ExogShockFn);
    end
    if isfield(simoptions_temp,'EiidShockFn') % If using ExogShockFn then figure out the parameter names
        simoptions_temp.EiidShockFnParamNames=getAnonymousFnInputNames(simoptions_temp.EiidShockFn);
    end
    
    if isfield(simoptions_temp,'z_grid_J')
        z_grid_J=simoptions_temp.z_grid_J;
    elseif isfield(simoptions_temp,'ExogShockFn')
        z_grid_J=zeros(sum(n_z_temp),N_j_temp);
        for jj=1:N_j_temp
            if isfield(simoptions_temp,'ExogShockFnParamNames')
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters_temp, simoptions_temp.ExogShockFnParamNames,jj);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for xx=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(xx,1)={ExogShockFnParamsVec(xx)};
                end
                [z_grid,~]=simoptions_temp.ExogShockFn(ExogShockFnParamsCell{:});
            else
                [z_grid,~]=simoptions_temp.ExogShockFn(jj);
            end
            z_grid_J(:,jj)=gather(z_grid);
        end
        simoptions_temp=rmfield(simoptions_temp,'ExogShockFn');
    else
        % This is just so that it makes things easier below because I can
        % take it as given that pi_z_J and z_grid_J exist
        z_grid_J=zeros(sum(n_z_temp),N_j_temp);
        for jj=1:N_j_temp
            z_grid_J(:,jj)=z_grid;
        end
    end
    % And same for e if that is used
    if isfield(simoptions_temp,'n_e')
        n_e_temp=simoptions_temp.n_e;
        N_e_temp=prod(n_e_temp);
    else
        N_e_temp=0;
    end
    if N_e_temp>0
        if isfield(simoptions_temp,'e_grid_J')
            e_grid_J=simoptions_temp.e_grid_J;
        elseif isfield(simoptions_temp,'EiidShockFn')
            e_grid_J=zeros(sum(n_e_temp),N_j_temp);
            for jj=1:N_j_temp
                if isfield(simoptions_temp,'EiidShockFnParamNames')
                    EiidShockFnParamsVec=CreateVectorFromParams(Parameters_temp, simoptions_temp.EiidShockFnParamNames,jj);
                    EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                    for xx=1:length(EiidShockFnParamsVec)
                        EiidShockFnParamsCell(xx,1)={EiidShockFnParamsVec(xx)};
                    end
                    [e_grid,~]=simoptions_temp.EiidShockFn(EiidShockFnParamsCell{:});
                else
                    [e_grid,~]=simoptions_temp.EiidShockFn(jj);
                end
                e_grid_J(:,jj)=gather(e_grid);
            end
            simoptions_temp=rmfield(simoptions_temp,'EiidShockFn');
        else
            % This is just so that it makes things easier below because I can
            % take it as given that pi_e_J and e_grid_J exist
            e_grid_J=zeros(sum(n_e_temp),N_j_temp);
            for jj=1:N_j_temp
                e_grid_J(:,jj)=simoptions_temp.e_grid;
            end
        end
    end
    
    simoptions_control_temp=simoptions_temp;
    simoptions_control_temp.z_grid_J=z_grid_J(:,TreatmentAgeRange(1):TreatmentAgeRange(2)+TreatmentDuration-1);
    if isfield(simoptions_temp,'n_e')
        simoptions_control_temp.e_grid=e_grid_J(:,TreatmentAgeRange(1):TreatmentAgeRange(2)+TreatmentDuration-1);
    end
    
    
    %% Get the age weights out of control group as we need the later for treatment group, then reweight control group ages to reflect treatment duration

    % Find the age weights for the treatment group (get them from the marginal distribution of the control group over ages)
    if isfield(simoptions_temp,'n_e')
        N_e_temp=prod(simoptions_temp.n_e);
        N_ze_temp=N_e_temp*prod(n_z_temp);
    else
        N_ze_temp=prod(n_z_temp);
    end
    if N_ze_temp==0
        TreatmentAgeWeights=sum(reshape(StationaryDist_control_temp,[prod(n_a_temp),N_j_temp_FieldExp],1));
    else
        TreatmentAgeWeights=shiftdim(sum(sum(reshape(StationaryDist_control_temp,[prod(n_a_temp),N_ze_temp,N_j_temp_FieldExp]),1),2),1);
    end
    % TreatmentAgeWeights is a row vector of just the weights for the initial ages
    TreatmentAgeWeights=TreatmentAgeWeights(1:end-TreatmentDuration+1);
    % Normalize to one
    TreatmentAgeWeights=TreatmentAgeWeights/sum(TreatmentAgeWeights);

    % We need to slightly reweight the control group ages to account for
    % the fact that they start at ages TreatmentAgeRange, and then are
    % there for TreatmentDuration.
    agereweight=[TreatmentAgeWeights'; zeros(TreatmentDuration-1,1)]; % First period of treatment
    for tt=2:TreatmentDuration % Other periods of treatment
        agereweight=agereweight+[zeros(tt-1,1); TreatmentAgeWeights'; zeros(TreatmentDuration-tt,1)];
    end
    agereweight=agereweight/sum(agereweight);
    StationaryDist_control_temp=reshape(StationaryDist_control_temp,[numel(StationaryDist_control_temp)/N_j_temp_FieldExp,N_j_temp_FieldExp]);
    StationaryDist_control_temp=StationaryDist_control_temp.*agereweight';
    StationaryDist_control_temp=StationaryDist_control_temp/sum(StationaryDist_control_temp(:));
    
    %% Calculate the aggregate variables for the control-group
    simoptions_control_temp.outputasstructure=0;
    if isfinite(N_j_temp)
        StatsFromDist_AggVars_Control_ii=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist_control_temp, PolicyIndexes_control_temp, FnsToEvaluate_temp, Parameters_control_temp, FnsToEvaluateParamNames_temp, n_d_temp, n_a_temp, n_z_temp, N_j_temp_FieldExp, d_grid_temp, a_grid_temp, z_grid_temp, Parallel_temp,simoptions_control_temp); % Note: simoptions_control_temp is only different from simoptions_temp in having z_grid_J and e_grid_J
    else % PType actually allows for infinite horizon as well
        error('Field experiments do not allow for infinite horizon problems')
    end
    
    
    %% Now deal with the treatment group
    % Modify the parameters to those for the treatment
    treatparamnames=fieldnames(TreatmentParams);
    for nn=1:length(treatparamnames)
        Parameters_temp.(treatparamnames{nn})=TreatmentParams.(treatparamnames{nn});
    end
    
    % Calculate the aggregate variables for the treatment-group
    simoptions_temp.outputasstructure=0;
    if isfinite(N_j_temp)
        StatsFromDist_AggVars_Treatment_ii=EvalFnOnAgentDist_AggVars_FHorz_Case1_FieldExp_Treatment(StationaryDist_treat_temp, PolicyIndexes_treat_temp, FnsToEvaluate_temp, Parameters_temp, FnsToEvaluateParamNames_temp, n_d_temp, n_a_temp, n_z_temp, N_j_temp, d_grid_temp, a_grid_temp, z_grid_temp, Parallel_temp, TreatmentAgeRange, TreatmentDuration, TreatmentAgeWeights, simoptions_temp); % Note: the input parameters should be those for the treatment group
    else % PType actually allows for infinite horizon as well
        error('Field experiments do not allow for infinite horizon problems')
    end
    
    
    %% Add things up across PTypes
    if simoptions.groupptypesforstats==1
        for kk=1:numFnsToEvaluate
            jj=WhichFnsForCurrentPType(kk);
            if jj>0
                AggVars_control(kk)=AggVars_control(kk)+StationaryDist.ptweights(ii)*StatsFromDist_AggVars_Control_ii(jj,:);
                AggVars_treatment(kk)=AggVars_treatment(kk)+StationaryDist.ptweights(ii)*StatsFromDist_AggVars_Treatment_ii(jj,:);
            end
        end
    else
        for kk=1:numFnsToEvaluate
            jj=WhichFnsForCurrentPType(kk);
            if jj>0
                AggVars_control(kk).(Names_i{ii})=StationaryDist.ptweights(ii)*StatsFromDist_AggVars_Control_ii(jj,:);
                AggVars_treatment(kk).(Names_i{ii})=StationaryDist.ptweights(ii)*StatsFromDist_AggVars_Treatment_ii(jj,:);
            end
        end
    end
end

% If using FnsToEvaluate as structure need to get in appropriate form for output
if isstruct(FnsToEvaluate)
    AggVarNames=fieldnames(FnsToEvaluate);
    % Change the output into a structure
    AggVars=struct();
    %     AggVarNames=fieldnames(FnsToEvaluate);
    if simoptions.groupptypesforstats==1
        for ff=1:length(AggVarNames)
            AggVars.(AggVarNames{ff}).control.Mean=AggVars_control(ff);
            AggVars.(AggVarNames{ff}).treatment.Mean=AggVars_treatment(ff);
        end
    else % simoptions.groupptypesforstats==0
        for ff=1:length(AggVarNames)
            for ii=1:N_i
                AggVars.(AggVarNames{ff}).control.(Names_i{ii}).Mean=AggVars_control(ff).(Names_i{ii});
                AggVars.(AggVarNames{ff}).treatment.(Names_i{ii}).Mean=AggVars_treatment(ff).(Names_i{ii});
            end
        end
    end
end


end
