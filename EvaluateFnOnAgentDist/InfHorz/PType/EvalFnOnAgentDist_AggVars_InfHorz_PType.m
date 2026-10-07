function AggVars=EvalFnOnAgentDist_AggVars_InfHorz_PType(StationaryDist, Policy, FnsToEvaluate, Parameters,n_d,n_a,n_z,Names_i,d_grid, a_grid, z_grid, simoptions)
% Allows for different permanent (fixed) types of agent.
% See ValueFnIter_InfHorz_PType for general idea.
%
% simoptions.verbose=1 will give feedback
% simoptions.verboseparams=1 will give further feedback on the param values of each permanent type
% simoptions.whichcombos (a vector of zeros/ones, one per FnsToEvaluate) selects which functions are evaluated; see below.
%
% Rest of this description describes how those inputs not already used for
% ValueFnIter_PType or StationaryDist_PType should be set up.
%
% jequaloneDist can either be same for all permanent types, or must be passed as a structure.
% AgeWeightParamNames is either same for all permanent types, or must be passed as a structure.
%
% The stationary distribution be a structure and will contain both the
% weights/distribution across the permanent types, as well as a pdf for the
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

% Set default of grouping all the PTypes together when reporting statistics. The per-type values are
% always returned as well (they are free - each one is computed anyway), so this option only controls
% whether the ptype-weighted aggregate is added alongside them. Same as the FHorz PType version.
if ~isfield(simoptions,'groupptypesforstats')
    simoptions.groupptypesforstats=1;
end
%% simoptions.whichcombos: which FnsToEvaluate to compute
% A vector of zeros/ones of length numFnsToEvaluate (row or column). Ones are evaluated, zeros skipped: their Mean (per type and
% grouped) is NaN. Default all ones. Intended for calibration/estimation, which only needs the targeted aggregates.
% (As EvalFnOnAgentDist_AggVars_FHorz_Case1_PType.)
if ~isfield(simoptions,'whichcombos')
    whichcombos=ones(numFnsToEvaluate,1);
else
    whichcombos=simoptions.whichcombos;
    if ~(isnumeric(whichcombos) || islogical(whichcombos)) || any(whichcombos(:)~=0 & whichcombos(:)~=1)
        error('simoptions.whichcombos must contain only zeros and ones')
    end
    if ~(isvector(whichcombos) && numel(whichcombos)==numFnsToEvaluate)
        error(['simoptions.whichcombos must be a vector of length ',num2str(numFnsToEvaluate),' (number of FnsToEvaluate)'])
    end
    whichcombos=double(whichcombos(:));
end

% One column per permanent type, so that each type's own values survive to the output; they are only
% weighted and summed at the end
AggVarsFull=zeros(numFnsToEvaluate,N_i,'gpuArray');

%%
for ii=1:N_i
    iistr=Names_i{ii};

    % First set up simoptions
    simoptions_temp=PType_Options(simoptions,iistr); % Note: already check for existence of simoptions and created it if it was not inputted
    if ~isfield(simoptions_temp,'verboseparams')
        simoptions_temp.verboseparams=0;
    end
    if ~isfield(simoptions_temp,'verbose')
        simoptions_temp.verbose=0;
    end

    if simoptions_temp.verbose>=1
        fprintf('Permanent type: %i of %i \n',ii, N_i)
    end

    PolicyIndexes_temp=gpuArray(Policy.(iistr)); % Just in case using vfoptions.ptypestorecpu=1
    StationaryDist_temp=gpuArray(StationaryDist.(iistr));

    %% Go through everything which might be dependent on fixed type (PType)
    [n_d_temp,n_a_temp,d_grid_temp,a_grid_temp]=PType_setup_da(iistr,n_d,n_a,d_grid,a_grid);

    % Exogenous shocks
    [n_z_temp,z_grid_temp,~,simoptions_temp]=PType_setup_ExogShocks(ii,iistr,N_i,n_z,z_grid,[],simoptions_temp,3);

    % Parameters
    Parameters_temp=PType_setup_Parameters(ii,iistr,N_i,Parameters,3);

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
    [FnsToEvaluate_temp,FnsToEvaluateParamNames_temp, WhichFnsForCurrentPType,~]=PType_FnsToEvaluate(FnsToEvaluate,Names_i,ii,l_d_temp,l_a_temp,l_z_temp,0);

    if ~any(whichcombos(WhichFnsForCurrentPType>0)) % none of the functions relevant to this type are wanted
        continue
    end
    if isfield(simoptions,'whichcombos')
        simoptions_temp.whichcombos=whichcombos(WhichFnsForCurrentPType>0); % the selection among the functions relevant to this type (same order)
    end

    simoptions_temp.outputasstructure=0;
    StatsFromDist_AggVars_ii=EvalFnOnAgentDist_AggVars_InfHorz(StationaryDist_temp, PolicyIndexes_temp, FnsToEvaluate_temp, Parameters_temp, FnsToEvaluateParamNames_temp, n_d_temp, n_a_temp, n_z_temp, d_grid_temp, a_grid_temp, z_grid_temp, simoptions_temp); % , EntryExitParamNames, PolicyWhenExiting

    for ff=1:numFnsToEvaluate
        jj=WhichFnsForCurrentPType(ff);
        if jj>0
            AggVarsFull(ff,ii)=StatsFromDist_AggVars_ii(jj,:); % weighting and summing happens after the loop
        end
    end
end

AggVarsFull(whichcombos==0,:)=NaN; % the functions that whichcombos skipped
AggVars2=sum(StationaryDist.ptweights'.*AggVarsFull,2); % sum across agents (ptweights stored as column)


%% If using FnsToEvaluate as structure need to get in appropriate form for output
if isstruct(FnsToEvaluate)
    AggVarNames=fieldnames(FnsToEvaluate);
    % Change the output into a structure
    AggVars=struct();
    for ff=1:length(AggVarNames)
        for ii=1:N_i
            AggVars.(AggVarNames{ff}).(Names_i{ii}).Mean=AggVarsFull(ff,ii);
        end
    end
    if simoptions.groupptypesforstats==1
        for ff=1:length(AggVarNames)
            AggVars.(AggVarNames{ff}).Mean=AggVars2(ff);
        end
    end
elseif simoptions.groupptypesforstats==1
    AggVars=AggVars2;
end


end
