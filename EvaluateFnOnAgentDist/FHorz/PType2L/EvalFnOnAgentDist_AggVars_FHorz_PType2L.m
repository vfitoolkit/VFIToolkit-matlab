function AggVars=EvalFnOnAgentDist_AggVars_FHorz_PType2L(StationaryDist,Policy,FnsToEvaluate,Parameters,n_d,n_a,n_z,N_j,Names_i,N_i,d_grid,a_grid,z_grid,simoptions)
% Two-level permanent type dispatcher (top level) for FHorz AggVars.
%
% Top level is named (Names_i, cell array) with structure-keyed dependence.
% Bottom level is numeric (N_i) and is handled by
% EvalFnOnAgentDist_AggVars_FHorz_Case1_PType.
%
% Expects StationaryDist as produced by StationaryDist_FHorz_PType2L:
%   StationaryDist.(topname)     inner StationaryDist struct (with .ptweights)
%   StationaryDist.topptweights  length-N_topi vector
%
% Output (FnsToEvaluate must be a structure):
%   AggVars.(fn).(topname).(ptypeNNN).Mean  per-(top,bottom) mean
%   AggVars.(fn).(topname).Mean             grouped over bottom (always set)
%   AggVars.(fn).Mean                       grouped over top (simoptions.grouptopptypesforstats=1)

if ~iscell(Names_i)
    error('Names_i must be a cell array of top-level PType names for the two-level PType command.')
end
N_topi=length(Names_i);

if ~isstruct(FnsToEvaluate)
    error('You can only use PType2L when FnsToEvaluate is a structure')
end
FnNames=fieldnames(FnsToEvaluate);

if ~exist('simoptions','var')
    simoptions=struct();
end
if ~isfield(simoptions,'grouptopptypesforstats')
    simoptions.grouptopptypesforstats=1;
end
if ~isfield(simoptions,'groupptypesforstats')
    simoptions.groupptypesforstats=1;
end
if ~isfield(simoptions,'verbose')
    simoptions.verbose=0;
end
if ~isfield(simoptions,'verboseparams')
    simoptions.verboseparams=0;
end
if ~isfield(simoptions,'ptypestorecpu')
    simoptions.ptypestorecpu=0;
end

AggVars=struct();

for ii_top=1:N_topi
    iistr=Names_i{ii_top};

    % First set up simoptions
    simoptions_temp=PType_Options_2L(simoptions,iistr);
    if ~isfield(simoptions_temp,'groupptypesforstats')
        simoptions_temp.groupptypesforstats=1; % needed so inner returns .(fn).Mean grouped over bottom
    end
    if ~isfield(simoptions_temp,'verbose')
        simoptions_temp.verbose=0;
    end
    if ~isfield(simoptions_temp,'verboseparams')
        simoptions_temp.verboseparams=0;
    end
    if ~isfield(simoptions_temp,'ptypestorecpu')
        simoptions_temp.ptypestorecpu=0;
    end

    if simoptions_temp.verbose==1
        fprintf('Top-level permanent type: %i of %i (%s)\n',ii_top,N_topi,iistr)
    end

    % Go through everything which might be dependent on the top-level fixed
    % type. Struct-only at this level; anything else passes through.
    [n_d_temp,n_a_temp,d_grid_temp,a_grid_temp]=PType_setup_da(iistr,n_d,n_a,d_grid,a_grid);

    if isstruct(N_j)
        N_j_temp=N_j.(iistr);
    else
        N_j_temp=N_j;
    end
    if isstruct(N_i)
        N_i_temp=N_i.(iistr);
    else
        N_i_temp=N_i;
    end

    % Exogenous shocks
    [n_z_temp,z_grid_temp,~,simoptions_temp]=PType_setup_ExogShocks(ii_top,iistr,N_topi,n_z,z_grid,[],simoptions_temp,1);

    % Parameters are only allowed to depend on top-level PType through a structure keyed by Names_i.
    Parameters_temp=PType_setup_Parameters(ii_top,iistr,N_topi,Parameters,1);

    if simoptions_temp.verboseparams==1
        fprintf('Parameter values for the current top-level permanent type\n')
        Parameters_temp
    end

    AggVars_tt=EvalFnOnAgentDist_AggVars_FHorz_Case1_PType(StationaryDist.(iistr),Policy.(iistr),FnsToEvaluate,Parameters_temp,n_d_temp,n_a_temp,n_z_temp,N_j_temp,N_i_temp,d_grid_temp,a_grid_temp,z_grid_temp,simoptions_temp);

    for ff=1:length(FnNames)
        AggVars.(FnNames{ff}).(iistr)=AggVars_tt.(FnNames{ff});
    end

    clear AggVars_tt
end

%% Group across top types
if simoptions.grouptopptypesforstats==1 && isfield(StationaryDist,'topptweights')
    topptweights=StationaryDist.topptweights;
    for ff=1:length(FnNames)
        m=0;
        for ii_top=1:N_topi
            iistr=Names_i{ii_top};
            if isfield(AggVars.(FnNames{ff}).(iistr),'Mean')
                m=m+topptweights(ii_top)*AggVars.(FnNames{ff}).(iistr).Mean;
            end
        end
        AggVars.(FnNames{ff}).Mean=m;
    end
end


end
