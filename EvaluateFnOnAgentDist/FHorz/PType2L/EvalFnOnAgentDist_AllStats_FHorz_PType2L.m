function AllStats=EvalFnOnAgentDist_AllStats_FHorz_PType2L(StationaryDist,Policy,FnsToEvaluate,Parameters,n_d,n_a,n_z,N_j,Names_i,N_i,d_grid,a_grid,z_grid,simoptions)
% Two-level permanent type dispatcher (top level) for FHorz AllStats.
%
% Top level is named (Names_i, cell array) with structure-keyed dependence.
% Bottom level is numeric (N_i) and is handled by
% EvalFnOnAgentDist_AllStats_FHorz_Case1_PType.
%
% Expects StationaryDist as produced by StationaryDist_FHorz_PType2L:
%   StationaryDist.(topname)     inner StationaryDist struct (with .ptweights)
%   StationaryDist.topptweights  length-N_topi vector
%
% Output (FnsToEvaluate must be a structure):
%   AllStats.(fn).(topname)       full inner AllStats for that top (per-bottom + grouped-over-bottom)
%   AllStats.(fn).Mean            grouped over top (if simoptions.grouptopptypesforstats=1)
%   AllStats.(fn).Variance        idem (law of total variance)
%   AllStats.(fn).StdDeviation    idem
%   AllStats.(fn).Minimum         idem (min over top of per-top Min)
%   AllStats.(fn).Maximum         idem (max over top of per-top Max)

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

AllStats=struct();

for tt=1:N_topi
    iistr=Names_i{tt};

    % First set up simoptions
    simoptions_temp=PType_Options_2L(simoptions,Names_i,tt);
    if ~isfield(simoptions_temp,'groupptypesforstats')
        simoptions_temp.groupptypesforstats=1; % needed so inner returns grouped-over-bottom stats
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
        fprintf('Top-level permanent type: %i of %i (%s)\n',tt,N_topi,iistr)
    end

    % Go through everything which might be dependent on the top-level fixed type.
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
    [n_z_temp,z_grid_temp,~,simoptions_temp]=PType_setup_ExogShocks(tt,iistr,N_topi,n_z,z_grid,[],simoptions_temp,1);

    % Parameters are only allowed to depend on top-level PType through a structure keyed by Names_i.
    Parameters_temp=PType_setup_Parameters(tt,iistr,N_topi,Parameters,1);

    if simoptions_temp.verboseparams==1
        fprintf('Parameter values for the current top-level permanent type\n')
        Parameters_temp
    end

    AllStats_tt=EvalFnOnAgentDist_AllStats_FHorz_Case1_PType(StationaryDist.(iistr),Policy.(iistr),FnsToEvaluate,Parameters_temp,n_d_temp,n_a_temp,n_z_temp,N_j_temp,N_i_temp,d_grid_temp,a_grid_temp,z_grid_temp,simoptions_temp);

    for ff=1:length(FnNames)
        AllStats.(FnNames{ff}).(iistr)=AllStats_tt.(FnNames{ff});
    end

    clear AllStats_tt
end

%% Group across top types
if simoptions.grouptopptypesforstats==1 && isfield(StationaryDist,'topptweights')
    topptweights=StationaryDist.topptweights(:);
    for ff=1:length(FnNames)
        fn=FnNames{ff};
        % Mean (linear in distribution)
        if isfield(AllStats.(fn).(Names_i{1}),'Mean')
            M=zeros(N_topi,1);
            for tt=1:N_topi
                M(tt)=AllStats.(fn).(Names_i{tt}).Mean;
            end
            AllStats.(fn).Mean=sum(topptweights.*M);
            Mbar=AllStats.(fn).Mean;
        else
            Mbar=[];
        end
        % Variance / StdDeviation via law of total variance
        if ~isempty(Mbar) && isfield(AllStats.(fn).(Names_i{1}),'Variance')
            Vsum=0;
            for tt=1:N_topi
                Vtt=AllStats.(fn).(Names_i{tt}).Variance;
                Mtt=AllStats.(fn).(Names_i{tt}).Mean;
                Vsum=Vsum+topptweights(tt)*(Vtt+Mtt^2);
            end
            AllStats.(fn).Variance=Vsum-Mbar^2;
            AllStats.(fn).StdDeviation=sqrt(max(AllStats.(fn).Variance,0));
        end
        % Min / Max across top
        if isfield(AllStats.(fn).(Names_i{1}),'Minimum')
            mn=Inf; mx=-Inf;
            for tt=1:N_topi
                mn=min(mn,AllStats.(fn).(Names_i{tt}).Minimum);
                mx=max(mx,AllStats.(fn).(Names_i{tt}).Maximum);
            end
            AllStats.(fn).Minimum=mn;
            AllStats.(fn).Maximum=mx;
        end
    end
end


end
