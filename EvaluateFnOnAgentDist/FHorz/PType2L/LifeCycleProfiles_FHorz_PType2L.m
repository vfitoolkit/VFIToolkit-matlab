function AgeConditionalStats=LifeCycleProfiles_FHorz_PType2L(StationaryDist,Policy,FnsToEvaluate,Parameters,n_d,n_a,n_z,N_j,Names_i,N_i,d_grid,a_grid,z_grid,simoptions)
% Two-level permanent type dispatcher (top level) for FHorz life-cycle profiles.
%
% Top level is named (Names_i, cell array) with structure-keyed dependence.
% Bottom level is numeric (N_i) and is handled by
% LifeCycleProfiles_FHorz_Case1_PType.
%
% Expects StationaryDist as produced by StationaryDist_FHorz_PType2L:
%   StationaryDist.(topname)     inner StationaryDist struct (with .ptweights)
%   StationaryDist.topptweights  length-N_topi vector
%
% Output (FnsToEvaluate must be a structure):
%   AgeConditionalStats.(fn).(topname)  full inner profile struct
%   AgeConditionalStats.(fn).Mean       grouped over top, vector over age groupings
%   AgeConditionalStats.(fn).Variance   idem (law of total variance, per age)
%   AgeConditionalStats.(fn).StdDeviation
%   AgeConditionalStats.(fn).Minimum    min over top per age
%   AgeConditionalStats.(fn).Maximum    max over top per age

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

AgeConditionalStats=struct();

for tt=1:N_topi
    iistr=Names_i{tt};

    % First set up simoptions
    simoptions_temp=PType_Options_2L(simoptions,Names_i,tt);
    if ~isfield(simoptions_temp,'groupptypesforstats')
        simoptions_temp.groupptypesforstats=1; % needed so inner returns grouped-over-bottom profiles
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

    Profiles_tt=LifeCycleProfiles_FHorz_Case1_PType(StationaryDist.(iistr),Policy.(iistr),FnsToEvaluate,Parameters_temp,n_d_temp,n_a_temp,n_z_temp,N_j_temp,N_i_temp,d_grid_temp,a_grid_temp,z_grid_temp,simoptions_temp);

    for ff=1:length(FnNames)
        AgeConditionalStats.(FnNames{ff}).(iistr)=Profiles_tt.(FnNames{ff});
    end

    clear Profiles_tt
end

%% Group across top types (per age)
if simoptions.grouptopptypesforstats==1 && isfield(StationaryDist,'topptweights')
    topptweights=StationaryDist.topptweights(:);
    for ff=1:length(FnNames)
        fn=FnNames{ff};
        % Mean
        if isfield(AgeConditionalStats.(fn).(Names_i{1}),'Mean')
            Mref=AgeConditionalStats.(fn).(Names_i{1}).Mean;
            Mbar=zeros(size(Mref),'like',Mref);
            for tt=1:N_topi
                Mbar=Mbar+topptweights(tt)*AgeConditionalStats.(fn).(Names_i{tt}).Mean;
            end
            AgeConditionalStats.(fn).Mean=Mbar;
        else
            Mbar=[];
        end
        % Variance / StdDeviation via law of total variance, per age
        if ~isempty(Mbar) && isfield(AgeConditionalStats.(fn).(Names_i{1}),'Variance')
            Vsum=zeros(size(Mbar),'like',Mbar);
            for tt=1:N_topi
                Vtt=AgeConditionalStats.(fn).(Names_i{tt}).Variance;
                Mtt=AgeConditionalStats.(fn).(Names_i{tt}).Mean;
                Vsum=Vsum+topptweights(tt)*(Vtt+Mtt.^2);
            end
            V=Vsum-Mbar.^2;
            AgeConditionalStats.(fn).Variance=V;
            AgeConditionalStats.(fn).StdDeviation=sqrt(max(V,0));
        end
        % Min / Max across top, per age
        if isfield(AgeConditionalStats.(fn).(Names_i{1}),'Minimum')
            mnref=AgeConditionalStats.(fn).(Names_i{1}).Minimum;
            mxref=AgeConditionalStats.(fn).(Names_i{1}).Maximum;
            mn=Inf(size(mnref),'like',mnref);
            mx=-Inf(size(mxref),'like',mxref);
            for tt=1:N_topi
                mn=min(mn,AgeConditionalStats.(fn).(Names_i{tt}).Minimum);
                mx=max(mx,AgeConditionalStats.(fn).(Names_i{tt}).Maximum);
            end
            AgeConditionalStats.(fn).Minimum=mn;
            AgeConditionalStats.(fn).Maximum=mx;
        end
    end
end


end
