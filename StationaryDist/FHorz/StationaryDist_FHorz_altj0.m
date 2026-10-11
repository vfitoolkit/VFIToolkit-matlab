function StationaryDist=StationaryDist_FHorz_altj0(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions)
% Agent distribution when the initial distribution is for an age j0>1, and/or the distribution is only wanted up
% to an age jlast<N_j.
%
% simoptions.jequaloneDistAge=j0 means jequaloneDist is the distribution of agents at age j0 (rather than
% j=1), there is no mass at ages 1,...,j0-1, and the distribution is iterated forward from j0 with the same
% Policy. Used for cohorts that enter the model at different ages (e.g., an initial distribution taken from
% data at an age other than the first model period).
% simoptions.lastDistAge=jlast means the distribution is iterated only up to age jlast, and ages jlast+1,...,N_j
% get zero mass. Used by the withCohorts estimation to stop each cohort at its last targeted age.
%
% Implemented by slicing everything age-dependent to ages j0:jlast (same approach as
% StationaryDist_FHorz_FieldExp_Treatment), calling StationaryDist_FHorz_Case1 on the shorter problem,
% and zero-padding ages 1,...,j0-1 and jlast+1,...,N_j. The age weights for ages j0:jlast are kept, so the
% output has total mass sum(AgeWeights(j0:jlast)) rather than one.
%
% Called by StationaryDist_FHorz_Case1 after the exogenous shocks have been set up, so:
%   pi_z_J is [N_z,N_z,N_j-1] (or N_j with V_Jplus1), or [] if no z
%   simoptions.z_gridvals_J exists if z_gridvals_J was created (experienceassetz/ze, residualasset)
%   simoptions.pi_e_J, simoptions.e_gridvals_J exist if there is an e
%   simoptions.pi_semiz_J, simoptions.semiz_gridvals_J exist if there is a semiz
%   Parameters.(AgeWeightParamNames{1}) is a row vector of length N_j
%   jequaloneDist has already been evaluated (if it was a function) and checked to be of mass one

j0=simoptions.jequaloneDistAge;
jlast=simoptions.lastDistAge;
if j0>N_j
    error('simoptions.jequaloneDistAge cannot be larger than N_j')
end
if jlast<j0 || jlast>N_j
    error('simoptions.lastDistAge must be between simoptions.jequaloneDistAge and N_j')
end
N_j_full=N_j;
N_j=jlast-j0+1;
ndrop=N_j_full-jlast; % number of trailing ages dropped (the age-dependent objects may have N_j_full or N_j_full-1 slices; j0:end-ndrop keeps whichever it is)

% Policy: age is the last dimension
Policyindex=repmat({':'},1,ndims(Policy));
Policyindex{end}=j0:jlast;
Policy=Policy(Policyindex{:});

% Age-dependent parameters: any parameter of length N_j (the rule used by StationaryDist_FHorz_FieldExp_Treatment)
paramnames=fieldnames(Parameters);
for nn=1:length(paramnames)
    if length(Parameters.(paramnames{nn}))==N_j_full
        temp=Parameters.(paramnames{nn});
        Parameters.(paramnames{nn})=temp(j0:jlast);
    end
end

% Age weights: use uniform weights for the shorter problem, and put the actual weights back afterwards
AgeWeights_j0=Parameters.(AgeWeightParamNames{1}); % row vector, already sliced to ages j0:jlast by the loop just above
Parameters.(AgeWeightParamNames{1})=ones(1,N_j)/N_j;

% Exogenous shocks: already in age-dependent joint-grid form with age as the last dimension
if ~isempty(pi_z_J)
    pi_z_J=pi_z_J(:,:,j0:end-ndrop); % last dim is N_j_full-1 (or N_j_full when using V_Jplus1); j0:end-ndrop keeps whichever it is
end
if isfield(simoptions,'z_gridvals_J')
    simoptions.z_grid=simoptions.z_gridvals_J(:,:,j0:end-ndrop); % with alreadygridvals=1, StationaryDist_FHorz_Case1 reads gridvals from simoptions.z_grid
    simoptions=rmfield(simoptions,'z_gridvals_J');
end
if isfield(simoptions,'pi_e_J')
    simoptions.pi_e_J=simoptions.pi_e_J(:,j0:end-ndrop);
end
if isfield(simoptions,'e_gridvals_J')
    simoptions.e_gridvals_J=simoptions.e_gridvals_J(:,:,j0:end-ndrop);
end
if isfield(simoptions,'pi_semiz_J')
    simoptions.pi_semiz_J=simoptions.pi_semiz_J(:,:,:,j0:end-ndrop);
end
if isfield(simoptions,'semiz_gridvals_J')
    simoptions.semiz_gridvals_J=simoptions.semiz_gridvals_J(:,:,j0:end-ndrop);
end

% Solve the shorter problem (the shocks are already set up, so the call must not redo them)
simoptions.jequaloneDistAge=1;
simoptions.lastDistAge=N_j; % (the shorter problem runs to its own last age)
simoptions.alreadygridvals=1;
simoptions.alreadygridvals_semiexo=1;
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);

% Put the actual age weights back, and zero-pad ages 1,...,j0-1 and jlast+1,...,N_j_full (age is the last dimension of the output)
fullsize=size(StationaryDist);
if N_j>1
    fullsize=fullsize(1:end-1); % drop the age dimension
end % (when N_j==1 the age dimension is a trailing singleton that size() does not report, so fullsize is already the state dimensions)
M=numel(StationaryDist)/N_j;
StationaryDist=reshape(StationaryDist,[M,N_j]).*(AgeWeights_j0*N_j); % uniform weights 1/N_j -> AgeWeights_j0
StationaryDist=reshape([zeros(M,j0-1,'like',StationaryDist),StationaryDist,zeros(M,ndrop,'like',StationaryDist)],[fullsize,N_j_full]);

end
