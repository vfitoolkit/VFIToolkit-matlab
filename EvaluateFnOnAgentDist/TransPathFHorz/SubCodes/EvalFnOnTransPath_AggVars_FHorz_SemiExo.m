function AggVarsPath=EvalFnOnTransPath_AggVars_FHorz_SemiExo(FnsToEvaluate, AgentDistPath, PolicyPath, PricePath, ParamPath, Parameters, T, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, transpathoptions, simoptions)
% AggVars along the transition path, with a semi-exogenous state.
% The semi-exogenous state is just part of the joint exogenous state for FnsToEvaluate (no transition machinery is needed here);
% EvalFnOnAgentDist_AggVars_FHorz_Case1 folds (semiz,z,e) into the joint exogenous grid via CreateGridvals_FnsToEvaluate_FHorz.
% Note: called from EvalFnOnTransPath_AggVars_Case1_FHorz before its (non-semiz) reshapes; this command does its own reshaping.

n_semiz=simoptions.n_semiz;
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_e=prod(simoptions.n_e);

if ~isscalar(n_a)
    error('Transition paths with semi-exogenous states only allow a single endogenous state (cannot have length(n_a)>1)')
end

% bothze composite (semiz,z,e), semiz fastest
if N_z==0
    if N_e==0
        N_bothze=N_semiz;
    else
        N_bothze=N_semiz*N_e;
    end
else
    if N_e==0
        N_bothze=N_semiz*N_z;
    else
        N_bothze=N_semiz*N_z*N_e;
    end
end

% EvalFnOnTransPath_AggVars uses the fastOLG=0 form of z_gridvals_J and e_gridvals_J
transpathoptions.fastOLG=0;

%% Internally PricePath is matrix of size T-by-'number of prices'.
[PricePath,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec]=PricePathParamPath_FHorz_StructToMatrix(PricePath,ParamPath,N_j,T);

%% Set up exogenous shock processes (gridpiboth=1: only need z_gridvals_J and e_gridvals_J)
if ~isfield(simoptions,'alreadygridvals')
    simoptions.alreadygridvals=0;
end
if simoptions.alreadygridvals==0
    [z_gridvals_J, ~, ~, e_gridvals_J, ~, ~, ~, transpathoptions, simoptions]=ExogShockSetup_FHorz_TPath(n_z,z_grid,[],N_a,N_j,Parameters,PricePathNames,ParamPathNames,transpathoptions,simoptions,1);
else
    z_gridvals_J=z_grid;
    e_gridvals_J=simoptions.e_gridvals_J;
end

if N_z>0 && transpathoptions.zpathtrivial==0
    error('Semi-exogenous states with z varying over the transition path are not yet implemented for EvalFnOnTransPath_AggVars (email me if you want this)')
end
if N_e>0 && transpathoptions.epathtrivial==0
    error('Semi-exogenous states with e varying over the transition path are not yet implemented for EvalFnOnTransPath_AggVars (email me if you want this)')
end

%% Set up simoptions so EvalFnOnAgentDist_AggVars_FHorz_Case1 treats semiz (and e) as part of the joint exogenous state
if ~isfield(simoptions,'d_grid')
    simoptions.d_grid=d_grid;
end
simoptions.alreadygridvals_semiexo=0; % EvalFnOnAgentDist will build semiz_gridvals_J itself
if N_e>0
    simoptions.e_gridvals_J=e_gridvals_J;
end

%% AggVarNames (for output structure)
AggVarNames=fieldnames(FnsToEvaluate);

%% Reshape AgentDistPath and PolicyPath over the bothze composite
AgentDistPath=reshape(AgentDistPath,[N_a,N_bothze,N_j,T]);
PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_bothze,N_j,T]);

AggVarsPath=struct();

for tt=1:T
    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
    end

    if N_z>0 && transpathoptions.zpathtrivial==0
        z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,tt);
    end

    AgentDist_tt=reshape(AgentDistPath(:,:,:,tt),[N_a,N_bothze,N_j]);
    Policy_tt=reshape(PolicyPath(:,:,:,:,tt),[size(PolicyPath,1),n_a,N_bothze,N_j]);

    % Pass FnsToEvaluate as a struct so that the parameter-name offset accounts for semiz (l_z includes semiz inside this command).
    % EvalFnOnAgentDist_AggVars_FHorz_Case1 folds semiz (via simoptions.n_semiz) and e (via simoptions.n_e) into the joint exogenous state.
    AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(AgentDist_tt, Policy_tt, FnsToEvaluate, Parameters, [], n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, simoptions);

    for ff=1:length(AggVarNames)
        AggVarsPath.(AggVarNames{ff}).Mean(tt)=AggVars.(AggVarNames{ff}).Mean;
    end
end

end
