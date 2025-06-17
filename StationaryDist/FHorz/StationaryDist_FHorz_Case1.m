function StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Parameters,simoptions)
%%
if isempty(n_d)
    n_d=0;
end
N_z=prod(n_z);

if exist('simoptions','var')==0
    simoptions.verbose=0;
    simoptions.tolerance=10^(-9);
    % Things that are really just for internal usage
    simoptions.parallel=1+(gpuDeviceCount>0); % 1 if cpu, 2 if gpu
    simoptions.outputkron=0; % If 1 then leave output in Kron form
    simoptions.tanimprovement=1;  % Mostly hardcoded, but in simplest case of (a,z) you can try out the alternatives
    simoptions.loopovere=0; % default is parallel over e, 1 will loop over e, 2 will parfor loop over e
    simoptions.gridinterplayer=0; % =1 Policy interpolates between grid points (must match vfoptions.interpgridlayer)
    % Alternative endo states
    simoptions.experienceasset=0;
    simoptions.experienceassetu=0;
    simoptions.riskyasset=0;
    simoptions.residualasset=0;
    % When calling as a subcommand, the following is used internally
    simoptions.alreadygridvals=0;
else
    %Check simoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
    end
    if ~isfield(simoptions,'tolerance')
        simoptions.tolerance=10^(-9);
    end
    % Things that are really just for internal usage
    if ~isfield(simoptions,'parallel')
            simoptions.parallel=1+(gpuDeviceCount>0); % 1 if cpu, 2 if gpu
    end
    if ~isfield(simoptions,'outputkron')
        simoptions.outputkron=0; % If 1 then leave output in Kron form
    end
    if ~isfield(simoptions,'tanimprovement')
        simoptions.tanimprovement=1; % Mostly hardcoded, but in simplest case of (a,z) you can try out the alternatives
    end
    if ~isfield(simoptions,'loopovere')
        simoptions.loopovere=0; % default is parallel over e, 1 will loop over e, 2 will parfor loop over e
    end
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0; % =1 Policy interpolates between grid points (must match vfoptions.interpgridlayer)
    elseif simoptions.gridinterplayer==1
        if ~isfield(simoptions,'ngridinterp')
            error('When using simoptions.gridinterplayer=1 you must set simoptions.ngridinterp (number of points to interpolate for aprime between each consecutive pair of points in a_grid)')
        end
    end
    % Alternative endo states
    if ~isfield(simoptions,'experienceasset')
        simoptions.experienceasset=0;
    end
    if ~isfield(simoptions,'experienceassetu')
        simoptions.experienceassetu=0;
    end
    if ~isfield(simoptions,'riskyasset')
        simoptions.riskyasset=0;
    end
    if ~isfield(simoptions,'residualasset')
        simoptions.residualasset=0;
    end
    if ~isfield(simoptions,'alreadygridvals')
        % When calling as a subcommand, the following is used internally
        simoptions.alreadygridvals=0;
    end
end

% Check for something that used to be an option, but no longer is
if isfield(simoptions,'iterate')
    if simoptions.iterate==0
        error('simoptions.iterate=0 is no longer supported (has been eliminated from VFI Toolkit)')
    end
end

%% Check for the age weights parameter, and make sure it is a row vector
if size(Parameters.(AgeWeightParamNames{1}),2)==1 % Seems like column vector
    Parameters.(AgeWeightParamNames{1})=Parameters.(AgeWeightParamNames{1})'; 
    % Note: assumed there is only one AgeWeightParamNames
end
% And check that the age weights sum to one
if abs((sum(Parameters.(AgeWeightParamNames{1}))-1))>10^(-15)
    warning('StationaryDist: The age-weights do not sum to one')
end

%% Exogenous shock grids
if simoptions.alreadygridvals==0
    % Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
    [~, pi_z_J, simoptions]=ExogShockSetup_FHorz(n_z,[],pi_z,N_j,Parameters,simoptions,2);
    % note: output z_gridvals_J, pi_z_J, and simoptions.e_gridvals_J, simoptions.pi_e_J
    %
    % size(z_gridvals_J)=[prod(n_z),length(n_z),N_j]
    % size(pi_z_J)=[prod(n_z),prod(n_z),N_j]
    % size(e_gridvals_J)=[prod(n_e),length(n_e),N_j]
    % size(pi_e_J)=[prod(n_e),N_j]
    % If no z, then z_gridvals_J=[] and pi_z_J=[]
    % If no e, then e_gridvals_J=[] and pi_e_J=[]
elseif simoptions.alreadygridvals==1
    % z_gridvals_J=z_grid;
    pi_z_J=pi_z;
end

%% Semi-exogenous shock gridvals and pi 
if isfield(simoptions,'n_semiz')
    % Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
    simoptions=SemiExogShockSetup_FHorz(n_d,N_j,simoptions.d_grid,Parameters,simoptions,1);
    % output: simoptions.semiz_gridvals_J, simoptions.pi_semiz_J
    % size(semiz_gridvals_J)=[prod(n_z),length(n_z),N_j]
    % size(pi_semiz_J)=[prod(n_semiz),prod(n_semiz),prod(n_dsemiz),N_j]
    % If no semiz, then simoptions just does not contain these field
end


%% If age one distribution is input as a function, then evaluate it
% Note: we might need to update z_grid based on ExogShockFn, so this has to come after the pi_z section
if isa(jequaloneDist, 'function_handle')
    jequaloneDistFn=jequaloneDist;
    clear jequaloneDist
    % figure out any parameters
    temp=getAnonymousFnInputNames(jequaloneDistFn);
    if length(temp)>4 % first 4 are a_grid,z_grid,n_a,n_z
        jequaloneDistFnParamNames={temp{5:end}}; % the first inputs will always be (a_grid,z_grid,n_a,n_z,...)
    else
        jequaloneDistFnParamNames={};
    end
    jequaloneParamsCell={};
    for pp=1:length(jequaloneDistFnParamNames)
        jequaloneParamsCell{pp}=Parameters.(jequaloneDistFnParamNames{pp});
    end
    % make sure a_grid and z_grid have been put in simoptions
    if ~isfield(simoptions,'a_grid')
        error('When using jequaloneDist as a function you must put a_grid into simoptions.a_grid')
    elseif ~isfield(simoptions,'z_grid')
        error('When using jequaloneDist as a function you must put z_grid into simoptions.z_grid')
    end

    jequaloneDist=jequaloneDistFn(simoptions.a_grid,simoptions.z_grid,n_a,n_z,jequaloneParamsCell{:});
end

% Check that the age one distribution is of mass one
if abs(sum(jequaloneDist(:))-1)>10^(-9)
    error('The jequaloneDist must be of mass one')
end


%% Non-standard endogenous states
if simoptions.experienceasset==1
    if isfield(simoptions,'n_semiz')
        StationaryDist=StationaryDist_FHorz_ExpAssetSemiExo(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,simoptions.n_semiz,n_z,N_j,simoptions.pi_semiz_J,pi_z_J,Parameters,simoptions);
        return
    end
    StationaryDist=StationaryDist_FHorz_ExpAsset(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);
    return
end
if simoptions.experienceassetu==1
    StationaryDist=StationaryDist_FHorz_ExpAssetu(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);
    return
end
if simoptions.riskyasset==1
    if isfield(simoptions,'n_semiz')
    StationaryDist=StationaryDist_FHorz_RiskyAssetSemiExo(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,simoptions.n_semiz,n_z,N_j,simoptions.pi_semiz_J,pi_z_J,Parameters,simoptions);
        return
    end
    StationaryDist=StationaryDist_FHorz_RiskyAsset(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);
    return
end
if simoptions.residualasset==1
    StationaryDist=StationaryDist_FHorz_ResidAsset(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);
    return
end

%% Standard endogenous states
if isfield(simoptions,'n_e')
    N_e=prod(simoptions.n_e);
else
    N_e=0;
end

if simoptions.gridinterplayer==0
    if isfield(simoptions,'n_semiz')
        if N_e==0
            if N_z==0
                StationaryDist=StationaryDist_FHorz_SemiExo_noz(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,simoptions.n_semiz,N_j,simoptions.pi_semiz_J,Parameters,simoptions);
            else
                StationaryDist=StationaryDist_FHorz_SemiExo(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,simoptions.n_semiz,n_z,N_j,simoptions.pi_semiz_J,pi_z_J,Parameters,simoptions);
            end
        else
            if N_z==0
                StationaryDist=StationaryDist_FHorz_SemiExo_noz_e(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,simoptions.n_semiz,simoptions.n_e,N_j,simoptions.pi_semiz_J,simoptions.pi_e_J,Parameters,simoptions);
            else
                StationaryDist=StationaryDist_FHorz_SemiExo_e(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,simoptions.n_semiz,n_z,simoptions.n_e,N_j,simoptions.pi_semiz_J,pi_z_J,simoptions.pi_e_J,Parameters,simoptions);
            end
        end
    else
        if N_e==0
            if N_z==0
                StationaryDist=StationaryDist_FHorz_noz(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,N_j,Parameters,simoptions);
            else
                StationaryDist=StationaryDist_FHorz_raw(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);
            end
        else
            if N_z==0
                StationaryDist=StationaryDist_FHorz_noz_e(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,simoptions.n_e,N_j,simoptions.pi_e_J,Parameters,simoptions);
            else
                StationaryDist=StationaryDist_FHorz_e(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,simoptions.n_e,N_j,pi_z_J,simoptions.pi_e_J,Parameters,simoptions);
            end
        end
    end

elseif simoptions.gridinterplayer==1
    if isfield(simoptions,'n_semiz')
        if N_e==0
            if N_z==0
                StationaryDist=StationaryDist_FHorz_SemiExo_GI_noz(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,simoptions.n_semiz,N_j,simoptions.pi_semiz_J,Parameters,simoptions);
            else
                StationaryDist=StationaryDist_FHorz_SemiExo_GI(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,simoptions.n_semiz,n_z,N_j,simoptions.pi_semiz_J,pi_z_J,Parameters,simoptions);
            end
        else
            if N_z==0
                StationaryDist=StationaryDist_FHorz_SemiExo_GI_noz_e(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,simoptions.n_semiz,simoptions.n_e,N_j,simoptions.pi_semiz_J,simoptions.pi_e_J,Parameters,simoptions);
            else
                StationaryDist=StationaryDist_FHorz_SemiExo_GI_e(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,simoptions.n_semiz,n_z,simoptions.n_e,N_j,simoptions.pi_semiz_J,pi_z_J,simoptions.pi_e_J,Parameters,simoptions);
            end
        end
    else
        if N_e==0
            if N_z==0
                StationaryDist=StationaryDist_FHorz_GI_noz(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,N_j,Parameters,simoptions);
            else
                StationaryDist=StationaryDist_FHorz_GI_raw(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);
            end
        else
            if N_z==0
                StationaryDist=StationaryDist_FHorz_GI_noz_e(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,simoptions.n_e,N_j,simoptions.pi_e_J,Parameters,simoptions);
            else
                StationaryDist=StationaryDist_FHorz_GI_e(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,simoptions.n_e,N_j,pi_z_J,simoptions.pi_e_J,Parameters,simoptions);
            end
        end
    end

end



end
