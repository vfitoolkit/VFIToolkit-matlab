function StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Parameters,simoptions)
%% Finite Horizon agent distribution. Solves using iteration (implemented with the Tan improvement).
% jequaloneDist is the distribution of agents in period j=1.

if exist('simoptions','var')==0
    simoptions.gridinterplayer=0; % =1 Policy interpolates between grid points (must match vfoptions.interpgridlayer)
    % Alternative endo states
    simoptions.experienceasset=0;
    simoptions.experienceassetu=0;
    simoptions.riskyasset=0;
    simoptions.residualasset=0;
    % Things that are really just for internal usage
    simoptions.parallel=1+(gpuDeviceCount>0);
    simoptions.outputkron=0; % If 1 then leave output in Kron form
    simoptions.alreadygridvals=0; % =1 when calling as a subcommand
else
    %Check simoptions for missing fields, if there are some fill them with the defaults
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
    % Things that are really just for internal usage
    if ~isfield(simoptions,'parallel')
        simoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(simoptions,'outputkron')
        simoptions.outputkron=0; % If 1 then leave output in Kron form
    end
    if ~isfield(simoptions,'alreadygridvals')
        simoptions.alreadygridvals=0; % =1 when calling as a subcommand
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

%% 
if simoptions.parallel==2
   % If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
   % Nothing to actually do here
else
   % CPU can be used, but only for the basics. Is kept separate here so that the rest of the codes can just assume you have GPU and work with it.
   StationaryDist=StationaryDist_FHorz_CPU(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Parameters,simoptions);
   return
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

%%
if isfield(simoptions,'n_semiz')
    N_semiz=prod(simoptions.n_semiz);
    if ~isfield(simoptions,'l_dsemiz')
        simoptions.l_dsemiz=1; % by default, just one decision variable is used for the semi-exo state
    end
else
    N_semiz=0;
end


%% Non-standard endogenous states
if simoptions.experienceasset==1
    if N_semiz==0
        StationaryDist=StationaryDist_FHorz_ExpAsset(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);
        return
    else
        StationaryDist=StationaryDist_FHorz_ExpAssetSemiExo(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,simoptions.n_semiz,n_z,N_j,simoptions.pi_semiz_J,pi_z_J,Parameters,simoptions);
        return
    end
end
if simoptions.experienceassetu==1
    if N_semiz==0
        StationaryDist=StationaryDist_FHorz_ExpAssetu(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);
        return

    else
        StationaryDist=StationaryDist_FHorz_ExpAssetuSemiExo(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,simoptions.n_semiz,n_z,N_j,simoptions.pi_semiz_J,pi_z_J,Parameters,simoptions);
        return
    end
end
if simoptions.riskyasset==1
    if ~isfield(simoptions,'refine_d')
        warning('Using simoptions.riskyasset=1 without setting simoptions.refine_d is outdated behaviour, it is strongly recommended you set simoptions.refine_d')
    end

    if N_semiz==0
        StationaryDist=StationaryDist_FHorz_RiskyAsset(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);
        return
    else
        StationaryDist=StationaryDist_FHorz_RiskyAssetSemiExo(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,simoptions.n_semiz,n_z,N_j,simoptions.pi_semiz_J,pi_z_J,Parameters,simoptions);
        return
    end
end
if simoptions.residualasset==1
    StationaryDist=StationaryDist_FHorz_ResidAsset(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);
    return
end



%% Standard endogenous states
if N_semiz>0
    StationaryDist=StationaryDist_FHorz_SemiExo(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,simoptions.n_semiz,n_z,N_j,simoptions.pi_semiz_J,pi_z_J,Parameters,simoptions);
    return
end





%% Now we are just left with the baseline: standard endogenous state, with/without markov z and with/without iid e

%% Key to understanding the below is to notice that the agent distribution evolves according to aprime, but d is irrelevant
% So the code is just cleaning up Policy_aprime, and then goes to the appropriate iteration command based on what shocks are in the model

if prod(n_d)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);

N_a=prod(n_a);
N_z=prod(n_z);
if isfield(simoptions,'n_e')
    N_e=prod(simoptions.n_e);
else
    N_e=0;
end

% Deal with the no z and no e first (as it needs different shapes to the rest)
if N_z==0 && N_e==0
    jequaloneDist=gpuArray(jequaloneDist); % make sure it is on gpu
    jequaloneDist=reshape(jequaloneDist,[N_a,1]);
    Policy=reshape(Policy,[size(Policy,1),N_a,N_j]);

    % Policy_aprime
    if l_a==1 % one endo state
        Policy_aprime=Policy(l_d+1,:,:);
    elseif l_a==2 % two endo states
        Policy_aprime=Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1);
    elseif l_a==3 % three endo states
        Policy_aprime=Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:)-1);
    elseif l_a==4 % four endo states
        Policy_aprime=Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(Policy(l_d+4,:,:)-1);
    else
        error('Not yet implemented standard endogenous states with length(n_a)>4')
    end
    Policy_aprime=shiftdim(Policy_aprime,1);

    if simoptions.gridinterplayer==0
        StationaryDist=StationaryDist_FHorz_Iteration_noz_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,N_a,N_j,Parameters);
    elseif simoptions.gridinterplayer==1
        % (a,1,j)
        Policy_aprime=reshape(Policy_aprime,[N_a,1,N_j]);
        Policy_aprime=repmat(Policy_aprime,1,2,1);
        PolicyProbs=ones([N_a,2,N_j],'gpuArray');
        % Policy_aprime(:,1,:) lower grid point for a1 is unchanged
        Policy_aprime(:,2,:)=Policy_aprime(:,2,:)+1; % add one to a1, to get upper grid point

        aprimeProbs_upper=reshape(shiftdim((Policy(end,:,:)-1)/(simoptions.ngridinterp+1),1),[N_a,1,N_j]); % probability of upper grid point (from L2 index)
        PolicyProbs(:,1,:)=PolicyProbs(:,1,:).*(1-aprimeProbs_upper); % lower a1
        PolicyProbs(:,2,:)=PolicyProbs(:,2,:).*aprimeProbs_upper; % upper a1

        StationaryDist=StationaryDist_FHorz_Iteration_nProbs_noz_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,PolicyProbs,2,N_a,N_j,Parameters);
    end
else
    if N_z==0
        n_ze=simoptions.n_e;
        N_ze=N_e;
    elseif N_e==0
        n_ze=n_z;
        N_ze=N_z;
    else % neither is zero
        n_ze=[n_z,simoptions.n_e];
        N_ze=N_z*N_e;
    end

    jequaloneDist=gpuArray(jequaloneDist); % make sure it is on gpu
    jequaloneDist=reshape(jequaloneDist,[N_a*N_ze,1]);
    Policy=reshape(Policy,[size(Policy,1),N_a,N_ze,N_j]);

    % Policy_aprime
    if l_a==1 % one endo state
        Policy_aprime=Policy(l_d+1,:,:,:);
    elseif l_a==2 % two endo states
        Policy_aprime=Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1);
    elseif l_a==3 % three endo states
        Policy_aprime=Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:,:)-1);
    elseif l_a==4 % four endo states
        Policy_aprime=Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(Policy(l_d+4,:,:,:)-1);
    else
        error('Not yet implemented standard endogenous states with length(n_a)>4')
    end
    Policy_aprime=shiftdim(Policy_aprime,1);

    
    %%
    if simoptions.gridinterplayer==0
        if N_z==0 && N_e==0 % handled separately above
            % StationaryDist=StationaryDist_FHorz_Iteration_noz_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,N_a,N_j,Parameters);
        elseif N_e==0 % just z
            StationaryDist=StationaryDist_FHorz_Iteration_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,N_a,N_z,N_j,pi_z_J,Parameters);
        elseif N_z==0 % just e
            StationaryDist=StationaryDist_FHorz_Iteration_noz_e_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,N_a,N_e,N_j,simoptions.pi_e_J,Parameters);
        else % both z and e
            StationaryDist=StationaryDist_FHorz_Iteration_e_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,N_a,N_z,N_e,N_j,pi_z_J,simoptions.pi_e_J,Parameters);
        end
    elseif simoptions.gridinterplayer==1
        % (a,z,1,j)
        Policy_aprime=reshape(Policy_aprime,[N_a,N_ze,1,N_j]);
        Policy_aprime=repmat(Policy_aprime,1,1,2,1);
        PolicyProbs=ones([N_a,N_ze,2,N_j],'gpuArray');
        % Policy_aprime(:,:,1,:) lower grid point for a1 is unchanged
        Policy_aprime(:,:,2,:)=Policy_aprime(:,:,2,:)+1; % add one to a1, to get upper grid point

        aprimeProbs_upper=reshape(shiftdim((Policy(end,:,:,:)-1)/(simoptions.ngridinterp+1),1),[N_a,N_ze,1,N_j]); % probability of upper grid point (from L2 index)
        PolicyProbs(:,:,1,:)=PolicyProbs(:,:,1,:).*(1-aprimeProbs_upper); % lower a1
        PolicyProbs(:,:,2,:)=PolicyProbs(:,:,2,:).*aprimeProbs_upper; % upper a1

        if N_z==0 && N_e==0 % handled separately above
            % StationaryDist=StationaryDist_FHorz_Iteration_nProbs_noz_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,PolicyProbs,2,N_a,N_j,Parameters);
        elseif N_e==0 % just z
            StationaryDist=StationaryDist_FHorz_Iteration_nProbs_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,PolicyProbs,2,N_a,N_z,N_j,pi_z_J,Parameters);
        elseif N_z==0 % just e
            StationaryDist=StationaryDist_FHorz_Iteration_nProbs_noz_e_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,PolicyProbs,2,N_a,N_e,N_j,simoptions.pi_e_J,Parameters);
        else % both z and e
            StationaryDist=StationaryDist_FHorz_Iteration_nProbs_e_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,PolicyProbs,2,N_a,N_z,N_e,N_j,pi_z_J,simoptions.pi_e_J,Parameters);
        end
    end
end


if N_z==0 && N_e==0
    if simoptions.outputkron==0
        StationaryDist=reshape(StationaryDist,[n_a,N_j]);
    else
        % If 1 then leave output in Kron form
        StationaryDist=reshape(StationaryDist,[N_a,N_j]);
    end
else
    if simoptions.outputkron==0
        StationaryDist=reshape(StationaryDist,[n_a,n_ze,N_j]);
    else
        % If 1 then leave output in Kron form
        StationaryDist=reshape(StationaryDist,[N_a,N_ze,N_j]);
    end
end

end
