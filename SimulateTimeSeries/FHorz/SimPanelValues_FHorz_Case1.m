function SimPanelValues=SimPanelValues_FHorz_Case1(InitialDist,Policy,FnsToEvaluate,Parameters,FnsToEvaluateParamNames,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,pi_z, simoptions)
% Simulates a panel based on PolicyIndexes of 'numbersims' agents of length
% 'simperiods' beginning from randomly drawn InitialDist.
% SimPanelValues is a 3-dimensional matrix with first dimension being the
% number of 'variables' to be simulated, second dimension is FHorz, and
% third dimension is the number-of-simulations
%
% InitialDist can be inputted as over the finite time-horizon (j), or
% without a time-horizon in which case it is assumed to be an InitialDist
% for time j=1. (So InitialDist is either n_a-by-n_z-by-n_j, or n_a-by-n_z)

%% Check which simoptions have been declared, set all others to defaults
if ~exist('simoptions','var')
    % If simoptions is not given, just use all the defaults
    simoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    simoptions.verbose=0;
    simoptions.simperiods=N_j;
    simoptions.numbersims=10^3;
    simoptions.lowmemory=0; % setting to 1 slows the simulations, but reduces memory
    % Model setup
    simoptions.gridinterplayer=0;
    simoptions.experienceasset=0;
    simoptions.experienceassetu=0;
    simoptions.experienceassetz=0;
    simoptions.experienceassete=0;
    simoptions.experienceassetze=0;
    simoptions.riskyasset=0;
    simoptions.n_semiz=0;
    simoptions.n_e=0;
    % When calling as a subcommand, the following is used internally
    simoptions.alreadygridvals=0;
    simoptions.alreadygridvals_semiexo=0;
    % For internal use only
    simoptions.keepoutputasmatrix=0;
    simoptions.simpanelindexkron=0;
else
    %Check simoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(simoptions,'parallel')
        simoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
    end
    if ~isfield(simoptions,'simperiods')
        simoptions.simperiods=N_j;
    end
    if ~isfield(simoptions,'numbersims')
        simoptions.numbersims=10^3;
    end
    if ~isfield(simoptions,'lowmemory')
        simoptions.lowmemory=0; % setting to 1 slows the simulations, but reduces memory
    end
    % Model setup
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0;
    end
    if ~isfield(simoptions,'experienceasset')
        simoptions.experienceasset=0;
    end
    if ~isfield(simoptions,'experienceassetu')
        simoptions.experienceassetu=0;
    end
    if ~isfield(simoptions,'experienceassetz')
        simoptions.experienceassetz=0;
    end
    if ~isfield(simoptions,'experienceassete')
        simoptions.experienceassete=0;
    end
    if ~isfield(simoptions,'experienceassetze')
        simoptions.experienceassetze=0;
    end
    if ~isfield(simoptions,'riskyasset')
        simoptions.riskyasset=0;
    end
    if ~isfield(simoptions,'n_semiz')
        simoptions.n_semiz=0;
    end
    if ~isfield(simoptions,'n_e')
        simoptions.n_e=0;
    end
    % When calling as a subcommand, the following is used internally
    if ~isfield(simoptions,'alreadygridvals')
        simoptions.alreadygridvals=0;
    end
    if ~isfield(simoptions,'alreadygridvals_semiexo')
        simoptions.alreadygridvals_semiexo=0;
    end
    % For internal use only
    if ~isfield(simoptions,'keepoutputasmatrix')
        simoptions.keepoutputasmatrix=0;
    end
    if ~isfield(simoptions,'simpanelindexkron')
        simoptions.simpanelindexkron=0;
    end
    % Some options require certain other inputs, and these have to be on the GPU
    if isfield(simoptions,'d_grid')
        simoptions.d_grid=gpuArray(simoptions.d_grid);
    elseif simoptions.experienceasset>=1 || simoptions.experienceassetz>=1 || simoptions.experienceassete>=1 || simoptions.experienceassetze>=1 || simoptions.experienceassetu>=1
        error('When using any kind of experience asset you must set simoptions.d_grid')
    elseif simoptions.riskyasset==1
        error('When using a risky asset you must set simoptions.d_grid')
    end
    if isfield(simoptions,'a_grid')
        simoptions.a_grid=gpuArray(simoptions.a_grid);
    elseif simoptions.experienceasset>=1 || simoptions.experienceassetz>=1 || simoptions.experienceassete>=1 || simoptions.experienceassetze>=1 || simoptions.experienceassetu>=1
        error('When using any kind of experience asset you must set simoptions.a_grid')
    end
    if isfield(simoptions,'z_grid')
        simoptions.z_grid=gpuArray(simoptions.z_grid);
    elseif simoptions.experienceassetz>=1
        error('When using experienceassetz you must set simoptions.z_grid')
    elseif simoptions.experienceassetze>=1
        error('When using experienceassetze you must set simoptions.z_grid')
    end
end

if isfield(simoptions,'ExogShockFn') % If using ExogShockFn then figure out the parameter names
    simoptions.ExogShockFnParamNames=getAnonymousFnInputNames(simoptions.ExogShockFn);
end

simoptions.simperiods=gather(simoptions.simperiods);
simoptions.numbersims=gather(simoptions.numbersims); % This is just to deal with weird error that matlab decided simoptions.numbersims was on gpu and so couldn't be an input to rand()


%%
if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);

N_d=prod(n_d);
N_a=prod(n_a);
% N_semizze set below

%% Move Policy and grids to GPU
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
Policy=gpuArray(Policy);

N_semiz=prod(simoptions.n_semiz);
N_z=prod(n_z);
N_e=prod(simoptions.n_e);

if N_z==0
    if N_e==0
        if N_semiz==0
            n_semizze=0;
        else
            n_semizze=simoptions.n_semiz;
        end
    else
        if N_semiz==0
            n_semizze=simoptions.n_e;
        else
            n_semizze=[simoptions.n_semiz,simoptions.n_e];
        end
    end
else
    if N_e==0
        if N_semiz==0
            n_semizze=n_z;
        else
            n_semizze=[simoptions.n_semiz,n_z];
        end
    else
        if N_semiz==0
            n_semizze=[n_z,simoptions.n_e];
        else
            n_semizze=[simoptions.n_semiz,n_z,simoptions.n_e];
        end
    end
end
N_semizze=prod(n_semizze);

%% Exogenous shock grids
if simoptions.alreadygridvals==0
    % Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
    [z_gridvals_J, pi_z_J, simoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,simoptions,3);
    % note: output z_gridvals_J, pi_z_J, and simoptions.e_gridvals_J, simoptions.pi_e_J
    %
    % size(z_gridvals_J)=[prod(n_z),length(n_z),N_j]
    % size(pi_z_J)=[prod(n_z),prod(n_z),N_j]
    % size(e_gridvals_J)=[prod(n_e),length(n_e),N_j]
    % size(pi_e_J)=[prod(n_e),N_j]
    % If no z, then z_gridvals_J=[] and pi_z_J=[]
    % If no e, then e_gridvals_J=[] and pi_e_J=[]
elseif simoptions.alreadygridvals==1
    z_gridvals_J=z_grid;
    pi_z_J=pi_z;
end

%% Send off to different simulation commands based on the setup (with/without z and e is handled within the commands)
simoptions.simpanelindexkron=1; % Keep the output as kron form as will want this later anyway for assigning the values
if simoptions.experienceasset>=1
    if N_semiz==0
        SimPanelIndexes=SimPanelIndexes_FHorz_ExpAsset(gather(InitialDist),Policy,n_d,n_a,n_z,N_j,gather(pi_z_J), Parameters, simoptions);
    else
        SimPanelIndexes=SimPanelIndexes_FHorz_ExpAsset_semiz(gather(InitialDist),Policy,n_d,n_a,n_z,N_j,gather(pi_z_J), Parameters, simoptions);
    end
elseif simoptions.experienceassetu>=1
    if N_semiz==0
        SimPanelIndexes=SimPanelIndexes_FHorz_ExpAssetU(gather(InitialDist),Policy,n_d,n_a,n_z,N_j,gather(pi_z_J), Parameters, simoptions);
    else
        SimPanelIndexes=SimPanelIndexes_FHorz_ExpAssetU_semiz(gather(InitialDist),Policy,n_d,n_a,n_z,N_j,gather(pi_z_J), Parameters, simoptions);
    end
elseif simoptions.experienceassetz>=1
    if N_semiz==0
        SimPanelIndexes=SimPanelIndexes_FHorz_ExpAssetz(gather(InitialDist),Policy,n_d,n_a,n_z,N_j,z_gridvals_J,gather(pi_z_J), Parameters, simoptions);
    else
        SimPanelIndexes=SimPanelIndexes_FHorz_ExpAssetz_semiz(gather(InitialDist),Policy,n_d,n_a,n_z,N_j,z_gridvals_J,gather(pi_z_J), Parameters, simoptions);
    end
elseif simoptions.experienceassete>=1
    if N_semiz==0
        SimPanelIndexes=SimPanelIndexes_FHorz_ExpAssete(gather(InitialDist),Policy,n_d,n_a,n_z,N_j,gather(pi_z_J), Parameters, simoptions);
    else
        SimPanelIndexes=SimPanelIndexes_FHorz_ExpAssete_semiz(gather(InitialDist),Policy,n_d,n_a,n_z,N_j,gather(pi_z_J), Parameters, simoptions);
    end
elseif simoptions.experienceassetze>=1
    if N_semiz==0
        SimPanelIndexes=SimPanelIndexes_FHorz_ExpAssetze(gather(InitialDist),Policy,n_d,n_a,n_z,N_j,z_gridvals_J,gather(pi_z_J), Parameters, simoptions);
    else
        SimPanelIndexes=SimPanelIndexes_FHorz_ExpAssetze_semiz(gather(InitialDist),Policy,n_d,n_a,n_z,N_j,z_gridvals_J,gather(pi_z_J), Parameters, simoptions);
    end
elseif simoptions.riskyasset==1
    if N_semiz==0
        SimPanelIndexes=SimPanelIndexes_FHorz_RiskyAsset(gather(InitialDist),Policy,n_d,n_a,n_z,N_j,gather(pi_z_J), Parameters, simoptions);
    else
        SimPanelIndexes=SimPanelIndexes_FHorz_RiskyAsset_semiz(gather(InitialDist),Policy,n_d,n_a,n_z,N_j,gather(pi_z_J), Parameters, simoptions);
    end
else
    if N_semiz==0
        SimPanelIndexes=SimPanelIndexes_FHorz(gather(InitialDist),Policy,n_d,n_a,n_z,N_j,gather(pi_z_J), simoptions);
    else
        SimPanelIndexes=SimPanelIndexes_FHorz_semiz(gather(InitialDist),Policy,n_d,n_a,n_z,N_j,gather(pi_z_J), Parameters, simoptions);
    end
end
% Note: SimPanelIndexes contains semiz,z,e all in a single dimension (as that suits how the gridvals are created below)

%% Exogenous shock grids (must come after the SimPanelIndexes as it then strips n_semiz and n_e out of simoptions)
% Create the combination of (semiz,z,e) as all three are the same for FnsToEvaluate
[~,semizze_gridvals_J,~,~,simoptions]=CreateGridvals_FnsToEvaluate_FHorz(n_z,z_grid,N_j,simoptions,Parameters);
% N_semizze=prod(n_semizze);
if N_semizze==0
    l_semizze=0;
else
    l_semizze=length(n_semizze);
end
% Note: semiz, z and e are from here on all just rolled together in n_z, z_gridvals_J, N_z and l_z

%% Implement new way of handling FnsToEvaluate
% Figure out l_aprime and l_daprime
l_aprime=l_a;
if simoptions.experienceasset>=1 || simoptions.experienceassetu>=1 || simoptions.experienceassetz>=1 || simoptions.experienceassete>=1 || simoptions.experienceassetze>=1
    l_aprime=l_aprime-1;
end
l_daprime=l_d+l_aprime;

% Note: l_semizze
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    FnsToEvalNames=fieldnames(FnsToEvaluate);
    for ff=1:length(FnsToEvalNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(FnsToEvalNames{ff}));
        if length(temp)>(l_daprime+l_a+l_semizze)
            FnsToEvaluateParamNames(ff).Names={temp{l_daprime+l_a+l_semizze+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(FnsToEvalNames{ff});
    end
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end
if isfield(simoptions,'keepoutputasmatrix')
    if simoptions.keepoutputasmatrix==1
        FnsToEvaluateStruct=0;
    end
end

nFnsToEvalute=length(FnsToEvaluate);


%% Precompute the gridvals vectors.
if N_semizze==0
    Policy=reshape(Policy,[size(Policy,1),N_a,1,N_j]);
else
    Policy=reshape(Policy,[size(Policy,1),N_a,N_semizze,N_j]);
end

% Note that dPolicy and aprimePolicy will depend on age
if N_semizze==0
    daprimePolicy_gridvals=zeros(N_a,l_daprime,N_j);
else
    daprimePolicy_gridvals=zeros(N_a*N_semizze,l_daprime,N_j);
end

for jj=1:N_j
    if l_aprime==0 
        % E.g., a model with just one endogenous state, which is an experience asset
        [dPolicy_gridvals_j,~]=CreateGridvals_Policy(Policy(:,:,:,jj),n_d,n_a,n_a,n_semizze,d_grid,a_grid,simoptions,1, 1); % handles gridinterplayer=1 in here
        daprimePolicy_gridvals(:,:,jj)=dPolicy_gridvals_j;
    else
        if N_d==0
            [~,aprimePolicy_gridvals_j]=CreateGridvals_Policy(Policy(:,:,:,jj),n_d,n_a,n_a,n_semizze,[],a_grid,simoptions,1, 1); % handles gridinterplayer=1 in here
            daprimePolicy_gridvals(:,:,jj)=aprimePolicy_gridvals_j;
        else
            [dPolicy_gridvals_j,aprimePolicy_gridvals_j]=CreateGridvals_Policy(Policy(:,:,:,jj),n_d,n_a,n_a,n_semizze,d_grid,a_grid,simoptions,1, 1); % handles gridinterplayer=1 in here
            daprimePolicy_gridvals(:,:,jj)=[dPolicy_gridvals_j, aprimePolicy_gridvals_j];
        end
    end
end

a_gridvals=CreateGridvals(n_a,a_grid,1); % 1 at end indicates output as matrices.


%% Now switch everything to gpu so can use arrayfun() to evaluates all the FnsToEvaluate
daprimePolicy_gridvals=gpuArray(daprimePolicy_gridvals);
SimPanelIndexes=gpuArray(SimPanelIndexes);

SimPanelValues=nan(length(FnsToEvaluate), N_j, simoptions.numbersims,'gpuArray'); % needs to be NaN to permit that some people might be 'born' later than age j=1
% Note, having the whole N_j at this stage makes assigning the values based on the indexes vastly faster


%% Create PanelValues from PanelIndexes
if N_semizze>0
    for jj=1:N_j
        SimPanelIndexes_jj=SimPanelIndexes(:,jj,:);

        relevantindices=(~isnan(SimPanelIndexes_jj(1,1,:))); % Note, is just across the ii dimension
        sumrelevantindices=sum(relevantindices);

        if sumrelevantindices>0 % Does the simulation even contain anyone of age jj?
            currentPanelIndexes_jj=SimPanelIndexes_jj(:,1,relevantindices);
            currentPanelValues_jj=zeros(sumrelevantindices,nFnsToEvalute); % transpose will be taken before storing

            az_ind=squeeze(currentPanelIndexes_jj(1,1,:)+N_a*(currentPanelIndexes_jj(2,1,:)-1));
            % a_ind=currentPanelIndexes_jj(1,1,:);
            % z_ind=currentPanelIndexes_jj(2,1,:); % this is semiz,z,e all together
            % j_ind=currentPanelIndexes_jj(3,1,:);

            a_val=a_gridvals(currentPanelIndexes_jj(1,1,:),:); % a_grid does depend on age
            z_val=semizze_gridvals_J(currentPanelIndexes_jj(2,1,:),:,jj);

            for vv=1:nFnsToEvalute
                if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'FnsToEvaluateParamNames={}'
                    ParamCell={};
                else
                    ValuesFnParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,jj);
                    ParamCell=num2cell(ValuesFnParamsVec);
                end
                daprime_val=daprimePolicy_gridvals(az_ind,:,jj);
                currentPanelValues_jj(:,vv)=EvalFnOnSimPanelIndex(FnsToEvaluate{vv},ParamCell,daprime_val,a_val,z_val,l_daprime,l_a,l_semizze);
            end
            SimPanelValues(:,jj,relevantindices)=reshape(currentPanelValues_jj',[nFnsToEvalute,1,sumrelevantindices]);

        end
    end
else % N_semizze==0
    for jj=1:N_j
        SimPanelIndexes_jj=SimPanelIndexes(:,jj,:);

        relevantindices=(~isnan(SimPanelIndexes_jj(1,1,:))); % Note, is just across the ii dimension
        sumrelevantindices=sum(relevantindices);

        if sumrelevantindices>0 % Does the simulation even contain anyone of age jj?
            currentPanelIndexes_jj=SimPanelIndexes_jj(:,1,relevantindices);
            currentPanelValues_jj=zeros(sumrelevantindices,nFnsToEvalute); % transpose will be taken before storing

            a_ind=squeeze(currentPanelIndexes_jj(1,1,:));
            % j_ind=currentPanelIndexes_jj(2,1,:);

            a_val=a_gridvals(a_ind,:); % a_grid does depend on age

            for vv=1:nFnsToEvalute
                if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'FnsToEvaluateParamNames={}'
                    ParamCell={};
                else
                    ValuesFnParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,jj);
                    ParamCell=num2cell(ValuesFnParamsVec);
                end
                daprime_val=daprimePolicy_gridvals(a_ind,:,jj);
                currentPanelValues_jj(:,vv)=EvalFnOnSimPanelIndex(FnsToEvaluate{vv},ParamCell,daprime_val,a_val,[],l_daprime,l_a,0);
            end
            SimPanelValues(:,jj,relevantindices)=reshape(currentPanelValues_jj',[nFnsToEvalute,1,sumrelevantindices]);

        end
    end
end


%% I SHOULD ADD OPTION HERE TO ONLY OUTPUT THE SIMULATED PERIODS AND NOT THE WHOLE N_j (WHEN simperiods<N_j)




%% Implement new way of handling FnsToEvaluate: convert results
if FnsToEvaluateStruct==1
    % Change the output into a structure
    SimPanelValues2=SimPanelValues;
    clear SimPanelValues
    SimPanelValues=struct();
    for ff=1:length(FnsToEvalNames)
        SimPanelValues.(FnsToEvalNames{ff})=shiftdim(SimPanelValues2(ff,:,:),1);
    end
end



end



