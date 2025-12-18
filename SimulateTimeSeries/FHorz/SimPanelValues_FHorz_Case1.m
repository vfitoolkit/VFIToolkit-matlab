function SimPanelValues=SimPanelValues_FHorz_Case1(InitialDist,Policy,FnsToEvaluate,Parameters,FnsToEvaluateParamNames,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,pi_z, simoptions)
% Simulates a panel based on PolicyIndexes of 'numbersims' agents of length
% 'simperiods' beginning from randomly drawn InitialDist.
% SimPanelValues is a 3-dimensional matrix with first dimension being the
% number of 'variables' to be simulated, second dimension is FHorz, and
% third dimension is the number-of-simulations
%
% InitialDist can be inputed as over the finite time-horizon (j), or
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
    simoptions.experienceassetu=0; % note, experienceassetu=1 not yet implmented
    simoptions.riskyasset=0; % note, riskyasset=1 not yet implmented
    simoptions.n_semiz=0;
    simoptions.n_e=0;
    % When calling as a subcommand, the following is used internally
    simoptions.alreadygridvals=0;
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
    if ~isfield(simoptions,'riskyasset')
        simoptions.riskyasset=0;
    end
    if ~isfield(simoptions,'experienceassetu')
        simoptions.experienceassetu=0;
    end
    if ~isfield(simoptions,'n_semiz')
        simoptions.n_semiz=0;
    end
    if ~isfield(simoptions,'n_e')
        simoptions.n_e=0;
    end
    % When calling as a subcommand, the following is used internally
    if ~isfield(simoptions, 'alreadygridvals')
        simoptions.alreadygridvals=0;
    end
    % For internal use only
    if ~isfield(simoptions,'keepoutputasmatrix')
        simoptions.keepoutputasmatrix=0;
    end
    if ~isfield(simoptions,'simpanelindexkron')
        simoptions.simpanelindexkron=0;
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

%% Simulate Panel Indexes
d_grid=gather(d_grid);
a_grid=gather(a_grid);

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

if N_z>0
    if ndims(pi_z)==3
        pi_z_J=pi_z;
    else
        pi_z_J=repelem(pi_z,1,1,N_j);
    end
else
    pi_z_J=[];
end

if N_e>0
    if size(simoptions.pi_e,2)==N_j
        simoptions.pi_e_J=simoptions.pi_e;
    else
        simoptions.pi_e_J=repelem(simoptions.pi_e,1,N_j);
    end
end


%% Send off to different simulation commands based on the setup (with/without z and e is handled within the commands)
simoptions.simpanelindexkron=1; % Keep the output as kron form as will want this later anyway for assigning the values
if simoptions.experienceasset==1
    if N_semiz==0
        SimPanelIndexes=SimPanelIndexes_FHorz_ExpAsset(gather(InitialDist),gather(Policy),n_d,n_a,n_z,N_j,gather(pi_z_J), Parameters, simoptions);
    else
        SimPanelIndexes=SimPanelIndexes_FHorz_ExpAsset_semiz(gather(InitialDist),gather(Policy),n_d,n_a,n_z,N_j,gather(pi_z_J), Parameters, simoptions);        
    end
elseif simoptions.experienceassetu==1
    error('Cannot yet simulate panel with simoptions.experienceassetu=1, ask on forum if you want this')
elseif simoptions.riskyasset==1    
    error('Cannot yet simulate panel with simoptions.riskyasset=1, ask on forum if you want this')
else
    if N_semiz==0
        SimPanelIndexes=SimPanelIndexes_FHorz(gather(InitialDist),gather(Policy),n_d,n_a,n_z,N_j,gather(pi_z_J), simoptions);
    else
        SimPanelIndexes=SimPanelIndexes_FHorz_semiz(gather(InitialDist),gather(Policy),n_d,n_a,n_z,N_j,gather(pi_z_J), Parameters, simoptions);
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
% Figure out l_daprime from Policy
l_daprime=size(Policy,1);
if simoptions.gridinterplayer==1
    l_daprime=l_daprime-1;
end

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
l_aprime=l_a;
if simoptions.experienceasset==1 || simoptions.experienceassetu==1
    l_aprime=l_aprime-1;
end
if N_d==0
    if N_semizze==0
        daprimePolicy_gridvals=zeros(N_a,l_aprime,N_j);
    else
        daprimePolicy_gridvals=zeros(N_a*N_semizze,l_aprime,N_j);
    end
else
    if N_semizze==0
        daprimePolicy_gridvals=zeros(N_a,l_d+l_aprime,N_j);
    else
        daprimePolicy_gridvals=zeros(N_a*N_semizze,l_d+l_aprime,N_j);
    end
end

for jj=1:N_j
    if N_d==0
        [~,aprimePolicy_gridvals_j]=CreateGridvals_Policy(Policy(:,:,:,jj),n_d,n_a,n_a,n_semizze,[],a_grid,simoptions,1, 1); % handles gridinterplayer=1 in here
        daprimePolicy_gridvals(:,:,jj)=aprimePolicy_gridvals_j;
    else
        [dPolicy_gridvals_j,aprimePolicy_gridvals_j]=CreateGridvals_Policy(Policy(:,:,:,jj),n_d,n_a,n_a,n_semizze,d_grid,a_grid,simoptions,1, 1); % handles gridinterplayer=1 in here
        daprimePolicy_gridvals(:,:,jj)=[dPolicy_gridvals_j, aprimePolicy_gridvals_j];
    end
end

a_gridvals=CreateGridvals(n_a,a_grid,1); % 1 at end indicates output as matrices.


%% Now switch everything to gpu so can use arrayfun() to evaluates all the FnsToEvaluate
daprimePolicy_gridvals=gpuArray(daprimePolicy_gridvals);
SimPanelIndexes=gpuArray(SimPanelIndexes);

SimPanelValues=nan(length(FnsToEvaluate), N_j, simoptions.numbersims,'gpuArray'); % needs to be NaN to permit that some people might be 'born' later than age j=1
% Note, having the whole N_j at this stage makes assiging the values based on the indexes vastly faster


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



