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
    %If simoptions is not given, just use all the defaults
    simoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    simoptions.verbose=0;
    simoptions.simperiods=N_j;
    simoptions.numbersims=10^3;
    simoptions.lowmemory=0; % setting to 1 slows the simulations, but reduces memory
    simoptions.keepoutputasmatrix=0;
    % When calling as a subcommand, the following is used internally
    simoptions.gridinterplayer=0;
    simoptions.alreadygridvals=0;
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
    if isfield(simoptions,'ExogShockFn') % If using ExogShockFn then figure out the parameter names
        simoptions.ExogShockFnParamNames=getAnonymousFnInputNames(simoptions.ExogShockFn);
    end
    if ~isfield(simoptions,'lowmemory')
        simoptions.lowmemory=0; % setting to 1 slows the simulations, but reduces memory
    end
    if ~isfield(simoptions,'keepoutputasmatrix')
        simoptions.keepoutputasmatrix=0;
    end
    % When calling as a subcommand, the following is used internally
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0;
    end
    if ~isfield(simoptions, 'alreadygridvals')
        simoptions.alreadygridvals=0;
    end
end

%%
if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_daprime=l_d+l_a; % Does not yet handle anything but basics

if ndims(pi_z)==3
    pi_z_J=pi_z;
else
    pi_z_J=repelem(pi_z,1,1,N_j);
end

%% Simulate Panel Indexes
d_grid=gather(d_grid);
a_grid=gather(a_grid);

if isfield(simoptions,'n_semiz')
    simoptions.Parameters=Parameters; % Need to be able to pass a copy of this to SimPanelIndexes
end
PolicyIndexes=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z, N_j,simoptions); % Create it here as want it both here and inside SimPanelIndexes_FHorz_Case1 (which will recognise that it is already in this form)
PolicyIndexes=gather(PolicyIndexes);

simoptions.simpanelindexkron=1; % Keep the output as kron form as will want this later anyway for assigning the values
SimPanelIndexes=SimPanelIndexes_FHorz_Case1(InitialDist,PolicyIndexes,n_d,n_a,n_z,N_j,pi_z_J, simoptions);


%% Exogenous shock grids (must come after the SimPanelIndexes as it then strips n_semiz and n_e out of simoptions)
% Create the combination of (semiz,z,e) as all three are the same for FnsToEvaluate 
[n_z,z_gridvals_J,N_z,l_z,simoptions]=CreateGridvals_FnsToEvaluate_FHorz(n_z,z_grid,N_j,simoptions,Parameters);

% Note: semiz, z and e are from here on all just rolled together in n_z, z_gridvals_J, N_z and l_z

%% Implement new way of handling FnsToEvaluate

% Note: l_z includes e and semiz (when appropriate)
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    FnsToEvalNames=fieldnames(FnsToEvaluate);
    for ff=1:length(FnsToEvalNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(FnsToEvalNames{ff}));
        if length(temp)>(l_daprime+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_daprime+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
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
N_a=prod(n_a);

a_gridvals=CreateGridvals(n_a,a_grid,1); % 1 at end indicates output as matrices.

% Note that dPolicy and aprimePolicy will depend on age
if n_d(1)==0
    daprimePolicy_gridvals=zeros(N_a*N_z,l_a,N_j);
else
    daprimePolicy_gridvals=zeros(N_a*N_z,l_d+l_a,N_j); % Note: N_e=1 if no e variables
end

for jj=1:N_j    
    if ~isfield(simoptions,'n_e')
        if n_d(1)==0
            [dPolicy_gridvals_j,aprimePolicy_gridvals_j]=CreateGridvals_PolicyKron(PolicyIndexes(:,:,jj),n_d,n_a,n_a,n_z,d_grid,a_grid,1, 1);
        else
            [dPolicy_gridvals_j,aprimePolicy_gridvals_j]=CreateGridvals_PolicyKron(PolicyIndexes(:,:,:,jj),n_d,n_a,n_a,n_z,d_grid,a_grid,1, 1);            
        end
    else
        if n_d(1)==0
            [dPolicy_gridvals_j,aprimePolicy_gridvals_j]=CreateGridvals_PolicyKron(reshape(PolicyIndexes(:,:,:,jj),[N_a,N_z]),n_d,n_a,n_a,n_z,d_grid,a_grid,1, 1);
        else
            [dPolicy_gridvals_j,aprimePolicy_gridvals_j]=CreateGridvals_PolicyKron(reshape(PolicyIndexes(:,:,:,:,jj),[2,N_a,N_z]),n_d,n_a,n_a,n_z,d_grid,a_grid,1, 1);
        end
    end
    if n_d(1)==0
        daprimePolicy_gridvals(:,:,jj)=aprimePolicy_gridvals_j;
    else
        daprimePolicy_gridvals(:,:,jj)=[dPolicy_gridvals_j, aprimePolicy_gridvals_j];
    end
end

%% Now switch everything to gpu so can use arrayfun() to evaluates all the FnsToEvaluate
daprimePolicy_gridvals=gpuArray(daprimePolicy_gridvals);
SimPanelIndexes=gpuArray(SimPanelIndexes);

SimPanelValues=nan(length(FnsToEvaluate), N_j, simoptions.numbersims,'gpuArray'); % needs to be NaN to permit that some people might be 'born' later than age j=1
% Note, having the whole N_j at this stage makes assiging the values based on the indexes vastly faster



%% For sure the following could be made faster by improving how I do it

for jj=1:N_j
    SimPanelIndexes_jj=SimPanelIndexes(:,jj,:);

    relevantindices=(~isnan(SimPanelIndexes_jj(1,1,:))); % Note, is just across the ii dimension
    sumrelevantindices=sum(relevantindices);

    if sumrelevantindices>0 % Does the simulation even contain anyone of age jj?
        currentPanelIndexes_jj=SimPanelIndexes_jj(:,1,relevantindices);
        currentPanelValues_jj=zeros(sumrelevantindices,nFnsToEvalute); % transpose will be taken before storing

        az_ind=squeeze(currentPanelIndexes_jj(1,1,:)+N_a*(currentPanelIndexes_jj(2,1,:)-1));
        % a_ind=currentPanelIndexes_jj(1,1,:);
        % z_ind=currentPanelIndexes_jj(2,1,:);
        % j_ind=currentPanelIndexes_jj(3,1,:);

        a_val=a_gridvals(currentPanelIndexes_jj(1,1,:),:); % a_grid does depend on age
        z_val=z_gridvals_J(currentPanelIndexes_jj(2,1,:),:,jj);

        for vv=1:nFnsToEvalute
            if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'FnsToEvaluateParamNames={}'
                ParamCell={};
            else
                ValuesFnParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,jj);
                ParamCell=num2cell(ValuesFnParamsVec);
            end
            daprime_val=daprimePolicy_gridvals(az_ind,:,jj);
            currentPanelValues_jj(:,vv)=EvalFnOnSimPanelIndex(FnsToEvaluate{vv},ParamCell,daprime_val,a_val,z_val,l_daprime,l_a,l_z);
        end
        SimPanelValues(:,jj,relevantindices)=reshape(currentPanelValues_jj',[nFnsToEvalute,1,sumrelevantindices]);

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



