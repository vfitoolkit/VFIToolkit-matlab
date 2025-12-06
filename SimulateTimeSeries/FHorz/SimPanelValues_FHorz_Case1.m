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
    simoptions.gridinterplayer=0;
    simoptions.keepoutputasmatrix=0;
    % When calling as a subcommand, the following is used internally
    simoptions.alreadygridvals=0;
    % Following are just for my convenience
    simoptions.n_semiz=0;
    simoptions.n_e=0;
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
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0;
    end
    if ~isfield(simoptions,'keepoutputasmatrix')
        simoptions.keepoutputasmatrix=0;
    end
    % When calling as a subcommand, the following is used internally
    if ~isfield(simoptions, 'alreadygridvals')
        simoptions.alreadygridvals=0;
    end
    % Following are just for my convenience
    if ~isfield(simoptions,'n_semiz')
        simoptions.n_semiz=0;
    end
    if ~isfield(simoptions,'n_e')
        simoptions.n_e=0;
    end
end

if isfield(simoptions,'ExogShockFn') % If using ExogShockFn then figure out the parameter names
    simoptions.ExogShockFnParamNames=getAnonymousFnInputNames(simoptions.ExogShockFn);
end

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

if ndims(pi_z)==3
    pi_z_J=pi_z;
else
    pi_z_J=repelem(pi_z,1,1,N_j);
end


%% Simulate Panel Indexes
d_grid=gather(d_grid);
a_grid=gather(a_grid);

if simoptions.n_e(1)==0
    if simoptions.n_semiz(1)==0
        n_semizze=n_z;
    else
        n_semizze=[simoptions.n_semiz,n_z];
    end
else
    if simoptions.n_semiz(1)==0
        n_semizze=[n_z,simoptions.n_e];
    else
        n_semizze=[simoptions.n_semiz,n_z,simoptions.n_e];
    end
end

if isfield(simoptions,'n_semiz')
    simoptions.Parameters=Parameters; % Need to be able to pass a copy of this to SimPanelIndexes
end
Policy=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_semizze, N_j,simoptions); % Create it here as want it both here and inside SimPanelIndexes_FHorz (which will recognise that it is already in this form)
Policy=gather(Policy);

simoptions.simpanelindexkron=1; % Keep the output as kron form as will want this later anyway for assigning the values
SimPanelIndexes=SimPanelIndexes_FHorz(InitialDist,Policy,n_d,n_a,n_z,N_j,pi_z_J, simoptions);

%% Exogenous shock grids (must come after the SimPanelIndexes as it then strips n_semiz and n_e out of simoptions)
% Create the combination of (semiz,z,e) as all three are the same for FnsToEvaluate 
[~,semizze_gridvals_J,~,~,simoptions]=CreateGridvals_FnsToEvaluate_FHorz(n_z,z_grid,N_j,simoptions,Parameters);
N_semizze=prod(n_semizze);
l_semizze=length(n_semizze);
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
if simoptions.gridinterplayer==0
    n_aprime=n_a;
    aprime_grid=a_grid;
else
    % Switch policy and aprime_grid to being on the fine grid
    Policy(l_d+1,:,:,:)=(simoptions.ngridinterp+1)*(Policy(l_d+1,:,:,:)-1)+Policy(end,:,:,:);
    Policy=Policy(1:end-1,:,:,:);
    if l_a==1
        n_aprime=1+(n_a-1)*(simoptions.ngridinterp+1);
        aprime_grid=interp1(1:simoptions.ngridinterp+1:n_aprime,a_grid,1:1:n_aprime)';
    else
        a1_grid=a_grid(1:n_a(1));
        n_a1=a1_grid;
        n_a1prime=1+(n_a1-1)*(simoptions.ngridinterp+1);
        a1prime_grid=interp1(1:simoptions.ngridinterp+1:n_a1prime,a1_grid,1:1:n_a1prime)';
        n_aprime=[n_a1,n_a(2:end)];
        aprime_grid=[a1prime_grid; a_grid(n_a(1)+1:end)];
    end
end


% Note that dPolicy and aprimePolicy will depend on age
if N_d==0
    daprimePolicy_gridvals=zeros(N_a*N_semizze,l_a,N_j);
else
    daprimePolicy_gridvals=zeros(N_a*N_semizze,l_d+l_a,N_j); % Note: N_e=1 if no e variables
end

for jj=1:N_j    
    if N_d==0
        [~,aprimePolicy_gridvals_j]=CreateGridvals_PolicyKron(Policy(:,:,jj),n_d,n_aprime,n_a,n_semizze,d_grid,aprime_grid,1, 1);
        daprimePolicy_gridvals(:,:,jj)=aprimePolicy_gridvals_j;
    else
        [dPolicy_gridvals_j,aprimePolicy_gridvals_j]=CreateGridvals_PolicyKron(Policy(:,:,:,jj),n_d,n_aprime,n_a,n_semizze,d_grid,aprime_grid,1, 1);
        daprimePolicy_gridvals(:,:,jj)=[dPolicy_gridvals_j, aprimePolicy_gridvals_j];
    end
end

a_gridvals=CreateGridvals(n_a,a_grid,1); % 1 at end indicates output as matrices.


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



