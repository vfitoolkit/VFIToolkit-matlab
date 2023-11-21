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
end

%%
if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end

l_a=length(n_a);
N_z=prod(n_z);
if N_z>0
    l_z=length(n_z);
else
    l_z=0;
end

l_daprime=l_d+l_a; % Does not yet handle anything but basics


%% Exogenous shock grids

% Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
% Gradually rolling these out so that all the commands build off of these
z_gridvals_J=zeros(prod(n_z),length(n_z),'gpuArray');
pi_z_J=zeros(prod(n_z),prod(n_z),'gpuArray');
if prod(n_z)==0 % no z
    z_gridvals_J=[];
elseif ndims(z_grid)==3 % already an age-dependent joint-grid
    if all(size(z_grid)==[prod(n_z),length(n_z),N_j])
        z_gridvals_J=z_grid;
    end
    pi_z_J=pi_z;
elseif all(size(z_grid)==[sum(n_z),N_j]) % age-dependent grid
    for jj=1:N_j
        z_gridvals_J(:,:,jj)=CreateGridvals(n_z,z_grid(:,jj),1);
    end
    pi_z_J=pi_z;
elseif all(size(z_grid)==[prod(n_z),length(n_z)]) % joint grid
    z_gridvals_J=z_grid.*ones(1,1,N_j,'gpuArray');
    pi_z_J=pi_z.*ones(1,1,N_j,'gpuArray');
elseif all(size(z_grid)==[sum(n_z),1]) % basic grid
    z_gridvals_J=CreateGridvals(n_z,z_grid,1).*ones(1,1,N_j,'gpuArray');
    pi_z_J=pi_z.*ones(1,1,N_j,'gpuArray');
end
if isfield(simoptions,'ExogShockFn')
    if isfield(simoptions,'ExogShockFnParamNames')
        for jj=1:N_j
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
            ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
            for ii=1:length(ExogShockFnParamsVec)
                ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
            end
            [z_grid,pi_z]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
            pi_z_J(:,jj)=gpuArray(pi_z);
            if all(size(z_grid)==[sum(n_z),1])
                z_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(n_z,z_grid,1));
            else % already joint-grid
                z_gridvals_J(:,:,jj)=gpuArray(z_grid,1);
            end
        end
    else
        for jj=1:N_j
            [z_grid,pi_z]=simoptions.ExogShockFn(N_j);
            pi_z_J(:,jj)=gpuArray(pi_z);
            if all(size(z_grid)==[sum(n_z),1])
                z_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(n_z,z_grid,1));
            else % already joint-grid
                z_gridvals_J(:,:,jj)=gpuArray(z_grid,1);
            end
        end
    end
end

% If using e variable, do same for this
if isfield(simoptions,'n_e')
    n_e=simoptions.n_e;
    N_e=prod(n_e);
    if isfield(simoptions,'n_e')
        if prod(simoptions.n_e)==0
            simoptions=rmfield(simoptions,'n_e');
        else
            if isfield(simoptions,'e_grid_J')
                error('No longer use simoptions.e_grid_J, instead just put the age-dependent grid in simoptions.e_grid (functionality of VFI Toolkit has changed to make it easier to use)')
            end
            if ~isfield(simoptions,'e_grid') % && ~isfield(simoptions,'e_grid_J')
                error('You are using an e (iid) variable, and so need to declare simoptions.e_grid')
            elseif ~isfield(simoptions,'pi_e')
                error('You are using an e (iid) variable, and so need to declare simoptions.pi_e')
            end

            e_gridvals_J=zeros(prod(simoptions.n_e),length(simoptions.n_e),'gpuArray');
            simoptions.pi_e_J=zeros(prod(simoptions.n_e),prod(simoptions.n_e),'gpuArray');
            if ndims(simoptions.e_grid)==3 % already an age-dependent joint-grid
                if all(size(simoptions.e_grid)==[prod(simoptions.n_e),length(simoptions.n_e),N_j])
                    e_gridvals_J=simoptions.e_grid;
                end
                simoptions.pi_e_J=simoptions.pi_e;
            elseif all(size(simoptions.e_grid)==[sum(simoptions.n_e),N_j]) % age-dependent stacked-grid
                for jj=1:N_j
                    e_gridvals_J(:,:,jj)=CreateGridvals(simoptions.n_e,simoptions.e_grid(:,jj),1);
                end
                simoptions.pi_e_J=simoptions.pi_e;
            elseif all(size(simoptions.e_grid)==[prod(simoptions.n_e),length(simoptions.n_e)]) % joint grid
                e_gridvals_J=simoptions.e_grid.*ones(1,1,N_j,'gpuArray');
                simoptions.pi_e_J=simoptions.pi_e.*ones(1,N_j,'gpuArray');
            elseif all(size(simoptions.e_grid)==[sum(simoptions.n_e),1]) % basic grid
                e_gridvals_J=CreateGridvals(simoptions.n_e,simoptions.e_grid,1).*ones(1,1,N_j,'gpuArray');
                simoptions.pi_e_J=simoptions.pi_e.*ones(1,N_j,'gpuArray');
            end
            if isfield(simoptions,'ExogShockFn')
                if isfield(simoptions,'ExogShockFnParamNames')
                    for jj=1:N_j
                        ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                        ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                        for ii=1:length(ExogShockFnParamsVec)
                            ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                        end
                        [simoptions.e_grid,simoptions.pi_e]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
                        simoptions.pi_e_J(:,jj)=gpuArray(simoptions.pi_e);
                        if all(size(simoptions.e_grid)==[sum(simoptions.n_e),1])
                            e_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(simoptions.n_e,simoptions.e_grid,1));
                        else % already joint-grid
                            e_gridvals_J(:,:,jj)=gpuArray(simoptions.e_grid,1);
                        end
                    end
                else
                    for jj=1:N_j
                        [simoptions.e_grid,simoptions.pi_e]=simoptions.ExogShockFn(N_j);
                        simoptions.pi_e_J(:,jj)=gpuArray(simoptions.pi_e);
                        if all(size(simoptions.e_grid)==[sum(simoptions.n_e),1])
                            e_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(simoptions.n_e,simoptions.e_grid,1));
                        else % already joint-grid
                            e_gridvals_J(:,:,jj)=gpuArray(simoptions.e_grid,1);
                        end
                    end
                end
            end
        end
    end
end

if isfield(simoptions,'n_e') % Note: N_z==0 is dealt with elsewhere
    if N_e>0
        n_ze=[n_z,n_e];
        l_ze=length(n_ze);
        N_ze=prod(n_ze);
    else
        n_ze=n_z;
        l_ze=l_z;
        N_ze=N_z;
    end
else
    n_ze=n_z;
    l_ze=length(n_z);
    N_ze=N_z;
end

%%
d_grid=gather(d_grid);
a_grid=gather(a_grid);

if isfield(simoptions,'n_semiz')
    simoptions.Parameters=Parameters; % Need to be able to pass a copy of this to SimPanelIndexes
    if N_z>0
        if isfield(simoptions,'n_e')
            PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, [simoptions.n_semiz,n_z], N_j,simoptions.n_e); % Create it here as want it both here and inside SimPanelIndexes_FHorz_Case1 (which will recognise that it is already in this form)
        else
            PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, [simoptions.n_semiz,n_z], N_j); % Create it here as want it both here and inside SimPanelIndexes_FHorz_Case1 (which will recognise that it is already in this form)
        end
    else
        if isfield(simoptions,'n_e')
            PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, simoptions.n_semiz, N_j,simoptions.n_e); % Create it here as want it both here and inside SimPanelIndexes_FHorz_Case1 (which will recognise that it is already in this form)
        else
            PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, simoptions.n_semiz, N_j); % Create it here as want it both here and inside SimPanelIndexes_FHorz_Case1 (which will recognise that it is already in this form)
        end
    end
else
    if isfield(simoptions,'n_e')
        PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z, N_j,simoptions.n_e); % Create it here as want it both here and inside SimPanelIndexes_FHorz_Case1 (which will recognise that it is already in this form)
    else
        PolicyIndexesKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z, N_j); % Create it here as want it both here and inside SimPanelIndexes_FHorz_Case1 (which will recognise that it is already in this form)
    end
end
PolicyIndexesKron=gather(PolicyIndexesKron);

simoptions.simpanelindexkron=1; % Keep the output as kron form as will want this later anyway for assigning the values
SimPanelIndexes=SimPanelIndexes_FHorz_Case1(InitialDist,PolicyIndexesKron,n_d,n_a,n_z,N_j,pi_z_J, simoptions);

if isfield(simoptions,'n_semiz')
    semiz_gridvals_J=CreateGridvals(simoptions.n_semiz,simoptions.semiz_grid,1).*ones(1,1,N_j);
    N_semiz=prod(simoptions.n_semiz);
    % From here on, can just treat semiz as another z
    if N_ze>0
        n_ze=[n_ze,simoptions.n_semiz];
        z_gridvals_J=[repmat(semiz_gridvals_J,N_z,1),repelem(z_gridvals_J,N_semiz,1)];
    else
        n_ze=simoptions.n_semiz;
        z_gridvals_J=semiz_gridvals_J;
    end
    N_ze=prod(n_ze);
    l_ze=length(n_ze);
end


%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
        if length(temp)>(l_d+l_a+l_a+l_ze)
            FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_a+l_ze+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(AggVarNames{ff});
    end
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end

if isfield(simoptions,'outputasstructure')
    if simoptions.outputasstructure==1
        FnsToEvaluateStruct=1;
        AggVarNames=simoptions.AggVarNames;
    elseif simoptions.outputasstructure==0
        FnsToEvaluateStruct=0;
    end
end

numFnsToEvalute=length(FnsToEvaluate);


%% Precompute the gridvals vectors.
N_a=prod(n_a);

a_gridvals=CreateGridvals(n_a,a_grid,1); % 1 at end indicates output as matrices.

% Note that dPolicy and aprimePolicy will depend on age
if n_d(1)==0
    daprimePolicy_gridvals=zeros(N_a*N_ze,l_a,N_j);
else
    daprimePolicy_gridvals=zeros(N_a*N_ze,l_d+l_a,N_j); % Note: N_e=1 if no e variables
end

for jj=1:N_j    
    if ~isfield(simoptions,'n_e')
        if n_d(1)==0
            [dPolicy_gridvals_j,aprimePolicy_gridvals_j]=CreateGridvals_PolicyKron(PolicyIndexesKron(:,:,jj),n_d,n_a,n_a,n_ze,d_grid,a_grid,1, 1);
        else
            [dPolicy_gridvals_j,aprimePolicy_gridvals_j]=CreateGridvals_PolicyKron(PolicyIndexesKron(:,:,:,jj),n_d,n_a,n_a,n_ze,d_grid,a_grid,1, 1);            
        end
    else
        if n_d(1)==0
            [dPolicy_gridvals_j,aprimePolicy_gridvals_j]=CreateGridvals_PolicyKron(reshape(PolicyIndexesKron(:,:,:,jj),[N_a,N_ze]),n_d,n_a,n_a,n_ze,d_grid,a_grid,1, 1);
        else
            [dPolicy_gridvals_j,aprimePolicy_gridvals_j]=CreateGridvals_PolicyKron(reshape(PolicyIndexesKron(:,:,:,:,jj),[2,N_a,N_ze]),n_d,n_a,n_a,n_ze,d_grid,a_grid,1, 1);
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
if ~isfield(simoptions,'n_e')
    for jj=1:N_j
        SimPanelIndexes_jj=SimPanelIndexes(:,jj,:);

        relevantindices=(~isnan(SimPanelIndexes_jj(1,1,:))); % Note, is just across the ii dimension
        sumrelevantindices=sum(relevantindices);
        
        if sumrelevantindices>0 % Does the simulation even contain anyone of age jj?
            currentPanelIndexes_jj=SimPanelIndexes_jj(:,1,relevantindices);
            currentPanelValues_jj=zeros(sumrelevantindices,numFnsToEvalute); % transpose will be taken before storing

            az_ind=squeeze(currentPanelIndexes_jj(1,1,:)+N_a*(currentPanelIndexes_jj(2,1,:)-1));
            % a_ind=currentPanelIndexes_jj(1,1,:);
            % z_ind=currentPanelIndexes_jj(2,1,:);
            % j_ind=currentPanelIndexes_jj(3,1,:);

            a_val=a_gridvals(currentPanelIndexes_jj(1,1,:),:); % a_grid does depend on age
            z_val=z_gridvals_J(currentPanelIndexes_jj(2,1,:),:,jj);

            for vv=1:numFnsToEvalute
                if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'FnsToEvaluateParamNames={}'
                    ParamCell={};
                else
                    ValuesFnParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,jj);
                    ParamCell=num2cell(ValuesFnParamsVec);
                end
                daprime_val=daprimePolicy_gridvals(az_ind,:,jj);
                currentPanelValues_jj(:,vv)=EvalFnOnSimPanelIndex(FnsToEvaluate{vv},ParamCell,daprime_val,a_val,z_val,l_daprime,l_a,l_ze);
            end
            SimPanelValues(:,jj,relevantindices)=reshape(currentPanelValues_jj',[numFnsToEvalute,1,sumrelevantindices]);

        end
    end

else
    %% Using e variable
    for jj=1:N_j
        SimPanelIndexes_jj=SimPanelIndexes(:,jj,:);

        relevantindices=(~isnan(SimPanelIndexes_jj(1,1,:))); % Note, is just across the ii dimension
        sumrelevantindices=sum(relevantindices);

        if sumrelevantindices>0 % Does the simulation even contain anyone of age jj?
            currentPanelIndexes_jj=SimPanelIndexes_jj(:,1,relevantindices);
            currentPanelValues_jj=zeros(sumrelevantindices,numFnsToEvalute); % transpose will be taken before storing

            az_ind=squeeze(currentPanelIndexes_jj(1,1,:)+N_a*(currentPanelIndexes_jj(2,1,:)-1));
            % a_ind=currentPanelIndexes_jj(1,1,:);
            % z_ind=currentPanelIndexes_jj(2,1,:);
            % e_ind=currentPanelIndexes_jj(3,1,:);
            % j_ind=currentPanelIndexes_jj(4,1,:);

            a_val=a_gridvals(currentPanelIndexes_jj(1,1,:),:); % a_grid does depend on age
            z_val=z_gridvals_J(currentPanelIndexes_jj(2,1,:),:,jj);
            e_val=e_gridvals_J(currentPanelIndexes_jj(3,1,:),:,jj);
            
            for vv=1:numFnsToEvalute
                if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'FnsToEvaluateParamNames={}'
                    ParamCell={};
                else
                    ValuesFnParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,jj);
                    ParamCell=num2cell(ValuesFnParamsVec);
                end
                daprime_val=daprimePolicy_gridvals(az_ind,:,jj);
                currentPanelValues_jj(:,vv)=EvalFnOnSimPanelIndex(FnsToEvaluate{vv},ParamCell,daprime_val,a_val,[z_val,e_val],l_daprime,l_a,l_ze);
            end
            SimPanelValues(:,jj,relevantindices)=reshape(currentPanelValues_jj',[numFnsToEvalute,1,sumrelevantindices]);
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
%     AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        SimPanelValues.(AggVarNames{ff})=shiftdim(SimPanelValues2(ff,:,:),1);
    end
end



end



