function StationaryDist_treatment=StationaryDist_FHorz_Case1_FieldExp_Treatment(StationaryDist_Control, AgeWeightParamNames, Policy,n_d,n_a,n_z,N_j,pi_z,Parameters, TreatmentAgeRange, TreatmentDuration,simoptions)
% The agent distribution of the control-group is effectively the 'initial' distribution from which 
% the treatment-group are drawn.
% The treatment group agent distribution does not include the information about the 'age weights' 
% because these are kept in the control group.

if exist('simoptions','var')==0
    simoptions.nsims=10^4;
    simoptions.parallel=3-(gpuDeviceCount>0); % 3 (sparse) if cpu, 2 if gpu
    simoptions.verbose=0;
    try 
        PoolDetails=gcp;
        simoptions.ncores=PoolDetails.NumWorkers;
    catch
        simoptions.ncores=1;
    end
    simoptions.iterate=1;
    simoptions.tolerance=10^(-9);
    simoptions.outputkron=0; % If 1 then leave output in Kron form
else
    %Check simoptions for missing fields, if there are some fill them with
    %the defaults
    if isfield(simoptions,'tolerance')==0
        simoptions.tolerance=10^(-9);
    end
    if isfield(simoptions,'nsims')==0
        simoptions.nsims=10^4;
    end
    if isfield(simoptions,'parallel')==0
        simoptions.parallel=3-(gpuDeviceCount>0); % 3 (sparse) if cpu, 2 if gpu
    end
    if isfield(simoptions,'verbose')==0
        simoptions.verbose=0;
    end
    if isfield(simoptions,'ncores')==0
        try
            PoolDetails=gcp;
            simoptions.ncores=PoolDetails.NumWorkers;
        catch
            simoptions.ncores=1;
        end
    end
    if isfield(simoptions,'iterate')==0
        simoptions.iterate=1;
    end
    if isfield(simoptions,'ExogShockFn') % If using ExogShockFn then figure out the parameter names
        simoptions.ExogShockFnParamNames=getAnonymousFnInputNames(simoptions.ExogShockFn);
    end
    if isfield(simoptions,'EiidShockFn') % If using ExogShockFn then figure out the parameter names
        simoptions.EiidShockFnParamNames=getAnonymousFnInputNames(simoptions.EiidShockFn);
    end
    if isfield(simoptions,'outputkron')==0
        simoptions.outputkron=0; % If 1 then leave output in Kron form
    end
end

%%
N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=0;
if isfield(simoptions,'n_e')
    N_e=prod(simoptions.n_e);
end
if N_z>0 && N_e>0
    N_ze=N_z*N_e;
else
    N_ze=N_z+N_e;
end
if N_ze>0
    StationaryDist_Control=reshape(StationaryDist_Control,[N_a,N_ze,N_j]);
    StationaryDist_treatment=zeros(N_a,N_ze,TreatmentDuration,TreatmentAgeRange(2)-TreatmentAgeRange(1)+1); % Note: temp(1:end-1) is going to be all the dimensions of that agent dist except age
    if N_e==0
        PolicyKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z,N_j);
    else
        PolicyKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z,N_j,simoptions.n_e);
    end
else
    StationaryDist_Control=reshape(StationaryDist_Control,[N_a,N_j]);
    StationaryDist_treatment=zeros(N_a,TreatmentDuration,TreatmentAgeRange(2)-TreatmentAgeRange(1)+1); % Note: temp(1:end-1) is going to be all the dimensions of that agent dist except age
    PolicyKron=KronPolicyIndexes_FHorz_Case1_noz(Policy, n_d, n_a,N_j);
end

if simoptions.parallel~=2 && simoptions.parallel~=4
    Policy=gather(Policy);
    StationaryDist_Control=gather(StationaryDist_Control);    
    pi_z=gather(pi_z);
end

%% Just in case z depends on j we need to create pi_z_J and then pass only the relevant part
if isfield(simoptions,'pi_z_J')
    pi_z_J=simoptions.pi_z_J;
elseif isfield(simoptions,'ExogShockFn')
    pi_z_J=zeros(N_z,N_z,N_j);
    for jj=1:N_j
        if isfield(simoptions,'ExogShockFnParamNames')
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
            ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
            for ii=1:length(ExogShockFnParamsVec)
                ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
            end
            [~,pi_z]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
        else
            [~,pi_z]=simoptions.ExogShockFn(jj);
        end
        pi_z_J(:,:,jj)=gather(pi_z);
    end
else
    % This is just so that it makes things easier below because I can
    % take it as given that pi_z_J and z_grid_J exist
    pi_z_J=zeros(N_z,N_z,N_j);
    for jj=1:N_j
        pi_z_J(:,:,jj)=pi_z;
    end
end
% And same for e if that is used
if N_e>0
    if isfield(simoptions,'pi_e_J')
        pi_e_J=simoptions.pi_e_J;
    elseif isfield(simoptions,'EiidShockFn')
        pi_e_J=zeros(N_e,N_j);
        for jj=1:N_j
            if isfield(simoptions,'EiidShockFnParamNames')
                EiidShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.EiidShockFnParamNames,jj);
                EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                for ii=1:length(EiidShockFnParamsVec)
                    EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                end
                [~,pi_e]=simoptions.EiidShockFn(EiidShockFnParamsCell{:});
            else
                [~,pi_e]=simoptions.EiidShockFn(jj);
            end
            pi_e_J(:,jj)=gather(pi_e);
        end
    else
        % This is just so that it makes things easier below because I can
        % take it as given that pi_e_J and e_grid_J exist
        pi_e_J=zeros(N_e,N_j);
        for jj=1:N_j
            pi_e_J(:,jj)=simoptions.pi_e;
        end
    end
end

% Identify the age-dependent parameters so that we can easily use just the
% parts relevant to the treatment ages (might be needed due to age-dependent z or e set up as functions)
AgeDepParams=struct();
paramnames=fieldnames(Parameters);
for nn=1:length(paramnames)
    if length(Parameters.(paramnames{nn}))==N_j
        AgeDepParams.(paramnames{nn})=Parameters.(paramnames{nn});
    end
end
agedepparamnames=fieldnames(AgeDepParams);

%%

for j_p=TreatmentAgeRange(1):TreatmentAgeRange(2)
    % Pull the appropraite initial distribution of agents
    if N_ze==0
        jequaloneDistKron=StationaryDist_Control(:,j_p);
        PolicyKron_treat=PolicyKron(:,:,j_p:j_p+TreatmentDuration-1);
    else
        jequaloneDistKron=StationaryDist_Control(:,:,j_p);
        if N_z==0 % just e
            PolicyKron_treat=PolicyKron(:,:,:,j_p:j_p+TreatmentDuration-1);
        elseif N_e==0 % Just z
            PolicyKron_treat=PolicyKron(:,:,:,j_p:j_p+TreatmentDuration-1);
        else % z and e
            PolicyKron_treat=PolicyKron(:,:,:,:,j_p:j_p+TreatmentDuration-1);
        end
    end
    % Normalize the mass of this initial distribution to one
    jequaloneDistKron=jequaloneDistKron./sum(jequaloneDistKron(:));
    
    % Replace all age dependent parameters with those required
    for nn=1:length(agedepparamnames)
        temp=AgeDepParams.(agedepparamnames{nn});
        Parameters.(agedepparamnames{nn})=temp(j_p:j_p+TreatmentDuration-1);
    end
    % Replace age weights with the appropriate age weights.
    Parameters.(AgeWeightParamNames{1})=ones(1,TreatmentDuration)/TreatmentDuration; % Note: This is ones() because it is inside the j_p loop. It is not the final age weights of the treatment group. (This is effectively assuming zero attrition in the field experiment.)

    % Create the agent dist for the current j_p
    if N_ze==0
        StationaryDist_treatment(:,:,j_p-TreatmentAgeRange(1)+1)=StationaryDist_FHorz_Case1_Iteration_noz_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron_treat,N_d,N_a,TreatmentDuration,Parameters,simoptions);
    else
        if N_z>0
            simoptions.pi_z_J=pi_z_J(:,:,j_p:j_p+TreatmentDuration);
        end
        if N_e>0
            simoptions.pi_e_J=pi_e_J(:,j_p:j_p+TreatmentDuration);
        end
        if N_e==0
            StationaryDist_treatment(:,:,:,j_p)=reshape(StationaryDist_FHorz_Case1_Iteration_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron_treat,N_d,N_a,N_z,TreatmentDuration,pi_z,Parameters,simoptions),[N_a,N_z,TreatmentDuration]);
        elseif N_z==0
            StationaryDist_treatment(:,:,:,j_p)=reshape(StationaryDist_FHorz_Case1_Iteration_noz_e_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron_treat,N_d,N_a,N_e,TreatmentDuration,simoptions.pi_e,Parameters,simoptions),[N_a,N_e,TreatmentDuration]); % e but no z
        else
            StationaryDist_treatment(:,:,:,j_p)=reshape(StationaryDist_FHorz_Case1_Iteration_e_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron_treat,N_d,N_a,N_z,N_e,TreatmentDuration,pi_z,simoptions.pi_e,Parameters,simoptions),[N_a,N_z*N_e,TreatmentDuration]);
        end
    end
    
end

% UnKron the output
if simoptions.outputkron==0
    if N_z>0 && N_e>0
        StationaryDist_treatment=reshape(StationaryDist_treatment,[n_a,n_z,n_e,TreatmentDuration,TreatmentAgeRange(2)-TreatmentAgeRange(1)+1]);
    elseif N_z>0
        StationaryDist_treatment=reshape(StationaryDist_treatment,[n_a,n_z,TreatmentDuration,TreatmentAgeRange(2)-TreatmentAgeRange(1)+1]);
    elseif N_e>0
        StationaryDist_treatment=reshape(StationaryDist_treatment,[n_a,n_e,TreatmentDuration,TreatmentAgeRange(2)-TreatmentAgeRange(1)+1]);
    else
        StationaryDist_treatment=reshape(StationaryDist_treatment,[n_a,TreatmentDuration,TreatmentAgeRange(2)-TreatmentAgeRange(1)+1]);
    end
end

end
