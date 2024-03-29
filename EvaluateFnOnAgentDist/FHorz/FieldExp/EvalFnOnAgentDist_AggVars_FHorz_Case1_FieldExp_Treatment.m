function AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1_FieldExp_Treatment(StationaryDist,PolicyIndexes, FnsToEvaluate,Parameters,FnsToEvaluateParamNames,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,Parallel, TreatmentAgeRange, TreatmentDuration, TreatmentAgeWeights, simoptions)
% Parallel is an optional input. If not given, will guess based on where StationaryDist
% Note: the input parameters should be those for the treatment group

if isempty(n_d)
    n_d=0;
    l_d=0;
elseif n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);
N_a=prod(n_a);
N_z=prod(n_z);

if ~exist('Parallel','var')
    if isa(StationaryDist, 'gpuArray')
        Parallel=2;
    else
        Parallel=1;
    end
else
    if isempty(Parallel)
        if isa(StationaryDist, 'gpuArray')
            Parallel=2;
        else
            Parallel=1;
        end
    end
end

if ~exist('simoptions','var')
    simoptions.lowmemory=0;
else
    if ~isfield(simoptions,'lowmemory')
        simoptions.lowmemory=0;
    end
end

%%
if N_z==0
    dbstack
    error('Not yet implemented AggVars for Field Exp with no z variable')
%    AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1_noz(StationaryDist,PolicyIndexes, FnsToEvaluate,Parameters,FnsToEvaluateParamNames, n_d,n_a,N_j,d_grid,a_grid,Parallel,simoptions);
%    return
end

%% This implementation is slightly inefficient when shocks are not age dependent, but speed loss is fairly trivial
if isfield(simoptions,'ExogShockFn') % If using ExogShockFn then figure out the parameter names
    simoptions.ExogShockFnParamNames=getAnonymousFnInputNames(simoptions.ExogShockFn);
end
if isfield(simoptions,'EiidShockFn') % If using ExogShockFn then figure out the parameter names
    simoptions.EiidShockFnParamNames=getAnonymousFnInputNames(simoptions.EiidShockFn);
end
eval('fieldexists_ExogShockFn=1;simoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;simoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')
eval('fieldexists_pi_z_J=1;simoptions.pi_z_J;','fieldexists_pi_z_J=0;')

if fieldexists_pi_z_J==1
    z_grid_J=simoptions.z_grid_J;
elseif fieldexists_ExogShockFn==1
    z_grid_J=zeros(sum(n_z),N_j);
    for jj=1:N_j
        if fieldexists_ExogShockFnParamNames==1
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
            ExogShockFnParamsCell=num2cell(ExogShockFnParamsVec);
            [z_grid,~]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
        else
            [z_grid,~]=simoptions.ExogShockFn(jj);
        end
        z_grid_J(:,jj)=z_grid;
    end
else
    if all(size(z_grid)==[sum(n_z),1])
        z_grid_J=repmat(z_grid,1,N_j);
    else % Joint-grid on shocks
        z_grid_J=repmat(z_grid,1,1,N_j);
    end
end
if Parallel==2
    z_grid_J=gpuArray(z_grid_J);
end
if ndims(z_grid_J)==2
    jointgridz=0;
else % Joint-grid on shocks
    jointgridz=1;
end

if isfield(simoptions,'n_e')
    % Because of how FnsToEvaluate works I can just get the e variables and then 'combine' them with z
    eval('fieldexists_EiidShockFn=1;simoptions.EiidShockFn;','fieldexists_EiidShockFn=0;')
    eval('fieldexists_EiidShockFnParamNames=1;simoptions.EiidShockFnParamNames;','fieldexists_EiidShockFnParamNames=0;')
    eval('fieldexists_pi_e_J=1;simoptions.pi_e_J;','fieldexists_pi_e_J=0;')
    
    n_e=simoptions.n_e;
    N_e=prod(n_e);
    l_e=length(n_e);
    
    if fieldexists_pi_e_J==1
        e_grid_J=simoptions.e_grid_J;
    elseif fieldexists_EiidShockFn==1
        e_grid_J=zeros(sum(simoptions.n_e),N_j);
        for jj=1:N_j
            if fieldexists_EiidShockFnParamNames==1
                EiidShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.EiidShockFnParamNames,jj);
                EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                for ii=1:length(EiidShockFnParamsVec)
                    EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                end
                [e_grid,~]=simoptions.EiidShockFn(EiidShockFnParamsCell{:});
            else
                [e_grid,~]=simoptions.EiidShockFn(jj);
            end
            e_grid_J(:,jj)=gather(e_grid);
        end
    else
        if all(size(simoptions.e_grid)==[sum(n_e),1])
            e_grid_J=repmat(simoptions.e_grid,1,N_j);
        else % Joint-grid on shocks
            e_grid_J=repmat(simoptions.e_grid,1,1,N_j);
        end
    end
    if ndims(e_grid_J)==2
        jointgride=0;
    else % Joint-grid on shocks
        jointgride=1;
    end
    
    if simoptions.lowmemory==1
        % Keep them seperate
    else
        % Now combine into z
        if n_z(1)==0
            l_z=l_e;
            n_z=n_e;
            z_grid_J=e_grid_J;
        else
            % Need to allow for possibility that one of the other is using joint-grids
            if jointgridz==0 && jointgride==0 % neither uses joint grids
                z_grid_J=[z_grid_J; e_grid_J];
            elseif jointgridz==0 % just e is joint
                z_grid_J_temp=z_grid_J;
                z_grid_J=zeros(N_e*N_z,l_e+l_z,N_j);
                for jj=1:N_j
                    z_grid_J(:,:,jj)=[kron(ones(N_e,1),CreateGridvals(n_z,z_grid_J_temp(:,jj),1)),kron(e_grid_J(:,:,jj) ,ones(N_z,1))]; % replace e with the joint-grid version
                end
            elseif jointgride==0 % just z is joint
                z_grid_J_temp=z_grid_J;
                z_grid_J=zeros(N_e*N_z,l_e+l_z,N_j);
                for jj=1:N_j
                    z_grid_J(:,:,jj)=[kron(ones(N_e,1),z_grid_J_temp(:,:,jj)),kron(CreateGridvals(n_e,e_grid_J(:,jj),1),ones(N_z,1))]; % replace e with the joint-grid version
                end
            else % both joint grids
                z_grid_J_temp=z_grid_J;
                z_grid_J=zeros(N_e*N_z,l_e+l_z,N_j);
                for jj=1:N_j
                    z_grid_J(:,:,jj)=[kron(ones(N_e,1),z_grid_J_temp(:,:,jj)),kron(e_grid_J(:,:,jj),ones(N_z,1))];
                end
            end
            n_z=[n_z,n_e];
        end
        N_z=prod(n_z);
    end
else
    N_e=0; % Just needed to decide which case when using parallel=2
    n_ze=n_z;
end

%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    if simoptions.lowmemory==0
        l_z_temp=l_z+l_e;
    elseif simoptions.lowmemory==1
        l_z_temp=l_z;
    end
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
        if length(temp)>(l_d+l_a+l_a+l_z_temp)
            FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_a+l_z_temp+1:end}}; % the first inputs will always be (d,aprime,a,z)
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

%%
if N_e>0 && simoptions.lowmemory==1
    % Loop over e
    AggVars=zeros(length(FnsToEvaluate),N_e,'gpuArray');
    StationaryDistVec=permute(reshape(StationaryDist,[N_a*N_z,N_e,TreatmentDuration,TreatmentAgeRange(2)-TreatmentAgeRange(1)+1]),[1,4,3,2]); % permute moves e to last dimension,  permute swaps treatment duration and treatment initial age

    % Set up policy in way that makes it easy to loop over later (just
    % keeping the treatment ages, and switching from indexes to values)
    PolicyValuesPermuteVec=struct();
    if n_d(1)>0
        dasize=l_d+l_a; %size(PolicyIndexes.(['treatmentage',num2str(TreatmentAgeRange(1))]),1);
    else
        dasize=l_a;
    end
    permuteindexes=[5,4,2,3,1]; %permute into [N_j,N_e,N_a,N_z,l_d+l_a]
    for j_p=1:(TreatmentAgeRange(2)-TreatmentAgeRange(1)+1)
        PolicyIndexes_temp=reshape(PolicyIndexes.(['treatmentage',num2str(j_p)]),[dasize,N_a,N_z*N_e,N_j]);
        PolicyValues=PolicyInd2Val_FHorz_Case1(PolicyIndexes_temp(:,:,:,j_p:j_p+TreatmentDuration-1),n_d,n_a,[n_z,n_e],TreatmentDuration,d_grid,a_grid); % Note PolicyIndexes input is the wrong shape, but because this is parellel=2 the first thing PolicyInd2Val does is anyway to reshape() it.
        for tt=1:TreatmentDuration
            PolicyValues=reshape(PolicyValues,[dasize,N_a,N_z,N_e,TreatmentDuration]);
            PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[N_j,N_e,N_a,N_z,l_d+l_a]
            for e_c=1:N_e
                PolicyValuesPermuteVec.(['treatmentage',num2str(j_p),'tt',num2str(tt),'ee',num2str(e_c)])=reshape(PolicyValuesPermute(tt,e_c,:,:,:),[n_a,n_z,l_d+l_a]);
            end
        end
    end

    for e_c=1:N_e
        StationaryDistVec_e=StationaryDistVec(:,:,:,e_c); % This is why I did the permute (to avoid a reshape here). Not actually sure whether all the reshapes would be faster than the permute or not?

        for ii=1:length(FnsToEvaluate)
            Values=nan(N_a*N_z,TreatmentAgeRange(2)-TreatmentAgeRange(1)+1,TreatmentDuration,'gpuArray');
            for j_p=1:(TreatmentAgeRange(2)-TreatmentAgeRange(1)+1)
                for tt=1:TreatmentDuration
                    jj=j_p+tt-1;
                    PolicyValuesPermuteVec_temp=PolicyValuesPermuteVec.(['treatmentage',num2str(j_p),'tt',num2str(tt),'ee',num2str(e_c)]);

                    if jointgridz==1
                        z_grid=z_grid_J(:,:,jj);
                    else
                        z_grid=z_grid_J(:,jj);
                    end
                    if jointgride==1
                        e_val=e_grid_J(e_c,:,jj);
                    else
                        e_val=e_grid_J(e_c,jj);
                    end

                    % Includes check for cases in which no parameters are actually required
                    if isempty(FnsToEvaluateParamNames(ii).Names) % || strcmp(FnsToEvaluateParamNames(1),'')) % check for 'FnsToEvaluateParamNames={}'
                        FnToEvaluateParamsVec=[];
                    else
                        FnToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj));
                    end
                    Values(:,j_p,tt)=reshape(EvalFnOnAgentDist_Grid_Case1(FnsToEvaluate{ii}, [FnToEvaluateParamsVec,e_val],PolicyValuesPermuteVec_temp,n_d,n_a,n_z,a_grid,z_grid,Parallel),[N_a*N_z,1]);
                end
            end

            AggVars(ii,e_c)=sum(TreatmentAgeWeights.*sum(sum(Values.*StationaryDistVec_e,1),3)); % sum over treatment period az & tt, then multiply by age weights, and then sum over j_p
        end
    end
    AggVars=sum(AggVars,2);

else % either no e vars, or just lowmemory=0
    AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');

    StationaryDistVec=reshape(StationaryDist,[N_a*N_z,TreatmentDuration,TreatmentAgeRange(2)-TreatmentAgeRange(1)+1]);
    StationaryDistVec=permute(reshape(StationaryDist,[N_a*N_z,TreatmentDuration,TreatmentAgeRange(2)-TreatmentAgeRange(1)+1]),[1,3,2]); % permute swaps treatment duration and treatment initial age

    % Set up policy in way that makes it easy to loop over later (just
    % keeping the treatment ages, and switching from indexes to values)
    PolicyValuesPermuteVec=struct();
    if n_d(1)>0
        dasize=l_d+l_a; %size(PolicyIndexes.(['treatmentage',num2str(TreatmentAgeRange(1))]),1);
    else
        dasize=l_a;
    end
    permuteindexes=[4,2,3,1]; %permute into [N_j,N_a,N_ze,l_d+l_a]
    for j_p=1:(TreatmentAgeRange(2)-TreatmentAgeRange(1)+1)
        PolicyIndexes_temp=reshape(PolicyIndexes.(['treatmentage',num2str(j_p)]),[dasize,N_a,N_z,N_j]);
        PolicyValues=PolicyInd2Val_FHorz_Case1(PolicyIndexes_temp(:,:,:,j_p:j_p+TreatmentDuration-1),n_d,n_a,n_z,TreatmentDuration,d_grid,a_grid); % Note PolicyIndexes input is the wrong shape, but because this is parellel=2 the first thing PolicyInd2Val does is anyway to reshape() it.
        for tt=1:TreatmentDuration
            PolicyValues=reshape(PolicyValues,[dasize,N_a,N_z,TreatmentDuration]);
            PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[N_j,N_a,N_ze,l_d+l_a]
            PolicyValuesPermuteVec.(['treatmentage',num2str(j_p),'tt',num2str(tt)])=reshape(PolicyValuesPermute(tt,:,:,:),[n_a,n_z,l_d+l_a]);
        end
    end

    for ii=1:length(FnsToEvaluate)
        Values=nan(N_a*N_z,TreatmentAgeRange(2)-TreatmentAgeRange(1)+1,TreatmentDuration,'gpuArray');
        for j_p=1:(TreatmentAgeRange(2)-TreatmentAgeRange(1)+1)
            for tt=1:TreatmentDuration
                jj=j_p+tt-1;
                PolicyValuesPermuteVec_temp=PolicyValuesPermuteVec.(['treatmentage',num2str(j_p),'tt',num2str(tt)]);

                if jointgridz==1
                    z_grid=z_grid_J(:,:,jj);
                else
                    z_grid=z_grid_J(:,jj);
                end

                % Includes check for cases in which no parameters are actually required
                if isempty(FnsToEvaluateParamNames(ii).Names) % || strcmp(FnsToEvaluateParamNames(1),'')) % check for 'FnsToEvaluateParamNames={}'
                    FnToEvaluateParamsVec=[];
                else
                    FnToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj));
                end
                Values(:,j_p,tt)=reshape(EvalFnOnAgentDist_Grid_Case1(FnsToEvaluate{ii}, FnToEvaluateParamsVec,PolicyValuesPermuteVec_temp,n_d,n_a,n_z,a_grid,z_grid,Parallel),[N_a*N_z,1]);
            end
        end

        AggVars(ii)=sum(TreatmentAgeWeights.*sum(sum(Values.*StationaryDistVec,1),3)); % sum over treatment period az & tt, then multiply by age weights, and then sum over j_p
    end
end



%%
if FnsToEvaluateStruct==1
    % Change the output into a structure
    AggVars2=AggVars;
    clear AggVars
    AggVars=struct();
%     AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        AggVars.(AggVarNames{ff}).Mean=AggVars2(ff);
    end
end


end
