function AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist,PolicyIndexes, FnsToEvaluate,Parameters,FnsToEvaluateParamNames,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,Parallel,simoptions)
% Parallel is an optional input. If not given, will guess based on where StationaryDist

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

if Parallel==1
    AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1_cpu(StationaryDist,PolicyIndexes, FnsToEvaluate,Parameters,FnsToEvaluateParamNames,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);
end

%%
l_a=length(n_a);
N_a=prod(n_a);
N_z=prod(n_z);


%% Exogenous shock grids

% Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
% Gradually rolling these out so that all the commands build off of these
if N_z>0
    z_gridvals_J=zeros(prod(n_z),length(n_z),'gpuArray');
    if prod(n_z)==0 % no z
        z_gridvals_J=[];
    elseif ndims(z_grid)==3 % already an age-dependent joint-grid
        if all(size(z_grid)==[prod(n_z),length(n_z),N_j])
            z_gridvals_J=z_grid;
        end
    elseif all(size(z_grid)==[sum(n_z),N_j]) % age-dependent grid
        for jj=1:N_j
            z_gridvals_J(:,:,jj)=CreateGridvals(n_z,z_grid(:,jj),1);
        end
    elseif all(size(z_grid)==[prod(n_z),length(n_z)]) % joint grid
        z_gridvals_J=z_grid.*ones(1,1,N_j,'gpuArray');
    elseif all(size(z_grid)==[sum(n_z),1]) % basic grid
        z_gridvals_J=CreateGridvals(n_z,z_grid,1).*ones(1,1,N_j,'gpuArray');
    end
    if isfield(simoptions,'ExogShockFn')
        if isfield(simoptions,'ExogShockFnParamNames')
            for jj=1:N_j
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [z_grid,~]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
                if all(size(z_grid)==[sum(n_z),1])
                    z_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(n_z,z_grid,1));
                else % already joint-grid
                    z_gridvals_J(:,:,jj)=gpuArray(z_grid,1);
                end
            end
        else
            for jj=1:N_j
                [z_grid,~]=simoptions.ExogShockFn(N_j);
                if all(size(z_grid)==[sum(n_z),1])
                    z_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(n_z,z_grid,1));
                else % already joint-grid
                    z_gridvals_J(:,:,jj)=gpuArray(z_grid,1);
                end
            end
        end
    end
end

% If using e variable, do same for this
if isfield(simoptions,'n_e')
    n_e=simoptions.n_e;
    N_e=prod(n_e);
    if N_e==0
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
        if ndims(simoptions.e_grid)==3 % already an age-dependent joint-grid
            if all(size(simoptions.e_grid)==[prod(simoptions.n_e),length(simoptions.n_e),N_j])
                e_gridvals_J=simoptions.e_grid;
            end
        elseif all(size(simoptions.e_grid)==[sum(simoptions.n_e),N_j]) % age-dependent grid
            for jj=1:N_j
                e_gridvals_J(:,:,jj)=CreateGridvals(simoptions.n_e,simoptions.e_grid(:,jj),1);
            end
        elseif all(size(simoptions.e_grid)==[prod(simoptions.n_e),length(simoptions.n_e)]) % joint grid
            e_gridvals_J=simoptions.e_grid.*ones(1,1,N_j,'gpuArray');
        elseif all(size(simoptions.e_grid)==[sum(simoptions.n_e),1]) % basic grid
            e_gridvals_J=CreateGridvals(simoptions.n_e,simoptions.e_grid,1).*ones(1,1,N_j,'gpuArray');
        end
        if isfield(simoptions,'ExogShockFn')
            if isfield(simoptions,'ExogShockFnParamNames')
                for jj=1:N_j
                    ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for ii=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                    end
                    [simoptions.e_grid,~]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
                    if all(size(simoptions.e_grid)==[sum(simoptions.n_e),1])
                        e_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(simoptions.n_e,simoptions.e_grid,1));
                    else % already joint-grid
                        e_gridvals_J(:,:,jj)=gpuArray(simoptions.e_grid,1);
                    end
                end
            else
                for jj=1:N_j
                    [simoptions.e_grid,simoptions.pi_e]=simoptions.ExogShockFn(N_j);
                    if all(size(simoptions.e_grid)==[sum(simoptions.n_e),1])
                        e_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(simoptions.n_e,simoptions.e_grid,1));
                    else % already joint-grid
                        e_gridvals_J(:,:,jj)=gpuArray(simoptions.e_grid,1);
                    end
                end
            end
        end

        % Now put e into z as that is easiest way to handle it from now on
        if N_z==0
            z_gridvals_J=e_gridvals_J;
            n_z=n_e;
            N_z=prod(n_z);
        else
            z_gridvals_J=[repmat(z_gridvals_J,N_e,1),repelem(e_gridvals_J,N_z,1)];
            n_z=[n_z,n_e];
            N_z=prod(n_z);
        end
    end
end

% Also semiz if that is used
if isfield(simoptions,'SemiExoStateFn') % If using semi-exogenous shocks
    if N_z==0
        n_z=simoptions.n_semiz;
        z_gridvals_J=CreateGridvals(simoptions.n_semiz,simoptions.semiz_grid,1);
    else
        % For purposes of function evaluation we can just treat the semi-exogenous states as exogenous states
        n_z=[simoptions.n_semiz,n_z];
        z_gridvals_J=[repmat(CreateGridvals(simoptions.n_semiz,simoptions.semiz_grid,1).*ones(1,1,N_j,'gpuArray'),N_z,1),repelem(z_gridvals_J,prod(simoptions.n_semiz),1)];
    end
end
N_z=prod(n_z);
if N_z==0
    l_z=0;
else
    l_z=length(n_z);
end

%% Implement new way of handling FnsToEvaluate
% Figure out l_daprime from Policy
l_daprime=size(PolicyIndexes,1);

% Note: l_z includes e and semiz (when appropriate)
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
        if length(temp)>(l_daprime+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_daprime+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
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
if N_z==0
    if simoptions.lowmemory==0
        AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');

        StationaryDist=reshape(StationaryDist,[N_a,N_j]);

        PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,0,N_j,d_grid,a_grid,simoptions,1);
        PolicyValuesPermute=permute(PolicyValues,[2,3,1]); % (N_a,N_j,l_daprime)

        a_gridvals=CreateGridvals(n_a,a_grid,1);

        for ff=1:length(FnsToEvaluate)
            % Values=nan(N_a,N_z,N_j,'gpuArray');

            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(ff).Names)
                ParamCell=cell(0,1);
            else
                % Create a matrix containing all the return function parameters (in order).
                % Each column will be a specific parameter with the values at every age.
                FnToEvaluateParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, FnsToEvaluateParamNames(ff).Names,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

                nFnToEvaluateParams=size(FnToEvaluateParamsAgeMatrix,2);

                ParamCell=cell(nFnToEvaluateParams,1);
                for ii=1:nFnToEvaluateParams
                    ParamCell(ii,1)={shiftdim(FnToEvaluateParamsAgeMatrix(:,ii),-1)}; % (a,j,l_d+l_a), so we want j to be after N_a
                end
            end
            
            Values=EvalFnOnAgentDist_Grid_J(FnsToEvaluate{ff},ParamCell,PolicyValuesPermute,l_daprime,n_a,0,a_gridvals,[]);
            AggVars(ff)=sum(sum(sum(Values.*StationaryDist)));
        end

    elseif simoptions.lowmemory==1 % Loop over age j
        AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');

        StationaryDist=reshape(StationaryDist,[N_a,N_j]);

        PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,0,N_j,d_grid,a_grid,simoptions,1);

        for ii=1:length(FnsToEvaluate)
            Values=nan(N_a,N_j,'gpuArray');
            for jj=1:N_j

                % Includes check for cases in which no parameters are actually required
                if isempty(FnsToEvaluateParamNames(ii).Names) % || strcmp(FnsToEvaluateParamNames(1),'')) % check for 'FnsToEvaluateParamNames={}'
                    FnToEvaluateParamsVec=[];
                else
                    FnToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj));
                end
                Values(:,jj)=EvalFnOnAgentDist_Grid(FnsToEvaluate{ii}, FnToEvaluateParamsVec,PolicyValues(:,:,:,jj),l_daprime,n_a,0,a_grid,[]);
            end
            AggVars(ii)=sum(sum(Values.*StationaryDist));
        end
    end

else % N_z

    if simoptions.lowmemory==0
        AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');

        StationaryDist=reshape(StationaryDist,[N_a,N_z,N_j]);

        PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,1);
        PolicyValuesPermute=permute(PolicyValues,[2,3,4,1]); % (N_a,N_z,N_j,l_daprime)

        a_gridvals=CreateGridvals(n_a,a_grid,1);

        for ff=1:length(FnsToEvaluate)
            % Values=nan(N_a,N_z,N_j,'gpuArray');

            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(ff).Names)
                ParamCell=cell(0,1);
            else
                % Create a matrix containing all the return function parameters (in order).
                % Each column will be a specific parameter with the values at every age.
                FnToEvaluateParamsAgeMatrix=CreateAgeMatrixFromParams(Parameters, FnsToEvaluateParamNames(ff).Names,N_j); % this will be a matrix, row indexes ages and column indexes the parameters (parameters which are not dependent on age appear as a constant valued column)

                nFnToEvaluateParams=size(FnToEvaluateParamsAgeMatrix,2);

                ParamCell=cell(nFnToEvaluateParams,1);
                for ii=1:nFnToEvaluateParams
                    ParamCell(ii,1)={shiftdim(FnToEvaluateParamsAgeMatrix(:,ii),-2)}; % (a,z,j,l_d+l_a), so we want j to be after N_a and N_z
                end
            end

            Values=EvalFnOnAgentDist_Grid_J(FnsToEvaluate{ff},ParamCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals_J);
            AggVars(ff)=sum(sum(sum(Values.*StationaryDist)));
        end

    elseif simoptions.lowmemory==1 % Loop over age j
        AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');

        StationaryDist=reshape(StationaryDist,[N_a*N_z,N_j]);

        PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,1);

        for ii=1:length(FnsToEvaluate)
            Values=nan(N_a*N_z,N_j,'gpuArray');
            for jj=1:N_j

                % Includes check for cases in which no parameters are actually required
                if isempty(FnsToEvaluateParamNames(ii).Names) % || strcmp(FnsToEvaluateParamNames(1),'')) % check for 'FnsToEvaluateParamNames={}'
                    FnToEvaluateParamsVec=[];
                else
                    FnToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj));
                end
                Values(:,jj)=EvalFnOnAgentDist_Grid(FnsToEvaluate{ii}, FnToEvaluateParamsVec,PolicyValues(:,:,:,jj),l_daprime,n_a,n_z,a_grid,z_gridvals_J(:,:,jj));
            end
            AggVars(ii)=sum(sum(Values.*StationaryDist));
        end
    elseif simoptions.lowmemory==2 % loop over age j, and over the exogenous state
        % Loop over z (which is all exogenous states, if there is an e or semiz variable it has just been rolled into z)
        AggVars=zeros(length(FnsToEvaluate),N_z,'gpuArray');
        StationaryDist=permute(reshape(StationaryDist,[N_a,N_z,N_j]),[1,3,2]); % permute moves z to last dimension
        if n_d(1)>0
            PolicyIndexes=reshape(PolicyIndexes,[size(PolicyIndexes,1),N_a,N_z,N_j]);
        else
            PolicyIndexes=reshape(PolicyIndexes,[N_a,N_z,N_j]);
        end

        for z_c=1:N_z
            StationaryDistVec_z=StationaryDist(:,:,z_c); % This is why I did the permute (to avoid a reshape here). Not actually sure whether all the reshapes would be faster than the permute or not?
            if n_d(1)>0
                PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes(:,:,z_c,:),n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,1); % Note PolicyIndexes input is the wrong shape, but because this is parellel=2 the first thing PolicyInd2Val does is anyway to reshape() it.
            else
                PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes(:,z_c,:),n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,1); % Note PolicyIndexes input is the wrong shape, but because this is parellel=2 the first thing PolicyInd2Val does is anyway to reshape() it.
            end

            for ii=1:length(FnsToEvaluate)
                Values=nan(N_a*N_z,N_j,'gpuArray');
                for jj=1:N_j
                    % Includes check for cases in which no parameters are actually required
                    if isempty(FnsToEvaluateParamNames(ii).Names) % || strcmp(FnsToEvaluateParamNames(1),'')) % check for 'FnsToEvaluateParamNames={}'
                        FnToEvaluateParamsVec=[];
                    else
                        FnToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj));
                    end
                    Values(:,jj)=EvalFnOnAgentDist_Grid(FnsToEvaluate{ii}, [e_gridvals_J(e_c,:,jj),FnToEvaluateParamsVec],PolicyValues(:,:,:,jj),l_daprime,n_a,n_z,a_grid,z_gridvals_J(:,:,jj));
                end
                AggVars(ii,z_c)=sum(sum(Values.*StationaryDistVec_z));
            end
        end
        AggVars=sum(AggVars,2); % sum over e (note that the weighting by pi_e is already implicit in the stationary dist)
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
