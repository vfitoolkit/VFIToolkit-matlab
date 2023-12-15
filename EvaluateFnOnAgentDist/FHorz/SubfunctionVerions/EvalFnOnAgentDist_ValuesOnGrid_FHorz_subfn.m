function ValuesOnGrid=EvalFnOnAgentDist_ValuesOnGrid_FHorz_subfn(PolicyValues, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, N_j, a_grid, z_grid, simoptions,keepoutputasmatrix)
% subfn version is GPU only, and uses PolicyValues instead of PolicyIndexes
% Still loops over j, I could speed it further by parallel over j

if ~exist('simoptions','var')
    simoptions=struct();
end
if ~exist('keepoutputasmatrix','var')
    keepoutputasmatrix=0;
end

l_a=length(n_a);
N_a=prod(n_a);
N_z=prod(n_z);

a_gridvals=CreateGridvals(n_a,a_grid,1);


%% Exogenous shock grids

if N_z>0
    % Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
    % Gradually rolling these out so that all the commands build off of these
    z_gridvals_J=zeros(prod(n_z),length(n_z),'gpuArray');
    if isfield(simoptions,'ExogShockFn')
        if ~isfield(simoptions,'ExogShockFnParamNames')
            simoptions.ExogShockFnParamNames=getAnonymousFnInputNames(simoptions.ExogShockFn);
        end
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
    elseif prod(n_z)==0 % no z
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
end

% If using e variable, do same for this
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

        if isfield(simoptions,'EiidShockFn')
            if ~isfield(simoptions,'EiidShockFnParamNames')
                simoptions.EiidShockFnParamNames=getAnonymousFnInputNames(simoptions.EiidShockFn);    
            end
            for jj=1:N_j
                EiidShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.EiidShockFnParamNames,jj);
                EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                for ii=1:length(EiidShockFnParamsVec)
                    EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                end
                [simoptions.e_grid,simoptions.pi_e]=simoptions.EiidShockFn(EiidShockFnParamsCell{:});
                if all(size(simoptions.e_grid)==[sum(simoptions.n_e),1])
                    e_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(simoptions.n_e,simoptions.e_grid,1));
                else % already joint-grid
                    e_gridvals_J(:,:,jj)=gpuArray(simoptions.e_grid,1);
                end
            end
        elseif ndims(simoptions.e_grid)==3 % already an age-dependent joint-grid
            if all(size(simoptions.e_grid)==[prod(simoptions.n_e),length(simoptions.n_e),N_j])
                e_gridvals_J=simoptions.e_grid;
            end
        elseif all(size(simoptions.e_grid)==[sum(simoptions.n_e),N_j]) % age-dependent stacked-grid
            for jj=1:N_j
                e_gridvals_J(:,:,jj)=CreateGridvals(simoptions.n_e,simoptions.e_grid(:,jj),1);
            end
        elseif all(size(simoptions.e_grid)==[prod(simoptions.n_e),length(simoptions.n_e)]) % joint grid
            e_gridvals_J=simoptions.e_grid.*ones(1,1,N_j,'gpuArray');
        elseif all(size(simoptions.e_grid)==[sum(simoptions.n_e),1]) % basic grid
            e_gridvals_J=CreateGridvals(simoptions.n_e,simoptions.e_grid,1).*ones(1,1,N_j,'gpuArray');
        end

        n_e=simoptions.n_e;
        N_e=prod(n_e);
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
l_daprime=size(PolicyValues,1);


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
if isfield(simoptions,'keepoutputasmatrix')
    if simoptions.keepoutputasmatrix==1
        FnsToEvaluateStruct=0;
    elseif simoptions.keepoutputasmatrix==2
        FnsToEvaluateStruct=2;
    end
end


%% Loop over j
if N_z==0
    ValuesOnGrid=zeros(N_a,N_j,length(FnsToEvaluate),'gpuArray');

    for ff=1:length(FnsToEvaluate)
        Values=nan(N_a,N_j,'gpuArray');
        for jj=1:N_j
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(ff).Names) % || strcmp(FnsToEvaluateParamNames(1),'')) % check for 'FnsToEvaluateParamNames={}'
                FnToEvaluateParamsVec=[];
            else
                FnToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj));
            end
            Values(:,jj)=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsVec,PolicyValues(:,:,jj),l_daprime,n_a,0,a_gridvals,[]);
        end
        ValuesOnGrid(:,:,ff)=Values;
    end
else
    ValuesOnGrid=zeros(N_a*N_z,N_j,length(FnsToEvaluate),'gpuArray');

    for ff=1:length(FnsToEvaluate)
        Values=nan(N_a*N_z,N_j,'gpuArray');
        for jj=1:N_j
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(ff).Names) % || strcmp(FnsToEvaluateParamNames(1),'')) % check for 'FnsToEvaluateParamNames={}'
                FnToEvaluateParamsVec=[];
            else
                FnToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj));
            end
            Values(:,jj)=reshape(EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsVec,PolicyValues(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj)),[N_a*N_z,1]);
        end
        ValuesOnGrid(:,:,ff)=Values;
    end
end


if FnsToEvaluateStruct==1
    ValuesOnGrid2=ValuesOnGrid;
    clear ValuesOnGrid
    ValuesOnGrid=struct();
    if N_z==0
        for ff=1:length(AggVarNames)
            ValuesOnGrid.(AggVarNames{ff})=reshape(ValuesOnGrid2(:,:,ff),[n_a,N_j]);
            % Change the ordering and size so that ProbDensityFns has same kind of shape as StationaryDist
        end
    else
        for ff=1:length(AggVarNames)
            ValuesOnGrid.(AggVarNames{ff})=reshape(ValuesOnGrid2(:,:,ff),[n_a,n_z,N_j]);
            % Change the ordering and size so that ProbDensityFns has same kind of shape as StationaryDist
        end
    end
elseif FnsToEvaluateStruct==0
    % Change the ordering and size so that ProbDensityFns has same kind of
    % shape as StationaryDist, except first dimension indexes the 'FnsToEvaluate'.
    ValuesOnGrid=permute(ValuesOnGrid,[3,1,2]);
    if N_z==0
        ValuesOnGrid=reshape(ValuesOnGrid,[length(FnsToEvaluate),n_a,N_j]);
    else
        ValuesOnGrid=reshape(ValuesOnGrid,[length(FnsToEvaluate),n_a,n_z,N_j]);
    end
elseif FnsToEvaluateStruct==2 % Just a rearranged version of FnsToEvaluateStruct=0 for use internally when length(FnsToEvaluate)==1
    %     ValuesOnGrid=reshape(ValuesOnGrid,[N_a,N_z,N_j]);
    % The output is already in this shape anyway, so no need to actually reshape it at all
end

end
