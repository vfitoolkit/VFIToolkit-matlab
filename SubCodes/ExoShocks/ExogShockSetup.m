function [z_gridvals, pi_z, options]=ExogShockSetup(n_z,z_grid,pi_z,Parameters,options,gridpiboth)
% Convert z and e to joint-grids and transtion matrix
% options will either be vfoptions or simoptions
% output: z_gridvals, pi_z, options.e_gridvals, options.pi_e

% gridpiboth=3: sometimes (value fn iter) we want both grid and transition probabilties
% gridpiboth=2: sometimes (agent dist)    we want just transition probabilties
% gridpiboth=1: sometimes (FnsToEvaluate) we want just grid

%% Check basic setup
if ~isfield(options,'n_e')
    n_e=0;
else
    n_e=options.n_e;
end

if isfield(options,'ExogShockFn')
    options.ExogShockFnParamNames=getAnonymousFnInputNames(options.ExogShockFn);
end
if isfield(options,'EiidShockFn')
    options.EiidShockFnParamNames=getAnonymousFnInputNames(options.EiidShockFn);
end


%% Deal with z variables
% Convert to z_gridvals (joint grids) and corresponding pi_z (transition matrix).
if prod(n_z)==0
    z_gridvals=[];
    pi_z=[];
else
    if gridpiboth==1 % for most FnsToEvaluate, we don't use pi_z
        pi_z=[];
        % Now just do z_gridvals
        % z_gridvals=zeros(prod(n_z),length(n_z),'gpuArray');
        if isfield(options,'ExogShockFn')
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, options.ExogShockFnParamNames);
            ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
            for ii=1:length(ExogShockFnParamsVec)
                ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
            end
            [z_grid,~]=options.ExogShockFn(ExogShockFnParamsCell{:});
        end
        if all(size(z_grid)==[prod(n_z),length(n_z)]) % joint grid
            z_gridvals=z_grid;
        elseif all(size(z_grid)==[sum(n_z),1]) % basic grid
            z_gridvals=CreateGridvals(n_z,z_grid,1);
        end
        z_gridvals=gpuArray(z_gridvals);
    elseif gridpiboth==2 % For agent dist, we don't use grid
        z_gridvals=[];
        % Now just do pi_z_J
        if isfield(options,'ExogShockFn')
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, options.ExogShockFnParamNames);
            ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
            for ii=1:length(ExogShockFnParamsVec)
                ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
            end
            [~,pi_z]=options.ExogShockFn(ExogShockFnParamsCell{:});
        end
        pi_z=gpuArray(pi_z);
    elseif gridpiboth==3
        % For value fn, both z_gridvals_J and pi_z_J
        % z_gridvals=zeros(prod(n_z),length(n_z),'gpuArray');
        if isfield(options,'ExogShockFn')
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, options.ExogShockFnParamNames);
            ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
            for ii=1:length(ExogShockFnParamsVec)
                ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
            end
            [z_gridvals,pi_z]=options.ExogShockFn(ExogShockFnParamsCell{:});
        end
        if all(size(z_grid)==[prod(n_z),length(n_z)]) % joint grid
            z_gridvals=z_grid;
        elseif all(size(z_grid)==[sum(n_z),1]) % basic grid
            z_gridvals=CreateGridvals(n_z,z_grid,1);
        end
        z_gridvals=gpuArray(z_gridvals);
        pi_z=gpuArray(pi_z);
    end
end



%% If using e variable, do same for this
if prod(n_e)==0
    options.e_gridvals=[];
    options.pi_e=[];
else
    if ~isfield(options,'e_grid') && ~isfield(options,'EiidShockFn')
        error('You are using an e (iid) variable, and so need to declare options.e_grid (options refers to either vfoptions or simoptions)')
    elseif ~isfield(options,'pi_e') && ~isfield(options,'EiidShockFn')
        error('You are using an e (iid) variable, and so need to declare options.pi_e (options refers to either vfoptions or simoptions)')
    end

    if gridpiboth==1 % for most FnsToEvaluate, we don't use pi_z
        options.pi_e=[];
        % Now just do e_gridvals_J
        options.e_gridvals=zeros(prod(options.n_e),length(options.n_e),'gpuArray');
        if isfield(options,'EiidShockFn')
            EiidShockFnParamsVec=CreateVectorFromParams(Parameters, options.EiidShockFnParamNames);
            EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
            for ii=1:length(EiidShockFnParamsVec)
                EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
            end
            [options.e_grid,~]=options.EiidShockFn(EiidShockFnParamsCell{:});
        end
        if all(size(options.e_grid)==[prod(options.n_e),length(options.n_e)]) % joint grid
            options.e_gridvals=options.e_grid;
        elseif all(size(options.e_grid)==[sum(options.n_e),1]) % basic grid
            options.e_gridvals=CreateGridvals(options.n_e,options.e_grid,1);
        end
    elseif gridpiboth==2 % For agent dist, we don't use grid
        options.e_gridvals=[];
        % Now just do pi_e_J
        if isfield(options,'EiidShockFn')
            EiidShockFnParamsVec=CreateVectorFromParams(Parameters, options.EiidShockFnParamNames);
            EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
            for ii=1:length(EiidShockFnParamsVec)
                EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
            end
            [~,options.pi_e]=options.EiidShockFn(EiidShockFnParamsCell{:});
        end
    elseif gridpiboth==3
        % For value fn, both e_gridvals_J and pi_e_J
        options.e_gridvals=zeros(prod(options.n_e),length(options.n_e),'gpuArray');
        if isfield(options,'EiidShockFn')
            EiidShockFnParamsVec=CreateVectorFromParams(Parameters, options.EiidShockFnParamNames);
            EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
            for ii=1:length(EiidShockFnParamsVec)
                EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
            end
            [options.e_grid,options.pi_e]=options.EiidShockFn(EiidShockFnParamsCell{:});
        end
        if all(size(options.e_grid)==[prod(options.n_e),length(options.n_e)]) % joint grid
            options.e_gridvals=options.e_grid;
        elseif all(size(options.e_grid)==[sum(options.n_e),1]) % basic grid
            options.e_gridvals=CreateGridvals(options.n_e,options.e_grid,1);
        end
    end
end


% We have evaluated ExogShockFn and EiidShockFn, so now remove them to keep things simple/clean
if isfield(options,'ExogShockFn')
    options=rmfield(options,'ExogShockFn');
end
if isfield(options,'ExogShockFnParamNames')
    options=rmfield(options,'ExogShockFnParamNames');
end
if isfield(options,'EiidShockFn')
    options=rmfield(options,'EiidShockFn');
end
if isfield(options,'EiidShockFnParamNames')
    options=rmfield(options,'EiidShockFnParamNames');
end


end