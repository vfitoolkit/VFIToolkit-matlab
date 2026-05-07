function [z_gridvals, pi_z, pi_z_sparse, e_gridvals, pi_e, pi_e_sparse, ze_gridvals, transpathoptions, options]=ExogShockSetup_TPath_InfHorz(n_z,z_grid,pi_z,N_a,Parameters,PricePathNames,ParamPathNames,transpathoptions,options,gridpiboth)
% Convert z and e to joint-grids and transition matrix
% output: z_gridvals, pi_z, e_gridvals, pi_e, transpathoptions,vfoptions,simoptions

% Sets up
% transpathoptions.zpathtrivial=1; % z_gridvals and pi_z are not varying over the path
%                              =0; % they vary over path, so z_gridvals_T and pi_z_T
% transpathoptions.epathtrivial=1; % e_gridvals and pi_e are not varying over the path
%                              =0; % they vary over path, so e_gridvals_T and pi_e_T
% and
% transpathoptions.gridsinGE=1; % grids depend on a GE parameter and so need to be recomputed every iteration
%                           =0; % grids are exogenous
%
% transpathoptions.zepathtrivial=0 when either of zpathtrival and epathtrivial both are zero

% gridpiboth=4: sometimes (trans path GE) we want both grid and transition probabilties, including pi_z_sparse transition probs
% gridpiboth=3: sometimes (value fn iter) we want both grid and transition probabilties
% gridpiboth=2: sometimes (agent dist)    we want just transition probabilties, including pi_z_sparse transition probs
% gridpiboth=1: sometimes (FnsToEvaluate) we want just grid


%% Check basic setup
N_z=prod(n_z);

if ~isfield(options,'n_e')
    n_e=0;
else
    n_e=options.n_e;
    options=rmfield(options,'n_e');
end
N_e=prod(n_e);

transpathoptions.gridsinGE=0; % will be overwritten if appropriate
transpathoptions.zpathtrivial=1;  % will be overwritten if appropriate
transpathoptions.epathtrivial=1;  % will be overwritten if appropriate

%% Check if z_grid and/or pi_z depend on prices. If not then create pi_z_J and z_grid_J for the entire transition before we start
% If 'exogenous shock fn' is used, then precompute it to save evaluating it numerous times
% Check if using 'exogenous shock fn' (exogenous state has a grid and transition matrix that depends on age)

z_gridvals=[];
if N_z>0
    if isfield(options,'ExogShockFn') % Just calculate grid and transition probabilities anyway
        options.ExogShockFnParamNames=getAnonymousFnInputNames(options.ExogShockFn);
        % First, check if ExogShockFn depends on a PricePath parameter
        overlap=0;
        for ii=1:length(options.ExogShockFnParamNames)
            if any(strcmp(options.ExogShockFnParamNames{ii},PricePathNames))
                overlap=1;
            end
        end
        if overlap==1
            transpathoptions.gridsinGE=1;
            transpathoptions.zpathtrivial=0; % z_grid and pi_z vary over the path
            error('Not yet implemented to use ExogShockFn which includes parameters from PricePath (contact me)')
        else % overlap==0
            % Next,check if ExogShockFn depends on a ParamPath parameter
            overlap2=0;
            for ii=1:length(options.ExogShockFnParamNames)
                if strcmp(options.ExogShockFnParamNames{ii},ParamPathNames)
                    overlap2=1;
                end
            end
            if overlap2==0
                pi_z=zeros(N_z,N_z,'gpuArray');
                z_grid=zeros(N_z,1,'gpuArray');
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, options.ExogShockFnParamNames);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [z_grid,pi_z]=options.ExogShockFn(ExogShockFnParamsCell{:});
                pi_z=gpuArray(pi_z);
                z_gridvals=CreateGridvals(n_z,gpuArray(z_grid),1);
            elseif overlap2==1 % ExogShockFn depends on a ParamPath parameter
                transpathoptions.zpathtrivial=0; % z_grid_J and pi_z_J vary over the path
                transpathoptions.pi_z_T=zeros(N_z,N_z,T,'gpuArray');
                transpathoptions.z_grid_T=zeros(sum(n_z),T,'gpuArray');
                for tt=1:T
                    for ii=1:length(ParamPathNames)
                        Parameters.(ParamPathNames{ii})=ParamPathStruct.(ParamPathNames{ii});
                    end
                    % Note, we know the PricePath is irrelevant for the current purpose
                    ExogShockFnParamsVec=CreateVectorFromParams(Parameters, options.ExogShockFnParamNames);
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for ii=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                    end
                    [z_grid,pi_z]=options.ExogShockFn(ExogShockFnParamsCell{:});
                    pi_z=gpuArray(pi_z);
                    z_gridvals=CreateGridvals(n_z,gpuArray(z_grid),1);

                    transpathoptions.pi_z_T(:,:,:,tt)=pi_z;
                    transpathoptions.z_grid_T(:,:,tt)=z_grid;
                end
            end
        end

    else % Not ExogShockFn, or at least not any more
        if gridpiboth==1 % for most FnsToEvaluate, we don't use pi_z
            pi_z=[];
            % Now just do z_gridvals
            z_gridvals=zeros(prod(n_z),length(n_z),'gpuArray');
            if all(size(z_grid)==[sum(n_z),1]) % stacked-column grid
                z_gridvals=CreateGridvals(n_z,z_grid,1);
            elseif all(size(z_grid)==[prod(n_z),length(n_z)]) % joint grid
                z_gridvals=z_grid;
            end
        elseif gridpiboth==2 % For agent dist, we don't use grid
            z_gridvals=[];
            % pi_z is fine as is
        elseif gridpiboth==3 || gridpiboth==4 % For value fn, both z_gridvals_J and pi_z_J
            if all(size(z_grid)==[prod(n_z),length(n_z)]) % joint grid
                z_gridvals=z_grid;
                % pi_z is fine as is
            elseif all(size(z_grid)==[sum(n_z),1]) % basic grid
                z_gridvals=CreateGridvals(n_z,z_grid,1);
                % pi_z is fine as is
            end
        end
    end
    
    if ~isfield(transpathoptions,'zpathtrivial')
        transpathoptions.zpathtrivial=1;
    end

    z_gridvals=gpuArray(z_gridvals);
    pi_z=gpuArray(pi_z);
    pi_z_sparse=sparse(gather(pi_z));
    % z_gridvals is [N_z,l_z]
    % pi_z is [N_z,N_z]
    % pi_z and z_gridvals are both gpuArrays
end

%% If using e variables do the same for e as we just did for z
e_gridvals=[];
pi_e=[];
pi_e_sparse=[];
l_e=0;
if N_e>0
    l_e=length(n_e);
    % Check if e_grid and/or pi_e depend on prices. If not then create pi_e_J and e_grid_J for the entire transition before we start

    transpathoptions.epathprecomputed=0;

    % Just calculate grid and transition probabilities anyway
    if isfield(options,'EiidShockFn')
        options.EiidShockFnParamNames=getAnonymousFnInputNames(options.EiidShockFn);
        % First, check if EiidShockFn depends on a PricePath parameter
        overlap=0;
        for ii=1:length(options.EiidShockFnParamNames)
            if strcmp(options.EiidShockFnParamNames{ii},PricePathNames)
                overlap=1;
            end
        end
        if overlap==1
            transpathoptions.gridsinGE=1;
            transpathoptions.epathtrivial=0; % e_grid_J and pi_e_J vary over the path
            error('Not yet implemented to use EiidShockFn which includes parameters from PricePath (contact me)')
        else % overlap==0
            % Next,check if EiidShockFn depends on a ParamPath parameter
            overlap2=0;
            for ii=1:length(options.EiidShockFnParamNames)
                if strcmp(options.EiidShockFnParamNames{ii},ParamPathNames)
                    overlap2=1;
                end
            end
            if overlap2==0
                EiidShockFnParamsVec=CreateVectorFromParams(Parameters, options.EiidShockFnParamNames);
                EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                for ii=1:length(EiidShockFnParamsVec)
                    EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                end
                [e_grid,pi_e]=options.EiidShockFn(EiidShockFnParamsCell{:});
                pi_e=gpuArray(pi_e);
                e_grid=gpuArray(e_grid);
            elseif overlap2==1 % ExogShockFn depends on a ParamPath parameter
                transpathoptions.epathtrivial=0; % e_grid and pi_e vary over the path
                transpathoptions.pi_e_T=zeros(N_e,N_e,T,'gpuArray');
                transpathoptions.e_grid_T=zeros(sum(n_e),T,'gpuArray');
                for tt=1:T
                    for ii=1:length(ParamPathNames)
                        Parameters.(ParamPathNames{ii})=ParamPathStruct.(ParamPathNames{ii});
                    end
                    % Note, we know the PricePath is irrelevant for the current purpose
                    EiidShockFnParamsVec=CreateVectorFromParams(Parameters, options.EiidShockFnParamNames);
                    EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                    for ii=1:length(ExogShockFnParamsVec)
                        EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                    end
                    [e_grid,pi_e]=options.ExogShockFn(EiidShockFnParamsCell{:});
                    pi_e=gpuArray(pi_e);
                    e_grid=gpuArray(e_grid);

                    transpathoptions.pi_e_T(:,tt)=pi_e;
                    transpathoptions.e_grid_T(:,:,tt)=e_grid;
                end
            end
        end

    else % Not ExogShockFn, or at least not any more
        if gridpiboth==1 % for most FnsToEvaluate, we don't use pi_z
            if isfield(options,'e_grid')
                if size(options.e_grid,2)==1 % stacked-column grid
                    e_gridvals=options.e_grid.*ones(1,1,N_j,'gpuArray');
                else
                    e_gridvals=options.e_grid;
                end
                options=rmfield(options,'e_grid');
            end
        elseif gridpiboth==2 % For agent dist, we don't use grid
            if isfield(options,'pi_e')
                pi_e=options.pi_e;
                options=rmfield(options,'pi_e');
            end
        elseif gridpiboth==3 || gridpiboth==4 % For value fn, both e_gridvals and pi_e
            if isfield(options,'pi_e')
                if size(options.e_grid,2)==1 % stacked-column grid
                    e_gridvals=options.e_grid.*ones(1,1,N_j,'gpuArray');
                else
                    e_gridvals=options.e_grid;
                end
                options=rmfield(options,'e_grid');
                pi_e=options.pi_e;
                options=rmfield(options,'pi_e');
            end
        end
    end
    % Make sure they are on grid
    pi_e=gpuArray(pi_e);
    e_gridvals=gpuArray(e_gridvals);

    if ~isfield(transpathoptions,'epathtrivial')
        transpathoptions.epathtrivial=1;
    end

    e_gridvals=gpuArray(e_gridvals);
    pi_e=gpuArray(pi_e);
    % e_gridvals is [N_e,l_e]
    % pi_e is [N_e,1]
    % pi_e and e_gridvals are both gpuArrays

    % vfoptions.e_grid=e_gridvals;
    % vfoptions.pi_e=pi_e;
    % simoptions.e_grid=e_gridvals;
    % simoptions.pi_e=pi_e;
end





%% Create ze_gridvals, which is used for AggVars. Will be z_gridvals or e_gridvals if only one of them is used
% gridpiboth=4: sometimes (trans path GE) we want both grid and transition probabilties, including pi_z_sparse
% gridpiboth=3: sometimes (value fn iter) we want both grid and transition probabilties
% gridpiboth=2: sometimes (agent dist)    we want just transition probabilties, including pi_z_sparse
% gridpiboth=1: sometimes (FnsToEvaluate) we want just grid
if gridpiboth==3 || gridpiboth==2
    ze_gridvals=[];
else
    if transpathoptions.zpathtrivial==0 || transpathoptions.epathtrivial==0
        transpathoptions.zepathtrivial=0;
    else
        transpathoptions.zepathtrivial=1;
    end

    if N_e==0 && N_z==0
        ze_gridvals=[];
    elseif N_z>0 && N_e==0
        ze_gridvals=z_gridvals;
    elseif N_z==0 && N_e>0
        ze_gridvals=e_gridvals;
    elseif N_z>0 && N_e>0
        ze_gridvals=[repmat(z_gridvals,N_e,1),repelem(e_gridvals,N_z,1)];
    end

    % If ze_gridvals depends on t, set up transpathoptions.ze_gridvals_T
    if transpathoptions.zepathtrivial==0
        if N_e==0 && N_z==0
            transpathoptions.ze_gridvals_T=[];
        elseif N_z>0 && N_e==0
            transpathoptions.ze_gridvals_T=transpathoptions.z_gridvals_T;
        elseif N_z==0 && N_e>0
            transpathoptions.ze_gridvals_T=transpathoptions.e_gridvals_T;
        elseif N_z>0 && N_e>0
            transpathoptions.ze_gridvals_T=[repmat(transpathoptions.z_gridvals_T,N_e,1),repelem(transpathoptions.e_gridvals_T,N_z,1)];
        end
    end
end


end