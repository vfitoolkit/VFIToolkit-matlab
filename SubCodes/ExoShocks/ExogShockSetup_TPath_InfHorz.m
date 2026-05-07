function [z_gridvals, pi_z, transpathoptions, options]=ExogShockSetup_TPath_InfHorz(n_z,z_grid,pi_z,N_a,Parameters,PricePathNames,ParamPathNames,transpathoptions,options,gridpiboth)
% Convert z and e to age-dependent joint-grids and transtion matrix
% Can input vfoptions OR simoptions
% output: z_gridvals, pi_z, transpathoptions, options

% Sets up
% transpathoptions.zpathtrivial=1; % z_gridvals and pi_z are not varying over the path
%                              =0; % they vary over path, so z_gridvals_T and pi_z_T
% and
% transpathoptions.gridsinGE=1; % grids depend on a GE parameter and so need to be recomputed every iteration
%                           =0; % grids are exogenous
%

% gridpiboth=4: sometimes (trans path GE) we want both grid and transition probabilties
% gridpiboth=3: sometimes (value fn iter) we want both grid and transition probabilties
% gridpiboth=2: sometimes (agent dist)    we want just transition probabilties
% gridpiboth=1: sometimes (FnsToEvaluate) we want just grid


%% Check basic setup
N_z=prod(n_z);

transpathoptions.gridsinGE=0; % will be overwritten if appropriate
transpathoptions.zpathtrivial=1;  % will be overwritten if appropriate

%% Check if z_grid and/or pi_z depend on prices.
% If 'exogenous shock fn' is used, then precompute it to save evaluating it numerous times
% Check if using 'exogenous shock fn' (exogenous state has a grid and transition matrix that depends on age)

l_z=0;
if N_z>0
    l_z=length(n_z);

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
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, options.ExogShockFnParamNames);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [z_grid,pi_z]=options.ExogShockFn(ExogShockFnParamsCell{:});
            elseif overlap2==1 % ExogShockFn depends on a ParamPath parameter
                transpathoptions.zpathtrivial=0; % z_grid and pi_z vary over the path
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
                    transpathoptions.pi_z_T(:,:,tt)=pi_z;
                    transpathoptions.z_grid_T(:,tt)=z_grid;
                end
            end
        end

    else % Not ExogShockFn, or at least not any more
        if gridpiboth==1 % for most FnsToEvaluate, we don't use pi_z
            if all(size(z_grid)==[prod(n_z),length(n_z)]) % joint grid
                z_gridvals=z_grid;
            elseif all(size(z_grid)==[sum(n_z),1]) % basic grid
                z_gridvals=CreateGridvals(n_z,z_grid,1);
            end
        elseif gridpiboth==2 % For agent dist, we don't use grid
            z_gridvals=[];
        elseif gridpiboth==3 || gridpiboth==4 % For value fn, both z_gridvals and pi_z
            if all(size(z_grid)==[prod(n_z),length(n_z)]) % joint grid
                z_gridvals=z_grid;
            elseif all(size(z_grid)==[sum(n_z),1]) % basic grid
                z_gridvals=gpuArray(CreateGridvals(n_z,z_grid,1));
            end
        end
    end
    % Make sure they are on grid
    pi_z=gpuArray(pi_z);
    z_gridvals=gpuArray(z_gridvals);

    if ~isfield(transpathoptions,'zpathtrivial')
        transpathoptions.zpathtrivial=1;
    end

    % z_gridvals is [N_z,l_z]
    % pi_z is [N_z,N_z]
    % pi_z and z_gridvals are both gpuArrays

end


end
