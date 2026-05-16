function [z_gridvals, pi_z, pi_z_sparse, e_gridvals, pi_e, pi_e_sparse, ze_gridvals, transpathoptions, options]=ExogShockSetup_InfHorz_TPath(n_z,z_grid,pi_z,Parameters,PricePathNames,ParamPathNames,transpathoptions,options,gridpiboth)
% Convert z and e to joint-grids and transition matrix
% output: z_gridvals, pi_z, pi_z_sparse, e_gridvals, pi_e, pi_e_sparse, ze_gridvals, transpathoptions, options

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

% gridpiboth=4: sometimes (trans path GE) we want both grid and transition probabilities, including pi_z_sparse transition probs
% gridpiboth=3: sometimes (value fn iter) we want both grid and transition probabilities
% gridpiboth=2: sometimes (agent dist)    we want just transition probabilities, including pi_z_sparse transition probs
% gridpiboth=1: sometimes (FnsToEvaluate) we want just grid

% Accepted input shapes:
%   z_grid:
%     [sum(n_z), 1]                 stacked column grid for markov z (time-invariant)
%     [prod(n_z), length(n_z)]      joint grid for markov z (time-invariant)
%     [sum(n_z), T]                 stacked column grid for markov z, time-varying (last dim is T)
%     [prod(n_z), length(n_z), T]   joint grid for markov z, time-varying (last dim is T)
%   pi_z:
%     [prod(n_z), prod(n_z)]        transition matrix for markov z (time-invariant)
%     [prod(n_z), prod(n_z), T]     transition matrix for markov z, time-varying (last dim is T)
%   options.e_grid:
%     [sum(n_e), 1]                 stacked column grid for iid e (time-invariant)
%     [prod(n_e), length(n_e)]      joint grid for iid e (time-invariant)
%     [sum(n_e), T]                 stacked column grid for iid e, time-varying (last dim is T)
%     [prod(n_e), length(n_e), T]   joint grid for iid e, time-varying (last dim is T)
%   options.pi_e:
%     [prod(n_e), 1]                iid distribution (time-invariant)
%     [prod(n_e), T]                iid distribution, time-varying (last dim is T)
%
% When a time-varying input is supplied for z, transpathoptions.zpathtrivial
% is set to 0 and the full path is stored in transpathoptions.z_gridvals_T
% (joint-grid form, [prod(n_z), length(n_z), T]) and transpathoptions.pi_z_T
% ([prod(n_z), prod(n_z), T]). The plain outputs z_gridvals and pi_z are
% populated with the tt=1 slice and act as placeholders only -- downstream
% code reads the full path from transpathoptions.*_T. Same convention for e
% (transpathoptions.epathtrivial, transpathoptions.e_gridvals_T,
% transpathoptions.pi_e_T; the placeholders are options.e_gridvals and
% options.pi_e). T is inferred from the trailing dimension of the input.
%
% Variation along the transition path can also be introduced if
% options.ExogShockFn / options.EiidShockFn depends on a name listed in
% PricePathNames or ParamPathNames; the shock-fn is then called per t to
% assemble the time-varying path. If neither a time-varying input nor a
% path-dependent shock-fn is supplied, the inputs are interpreted as
% time-invariant.
%
% Stacked column grid: each of the underlying univariate grids written one
% beneath the next in a single column of length sum(n_z). Compact, but the
% joint state space is only implicit. For example with two markov variables
% of sizes n_z=[3,2], the column contains the 3 values of z1 followed by the
% 2 values of z2, giving a 5x1 vector. The time-varying form adds a
% trailing dimension of length T.
%
% Joint grid: every point in the product space listed explicitly, one per
% row, with each variable in its own column. The number of rows is
% prod(n_z) and the number of columns is length(n_z). Continuing the
% example, a joint grid is 6x2: each row pairs one z1 value with one z2
% value, covering all 6 combinations. The time-varying form adds a
% trailing dimension of length T.
%
% Output shapes (function returns):
%   z_gridvals:
%     [prod(n_z), length(n_z)]    joint grid; if transpathoptions.zpathtrivial==1, this IS the
%                                 time-invariant grid; if ==0, this is just the tt=1 slice
%                                 placeholder — downstream code reads transpathoptions.z_gridvals_T
%     []                          if gridpiboth==2 or N_z==0
%   pi_z:
%     [prod(n_z), prod(n_z)]      transition matrix (analogous placeholder convention)
%     []                          if gridpiboth==1 or N_z==0
%   pi_z_sparse:
%     sparse([prod(n_z), prod(n_z)])   CPU-side sparse copy of pi_z
%     []                               if N_z==0
%   e_gridvals:
%     [prod(n_e), length(n_e)]    joint grid for iid e (placeholder convention as above)
%     []                          if gridpiboth==2 or N_e==0
%   pi_e:
%     [prod(n_e), 1]              iid distribution (placeholder convention)
%     []                          if gridpiboth==1 or N_e==0
%   pi_e_sparse:
%     sparse([prod(n_e), 1])      CPU-side sparse copy of pi_e
%     []                          if N_e==0
%   ze_gridvals:
%     [prod(n_z)*prod(n_e), length(n_z)+length(n_e)]   combined grid used for AggVars (gridpiboth==1 or 4)
%     []                          if gridpiboth==2 or gridpiboth==3 (or both N_z==0 and N_e==0)
%
% transpathoptions fields populated:
%   .zpathtrivial    1 if z_gridvals/pi_z don't vary along the transition path, 0 otherwise
%   .epathtrivial    analogous for e
%   .zepathtrivial   =0 iff either of the above is 0 (only set when gridpiboth is 1 or 4)
%   .gridsinGE       1 if grids depend on a PricePath parameter (recompute every GE iteration); else 0
% When zpathtrivial==0:
%   .z_gridvals_T   [prod(n_z), length(n_z), T]
%   .pi_z_T         [prod(n_z), prod(n_z), T]
% When epathtrivial==0:
%   .e_gridvals_T   [prod(n_e), length(n_e), T]
%   .pi_e_T         [prod(n_e), T]
% When zepathtrivial==0:
%   .ze_gridvals_T  [prod(n_z)*prod(n_e), length(n_z)+length(n_e), T]
% T is inferred from the trailing dimension of whichever time-varying input is supplied.

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
                if any(strcmp(options.ExogShockFnParamNames{ii},ParamPathNames))
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
                transpathoptions.z_gridvals_T=zeros(prod(n_z),length(n_z),T,'gpuArray');
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

                    transpathoptions.pi_z_T(:,:,tt)=pi_z;
                    transpathoptions.z_gridvals_T(:,:,tt)=z_gridvals;
                end
            end
        end

    else % Not ExogShockFn, or at least not any more
        % Detect whether the raw z_grid / pi_z inputs are time-varying.
        % z_grid time-varying shapes are 3D [prod(n_z),length(n_z),T] or
        % 2D [sum(n_z),T] (where the 2D form is only unambiguous when
        % size(2)>1 and the shape doesn't match the time-invariant joint grid).
        z_grid_timevarying = (ndims(z_grid)==3) || ...
            (ndims(z_grid)==2 && size(z_grid,2)>1 && size(z_grid,1)==sum(n_z) && ~all(size(z_grid)==[prod(n_z),length(n_z)]));
        pi_z_timevarying = (ndims(pi_z)==3);

        if z_grid_timevarying || pi_z_timevarying
            % Infer T from whichever input is time-varying
            if pi_z_timevarying
                T=size(pi_z,3);
            elseif ndims(z_grid)==3
                T=size(z_grid,3);
            else
                T=size(z_grid,2); % 2D stacked-column time-varying
            end
            transpathoptions.zpathtrivial=0;

            % Build transpathoptions.z_gridvals_T as [prod(n_z),length(n_z),T]
            if z_grid_timevarying
                if ndims(z_grid)==3
                    if all(size(z_grid)==[prod(n_z),length(n_z),T])
                        transpathoptions.z_gridvals_T=gpuArray(z_grid);
                    else
                        error('z_grid is 3D but size does not match [prod(n_z), length(n_z), T]')
                    end
                else
                    if all(size(z_grid)==[sum(n_z),T])
                        transpathoptions.z_gridvals_T=zeros(prod(n_z),length(n_z),T,'gpuArray');
                        for tt=1:T
                            transpathoptions.z_gridvals_T(:,:,tt)=CreateGridvals(n_z,gpuArray(z_grid(:,tt)),1);
                        end
                    else
                        error('z_grid 2D with trailing dim > 1 but size does not match [sum(n_z), T]')
                    end
                end
            else
                % z_grid is time-invariant: broadcast across t
                if all(size(z_grid)==[prod(n_z),length(n_z)])
                    z_gridvals_static=gpuArray(z_grid);
                elseif all(size(z_grid)==[sum(n_z),1])
                    z_gridvals_static=CreateGridvals(n_z,gpuArray(z_grid),1);
                else
                    error('z_grid time-invariant but size does not match any expected shape')
                end
                transpathoptions.z_gridvals_T=repmat(z_gridvals_static,1,1,T);
            end

            % Build transpathoptions.pi_z_T as [prod(n_z),prod(n_z),T]
            if pi_z_timevarying
                if all(size(pi_z)==[prod(n_z),prod(n_z),T])
                    transpathoptions.pi_z_T=gpuArray(pi_z);
                else
                    error('pi_z is 3D but size does not match [prod(n_z), prod(n_z), T]')
                end
            else
                if all(size(pi_z)==[prod(n_z),prod(n_z)])
                    transpathoptions.pi_z_T=repmat(gpuArray(pi_z),1,1,T);
                else
                    error('pi_z time-invariant but size does not match [prod(n_z), prod(n_z)]')
                end
            end

            % Plain outputs are placeholders for tt=1; downstream code uses the _T fields
            if gridpiboth==1
                z_gridvals=transpathoptions.z_gridvals_T(:,:,1);
                pi_z=[];
            elseif gridpiboth==2
                z_gridvals=[];
                pi_z=transpathoptions.pi_z_T(:,:,1);
            elseif gridpiboth==3 || gridpiboth==4
                z_gridvals=transpathoptions.z_gridvals_T(:,:,1);
                pi_z=transpathoptions.pi_z_T(:,:,1);
            end
        else
            % Time-invariant inputs: existing logic
            if gridpiboth==1 % for most FnsToEvaluate, we don't use pi_z
                pi_z=[];
                % Now just do z_gridvals
                if all(size(z_grid)==[sum(n_z),1]) % stacked-column grid
                    z_gridvals=CreateGridvals(n_z,z_grid,1);
                elseif all(size(z_grid)==[prod(n_z),length(n_z)]) % joint grid
                    z_gridvals=z_grid;
                else
                    error('z_grid size does not match any expected shape')
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
                else
                    error('z_grid size does not match any expected shape')
                end
            end
        end
    end

    if ~isfield(transpathoptions,'zpathtrivial')
        transpathoptions.zpathtrivial=1;
    end

    z_gridvals=gpuArray(z_gridvals);
    if gridpiboth==2
        pi_z=gather(pi_z); % Agent distribution iteration is performed on cpu
    else
        pi_z=gpuArray(pi_z);
    end
    pi_z_sparse=sparse(gather(pi_z));
    % z_gridvals is [N_z,l_z]
    % pi_z is [N_z,N_z]
    % pi_z is a gpuArray except when gridpiboth==2 (agent dist) when it is on cpu; z_gridvals is a gpuArray
end

%% If using e variables do the same for e as we just did for z
e_gridvals=[];
pi_e=[];
pi_e_sparse=[];
l_e=0;
if N_e>0
    l_e=length(n_e);
    % Check if e_grid and/or pi_e depend on prices. If not then create pi_e_J and e_grid_J for the entire transition before we start


    % Just calculate grid and transition probabilities anyway
    if isfield(options,'EiidShockFn')
        options.EiidShockFnParamNames=getAnonymousFnInputNames(options.EiidShockFn);
        % First, check if EiidShockFn depends on a PricePath parameter
        overlap=0;
        for ii=1:length(options.EiidShockFnParamNames)
            if any(strcmp(options.EiidShockFnParamNames{ii},PricePathNames))
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
                if any(strcmp(options.EiidShockFnParamNames{ii},ParamPathNames))
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
                transpathoptions.pi_e_T=zeros(N_e,T,'gpuArray');
                transpathoptions.e_gridvals_T=zeros(prod(n_e),length(n_e),T,'gpuArray');
                for tt=1:T
                    for ii=1:length(ParamPathNames)
                        Parameters.(ParamPathNames{ii})=ParamPathStruct.(ParamPathNames{ii});
                    end
                    % Note, we know the PricePath is irrelevant for the current purpose
                    EiidShockFnParamsVec=CreateVectorFromParams(Parameters, options.EiidShockFnParamNames);
                    EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                    for ii=1:length(EiidShockFnParamsVec)
                        EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                    end
                    [e_grid,pi_e]=options.EiidShockFn(EiidShockFnParamsCell{:});
                    pi_e=gpuArray(pi_e);
                    e_gridvals=CreateGridvals(n_e,gpuArray(e_grid),1);

                    transpathoptions.pi_e_T(:,tt)=pi_e;
                    transpathoptions.e_gridvals_T(:,:,tt)=e_gridvals;
                end
            end
        end

    else % Not EiidShockFn, or at least not any more
        % Detect whether the raw options.e_grid / options.pi_e inputs are time-varying.
        if isfield(options,'e_grid')
            e_grid_timevarying = (ndims(options.e_grid)==3) || ...
                (ndims(options.e_grid)==2 && size(options.e_grid,2)>1 && size(options.e_grid,1)==sum(n_e) && ~all(size(options.e_grid)==[prod(n_e),length(n_e)]));
        else
            e_grid_timevarying = false;
        end
        if isfield(options,'pi_e')
            pi_e_timevarying = (size(options.pi_e,2)>1);
        else
            pi_e_timevarying = false;
        end

        if e_grid_timevarying || pi_e_timevarying
            % Infer T from whichever input is time-varying
            if pi_e_timevarying
                T=size(options.pi_e,2);
            elseif ndims(options.e_grid)==3
                T=size(options.e_grid,3);
            else
                T=size(options.e_grid,2);
            end
            transpathoptions.epathtrivial=0;

            % Build transpathoptions.e_gridvals_T as [prod(n_e),length(n_e),T]
            if e_grid_timevarying
                if ndims(options.e_grid)==3
                    if all(size(options.e_grid)==[prod(n_e),length(n_e),T])
                        transpathoptions.e_gridvals_T=gpuArray(options.e_grid);
                    else
                        error('options.e_grid is 3D but size does not match [prod(n_e), length(n_e), T]')
                    end
                else
                    if all(size(options.e_grid)==[sum(n_e),T])
                        transpathoptions.e_gridvals_T=zeros(prod(n_e),length(n_e),T,'gpuArray');
                        for tt=1:T
                            transpathoptions.e_gridvals_T(:,:,tt)=CreateGridvals(n_e,gpuArray(options.e_grid(:,tt)),1);
                        end
                    else
                        error('options.e_grid 2D with trailing dim > 1 but size does not match [sum(n_e), T]')
                    end
                end
            else
                % e_grid is time-invariant: broadcast across t
                if all(size(options.e_grid)==[prod(n_e),length(n_e)])
                    e_gridvals_static=gpuArray(options.e_grid);
                elseif all(size(options.e_grid)==[sum(n_e),1])
                    e_gridvals_static=CreateGridvals(n_e,gpuArray(options.e_grid),1);
                else
                    error('options.e_grid time-invariant but size does not match any expected shape')
                end
                transpathoptions.e_gridvals_T=repmat(e_gridvals_static,1,1,T);
            end

            % Build transpathoptions.pi_e_T as [prod(n_e),T]
            if pi_e_timevarying
                if all(size(options.pi_e)==[prod(n_e),T])
                    transpathoptions.pi_e_T=gpuArray(options.pi_e);
                else
                    error('options.pi_e 2D with trailing dim > 1 but size does not match [prod(n_e), T]')
                end
            else
                if all(size(options.pi_e)==[prod(n_e),1])
                    transpathoptions.pi_e_T=repmat(gpuArray(options.pi_e),1,T);
                else
                    error('options.pi_e time-invariant but size does not match [prod(n_e), 1]')
                end
            end

            % Plain outputs are placeholders for tt=1; downstream code uses the _T fields
            if gridpiboth==1
                e_gridvals=transpathoptions.e_gridvals_T(:,:,1);
                pi_e=[];
            elseif gridpiboth==2
                e_gridvals=[];
                pi_e=transpathoptions.pi_e_T(:,1);
            elseif gridpiboth==3 || gridpiboth==4
                e_gridvals=transpathoptions.e_gridvals_T(:,:,1);
                pi_e=transpathoptions.pi_e_T(:,1);
            end

            if isfield(options,'e_grid'); options=rmfield(options,'e_grid'); end
            if isfield(options,'pi_e'); options=rmfield(options,'pi_e'); end
        else
            % Time-invariant inputs: existing logic
            if gridpiboth==1 % for most FnsToEvaluate, we don't use pi_e
                if isfield(options,'e_grid')
                    if all(size(options.e_grid)==[sum(n_e),1]) % stacked-column grid
                        e_gridvals=CreateGridvals(n_e,gpuArray(options.e_grid),1);
                    elseif all(size(options.e_grid)==[prod(n_e),length(n_e)]) % joint grid
                        e_gridvals=options.e_grid;
                    else
                        error('options.e_grid time-invariant but size does not match any expected shape')
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
                    if all(size(options.e_grid)==[sum(n_e),1]) % stacked-column grid
                        e_gridvals=CreateGridvals(n_e,gpuArray(options.e_grid),1);
                    elseif all(size(options.e_grid)==[prod(n_e),length(n_e)]) % joint grid
                        e_gridvals=options.e_grid;
                    else
                        error('options.e_grid time-invariant but size does not match any expected shape')
                    end
                    options=rmfield(options,'e_grid');
                    pi_e=options.pi_e;
                    options=rmfield(options,'pi_e');
                end
            end
        end
    end
    % Make sure they are on grid
    if gridpiboth==2
        pi_e=gather(pi_e); % Agent distribution iteration is performed on cpu
    else
        pi_e=gpuArray(pi_e);
    end
    e_gridvals=gpuArray(e_gridvals);
    pi_e_sparse=sparse(gather(pi_e));

    if ~isfield(transpathoptions,'epathtrivial')
        transpathoptions.epathtrivial=1;
    end

    % e_gridvals is [N_e,l_e]
    % pi_e is [N_e,1]
    % pi_e and e_gridvals are both gpuArrays

    % vfoptions.e_grid=e_gridvals;
    % vfoptions.pi_e=pi_e;
    % simoptions.e_grid=e_gridvals;
    % simoptions.pi_e=pi_e;
end





%% Create ze_gridvals, which is used for AggVars. Will be z_gridvals or e_gridvals if only one of them is used
% gridpiboth=4: sometimes (trans path GE) we want both grid and transition probabilities, including pi_z_sparse
% gridpiboth=3: sometimes (value fn iter) we want both grid and transition probabilities
% gridpiboth=2: sometimes (agent dist)    we want just transition probabilities, including pi_z_sparse
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