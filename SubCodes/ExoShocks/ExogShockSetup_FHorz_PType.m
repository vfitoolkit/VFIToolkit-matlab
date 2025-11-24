function [z_gridvals_J, pi_z_J, options]=ExogShockSetup_FHorz_PType(n_z,z_grid,pi_z,N_j,Names_i,Parameters,options,gridpiboth)
% Convert z and e to age-dependent joint-grids and transtion matrix
% options will either be vfoptions or simoptions
% output: z_gridvals_J, pi_z_J, options.e_gridvals_J, options.pi_e_J
% All outputs are made dependent on permanent type (as that way the rest of the codes can just assume this)

% gridpiboth=3: sometimes (value fn iter) we want both grid and transition probabilties
% gridpiboth=2: sometimes (agent dist)    we want just transition probabilties
% gridpiboth=1: sometimes (FnsToEvaluate) we want just grid

% Note: must be run after Names_i is setup (so that this is Names_i and not N_i)

%% Check what we are doing in terms of dependence on permanent type for z
zdependsonptype=0;
if isstruct(z_grid) || isstruct(pi_z)
    zdependsonptype=1;
end
if isfield(options,'ExogShockFn')
    if isstruct(options.ExogShockFn)
        zdependsonptype=1;
        for ii=1:length(Names_i)
            options.ExogShockFnParamNames.(Names_i{ii})=getAnonymousFnInputNames(options.ExogShockFn.(Names_i{ii}));
        end
    else
        options.ExogShockFnParamNames=getAnonymousFnInputNames(options.ExogShockFn);
    end
end


%% Check what we are doing in terms of dependence on permanent type for e
edependsonptype=0;
if ~isfield(options,'n_e')
    n_e=0;
else
    n_e=options.n_e;

    if ~isfield(options,'e_grid') && ~isfield(options,'EiidShockFn')
        error('You are using an e (iid) variable, and so need to declare options.e_grid (options refers to either vfoptions or simoptions)')
    elseif ~isfield(options,'pi_e') && ~isfield(options,'EiidShockFn')
        error('You are using an e (iid) variable, and so need to declare options.pi_e (options refers to either vfoptions or simoptions)')
    end

    if isstruct(options.e_grid) || isstruct(options.pi_e)
        edependsonptype=1;
    end
    if isfield(options,'EiidShockFn')
        if isstruct(options.EiidShockFn)
            edependsonptype=1;
            for ii=1:length(Names_i)
                options.EiidShockFnParamNames.(Names_i{ii})=getAnonymousFnInputNames(options.EiidShockFn.(Names_i{ii}));
            end
        else
            options.EiidShockFnParamNames=getAnonymousFnInputNames(options.EiidShockFn);
        end
    end
end


%% Deal with z variables
if zdependsonptype==0
    % Convert to z_gridvals_J (age-dependent joint grids) and corresponding
    % pi_z_J (age-dependent transition matrix).
    if prod(n_z)==0
        z_gridvals_J=[];
        pi_z_J=[];
    else
        if gridpiboth==1 % for most FnsToEvaluate, we don't use pi_z
            pi_z_J=[];
            % Now just do z_gridvals_J
            z_gridvals_J=zeros(prod(n_z),length(n_z),'gpuArray');
            if isfield(options,'ExogShockFn')
                for jj=1:N_j
                    ExogShockFnParamsVec=CreateVectorFromParams(Parameters, options.ExogShockFnParamNames,jj);
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for ii=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                    end
                    [z_grid,~]=options.ExogShockFn(ExogShockFnParamsCell{:});
                    if all(size(z_grid)==[sum(n_z),1])
                        z_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(n_z,z_grid,1));
                    else % already joint-grid
                        z_gridvals_J(:,:,jj)=gpuArray(z_grid,1);
                    end
                end
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
        elseif gridpiboth==2 % For agent dist, we don't use grid
            z_gridvals_J=[];
            % Now just do pi_z_J
            pi_z_J=zeros(prod(n_z),prod(n_z),'gpuArray');
            if isfield(options,'ExogShockFn')
                for jj=1:N_j
                    ExogShockFnParamsVec=CreateVectorFromParams(Parameters, options.ExogShockFnParamNames,jj);
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for ii=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                    end
                    [~,pi_z]=options.ExogShockFn(ExogShockFnParamsCell{:});
                    pi_z_J(:,:,jj)=gpuArray(pi_z);
                end
            else
                % whether or not pi_z depends on age, we can just do
                pi_z_J=pi_z.*ones(1,1,N_j,'gpuArray');
            end
        elseif gridpiboth==3
            % For value fn, both z_gridvals_J and pi_z_J
            z_gridvals_J=zeros(prod(n_z),length(n_z),'gpuArray');
            pi_z_J=zeros(prod(n_z),prod(n_z),'gpuArray');
            if isfield(options,'ExogShockFn')
                for jj=1:N_j
                    ExogShockFnParamsVec=CreateVectorFromParams(Parameters, options.ExogShockFnParamNames,jj);
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for ii=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                    end
                    [z_grid,pi_z]=options.ExogShockFn(ExogShockFnParamsCell{:});
                    pi_z_J(:,:,jj)=gpuArray(pi_z);
                    if all(size(z_grid)==[sum(n_z),1])
                        z_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(n_z,z_grid,1));
                    else % already joint-grid
                        z_gridvals_J(:,:,jj)=gpuArray(z_grid);
                    end
                end
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
        end
    end

elseif zdependsonptype==1

    for ii=1:length(Names_i)
        if ~isstruct(n_z)
            n_z_temp=n_z;
        else
            n_z_temp=n_z.(Names_i{ii});
        end
        if ~isstruct(N_j)
            N_j_temp=N_j;
        else
            N_j_temp=N_j.(Names_i{ii});
        end
        if ~isstruct(z_grid)
            z_grid_temp=z_grid;
        else
            z_grid_temp=z_grid.(Names_i{ii});
        end
        if ~isstruct(pi_z)
            pi_z_temp=pi_z;
        else
            pi_z_temp=pi_z.(Names_i{ii});
        end
        

        if prod(n_z_temp)==0
            z_gridvals_J.(Names_i{ii})=z_gridvals_J_temp;
            pi_z_J.(Names_i{ii})=pi_z_J_temp;
        else
            if gridpiboth==1 % for most FnsToEvaluate, we don't use pi_z
                pi_z_J.(Names_i{ii})=[];
                % Now just do z_gridvals_J
                z_gridvals_J_temp=zeros(prod(n_z_temp),length(n_z_temp),'gpuArray');
                if isfield(options,'ExogShockFn')
                    for jj=1:N_j_temp
                        ExogShockFnParamsVec=CreateVectorFromParams(Parameters, options.ExogShockFnParamNames.(Names_i{ii}),jj);
                        ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                        for pp=1:length(ExogShockFnParamsVec)
                            ExogShockFnParamsCell(pp,1)={ExogShockFnParamsVec(pp)};
                        end
                        temp=options.ExogShockFn.(Names_i{ii});
                        [z_grid_temp,~]=temp(ExogShockFnParamsCell{:});
                        if all(size(z_grid_temp)==[sum(n_z_temp),1])
                            z_gridvals_J_temp(:,:,jj)=gpuArray(CreateGridvals(n_z_temp,z_grid_temp,1));
                        else % already joint-grid
                            z_gridvals_J_temp(:,:,jj)=gpuArray(z_grid_temp,1);
                        end
                    end
                elseif ndims(z_grid_temp)==3 % already an age-dependent joint-grid
                    if all(size(z_grid_temp)==[prod(n_z_temp),length(n_z_temp),N_j_temp])
                        z_gridvals_J_temp=z_grid_temp;
                    end
                elseif all(size(z_grid_temp)==[sum(n_z_temp),N_j_temp]) % age-dependent grid
                    for jj=1:N_j_temp
                        z_gridvals_J_temp(:,:,jj)=CreateGridvals(n_z_temp,z_grid_temp(:,jj),1);
                    end
                elseif all(size(z_grid_temp)==[prod(n_z_temp),length(n_z_temp)]) % joint grid
                    z_gridvals_J_temp=z_grid_temp.*ones(1,1,N_j_temp,'gpuArray');
                elseif all(size(z_grid_temp)==[sum(n_z_temp),1]) % basic grid
                    z_gridvals_J_temp=CreateGridvals(n_z_temp,z_grid_temp,1).*ones(1,1,N_j_temp,'gpuArray');
                end
                z_gridvals_J.(Names_i{ii})=z_gridvals_J_temp;
            elseif gridpiboth==2 % For agent dist, we don't use grid
                z_gridvals_J.(Names_i{ii})=[];
                % Now just do pi_z_J
                pi_z_J_temp=zeros(prod(n_z_temp),prod(n_z_temp),'gpuArray');
                if isfield(options,'ExogShockFn')
                    for jj=1:N_j_temp
                        ExogShockFnParamsVec=CreateVectorFromParams(Parameters, options.ExogShockFnParamNames.(Names_i{ii}),jj);
                        ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                        for pp=1:length(ExogShockFnParamsVec)
                            ExogShockFnParamsCell(pp,1)={ExogShockFnParamsVec(pp)};
                        end
                        temp=options.ExogShockFn.(Names_i{ii});
                        [~,pi_z_temp]=temp(ExogShockFnParamsCell{:});
                        pi_z_J_temp(:,:,jj)=gpuArray(pi_z_temp);
                    end
                else
                    % whether or not pi_z depends on age, we can just do
                    pi_z_J_temp=pi_z_temp.*ones(1,1,N_j_temp,'gpuArray');
                end
                pi_z_J.(Names_i{ii})=pi_z_J_temp;
            elseif gridpiboth==3
                % For value fn, both z_gridvals_J and pi_z_J
                z_gridvals_J_temp=zeros(prod(n_z_temp),length(n_z_temp),'gpuArray');
                pi_z_J_temp=zeros(prod(n_z_temp),prod(n_z_temp),'gpuArray');
                if isfield(options,'ExogShockFn')
                    for jj=1:N_j_temp
                        ExogShockFnParamsVec=CreateVectorFromParams(Parameters, options.ExogShockFnParamNames.(Names_i{ii}),jj);
                        ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                        for pp=1:length(ExogShockFnParamsVec)
                            ExogShockFnParamsCell(pp,1)={ExogShockFnParamsVec(pp)};
                        end
                        temp=options.ExogShockFn.(Names_i{ii});
                        [z_grid_temp,pi_z_temp]=temp(ExogShockFnParamsCell{:});
                        pi_z_J_temp(:,:,jj)=gpuArray(pi_z_temp);
                        if all(size(z_grid_temp)==[sum(n_z_temp),1])
                            z_gridvals_J_temp(:,:,jj)=gpuArray(CreateGridvals(n_z_temp,z_grid_temp,1));
                        else % already joint-grid
                            z_gridvals_J_temp(:,:,jj)=gpuArray(z_grid_temp,1);
                        end
                    end
                elseif ndims(z_grid_temp)==3 % already an age-dependent joint-grid
                    if all(size(z_grid_temp)==[prod(n_z_temp),length(n_z_temp),N_j_temp])
                        z_gridvals_J_temp=z_grid_temp;
                    end
                    pi_z_J_temp=pi_z_temp;
                elseif all(size(z_grid_temp)==[sum(n_z_temp),N_j_temp]) % age-dependent grid
                    for jj=1:N_j_temp
                        z_gridvals_J_temp(:,:,jj)=CreateGridvals(n_z_temp,z_grid_temp(:,jj),1);
                    end
                    pi_z_J_temp=pi_z_temp;
                elseif all(size(z_grid_temp)==[prod(n_z_temp),length(n_z_temp)]) % joint grid
                    z_gridvals_J_temp=z_grid_temp.*ones(1,1,N_j_temp,'gpuArray');
                    pi_z_J_temp=pi_z_temp.*ones(1,1,N_j_temp,'gpuArray');
                elseif all(size(z_grid_temp)==[sum(n_z_temp),1]) % basic grid
                    z_gridvals_J_temp=CreateGridvals(n_z_temp,z_grid_temp,1).*ones(1,1,N_j_temp,'gpuArray');
                    pi_z_J_temp=pi_z_temp.*ones(1,1,N_j_temp,'gpuArray');
                end
                z_gridvals_J.(Names_i{ii})=z_gridvals_J_temp;
                pi_z_J.(Names_i{ii})=pi_z_J_temp;
            end
        end
    end
end


%% If using e variable, do same for this
if edependsonptype==0
    if prod(n_e)==0
        options.e_gridvals_J=[];
        options.pi_e_J=[];
    else
        if ~isfield(options,'e_grid') && ~isfield(options,'EiidShockFn')
            error('You are using an e (iid) variable, and so need to declare options.e_grid (options refers to either vfoptions or simoptions)')
        elseif ~isfield(options,'pi_e') && ~isfield(options,'EiidShockFn')
            error('You are using an e (iid) variable, and so need to declare options.pi_e (options refers to either vfoptions or simoptions)')
        end

        if gridpiboth==1 % for most FnsToEvaluate, we don't use pi_z
            options.pi_e_J=[];
            % Now just do e_gridvals_J
            options.e_gridvals_J=zeros(prod(options.n_e),length(options.n_e),'gpuArray');
            if isfield(options,'EiidShockFn')
                for jj=1:N_j
                    EiidShockFnParamsVec=CreateVectorFromParams(Parameters, options.EiidShockFnParamNames,jj);
                    EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                    for ii=1:length(EiidShockFnParamsVec)
                        EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                    end
                    [options.e_grid,~]=options.EiidShockFn(EiidShockFnParamsCell{:});
                    if all(size(options.e_grid)==[sum(options.n_e),1])
                        options.e_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(options.n_e,options.e_grid,1));
                    else % already joint-grid
                        options.e_gridvals_J(:,:,jj)=gpuArray(options.e_grid,1);
                    end
                end
            elseif ndims(options.e_grid)==3 % already an age-dependent joint-grid
                if all(size(options.e_grid)==[prod(options.n_e),length(options.n_e),N_j])
                    options.e_gridvals_J=options.e_grid;
                end
            elseif all(size(options.e_grid)==[sum(options.n_e),N_j]) % age-dependent stacked-grid
                for jj=1:N_j
                    options.e_gridvals_J(:,:,jj)=CreateGridvals(options.n_e,options.e_grid(:,jj),1);
                end
            elseif all(size(options.e_grid)==[prod(options.n_e),length(options.n_e)]) % joint grid
                options.e_gridvals_J=options.e_grid.*ones(1,1,N_j,'gpuArray');
            elseif all(size(options.e_grid)==[sum(options.n_e),1]) % basic grid
                options.e_gridvals_J=CreateGridvals(options.n_e,options.e_grid,1).*ones(1,1,N_j,'gpuArray');
            end
        elseif gridpiboth==2 % For agent dist, we don't use grid
            options.e_gridvals_J=[];
            % Now just do pi_e_J
            options.pi_e_J=zeros(prod(options.n_e),prod(options.n_e),'gpuArray');
            if isfield(options,'EiidShockFn')
                for jj=1:N_j
                    EiidShockFnParamsVec=CreateVectorFromParams(Parameters, options.EiidShockFnParamNames,jj);
                    EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                    for ii=1:length(EiidShockFnParamsVec)
                        EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                    end
                    [~,options.pi_e]=options.EiidShockFn(EiidShockFnParamsCell{:});
                    options.pi_e_J(:,jj)=gpuArray(options.pi_e);
                end
            else
                options.pi_e_J=options.pi_e.*ones(1,N_j,'gpuArray');
            end
        elseif gridpiboth==3
            % For value fn, both e_gridvals_J and pi_e_J
            options.e_gridvals_J=zeros(prod(options.n_e),length(options.n_e),'gpuArray');
            options.pi_e_J=zeros(prod(options.n_e),prod(options.n_e),'gpuArray');

            if isfield(options,'EiidShockFn')
                for jj=1:N_j
                    EiidShockFnParamsVec=CreateVectorFromParams(Parameters, options.EiidShockFnParamNames,jj);
                    EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                    for ii=1:length(EiidShockFnParamsVec)
                        EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                    end
                    [options.e_grid,options.pi_e]=options.EiidShockFn(EiidShockFnParamsCell{:});
                    options.pi_e_J(:,jj)=gpuArray(options.pi_e);
                    if all(size(options.e_grid)==[sum(options.n_e),1])
                        options.e_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(options.n_e,options.e_grid,1));
                    else % already joint-grid
                        options.e_gridvals_J(:,:,jj)=gpuArray(options.e_grid,1);
                    end
                end
            elseif ndims(options.e_grid)==3 % already an age-dependent joint-grid
                if all(size(options.e_grid)==[prod(options.n_e),length(options.n_e),N_j])
                    options.e_gridvals_J=options.e_grid;
                end
                options.pi_e_J=options.pi_e;
            elseif all(size(options.e_grid)==[sum(options.n_e),N_j]) % age-dependent stacked-grid
                for jj=1:N_j
                    options.e_gridvals_J(:,:,jj)=CreateGridvals(options.n_e,options.e_grid(:,jj),1);
                end
                options.pi_e_J=options.pi_e;
            elseif all(size(options.e_grid)==[prod(options.n_e),length(options.n_e)]) % joint grid
                options.e_gridvals_J=options.e_grid.*ones(1,1,N_j,'gpuArray');
                options.pi_e_J=options.pi_e.*ones(1,N_j,'gpuArray');
            elseif all(size(options.e_grid)==[sum(options.n_e),1]) % basic grid
                options.e_gridvals_J=CreateGridvals(options.n_e,options.e_grid,1).*ones(1,1,N_j,'gpuArray');
                options.pi_e_J=options.pi_e.*ones(1,N_j,'gpuArray');
            end
        end
    end
elseif edependsonptype==1
    for ii=1:length(Names_i)
        if ~isstruct(n_e)
            n_e_temp=n_e;
        else
            n_e_temp=n_e.(Names_i{ii});
        end
        if ~isstruct(N_j)
            N_j_temp=N_j;
        else
            N_j_temp=N_j.(Names_i{ii});
        end
        if ~isstruct(options.e_grid)
            e_grid_temp=options.e_grid;
        else
            e_grid_temp=options.e_grid.(Names_i{ii});
        end
        if ~isstruct(options.pi_e)
            pi_e_temp=options.pi_e;
        else
            pi_e_temp=options.pi_e.(Names_i{ii});
        end

        if prod(n_e_temp)==0
            options.e_gridvals_J.(Names_i{ii})=[];
            options.pi_e_J.(Names_i{ii})=[];
        else
            if gridpiboth==1 % for most FnsToEvaluate, we don't use pi_z
                options.pi_e_J.(Names_i{ii})=[];
                % Now just do e_gridvals_J
                e_gridvals_J_temp=zeros(prod(n_e_temp),length(n_e_temp),'gpuArray');
                if isfield(options,'EiidShockFn')
                    for jj=1:N_j_temp
                        EiidShockFnParamsVec=CreateVectorFromParams(Parameters, options.EiidShockFnParamNames.(Names_i{ii}),jj);
                        EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                        for pp=1:length(EiidShockFnParamsVec)
                            EiidShockFnParamsCell(pp,1)={EiidShockFnParamsVec(pp)};
                        end
                        temp=options.EiidShockFn.(Names_i{ii});
                        [e_grid_temp,~]=temp(EiidShockFnParamsCell{:});
                        if all(size(e_grid_temp)==[sum(n_e_temp),1])
                            e_gridvals_J_temp(:,:,jj)=gpuArray(CreateGridvals(n_e_temp,e_grid_temp,1));
                        else % already joint-grid
                            e_gridvals_J_temp(:,:,jj)=gpuArray(e_grid_temp,1);
                        end
                    end
                elseif ndims(options.e_grid)==3 % already an age-dependent joint-grid
                    if all(size(e_grid_temp)==[prod(n_e_temp),length(n_e_temp),N_j_temp])
                        e_gridvals_J_temp=e_grid_temp;
                    end
                elseif all(size(e_grid_temp)==[sum(n_e_temp),N_j_temp]) % age-dependent stacked-grid
                    for jj=1:N_j_temp
                        e_gridvals_J_temp(:,:,jj)=CreateGridvals(n_e_temp,e_grid_temp(:,jj),1);
                    end
                elseif all(size(e_grid_temp)==[prod(n_e_temp),length(n_e_temp)]) % joint grid
                    e_gridvals_J_temp=e_grid_temp.*ones(1,1,N_j_temp,'gpuArray');
                elseif all(size(e_grid_temp)==[sum(n_e_temp),1]) % basic grid
                    e_gridvals_J_temp=CreateGridvals(n_e_temp,e_grid_temp,1).*ones(1,1,N_j_temp,'gpuArray');
                end
                options.e_gridvals_J.(Names_i{ii})=e_gridvals_J_temp;
            elseif gridpiboth==2 % For agent dist, we don't use grid
                options.e_gridvals_J.(Names_i{ii})=[];
                % Now just do pi_e_J
                pi_e_J_temp=zeros(prod(n_e_temp),prod(n_e_temp),'gpuArray');
                if isfield(options,'EiidShockFn')
                    for jj=1:N_j_temp
                        EiidShockFnParamsVec=CreateVectorFromParams(Parameters, options.EiidShockFnParamNames.(Names_i{ii}),jj);
                        EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                        for pp=1:length(EiidShockFnParamsVec)
                            EiidShockFnParamsCell(pp,1)={EiidShockFnParamsVec(pp)};
                        end
                        temp=options.EiidShockFn.(Names_i{ii});
                        [~,pi_e_temp]=temp(EiidShockFnParamsCell{:});
                        pi_e_J_temp(:,jj)=gpuArray(pi_e_temp);
                    end
                else
                    pi_e_J_temp=pi_e_temp.*ones(1,N_j_temp,'gpuArray');
                end
                options.pi_e_J.(Names_i{ii})=pi_e_J_temp;
            elseif gridpiboth==3
                % For value fn, both e_gridvals_J and pi_e_J
                e_gridvals_J_temp=zeros(prod(n_e_temp),length(n_e_temp),'gpuArray');
                pi_e_J_temp=zeros(prod(n_e_temp),prod(n_e_temp),'gpuArray');

                if isfield(options,'EiidShockFn')
                    for jj=1:N_j_temp
                        EiidShockFnParamsVec=CreateVectorFromParams(Parameters, options.EiidShockFnParamNames.(Names_i{ii}),jj);
                        EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                        for pp=1:length(EiidShockFnParamsVec)
                            EiidShockFnParamsCell(pp,1)={EiidShockFnParamsVec(pp)};
                        end
                        temp=options.EiidShockFn.(Names_i{ii});
                        [e_grid_temp,pi_e_temp]=temp(EiidShockFnParamsCell{:});
                        pi_e_J_temp(:,jj)=gpuArray(pi_e_temp);
                        if all(size(e_grid_temp)==[sum(n_e_temp),1])
                            e_gridvals_J_temp(:,:,jj)=gpuArray(CreateGridvals(n_e_temp,e_grid_temp,1));
                        else % already joint-grid
                            e_gridvals_J_temp(:,:,jj)=gpuArray(e_grid_temp,1);
                        end
                    end
                elseif ndims(e_grid_temp)==3 % already an age-dependent joint-grid
                    if all(size(e_grid_temp)==[prod(n_e_temp),length(n_e_temp),N_j_temp])
                        e_gridvals_J_temp=e_grid_temp;
                    end
                    pi_e_J_temp=pi_e_temp;
                elseif all(size(e_grid_temp)==[sum(n_e_temp),N_j_temp]) % age-dependent stacked-grid
                    for jj=1:N_j_temp
                        e_gridvals_J_temp(:,:,jj)=CreateGridvals(n_e_temp,e_grid_temp(:,jj),1);
                    end
                    pi_e_J_temp=pi_e_temp;
                elseif all(size(e_grid_temp)==[prod(n_e_temp),length(n_e_temp)]) % joint grid
                    e_gridvals_J_temp=e_grid_temp.*ones(1,1,N_j_temp,'gpuArray');
                    pi_e_J_temp=pi_e_temp.*ones(1,N_j_temp,'gpuArray');
                elseif all(size(e_grid_temp)==[sum(n_e_temp),1]) % basic grid
                    e_gridvals_J_temp=CreateGridvals(n_e_temp,e_grid_temp,1).*ones(1,1,N_j_temp,'gpuArray');
                    pi_e_J_temp=pi_e_temp.*ones(1,N_j_temp,'gpuArray');
                end
                options.e_gridvals_J.(Names_i{ii})=e_gridvals_J_temp;
                options.pi_e_J.(Names_i{ii})=pi_e_J_temp;
            end
        end
    end
end


%% We have evaluated ExogShockFn and EiidShockFn, so now remove them to keep things simple/clean
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