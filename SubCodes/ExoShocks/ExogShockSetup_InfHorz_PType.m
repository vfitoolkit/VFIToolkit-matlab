function [z_gridvals, pi_z, options]=ExogShockSetup_InfHorz_PType(n_z,z_grid,pi_z,Names_i,Parameters,options,gridpiboth)
% Convert z and e to joint-grids and transition matrix
% options will either be vfoptions or simoptions
% output: z_gridvals, pi_z, options.e_gridvals, options.pi_e
% All outputs are made dependent on permanent type (as that way the rest of the codes can just assume this)

% gridpiboth=3: sometimes (value fn iter) we want both grid and transition probabilities
% gridpiboth=2: sometimes (agent dist)    we want just transition probabilities
% gridpiboth=1: sometimes (FnsToEvaluate) we want just grid

% Note: must be run after Names_i is setup (so that this is Names_i and not N_i)

% Accepted input shapes:
%   z_grid:
%     struct keyed by Names_i              each field is one of the per-ptype shapes below
%     [sum(n_z), 1]                        stacked column grid for markov z (ptype-independent)
%     [prod(n_z), length(n_z)]             joint grid for markov z (ptype-independent)
%     [prod(n_z), N_i]                     stacked column grid for markov z, ptype-dependent (last dim is ptype)
%     [prod(n_z), length(n_z), N_i]        joint grid for markov z, ptype-dependent
%   pi_z:
%     struct keyed by Names_i              each field is one of the per-ptype shapes below
%     [prod(n_z), prod(n_z)]               transition matrix for markov z (ptype-independent)
%     [prod(n_z), prod(n_z), N_i]          transition matrix for markov z, ptype-dependent (last dim is ptype)
%   options.e_grid:
%     struct keyed by Names_i              each field is one of the per-ptype shapes below
%     [sum(n_e), 1]                        stacked column grid for iid e (ptype-independent)
%     [prod(n_e), length(n_e)]             joint grid for iid e (ptype-independent)
%     [prod(n_e), N_i]                     stacked column grid for iid e, ptype-dependent (last dim is ptype)
%     [prod(n_e), length(n_e), N_i]        joint grid for iid e, ptype-dependent
%   options.pi_e:
%     struct keyed by Names_i              each field is one of the per-ptype shapes below
%     [prod(n_e), 1]                       iid distribution (ptype-independent)
%     [prod(n_e), N_i]                     iid distribution, ptype-dependent (last dim is ptype)
%
% If options.ExogShockFn is supplied, it is called to produce [z_grid, pi_z]
% in one of the ptype-independent shapes above (and, if ExogShockFn is itself
% a struct keyed by Names_i, once per ptype). The raw z_grid / pi_z inputs
% are then ignored. Likewise options.EiidShockFn for the iid e variables.
%
% Stacked column grid: each of the underlying univariate grids written one
% beneath the next in a single column of length sum(n_z). Compact, but the
% joint state space is only implicit. For example with two markov variables
% of sizes n_z=[3,2], the column contains the 3 values of z1 followed by the
% 2 values of z2, giving a 5x1 vector. The ptype-dependent version adds a
% trailing dimension of length N_i.
%
% Joint grid: every point in the product space listed explicitly, one per
% row, with each variable in its own column. The number of rows is
% prod(n_z) and the number of columns is length(n_z). Continuing the
% example, a joint grid is 6x2: each row pairs one z1 value with one z2
% value, covering all 6 combinations. The ptype-dependent version adds a
% trailing dimension of length N_i.
%
% Output shapes (function returns):
% Outputs are always structs keyed by Names_i (regardless of whether the
% underlying inputs were ptype-dependent — for ptype-independent inputs the
% same per-ptype payload is replicated across all fields). This lets
% downstream code uniformly treat the outputs as ptype-dependent.
%   z_gridvals:
%     struct keyed by Names_i    each field is [prod(n_z_i), length(n_z_i)] (joint grid for ptype i),
%                                or [] if that ptype has prod(n_z_i)==0 or gridpiboth==2
%   pi_z:
%     struct keyed by Names_i    each field is [prod(n_z_i), prod(n_z_i)] transition matrix,
%                                or [] if that ptype has prod(n_z_i)==0 or gridpiboth==1
%   options.e_gridvals:
%     struct keyed by Names_i    each field is [prod(n_e_i), length(n_e_i)] joint grid for iid e,
%                                or [] if that ptype has prod(n_e_i)==0 or gridpiboth==2
%   options.pi_e:
%     struct keyed by Names_i    each field is [prod(n_e_i), 1] iid distribution,
%                                or [] if that ptype has prod(n_e_i)==0 or gridpiboth==1
%
% In the ptype-independent case (n_z / n_e is a scalar or vector, not a
% struct), n_z_i / n_e_i is just n_z / n_e — i.e. all per-ptype fields share
% one common shape.

%% Check what we are doing in terms of dependence on permanent type for z
zdependsonptype=0;
if isstruct(z_grid) || isstruct(pi_z)
    zdependsonptype=1;
elseif size(z_grid,ndims(z_grid))==length(Names_i) || size(pi_z,ndims(pi_z))==length(Names_i) % last dimension of z_grid/pi_z is of length N_i
    zdependsonptype=2;
    N_i=length(Names_i);
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
    elseif size(options.e_grid,ndims(options.e_grid))==length(Names_i) || size(options.pi_e,ndims(options.pi_e))==length(Names_i) % last dimension of e_grid/pi_e is of length N_i
        edependsonptype=2;
        N_i=length(Names_i);
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
    if prod(n_z)==0
        z_gridvals=[];
        pi_z=[];
    else
        if gridpiboth==1 % for most FnsToEvaluate, we don't use pi_z
            pi_z=[];
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
            elseif all(size(z_grid)==[sum(n_z),1]) % stacked column grid
                z_gridvals=CreateGridvals(n_z,z_grid,1);
            else
                error('z_grid size does not match any expected shape')
            end
            z_gridvals=gpuArray(z_gridvals);
        elseif gridpiboth==2 % For agent dist, we don't use grid
            z_gridvals=[];
            if isfield(options,'ExogShockFn')
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, options.ExogShockFnParamNames);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [~,pi_z]=options.ExogShockFn(ExogShockFnParamsCell{:});
            end
            pi_z=gather(pi_z); % Agent distribution iteration is performed on cpu
        elseif gridpiboth==3
            % For value fn, both z_gridvals and pi_z
            if isfield(options,'ExogShockFn')
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, options.ExogShockFnParamNames);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [z_grid,pi_z]=options.ExogShockFn(ExogShockFnParamsCell{:});
            end
            if all(size(z_grid)==[prod(n_z),length(n_z)]) % joint grid
                z_gridvals=z_grid;
            elseif all(size(z_grid)==[sum(n_z),1]) % stacked column grid
                z_gridvals=CreateGridvals(n_z,z_grid,1);
            else
                error('z_grid size does not match any expected shape')
            end
            z_gridvals=gpuArray(z_gridvals);
            pi_z=gpuArray(pi_z);
        end
    end
    % Broadcast bare arrays into struct keyed by Names_i, so that downstream
    % code can always treat the outputs as ptype-dependent.
    z_gridvals_bare=z_gridvals;
    pi_z_bare=pi_z;
    z_gridvals=struct();
    pi_z=struct();
    for ii=1:length(Names_i)
        z_gridvals.(Names_i{ii})=z_gridvals_bare;
        pi_z.(Names_i{ii})=pi_z_bare;
    end

elseif zdependsonptype==1

    for ii=1:length(Names_i)
        if ~isstruct(n_z)
            n_z_temp=n_z;
        else
            n_z_temp=n_z.(Names_i{ii});
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
            z_gridvals_out.(Names_i{ii})=[];
            pi_z_out.(Names_i{ii})=[];
        else
            if gridpiboth==1 % for most FnsToEvaluate, we don't use pi_z
                pi_z_out.(Names_i{ii})=[];
                if isfield(options,'ExogShockFn')
                    ExogShockFnParamsVec=CreateVectorFromParams(Parameters, options.ExogShockFnParamNames.(Names_i{ii}));
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for pp=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(pp,1)={ExogShockFnParamsVec(pp)};
                    end
                    temp=options.ExogShockFn.(Names_i{ii});
                    [z_grid_temp,~]=temp(ExogShockFnParamsCell{:});
                end
                if all(size(z_grid_temp)==[prod(n_z_temp),length(n_z_temp)]) % joint grid
                    z_gridvals_temp=z_grid_temp;
                elseif all(size(z_grid_temp)==[sum(n_z_temp),1]) % stacked column grid
                    z_gridvals_temp=CreateGridvals(n_z_temp,z_grid_temp,1);
                else
                    error('z_grid_temp size does not match any expected shape')
                end
                z_gridvals_out.(Names_i{ii})=gpuArray(z_gridvals_temp);
            elseif gridpiboth==2 % For agent dist, we don't use grid
                z_gridvals_out.(Names_i{ii})=[];
                if isfield(options,'ExogShockFn')
                    ExogShockFnParamsVec=CreateVectorFromParams(Parameters, options.ExogShockFnParamNames.(Names_i{ii}));
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for pp=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(pp,1)={ExogShockFnParamsVec(pp)};
                    end
                    temp=options.ExogShockFn.(Names_i{ii});
                    [~,pi_z_temp]=temp(ExogShockFnParamsCell{:});
                end
                pi_z_out.(Names_i{ii})=gather(pi_z_temp); % Agent distribution iteration is performed on cpu
            elseif gridpiboth==3
                if isfield(options,'ExogShockFn')
                    ExogShockFnParamsVec=CreateVectorFromParams(Parameters, options.ExogShockFnParamNames.(Names_i{ii}));
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for pp=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(pp,1)={ExogShockFnParamsVec(pp)};
                    end
                    temp=options.ExogShockFn.(Names_i{ii});
                    [z_grid_temp,pi_z_temp]=temp(ExogShockFnParamsCell{:});
                end
                if all(size(z_grid_temp)==[prod(n_z_temp),length(n_z_temp)]) % joint grid
                    z_gridvals_temp=z_grid_temp;
                elseif all(size(z_grid_temp)==[sum(n_z_temp),1]) % stacked column grid
                    z_gridvals_temp=CreateGridvals(n_z_temp,z_grid_temp,1);
                else
                    error('z_grid_temp size does not match any expected shape')
                end
                z_gridvals_out.(Names_i{ii})=gpuArray(z_gridvals_temp);
                pi_z_out.(Names_i{ii})=gpuArray(pi_z_temp);
            end
        end
    end
    z_gridvals=z_gridvals_out;
    pi_z=pi_z_out;

elseif zdependsonptype==2 % dependence of ptype via last dimension of matrix for z_grid &/or pi_z

    for ii=1:length(Names_i)
        if ~isstruct(n_z)
            n_z_temp=n_z;
        else
            n_z_temp=n_z.(Names_i{ii});
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
            z_gridvals_out.(Names_i{ii})=[];
            pi_z_out.(Names_i{ii})=[];
        else
            if gridpiboth==1 % for most FnsToEvaluate, we don't use pi_z
                pi_z_out.(Names_i{ii})=[];
                if isfield(options,'ExogShockFn')
                    ExogShockFnParamsVec=CreateVectorFromParams(Parameters, options.ExogShockFnParamNames.(Names_i{ii}));
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for pp=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(pp,1)={ExogShockFnParamsVec(pp)};
                    end
                    temp=options.ExogShockFn.(Names_i{ii});
                    [z_grid_temp,~]=temp(ExogShockFnParamsCell{:});
                    if all(size(z_grid_temp)==[sum(n_z_temp),1])
                        z_gridvals_temp=CreateGridvals(n_z_temp,z_grid_temp,1);
                    else % already joint-grid
                        z_gridvals_temp=z_grid_temp;
                    end
                elseif ndims(z_grid_temp)==3
                    if all(size(z_grid_temp)==[prod(n_z_temp),length(n_z_temp),N_i])
                        z_gridvals_temp=z_grid_temp(:,:,ii);
                    else
                        error('z_grid_temp is 3D but its size does not match [prod(n_z_temp), length(n_z_temp), N_i]')
                    end
                elseif ndims(z_grid_temp)==2
                    if all(size(z_grid_temp)==[prod(n_z_temp),length(n_z_temp)]) % joint grid
                        z_gridvals_temp=z_grid_temp;
                    elseif all(size(z_grid_temp)==[prod(n_z_temp),N_i])
                        z_gridvals_temp=z_grid_temp(:,ii);
                    elseif all(size(z_grid_temp)==[sum(n_z_temp),1]) % stacked column grid
                        z_gridvals_temp=CreateGridvals(n_z_temp,z_grid_temp,1);
                    else
                        error('z_grid_temp is 2D but its size does not match any expected shape')
                    end
                else
                    error('z_grid_temp has unexpected number of dimensions (expected 2 or 3)')
                end
                z_gridvals_out.(Names_i{ii})=gpuArray(z_gridvals_temp);
            elseif gridpiboth==2 % For agent dist, we don't use grid
                z_gridvals_out.(Names_i{ii})=[];
                if isfield(options,'ExogShockFn')
                    ExogShockFnParamsVec=CreateVectorFromParams(Parameters, options.ExogShockFnParamNames.(Names_i{ii}));
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for pp=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(pp,1)={ExogShockFnParamsVec(pp)};
                    end
                    temp=options.ExogShockFn.(Names_i{ii});
                    [~,pi_z_temp]=temp(ExogShockFnParamsCell{:});
                elseif size(pi_z_temp,ndims(pi_z_temp))==N_i
                    otherdims = repmat({':'},1,ndims(pi_z_temp)-1);
                    pi_z_temp=pi_z_temp(otherdims{:},ii);
                end
                pi_z_out.(Names_i{ii})=gather(pi_z_temp); % Agent distribution iteration is performed on cpu
            elseif gridpiboth==3
                if isfield(options,'ExogShockFn')
                    ExogShockFnParamsVec=CreateVectorFromParams(Parameters, options.ExogShockFnParamNames.(Names_i{ii}));
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for pp=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(pp,1)={ExogShockFnParamsVec(pp)};
                    end
                    temp=options.ExogShockFn.(Names_i{ii});
                    [z_grid_temp,pi_z_temp]=temp(ExogShockFnParamsCell{:});
                    if all(size(z_grid_temp)==[sum(n_z_temp),1])
                        z_gridvals_temp=CreateGridvals(n_z_temp,z_grid_temp,1);
                    else % already joint-grid
                        z_gridvals_temp=z_grid_temp;
                    end
                else
                    if ndims(z_grid_temp)==3
                        if all(size(z_grid_temp)==[prod(n_z_temp),length(n_z_temp),N_i])
                            z_gridvals_temp=z_grid_temp(:,:,ii);
                        else
                            error('z_grid_temp is 3D but its size does not match [prod(n_z_temp), length(n_z_temp), N_i]')
                        end
                    elseif ndims(z_grid_temp)==2
                        if all(size(z_grid_temp)==[prod(n_z_temp),length(n_z_temp)]) % joint grid
                            z_gridvals_temp=z_grid_temp;
                        elseif all(size(z_grid_temp)==[prod(n_z_temp),N_i])
                            z_gridvals_temp=z_grid_temp(:,ii);
                        elseif all(size(z_grid_temp)==[sum(n_z_temp),1]) % stacked column grid
                            z_gridvals_temp=CreateGridvals(n_z_temp,z_grid_temp,1);
                        else
                            error('z_grid_temp is 2D but its size does not match any expected shape')
                        end
                    else
                        error('z_grid_temp has unexpected number of dimensions (expected 2 or 3)')
                    end
                    if size(pi_z_temp,ndims(pi_z_temp))==N_i
                        otherdims = repmat({':'},1,ndims(pi_z_temp)-1);
                        pi_z_temp=pi_z_temp(otherdims{:},ii);
                    end
                end
                z_gridvals_out.(Names_i{ii})=gpuArray(z_gridvals_temp);
                pi_z_out.(Names_i{ii})=gpuArray(pi_z_temp);
            end
        end
    end
    z_gridvals=z_gridvals_out;
    pi_z=pi_z_out;
end


%% If using e variable, do same for this
if edependsonptype==0
    if prod(n_e)==0
        options.e_gridvals=[];
        options.pi_e=[];
    else
        if gridpiboth==1 % for most FnsToEvaluate, we don't use pi_e
            options.pi_e=[];
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
            elseif all(size(options.e_grid)==[sum(options.n_e),1]) % stacked column grid
                options.e_gridvals=CreateGridvals(options.n_e,options.e_grid,1);
            else
                error('options.e_grid size does not match any expected shape')
            end
            options.e_gridvals=gpuArray(options.e_gridvals);
        elseif gridpiboth==2 % For agent dist, we don't use grid
            options.e_gridvals=[];
            if isfield(options,'EiidShockFn')
                EiidShockFnParamsVec=CreateVectorFromParams(Parameters, options.EiidShockFnParamNames);
                EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                for ii=1:length(EiidShockFnParamsVec)
                    EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                end
                [~,options.pi_e]=options.EiidShockFn(EiidShockFnParamsCell{:});
            end
            options.pi_e=gather(options.pi_e); % Agent distribution iteration is performed on cpu
        elseif gridpiboth==3
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
            elseif all(size(options.e_grid)==[sum(options.n_e),1]) % stacked column grid
                options.e_gridvals=CreateGridvals(options.n_e,options.e_grid,1);
            else
                error('options.e_grid size does not match any expected shape')
            end
            options.e_gridvals=gpuArray(options.e_gridvals);
            options.pi_e=gpuArray(options.pi_e);
        end
    end
    % Broadcast bare arrays into struct keyed by Names_i.
    e_gridvals_bare=options.e_gridvals;
    pi_e_bare=options.pi_e;
    options.e_gridvals=struct();
    options.pi_e=struct();
    for ii=1:length(Names_i)
        options.e_gridvals.(Names_i{ii})=e_gridvals_bare;
        options.pi_e.(Names_i{ii})=pi_e_bare;
    end

elseif edependsonptype==1

    for ii=1:length(Names_i)
        if ~isstruct(n_e)
            n_e_temp=n_e;
        else
            n_e_temp=n_e.(Names_i{ii});
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
            e_gridvals_out.(Names_i{ii})=[];
            pi_e_out.(Names_i{ii})=[];
        else
            if gridpiboth==1 % for most FnsToEvaluate, we don't use pi_e
                pi_e_out.(Names_i{ii})=[];
                if isfield(options,'EiidShockFn')
                    EiidShockFnParamsVec=CreateVectorFromParams(Parameters, options.EiidShockFnParamNames.(Names_i{ii}));
                    EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                    for pp=1:length(EiidShockFnParamsVec)
                        EiidShockFnParamsCell(pp,1)={EiidShockFnParamsVec(pp)};
                    end
                    temp=options.EiidShockFn.(Names_i{ii});
                    [e_grid_temp,~]=temp(EiidShockFnParamsCell{:});
                end
                if all(size(e_grid_temp)==[prod(n_e_temp),length(n_e_temp)]) % joint grid
                    e_gridvals_temp=e_grid_temp;
                elseif all(size(e_grid_temp)==[sum(n_e_temp),1]) % stacked column grid
                    e_gridvals_temp=CreateGridvals(n_e_temp,e_grid_temp,1);
                else
                    error('e_grid_temp size does not match any expected shape')
                end
                e_gridvals_out.(Names_i{ii})=gpuArray(e_gridvals_temp);
            elseif gridpiboth==2 % For agent dist, we don't use grid
                e_gridvals_out.(Names_i{ii})=[];
                if isfield(options,'EiidShockFn')
                    EiidShockFnParamsVec=CreateVectorFromParams(Parameters, options.EiidShockFnParamNames.(Names_i{ii}));
                    EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                    for pp=1:length(EiidShockFnParamsVec)
                        EiidShockFnParamsCell(pp,1)={EiidShockFnParamsVec(pp)};
                    end
                    temp=options.EiidShockFn.(Names_i{ii});
                    [~,pi_e_temp]=temp(EiidShockFnParamsCell{:});
                end
                pi_e_out.(Names_i{ii})=gather(pi_e_temp); % Agent distribution iteration is performed on cpu
            elseif gridpiboth==3
                if isfield(options,'EiidShockFn')
                    EiidShockFnParamsVec=CreateVectorFromParams(Parameters, options.EiidShockFnParamNames.(Names_i{ii}));
                    EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                    for pp=1:length(EiidShockFnParamsVec)
                        EiidShockFnParamsCell(pp,1)={EiidShockFnParamsVec(pp)};
                    end
                    temp=options.EiidShockFn.(Names_i{ii});
                    [e_grid_temp,pi_e_temp]=temp(EiidShockFnParamsCell{:});
                end
                if all(size(e_grid_temp)==[prod(n_e_temp),length(n_e_temp)]) % joint grid
                    e_gridvals_temp=e_grid_temp;
                elseif all(size(e_grid_temp)==[sum(n_e_temp),1]) % stacked column grid
                    e_gridvals_temp=CreateGridvals(n_e_temp,e_grid_temp,1);
                else
                    error('e_grid_temp size does not match any expected shape')
                end
                e_gridvals_out.(Names_i{ii})=gpuArray(e_gridvals_temp);
                pi_e_out.(Names_i{ii})=gpuArray(pi_e_temp);
            end
        end
    end
    options.e_gridvals=e_gridvals_out;
    options.pi_e=pi_e_out;

elseif edependsonptype==2 % dependence of ptype via last dimension of matrix for e_grid &/or pi_e

    for ii=1:length(Names_i)
        if ~isstruct(n_e)
            n_e_temp=n_e;
        else
            n_e_temp=n_e.(Names_i{ii});
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
            e_gridvals_out.(Names_i{ii})=[];
            pi_e_out.(Names_i{ii})=[];
        else
            if gridpiboth==1 % for most FnsToEvaluate, we don't use pi_e
                pi_e_out.(Names_i{ii})=[];
                if isfield(options,'EiidShockFn')
                    EiidShockFnParamsVec=CreateVectorFromParams(Parameters, options.EiidShockFnParamNames.(Names_i{ii}));
                    EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                    for pp=1:length(EiidShockFnParamsVec)
                        EiidShockFnParamsCell(pp,1)={EiidShockFnParamsVec(pp)};
                    end
                    temp=options.EiidShockFn.(Names_i{ii});
                    [e_grid_temp,~]=temp(EiidShockFnParamsCell{:});
                    if all(size(e_grid_temp)==[sum(n_e_temp),1])
                        e_gridvals_temp=CreateGridvals(n_e_temp,e_grid_temp,1);
                    else % already joint-grid
                        e_gridvals_temp=e_grid_temp;
                    end
                elseif ndims(e_grid_temp)==3
                    if all(size(e_grid_temp)==[prod(n_e_temp),length(n_e_temp),N_i])
                        e_gridvals_temp=e_grid_temp(:,:,ii);
                    else
                        error('e_grid_temp is 3D but its size does not match [prod(n_e_temp), length(n_e_temp), N_i]')
                    end
                elseif ndims(e_grid_temp)==2
                    if all(size(e_grid_temp)==[prod(n_e_temp),length(n_e_temp)]) % joint grid
                        e_gridvals_temp=e_grid_temp;
                    elseif all(size(e_grid_temp)==[prod(n_e_temp),N_i])
                        e_gridvals_temp=e_grid_temp(:,ii);
                    elseif all(size(e_grid_temp)==[sum(n_e_temp),1]) % stacked column grid
                        e_gridvals_temp=CreateGridvals(n_e_temp,e_grid_temp,1);
                    else
                        error('e_grid_temp is 2D but its size does not match any expected shape')
                    end
                else
                    error('e_grid_temp has unexpected number of dimensions (expected 2 or 3)')
                end
                e_gridvals_out.(Names_i{ii})=gpuArray(e_gridvals_temp);
            elseif gridpiboth==2 % For agent dist, we don't use grid
                e_gridvals_out.(Names_i{ii})=[];
                if isfield(options,'EiidShockFn')
                    EiidShockFnParamsVec=CreateVectorFromParams(Parameters, options.EiidShockFnParamNames.(Names_i{ii}));
                    EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                    for pp=1:length(EiidShockFnParamsVec)
                        EiidShockFnParamsCell(pp,1)={EiidShockFnParamsVec(pp)};
                    end
                    temp=options.EiidShockFn.(Names_i{ii});
                    [~,pi_e_temp]=temp(EiidShockFnParamsCell{:});
                elseif size(pi_e_temp,ndims(pi_e_temp))==N_i
                    otherdims = repmat({':'},1,ndims(pi_e_temp)-1);
                    pi_e_temp=pi_e_temp(otherdims{:},ii);
                end
                pi_e_out.(Names_i{ii})=gather(pi_e_temp); % Agent distribution iteration is performed on cpu
            elseif gridpiboth==3
                if isfield(options,'EiidShockFn')
                    EiidShockFnParamsVec=CreateVectorFromParams(Parameters, options.EiidShockFnParamNames.(Names_i{ii}));
                    EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                    for pp=1:length(EiidShockFnParamsVec)
                        EiidShockFnParamsCell(pp,1)={EiidShockFnParamsVec(pp)};
                    end
                    temp=options.EiidShockFn.(Names_i{ii});
                    [e_grid_temp,pi_e_temp]=temp(EiidShockFnParamsCell{:});
                    if all(size(e_grid_temp)==[sum(n_e_temp),1])
                        e_gridvals_temp=CreateGridvals(n_e_temp,e_grid_temp,1);
                    else % already joint-grid
                        e_gridvals_temp=e_grid_temp;
                    end
                else
                    if ndims(e_grid_temp)==3
                        if all(size(e_grid_temp)==[prod(n_e_temp),length(n_e_temp),N_i])
                            e_gridvals_temp=e_grid_temp(:,:,ii);
                        else
                            error('e_grid_temp is 3D but its size does not match [prod(n_e_temp), length(n_e_temp), N_i]')
                        end
                    elseif ndims(e_grid_temp)==2
                        if all(size(e_grid_temp)==[prod(n_e_temp),length(n_e_temp)]) % joint grid
                            e_gridvals_temp=e_grid_temp;
                        elseif all(size(e_grid_temp)==[prod(n_e_temp),N_i])
                            e_gridvals_temp=e_grid_temp(:,ii);
                        elseif all(size(e_grid_temp)==[sum(n_e_temp),1]) % stacked column grid
                            e_gridvals_temp=CreateGridvals(n_e_temp,e_grid_temp,1);
                        else
                            error('e_grid_temp is 2D but its size does not match any expected shape')
                        end
                    else
                        error('e_grid_temp has unexpected number of dimensions (expected 2 or 3)')
                    end
                    if size(pi_e_temp,ndims(pi_e_temp))==N_i
                        otherdims = repmat({':'},1,ndims(pi_e_temp)-1);
                        pi_e_temp=pi_e_temp(otherdims{:},ii);
                    end
                end
                e_gridvals_out.(Names_i{ii})=gpuArray(e_gridvals_temp);
                pi_e_out.(Names_i{ii})=gpuArray(pi_e_temp);
            end
        end
    end
    options.e_gridvals=e_gridvals_out;
    options.pi_e=pi_e_out;
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
