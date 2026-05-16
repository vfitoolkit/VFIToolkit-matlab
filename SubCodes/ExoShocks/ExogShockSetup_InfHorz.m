function [z_gridvals, pi_z, options]=ExogShockSetup_InfHorz(n_z,z_grid,pi_z,Parameters,options,gridpiboth)
% Convert z and e to joint-grids and transtion matrix
% options will either be vfoptions or simoptions
% output: z_gridvals, pi_z, options.e_gridvals, options.pi_e

% gridpiboth=3: sometimes (value fn iter) we want both grid and transition probabilities
% gridpiboth=2: sometimes (agent dist)    we want just transition probabilities
% gridpiboth=1: sometimes (FnsToEvaluate) we want just grid

% Accepted input shapes:
%   z_grid:
%     [sum(n_z), 1]                 stacked column grid for markov z
%     [prod(n_z), length(n_z)]      joint grid for markov z
%   pi_z:
%     [prod(n_z), prod(n_z)]        transition matrix for markov z (rows = from-state, cols = to-state)
%   options.e_grid:
%     [sum(n_e), 1]                 stacked column grid for iid e
%     [prod(n_e), length(n_e)]      joint grid for iid e
%   options.pi_e:
%     [prod(n_e), 1]                iid distribution (column vector of probabilities)
%
% If options.ExogShockFn is supplied, it is called to produce [z_grid, pi_z]
% in one of the shapes above; the raw z_grid / pi_z inputs are then ignored.
% Likewise, if options.EiidShockFn is supplied, it produces [options.e_grid,
% options.pi_e] in one of the shapes above; the raw options.e_grid /
% options.pi_e inputs are then ignored.
%
% Stacked column grid: each of the underlying univariate grids written one
% beneath the next in a single column of length sum(n_z). Compact, but the
% joint state space is only implicit. For example with two markov variables
% of sizes n_z=[3,2], the column contains the 3 values of z1 followed by the
% 2 values of z2, giving a 5x1 vector.
%
% Joint grid: every point in the product space listed explicitly, one per
% row, with each variable in its own column. The number of rows is
% prod(n_z) and the number of columns is length(n_z). Continuing the
% example, a joint grid is 6x2: each row pairs one z1 value with one z2
% value, covering all 6 combinations.
%
% Output shapes (function returns):
%   z_gridvals:
%     [prod(n_z), length(n_z)]  joint grid (always in joint form, regardless of input shape)
%     []                        if gridpiboth==2 (only pi_z requested) or prod(n_z)==0
%   pi_z:
%     [prod(n_z), prod(n_z)]    transition matrix (rows = from-state, cols = to-state)
%     []                        if gridpiboth==1 (only grid requested) or prod(n_z)==0
%   options.e_gridvals:
%     [prod(n_e), length(n_e)]  joint grid for iid e
%     []                        if gridpiboth==2 or no e variable
%   options.pi_e:
%     [prod(n_e), 1]            iid distribution (column of probabilities)
%     []                        if gridpiboth==1 or no e variable

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
        else
            error('z_grid size does not match any expected shape')
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
        pi_z=gather(pi_z); % Agent distribution iteration is performed on cpu
    elseif gridpiboth==3
        % For value fn, both z_gridvals_J and pi_z_J
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
        elseif all(size(z_grid)==[sum(n_z),1]) % basic grid
            z_gridvals=CreateGridvals(n_z,z_grid,1);
        else
            error('z_grid size does not match any expected shape')
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
        else
            error('options.e_grid size does not match any expected shape')
        end
        options.e_gridvals=gpuArray(options.e_gridvals);
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
        options.pi_e=gather(options.pi_e); % Agent distribution iteration is performed on cpu
    elseif gridpiboth==3
        % For value fn, both e_gridvals_J and pi_e_J
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
        else
            error('options.e_grid size does not match any expected shape')
        end
        options.e_gridvals=gpuArray(options.e_gridvals);
        options.pi_e=gpuArray(options.pi_e);
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