function [z_gridvals_J, pi_z_J, options]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,options,gridpiboth)
% Convert z and e to age-dependent joint-grids and transtion matrix
% options will either be vfoptions or simoptions
% output: z_gridvals_J, pi_z_J, options.e_gridvals_J, options.pi_e_J

% gridpiboth=3: sometimes (value fn iter) we want both grid and transition probabilities
% gridpiboth=2: sometimes (agent dist)    we want just transition probabilities
% gridpiboth=1: sometimes (FnsToEvaluate) we want just grid

% Accepted input shapes:
%   z_grid:
%     [sum(n_z), 1]                       stacked column grid for markov z (age-independent)
%     [prod(n_z), length(n_z)]            joint grid for markov z (age-independent)
%     [sum(n_z), N_j]                     stacked column grid for markov z, age-dependent (one column per age)
%     [prod(n_z), length(n_z), N_j]       joint grid for markov z, age-dependent (one slice per age)
%   pi_z:
%     [prod(n_z), prod(n_z)]              transition matrix for markov z (age-independent)
%     [prod(n_z), prod(n_z), N_j]         transition matrix for markov z, age-dependent (one slice per age)
%   options.e_grid:
%     [sum(n_e), 1]                       stacked column grid for iid e (age-independent)
%     [prod(n_e), length(n_e)]            joint grid for iid e (age-independent)
%     [sum(n_e), N_j]                     stacked column grid for iid e, age-dependent (one column per age)
%     [prod(n_e), length(n_e), N_j]       joint grid for iid e, age-dependent (one slice per age)
%   options.pi_e:
%     [prod(n_e), 1]                      iid distribution (age-independent)
%     [prod(n_e), N_j]                    iid distribution, age-dependent (one column per age)
%
% If options.ExogShockFn is supplied, it is called once per age j to produce
% a single age's [z_grid, pi_z] (using the age-independent shapes above); the
% raw z_grid / pi_z inputs are then ignored. Likewise options.EiidShockFn,
% if supplied, is called once per age j to produce [options.e_grid,
% options.pi_e]; the raw inputs are then ignored.
%
% Stacked column grid: each of the underlying univariate grids written one
% beneath the next in a single column of length sum(n_z). Compact, but the
% joint state space is only implicit. For example with two markov variables
% of sizes n_z=[3,2], the column contains the 3 values of z1 followed by the
% 2 values of z2, giving a 5x1 vector. In the age-dependent form, each age
% has its own such column, stacked side-by-side.
%
% Joint grid: every point in the product space listed explicitly, one per
% row, with each variable in its own column. The number of rows is
% prod(n_z) and the number of columns is length(n_z). Continuing the
% example, a joint grid is 6x2: each row pairs one z1 value with one z2
% value, covering all 6 combinations. In the age-dependent form, the same
% joint grid is given per age along the third dimension.
%
% Output shapes (function returns):
%   z_gridvals_J:
%     [prod(n_z), length(n_z), N_j]  age-dependent joint grid (always in joint form, regardless of input shape)
%     []                             if gridpiboth==2 (only pi_z_J requested) or prod(n_z)==0
%   pi_z_J:
%     [prod(n_z), prod(n_z), N_j]    age-dependent transition matrix (rows = from-state, cols = to-state)
%     []                             if gridpiboth==1 (only grid requested) or prod(n_z)==0
%   options.e_gridvals_J:
%     [prod(n_e), length(n_e), N_j]  age-dependent joint grid for iid e
%     []                             if gridpiboth==2 or no e variable
%   options.pi_e_J:
%     [prod(n_e), N_j]               age-dependent iid distribution (one column per age)
%     []                             if gridpiboth==1 or no e variable
%
% Age-independent inputs are broadcast across the N_j dimension; age-dependent
% inputs are passed through (or converted from stacked to joint form per age).

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
% Convert to z_gridvals_J (age-dependent joint grids) and corresponding
% pi_z_J (age-dependent transition matrix).
if prod(n_z)==0
    z_gridvals_J=[];
    pi_z_J=[];
else
    if gridpiboth==1 % for most FnsToEvaluate, we don't use pi_z
        pi_z_J=[];
        % Now just do z_gridvals_J
        z_gridvals_J=zeros(prod(n_z),length(n_z),N_j,'gpuArray');
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
                    z_gridvals_J(:,:,jj)=gpuArray(z_grid);
                end
            end
        elseif ndims(z_grid)==3 % already an age-dependent joint-grid
            if all(size(z_grid)==[prod(n_z),length(n_z),N_j])
                z_gridvals_J=z_grid;
            else
                error('z_grid is 3D but its size does not match [prod(n_z), length(n_z), N_j]')
            end
        elseif all(size(z_grid)==[sum(n_z),N_j]) % age-dependent grid
            for jj=1:N_j
                z_gridvals_J(:,:,jj)=CreateGridvals(n_z,z_grid(:,jj),1);
            end
        elseif all(size(z_grid)==[prod(n_z),length(n_z)]) % joint grid
            z_gridvals_J=z_grid.*ones(1,1,N_j,'gpuArray');
        elseif all(size(z_grid)==[sum(n_z),1]) % basic grid
            z_gridvals_J=CreateGridvals(n_z,z_grid,1).*ones(1,1,N_j,'gpuArray');
        else
            error('z_grid size does not match any expected shape')
        end
    elseif gridpiboth==2 % For agent dist, we don't use grid
        z_gridvals_J=[];
        % Now just do pi_z_J
        pi_z_J=zeros(prod(n_z),prod(n_z),N_j,'gpuArray');
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
        z_gridvals_J=zeros(prod(n_z),length(n_z),N_j,'gpuArray');
        pi_z_J=zeros(prod(n_z),prod(n_z),N_j,'gpuArray');
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
            else
                error('z_grid is 3D but its size does not match [prod(n_z), length(n_z), N_j]')
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
        else
            error('z_grid size does not match any expected shape')
        end
    end
end



%% If using e variable, do same for this
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
        options.e_gridvals_J=zeros(prod(options.n_e),length(options.n_e),N_j,'gpuArray');
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
                    options.e_gridvals_J(:,:,jj)=gpuArray(options.e_grid);
                end
            end
        elseif ndims(options.e_grid)==3 % already an age-dependent joint-grid
            if all(size(options.e_grid)==[prod(options.n_e),length(options.n_e),N_j])
                options.e_gridvals_J=options.e_grid;
            else
                error('options.e_grid is 3D but its size does not match [prod(n_e), length(n_e), N_j]')
            end
        elseif all(size(options.e_grid)==[sum(options.n_e),N_j]) % age-dependent stacked-grid
            for jj=1:N_j
                options.e_gridvals_J(:,:,jj)=CreateGridvals(options.n_e,options.e_grid(:,jj),1);
            end
        elseif all(size(options.e_grid)==[prod(options.n_e),length(options.n_e)]) % joint grid
            options.e_gridvals_J=options.e_grid.*ones(1,1,N_j,'gpuArray');
        elseif all(size(options.e_grid)==[sum(options.n_e),1]) % basic grid
            options.e_gridvals_J=CreateGridvals(options.n_e,options.e_grid,1).*ones(1,1,N_j,'gpuArray');
        else
            error('options.e_grid size does not match any expected shape')
        end
    elseif gridpiboth==2 % For agent dist, we don't use grid
        options.e_gridvals_J=[];
        % Now just do pi_e_J
        options.pi_e_J=zeros(prod(options.n_e),N_j,'gpuArray');
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
        options.e_gridvals_J=zeros(prod(options.n_e),length(options.n_e),N_j,'gpuArray');
        options.pi_e_J=zeros(prod(options.n_e),N_j,'gpuArray');
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
                    options.e_gridvals_J(:,:,jj)=gpuArray(options.e_grid);
                end
            end
        elseif ndims(options.e_grid)==3 % already an age-dependent joint-grid
            if all(size(options.e_grid)==[prod(options.n_e),length(options.n_e),N_j])
                options.e_gridvals_J=options.e_grid;
            else
                error('options.e_grid is 3D but its size does not match [prod(n_e), length(n_e), N_j]')
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
        else
            error('options.e_grid size does not match any expected shape')
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