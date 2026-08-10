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
%     [prod(n_z), prod(n_z), N_j]         transition matrix for markov z, age-dependent (slice j = transition from period j to j+1; the final slice is dropped internally unless options.V_Jplus1 is being used)
%     [prod(n_z), prod(n_z), N_j-1]       transition matrix for markov z, age-dependent with the (never-read) final-period slice omitted (not valid together with options.V_Jplus1)
%   options.e_grid:
%     [sum(n_e), 1]                       stacked column grid for iid e (age-independent)
%     [prod(n_e), length(n_e)]            joint grid for iid e (age-independent)
%     [sum(n_e), N_j]                     stacked column grid for iid e, age-dependent (one column per age)
%     [prod(n_e), length(n_e), N_j]       joint grid for iid e, age-dependent (one slice per age)
%   options.pi_e:
%     [prod(n_e), 1]                      iid distribution (age-independent)
%     [prod(n_e), N_j]                    iid distribution, age-dependent (column j = distribution of the e realized in period j; not valid together with options.V_Jplus1)
%     [prod(n_e), N_j+1]                  iid distribution, age-dependent including the V_Jplus1 period (final column is dropped internally unless options.V_Jplus1 is being used)
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
%     [prod(n_z), prod(n_z), N_j-1]  age-dependent transition matrix; slice jj is the transition from period jj to jj+1 (rows = from-state on the period-jj grid, cols = to-state on the period-jj+1 grid). There is no final-period slice as nothing can ever be read from it.
%     [prod(n_z), prod(n_z), N_j]    if options.V_Jplus1 is being used (the final slice is the transition from period N_j into the V_Jplus1 period)
%     []                             if gridpiboth==1 (only grid requested) or prod(n_z)==0
%   options.e_gridvals_J:
%     [prod(n_e), length(n_e), N_j]  age-dependent joint grid for iid e
%     []                             if gridpiboth==2 or no e variable
%   options.pi_e_J:
%     [prod(n_e), N_j]               age-dependent iid distribution; column jj is the distribution of the e realized in period jj (column 1 is unused by convention: the period-1 e comes from jequaloneDist)
%     [prod(n_e), N_j+1]             if options.V_Jplus1 is being used (the final column is the distribution of e in the V_Jplus1 period)
%     []                             if gridpiboth==1 or no e variable
%
% Age-independent inputs are broadcast across the N_j dimension; age-dependent
% inputs are passed through (or converted from stacked to joint form per age).
%
% Timing conventions are documented in docs/ExogenousShocks.md (section 'Timing').

%% Check basic setup
if isempty(n_z)
    error('If you have no z (exogenous markov) variables, set n_z=0 (not n_z=[])')
end

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

% pi_z_J gets N_jpiz slices and pi_e_J gets N_jpie columns. Nothing can ever be read from a
% final-period transition slice, nor from an e-distribution column for a period beyond the
% model, so these are not stored -- except when options.V_Jplus1 is being used, in which case
% the transition from period N_j into the V_Jplus1 period, and the distribution of e in the
% V_Jplus1 period, are both needed.
if isfield(options,'V_Jplus1')
    N_jpiz=N_j;
    N_jpie=N_j+1;
else
    N_jpiz=N_j-1;
    N_jpie=N_j;
end
if isfield(options,'V_Jplus1') && isfield(options,'EiidShockFn')
    error('Cannot combine vfoptions.V_Jplus1 with vfoptions.EiidShockFn: the distribution of e in the V_Jplus1 period is needed, but EiidShockFn cannot be evaluated for period N_j+1. Give vfoptions.pi_e as [N_e,1] or [N_e,N_j+1] instead.')
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
                error('z_grid is 3D but its size does not match [prod(n_z),length(n_z),N_j]; got [%s]',num2str(size(z_grid)))
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
            error('z_grid is not the correct shape. Expected one of: [sum(n_z),1] (stacked vector), [prod(n_z),length(n_z)] (joint grid), [sum(n_z),N_j] (age-dependent stacked vector), or [prod(n_z),length(n_z),N_j] (age-dependent joint grid). Got [%s]',num2str(size(z_grid)))
        end
    elseif gridpiboth==2 % For agent dist, we don't use grid
        z_gridvals_J=[];
        if isfield(options,'ExogShockFn')
            pi_z_J=zeros(prod(n_z),prod(n_z),N_jpiz,'gpuArray');
            for jj=1:N_jpiz
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, options.ExogShockFnParamNames,jj);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [~,pi_z]=options.ExogShockFn(ExogShockFnParamsCell{:});
                pi_z_J(:,:,jj)=gpuArray(pi_z);
            end
        else
            if ~isequal(size(pi_z),[prod(n_z),prod(n_z)]) && ~isequal(size(pi_z),[prod(n_z),prod(n_z),N_j]) && ~isequal(size(pi_z),[prod(n_z),prod(n_z),N_j-1])
                error('pi_z is the wrong shape: expected [N_z,N_z], [N_z,N_z,N_j] or [N_z,N_z,N_j-1] (where N_z=prod(n_z)), got [%s]',num2str(size(pi_z)))
            end
            if ndims(pi_z)==2 % age-independent: broadcast
                pi_z_J=pi_z.*ones(1,1,N_jpiz,'gpuArray');
            elseif size(pi_z,3)>=N_jpiz
                pi_z_J=gpuArray(pi_z(:,:,1:N_jpiz)); % if a (never-read) final-period slice was given, it is dropped here
            else % N_j-1 slices given, but N_jpiz=N_j because options.V_Jplus1 is being used
                error('When using vfoptions.V_Jplus1 you must give pi_z with N_j slices (the final slice is the transition from period N_j into the V_Jplus1 period)')
            end
        end
        pi_z_J=gather(pi_z_J); % Agent distribution iteration is performed on cpu
    elseif gridpiboth==3
        % For value fn, both z_gridvals_J and pi_z_J
        z_gridvals_J=zeros(prod(n_z),length(n_z),N_j,'gpuArray');
        if isfield(options,'ExogShockFn')
            pi_z_J=zeros(prod(n_z),prod(n_z),N_jpiz,'gpuArray');
            for jj=1:N_j
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, options.ExogShockFnParamNames,jj);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [z_grid,pi_z]=options.ExogShockFn(ExogShockFnParamsCell{:});
                if jj<=N_jpiz
                    pi_z_J(:,:,jj)=gpuArray(pi_z);
                end
                if all(size(z_grid)==[sum(n_z),1])
                    z_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(n_z,z_grid,1));
                else % already joint-grid
                    z_gridvals_J(:,:,jj)=gpuArray(z_grid);
                end
            end
        else
            if ~isequal(size(pi_z),[prod(n_z),prod(n_z)]) && ~isequal(size(pi_z),[prod(n_z),prod(n_z),N_j]) && ~isequal(size(pi_z),[prod(n_z),prod(n_z),N_j-1])
                error('pi_z is the wrong shape: expected [N_z,N_z], [N_z,N_z,N_j] or [N_z,N_z,N_j-1] (where N_z=prod(n_z)), got [%s]',num2str(size(pi_z)))
            end
            if ndims(pi_z)==2 % age-independent: broadcast
                pi_z_J=pi_z.*ones(1,1,N_jpiz,'gpuArray');
            elseif size(pi_z,3)>=N_jpiz
                pi_z_J=gpuArray(pi_z(:,:,1:N_jpiz)); % if a (never-read) final-period slice was given, it is dropped here
            else % N_j-1 slices given, but N_jpiz=N_j because options.V_Jplus1 is being used
                error('When using vfoptions.V_Jplus1 you must give pi_z with N_j slices (the final slice is the transition from period N_j into the V_Jplus1 period)')
            end
            if ndims(z_grid)==3 % already an age-dependent joint-grid
                if all(size(z_grid)==[prod(n_z),length(n_z),N_j])
                    z_gridvals_J=z_grid;
                else
                    error('z_grid is 3D but its size does not match [prod(n_z),length(n_z),N_j]; got [%s]',num2str(size(z_grid)))
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
                error('z_grid is not the correct shape. Expected one of: [sum(n_z),1] (stacked vector), [prod(n_z),length(n_z)] (joint grid), [sum(n_z),N_j] (age-dependent stacked vector), or [prod(n_z),length(n_z),N_j] (age-dependent joint grid). Got [%s]',num2str(size(z_grid)))
            end
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
                error('options.e_grid is 3D but its size does not match [prod(n_e),length(n_e),N_j]; got [%s]',num2str(size(options.e_grid)))
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
            error('options.e_grid is not the correct shape. Expected one of: [sum(n_e),1] (stacked vector), [prod(n_e),length(n_e)] (joint grid), [sum(n_e),N_j] (age-dependent stacked vector), or [prod(n_e),length(n_e),N_j] (age-dependent joint grid). Got [%s]',num2str(size(options.e_grid)))
        end
    elseif gridpiboth==2 % For agent dist, we don't use grid
        options.e_gridvals_J=[];
        if isfield(options,'EiidShockFn')
            options.pi_e_J=zeros(prod(options.n_e),N_jpie,'gpuArray');
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
            if ~isequal(size(options.pi_e),[prod(options.n_e),1]) && ~isequal(size(options.pi_e),[prod(options.n_e),N_j]) && ~isequal(size(options.pi_e),[prod(options.n_e),N_j+1])
                error('options.pi_e is the wrong shape: expected [N_e,1], [N_e,N_j] or [N_e,N_j+1] (where N_e=prod(n_e)), got [%s]',num2str(size(options.pi_e)))
            end
            if size(options.pi_e,2)==1 % age-independent: broadcast
                options.pi_e_J=options.pi_e.*ones(1,N_jpie,'gpuArray');
            elseif size(options.pi_e,2)>=N_jpie
                options.pi_e_J=gpuArray(options.pi_e(:,1:N_jpie)); % if a period-N_j+1 column was given without options.V_Jplus1, it is dropped here
            else % N_j columns given, but N_jpie=N_j+1 because options.V_Jplus1 is being used
                error('When using vfoptions.V_Jplus1, an age-dependent vfoptions.pi_e must be [N_e,N_j+1] (the final column is the distribution of e in the V_Jplus1 period)')
            end
        end
        options.pi_e_J=gather(options.pi_e_J); % Agent distribution iteration is performed on cpu
    elseif gridpiboth==3
        % For value fn, both e_gridvals_J and pi_e_J
        options.e_gridvals_J=zeros(prod(options.n_e),length(options.n_e),N_j,'gpuArray');
        if isfield(options,'EiidShockFn')
            options.pi_e_J=zeros(prod(options.n_e),N_jpie,'gpuArray');
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
        else
            if ~isequal(size(options.pi_e),[prod(options.n_e),1]) && ~isequal(size(options.pi_e),[prod(options.n_e),N_j]) && ~isequal(size(options.pi_e),[prod(options.n_e),N_j+1])
                error('options.pi_e is the wrong shape: expected [N_e,1], [N_e,N_j] or [N_e,N_j+1] (where N_e=prod(n_e)), got [%s]',num2str(size(options.pi_e)))
            end
            if size(options.pi_e,2)==1 % age-independent: broadcast
                options.pi_e_J=options.pi_e.*ones(1,N_jpie,'gpuArray');
            elseif size(options.pi_e,2)>=N_jpie
                options.pi_e_J=gpuArray(options.pi_e(:,1:N_jpie)); % if a period-N_j+1 column was given without options.V_Jplus1, it is dropped here
            else % N_j columns given, but N_jpie=N_j+1 because options.V_Jplus1 is being used
                error('When using vfoptions.V_Jplus1, an age-dependent vfoptions.pi_e must be [N_e,N_j+1] (the final column is the distribution of e in the V_Jplus1 period)')
            end
            if ndims(options.e_grid)==3 % already an age-dependent joint-grid
                if all(size(options.e_grid)==[prod(options.n_e),length(options.n_e),N_j])
                    options.e_gridvals_J=options.e_grid;
                else
                    error('options.e_grid is 3D but its size does not match [prod(n_e),length(n_e),N_j]; got [%s]',num2str(size(options.e_grid)))
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
                error('options.e_grid is not the correct shape. Expected one of: [sum(n_e),1] (stacked vector), [prod(n_e),length(n_e)] (joint grid), [sum(n_e),N_j] (age-dependent stacked vector), or [prod(n_e),length(n_e),N_j] (age-dependent joint grid). Got [%s]',num2str(size(options.e_grid)))
            end
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