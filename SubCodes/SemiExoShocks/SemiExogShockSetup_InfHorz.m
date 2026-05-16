function options=SemiExogShockSetup_InfHorz(n_d,d_grid,Parameters,options,gridpiboth)
% Convert semiz to joint-grids and transtion matrix (infinite horizon -- no age dimension)
% options will either be vfoptions or simoptions
% output: options.semiz_gridvals, options.pi_semiz

% gridpiboth=3: sometimes (value fn iter) we want both grid and transition probabilities
% gridpiboth=2: sometimes (agent dist)    we want just transition probabilities
% gridpiboth=1: sometimes (FnsToEvaluate) we want just grid

%% Check basic setup
if ~isfield(options,'n_semiz')
    return
end
if ~isfield(options,'l_dsemiz')
    options.l_dsemiz=1; % by default, only one decision variable influences the semi-exogenous state
end


if gridpiboth==3 || gridpiboth==2
    if ~isfield(options,'SemiExoStateFn') && ~isfield(options,'pi_semiz')
        error('When using options.n_semiz you must declare options.SemiExoStateFn or options.pi_semiz (options refers to either vfoptions or simoptions)')
    end
end
if gridpiboth==3 || gridpiboth==1
    if ~isfield(options,'semiz_grid')
        error('When using options.n_semiz you must declare options.semiz_grid (options refers to either vfoptions or simoptions)')
    elseif size(options.semiz_grid,1)==1 && sum(options.n_semiz)>1
        error('options.semiz_grid must be a column vector (you have a row vector) (options refers to either vfoptions or simoptions)')
    end
end

%% Create semiz_gridvals (joint grid on semiz)
if gridpiboth==3 || gridpiboth==1 || isfield(options,'SemiExoStateFn')
    % Regardless of whether we output semiz_gridvals, we sometimes have to create it as it is needed for evaluating SemiExoStateFn
    if all(size(options.semiz_grid)==[sum(options.n_semiz),1])
        % need to convert to joint-grid
        semiz_gridvals=CreateGridvals(options.n_semiz,options.semiz_grid,1);
    elseif all(size(options.semiz_grid)==[prod(options.n_semiz),length(options.n_semiz)])
        % already joint-grid
        semiz_gridvals=options.semiz_grid;
    else
        error('options.semiz_grid size does not match any expected shape')
    end
    semiz_gridvals=gpuArray(semiz_gridvals); % Make sure it is on GPU
end


if gridpiboth==3 || gridpiboth==2
    if isempty(n_d)
        % FnsToEvaluate don't need pi_semiz
        pi_semiz=[];
    else
        %% Find decision variables that matter for semiz (can differ by setting)
        if ~isfield(options,'riskyasset')
            options.riskyasset=0;
        end

        if options.riskyasset==1
            n_dsemiz=n_d(sum(options.refine_d(1:3))+1:end);
            n_predsemiz=n_d(1:sum(options.refine_d(1:3)));
        else
            % Last decision variable(s) is the one influencing the semiz
            n_dsemiz=n_d(length(n_d)-options.l_dsemiz+1:end); % last options.numd_semiz decision variables
            n_predsemiz=n_d(1:length(n_d)-options.l_dsemiz);
        end
        l_dsemiz=length(n_dsemiz);
        N_dsemiz=prod(n_dsemiz);
        dsemiz_grid=d_grid(sum(n_predsemiz)+1:sum(n_d));

        %% Create pi_semiz

        % Create the transition matrix in terms of (semiz,semizprime,dsemiz) for the semi-exogenous states
        N_semiz=prod(options.n_semiz);
        l_semiz=length(options.n_semiz);
        if isfield(options,'SemiExoStateFn')
            temp=getAnonymousFnInputNames(options.SemiExoStateFn);
            if length(temp)>(l_semiz+l_semiz+l_dsemiz) % This is largely pointless, the SemiExoShockFn is always going to have some parameters
                SemiExoStateFnParamNames={temp{l_semiz+l_semiz+options.l_dsemiz+1:end}}; % the first inputs will always be (d,semizprime,semiz)
            else
                SemiExoStateFnParamNames={};
            end
            % Create pi_semiz
            SemiExoStateFnParamValues=CreateVectorFromParams(Parameters,SemiExoStateFnParamNames);
            pi_semiz=gpuArray(CreatePiSemiZ(n_dsemiz,options.n_semiz,dsemiz_grid,semiz_gridvals,options.SemiExoStateFn,SemiExoStateFnParamValues));
        else
            % User already inputted options.pi_semiz
            % So just check it is the right size
            if ndims(options.pi_semiz)==3
                if all(size(options.pi_semiz)==[N_semiz,N_semiz,N_dsemiz])
                    pi_semiz=gpuArray(options.pi_semiz);
                else
                    error('options.pi_semiz matrix is the wrong size (expected [N_semiz, N_semiz, N_dsemiz])')
                end
            else
                error('options.pi_semiz matrix has wrong number of dimensions (expected 3)')
            end
        end

        %% Check that pi_semiz has rows summing to one
        temp=abs(sum(pi_semiz,2)-1);
        if any(temp(:)>1e-14)
            warning('Using semi-exo shocks, your transition matrix has some rows that dont sum to one')
        end
        % Check that pi_semiz has no negative entries
        if min(pi_semiz(:))<0
            warning('Using semi-exo shocks, your transition matrix contains negative values')
        end

    end
end


%% Clean up output
if gridpiboth==3
    options.pi_semiz=pi_semiz;
    options.semiz_gridvals=semiz_gridvals;
elseif gridpiboth==2
    options.pi_semiz=gather(pi_semiz); % Agent distribution iteration is performed on cpu
elseif gridpiboth==1
    options.semiz_gridvals=semiz_gridvals;
end
% clean up options, so we don't accidently reuse these things
if isfield(options,'SemiExoStateFn')
    options=rmfield(options,'SemiExoStateFn');
end

end
