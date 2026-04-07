function options=SemiExogShockSetup_FHorz_PType(n_d,N_j,Names_i,d_grid,Parameters,options,Parallel,gridpiboth)
% Convert semiz to age-dependent joint-grids and transtion matrix
% options will either be options or simoptions
% output: options.semiz_gridvals_J, options.pi_semiz_J

% gridpiboth=3: sometimes (value fn iter) we want both grid and transition probabilties
% gridpiboth=2: sometimes (agent dist)    we want just transition probabilties
% gridpiboth=1: sometimes (FnsToEvaluate) we want just grid

%% Check basic setup
if ~isfield(options,'n_semiz')
    return
end
if ~isfield(options,'l_dsemiz')
    options.l_dsemiz=1; % by default, only one decision variable influences the semi-exogenous state
end

N_i=length(Names_i);

for ii=1:N_i

    n_semiz_ii=0;
    if isstruct(options.n_semiz)
        if isfield(options.n_semiz,Names_i{ii})
            n_semiz_ii=options.n_semiz.(Names_i{ii});
        end
    else
        n_semiz_ii=options.n_semiz;
    end

    l_dsemiz_ii=0;
    if isstruct(options.l_dsemiz)
        if isfield(options.l_dsemiz,Names_i{ii})
            l_dsemiz_ii=options.l_dsemiz.(Names_i{ii});
        end
    else
        l_dsemiz_ii=options.l_dsemiz;
    end

    semiz_grid_ii=[];
    if isstruct(options.semiz_grid)
        if isfield(options.semiz_grid,Names_i{ii})
            semiz_grid_ii=options.semiz_grid.(Names_i{ii});
        end
    else
        if size(options.semiz_grid,ndims(options.semiz_grid))==N_i
            otherdims = repmat({':'},1,ndims(options.semiz_grid)-1);
            semiz_grid_ii=options.semiz_grid(otherdims{:},ii);
        else
            semiz_grid_ii=options.semiz_grid;
        end
    end

    if isfield(options,'SemiExoStateFn')
        if isstruct(options.SemiExoStateFn)
            if isfield(options.SemiExoStateFn,Names_i{ii})
                SemiExoStateFn_ii=options.SemiExoStateFn.(Names_i{ii});
            else
                SemiExoStateFn_ii=[];
            end
        else
            SemiExoStateFn_ii=options.SemiExoStateFn;
        end
    else
        SemiExoStateFn_ii=[];
    end

    if isfield(options,'pi_semiz_ii')
        if isstruct(options.pi_semiz)
            if isfield(options.pi_semiz,Names_i{ii})
                pi_semiz_ii=options.pi_semiz.(Names_i{ii});
            end
        else
            if size(options.pi_semiz,ndims(options.pi_semiz))==N_i
                otherdims = repmat({':'},1,ndims(options.pi_semiz)-1);
                pi_semiz_ii=options.pi_semiz(otherdims{:},ii);
            else
                pi_semiz_ii=options.pi_semiz;
            end
        end
    else
        pi_semiz_ii=[];
    end

    if gridpiboth==3 || gridpiboth==2
        if isempty(SemiExoStateFn_ii) && isempty(pi_semiz_ii)
            error('When using options.n_semiz you must declare options.SemiExoStateFn or options.pi_semiz (options refers to either options or simoptions)')
        end
    end
    if gridpiboth==3 || gridpiboth==1
        if isempty(semiz_grid_ii)
            error('When using options.n_semiz you must declare options.semiz_grid (options refers to either options or simoptions)')
        elseif size(semiz_grid_ii,1)==1 && sum(n_semiz_ii)>1
            error('options.semiz_grid must be a column vector (you have a row vector) (options refers to either options or simoptions)')
        end
    end

    
    if n_semiz_ii>0

        %% Create semiz_gridvals_J (joint grid on semiz)
        if gridpiboth==3 || gridpiboth==1 || ~isempty(SemiExoStateFn_ii)
            % Regardless of whether we output semiz_gridvals_J, we sometimes have to create it as it is needed for evaluting SemiExogShockFn
            if ndims(semiz_grid_ii)==3
                if all(size(semiz_grid_ii)==[prod(n_semiz_ii),length(n_semiz_ii),N_j])
                    % already age-dependent joint-grid
                    semiz_gridvals_J=semiz_grid_ii;
                end
            elseif ndims(semiz_grid_ii)==2
                if all(size(semiz_grid_ii)==[sum(n_semiz_ii),1])
                    % need to convert to joint-grid, and make age-dependent
                    semiz_gridvals_J=CreateGridvals(n_semiz_ii,semiz_grid_ii,1).*ones(1,1,N_j,'gpuArray');
                elseif all(size(semiz_grid_ii)==[prod(n_semiz_ii),length(n_semiz_ii)]) % joint grid
                    % already joint-grid, need to make age-dependent
                    semiz_gridvals_J=semiz_grid_ii.*ones(1,1,N_j,'gpuArray');
                elseif all(size(semiz_grid_ii)==[sum(n_semiz_ii),N_j])
                    % already age-dependent, but need to convert to joint-grid
                    semiz_gridvals_J=zeros(prod(n_semiz_ii),length(n_semiz_ii),N_j,'gpuArray');
                    for jj=1:N_j
                        semiz_gridvals_J(:,:,jj)=CreateGridvals(n_semiz_ii,semiz_grid_ii(:,jj),1);
                    end
                end
            end
            semiz_gridvals_J=gpuArray(semiz_gridvals_J); % Make sure it is on GPU
        end


        if gridpiboth==3 || gridpiboth==2
            if isempty(n_d)
                % FnsToEvaluate don't need pi_semiz_J
                pi_semiz_J=[];
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
                    n_dsemiz=n_d(length(n_d)-l_dsemiz_ii+1:end); % last options.numd_semiz decision variables
                    n_predsemiz=n_d(1:length(n_d)-l_dsemiz_ii);
                end
                l_dsemiz=length(n_dsemiz);
                N_dsemiz=prod(n_dsemiz);
                dsemiz_grid=d_grid(sum(n_predsemiz)+1:sum(n_d));

                %% Create pi_semiz_J

                % Create the transition matrix in terms of (semiz,semizprime,dsemiz,j) for the semi-exogenous states for each age
                N_semiz=prod(n_semiz_ii);
                l_semiz=length(n_semiz_ii);
                if isfield(options,'SemiExoStateFn')
                    temp=getAnonymousFnInputNames(SemiExoStateFn_ii);
                    if length(temp)>(l_semiz+l_semiz+l_dsemiz) % This is largely pointless, the SemiExoShockFn is always going to have some parameters
                        SemiExoStateFnParamNames={temp{l_semiz+l_semiz+l_dsemiz_ii+1:end}}; % the first inputs will always be (d,semizprime,semiz)
                    else
                        SemiExoStateFnParamNames={};
                    end
                    % Create pi_semiz_J
                    pi_semiz_J=zeros(N_semiz,N_semiz,N_dsemiz,N_j,'gpuArray');
                    for jj=1:N_j
                        SemiExoStateFnParamValues=CreateVectorFromParams(Parameters,SemiExoStateFnParamNames,jj);
                        pi_semiz_J(:,:,:,jj)=gpuArray(CreatePiSemiZ(n_dsemiz,n_semiz_ii,dsemiz_grid,semiz_gridvals_J(:,:,jj),SemiExoStateFn_ii,SemiExoStateFnParamValues));
                    end
                else
                    % User already inputted options.pi_semiz
                    % So just check it is the right size
                    if ndims(pi_semiz_ii)==4
                        if all(size(pi_semiz_ii)==[N_semiz,N_semiz,N_dsemiz,N_j])
                            pi_semiz_J=gpuArray(pi_semiz_ii);
                        else
                            error('options.pi_semiz matrix is the wrong size')
                        end
                    elseif ndims(pi_semiz_ii)==3
                        if all(size(pi_semiz_ii)==[N_semiz,N_semiz,N_dsemiz])
                            pi_semiz_J=repelem(gpuArray(pi_semiz_ii),1,1,1,N_j);
                        else
                            error('options.pi_semiz matrix is the wrong size')
                        end
                    else
                        error('options.pi_semiz matrix is the wrong size')
                    end
                end

                %% Check that pi_semiz_J has rows summing to one
                % Check that pi_semiz_J has rows summing to one, if not, print a warning
                for jj=1:N_j
                    temp=abs(sum(pi_semiz_J(:,:,:,jj),2)-1);
                    if any(temp(:)>1e-14)
                        warning('Using semi-exo shocks, your transition matrix has some rows that dont sum to one for age %i',jj)
                    end
                end
                % Check that pi_semiz_J has no negative entries, if not, print a warning
                if min(pi_semiz_J(:))<0
                    warning('Using semi-exo shocks, your transition matrix contains negative values')
                end

            end
        end


        %% Clean up output
        if Parallel==2 % gpu, for value fn and FnsToEvaluate
            if gridpiboth==3
                options.pi_semiz_J.(Names_i{ii})=pi_semiz_J;
                options.semiz_gridvals_J.(Names_i{ii})=semiz_gridvals_J;
            elseif gridpiboth==2
                options.pi_semiz_J.(Names_i{ii})=pi_semiz_J;
            elseif gridpiboth==1
                options.semiz_gridvals_J.(Names_i{ii})=semiz_gridvals_J;
            end
        else % cpu, for agent dist
            if gridpiboth==3
                options.semiz_gridvals_J.(Names_i{ii})=gather(semiz_gridvals_J);
                options.pi_semiz_J.(Names_i{ii})=gather(pi_semiz_J);
            elseif gridpiboth==2
                options.pi_semiz_J.(Names_i{ii})=gather(pi_semiz_J);
            elseif gridpiboth==1
                options.semiz_gridvals_J.(Names_i{ii})=gather(semiz_gridvals_J);
            end
        end
    end
end


% clean up options, so we don't accidently reuse these things
if isfield(options,'SemiExoStateFn')
    options=rmfield(options,'SemiExoStateFn');
end
if isfield(options,'pi_semiz')
    options=rmfield(options,'pi_semiz');
end










end