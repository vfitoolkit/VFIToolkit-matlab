function options=SemiExogShockSetup_FHorz(n_d,N_j,d_grid,Parameters,options,Parallel)
% Convert semiz to age-dependent joint-grids and transtion matrix
% options will either be options or simoptions
% output: options.semiz_gridvals_J, options.pi_semiz_J


%% Check basic setup
if ~isfield(options,'n_semiz')
    return
end

if ~isfield(options,'SemiExoStateFn')
    error('When using options.n_semiz you must declare options.SemiExoStateFn (options refers to either options or simoptions)')
end
if ~isfield(options,'semiz_grid')
    error('When using options.n_semiz you must declare options.semiz_grid (options refers to either options or simoptions)')
elseif all(size(options.semiz_grid)==[1,sum(options.n_semiz)]) && sum(options.n_semiz)>1
    error('options.semiz_grid must be a column vector (you have a row vector) (options refers to either options or simoptions)')
end

if ~isfield(options,'l_dsemiz')
    options.l_dsemiz=1; % by default, only one decision variable influences the semi-exogenous state
end


%% Create semiz_gridvals_J
if ndims(options.semiz_grid)==2
    if all(size(options.semiz_grid)==[sum(options.n_semiz),1])
        semiz_gridvals_J=CreateGridvals(options.n_semiz,options.semiz_grid,1).*ones(1,1,N_j,'gpuArray');
    elseif all(size(options.semiz_grid)==[prod(options.n_semiz),length(options.n_semiz)])
        semiz_gridvals_J=options.semiz_grid.*ones(1,1,N_j,'gpuArray');
    end
else % Already age-dependent
    if all(size(options.semiz_grid)==[sum(options.n_semiz),N_j])
        semiz_gridvals_J=zeros(prod(options.n_semiz),length(options.n_semiz),N_j,'gpuArray');
        for jj=1:N_j
            semiz_gridvals_J(:,:,jj)=CreateGridvals(options.n_semiz,options.semiz_grid(:,jj),1);
        end
    elseif all(size(options.semiz_grid)==[prod(options.n_semiz),length(options.n_semiz),N_j])
        semiz_gridvals_J=options.semiz_grid;
    end
end

if isempty(n_d)
    % FnsToEvaluate don't need pi_semiz_J
    pi_semiz_J=[];
else
    %% Find decision variables that matter for semiz (can differ by setting)
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

    %% Create pi_semiz_J

    % Create the transition matrix in terms of (semiz,semizprime,dsemiz,j) for the semi-exogenous states for each age
    N_semiz=prod(options.n_semiz);
    l_semiz=length(options.n_semiz);
    temp=getAnonymousFnInputNames(options.SemiExoStateFn);
    if length(temp)>(l_semiz+l_semiz+l_dsemiz) % This is largely pointless, the SemiExoShockFn is always going to have some parameters
        SemiExoStateFnParamNames={temp{l_semiz+l_semiz+options.l_dsemiz+1:end}}; % the first inputs will always be (d,semizprime,semiz)
    else
        SemiExoStateFnParamNames={};
    end
    % Create pi_semiz_J
    pi_semiz_J=zeros(N_semiz,N_semiz,N_dsemiz,N_j,'gpuArray');
    for jj=1:N_j
        SemiExoStateFnParamValues=CreateVectorFromParams(Parameters,SemiExoStateFnParamNames,jj);
        pi_semiz_J(:,:,:,jj)=gpuArray(CreatePiSemiZ(n_dsemiz,options.n_semiz,dsemiz_grid,semiz_gridvals_J,options.SemiExoStateFn,SemiExoStateFnParamValues));
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

%% Clean up output
if Parallel==2 % gpu, for value fn and FnsToEvaluate
    options.semiz_gridvals_J=semiz_gridvals_J;
    options.pi_semiz_J=pi_semiz_J;
else % cpu, for agent dist
    options.semiz_gridvals_J=gather(semiz_gridvals_J);
    options.pi_semiz_J=gather(pi_semiz_J);
end











end