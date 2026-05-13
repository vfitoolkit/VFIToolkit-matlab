function varargout=ValueFnIter_InfHorz_CPU(n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Solves infinite-horizon value function problems on the CPU
% Because this is the CPU code, you just get (d,a,z) and nothing that needs vfoptions to be set.
% varargoutput={V,Policy};

V=nan; % Matlab was complaining that V was not assigned

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

%% Check the sizes of some of the inputs
if strcmp(vfoptions.solnmethod,'purediscretization') || strcmp(vfoptions.solnmethod,'purediscretization_refinement') || strcmp(vfoptions.solnmethod,'localpolicysearch')
    if N_d>0 && ~all(size(d_grid)==[sum(n_d), 1])
        error('d_grid is not the correct shape (should be of size sum(n_d)-by-1)')
    elseif ~all(size(a_grid)==[sum(n_a), 1])
        error('a_grid is not the correct shape (should be of size sum(n_a)-by-1)')

        % Check z_grid inputs
    elseif isfield(vfoptions,'ExogShockFn')
            % okay
    elseif N_z>0
        if ~all(size(z_grid)==[sum(n_z), 1])
            if all(size(z_grid)==[prod(n_z),length(n_z)])
                % Using joint grids
            else
                error('z_grid is not the correct shape (should be of size sum(n_z)-by-1)')
            end
        elseif ~all(size(pi_z)==[N_z, N_z])
            error('pi is not of size N_z-by-N_z')
        end
    elseif n_z(end)>1 % Ignores this final check if last dimension of n_z is singleton as will cause an error
        if ndims(V0)>2
            if ~all(size(V0)==[n_a,n_z]) % Allow for input to be already transformed into Kronecker form
                error('Starting choice for ValueFn is not of size [n_a,n_z]')
            end
        elseif ~all(size(V0)==[N_a,N_z]) % Allows for possiblity that V0 is already in kronecker form
            error('Starting choice for ValueFn is not of size [n_a,n_z]')
        end
    end
end

if N_z>0
    if min(min(pi_z))<0
        error('Problem with pi_z in ValueFnIter_Case1: min(min(pi_z))<0 \n')
    elseif vfoptions.piz_strictonrowsaddingtoone==1
        if max(sum(pi_z,2))~=1 || min(sum(pi_z,2))~=1
            error('Problem with pi_z in ValueFnIter_Case1: rows do not sum to one \n')
        end
    elseif vfoptions.piz_strictonrowsaddingtoone==0
        if max(abs((sum(pi_z,2))-1)) > 10^(-13)
            error('Problem with pi_z in ValueFnIter_Case1: rows do not sum to one \n')
        end
    end
end


%% Anything but the basics is only for GPU
if vfoptions.experienceasset==1
    error('Cannot use vfoptions.experienceasset=1 without a GPU')
end


%% V0 (initial guess)
if isfield(vfoptions,'V0')
    V0=reshape(vfoptions.V0,[N_a,N_z]);
    vfoptions.actualV0=1;
else
    V0=zeros([N_a,N_z]);
    vfoptions.actualV0=0; % DC2 has different way of creating inital guess so this will be ignored
end


%% Switch to z_gridvals
if vfoptions.alreadygridvals==0
    % only basics allowed with cpu (can have two z variables, but that is as complex as it gets)
    z_gridvals=CreateGridvals(n_z,z_grid,1);
elseif vfoptions.alreadygridvals==1
    z_gridvals=z_grid;
end

%% Implement new way of handling ReturnFn inputs
if isempty(ReturnFnParamNames)
    ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,0,vfoptions,Parameters);
end
% Basic setup: the first inputs of ReturnFn will be (d,aprime,a,z,..) and everything after this is a parameter, so we get the names of all these parameters.
% But this changes if you have e, semiz, or just multiple d, and if you use riskyasset, expasset, etc.
% So figure out which setup we have, and get the relevant ReturnFnParamNames


%% Entry and Exit
if vfoptions.endogenousexit==1
    % ExitPolicy is binary decision to exit (1 is exit, 0 is 'not exit').
    [V, Policy,ExitPolicy]=ValueFnIter_InfHorz_EndogExit(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    varargout={V,Policy,ExitPolicy};
    return
elseif vfoptions.endogenousexit==2 % Mixture of endogenous and exogenous exit.
    % ExitPolicy is binary decision to exit (1 is exit, 0 is 'not exit').
    % Policy is for those who remain.
    % PolicyWhenExit is current period decisions of those who will exit at end of period.
    [V, Policy, PolicyWhenExit, ExitPolicy]=ValueFnIter_InfHorz_EndogExit2(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    varargout={V,Policy, PolicyWhenExit,ExitPolicy};
    return
end

%% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec); % Infinite horizon, so just do this once.

%%
if length(n_d)>1
    d_gridvals=CreateGridvals(n_d,d_grid,1);
else
    d_gridvals=d_grid;
end

%% On CPU, pure-discretization is the only solution algorithm
if strcmp(vfoptions.solnmethod,'purediscretization')

    N_d=prod(n_d);

    if vfoptions.lowmemory==0

        %% CreateReturnFnMatrix_Disc_CPU creates a matrix of dimension (d and aprime)-by-a-by-z.
        % Since the return function is independent of time creating it once and
        % then using it every iteration is good for speed, but it does use a lot of memory.

        if vfoptions.verbose==1
            disp('Creating return fn matrix')
        end

        ReturnMatrix=CreateReturnFnMatrix_Disc_CPU(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, ReturnFnParamsVec);

        if vfoptions.verbose==1
            fprintf('Starting Value Function \n')
        end

        if N_d==0
            if vfoptions.parallel==0 % On CPU
                [VKron,Policy]=ValueFnIter_InfHorz_nod_Par0_raw(V0, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            elseif vfoptions.parallel==1 % On Parallel CPU
                [VKron,Policy]=ValueFnIter_InfHorz_nod_Par1_raw(V0, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            end
        else
            if vfoptions.parallel==0  % On CPU
                [VKron, Policy]=ValueFnIter_InfHorz_Par0_raw(V0, N_d,N_a,N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, vfoptions.maxiter);
            elseif vfoptions.parallel==1 % On Parallel CPU
                [VKron, Policy]=ValueFnIter_InfHorz_Par1_raw(V0, N_d,N_a,N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, vfoptions.maxiter);
            end
        end

    elseif vfoptions.lowmemory==1
        error('Can only use lowmemory on GPU')
    end
end

%%
if vfoptions.verbose==1
    disp('Transforming Value Fn and Optimal Policy matrices back out of Kronecker Form')
    tic;
end


%% Cleaning up the output
if vfoptions.outputkron==0
    V=reshape(VKron,[n_a,n_z]);
    Policy=UnKronPolicyIndexes_InfHorz_CPU(Policy, n_d, n_a, n_z);
else
    Policy=reshape(Policy,[1,N_a,N_z]);
    varargout={VKron,Policy};
    return
end

varargout={V,Policy};

end
