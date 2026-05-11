function varargout=ValueFnIter_InfHorz(n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Solves infinite-horizon value function problems.
% Typically, varargoutput={V,Policy};

V=nan; % Matlab was complaining that V was not assigned

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

%% Check which vfoptions have been used, set all others to defaults 
if ~exist('vfoptions','var')
    disp('No vfoptions given, using defaults')
    % If vfoptions is not given, just use all the defaults
    vfoptions.verbose=0;
    vfoptions.tolerance=10^(-9); % Convergence tolerance (for ||V_n - V_{n-1}|| )
    vfoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    vfoptions.maxiter=10^4; % Can be used to stop the VFI after a finite number of iterations
    % Options relating to how the model is solved (which algorithm)
    vfoptions.solnmethod='purediscretization_refinement'; % if no d variable, will be set to 'purediscretization' below
    vfoptions.divideandconquer=0;
    vfoptions.gridinterplayer=0; % grid interpolation layer
    vfoptions.lowmemory=0;
    % Howards improvement
    vfoptions.howardsgreedy=0; % =0 iterated (aka modified-Policy Fn Iteration) or =1 greedy (aka Policy Fn Iteration)
        % Note: for small models, Howards greedy is faster, but cannot handle that V takes -Inf.
    % When doing Howards iterations, the following are some suboptions
    vfoptions.howards=150; % based on some tests, 80 to 150 was fastest, but 150 was best on average
    vfoptions.maxhowards=500; % Turn howards off after this many times (just so it cannot cause convergence to fail if thing are going wrong)
    if N_a>1200 && N_z>100
        vfoptions.howardssparse=1; % Do Howards iteration using a sparse matrix (rather than indexing). Sparse is only faster for bigger models.
    else
        vfoptions.howardssparse=0;
    end
    % Different asset types
    vfoptions.endogenousexit=0;
    vfoptions.incrementaltype=0; % (vector indicating endogenous state is an incremental endogenous state variable)
    vfoptions.experienceasset=0;
    vfoptions.inheritanceasset=0;
%     vfoptions.exoticpreferences % default is not to declare it
%     vfoptions.SemiEndogShockFn % default is not to declare it
    % Exogenous shocks
    vfoptions.n_e=0;
    vfoptions.n_semiz=0;
    % Other options
    vfoptions.piz_strictonrowsaddingtoone=0;
    vfoptions.separableReturnFn=0; % advanced option to split ReturnFn into two parts (ReturnFn.R1 and ReturnFn.R2)
    vfoptions.outputkron=0;
    % When calling as a subcommand, the following is used internally
    vfoptions.alreadygridvals=0;
else
    % Check vfoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(vfoptions,'verbose')
        vfoptions.verbose=0;
    end
    if ~isfield(vfoptions,'tolerance')
        vfoptions.tolerance=10^(-9);
    end
    if ~isfield(vfoptions,'parallel')
        vfoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    end
    if ~isfield(vfoptions,'maxiter')
        vfoptions.maxiter=10^4;
    end
    if ~isfield(vfoptions,'solnmethod')
        vfoptions.solnmethod='purediscretization_refinement'; % if no d variable, will be set to 'purediscretization' below
    end
    if ~isfield(vfoptions,'divideandconquer')
        vfoptions.divideandconquer=0;
    end
    if ~isfield(vfoptions,'gridinterplayer')
        vfoptions.gridinterplayer=0; % grid interpolation layer
    elseif vfoptions.gridinterplayer==1
        if ~isfield(vfoptions,'preGI')
            vfoptions.preGI=0; % post GI is faster
        end
        if ~isfield(vfoptions,'ngridinterp')
            error('When using vfoptions.gridinterplayer=1 you must set vfoptions.ngridinterp')
        end
    end
    if ~isfield(vfoptions,'lowmemory')
        vfoptions.lowmemory=0;
    end
    % Howards improvement
    if ~isfield(vfoptions,'howardsgreedy')
        vfoptions.howardsgreedy=0; % =0 iterated (aka modified-Policy Fn Iteration) or =1 greedy (aka Policy Fn Iteration)
        % Note: for small models, Howards greedy is faster, but cannot handle that V takes -Inf.
        % Note: Howards greedy =1 can only be used without grid interpolation layer (gridinterplayer=0)
    end
    % When doing Howards iterations, the following are some suboptions
    if ~isfield(vfoptions,'howards')
        vfoptions.howards=150; % based on some tests, 80 to 150 was fastest, but 150 was best on average
    end  
    if ~isfield(vfoptions,'maxhowards')
        vfoptions.maxhowards=500; % Turn howards off after this many times (just so it cannot cause convergence to fail if thing are going wrong)
    end
    if ~isfield(vfoptions,'howardssparse')
        if N_a>1200 && N_z>100
            vfoptions.howardssparse=1; % Do Howards iteration using a sparse matrix (rather than indexing). Sparse is only faster for bigger models.
        else
            vfoptions.howardssparse=0;
        end
    end
    % Different asset types
    if ~isfield(vfoptions,'endogenousexit')
        vfoptions.endogenousexit=0;
    end
    if ~isfield(vfoptions,'incrementaltype')
        vfoptions.incrementaltype=0; % (vector indicating endogenous state is an incremental endogenous state variable)
    end
    if ~isfield(vfoptions,'experienceasset')
        vfoptions.experienceasset=0;
    end
    if ~isfield(vfoptions,'inheritanceasset')
        vfoptions.inheritanceasset=0;
    end
%     vfoptions.exoticpreferences % default is not to declare it
%     vfoptions.SemiEndogShockFn % default is not to declare it    
    % Exogenous shocks
    if ~isfield(vfoptions,'n_e')
        vfoptions.n_e=0;
    end
    if ~isfield(vfoptions,'n_semiz')
        vfoptions.n_semiz=0;
    end
    % Other options
    if ~isfield(vfoptions,'piz_strictonrowsaddingtoone')
        vfoptions.piz_strictonrowsaddingtoone=0;
    end
    if ~isfield(vfoptions,'separableReturnFn')
        vfoptions.separableReturnFn=0; % advanced option to split ReturnFn into two parts (ReturnFn.R1 and ReturnFn.R2)
    end
    if ~isfield(vfoptions,'outputkron')
        vfoptions.outputkron=0;
    end
    % When calling as a subcommand, the following is used internally
    if ~isfield(vfoptions,'alreadygridvals')
        vfoptions.alreadygridvals=0;
    end
end


% If setting refinement without a d variable, then just shift to standard purediscretization (as refinement only makes sense if there is a d variable)
if strcmp(vfoptions.solnmethod,'purediscretization_refinement')
    if n_d(1)==0
        vfoptions.solnmethod='purediscretization';
    end
    if vfoptions.parallel~=2 % CPU just does the most basic thing
        vfoptions.solnmethod='purediscretization';
    end
end
% Divide-and-conquer with one endogenous state is implemented, but is too slow to be something you would ever want. 
% Especially becuase you cannot refine while doing divideandconquer
if vfoptions.divideandconquer==1 && isscalar(n_a)
    vfoptions.divideandconquer=0; 
end


%% Report on Setup
if vfoptions.verbose==1
    vfoptions
end

%% If using CPU, then go off to dedicated function
if vfoptions.parallel<2
    [V,Policy]=ValueFnIter_InfHorz_CPU(n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    varargout={V,Policy};
    return
end

%% V0 (initial guess)
if isfield(vfoptions,'V0')
    V0=reshape(gpuArray(vfoptions.V0),[N_a,N_z]);
    vfoptions.actualV0=1;
else
    V0=zeros([N_a,N_z], 'gpuArray');
    vfoptions.actualV0=0; % DC2 has different way of creating inital guess so this will be ignored
end

%% Check the sizes of some of the inputs
if strcmp(vfoptions.solnmethod,'purediscretization') || strcmp(vfoptions.solnmethod,'purediscretization_refinement')
    if N_d>0 && ~all(size(d_grid)==[sum(n_d), 1]) && ~all(size(d_grid)==[prod(n_d), length(n_d)]) % if you are using d, then should be stacked-column d_grid or joint-grid d_gridvals
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


if max(vfoptions.incrementaltype)==1
    if ~strcmp(vfoptions.solnmethod,'purediscretization') 
        error('Using vfoptions.incrementaltype only works with vfoptions.solnmethod as purediscretization')
    end
end

%% Make sure all the relevant inputs are GPU arrays (not standard arrays)
pi_z=gpuArray(pi_z);
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
z_grid=gpuArray(z_grid);

%% Switch to z_gridvals
if vfoptions.alreadygridvals==0
    [z_gridvals, pi_z, vfoptions]=ExogShockSetup_InfHorz(n_z,z_grid,pi_z,Parameters,vfoptions,3);
elseif vfoptions.alreadygridvals==1
    z_gridvals=z_grid;
end

%% Separable Return Fn
if vfoptions.separableReturnFn==1
    % Split it off here, as messes with ReturnFnParamNames and ReturnFnParamsVec
    [V,Policy]=ValueFnIter_SeparableReturnFn(V0,n_d,n_a,n_z,d_grid,a_grid,z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    varargout={V,Policy};
    return
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
    [V, Policy,ExitPolicy]=ValueFnIter_Case1_EndogExit(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    varargout={V,Policy,ExitPolicy};
    return
elseif vfoptions.endogenousexit==2 % Mixture of endogenous and exogenous exit.
    % ExitPolicy is binary decision to exit (1 is exit, 0 is 'not exit').
    % Policy is for those who remain.
    % PolicyWhenExit is current period decisions of those who will exit at end of period.
    [V, Policy, PolicyWhenExit, ExitPolicy]=ValueFnIter_Case1_EndogExit2(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    varargout={V,Policy, PolicyWhenExit,ExitPolicy};
    return
end

%% Exotic Preferences
if isfield(vfoptions,'exoticpreferences')
    if strcmp(vfoptions.exoticpreferences,'None')
        % Just ignore and will then continue on.
    elseif strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
        [V, Policy]=ValueFnIter_InfHorz_QuasiHyperbolic(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, vfoptions,Parameters,ReturnFnParamNames);
        varargout={V,Policy};
        return
    elseif strcmp(vfoptions.exoticpreferences,'EpsteinZin')
        [V, Policy]=ValueFnIter_InfHorz_EpsteinZin(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, vfoptions,Parameters,ReturnFnParamNames);
        varargout={V,Policy};
        return
    elseif vfoptions.exoticpreferences==3 % Allow the discount factor to depend on the (next period) exogenous state.
        % To implement this, can actually just replace the discount factor by 1, and adjust pi_z appropriately.
        % Note that distinguishing the discount rate and pi_z is important in almost all other contexts. Just not in this one.
        
        % Create a matrix containing the DiscountFactorParams,
        nDiscFactors=length(DiscountFactorParamNames);
        DiscountFactorParamsMatrix=Parameters.(DiscountFactorParamNames{1});
        if nDiscFactors>1
            for ii=2:nDiscFactors
                DiscountFactorParamsMatrix=DiscountFactorParamsMatrix.*(Parameters.(DiscountFactorParamNames{ii}));
            end
        end
        DiscountFactorParamsMatrix=DiscountFactorParamsMatrix.*ones(N_z,N_z,'gpuArray'); % Make it of size z-by-zprime, so that I can later just assume that it takes this shape
        DiscountFactorParamsMatrix=gpuArray(DiscountFactorParamsMatrix);
        % Set the 'fake discount factor to one.
        DiscountFactorParamsVec=1;
        % Set pi_z to include the state-dependent discount factors
        pi_z=pi_z.*DiscountFactorParamsMatrix;
    end
end

%% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
if isfield(vfoptions,'exoticpreferences')
    if vfoptions.exoticpreferences~=3
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
        if vfoptions.exoticpreferences==0
            DiscountFactorParamsVec=prod(DiscountFactorParamsVec); % Infinite horizon, so just do this once.
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec); % Infinite horizon, so just do this once.
end

%% Experience asset
if vfoptions.experienceasset==1
    % It is simply assumed that the experience asset is the last asset, and that the decision that influences it is the last decision.
    
    % Split decision variables into the standard ones and the one relevant to the experience asset
    if isscalar(n_d)
        n_d1=0;
    else
        n_d1=n_d(1:end-1);
    end
    n_d2=n_d(end); % n_d2 is the decision variable that influences next period vale of the experience asset
    d1_grid=d_grid(1:sum(n_d1));
    d2_grid=d_grid(sum(n_d1)+1:end);

    % Split endogenous assets into the standard ones and the experience asset
    if isscalar(n_a)
        n_a1=0;
    else
        n_a1=n_a(1:end-1);
    end
    n_a2=n_a(end); % n_a2 is the experience asset
    a1_grid=a_grid(1:sum(n_a1));
    a2_grid=a_grid(sum(n_a1)+1:end);

    % Now just send all this to the right value fn iteration command
    [V,Policy]=ValueFnIter_InfHorz_ExpAsset(V0,n_d1,n_d2,n_a1,n_a2,n_z, d1_grid , d2_grid, a1_grid, a2_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);

    varargout={V,Policy};
    return
end

%% Inheritance asset
if vfoptions.inheritanceasset==1
    % It is simply assumed that the inheritance asset is the last asset, and that the decision that influences it is the last decision.

    % Split decision variables into the standard ones and the one relevant to the experience asset
    if isscalar(n_d)
        n_d1=0;
    else
        n_d1=n_d(1:end-1);
    end
    n_d2=n_d(end); % n_d2 is the decision variable that influences next period vale of the experience asset
    d1_grid=d_grid(1:sum(n_d1));
    d2_grid=d_grid(sum(n_d1)+1:end);

    % Split endogenous assets into the standard ones and the experience asset
    if isscalar(n_a)
        n_a1=0;
    else
        n_a1=n_a(1:end-1);
    end
    n_a2=n_a(end); % n_a2 is the experience asset
    a1_grid=a_grid(1:sum(n_a1));
    a2_grid=a_grid(sum(n_a1)+1:end);

    % Now just send all this to the right value fn iteration command
    [V,Policy]=ValueFnIter_InfHorz_InheritAsset(V0,n_d1,n_d2,n_a1,n_a2,n_z, d1_grid , d2_grid, a1_grid, a2_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);

    varargout={V,Policy};
    return
end


%% Semi-endogenous state
% The transition matrix of the exogenous shocks depends on the value of the endogenous state.
if isfield(vfoptions,'SemiEndogShockFn')
    [V,Policy]=ValueFnIter_SemiEndo(V0, n_d, n_a, n_z, d_grid, a_grid, z_gridvals, DiscountFactorParamsVec, ReturnFn, vfoptions);
    varargout={V,Policy};
    return
end

%% Detect if using incremental endogenous states and solve this using purediscretization, prior to the main purediscretization routines
if any(vfoptions.incrementaltype==1) && strcmp(vfoptions.solnmethod,'purediscretization')
    % Incremental Endogenous States: aprime either equals a, or one grid point higher (unchanged on incremental increase)
    [VKron,Policy]=ValueFnIter_Case1_Increment(V0,n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z,ReturnFn,ReturnFnParamsVec,DiscountFactorParamsVec,vfoptions);
end


















%% Rest is all just different ways of solving the standard problem


%%
if strcmp(vfoptions.solnmethod,'purediscretization_relativeVFI') 
    % Note: have only implemented Relative VFI on the GPU
    warning('Relative VFI is unstable if you have substantial discretization (has difficulty converging if you dont use enough points)')
    [VKron,Policy]=ValueFnIter_Case1_RelativeVFI(V0,n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z,ReturnFn,ReturnFnParamsVec,DiscountFactorParamsVec,vfoptions,n_SDP,SDP1,SDP2,SDP3);
end

%%
if strcmp(vfoptions.solnmethod,'purediscretization_endogenousVFI') 
    % Note: have only implemented Endogenous VFI on the GPU
    error('Endogenous VFI is not yet working')
%     [VKron,Policy]=ValueFnIter_Case1_EndoVFI(V0,n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z,ReturnFn,ReturnFnParamsVec,DiscountFactorParamsVec,vfoptions,n_SDP,SDP1,SDP2,SDP3);
end


%%
if size(d_grid,2)==1
    d_gridvals=CreateGridvals(n_d,d_grid,1);
else % already d_gridvals
    d_gridvals=d_grid;
end

%% Divide-and-conquer together with grid interpolation layer is not yet done in InfHorz. It is assumed you just want the grid interpolation layer
if vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==1
    vfoptions.divideandconquer=0;
    warning('Cannot yet use divide-and-conquer with grid interpolation layer for InfHorz, so just ignoring the divide-and-conquer (and doing the grid interpolation layer)')
end

%% Divide-and-conquer
if vfoptions.divideandconquer==1
    warning('Divide-and-Conquer tends to be a slow option in Infinite Horizon problems')
    [V,Policy]=ValueFnIter_DivideConquer(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
    varargout={V,Policy};
    return
end
%% Grid interpolation layer
if vfoptions.gridinterplayer==1
    [V,Policy]=ValueFnIter_GridInterpLayer(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
    varargout={V,Policy};
    return
end


%%
if strcmp(vfoptions.solnmethod,'purediscretization')

    N_d=prod(n_d);

    if vfoptions.lowmemory==0
        
        %% CreateReturnFnMatrix_Disc_CPU creates a matrix of dimension (d and aprime)-by-a-by-z.
        % Since the return function is independent of time creating it once and
        % then using it every iteration is good for speed, but it does use a lot of memory.
        
        if vfoptions.verbose==1
            disp('Creating return fn matrix')
        end

        if N_d==0
            ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, [], a_grid, z_gridvals, ReturnFnParamsVec);
        else
            ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, ReturnFnParamsVec,0);
        end
        
        if vfoptions.verbose==1
            fprintf('Starting Value Function \n')
        end
        
        if N_d==0
            if vfoptions.howardsgreedy==1
                [VKron,Policy]=ValueFnIter_nod_HowardGreedy_raw(V0, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            elseif vfoptions.howardsgreedy==0
                if vfoptions.howardssparse==0
                    [VKron,Policy]=ValueFnIter_nod_raw(V0, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
                elseif vfoptions.howardssparse==1
                    [VKron,Policy]=ValueFnIter_sparse_nod_raw(V0, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
                end
            end
        else
            % Can't be bothered implementing HowardGreedy here, as for good runtimes you should anyway be doing Refine so wouldn't get here
            [VKron, Policy]=ValueFnIter_raw(V0, n_d,n_a,n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, vfoptions.maxiter);
        end
        
    elseif vfoptions.lowmemory==1
                
        if vfoptions.verbose==1
            disp('Starting Value Function')
        end
        
        if N_d==0
            if vfoptions.howardssparse==0
                [VKron,Policy]=ValueFnIter_LowMem_nod_raw(V0, n_a, n_z, a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            elseif vfoptions.howardssparse==1
                [VKron,Policy]=ValueFnIter_LowMem_sparse_nod_raw(V0, n_a, n_z, a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            end
        else
            [VKron, Policy]=ValueFnIter_LowMem_raw(V0, n_d,n_a,n_z, d_gridvals, a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, vfoptions.maxiter);
        end
    end
end

%% VFI with Refine
% If we get to refinement then there must be d variable
if strcmp(vfoptions.solnmethod,'purediscretization_refinement') 
    % Refinement: Presolve for dstar(aprime,a,z). Then solve value function for just aprime,a,z. 
    [VKron,Policy]=ValueFnIter_Refine(V0,n_d,n_a,n_z,d_gridvals,a_grid,z_gridvals,pi_z,ReturnFn,ReturnFnParamsVec,DiscountFactorParamsVec,vfoptions);
end


if vfoptions.verbose==1
    disp('Transforming Value Fn and Optimal Policy matrices back out of Kronecker Form')
    tic;
end

%% Cleaning up the output
if vfoptions.outputkron==0
    V=reshape(VKron,[n_a,n_z]);
    Policy=UnKronPolicyIndexes_Case1(Policy, n_d, n_a, n_z,vfoptions);
else
    Policy=reshape(Policy,[1,N_a,N_z]);
    varargout={VKron,Policy};
    return
end

varargout={V,Policy};

end
