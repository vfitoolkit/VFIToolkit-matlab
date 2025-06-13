function varargout=ValueFnIter_Case1(n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Solves infinite-horizon 'Case 1' value function problems.
% Typically, varargoutput={V,Policy};

V=nan; % Matlab was complaining that V was not assigned

%% Check which vfoptions have been used, set all others to defaults 
if ~exist('vfoptions','var')
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
    vfoptions.solnmethod='purediscretization_refinement'; % if no d variable, will be set to 'purediscretization' below
    vfoptions.divideandconquer=0;
    vfoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    if vfoptions.parallel==2
        vfoptions.returnmatrix=2; % On GPU, must use this option
    end
    if isfield(vfoptions,'returnmatrix')==0
        if isa(ReturnFn,'function_handle')==1
            vfoptions.returnmatrix=0;
        else
            vfoptions.returnmatrix=1;
        end
    end
    vfoptions.lowmemory=0;
    vfoptions.verbose=0;
    vfoptions.tolerance=10^(-9);
    vfoptions.howards=80;
    vfoptions.maxhowards=500;
    vfoptions.maxiter=Inf;
    vfoptions.endogenousexit=0;
    vfoptions.endotype=0; % (vector indicating endogenous state is a type)
    vfoptions.incrementaltype=0; % (vector indicating endogenous state is an incremental endogenous state variable)
%     vfoptions.exoticpreferences % default is not to declare it
%     vfoptions.SemiEndogShockFn % default is not to declare it    
    vfoptions.experienceasset=0;
    vfoptions.polindorval=1;
    vfoptions.policy_forceintegertype=0;
    vfoptions.piz_strictonrowsaddingtoone=0;
    vfoptions.separableReturnFn=0; % advanced option to split ReturnFn into two parts (ReturnFn.R1 and ReturnFn.R2)
    vfoptions.outputkron=0;
    % When calling as a subcommand, the following is used internally
    vfoptions.alreadygridvals=0;
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(vfoptions,'solnmethod')
        vfoptions.solnmethod='purediscretization_refinement'; % if no d variable, will be set to 'purediscretization' below
    end
    if ~isfield(vfoptions,'divideandconquer')
        vfoptions.divideandconquer=0;
    end
    if ~isfield(vfoptions,'parallel')
        vfoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    end
    if vfoptions.parallel==2
        vfoptions.returnmatrix=2; % On GPU, must use this option
    end
    if ~isfield(vfoptions,'returnmatrix')
        if isa(ReturnFn,'function_handle')
            vfoptions.returnmatrix=0;
        else
            vfoptions.returnmatrix=1;
        end
    end
    if ~isfield(vfoptions,'lowmemory')
        vfoptions.lowmemory=0;
    end
    if ~isfield(vfoptions,'verbose')
        vfoptions.verbose=0;
    end
    if ~isfield(vfoptions,'tolerance')
        vfoptions.tolerance=10^(-9);
    end
    if ~isfield(vfoptions,'howards')
        vfoptions.howards=80;
    end  
    if ~isfield(vfoptions,'maxhowards')
        vfoptions.maxhowards=500;
    end  
    if ~isfield(vfoptions,'maxiter')
        vfoptions.maxiter=Inf;
    end
    if ~isfield(vfoptions,'endogenousexit')
        vfoptions.endogenousexit=0;
    end
    if ~isfield(vfoptions,'endotype')
        vfoptions.endotype=0; % (vector indicating endogenous state is a type)
    end
    if ~isfield(vfoptions,'incrementaltype')
        vfoptions.incrementaltype=0; % (vector indicating endogenous state is an incremental endogenous state variable)
    end
%     vfoptions.exoticpreferences % default is not to declare it
%     vfoptions.SemiEndogShockFn % default is not to declare it    
    if ~isfield(vfoptions,'experienceasset')
        vfoptions.experienceasset=0;
    end
    if ~isfield(vfoptions,'polindorval')
        vfoptions.polindorval=1;
    end
    if ~isfield(vfoptions,'policy_forceintegertype')
        vfoptions.policy_forceintegertype=0;
    end
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
if strcmp(vfoptions.solnmethod,'purediscretization_refinement') || strcmp(vfoptions.solnmethod,'purediscretization_refinement2') 
    if n_d(1)==0
        vfoptions.solnmethod='purediscretization';
    end
end
% Divide-and-conquer with one endogenous state is implemented, but is too slow to be something you would ever want. 
% Especially becuase you cannot refine while doing divideandconquer
if vfoptions.divideandconquer==1 && isscalar(n_a)
    vfoptions.divideandconquer=0; 
end


N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if isfield(vfoptions,'V0')
    V0=reshape(vfoptions.V0,[N_a,N_z]);
    vfoptions.actualV0=1;
else
    if vfoptions.parallel==2
        V0=zeros([N_a,N_z], 'gpuArray');
    else
        V0=zeros([N_a,N_z]);
    end
    vfoptions.actualV0=0; % DC2 has different way of creating inital guess so this will be ignored
end


%% Check the sizes of some of the inputs
if strcmp(vfoptions.solnmethod,'purediscretization') || strcmp(vfoptions.solnmethod,'purediscretization_refinement') || strcmp(vfoptions.solnmethod,'localpolicysearch')
    if N_d>0 && ~all(size(d_grid)==[sum(n_d), 1])
        error('d_grid is not the correct shape (should be of size sum(n_d)-by-1)')
    elseif ~all(size(a_grid)==[sum(n_a), 1])
        error('a_grid is not the correct shape (should be of size sum(n_a)-by-1)')
    elseif N_z>0
        if ~all(size(z_grid)==[sum(n_z), 1])
            if all(size(z_grid)==[prod(n_z),length(n_z)])
                % Using joint grids
            else
                error('z_grid is not the correct shape (should be of size sum(n_z)-by-1)')
            end
        elseif size(pi_z)~=[N_z, N_z]
            error('pi is not of size N_z-by-N_z')
        end
    elseif n_z(end)>1 % Ignores this final check if last dimension of n_z is singleton as will cause an error
        if ndims(V0)>2
            if size(V0)~=[n_a,n_z] % Allow for input to be already transformed into Kronecker form
                error('Starting choice for ValueFn is not of size [n_a,n_z]')
            end
        elseif size(V0)~=[N_a,N_z] % Allows for possiblity that V0 is already in kronecker form
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

if max(vfoptions.endotype)==1
    if ~strcmp(vfoptions.solnmethod,'purediscretization_refinement2') 
        error('Using vfoptions.endotype only works with vfoptions.solnmethod as purediscretization_refinement2')
    end
end
if max(vfoptions.incrementaltype)==1
    if ~strcmp(vfoptions.solnmethod,'purediscretization') 
        error('Using vfoptions.incrementaltype only works with vfoptions.solnmethod as purediscretization')
    end
end

%% Separable Return Fn
if vfoptions.separableReturnFn==1
    % Split it off here, as messes with ReturnFnParamNames and ReturnFnParamsVec
    [V,Policy]=ValueFnIter_SeparableReturnFn(V0,n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
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



%%

if vfoptions.parallel==2 
   % If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
   V0=gpuArray(V0);
   pi_z=gpuArray(pi_z);
   d_grid=gpuArray(d_grid);
   a_grid=gpuArray(a_grid);
   z_grid=gpuArray(z_grid);
end

if vfoptions.verbose==1
    vfoptions
end

%% Switch to z_gridvals
if vfoptions.alreadygridvals==0
    if vfoptions.parallel<2
        % only basics allowed with cpu
        z_gridvals=z_grid;
    else
        [z_gridvals, pi_z, vfoptions]=ExogShockSetup(n_z,z_grid,pi_z,Parameters,vfoptions,3);
    end
elseif vfoptions.alreadygridvals==1
    z_gridvals=z_grid;
end


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
        [V, Policy]=ValueFnIter_Case1_QuasiHyperbolic(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, vfoptions,Parameters,ReturnFnParamNames);
        varargout={V,Policy};
        return
    elseif strcmp(vfoptions.exoticpreferences,'EpsteinZin')
        [V, Policy]=ValueFnIter_Case1_EpsteinZin(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, vfoptions,Parameters,ReturnFnParamNames);
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
        DiscountFactorParamsMatrix=DiscountFactorParamsMatrix.*ones(N_z,N_z); % Make it of size z-by-zprime, so that I can later just assume that it takes this shape
        if vfoptions.parallel==2
            DiscountFactorParamsMatrix=gpuArray(DiscountFactorParamsMatrix);
        end
        % Set the 'fake discount factor to one.
        DiscountFactorParamsVec=1;
        % Set pi_z to include the state-dependent discount factors
        pi_z=pi_z.*DiscountFactorParamsMatrix;
    end
end

%% State Dependent Parameters
n_SDP=0;
SDP1=[]; SDP2=[]; SDP3=[];
if isfield(vfoptions,'statedependentparams')
    % Remove the statedependentparams from ReturnFnParamNames
    ReturnFnParamNames=setdiff(ReturnFnParamNames,vfoptions.statedependentparams.names);
    % Note that the codes assume that the statedependentparams are the first elements in ReturnFnParamNames
    % Codes currently allow up to three state dependent parameters
    n_SDP=length(vfoptions.statedependentparams.names);
    if N_d>1
        l_d=length(n_d);
        n_full=[n_d,n_a,n_a,n_z];
    else
        l_d=0;
        n_full=[n_a,n_a,n_z];
    end
    l_a=length(n_a);
    l_z=length(n_z);
    
    % First state dependent parameter, get into form needed for the valuefn
    SDP1=Params.(vfoptions.statedependentparams.names{1});
    SDP1_dims=vfoptions.statedependentparams.dimensions.(vfoptions.statedependentparams.names{1});
%     vfoptions.statedependentparams.dimensions.kmax=[3,4,5,6,7]; % The d,a & z variables (in VFI toolkit notation)
    temp=ones(1,l_d+l_a+l_a+l_z);
    for jj=1:max(SDP1_dims)
        [v,ind]=max(SDP1_dims==jj);
        if v==1
            temp(jj)=n_full(ind);
        end
    end
    if isscalar(SDP1)
        SDP1=SDP1*ones(temp);
    else
        SDP1=reshape(SDP1,temp);
    end
    if n_SDP>=2
        % Second state dependent parameter, get into form needed for the valuefn
        SDP2=Params.(vfoptions.statedependentparams.names{2});
        SDP2_dims=vfoptions.statedependentparams.dimensions.(vfoptions.statedependentparams.names{2});
        temp=ones(1,l_d+l_a+l_a+l_z);
        for jj=1:max(SDP2_dims)
            [v,ind]=max(SDP2_dims==jj);
            if v==1
                temp(jj)=n_full(ind);
            end
        end
        if isscalar(SDP2)
            SDP2=SDP2*ones(temp);
        else
            SDP2=reshape(SDP2,temp);
        end
    end
    if n_SDP>=3
        % Third state dependent parameter, get into form needed for the valuefn
        SDP3=Params.(vfoptions.statedependentparams.names{3});
        SDP3_dims=vfoptions.statedependentparams.dimensions.(vfoptions.statedependentparams.names{3});
        temp=ones(1,l_d+l_a+l_a+l_z);
        for jj=1:max(SDP3_dims)
            [v,ind]=max(SDP3_dims==jj);
            if v==1
                temp(jj)=n_full(ind);
            end
        end
        if isscalar(SDP3)
            SDP3=SDP3*ones(temp);
        else
            SDP3=reshape(SDP3,temp);
        end
    end
    if n_SDP>3
        fprintf('WARNING: currently only three state dependent parameters are allowed. If you have a need for more please email robertdkirkby@gmail.com and let me know (I can easily implement more if needed) \n')
        dbstack
        return
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
    
    if length(n_a)==1
        error('experienceasset in InfHorz is only coded alongside a standard endogenous state')
    elseif length(n_a)==2
        % Split decision variables into the standard ones and the one relevant to the experience asset
        if length(n_d)==1
            n_d1=0;
        else
            n_d1=n_d(1:end-1);
        end
        n_d2=n_d(end); % n_d2 is the decision variable that influences next period vale of the experience asset
        d1_grid=d_grid(1:sum(n_d1));
        d2_grid=d_grid(sum(n_d1)+1:end);
        % Split endogenous assets into the standard ones and the experience asset
        if length(n_a)==1
            n_a1=0;
        else
            n_a1=n_a(1:end-1);
        end
        n_a2=n_a(end); % n_a2 is the experience asset
        a1_grid=a_grid(1:sum(n_a1));
        a2_grid=a_grid(sum(n_a1)+1:end);

        % Now just send all this to the right value fn iteration command
        [V,Policy]=ValueFnIter_Case1_ExpAsset(V0,n_d1,n_d2,n_a1,n_a2,n_z, d1_grid , d2_grid, a1_grid, a2_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
    else
        error('experienceasset in InfHorz is only coded alongside a single standard endogenous state (you have length(n_a)>2)')
    end
    varargout={V,Policy};
    return
end

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

%% Divide-and-conquer
if vfoptions.divideandconquer==1
    [V,Policy]=ValueFnIter_DivideConquer(V0, n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
    varargout={V,Policy};
    return
end


%%
if strcmp(vfoptions.solnmethod,'purediscretization')
    if vfoptions.parallel==1 && vfoptions.lowmemory==2
        fprintf('Use of vfoptions.lowmemory=2 in not supported for cpu, have switched to vfoptions.lowmemory=1 \n')
        vfoptions.lowmemory=1;
    end

    if vfoptions.lowmemory==0
        
        %% CreateReturnFnMatrix_Case1_Disc creates a matrix of dimension (d and aprime)-by-a-by-z.
        % Since the return function is independent of time creating it once and
        % then using it every iteration is good for speed, but it does use a lot of memory.
        
        if vfoptions.verbose==1
            disp('Creating return fn matrix')
        end
        
        if isfield(vfoptions,'statedependentparams')
            if vfoptions.returnmatrix==2 % GPU
                if n_SDP==3
                    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2_SDP(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_gridvals, ReturnFnParamsVec,SDP1,SDP2,SDP3);
                elseif n_SDP==2
                    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2_SDP(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_gridvals, ReturnFnParamsVec,SDP1,SDP2);
                elseif n_SDP==1
                    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2_SDP(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_gridvals, ReturnFnParamsVec,SDP1);
                end
            else
                fprintf('ERROR: statedependentparams only works with GPU (parallel=2) \n')
                dbstack
            end
        else % Following is the normal/standard behavior
            if vfoptions.returnmatrix==0
                ReturnMatrix=CreateReturnFnMatrix_Case1_Disc(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_gridvals, vfoptions.parallel, ReturnFnParamsVec);
            elseif vfoptions.returnmatrix==1
                ReturnMatrix=ReturnFn;
            elseif vfoptions.returnmatrix==2 % GPU
                ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_gridvals, ReturnFnParamsVec);
            end
        end
        
        if vfoptions.verbose==1
            fprintf('Starting Value Function \n')
        end
                
        if n_d(1)==0
            if vfoptions.parallel==0     % On CPU
                [VKron,Policy]=ValueFnIter_Case1_NoD_raw(V0, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
            elseif vfoptions.parallel==1 % On Parallel CPU
                [VKron,Policy]=ValueFnIter_Case1_NoD_Par1_raw(V0, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
            elseif vfoptions.parallel==2 % On GPU
                [VKron,Policy]=ValueFnIter_Case1_NoD_Par2_raw(V0, n_a, n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter); %  a_grid, z_grid,
            end
        else
            if vfoptions.parallel==0 % On CPU
                [VKron, Policy]=ValueFnIter_Case1_raw(V0, N_d,N_a,N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
            elseif vfoptions.parallel==1 % On Parallel CPU
                [VKron, Policy]=ValueFnIter_Case1_Par1_raw(V0, N_d,N_a,N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
            elseif vfoptions.parallel==2 % On GPU
                [VKron, Policy]=ValueFnIter_Case1_Par2_raw(V0, n_d,n_a,n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
            end
        end
        
    elseif vfoptions.lowmemory==1
                
        if vfoptions.verbose==1
            disp('Starting Value Function')
        end
        
        if n_d(1)==0
            if vfoptions.parallel==0
                [VKron,Policy]=ValueFnIter_Case1_LowMem_NoD_raw(V0, n_a, n_z, a_grid, z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
            elseif vfoptions.parallel==1
                [VKron,Policy]=ValueFnIter_Case1_LowMem_NoD_Par1_raw(V0, n_a, n_z, a_grid, z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.verbose);
            elseif vfoptions.parallel==2 % On GPU
                [VKron,Policy]=ValueFnIter_Case1_LowMem_NoD_Par2_raw(V0, n_a, n_z, a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
            end
        else
            if vfoptions.parallel==0
                [VKron, Policy]=ValueFnIter_Case1_LowMem_raw(V0, n_d,n_a,n_z, d_grid,a_grid,z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
            elseif vfoptions.parallel==1
                [VKron, Policy]=ValueFnIter_Case1_LowMem_Par1_raw(V0, n_d,n_a,n_z, d_grid,a_grid,z_grid,pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, vfoptions.verbose);
            elseif vfoptions.parallel==2 % On GPU
                [VKron, Policy]=ValueFnIter_Case1_LowMem_Par2_raw(V0, n_d,n_a,n_z, d_grid, a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
            end
        end
    end
end

%%
% If we get to refinement and refinement2 then there is no d variable
if strcmp(vfoptions.solnmethod,'purediscretization_refinement') 
    % Refinement: Presolve for dstar(aprime,a,z). Then solve value function for just aprime,a,z. 
    [VKron,Policy]=ValueFnIter_Case1_Refine(V0,n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z,ReturnFn,ReturnFnParamsVec,DiscountFactorParamsVec,vfoptions);
end

if strcmp(vfoptions.solnmethod,'purediscretization_refinement2') 
    % Refinement: Presolve for dstar(aprime,a,z). Then solve value function for just aprime,a,z. 
    % Refinement 2: Multigrid approach when presolving for dstar(aprime,a,z).
    
    % Check that the info about layers is provided
    if ~isfield(vfoptions,'refine_pts') % points per dimension per layer
        error('Using vfoptions.solnmethod purediscretization_refinement2 you must declare vfoptions.refine_pts')
    else
        if rem(vfoptions.refine_pts,2)~=1
            error('vfoptions.refine_pts must be an odd number')
        end
    end
    if ~isfield(vfoptions,'refine_iter') % number of layers
        error('Using vfoptions.solnmethod purediscretization_refinement2 you must declare vfoptions.refine_iter')
    end
    
    % Check that grid size for d variables matches the ptsperlayer and 
    RequiredGridPoints=nGridPointsWithLayers(vfoptions);
    for ii=1:length(n_d)
        if n_d(ii)~=RequiredGridPoints
            fprintf('Problem with the %i-th decision variable \n',ii)
            fprintf('With current settings for layers (in vfoptions) you should be using %i points for each decision variable \n',RequiredGridPoints)
            error('The number of points in the grid for the i-th variable is does not fit layers')
        end
    end
    
    if max(vfoptions.endotype)==0 % If they are all zeros, no endo types are used
        [VKron,Policy]=ValueFnIter_Case1_Refine2(V0,l_d,N_a,N_z,n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z,ReturnFn,ReturnFnParamsVec,DiscountFactorParamsVec,vfoptions);
    else
        % Need to seperate endogenous states from endogenous types to take advantage of them
        n_endostate=n_a(logical(1-vfoptions.endotype));
        n_endotype=n_a(logical(vfoptions.endotype));
        endostate_grid=zeros(sum(n_endostate),1);
        endotype_grid=zeros(sum(n_endotype),1);
        endostate_c=1;
        endotype_c=1;
        if vfoptions.endotype(1)==1 % Endogenous type
            endotype_grid(1:n_a(1))=a_grid(1:n_a(1));
            endotype_c=endotype_c+1;
        else % Endogenous state
            endostate_grid(1:n_a(1))=a_grid(1:n_a(1));
            endostate_c=endostate_c+1;
        end
        for ii=2:length(n_a)
            if vfoptions.endotype(ii)==1 % Endogenous type
                if endotype_c==1
                    endotype_grid(1:n_endotype(1))=a_grid(1+sum(n_a(1:ii-1)):sum(n_a(1:ii)));
                else
                    endotype_grid(1+sum(n_endotype(1:endotype_c-1)):sum(n_endotype(1:endotype_c)))=a_grid(1+sum(n_a(1:ii-1)):sum(n_a(1:ii)));
                end
                endotype_c=endotype_c+1;
            else % Endogenous state
                if endotype_c==1
                    endostate_grid(1:n_endostate(1))=a_grid(1+sum(n_a(1:ii-1)):sum(n_a(1:ii)));
                else
                    endostate_grid(1+sum(n_endostate(1:endostate_c-1)):sum(n_endostate(1:endostate_c)))=a_grid(1+sum(n_a(1:ii-1)):sum(n_a(1:ii)));
                end
                endostate_c=endostate_c+1;
            end
        end
        
        [VKron,Policy]=ValueFnIter_Case1_EndoType_Refine2(V0,l_d,prod(n_endostate),N_z,n_d,n_endostate,n_z,n_endotype,d_grid,endostate_grid,z_grid,endotype_grid,pi_z,ReturnFn,ReturnFnParamsVec,DiscountFactorParamsVec,vfoptions);
    end
    % To be able to resize the output we need to treat endotype is just
    % another endogenous state. This will happen because of how we have n_a setup.
end

%%
if strcmp(vfoptions.solnmethod,'purediscretization_PFI') 
    % Note: have only implemented PFI on the GPU
    [VKron,Policy]=ValueFnIter_Case1_PolicyFnIter(V0,n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z,ReturnFn,ReturnFnParamsVec,DiscountFactorParamsVec,vfoptions,n_SDP,SDP1,SDP2,SDP3);
end

if vfoptions.verbose==1
    time=toc;
    fprintf('Time to solve for Value Fn and Policy: %8.4f \n', time)
    disp('Transforming Value Fn and Optimal Policy matrices back out of Kronecker Form')
    tic;
end

%% Cleaning up the output
if vfoptions.outputkron==0
    V=reshape(VKron,[n_a,n_z]);
    Policy=UnKronPolicyIndexes_Case1(Policy, n_d, n_a, n_z,vfoptions);
    if vfoptions.verbose==1
        time=toc;
        fprintf('Time to create UnKron Value Fn and Policy: %8.4f \n', time)
    end
else
    varargout={VKron,Policy};
    return
end

if vfoptions.polindorval==2
    Policy=PolicyInd2Val_Case1(Policy,n_d,n_a,n_z,d_grid, a_grid);
end
    
% Sometimes numerical rounding errors (of the order of 10^(-16) can mean
% that Policy is not integer valued. The following corrects this by converting to int64 and then
% makes the output back into double as Matlab otherwise cannot use it in
% any arithmetical expressions.
if vfoptions.policy_forceintegertype==1
    Policy=uint64(Policy);
    Policy=double(Policy);
end

varargout={V,Policy};

end
