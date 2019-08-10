function [V, Policy]=ValueFnIter_Case1(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Solves infinite-horizon 'Case 1' value function problems.

V=nan; % Matlab was complaining that V was not assigned

%% Check which vfoptions have been used, set all others to defaults 
if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
    vfoptions.solnmethod='purediscretization';
    vfoptions.parallel=2;
    vfoptions.returnmatrix=2;
    vfoptions.lowmemory=0;
    vfoptions.verbose=0;
    vfoptions.tolerance=10^(-9);
    vfoptions.howards=80;
    vfoptions.maxhowards=500;
    vfoptions.exoticpreferences=0;
    vfoptions.polindorval=1;
    vfoptions.policy_forceintegertype=0;
    vfoptions.piz_strictonrowsaddingtoone=0;
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if isfield(vfoptions,'solnmethod')==0
        vfoptions.solnmethod='purediscretization';
    end
    if isfield(vfoptions,'parallel')==0
        vfoptions.parallel=2;
    end
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
    if isfield(vfoptions,'lowmemory')==0
        vfoptions.lowmemory=0;
    end
    if isfield(vfoptions,'verbose')==0
        vfoptions.verbose=0;
    end
     if isfield(vfoptions,'tolerance')==0
        vfoptions.tolerance=10^(-9);
    end
    if isfield(vfoptions,'howards')==0
        vfoptions.howards=80;
    end  
    if isfield(vfoptions,'maxhowards')==0
        vfoptions.maxhowards=500;
    end  
    if isfield(vfoptions,'exoticpreferences')==0
        vfoptions.exoticpreferences=0;
    end  
    if isfield(vfoptions,'polindorval')==0
        vfoptions.polindorval=1;
    end
    if isfield(vfoptions,'policy_forceintegertype')==0
        vfoptions.policy_forceintegertype=0;
    end
    if isfield(vfoptions,'piz_strictonrowsaddingtoone')==0
        vfoptions.piz_strictonrowsaddingtoone=0;
    end
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);


%% Check the sizes of some of the inputs
if strcmp(vfoptions.solnmethod,'purediscretization')
    if size(d_grid)~=[sum(n_d), 1]
        disp('ERROR: d_grid is not the correct shape (should be of size sum(n_d)-by-1)')
        dbstack
        return
    elseif size(a_grid)~=[sum(n_a), 1]
        disp('ERROR: a_grid is not the correct shape (should be of size sum(n_a)-by-1)')
        dbstack
        return
    elseif size(z_grid)~=[sum(n_z), 1]
        disp('ERROR: z_grid is not the correct shape (should be of size sum(n_z)-by-1)')
        dbstack
        return
    elseif size(pi_z)~=[N_z, N_z]
        disp('ERROR: pi is not of size N_z-by-N_z')
        dbstack
        return
    elseif n_z(end)>1 % Ignores this final check if last dimension of n_z is singleton as will cause an error
        if size(V0)~=[n_a,n_z] % Allow for input to be already transformed into Kronecker form
            disp('ERROR: Starting choice for ValueFn is not of size [n_a,n_z]')
            dbstack
            return
        end
    end
end

if min(min(pi_z))<0
    fprintf('WARNING: Problem with pi_z in ValueFnIter_Case1: min(min(pi_z))<0 \n')
    dbstack
    return
elseif vfoptions.piz_strictonrowsaddingtoone==1
    if max(sum(pi_z,2))~=1 || min(sum(pi_z,2))~=1
        fprintf('WARNING: Problem with pi_z in ValueFnIter_Case1: rows do not sum to one \n')
        dbstack
        return
    end
elseif vfoptions.piz_strictonrowsaddingtoone==0
    if max(abs((sum(pi_z,2))-1)) > 10^(-13)
        fprintf('WARNING: Problem with pi_z in ValueFnIter_Case1: rows do not sum to one \n')
        dbstack
        return
    end
end

%%

if vfoptions.parallel==2 
   % If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
   V0=gpuArray(V0);
   pi_z=gpuArray(pi_z);
   d_grid=gpuArray(d_grid);
   a_grid=gpuArray(a_grid);
   z_grid=gpuArray(z_grid);
% else
%    % If using CPU make sure all the relevant inputs are CPU arrays (not standard arrays)
%    % This may be completely unnecessary.
%    V0=gather(V0);
%    pi_z=gather(pi_z);
%    d_grid=gather(d_grid);
%    a_grid=gather(a_grid);
%    z_grid=gather(z_grid);
end

if vfoptions.verbose==1
    vfoptions
end

if vfoptions.exoticpreferences==0
    if length(DiscountFactorParamNames)~=1
        disp('WARNING: There should only be a single Discount Factor (in DiscountFactorParamNames) when using standard VFI')
        dbstack
    end
elseif vfoptions.exoticpreferences==1 % 'alpha-beta' quasi-geometric discounting
    %NOT YET IMPLEMENTED
%    [V, Policy]=ValueFnIter_Case1_QuasiGeometric(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, vfoptions,Parameters,ReturnFnParamNames);
%    return
elseif vfoptions.exoticpreferences==2 % Epstein-Zin preferences
    [V, Policy]=ValueFnIter_Case1_EpsteinZin(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, vfoptions,Parameters,ReturnFnParamNames);
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

if strcmp(vfoptions.solnmethod,'smolyak_chebyshev') 
    % Solve value function using smolyak grids and chebyshev polynomials (see Judd, Maliar, Maliar & Valero (2014).
    [V, Policy]=ValueFnIter_Case1_SmolyakChebyshev(V0, n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z, DiscountFactorParamNames, ReturnFn, Parameters, ReturnFnParamNames, vfoptions);
    return
end

%%
% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
if vfoptions.exoticpreferences~=3
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
end

%%
if vfoptions.lowmemory==0
    
    %% CreateReturnFnMatrix_Case1_Disc creates a matrix of dimension (d and aprime)-by-a-by-z.
    % Since the return function is independent of time creating it once and
    % then using it every iteration is good for speed, but it does use a
    % lot of memory.

    if vfoptions.verbose==1
        disp('Creating return fn matrix')
        tic;
    end
    
    if vfoptions.returnmatrix==0
        fprintf('NOTE: You are not using GPU parallelization. \n NOTE: Your codes will run slowly unless you use vfoptions.returnmatrix=1 \n NOTE: (rather than current vfoptions.returnmatrix=0). \n NOTE: See documentation on vfoptions.returnmatrix option for more.')
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, vfoptions.parallel, ReturnFnParamsVec);
    elseif vfoptions.returnmatrix==1
        ReturnMatrix=ReturnFn;
    elseif vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
    end
    
    if vfoptions.verbose==1
        time=toc;
        fprintf('Time to create return fn matrix: %8.4f \n', time)
        disp('Starting Value Function')
        tic;
    end
        
    %%
    V0Kron=reshape(V0,[N_a,N_z]);
    
    if n_d(1)==0
        if vfoptions.parallel==0     % On CPU
            [VKron,Policy]=ValueFnIter_Case1_NoD_raw(V0Kron, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
        elseif vfoptions.parallel==1 % On Parallel CPU
            [VKron,Policy]=ValueFnIter_Case1_NoD_Par1_raw(V0Kron, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
        elseif vfoptions.parallel==2 % On GPU
            [VKron,Policy]=ValueFnIter_Case1_NoD_Par2_raw(V0Kron, n_a, n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance); %  a_grid, z_grid,
        end
    else
        if vfoptions.parallel==0 % On CPU
            [VKron, Policy]=ValueFnIter_Case1_raw(V0Kron, N_d,N_a,N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
        elseif vfoptions.parallel==1 % On Parallel CPU
            [VKron, Policy]=ValueFnIter_Case1_Par1_raw(V0Kron, N_d,N_a,N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
        elseif vfoptions.parallel==2 % On GPU
            [VKron, Policy]=ValueFnIter_Case1_Par2_raw(V0Kron, n_d,n_a,n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
        end
    end
    
elseif vfoptions.lowmemory==1    
    
    V0Kron=reshape(V0,[N_a,N_z]);
    
    if vfoptions.verbose==1
        disp('Starting Value Function')
        tic;
    end
    
    if n_d(1)==0
        if vfoptions.parallel==0
            [VKron,Policy]=ValueFnIter_Case1_LowMem_NoD_raw(V0Kron, n_a, n_z, a_grid, z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
        elseif vfoptions.parallel==1
            [VKron,Policy]=ValueFnIter_Case1_LowMem_NoD_Par1_raw(V0Kron, n_a, n_z, a_grid, z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.verbose);
        elseif vfoptions.parallel==2 % On GPU
            [VKron,Policy]=ValueFnIter_Case1_LowMem_NoD_Par2_raw(V0Kron, n_a, n_z, a_grid, z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
        end
    else
        if vfoptions.parallel==0
            [VKron, Policy]=ValueFnIter_Case1_LowMem_raw(V0Kron, n_d,n_a,n_z, d_grid,a_grid,z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
        elseif vfoptions.parallel==1
            [VKron, Policy]=ValueFnIter_Case1_LowMem_Par1_raw(V0Kron, n_d,n_a,n_z, d_grid,a_grid,z_grid,pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, vfoptions.verbose);
        elseif vfoptions.parallel==2 % On GPU
            [VKron, Policy]=ValueFnIter_Case1_LowMem_Par2_raw(V0Kron, n_d,n_a,n_z, d_grid, a_grid, z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
        end
    end
    
elseif vfoptions.lowmemory==2
    
    V0Kron=reshape(V0,[N_a,N_z]);
    
    if vfoptions.verbose==1
        disp('Starting Value Function')
        tic;
    end
    
    if n_d(1)==0
        if vfoptions.parallel==2
            [VKron,Policy]=ValueFnIter_Case1_LowMem2_NoD_Par2_raw(V0Kron, n_a, n_z, a_grid, z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions.howards, vfoptions.tolerance);
        end
    else
        if vfoptions.parallel==2 % On GPU
            [VKron, Policy]=ValueFnIter_Case1_LowMem2_Par2_raw(V0Kron, n_d,n_a,n_z, d_grid, a_grid, z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec ,vfoptions.howards,vfoptions.tolerance);
        end
    end
end

if vfoptions.verbose==1
    time=toc;
    fprintf('Time to solve for Value Fn and Policy: %8.4f \n', time)
    disp('Transforming Value Fn and Optimal Policy matrices back out of Kronecker Form')
    tic;
end
%% Cleaning up the output
V=reshape(VKron,[n_a,n_z]);
Policy=UnKronPolicyIndexes_Case1(Policy, n_d, n_a, n_z,vfoptions);
if vfoptions.verbose==1
    time=toc;
    fprintf('Time to create UnKron Value Fn and Policy: %8.4f \n', time)
end

if vfoptions.polindorval==2
    Policy=PolicyInd2Val_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,vfoptions.parallel);
end
    
% Sometimes numerical rounding errors (of the order of 10^(-16) can mean
% that Policy is not integer valued. The following corrects this by converting to int64 and then
% makes the output back into double as Matlab otherwise cannot use it in
% any arithmetical expressions.
if vfoptions.policy_forceintegertype==1
    Policy=uint64(Policy);
    Policy=double(Policy);
end

end