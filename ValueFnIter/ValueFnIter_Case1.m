function [V, Policy]=ValueFnIter_Case1(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, vfoptions,Parameters,ReturnFnParamNames)

%% Check which vfoptions have been used, set all others to defaults 
if nargin<11
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
    vfoptions.tolerance=10^(-9);
    vfoptions.howards=80;
    vfoptions.maxhowards=500;
    vfoptions.exoticpreferences=0;
    vfoptions.parallel=2;
    vfoptions.returnmatrix=2;
    vfoptions.verbose=0;
    vfoptions.lowmemory=0;
    vfoptions.polindorval=1;
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    eval('fieldexists=1;vfoptions.parallel;','fieldexists=0;')
    if fieldexists==0
        vfoptions.parallel=2;
    end
    if vfoptions.parallel==2
        vfoptions.returnmatrix=2; % On GPU, must use this option
    end
    eval('fieldexists=1;vfoptions.lowmemory;','fieldexists=0;')
    if fieldexists==0
        vfoptions.lowmemory=0;
    end
    eval('fieldexists=1;vfoptions.howards;','fieldexists=0;')
    if fieldexists==0
        vfoptions.howards=80;
    end  
    eval('fieldexists=1;vfoptions.maxhowards;','fieldexists=0;')
    if fieldexists==0
        vfoptions.maxhowards=500;
    end  
    eval('fieldexists=1;vfoptions.exoticpreferences;','fieldexists=0;')
    if fieldexists==0
        vfoptions.exoticpreferences=0;
    end  
    eval('fieldexists=1;vfoptions.verbose;','fieldexists=0;')
    if fieldexists==0
        vfoptions.verbose=0;
    end
    eval('fieldexists=1;vfoptions.returnmatrix;','fieldexists=0;')
    if fieldexists==0 % If still doesn't exist by now, then not using GPU
        vfoptions.returnmatrix=1;
    end
    eval('fieldexists=1;vfoptions.tolerance;','fieldexists=0;')
    if fieldexists==0
        vfoptions.tolerance=10^(-9);
    end
    eval('fieldexists=1;vfoptions.polindorval;','fieldexists=0;')
    if fieldexists==0
        vfoptions.polindorval=1;
    end
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);


%% Check the sizes of some of the inputs
if size(d_grid)~=[N_d, 1]
    disp('ERROR: d_grid is not the correct shape (should be  of size N_d-by-1)')
    dbstack
    return
elseif size(a_grid)~=[N_a, 1]
    disp('ERROR: a_grid is not the correct shape (should be  of size N_a-by-1)')
    dbstack
    return
elseif size(z_grid)~=[N_z, 1]
    disp('ERROR: z_grid is not the correct shape (should be  of size N_z-by-1)')
    dbstack
    return
elseif size(pi_z)~=[N_z, N_z]
    disp('ERROR: pi is not of size N_z-by-N_z')
    dbstack
    return
elseif size(V0)~=[n_a,n_z]
    disp('ERROR: Starting choice for ValueFn is not of size [n_a,n_z]')
    dbstack
    return
end

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
end


%%
% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);

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
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, vfoptions.parallel);
    elseif vfoptions.returnmatrix==1
        ReturnMatrix=ReturnFn;
    elseif vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid,ReturnFnParamsVec);
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
            [VKron,Policy]=ValueFnIter_Case1_LowMem_NoD_Par1_raw(V0Kron, n_a, n_z, a_grid, z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
        elseif vfoptions.parallel==2 % On GPU
            [VKron,Policy]=ValueFnIter_Case1_LowMem_NoD_Par2_raw(V0Kron, n_a, n_z, a_grid, z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
        end
    else
        if vfoptions.parallel==0
            [VKron, Policy]=ValueFnIter_Case1_LowMem_raw(V0Kron, n_d,n_a,n_z, d_grid,a_grid,z_grid, pi_z, DiscountFactorParamsVec, ReturnFn,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
        elseif vfoptions.parallel==1
            [VKron, Policy]=ValueFnIter_Case1_LowMem_Par1_raw(V0Kron, n_d,n_a,n_z, d_grid,a_grid,z_grid,pi_z, DiscountFactorParamsVec, ReturnFn,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
        elseif vfoptions.parallel==2 % On GPU
            [VKron, Policy]=ValueFnIter_Case1_LowMem_Par2_raw(V0Kron, n_d,n_a,n_z, d_grid, a_grid, z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
        end
    end
    
% elseif vfoptions.lowmemory==2
%     
%     V0Kron=reshape(V0,[N_a,N_z]);
%     
%     if vfoptions.verbose==1
%         disp('Starting Value Function')
%         tic;
%     end
%     
%     if n_d(1)==0
%         if vfoptions.parallel==2
%             [VKron,Policy]=ValueFnIter_Case1_LowMem2_NoD_Par2_raw(V0Kron, n_a, n_z, a_grid, z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamNames, ReturnFnParams, vfoptions.howards, vfoptions.tolerance);
%         end
%     else
%         if vfoptions.parallel==2 % On GPU
%             [VKron, Policy]=ValueFnIter_Case1_LowMem2_Par2_raw(V0Kron, n_d,n_a,n_z, d_grid, a_grid, z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamNames, ReturnFnParams,vfoptions.howards,vfoptions.tolerance);
%         end
%     end
end

if vfoptions.verbose==1
    time=toc;
    fprintf('Time to solve for Value Fn and Policy: %8.4f \n', time)
    disp('Transforming Value Fn and Optimal Policy matrices back out of Kronecker Form')
    tic;
end
%%
V=reshape(VKron,[n_a,n_z]);
Policy=UnKronPolicyIndexes_Case1(Policy, n_d, n_a, n_z,vfoptions);
if vfoptions.verbose==1
    time=toc;
    fprintf('Time to create UnKron Value Fn and Policy: %8.4f \n', time)
end

if vfoptions.polindorval==2
    Policy=PolicyInd2Val_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,vfoptions.parallel);
end
    

end