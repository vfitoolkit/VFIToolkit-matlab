function [V, Policy]=ValueFnIter_Case1(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, beta, ReturnFn, vfoptions,ReturnFnParams)

%% Check which vfoptions have been used, set all others to defaults 
if nargin<11
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
    vfoptions.tolerance=10^(-9);
    vfoptions.howards=80;
    vfoptions.maxhowards=500;
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



% %Check the sizes of some of the inputs
% if length(n_z)==1 && n_z(1)==1
%     if size(pi_z)~=[N_z, N_z]
%         disp('Error: pi is not of size N_z-by-N_z')
%     elseif size(Fmatrix)~=[n_d, n_a, n_a]
%         disp('Error: Fmatrix is not of size [n_d, n_a, n_a, n_z]')
%     elseif size(V0)~=[n_a]
%         disp('Error: Starting choice for ValueFn is not of size [n_a,n_z]')
%     end
% else
%     if size(pi_z)~=[N_z, N_z]
%         disp('Error: pi is not of size N_z-by-N_z')
%     elseif size(Fmatrix)~=[n_d, n_a, n_a, n_z]
%         disp('Error: Fmatrix is not of size [n_d, n_a, n_a, n_z]')
%     elseif size(V0)~=[n_a,n_z]
%         disp('Error: Starting choice for ValueFn is not of size [n_a,n_z]')
%     end
% end

%%
% % If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
% if vfoptions.parallel==2 
%    V0=gpuArray(V0);
%    pi_z=gpuArray(pi_z);
% end

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
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid,ReturnFnParams);
%     elseif vfoptions.returnmatrix==3 %An attempt to use spmd to parallelize on CPU
%         spmd
%            ReturnMatrix = codistributed(ReturnFn);
%         end
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
            [VKron,Policy]=ValueFnIter_Case1_NoD_raw(V0Kron, N_a, N_z, pi_z, beta, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
        elseif vfoptions.parallel==1 % On Parallel CPU
            [VKron,Policy]=ValueFnIter_Case1_NoD_Par1_raw(V0Kron, N_a, N_z, pi_z, beta, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
        elseif vfoptions.parallel==2 % On GPU
            [VKron,Policy]=ValueFnIter_Case1_NoD_Par2_raw(V0Kron, n_a, n_z, pi_z, beta, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance); %  a_grid, z_grid,
        end
    else
        if vfoptions.parallel==0 % On CPU
            [VKron, Policy]=ValueFnIter_Case1_raw(V0Kron, N_d,N_a,N_z, pi_z, beta, ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
        elseif vfoptions.parallel==1 % On Parallel CPU
            [VKron, Policy]=ValueFnIter_Case1_Par1_raw(V0Kron, N_d,N_a,N_z, pi_z, beta, ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
        elseif vfoptions.parallel==2 % On GPU
            [VKron, Policy]=ValueFnIter_Case1_Par2_raw(V0Kron, n_d,n_a,n_z, pi_z, beta, ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
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
            [VKron,Policy]=ValueFnIter_Case1_LowMem_NoD_raw(V0Kron, n_a, n_z, a_grid, z_grid, pi_z, beta, ReturnFn, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
        elseif vfoptions.parallel==1
            [VKron,Policy]=ValueFnIter_Case1_LowMem_NoD_Par1_raw(V0Kron, n_a, n_z, a_grid, z_grid, pi_z, beta, ReturnFn, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
        elseif vfoptions.parallel==2 % On GPU
            [VKron,Policy]=ValueFnIter_Case1_LowMem_NoD_Par2_raw(V0Kron, n_a, n_z, a_grid, z_grid, pi_z, beta, ReturnFn, ReturnFnParams, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
        end
    else
        if vfoptions.parallel==0
            [VKron, Policy]=ValueFnIter_Case1_LowMem_raw(V0Kron, n_d,n_a,n_z, d_grid,a_grid,z_grid, pi_z, beta, ReturnFn,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
        elseif vfoptions.parallel==1
            [VKron, Policy]=ValueFnIter_Case1_LowMem_Par1_raw(V0Kron, n_d,n_a,n_z, d_grid,a_grid,z_grid,pi_z, beta, ReturnFn,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
        elseif vfoptions.parallel==2 % On GPU
            [VKron, Policy]=ValueFnIter_Case1_LowMem_Par2_raw(V0Kron, n_d,n_a,n_z, d_grid, a_grid, z_grid, pi_z, beta, ReturnFn, ReturnFnParams,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
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
%             [VKron,Policy]=ValueFnIter_Case1_LowMem2_NoD_Par2_raw(V0Kron, n_a, n_z, a_grid, z_grid, pi_z, beta, ReturnFn, ReturnFnParamNames, ReturnFnParams, vfoptions.howards, vfoptions.tolerance);
%         end
%     else
%         if vfoptions.parallel==2 % On GPU
%             [VKron, Policy]=ValueFnIter_Case1_LowMem2_Par2_raw(V0Kron, n_d,n_a,n_z, d_grid, a_grid, z_grid, pi_z, beta, ReturnFn, ReturnFnParamNames, ReturnFnParams,vfoptions.howards,vfoptions.tolerance);
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