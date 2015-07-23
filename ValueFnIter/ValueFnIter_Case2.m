function [V, Policy]=ValueFnIter_Case2(V0, n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z,beta, ReturnFn, Phi_aprime, Case2_Type, vfoptions,ReturnFnParams)

%% Check which vfoptions have been used, set all others to defaults 
if nargin<13
    %If vfoptions is not given, just use all the defaults
    vfoptions.lowmemory=0;
    vfoptions.polindorval=1;
    vfoptions.howards=80;
    vfoptions.maxhowards=500;
    vfoptions.parallel=0;
    vfoptions.verbose=0;
    vfoptions.returnmatrix=0;
    vfoptions.tolerance=10^(-9);
else
    %Check vfoptions for missing fields, if there are some fill them with
    %the defaults
    eval('fieldexists=1;vfoptions.lowmemory;','fieldexists=0;')
    if fieldexists==0
        vfoptions.lowmemory=0;
    end
    eval('fieldexists=1;vfoptions.polindorval;','fieldexists=0;')
    if fieldexists==0
        vfoptions.polindorval=1;
    end
    eval('fieldexists=1;vfoptions.howards;','fieldexists=0;')
    if fieldexists==0
        vfoptions.howards=80;
    end
    eval('fieldexists=1;vfoptions.maxhowards;','fieldexists=0;')
    if fieldexists==0
        vfoptions.maxhowards=500;
    end
    eval('fieldexists=1;vfoptions.parallel;','fieldexists=0;')
    if fieldexists==0
        vfoptions.parallel=0;
    end
    eval('fieldexists=1;vfoptions.verbose;','fieldexists=0;')
    if fieldexists==0
        vfoptions.verbose=0;
    end
    eval('fieldexists=1;vfoptions.tolerance;','fieldexists=0;')
    if fieldexists==0
        vfoptions.tolerance=10^(-9);
    end
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

vfoptions


%%
if vfoptions.lowmemory==0
    
    %% CreateReturnFnMatrix_Case1_Disc creates a matrix of dimension d-by-a-by-z.
    % Since the return function is independent of time creating it once and
    % then using it every iteration is good for speed, but it does use a
    % lot of memory.
    if vfoptions.returnmatrix==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, vfoptions.parallel);
    elseif vfoptions.returnmatrix==1
        ReturnMatrix=ReturnFn;
    elseif vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid,ReturnFnParams);
    end
    
    %     %% Create Phi_aprimeKron
%     if vfoptions.phiaprime==2
%         tic;
%         Phi_aprime=CreatePhiaprimeMatrix_Case2_Disc_Parallel2(PhiFn, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeFnParams);
%         time0=toc  
%     else
%         Phi_aprime=PhiFn;
%     end
        
    %% The Value Function Iteration
    V0Kron=reshape(V0,[N_a,N_z]);
    
    if vfoptions.parallel==0 % On CPU
        [VKron, Policy]=ValueFnIter_Case2_raw(V0Kron, n_d,n_a,n_z, pi_z, beta, ReturnMatrix,Phi_aprime,Case2_Type,vfoptions.howards,vfoptions.maxhowards,vfoptions.verbose,vfoptions.tolerance); 
    elseif vfoptions.parallel==1 % On Parallel CPU
        [VKron, Policy]=ValueFnIter_Case2_Par1_raw(V0Kron, n_d,n_a,n_z, pi_z, beta, ReturnMatrix,Phi_aprime,Case2_Type,vfoptions.howards,vfoptions.maxhowards,vfoptions.verbose,vfoptions.tolerance);
    elseif vfoptions.parallel==2 % On GPU
        [VKron, Policy]=ValueFnIter_Case2_Par2_raw(V0Kron, n_d,n_a,n_z, pi_z, beta, ReturnMatrix,Phi_aprime,Case2_Type,vfoptions.howards,vfoptions.maxhowards,vfoptions.verbose,vfoptions.tolerance);
    end
    
    %% Sort out Policy
    if vfoptions.polindorval==2
        Policy=PolicyInd2Val_Case2(Policy,n_d,n_a,n_z,d_grid,vfoptions.parallel);
    end
end

V=reshape(VKron,[n_a,n_z]);
if vfoptions.polindorval==1
    Policy=UnKronPolicyIndexes_Case2(Policy, n_d, n_a, n_z,vfoptions);
end


end