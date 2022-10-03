function [V, Policy, ExitPolicy]=ValueFnIter_Case1_EndogExit(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Solves infinite-horizon 'Case 1' value function problems with Endogenous Exit

% It is only intended that this is called indirectly by ValueFnIter_Case1()

if isfield(vfoptions,'endogenousexit')==0
    vfoptions.endogenousexit=0;
end
if isfield(vfoptions,'endofperiodexit')==0
    vfoptions.endofperiodexit=0; % This has not yet been implemented as an option that can be activated.
end
if isfield(vfoptions,'keeppolicyonexit')==0
    vfoptions.keeppolicyonexit=0;
end

V=nan; % Matlab was complaining that V was not assigned

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

l_a=length(n_a);
l_z=length(n_z);

% Make sure that the inputs specifically required for endogenous exit have been included.
if isfield(vfoptions,'ReturnToExitFn')==0
    fprintf('ERROR: vfoptions.endogenousexit=1 requires that you specify vfoptions.ReturnToExitFn \n');
    return
end

%%
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
% The 'return to exit function' parameters (in order)
temp=getAnonymousFnInputNames(vfoptions.ReturnToExitFn);
if length(temp)>(l_a+l_z)
    ReturnToExitFnParamNames={temp{l_a+l_z+1:end}}; % the first inputs will always be (a,z)
else
    ReturnToExitFnParamNames={};
end
ReturnToExitFnParamsVec=CreateVectorFromParams(Parameters, ReturnToExitFnParamNames);

%%
if vfoptions.lowmemory==0
    
    %% CreateReturnFnMatrix_Case1_Disc creates a matrix of dimension (d and aprime)-by-a-by-z.
    % Since the return function is independent of time creating it once and
    % then using it every iteration is good for speed, but it does use a
    % lot of memory.

    if vfoptions.verbose==1
        disp('Creating return fn matrix')
        tic;
        if vfoptions.returnmatrix==0
            fprintf('NOTE: When using CPU you can speed things up by giving return fn as a matrix; see vfoptions.returnmatrix=1 in VFI Toolkit documentation. \n')
        end
    end
    
    if vfoptions.returnmatrix==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, vfoptions.parallel, ReturnFnParamsVec);
        ReturnToExitMatrix=CreateReturnToExitFnMatrix_Case1_Disc(vfoptions.ReturnToExitFn, n_a, n_z, a_grid, z_grid, vfoptions.parallel, ReturnToExitFnParamsVec);
    elseif vfoptions.returnmatrix==1
        ReturnMatrix=ReturnFn;
        ReturnToExitMatrix=vfoptions.ReturnToExitFn; % It is simply assumed that you are doing this for both.
    elseif vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
        ReturnToExitMatrix=CreateReturnToExitFnMatrix_Case1_Disc_Par2(vfoptions.ReturnToExitFn, n_a, n_z, a_grid, z_grid, ReturnToExitFnParamsVec);
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
        if isfield(vfoptions,'SemiEndogShock')
            pi_z_semiendog=vfoptions.SemiEndogShock;
            if vfoptions.parallel==0     % On CPU
                [VKron,Policy,ExitPolicy]=ValueFnIter_Case1_EndogExit_SemiEndog_NoD_raw(V0Kron, N_a, N_z, pi_z_semiendog, DiscountFactorParamsVec, ReturnMatrix, ReturnToExitMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.keeppolicyonexit);
            elseif vfoptions.parallel==1 % On Parallel CPU
                [VKron,Policy,ExitPolicy]=ValueFnIter_Case1_EndogExit_SemiEndog_NoD_Par1_raw(V0Kron, N_a, N_z, pi_z_semiendog, DiscountFactorParamsVec, ReturnMatrix, ReturnToExitMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.keeppolicyonexit);
            elseif vfoptions.parallel==2 % On GPU
                pi_z_semiendog=gpuArray(pi_z_semiendog);
                [VKron,Policy,ExitPolicy]=ValueFnIter_Case1_EndogExit_SemiEndog_NoD_Par2_raw(V0Kron, n_a, n_z, pi_z_semiendog, DiscountFactorParamsVec, ReturnMatrix, ReturnToExitMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.keeppolicyonexit); %  a_grid, z_grid,
            end
        else
            if vfoptions.parallel==0     % On CPU
                [VKron,Policy,ExitPolicy]=ValueFnIter_Case1_EndogExit_NoD_raw(V0Kron, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, ReturnToExitMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.keeppolicyonexit);
            elseif vfoptions.parallel==1 % On Parallel CPU
                [VKron,Policy,ExitPolicy]=ValueFnIter_Case1_EndogExit_NoD_Par1_raw(V0Kron, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, ReturnToExitMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.keeppolicyonexit);
            elseif vfoptions.parallel==2 % On GPU
                [VKron,Policy,ExitPolicy]=ValueFnIter_Case1_EndogExit_NoD_Par2_raw(V0Kron, n_a, n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, ReturnToExitMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.keeppolicyonexit); %  a_grid, z_grid,
            end
        end
    else
        if vfoptions.parallel==0 % On CPU
            [VKron, Policy,ExitPolicy]=ValueFnIter_Case1_EndogExit_raw(V0Kron, N_d,N_a,N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, ReturnToExitMatrix, vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, vfoptions.keeppolicyonexit);
        elseif vfoptions.parallel==1 % On Parallel CPU
            [VKron, Policy,ExitPolicy]=ValueFnIter_Case1_EndogExit_Par1_raw(V0Kron, N_d,N_a,N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, ReturnToExitMatrix, vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, vfoptions.keeppolicyonexit);
        elseif vfoptions.parallel==2 % On GPU
            [VKron, Policy,ExitPolicy]=ValueFnIter_Case1_EndogExit_Par2_raw(V0Kron, n_d,n_a,n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, ReturnToExitMatrix, vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, vfoptions.keeppolicyonexit);
        end
    end
    
    
elseif vfoptions.lowmemory==1    
    fprintf('ERROR: endogenousexit does not yet allow for vfoptions.lowmemory=1, please contact robertdkirkby@gmail.com if this is something you want/need \n');
    dbstack
    return
elseif vfoptions.lowmemory==2
    fprintf('ERROR: endogenousexit does not yet allow for vfoptions.lowmemory=2, please contact robertdkirkby@gmail.com if this is something you want/need \n');
    dbstack
    return
end

if vfoptions.verbose==1
    time=toc;
    fprintf('Time to solve for Value Fn and Policy: %8.4f \n', time)
    disp('Transforming Value Fn and Optimal Policy matrices back out of Kronecker Form')
    tic;
end
%% Cleaning up the output
V=reshape(VKron,[n_a,n_z]);
ExitPolicy=reshape(ExitPolicy,[n_a,n_z]);
Policy=UnKronPolicyIndexes_Case1(Policy, n_d, n_a, n_z,vfoptions);
if vfoptions.verbose==1
    time=toc;
    fprintf('Time to create UnKron Value Fn and Policy: %8.4f \n', time)
end

if vfoptions.polindorval==2
    Policy=PolicyInd2Val_Case1(Policy,n_d,n_a,n_z,d_grid,a_grid,vfoptions.parallel);
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