function [V, Policy]=ValueFnIter_Case1_QuasiHyperbolic(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, vfoptions,Parameters,ReturnFnParamNames)
% (last two entries of) DiscountFactorParamNames contains the names for the two parameters relating to
% Quasi-hyperbolic preferences.
% Let V_j be the standard (exponential discounting) solution to the value fn problem
% The 'Naive' quasi-hyperbolic solution takes current actions as if the
% future agent take actions as if having time-consistent (exponential discounting) preferences.
% V_naive_j: Vtilde= u_t+ beta0beta *E[V_{j+1}]
% The 'Sophisticated' quasi-hyperbolic solution takes into account the time-inconsistent behaviour of their future self.
% Let Vunderbar_j be the exponential discounting value fn of the
% time-inconsistent policy function (aka. the policy-greedy exponential discounting value function of the time-inconsistent policy function)
% V_sophisticated_j: Vhat=u_t+beta0beta*E[Vundberbar_{j+1}]
% See documentation for a fuller explanation of this.

V=nan; % The value function of the quasi-hyperbolic agent
Policy=nan; % The value function of the quasi-hyperbolic agent

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if ~isfield(vfoptions,'quasi_hyperbolic')
    vfoptions.quasi_hyperbolic='Naive'; % This is the default, alternative is 'Sophisticated'.
elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
    vfoptions.maxiter=1000; 
    % Sophisticated quasihyperbolic in infinite horizon seems to struggle to converge, so end after fixed number of grid points
    % My impression is that accurate convergence would require insane number of grid points.
    vfoptions.verbose=1;
    % I set verbose so you can see if it appears to have gotten as close to
    % convergence as it can before reaching maxiter (if currdist is cycling
    % through numbers of the same magnitude for a while before stopping)
elseif ~strcmp(vfoptions.quasi_hyperbolic,'Naive') && ~strcmp(vfoptions.quasi_hyperbolic,'Sophisticated') 
    % Check that one of the possible options have been used. If not then error.
    fprintf('ERROR: vfoptions.quasi_hyperbolic must be either Naive or Sophisticated (check spelling and capital letter) \n')
    dbstack
    return
end

%%

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
if length(DiscountFactorParamsVec)>3
    DiscountFactorParamsVec=[prod(DiscountFactorParamsVec(1:end-2));DiscountFactorParamsVec(end-1);DiscountFactorParamsVec(end)];
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
    if vfoptions.parallel==2 % On GPU (Only implemented for GPU)
        if strcmp(vfoptions.quasi_hyperbolic,'Naive') % For Naive, just solve the standard value function problem, and then just one step following that.
            if n_d(1)==0
                % First calculate the exponential discounting solution
                [V,~]=ValueFnIter_Case1_NoD_Par2_raw(V0, n_a, n_z, pi_z, prod(DiscountFactorParamsVec(1:end-1)), ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
                % Then the Naive quasi-hyperbolic from this
                [V,Policy]=ValueFnIter_Case1_QuasiHyperbolicNaive_NoD_Par2_raw(V, n_a, n_z, pi_z, DiscountFactorParamsVec(end), ReturnMatrix);
            else
                % First calculate the exponential discounting solution
                [V, ~]=ValueFnIter_Case1_Par2_raw(V0, n_d,n_a,n_z, pi_z, prod(DiscountFactorParamsVec(1:end-1)), ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
                % Then the Naive quasi-hyperbolic from this
                [V, Policy]=ValueFnIter_Case1_QuasiHyperbolicNaive_Par2_raw(V, n_d,n_a,n_z, pi_z, DiscountFactorParamsVec(end), ReturnMatrix);
            end
        elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated') % For Naive, just solve the standard value function problem, and then just one step following that.
            if n_d(1)==0
                [V,Policy]=ValueFnIter_Case1_QuasiHyperbolic_NoD_Par2_raw(V0, n_a, n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter);
            else
                [V, Policy]=ValueFnIter_Case1_QuasiHyperbolic_Par2_raw(V0, n_d,n_a,n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, vfoptions.maxiter);
            end
        end
    elseif vfoptions.parallel==0 || vfoptions.parallel==1
        disp('ERROR: Quasi-Hyperbolic currently only implemented for Parallel=2: email robertdkirkby@gmail.com')
        dbstack
        return
    end
    
elseif vfoptions.lowmemory==1    
    
    if vfoptions.verbose==1
        disp('Starting Value Function')
        tic;
    end

    if vfoptions.parallel==2 % On GPU
        if strcmp(vfoptions.quasi_hyperbolic,'Naive') % For Naive, just solve the standard value function problem, and then just one step following that.
            if n_d(1)==0
                % First calculate the exponential discounting solution
                [V,~]=ValueFnIter_Case1_LowMem_NoD_Par2_raw(V0, n_a, n_z, a_grid, z_grid, pi_z, prod(DiscountFactorParamsVec(1:end-1)), ReturnFn, ReturnFnParamsVec, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
                % Then the Naive quasi-hyperbolic from this
                [V,Policy]=ValueFnIter_Case1_QuasiHyperbolicNaive_LowMem_NoD_Par2_raw(V, n_a, n_z, pi_z, DiscountFactorParamsVec(end), ReturnFn, ReturnFnParamsVec);
            else
                % First calculate the exponential discounting solution
                [V, ~]=ValueFnIter_Case1_LowMem_Par2_raw(V0, n_d,n_a,n_z, d_grid, a_grid, z_grid, pi_z, prod(DiscountFactorParamsVec(1:end-1)), ReturnFn, ReturnFnParamsVec,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
                % Then the Naive quasi-hyperbolic from this
                [V, Policy]=ValueFnIter_Case1_QuasiHyperbolicNaive_LowMem_Par2_raw(V, n_d, n_a, n_z, pi_z, DiscountFactorParamsVec(end), ReturnFn, ReturnFnParamsVec);
            end
        elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated') % For Naive, just solve the standard value function problem, and then just one step following that.
            if n_d(1)==0
                [V,Policy]=ValueFnIter_Case1_QuasiHyperbolic_LowMem_NoD_Par2_raw(V0, n_a, n_z, a_grid, z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
            else
                [V, Policy]=ValueFnIter_Case1_QuasiHyperbolic_LowMem_Par2_raw(V0, n_d,n_a,n_z, d_grid, a_grid, z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
            end
        end
    elseif vfoptions.parallel==0 || vfoptions.parallel==1
        disp('ERROR: Quasi-Hyperbolic currently only implemented for Parallel=2: email robertdkirkby@gmail.com')
        dbstack
    return

    end
    
end

if vfoptions.verbose==1
    time=toc;
    fprintf('Time to solve for Value Fn and Policy: %8.4f \n', time)
    disp('Transforming Value Fn and Optimal Policy matrices back out of Kronecker Form')
    tic;
end

%%
V=reshape(V,[n_a,n_z]);
Policy=UnKronPolicyIndexes_Case1(Policy, n_d, n_a, n_z,vfoptions);
if vfoptions.verbose==1
    time=toc;
    fprintf('Time to create UnKron Value Fn and Policy: %8.4f \n', time)
end

if vfoptions.polindorval==2
    Policy=PolicyInd2Val_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,vfoptions.parallel);
end
    

end