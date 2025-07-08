function [V, Policy]=ValueFnIter_Case1_EpsteinZin(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, vfoptions,Parameters,ReturnFnParamNames)
% Epstein-Zin preferences
% Formulation depends on whether using utility-units or consumption-units

%% Some Epstein-Zin specific options need to be set if they are not already declared
if ~isfield(vfoptions,'EZriskaversion')
    error('When using Epstein-Zin preferences you must declare vfoptions.EZriskaversion (coefficient controlling risk aversion)')
end
if ~isfield(vfoptions,'EZutils')
    vfoptions.EZutils=1; % Use EZ preferences with general utility function (0 gives traditional EZ with exogenous labor, 2 gives traditional EZ with endogenous labor)
end
if vfoptions.EZutils==1
    % Have to do EZ preferences differently depending on whether the utility function is >=0 or <=0.
    % vfoptions.EZpositiveutility=1 if utility is positive; Note, in this case when EZriskaversion is higher, the risk aversion is larger (EZriskaversion>0 is risk averse)
    % vfoptions.EZpositiveutility=0 if utility is negative; Note, in this case when EZriskaversion is lower, the risk aversion is larger  (EZriskaversion<0 is risk averse)
    if ~isfield(vfoptions,'EZpositiveutility')
        warning('Using Epstein-Zin preferences it is assumed the utility/return function is negative valued, if not you need to set vfoptions.EZpositiveutility=1')
        vfoptions.EZpositiveutility=0; % User did not specify. Guess that it is negative as most common things (like CES) are negative valued.
    end
else
    % Traditional EZ preferences requires you to specify the EIS parameter
    if ~isfield(vfoptions,'EZeis')
        error('When using Epstein-Zin preferences you must declare vfoptions.EZeis (elasticity of intertemporal substitution)')
    end
end
if ~isfield(vfoptions,'EZoneminusbeta')
    vfoptions.EZoneminusbeta=0; % Put a (1-beta)* term on the this period return (sometimes people want this for traditional EZ, I don't actually know why)
end
% Set up sj
if isfield(vfoptions,'survivalprobability') || isfield(vfoptions,'WarmGlowBequestsFn')
    error('Epstein-Zin preferences in infinite-horizon do not support vfoptions.survivalprobability nor vfoptions.WarmGlowBequestsFn')
end

%% Based on the settings, define a bunch of variables that are used to implement the EZ preferences
% Note that the discount factor and survival probabilities can depend on jj (age/period)
% But the 'relative risk aversion' and 'elasticity of intertemporal substititution' cannot depend on jj
crisk=Parameters.(vfoptions.EZriskaversion);
if vfoptions.EZutils==0
    ceis=Parameters.(vfoptions.EZeis);
    % Traditional EZ in consumption units
    ezc1=1; % used if vfoptions.EZoneminusbeta=1
    ezc2=1-1/ceis; % ezc3 is same in both cases
    ezc3=1;
    ezc4=1;
    ezc5=1-crisk;
    ezc6=(1-1/ceis)/(1-crisk);
    ezc7=1/(1-1/ceis);
elseif vfoptions.EZutils==1
    % EZ in utility-units
    ezc1=1; % used if vfoptions.EZoneminusbeta=1
    ezc2=1; % ezc3 is same in both cases
    % If the utility is negative you need to multiply it by -1 in two places
    if vfoptions.EZpositiveutility==1
        ezc3=1; % will be -1 if vfoptions.EZpositiveutility=0
        ezc4=1; % will be -1 if vfoptions.EZpositiveutility=0
    elseif vfoptions.EZpositiveutility==0
        ezc3=-1;
        ezc4=-1;
    end
    % If the utility is negative use 1+crisk instead of 1-crisk. This way
    % the interpretation of crisk is identical in both cases
    if vfoptions.EZpositiveutility==1
        ezc5=1-crisk;
        ezc6=1/(1-crisk);
    elseif vfoptions.EZpositiveutility==0
        ezc5=1+crisk; % essentially, just use crisk as being - what it would otherwise be
        ezc6=1/(1+crisk);
    end
    ezc7=1;
end

if vfoptions.EZoneminusbeta==1
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames); 
    ezc1=1-prod(DiscountFactorParamsVec); % (This will be changed later if it depends on age)
end

if vfoptions.EZutils==0
    if crisk<1
        error('Cannot use EZriskaversion parameter less than one (must be risk averse) with Epstein-Zin preferences')
    end
    if ceis<=0
        error('Cannot use EZeis parameter less than zero with Epstein-Zin preferences')
    end
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
        if n_d(1)==0
            [VKron,Policy]=ValueFnIter_Case1_EpsteinZin_NoD_Par2_raw(V0, n_a, n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7);
        else
            [VKron, Policy]=ValueFnIter_Case1_EpsteinZin_Par2_raw(V0, n_d,n_a,n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7);
        end
    end
    
elseif vfoptions.lowmemory==1    
    
    if vfoptions.verbose==1
        disp('Starting Value Function')
        tic;
    end

    if vfoptions.parallel==2 % On GPU
        if n_d(1)==0
            [VKron,Policy]=ValueFnIter_Case1_EpsteinZin_LowMem_NoD_Par2_raw(V0, n_a, n_z, a_grid, z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7);
        else
            [VKron, Policy]=ValueFnIter_Case1_EpsteinZin_LowMem_Par2_raw(V0, n_d,n_a,n_z, d_grid, a_grid, z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7);
        end
    end
    
elseif vfoptions.lowmemory==2  
    error('lowmemory=2 is not an option for Epstein-Zin preferences in infinite horizon')
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