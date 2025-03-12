function [V, Policy]=ValueFnIter_Case1_FHorz_EpsteinZin_RiskyAsset(n_d,n_a1,n_a2,n_z,n_u,N_j,d_grid,a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, aprimeFn, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)


N_a1=prod(n_a1);
N_z=prod(n_z);

%% Get aprimeFnParamNames
l_d=length(n_d); % because it is a risky asset there must be some decision variables
if isfield(vfoptions,'refine_d')
    l_d=l_d-vfoptions.refine_d(1);
    if length(vfoptions.refine_d)==4 % only relevant if using semiz
        l_d=l_d-vfoptions.refine_d(4);
    end
end
l_u=length(n_u);
temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d+l_u)
    aprimeFnParamNames={temp{l_d+l_u+1:end}}; % the first inputs will always be (d,u)
else
    aprimeFnParamNames={};
end

%% 
% Make sure all the relevant inputs are GPU arrays (not standard arrays)
pi_u=gpuArray(pi_u);
u_grid=gpuArray(u_grid);
% Check pi_u and u_grid are the right size
if all(size(pi_u)==[prod(n_u),1])
    % good
elseif all(size(pi_u)==[1,prod(n_u)])
    error('pi_u should be a column vector (it is a row vector, you need to transpose it')
else
    error('pi_u is the wrong size (it should be a column vector of size prod(n_u)-by-1)')
end
if all(size(u_grid)==[prod(n_u),1])
    % good
elseif all(size(u_grid)==[1,prod(n_u)])
    error('u_grid should be a column vector (it is a row vector, you need to transpose it')
else
    error('u_grid is the wrong size (it should be a column vector of size prod(n_u)-by-1)')
end

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
    vfoptions.EZoneminusbeta=0; % default essentially does nothing
    %=1 Put a (1-beta)* term on the this period return
    %=2 Put a (1-sj*beta)* term on the this period return
end
% Set up sj
if isfield(vfoptions,'survivalprobability')
    sj=Parameters.(vfoptions.survivalprobability);
    if length(sj)~=N_j
        error('Survival probabilities must be of the same length as N_j')
    end
elseif isfield(vfoptions,'WarmGlowBequestsFn')
    % If you have warm-glow but do not specify survival probabilites it is assumed you only get it at end of final period
    sj=ones(N_j,1); % conditional survival probabilities
    sj(end)=0;
    warning('You have used vfoptons.WarmGlowBequestsFn, but have not set vfoptions.survivalprobability, it is assumed you only want to have the warm-glow at the end of the final period')
else
    sj=ones(N_j,1); % conditional survival probabilities
end
% Declare warmglow indicator
if isfield(vfoptions,'WarmGlowBequestsFn')
    warmglow=1;
    temp=getAnonymousFnInputNames(vfoptions.WarmGlowBequestsFn);
    vfoptions.WarmGlowBequestsFnParamsNames={temp{2:end}};
else
    warmglow=0;
end

%% Based on the settings, define a bunch of variables that are used to implement the EZ preferences
% Note that the discount factor and survival probabilities can depend on jj (age/period)
% But the 'relative risk aversion' and 'elasticity of intertemporal substititution' cannot depend on jj
crisk=Parameters.(vfoptions.EZriskaversion);
if vfoptions.EZutils==0
    ceis=Parameters.(vfoptions.EZeis);
    % Traditional EZ in consumption units
    ezc1=1; % used if vfoptions.EZoneminusbeta=1
    ezc2=1-1./ceis; % ezc3 is same in both cases
    ezc3=1;
    ezc4=1;
    ezc5=1-crisk;
    ezc6=(1-1./ceis)./(1-crisk);
    ezc7=1./(1-1./ceis);
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
        ezc6=1./(1-crisk);
    elseif vfoptions.EZpositiveutility==0
        ezc5=1+crisk; % essentially, just use crisk as being - what it would otherwise be
        ezc6=1./(1+crisk);
    end
    ezc7=1;
end
% Can do a double Epstein-Zin, this involves changing a fair few of these
% (inner EZ is about risk, outer EZ is about mortality-risk)
if isfield(vfoptions,'EZmortalityriskaversion')
    mrisk=Parameters.(vfoptions.EZmortalityriskaversion);
    if vfoptions.EZutils==0
        ezc6=(1-1./ceis)./(1-mrisk);
        ezc8=(1-mrisk)./(1-crisk);
    elseif vfoptions.EZutils==1
        if vfoptions.EZpositiveutility==1
            ezc6=1./(1-mrisk);
            ezc8=(1-mrisk)./(1-crisk);
        elseif vfoptions.EZpositiveutility==0
            ezc6=1./(1+mrisk);
            ezc8=(1+mrisk)./(1+crisk);
        end
    end
    % Note: the baseline codes apply ezc5 to the warm-glow, so we also want
    % to use ezc8 on the warm-glow to get rid of this (even though in Case1
    % raising to ezc5 and then getting rid of it as part of ezc8 is just a
    % waste, I don't feel like recoding the whole thing)
else
    % This wont do anything
    ezc8=1;
end
% When doing the refine, if ezc7 is negative, need to take min(X) instead of
% max(X) as part of Refine. I do this by taking ezc9*max(ezc9*X) and having
% ezc9=1 normally but ezc=9 when ezc7 is negative.
ezc9=1;
if ezc7(1)<0
    ezc9=-1; % Not allowed to vary by age
end
if ~isscalar(ezc7)
    temp1=any(ezc7<0);
    temp2=any(ezc7>0);
    if temp1 && temp2
        error('Epstein-Zin preferences: you have set the elasticity-of-intertemporal-substution parameter (vfoptions.EZeis) to depend on age. You must have it either <1 or >1 for all ages (cannot be <1 at some ages and >1 at other ages, which is what you currently have).')
    end
end

% setup to permit age-dependence of these (and make them column vectors if they are not already)
% Note: the only ones that need to permit this are ezc2, ezc5, ezc6, ezc7, ezc8
if size(ezc2,1)==1
    ezc2=ezc2';
end
ezc2=ezc2.*ones(N_j,1); % this will work whether it starts N_j-by-1 or 1-by-1
if size(ezc5,1)==1
    ezc5=ezc5';
end
ezc5=ezc5.*ones(N_j,1); % this will work whether it starts N_j-by-1 or 1-by-1
if size(ezc6,1)==1
    ezc6=ezc6';
end
ezc6=ezc6.*ones(N_j,1); % this will work whether it starts N_j-by-1 or 1-by-1
if size(ezc7,1)==1
    ezc7=ezc7';
end
ezc7=ezc7.*ones(N_j,1); % this will work whether it starts N_j-by-1 or 1-by-1
if size(ezc8,1)==1
    ezc8=ezc8';
end
ezc8=ezc8.*ones(N_j,1); % this will work whether it starts N_j-by-1 or 1-by-1

if vfoptions.EZoneminusbeta==1
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j); 
    ezc1=1-prod(DiscountFactorParamsVec); % (This will be changed later if it depends on age)
elseif vfoptions.EZoneminusbeta==2
    % Some formulations using bequests multiply the period utility function by (1-sj*beta) 
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    ezc1=1-sj(N_j)*prod(DiscountFactorParamsVec);
end

if vfoptions.EZutils==0
    if crisk<1
        error('Cannot use EZriskaversion parameter less than one (must be risk averse) with Epstein-Zin preferences')
    end
    if ceis<=0
        error('Cannot use EZeis parameter less than zero with Epstein-Zin preferences')
    end
    if ceis==1
        error('Cannot use EZeis parameter equal to one with Epstein-Zin preferences (look at formula, it would mean having to raise to the power of zero; you can always put 0.99 or 1.01)')
    end
end


%% Just do the standard case
if length(n_a2)>1
    error('Have not yet implemented riskyasset for more than one riskyasset')
end
if isfield(vfoptions,'n_e')
    N_e=prod(vfoptions.n_e);
else
    N_e=0;
end
if isfield(vfoptions,'refine_d')
    if vfoptions.refine_d(1)==0
        N_d1=0;
    else
        N_d1=prod(n_d(1:vfoptions.refine_d(1)));
    end
    if vfoptions.refine_d(1)==0 && vfoptions.refine_d(2)==0
        % when only d3, refine does nothing anyway, so just turn it off
        vfoptions=rmfield(vfoptions, 'refine_d');
    end
end

if ~isfield(vfoptions,'refine_d')
    warning('When using vfoptions.riskyasset you should also set vfoptions.refine_d (you do not have refine_d, and this is making codes way slower than they should be)')
    if N_e>0
        if N_a1==0
            if N_z==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_noa1_noz_e_raw(n_d, n_a2, vfoptions.n_e, n_u, N_j, d_grid, a2_grid, vfoptions.e_gridvals_J, u_grid, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
            else
                [VKron,PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_noa1_e_raw(n_d, n_a2, n_z, vfoptions.n_e, n_u, N_j, d_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
            end
        else
            if N_z==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_noz_e_raw(n_d, n_a1,n_a2, vfoptions.n_e, n_u, N_j, d_grid, a1_grid,a2_grid, vfoptions.e_gridvals_J, u_grid, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
            else
                [VKron,PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_e_raw(n_d, n_a1,n_a2, n_z, vfoptions.n_e, n_u, N_j, d_grid, a1_grid,a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
            end
        end
    else % N_e==0
        if N_a1==0
            if N_z==0
                error('Cannot use Epstein-Zin preferences without any shocks (what is the point?); you have n_z=0 and no e variables')
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_noa1_raw(n_d,n_a2,n_z,n_u, N_j, d_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
            end
        else
            if N_z==0
                error('Cannot use Epstein-Zin preferences without any shocks (what is the point?); you have n_z=0 and no e variables')
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_raw(n_d,n_a1,n_a2,n_z,n_u, N_j, d_grid, a1_grid,a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
            end
        end
    end
else
    if sum(vfoptions.refine_d)~=length(n_d)
        error('vfoptions.refine_d seems to be set up wrong, it is inconsistent with n_d')
    end
    if any(vfoptions.refine_d(2:3)==0)
        error('vfoptions.refine_d cannot contain zeros for d2 or d3 (you can do no d1, but you cannot do no d2 nor no d3)')
    end
    n_d1=n_d(1:vfoptions.refine_d(1));
    n_d2=n_d(vfoptions.refine_d(1)+1:vfoptions.refine_d(1)+vfoptions.refine_d(2));
    n_d3=n_d(vfoptions.refine_d(1)+vfoptions.refine_d(2)+1:end);
    d1_grid=d_grid(1:sum(n_d1));
    d2_grid=d_grid(sum(n_d1)+1:sum(n_d1)+sum(n_d2));
    d3_grid=d_grid(sum(n_d1)+sum(n_d2)+1:end);
    if N_e>0
        if N_a1==0
            if N_z==0
                if N_d1==0
                    error('I have not yet implemented Epstein-Zin preferences for riskyasset with e but without z shocks (contact me if you need this)')
                else
                    error('I have not yet implemented Epstein-Zin preferences for riskyasset with e but without z shocks (contact me if you need this)')
                end
            else
                if N_d1==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_Refine_nod1_noa1_e_raw(n_d2,n_d3, n_a2, n_z, vfoptions.n_e, n_u, N_j, d2_grid, d3_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_Refine_noa1_e_raw(n_d1,n_d2,n_d3, n_a2, n_z, vfoptions.n_e, n_u, N_j, d1_grid, d2_grid, d3_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);
                end
            end
        else % N_a1>0
            if N_z==0
                if N_d1==0
                    error('I have not yet implemented Epstein-Zin preferences for riskyasset with e but without z shocks (contact me if you need this)')
                else
                    error('I have not yet implemented Epstein-Zin preferences for riskyasset with e but without z shocks (contact me if you need this)')
                end
            else
                if N_d1==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_Refine_nod1_e_raw(n_d2,n_d3, n_a1,n_a2, n_z, vfoptions.n_e, n_u, N_j, d2_grid, d3_grid, a1_grid,a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_Refine_e_raw(n_d1,n_d2,n_d3, n_a1,n_a2, n_z, vfoptions.n_e, n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid,a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);
                end
            end
        end
    else % N_e==0
        if N_a1==0
            if N_z==0
                error('Cannot use Epstein-Zin preferences without any shocks (what is the point?); you have n_z=0 and no e variables')
            else
                if N_d1==0
                    [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_Refine_nod1_noa1_raw(n_d2,n_d3,n_a2,n_z,n_u, N_j, d2_grid, d3_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);       
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_Refine_noa1_raw(n_d1,n_d2,n_d3,n_a2,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);
                end
            end
        else % N_a1>0
            if N_z==0
                error('Cannot use Epstein-Zin preferences without any shocks (what is the point?); you have n_z=0 and no e variables')
            else
                if N_d1==0
                    [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_Refine_nod1_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_u, N_j, d2_grid, d3_grid, a1_grid,a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_Refine_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid,a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);
                end
            end
        end
    end
end



%%
if N_a1==0
    n_a=n_a2;
else
    n_a=[n_a1,n_a2];
    n_d=[n_d,n_a1]; % just to UnKron
end

if vfoptions.outputkron==0
    %Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
    if isfield(vfoptions,'n_e')
        if N_z==0
            V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz(PolicyKron, n_d, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
        else
            V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz_e(PolicyKron, n_d, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
        end
    else
        V=reshape(VKron,[n_a,n_z,N_j]);
        Policy=UnKronPolicyIndexes_Case2_FHorz(PolicyKron, n_d, n_a, n_z, N_j, vfoptions);
    end
else
    V=VKron;
    Policy=PolicyKron;
end

% Sometimes numerical rounding errors (of the order of 10^(-16) can mean
% that Policy is not integer valued. The following corrects this by converting to int64 and then
% makes the output back into double as Matlab otherwise cannot use it in
% any arithmetical expressions.
if vfoptions.policy_forceintegertype==1
    fprintf('USING vfoptions to force integer... \n')
    % First, give some output on the size of any changes in Policy as a
    % result of turning the values into integers
    temp=max(max(max(abs(round(Policy)-Policy))));
    while ndims(temp)>1
        temp=max(temp);
    end
    fprintf('  CHECK: Maximum change when rounding values of Policy is %8.6f (if these are not of numerical rounding error size then something is going wrong) \n', temp)
    % Do the actual rounding to integers
    Policy=round(Policy);
    % Somewhat unrelated, but also do a double-check that Policy is now all positive integers
    temp=min(min(min(Policy)));
    while ndims(temp)>1
        temp=min(temp);
    end
    fprintf('  CHECK: Minimum value of Policy is %8.6f (if this is <=0 then something is wrong) \n', temp)
%     Policy=uint64(Policy);
%     Policy=double(Policy);
elseif vfoptions.policy_forceintegertype==2
    % Do the actual rounding to integers
    Policy=round(Policy);
end

end