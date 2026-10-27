function [V, Policy]=ValueFnIter_InfHorz_EpsteinZin(V0, n_d,n_a,n_z,d_gridvals,a_grid,z_gridvals, pi_z, DiscountFactorParamNames, ReturnFn, vfoptions,Parameters,ReturnFnParamNames)
% Epstein-Zin preferences
% Formulation depends on whether using utility-units or consumption-units
% See appendix to the 'Intro to Life-Cycle models' for an explanation (the InfHorz recursion is
% the FHorz one of ValueFnIter_FHorz_EpsteinZin with no ages, no survival probabilities and no
% warm-glow: V = [ezc1*F^ezc2 + ezc3*beta*(E[(ezc4*V')^ezc5])^ezc6]^ezc7, iterated to a fixed point)
%
% Paths: plain (lowmemory 0/1, with/without d) and, with vfoptions.gridinterplayer=1, the postGI
% grid interpolation layer (with/without d; d is refined out of the return function first, which
% is preference-independent). Feasibility conventions: EZ_feasibility_conventions_proposal.md at
% the toolkit root (a zero return is a value; -Inf stays -Inf under both sign conventions).

%% Interface checks: the shapes and options this dispatcher does not handle
% Everything it does handle is dispatched below. Anything else must be refused HERE, with a
% message that says so, because a flag that no raw reads is otherwise silently ignored: the
% solve runs, and returns an answer to a different problem than the one asked for.
if prod(n_z)==0
    error('Epstein-Zin preferences in infinite-horizon require an exogenous state z (n_z=0 is not implemented: every Epstein-Zin InfHorz raw loops over z, and with no z that loop never runs, so the solve would return V=0 and Policy=0 without an error). A single-point z with z_grid=1 and pi_z=1 reproduces a no-shock model (and the warning that Epstein-Zin preferences do not make much sense without any shocks applies to it).')
end
if ~(vfoptions.lowmemory==0 || vfoptions.lowmemory==1)
    error('Epstein-Zin preferences in infinite-horizon only implement vfoptions.lowmemory=0 or 1')
end
if isfield(vfoptions,'EZmortalityriskaversion')
    error('vfoptions.EZmortalityriskaversion (a separate risk aversion over mortality risk) has no meaning in infinite-horizon, where there are no survival probabilities; remove it (it was previously ignored without an error)')
end
if vfoptions.gridinterplayer==1
    % The Epstein-Zin grid interpolation layer is postGI only, with dense two-point Howards.
    if vfoptions.preGI==1
        error('Epstein-Zin preferences in infinite-horizon implement the grid interpolation layer as postGI only (vfoptions.preGI=1 is not implemented)')
    end
    if ~isscalar(n_a)
        error('Epstein-Zin preferences in infinite-horizon implement the grid interpolation layer for a single endogenous state only (vfoptions.gridinterplayer=1 with two endogenous states is not implemented; without the grid interpolation layer two endogenous states work)')
    end
    if vfoptions.howardsgreedy>0
        error('Epstein-Zin preferences in infinite-horizon do not implement greedy Howards (vfoptions.howardsgreedy>0): the Epstein-Zin Howards step is not linear in V, so the direct solve the greedy raws use does not apply. Use vfoptions.howardsgreedy=0.')
    end
    if isfield(vfoptions,'postGIrepeat') && vfoptions.postGIrepeat>0
        error('Epstein-Zin preferences in infinite-horizon do not implement vfoptions.postGIrepeat>0 with the grid interpolation layer')
    end
    % vfoptions.howardssparse is NOT read by the Epstein-Zin grid-interpolation raws: their Howards
    % step is the dense two-point form (the sparse form is a matrix product with V, and the
    % Epstein-Zin step is not linear in V). ValueFnIter_InfHorz sets howardssparse=1 by default
    % under the grid interpolation layer, so this is the one option the raws do not honour; it is
    % the same on the quasi-hyperbolic path.
    if ~isfield(vfoptions,'multigridswitch')
        vfoptions.multigridswitch=10; % postGI default (ValueFnIter_InfHorz_GridInterpLayer sets this itself; this dispatcher bypasses it)
    end
    if ~isfield(vfoptions,'maxaprimediff')
        % Mirrored from ValueFnIter_InfHorz_GridInterpLayer: the small default is only safe when
        % aprime is the sole choice; with a decision variable the coarse argmax can sit a long way
        % from the fine one, which is what the window is centred on.
        if prod(n_d)==0
            vfoptions.maxaprimediff=5;
        else
            if n_a(1)<300
                vfoptions.maxaprimediff=ceil(n_a(1)/5);
            else
                vfoptions.maxaprimediff=ceil(n_a(1)/10);
            end
        end
    end
end

%% Some Epstein-Zin specific options need to be set if they are not already declared
if ~isfield(vfoptions,'EZriskaversion')
    error('When using Epstein-Zin preferences you must declare vfoptions.EZriskaversion (coefficient controlling risk aversion)')
end
if ~isfield(vfoptions,'EZutils')
    vfoptions.EZutils=1; % Use EZ preferences with general utility function (0 gives traditional EZ with exogenous labor)
elseif ~(vfoptions.EZutils==0 || vfoptions.EZutils==1)
    error('vfoptions.EZutils must be 0 (traditional Epstein-Zin in consumption units) or 1 (Epstein-Zin in utility units); there is no other option (an older comment mentioned 2 for traditional EZ with endogenous labor, which has never been implemented)')
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
elseif ~(vfoptions.EZoneminusbeta==0 || vfoptions.EZoneminusbeta==1)
    error('vfoptions.EZoneminusbeta must be 0 or 1 in infinite-horizon (the FHorz option 2, a (1-sj*beta) term, needs survival probabilities, which infinite-horizon does not have; it was previously ignored without an error)')
end
% Set up sj
if isfield(vfoptions,'survivalprobability') || isfield(vfoptions,'WarmGlowBequestsFn')
    error('Epstein-Zin preferences in infinite-horizon do not support vfoptions.survivalprobability nor vfoptions.WarmGlowBequestsFn')
end


%% Based on the settings, define a bunch of variables that are used to implement the EZ preferences
% Note that the discount factor and survival probabilities can depend on jj (age/period)
% But the 'relative risk aversion' and 'elasticity of intertemporal substitution' cannot depend on jj
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
    if ceis==1
        error('Cannot use EZeis parameter equal to one with Epstein-Zin preferences (look at formula, it would mean having to raise to the power of zero; you can always put 0.99 or 1.01)')
    end
end



%%

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

%% Starting value for the iteration
% A zero continuation is a genuine value under Epstein-Zin: with risk aversion above one the certainty
% equivalent of a lottery that can give zero is zero (a catastrophe), and in consumption units with
% EIS<1 a V of zero everywhere is a FIXED POINT of the aggregator. So the iteration cannot start from
% the zeros that ValueFnIter_InfHorz supplies by default: it would never leave them. It starts instead
% from the myopic value, the best return today with no continuation, (ezc1*max_{d,aprime} F^ezc2)^ezc7,
% which is finite and nonzero at every state with a feasible choice. (Until 2026-10-27 the raws treated
% V==0 as 'contributes nothing to the expectation', which made the zeros start work but mis-valued a
% genuine zero, and treated a return of exactly zero as infeasible; see
% EZ_feasibility_conventions_proposal.md at the toolkit root.)
% A user-supplied vfoptions.V0 is used as given; zeros in it are catastrophes when ezc5<0, hence the warning.
if vfoptions.actualV0==1 && ezc5<0 && any(V0(:)==0)
    warning('vfoptions.V0 contains exact zeros. Under Epstein-Zin preferences (consumption units, or utility units with a positive-valued utility fn) a zero continuation value is a catastrophe (certainty equivalent zero), not an uninitialised entry; the iteration may not recover from it. Omit vfoptions.V0 to start from the myopic value instead.')
end
if vfoptions.actualV0==0
    % The myopic start, one z at a time (cheap, and the same code for every path below): ezc1*F^ezc2 on
    % the finite entries, then ^ezc7 on everything but -Inf, then the max over (d,aprime)
    l_z=length(n_z);
    V0=zeros(N_a,N_z,'gpuArray');
    for z_c=1:N_z
        if N_d==0
            ReturnMatrix_z=CreateReturnFnMatrix_Disc(ReturnFn,0, n_a, ones(l_z,1),0, a_grid, z_gridvals(z_c,:),ReturnFnParamsVec,0);
        else
            ReturnMatrix_z=CreateReturnFnMatrix_Disc(ReturnFn,n_d, n_a, ones(l_z,1),d_gridvals, a_grid, z_gridvals(z_c,:),ReturnFnParamsVec,0);
        end
        temp2_z=ReturnMatrix_z;
        temp2_z(isfinite(ReturnMatrix_z))=ezc1*ReturnMatrix_z(isfinite(ReturnMatrix_z)).^ezc2;
        temp2_z(~(temp2_z==-Inf))=temp2_z(~(temp2_z==-Inf)).^ezc7;
        V0(:,z_c)=reshape(max(temp2_z,[],1),[N_a,1]);
    end
end

%%
if vfoptions.gridinterplayer==1

    if N_d==0
        [VKron,Policy]=ValueFnIter_InfHorz_EpsteinZin_postGI_nod_raw(V0, n_a, n_z, a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7);
    else
        [VKron,Policy]=ValueFnIter_InfHorz_EpsteinZin_Refine_postGI_raw(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7);
    end

elseif vfoptions.lowmemory==0

    %% CreateReturnFnMatrix_Disc_CPU creates a matrix of dimension (d and aprime)-by-a-by-z.
    % Since the return function is independent of time creating it once and then using it every iteration is good
    % for speed, but it does use a lot of memory.
    ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals,ReturnFnParamsVec,0);

    if N_d==0
        [VKron,Policy]=ValueFnIter_InfHorz_EpsteinZin_nod_raw(V0, n_a, n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter, vfoptions.verbose, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7);
        Policy=shiftdim(Policy,-1);
    else
        [VKron, Policy]=ValueFnIter_InfHorz_EpsteinZin_raw(V0, n_d,n_a,n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, vfoptions.maxiter, vfoptions.verbose, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7);
    end

elseif vfoptions.lowmemory==1

    if N_d==0
        [VKron,Policy]=ValueFnIter_InfHorz_EpsteinZin_LowMem_nod_raw(V0, n_a, n_z, a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.maxiter, vfoptions.verbose, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7);
        Policy=shiftdim(Policy,-1);
    else
        [VKron, Policy]=ValueFnIter_InfHorz_EpsteinZin_LowMem_raw(V0, n_d,n_a,n_z, d_gridvals, a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, vfoptions.maxiter, vfoptions.verbose, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7);
    end
end


%%
V=reshape(VKron,[n_a,n_z]);
% (UnKron handles the grid-interpolation Policy layout, rows L1/L2/L2flag after any d, itself)
if N_d==0
    Policy=UnKronPolicyIndexes1_z(Policy, n_a, n_a, n_z, vfoptions);
else
    Policy=UnKronPolicyIndexes2_z(Policy, n_d, n_a, n_a, n_z, vfoptions);
end

end
