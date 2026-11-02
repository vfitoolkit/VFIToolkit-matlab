function [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EpsteinZin(VKron,n_d,n_a,n_z,n_e,N_j,d_gridvals,a_grid,z_gridvals_J,e_gridvals_J,pi_z_J,pi_e_J,ReturnFn,Parameters,DiscountFactorParamNames,ReturnFnParamNames,vfoptions,fastOLG)
% Epstein-Zin preferences, one period of the FHorz transition path. The VKron input is next period value
% fn, the VKron output is this period.
%
% Called by all eight single-step dispatchers (ValueFnIter_FHorz_TPath_SingleStep, _noz, _e, _noz_e and
% their _fastOLG versions); n_z=0 with no z and n_e=0 with no e, and fastOLG says which layout VKron is in.
% Returns VKron and PolicyKron in exactly the form the dispatcher's own raws do, so the dispatcher's UnKron
% applies unchanged.
%
% Builds this period's Epstein-Zin constants from Parameters, which hold this period's prices and
% parameters: so risk aversion, the EIS, the survival probabilities and the discount factor may all move
% along the path. Everything that does not depend on the period -- the option checks and defaults
% (EZriskaversion, EZeis, EZutils, EZpositiveutility, EZoneminusbeta), the length of the survival
% probabilities, and vfoptions.WarmGlowBequestsFnParamsNames -- is done ONCE, in the top-level commands
% (TransitionPath_Case1_FHorz and ValueFnOnTransPath_Case1_FHorz). For the formulation (what each ezc
% constant is) see ValueFnIter_FHorz_EpsteinZin, from which the block below is taken.

N_d=prod(n_d);
N_z=prod(n_z);
N_e=prod(n_e);

% vfoptions.EVpre=1 is set internally by the Matched Expectations Path, never by a user: the input V is then
% already an expectation, which the Epstein-Zin raws (which take the certainty equivalent of next period's V
% themselves) would silently integrate a second time. Refused rather than ignored.
if isfield(vfoptions,'EVpre')
    if vfoptions.EVpre==1
        error('Epstein-Zin preferences are not implemented for the Matched Expectations Path (vfoptions.EVpre=1)')
    end
end

%% Survival probabilities and the warm-glow
if isfield(vfoptions,'survivalprobability')
    sj=Parameters.(vfoptions.survivalprobability);
elseif isfield(vfoptions,'WarmGlowBequestsFn')
    % Warm-glow without survival probabilities: the warm-glow only at the end of the final period
    sj=ones(N_j,1);
    sj(end)=0;
else
    sj=ones(N_j,1);
end
if isfield(vfoptions,'WarmGlowBequestsFn')
    warmglow=1;
else
    warmglow=0;
end

%% This period's Epstein-Zin constants
% The discount factor and survival probabilities can depend on jj (age); the 'relative risk aversion' and
% 'elasticity of intertemporal substitution' cannot depend on jj, but can depend on the period.
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
else % vfoptions.EZutils==1
    % EZ in utility-units
    ezc1=1; % used if vfoptions.EZoneminusbeta=1
    ezc2=1; % ezc3 is same in both cases
    % If the utility is negative you need to multiply it by -1 in two places
    if vfoptions.EZpositiveutility==1
        ezc3=1; % will be -1 if vfoptions.EZpositiveutility=0
        ezc4=1; % will be -1 if vfoptions.EZpositiveutility=0
    else
        ezc3=-1;
        ezc4=-1;
    end
    % If the utility is negative use 1+crisk instead of 1-crisk. This way
    % the interpretation of crisk is identical in both cases
    if vfoptions.EZpositiveutility==1
        ezc5=1-crisk;
        ezc6=1./(1-crisk);
    else
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
    else
        if vfoptions.EZpositiveutility==1
            ezc6=1./(1-mrisk);
            ezc8=(1-mrisk)./(1-crisk);
        else
            ezc6=1./(1+mrisk);
            ezc8=(1+mrisk)./(1+crisk);
        end
    end
else
    % This wont do anything
    ezc8=1;
end

% Permit age-dependence of these (and make them column vectors if they are not already)
% Note: the only ones that need to permit this are ezc2, ezc5, ezc6, ezc7, ezc8
if size(ezc2,1)==1
    ezc2=ezc2';
end
ezc2=ezc2.*ones(N_j,1); % this will work whether it starts N_j-by-1 or 1-by-1
if size(ezc5,1)==1
    ezc5=ezc5';
end
ezc5=ezc5.*ones(N_j,1);
if size(ezc6,1)==1
    ezc6=ezc6';
end
ezc6=ezc6.*ones(N_j,1);
if size(ezc7,1)==1
    ezc7=ezc7';
end
ezc7=ezc7.*ones(N_j,1);
if size(ezc8,1)==1
    ezc8=ezc8';
end
ezc8=ezc8.*ones(N_j,1);

if vfoptions.EZoneminusbeta==1
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    ezc1=1-prod(DiscountFactorParamsVec); % (The raws recompute it age by age)
elseif vfoptions.EZoneminusbeta==2
    % Some formulations using bequests multiply the period utility function by (1-sj*beta)
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    ezc1=1-sj(N_j)*prod(DiscountFactorParamsVec);
end

if vfoptions.divideandconquer==1
    vfoptions.level1n=min(vfoptions.level1n,n_a);
end

%% Solve this period
if fastOLG==0
    if vfoptions.divideandconquer==0 && vfoptions.gridinterplayer==0
        if N_e==0
            if N_z==0
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_nod_noz_raw(VKron,n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_noz_raw(VKron,n_d,n_a, N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            else
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            end
        else % N_e
            if N_z==0
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_nod_noz_e_raw(VKron,n_a, n_e, N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_noz_e_raw(VKron,n_d, n_a, n_e, N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            else
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_nod_e_raw(VKron,n_a, n_z, n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_e_raw(VKron,n_d, n_a, n_z, n_e, N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            end
        end
    elseif vfoptions.divideandconquer==0 % && vfoptions.gridinterplayer==1
        if N_e==0
            if N_z==0
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_GI1_nod_noz_raw(VKron,n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_GI1_noz_raw(VKron,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            else
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_GI1_nod_raw(VKron,n_a,n_z,N_j, a_grid, z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_GI1_raw(VKron,n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            end
        else % N_e
            if N_z==0
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_GI1_nod_noz_e_raw(VKron,n_a,n_e,N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_GI1_noz_e_raw(VKron,n_d,n_a,n_e,N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            else
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_GI1_nod_e_raw(VKron,n_a,n_z,n_e,N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_GI1_e_raw(VKron,n_d,n_a,n_z,n_e,N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            end
        end
    elseif vfoptions.gridinterplayer==0 % && vfoptions.divideandconquer==1
        if N_e==0
            if N_z==0
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_DC1_nod_noz_raw(VKron,n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_DC1_noz_raw(VKron,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            else
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_DC1_nod_raw(VKron,n_a,n_z,N_j, a_grid, z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_DC1_raw(VKron,n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            end
        else % N_e
            if N_z==0
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_DC1_nod_noz_e_raw(VKron,n_a,n_e,N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_DC1_noz_e_raw(VKron,n_d,n_a,n_e,N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            else
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_DC1_nod_e_raw(VKron,n_a,n_z,n_e,N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_DC1_e_raw(VKron,n_d,n_a,n_z,n_e,N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            end
        end
    else % vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==1
        if N_e==0
            if N_z==0
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_DC1_GI1_nod_noz_raw(VKron,n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_DC1_GI1_noz_raw(VKron,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            else
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_DC1_GI1_nod_raw(VKron,n_a,n_z,N_j, a_grid, z_gridvals_J,pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_DC1_GI1_raw(VKron,n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            end
        else % N_e
            if N_z==0
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_DC1_GI1_nod_noz_e_raw(VKron,n_a,n_e,N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_DC1_GI1_noz_e_raw(VKron,n_d,n_a,n_e,N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            else
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_DC1_GI1_nod_e_raw(VKron,n_a,n_z,n_e,N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_DC1_GI1_e_raw(VKron,n_d,n_a,n_z,n_e,N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            end
        end
    end
else % fastOLG==1
    if vfoptions.divideandconquer==0 && vfoptions.gridinterplayer==0
        if N_e==0
            if N_z==0
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_nod_noz_raw(VKron,n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_noz_raw(VKron,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            else
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_nod_raw(VKron,n_a,n_z,N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_raw(VKron,n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            end
        else % N_e
            if N_z==0
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_nod_noz_e_raw(VKron,n_a,n_e,N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_noz_e_raw(VKron,n_d,n_a,n_e,N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            else
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_nod_e_raw(VKron,n_a,n_z,n_e,N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_e_raw(VKron,n_d,n_a,n_z,n_e,N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            end
        end
    elseif vfoptions.divideandconquer==0 % && vfoptions.gridinterplayer==1
        if N_e==0
            if N_z==0
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_GI1_nod_noz_raw(VKron,n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_GI1_noz_raw(VKron,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            else
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_GI1_nod_raw(VKron,n_a,n_z,N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_GI1_raw(VKron,n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            end
        else % N_e
            if N_z==0
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_GI1_nod_noz_e_raw(VKron,n_a,n_e,N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_GI1_noz_e_raw(VKron,n_d,n_a,n_e,N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            else
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_GI1_nod_e_raw(VKron,n_a,n_z,n_e,N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_GI1_e_raw(VKron,n_d,n_a,n_z,n_e,N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            end
        end
    elseif vfoptions.gridinterplayer==0 % && vfoptions.divideandconquer==1
        if N_e==0
            if N_z==0
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_DC1_nod_noz_raw(VKron,n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_DC1_noz_raw(VKron,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            else
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_DC1_nod_raw(VKron,n_a,n_z,N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_DC1_raw(VKron,n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            end
        else % N_e
            if N_z==0
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_DC1_nod_noz_e_raw(VKron,n_a,n_e,N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_DC1_noz_e_raw(VKron,n_d,n_a,n_e,N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            else
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_DC1_nod_e_raw(VKron,n_a,n_z,n_e,N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_DC1_e_raw(VKron,n_d,n_a,n_z,n_e,N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            end
        end
    else % vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==1
        if N_e==0
            if N_z==0
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_DC1_GI1_nod_noz_raw(VKron,n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_DC1_GI1_noz_raw(VKron,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            else
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_DC1_GI1_nod_raw(VKron,n_a,n_z,N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_DC1_GI1_raw(VKron,n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            end
        else % N_e
            if N_z==0
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_DC1_GI1_nod_noz_e_raw(VKron,n_a,n_e,N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_DC1_GI1_noz_e_raw(VKron,n_d,n_a,n_e,N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            else
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_DC1_GI1_nod_e_raw(VKron,n_a,n_z,n_e,N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                else
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_EZ_fastOLG_DC1_GI1_e_raw(VKron,n_d,n_a,n_z,n_e,N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
                end
            end
        end
    end
end

end
