function V=ValueFnFromPolicy_InfHorz_EpsteinZin(Policy,n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% Compute V from a given Policy when the infinite-horizon model uses Epstein-Zin preferences.
% This is the InfHorz mirror of ValueFnFromPolicy_FHorz_EpsteinZin: it evaluates the Epstein-Zin
% recursion along the given policy (all three EZ cases: consumption-units EZutils=0, utility-units
% EZutils=1 with positive or negative valued utility fn), iterating the policy-evaluation fixed point
%   V = [ ezc1*F(Policy)^ezc2 + ezc3*beta*( E[(ezc4*V')^ezc5 | z] at Policy )^ezc6 ]^ezc7
% to vfoptions.tolerance (bounded by vfoptions.maxiter). Evaluating at the argmax policy of
% ValueFnIter_InfHorz_EpsteinZin reproduces its V to convergence tolerance.
%
% With vfoptions.gridinterplayer=1 the policy is a (lower node, L2) pair and the continuation is the
% two-point interpolation of the TRANSFORMED expectation E[(ezc4*V')^ezc5] between the two coarse
% nodes, with the certainty-equivalent power ^ezc6 applied pointwise afterwards: the same object the
% Epstein-Zin grid-interpolation raws interpolate (mirrors ValueFnFromPolicy_FHorz_EpsteinZin's GI1
% branch and the two-point lookup of ValueFnFromPolicy_InfHorz_QuasiHyperbolic). Each weighted term is
% zeroed where its weight is exactly zero, because 0*(-Inf) is NaN.
%
% The arithmetic (the ezc constants, and the finite/-Inf conventions around every power) is exactly
% that of the ValueFnIter_InfHorz_EpsteinZin raws, so that this agrees with the solver and so that a
% Policy that is NOT the argmax is valued under the same conventions. Infinite horizon has no
% survival probabilities and no warm-glow, so there is no sj, no ezc8 and no terminal-age composition.
% Feasibility conventions: EZ_feasibility_conventions_proposal.md at the toolkit root.

%% Scope limits (the same shapes ValueFnIter_InfHorz_EpsteinZin refuses)
if prod(n_z)==0
    error('ValueFnFromPolicy_InfHorz_EpsteinZin: Epstein-Zin preferences in infinite-horizon require an exogenous state z (n_z=0 is not implemented)')
end
if prod(vfoptions.n_e)>0
    error('vfoptions.n_e (iid e) is not yet implemented for ValueFnFromPolicy_InfHorz with vfoptions.exoticpreferences=EpsteinZin')
end
if prod(vfoptions.n_semiz)>0
    error('vfoptions.n_semiz (semi-exogenous states) is not yet implemented for ValueFnFromPolicy_InfHorz with vfoptions.exoticpreferences=EpsteinZin')
end
if vfoptions.gridinterplayer==1 && ~isscalar(n_a)
    error('ValueFnFromPolicy_InfHorz_EpsteinZin: the grid interpolation layer is implemented for a single endogenous state only')
end
if isfield(vfoptions,'EZmortalityriskaversion')
    error('vfoptions.EZmortalityriskaversion (a separate risk aversion over mortality risk) has no meaning in infinite-horizon, where there are no survival probabilities; remove it')
end
if isfield(vfoptions,'survivalprobability') || isfield(vfoptions,'WarmGlowBequestsFn')
    error('Epstein-Zin preferences in infinite-horizon do not support vfoptions.survivalprobability nor vfoptions.WarmGlowBequestsFn')
end

%% Some Epstein-Zin specific options need to be set if they are not already declared
% (this preamble mirrors ValueFnIter_InfHorz_EpsteinZin)
if ~isfield(vfoptions,'EZriskaversion')
    error('When using Epstein-Zin preferences you must declare vfoptions.EZriskaversion (coefficient controlling risk aversion)')
end
if ~isfield(vfoptions,'EZutils')
    vfoptions.EZutils=1; % Use EZ preferences with general utility function (0 gives traditional EZ with exogenous labor)
elseif ~(vfoptions.EZutils==0 || vfoptions.EZutils==1)
    error('vfoptions.EZutils must be 0 (traditional Epstein-Zin in consumption units) or 1 (Epstein-Zin in utility units); there is no other option')
end
if vfoptions.EZutils==1
    if ~isfield(vfoptions,'EZpositiveutility')
        warning('Using Epstein-Zin preferences it is assumed the utility/return function is negative valued, if not you need to set vfoptions.EZpositiveutility=1')
        vfoptions.EZpositiveutility=0;
    end
else
    if ~isfield(vfoptions,'EZeis')
        error('When using Epstein-Zin preferences you must declare vfoptions.EZeis (elasticity of intertemporal substitution)')
    end
end
if ~isfield(vfoptions,'EZoneminusbeta')
    vfoptions.EZoneminusbeta=0;
elseif ~(vfoptions.EZoneminusbeta==0 || vfoptions.EZoneminusbeta==1)
    error('vfoptions.EZoneminusbeta must be 0 or 1 in infinite-horizon')
end

%% Based on the settings, define a bunch of variables that are used to implement the EZ preferences
crisk=Parameters.(vfoptions.EZriskaversion);
if vfoptions.EZutils==0
    ceis=Parameters.(vfoptions.EZeis);
    % Traditional EZ in consumption units
    ezc1=1; % used if vfoptions.EZoneminusbeta=1
    ezc2=1-1/ceis;
    ezc3=1;
    ezc4=1;
    ezc5=1-crisk;
    ezc6=(1-1/ceis)/(1-crisk);
    ezc7=1/(1-1/ceis);
elseif vfoptions.EZutils==1
    % EZ in utility-units
    ezc1=1; % used if vfoptions.EZoneminusbeta=1
    ezc2=1;
    if vfoptions.EZpositiveutility==1
        ezc3=1;
        ezc4=1;
    elseif vfoptions.EZpositiveutility==0
        ezc3=-1;
        ezc4=-1;
    end
    if vfoptions.EZpositiveutility==1
        ezc5=1-crisk;
        ezc6=1/(1-crisk);
    elseif vfoptions.EZpositiveutility==0
        ezc5=1+crisk;
        ezc6=1/(1+crisk);
    end
    ezc7=1;
end

DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames)); % Infinite horizon, so just do this once
if vfoptions.EZoneminusbeta==1
    ezc1=1-DiscountFactorParamsVec;
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

%% Setup (mirrors ValueFnFromPolicy_InfHorz)
N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if vfoptions.gridinterplayer==1
    l_daprime=size(Policy,1)-2; % -2 for the L2index and L2flag
elseif N_d==0 && isscalar(n_a)
    l_daprime=1;
else
    l_daprime=size(Policy,1);
end

a_gridvals=CreateGridvals(n_a,a_grid,1);
% Switch to z_gridvals
[z_gridvals, pi_z, vfoptions]=ExogShockSetup_InfHorz(n_z,z_grid,pi_z,Parameters,vfoptions,3,0);

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,0,vfoptions,Parameters);
ReturnFnParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames);

%% Calculate FofPolicy (the return fn evaluated at the Policy; under the grid interpolation layer this is at the fine point)
PolicyValues=PolicyInd2Val_InfHorz(Policy,n_d,n_a,n_z,d_grid,a_grid, vfoptions);
PolicyValuesPermute=permute(reshape(PolicyValues,[size(PolicyValues,1),N_a,N_z]),[2,3,1]); %[N_a,N_z,l_d+l_a]
FofPolicy=EvalFnOnAgentDist_Grid(ReturnFn, ReturnFnParamsCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);
FofPolicy=reshape(FofPolicy,[N_a,N_z]);

% Modify the Return Function appropriately for Epstein-Zin Preferences (as the raws do)
becareful=isfinite(FofPolicy); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
temp2=FofPolicy;
temp2(becareful)=FofPolicy(becareful).^ezc2;

%% The aprime index of the policy, in the form the expectation step needs
% Row (z-1)*N_a+a of pi_z_howards is pi_z(z,:), matching the ordering of a (N_a*N_z)-by-N_z lookup
pi_z_howards=repelem(pi_z,N_a,1);
Policy=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, n_z, 0, vfoptions);
if vfoptions.gridinterplayer==1
    % rows after any d: lower node L1, then L2 (the Kron drops L2flag)
    index_a1=1+(N_d>0);
    alowerindex=reshape(ceil(Policy(index_a1,:,:)),[N_a*N_z,1]);
    wupper=reshape((ceil(Policy(end,:,:))-1)/(vfoptions.ngridinterp+1),[N_a*N_z,1]); % weight on the upper node
    wlower=1-wupper;
    lowerweightzero=(wlower==0); % the chosen point is the upper coarse node
    upperweightzero=(wupper==0); % the chosen point is the lower coarse node
else
    if N_d==0
        Policy_a=shiftdim(Policy(1,:,:),1);
    else
        Policy_a=shiftdim(ceil(Policy(2,:,:)),1);
    end
end

%% Iterate the policy-evaluation fixed point
% Start from the myopic value at the policy, (ezc1*F(Policy)^ezc2)^ezc7, as the solver starts from the
% myopic value over all choices: a zero continuation is a genuine value under Epstein-Zin (a catastrophe
% when risk aversion is above one, and in consumption units with EIS<1 a fixed point of the aggregator),
% so a zeros start could never leave it. See EZ_feasibility_conventions_proposal.md at the toolkit root.
VKron=ezc1*temp2;
VKron(~(VKron==-Inf))=VKron(~(VKron==-Inf)).^ezc7;
currdist=Inf;
itercount=1;
while currdist>vfoptions.tolerance && itercount<=vfoptions.maxiter
    VKronold=VKron;

    % Part of Epstein-Zin is before taking expectation
    temp=VKronold;
    temp(isfinite(VKronold))=(ezc4*VKronold(isfinite(VKronold))).^ezc5;
    temp(VKronold==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf

    % The expectation term (except beta), at the policy
    if vfoptions.gridinterplayer==1
        EVlower=wlower.*temp(alowerindex,:); % [N_a*N_z,N_z], last dimension is zprime
        EVlower(lowerweightzero,:)=0; % zero weight times -Inf gives NaN, a zero weight contributes nothing
        EVupper=wupper.*temp(alowerindex+1,:);
        EVupper(upperweightzero,:)=0;
        EVKrontemp=(EVlower+EVupper).*pi_z_howards;
    else
        EVKrontemp=temp(Policy_a,:);
        EVKrontemp=EVKrontemp.*pi_z_howards;
    end
    EVKrontemp(isnan(EVKrontemp))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);

    temp4=EVKrontemp;
    temp4(~(temp4==-Inf))=temp4(~(temp4==-Inf)).^ezc6;
    temp4(EVKrontemp==0)=0;

    VKron=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4;
    VKron(isnan(VKron))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
    temp5=~(VKron==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
    VKron(temp5)=VKron(temp5).^ezc7;

    VKrondist=VKron(:)-VKronold(:); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));
    itercount=itercount+1;
end

if currdist>vfoptions.tolerance
    warning(['ValueFnFromPolicy_InfHorz_EpsteinZin: the policy-evaluation fixed point stopped on reaching ', ...
             'the maximum number of iterations, not on convergence (set by vfoptions.maxiter). ', ...
             'The returned V is not the value of the given Policy. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], currdist, vfoptions.tolerance)
end

V=reshape(VKron,[n_a,n_z]);

end
