function [V,Valt]=ValueFnFromPolicy_InfHorz_QuasiHyperbolic(Policy,n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% Compute V and Valt from a given Policy when the infinite-horizon model uses Quasi-Hyperbolic
% discounting. This is the InfHorz mirror of ValueFnFromPolicy_FHorz_QuasiHyperbolic.
%
% Returns [V, Valt] matching the convention of ValueFnIter_InfHorz_QuasiHyperbolic:
%   Naive:         V    = Vtilde, the quasi-hyperbolic value at Policy
%                  Valt = V_std, the exponential-discounter value at Policyalt
%                  Requires vfoptions.Policyalt (the exponential-discounter argmax, available as
%                  the 4th output of ValueFnIter_InfHorz for Naive). Naive stores Policy at the
%                  beta0*beta argmax while Valt is the exponential value at the std argmax, so
%                  Policy alone cannot reconstruct both.
%   Sophisticated: V    = Vhat
%                  Valt = Vunderbar, the realised continuation under future selves' own choices
%                  Policy alone suffices; no Policyalt.
%
% The arithmetic, in both cases, is one policy-evaluation fixed point plus one extra step:
%   Naive:         Valt      = u(Policyalt) + beta      *E[Valt      at Policyalt]   (fixed point)
%                  Vtilde    = u(Policy)    + beta0*beta*E[Valt      at Policy]      (one step)
%   Sophisticated: Vunderbar = u(Policy)    + beta      *E[Vunderbar at Policy]      (fixed point)
%                  Vhat      = u(Policy)    + beta0*beta*E[Vunderbar at Policy]      (one step)
% Everything below mirrors ValueFnFromPolicy_InfHorz (gridinterplayer=0) and
% ValueFnFromPolicy_InfHorz_GI (gridinterplayer=1); the grid-interpolation branch is inlined here
% rather than sent to a separate file, matching ValueFnFromPolicy_FHorz_QuasiHyperbolic.
%
% Note: ValueFnIter_InfHorz_QuasiHyperbolic has no 'without z' code path (every quasi-hyperbolic
% InfHorz raw indexes pi_z and loops over z), so no InfHorz solve can currently hand this command
% a noz Policy. The N_z==0 branches below are written for symmetry with the rest of
% ValueFnFromPolicy and are UNTESTED for that reason.

%% Which quasi-hyperbolic solution, and the additional discount factor
if ~isfield(vfoptions,'quasi_hyperbolic')
    vfoptions.quasi_hyperbolic='Naive'; % This is the default, alternative is 'Sophisticated'.
elseif ~strcmp(vfoptions.quasi_hyperbolic,'Naive') && ~strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
    error('vfoptions.quasi_hyperbolic must be either Naive or Sophisticated (check spelling and capital letter)')
end
isNaive=strcmp(vfoptions.quasi_hyperbolic,'Naive');

if ~isfield(vfoptions,'QHadditionaldiscount')
    error('You must declare vfoptions.QHadditionaldiscount when using quasi-hyperbolic discounting (you have vfoptions.exoticpreferences set to QuasiHyperbolic)')
elseif ~ischar(vfoptions.QHadditionaldiscount)
    error('vfoptions.QHadditionaldiscount must be the name of the additional discount parameter, given as a character vector such as ''beta0'' (this matches how vfoptions.EZriskaversion and vfoptions.survivalprobability are declared)')
end
beta0=Parameters.(vfoptions.QHadditionaldiscount);
if ~isscalar(beta0)
    error('The quasi-hyperbolic additional discount factor (the parameter named by vfoptions.QHadditionaldiscount) must be a scalar; it cannot depend on age')
end

%% Naive requires Policyalt
if isNaive
    if ~isfield(vfoptions,'Policyalt')
        error('ValueFnFromPolicy_InfHorz_QuasiHyperbolic (Naive): vfoptions.Policyalt is required. Naive quasi-hyperbolic stores Policy at the quasi-hyperbolic (beta0*beta) argmax but Valt is the exponential-discounter value at the std argmax. To reconstruct V and Valt from policies alone, pass the exponential-discounter argmax (Policyalt) via vfoptions.Policyalt. It is returned as the 4th output of ValueFnIter_InfHorz for Naive.')
    end
    Policyalt=gpuArray(vfoptions.Policyalt);
end

%% Setup (mirrors ValueFnFromPolicy_InfHorz)
N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
l_a=length(n_a);

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

%% Calculate FofPolicy (the return fn evaluated at the Policy), and at Policyalt for Naive
PolicyValues=PolicyInd2Val_InfHorz(Policy,n_d,n_a,n_z,d_grid,a_grid, vfoptions);
if N_z==0
    PolicyValuesPermute=permute(reshape(PolicyValues,[size(PolicyValues,1),N_a]),[2,1]); %[N_a,l_d+l_a]
else
    PolicyValuesPermute=permute(reshape(PolicyValues,[size(PolicyValues,1),N_a,N_z]),[2,3,1]); %[N_a,N_z,l_d+l_a]
end
FofPolicy=EvalFnOnAgentDist_Grid(ReturnFn, ReturnFnParamsCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);

if isNaive
    PolicyaltValues=PolicyInd2Val_InfHorz(Policyalt,n_d,n_a,n_z,d_grid,a_grid, vfoptions);
    if N_z==0
        PolicyaltValuesPermute=permute(reshape(PolicyaltValues,[size(PolicyaltValues,1),N_a]),[2,1]);
    else
        PolicyaltValuesPermute=permute(reshape(PolicyaltValues,[size(PolicyaltValues,1),N_a,N_z]),[2,3,1]);
    end
    FofPolicyalt=EvalFnOnAgentDist_Grid(ReturnFn, ReturnFnParamsCell,PolicyaltValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);
end

beta=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames)); % Discount rate between two future periods
beta0beta=beta0*beta; % Discount rate between present period and next period

% The fixed point is evaluated at Policyalt for Naive and at Policy for Sophisticated, so the flow
% payoff that drives it, and the rough starting guess, differ by case.
if isNaive
    Ffixedpoint=FofPolicyalt;
else
    Ffixedpoint=FofPolicy;
end

%% Grid interpolation layer
% Mirrors ValueFnFromPolicy_InfHorz_GI: the aprime lookup is at (alower, alower+1) with weights
% from L2, rather than at a single integer aprime index.
if vfoptions.gridinterplayer==1
    index_a1=1+(N_d>0); % row of a1 in the Kron'd policy (comes after d, when there is a d)

    if N_z==0
        PolicyKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, 1, 0, vfoptions);
        alowerindex=reshape(ceil(PolicyKron(index_a1,:,:)),[1,N_a]);
        if l_a>=2 % GI2A: fold the remaining endogenous state(s) into the linear index
            alowerindex=alowerindex+n_a(1)*(reshape(ceil(PolicyKron(index_a1+1,:,:)),[1,N_a])-1);
        end
        aprimeindex=[alowerindex; alowerindex+1]; % [2,N_a]
        PolicyProbs=reshape(ceil(PolicyKron(end,:,:)),[1,N_a]); % L2 (Kron drops L2flag)
        PolicyProbs=(PolicyProbs-1)/(vfoptions.ngridinterp+1); % prob of upper point
        PolicyProbs=[1-PolicyProbs; PolicyProbs]; % [2,N_a]

        if isNaive
            PolicyaltKron=KronPolicyIndexes_forValueFnFromPolicy(Policyalt, n_d, n_a, 1, 0, vfoptions);
            alowerindexalt=reshape(ceil(PolicyaltKron(index_a1,:,:)),[1,N_a]);
            if l_a>=2
                alowerindexalt=alowerindexalt+n_a(1)*(reshape(ceil(PolicyaltKron(index_a1+1,:,:)),[1,N_a])-1);
            end
            aprimeindexalt=[alowerindexalt; alowerindexalt+1];
            PolicyProbsalt=reshape(ceil(PolicyaltKron(end,:,:)),[1,N_a]);
            PolicyProbsalt=(PolicyProbsalt-1)/(vfoptions.ngridinterp+1);
            PolicyProbsalt=[1-PolicyProbsalt; PolicyProbsalt];
            aprimeindexfp=aprimeindexalt; % the fixed point is evaluated at Policyalt
            PolicyProbsfp=PolicyProbsalt;
        else
            aprimeindexfp=aprimeindex; % the fixed point is evaluated at Policy
            PolicyProbsfp=PolicyProbs;
        end

        % Fixed point: Valt (Naive) or Vunderbar (Sophisticated)
        currdist=Inf;
        itercount=1;
        VfpKron=Ffixedpoint/(1-beta); % rough guess
        while currdist>vfoptions.tolerance && itercount<vfoptions.maxiter
            VfpKronold=VfpKron;

            EVKrontemp=reshape(VfpKron(aprimeindexfp,:),[2,N_a]);
            EVKrontemp=PolicyProbsfp.*EVKrontemp;
            % A zero interpolation weight on a grid point where V is -Inf gives 0*(-Inf)=NaN, which
            % then contaminates the sum; a zero weight is meant to contribute nothing.
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=shiftdim(sum(EVKrontemp,1),1);

            VfpKron=Ffixedpoint+beta*EVKrontemp;

            currdist=max(max(abs(VfpKron-VfpKronold)));
            itercount=itercount+1;
        end

        % One extra step, at Policy, with the quasi-hyperbolic discount factor
        EVKrontemp=reshape(VfpKron(aprimeindex,:),[2,N_a]);
        EVKrontemp=PolicyProbs.*EVKrontemp;
        EVKrontemp(isnan(EVKrontemp))=0;
        EVKrontemp=shiftdim(sum(EVKrontemp,1),1);
        VqhKron=FofPolicy+beta0beta*EVKrontemp;

        V=reshape(VqhKron,[n_a,1]);
        Valt=reshape(VfpKron,[n_a,1]);
    else % N_z>0
        PolicyKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, n_z, 0, vfoptions);
        pi_z_howards=repelem(pi_z,N_a,1);

        alowerindex=reshape(ceil(PolicyKron(index_a1,:,:)),[1,N_a*N_z]);
        if l_a>=2 % GI2A: fold the remaining endogenous state(s) into the linear index
            alowerindex=alowerindex+n_a(1)*(reshape(ceil(PolicyKron(index_a1+1,:,:)),[1,N_a*N_z])-1);
        end
        aprimeindex=[alowerindex; alowerindex+1]; % [2,N_a*N_z]
        PolicyProbs=reshape(ceil(PolicyKron(end,:,:)),[1,N_a*N_z]); % L2 (Kron drops L2flag)
        PolicyProbs=(PolicyProbs-1)/(vfoptions.ngridinterp+1); % prob of upper point
        PolicyProbs=[1-PolicyProbs; PolicyProbs]; % [2,N_a*N_z]

        if isNaive
            PolicyaltKron=KronPolicyIndexes_forValueFnFromPolicy(Policyalt, n_d, n_a, n_z, 0, vfoptions);
            alowerindexalt=reshape(ceil(PolicyaltKron(index_a1,:,:)),[1,N_a*N_z]);
            if l_a>=2
                alowerindexalt=alowerindexalt+n_a(1)*(reshape(ceil(PolicyaltKron(index_a1+1,:,:)),[1,N_a*N_z])-1);
            end
            aprimeindexalt=[alowerindexalt; alowerindexalt+1];
            PolicyProbsalt=reshape(ceil(PolicyaltKron(end,:,:)),[1,N_a*N_z]);
            PolicyProbsalt=(PolicyProbsalt-1)/(vfoptions.ngridinterp+1);
            PolicyProbsalt=[1-PolicyProbsalt; PolicyProbsalt];
            aprimeindexfp=aprimeindexalt;
            PolicyProbsfp=PolicyProbsalt;
        else
            aprimeindexfp=aprimeindex;
            PolicyProbsfp=PolicyProbs;
        end

        % Fixed point: Valt (Naive) or Vunderbar (Sophisticated)
        currdist=Inf;
        itercount=1;
        VfpKron=Ffixedpoint/(1-beta); % rough guess
        while currdist>vfoptions.tolerance && itercount<vfoptions.maxiter
            VfpKronold=VfpKron;

            EVKrontemp=reshape(VfpKron(aprimeindexfp,:),[2,N_a*N_z,N_z]); % last dimension is zprime
            EVKrontemp=PolicyProbsfp.*EVKrontemp;
            % A zero interpolation weight on a grid point where V is -Inf gives 0*(-Inf)=NaN, which
            % then contaminates the sum; a zero weight is meant to contribute nothing.
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=shiftdim(sum(EVKrontemp,1),1); % [N_a*N_z,N_z]

            EVKrontemp=EVKrontemp.*pi_z_howards;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VfpKron=Ffixedpoint+beta*EVKrontemp;

            currdist=max(max(abs(VfpKron-VfpKronold)));
            itercount=itercount+1;
        end

        % One extra step, at Policy, with the quasi-hyperbolic discount factor
        EVKrontemp=reshape(VfpKron(aprimeindex,:),[2,N_a*N_z,N_z]);
        EVKrontemp=PolicyProbs.*EVKrontemp;
        EVKrontemp(isnan(EVKrontemp))=0;
        EVKrontemp=shiftdim(sum(EVKrontemp,1),1);
        EVKrontemp=EVKrontemp.*pi_z_howards;
        EVKrontemp(isnan(EVKrontemp))=0;
        EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
        VqhKron=FofPolicy+beta0beta*EVKrontemp;

        V=reshape(VqhKron,[n_a,n_z]);
        Valt=reshape(VfpKron,[n_a,n_z]);
    end
else % No grid interpolation layer
    if N_z==0
        PolicyKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, 1, 0, vfoptions);
        if N_d==0
            Policy_a=shiftdim(PolicyKron(1,:),1);
        else
            Policy_a=shiftdim(ceil(PolicyKron(2,:)),1);
        end

        if isNaive
            PolicyaltKron=KronPolicyIndexes_forValueFnFromPolicy(Policyalt, n_d, n_a, 1, 0, vfoptions);
            if N_d==0
                Policyalt_a=shiftdim(PolicyaltKron(1,:),1);
            else
                Policyalt_a=shiftdim(ceil(PolicyaltKron(2,:)),1);
            end
            Policy_afp=Policyalt_a; % the fixed point is evaluated at Policyalt
        else
            Policy_afp=Policy_a; % the fixed point is evaluated at Policy
        end

        % Fixed point: Valt (Naive) or Vunderbar (Sophisticated)
        currdist=Inf;
        itercount=1;
        VfpKron=Ffixedpoint/(1-beta); % rough guess
        while currdist>vfoptions.tolerance && itercount<vfoptions.maxiter
            VfpKronold=VfpKron;

            EVKrontemp=VfpKron(Policy_afp,:);

            VfpKron=Ffixedpoint+beta*EVKrontemp;

            currdist=max(max(abs(VfpKron-VfpKronold)));
            itercount=itercount+1;
        end

        % One extra step, at Policy, with the quasi-hyperbolic discount factor
        VqhKron=FofPolicy+beta0beta*VfpKron(Policy_a,:);

        V=reshape(VqhKron,[n_a,1]);
        Valt=reshape(VfpKron,[n_a,1]);
    else % N_z>0
        PolicyKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, n_z, 0, vfoptions);
        pi_z_howards=repelem(pi_z,N_a,1);

        if N_d==0
            Policy_a=shiftdim(PolicyKron(1,:,:),1);
        else
            Policy_a=shiftdim(ceil(PolicyKron(2,:,:)),1);
        end

        if isNaive
            PolicyaltKron=KronPolicyIndexes_forValueFnFromPolicy(Policyalt, n_d, n_a, n_z, 0, vfoptions);
            if N_d==0
                Policyalt_a=shiftdim(PolicyaltKron(1,:,:),1);
            else
                Policyalt_a=shiftdim(ceil(PolicyaltKron(2,:,:)),1);
            end
            Policy_afp=Policyalt_a; % the fixed point is evaluated at Policyalt
        else
            Policy_afp=Policy_a; % the fixed point is evaluated at Policy
        end

        % Fixed point: Valt (Naive) or Vunderbar (Sophisticated)
        currdist=Inf;
        itercount=1;
        VfpKron=Ffixedpoint/(1-beta); % rough guess
        while currdist>vfoptions.tolerance && itercount<vfoptions.maxiter
            VfpKronold=VfpKron;

            EVKrontemp=VfpKron(Policy_afp,:);

            EVKrontemp=EVKrontemp.*pi_z_howards;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
            VfpKron=Ffixedpoint+beta*EVKrontemp;

            currdist=max(max(abs(VfpKron-VfpKronold)));
            itercount=itercount+1;
        end

        % One extra step, at Policy, with the quasi-hyperbolic discount factor
        EVKrontemp=VfpKron(Policy_a,:);
        EVKrontemp=EVKrontemp.*pi_z_howards;
        EVKrontemp(isnan(EVKrontemp))=0;
        EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
        VqhKron=FofPolicy+beta0beta*EVKrontemp;

        V=reshape(VqhKron,[n_a,n_z]);
        Valt=reshape(VfpKron,[n_a,n_z]);
    end
end

if currdist>vfoptions.tolerance
    warning(['ValueFnFromPolicy_InfHorz_QuasiHyperbolic: the policy-evaluation fixed point stopped ', ...
             'on reaching the maximum number of iterations, not on convergence (set by vfoptions.maxiter). ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], currdist, vfoptions.tolerance)
end

end
