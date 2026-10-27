function [VKron, PolicyKron, aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep(VKron,n_d,n_a,n_z,N_j,d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% The VKron input is next period value fn, the VKron output is this period.
%
% aprimeReferencePolicy and the third output are for vfoptions.localsearch, and mean exactly what they
% mean in ValueFnIter_InfHorz_TPath_SingleStep, with an age dimension: the aprime index at the centre
% of each window, indexed by everything the search conditions on, [N_dref,N_a,N_z,N_j] with N_dref=1
% with no d and N_d with one. The third output comes back empty unless this is a localsearch sweep or
% a seed sweep.
aprimeReferencePolicyNew=[];

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

% Both defaulted HERE, ABOVE THEIR FIRST READ, because most callers of this command (the Jacobian
% builds, ValueFnOnTransPath_Case1_FHorz, the test bank) pass a vfoptions that has never heard of
% them. localsearchseed is internal, set only by Step1.
if ~isfield(vfoptions,'localsearch')
    vfoptions.localsearch=0;
end
if ~isfield(vfoptions,'localsearchseed')
    vfoptions.localsearchseed=0;
end
% ONLY 0 AND 1 ARE MEANINGFUL HERE. ValueFnOnTransPath_Case1_FHorz reads a third value (2 = force a
% restricted search it would otherwise ignore) and translates it before calling this. A 2 that leaks
% through would fall past every ==1 test below and run unrestricted, silently. So it is refused by name.
if vfoptions.localsearch~=0 && vfoptions.localsearch~=1
    error('vfoptions.localsearch must be 0 or 1 here; the value 2 is only meaningful to ValueFnOnTransPath_Case1_FHorz, which uses it to force a restricted search it would otherwise ignore')
end
if (vfoptions.localsearch==1 || vfoptions.localsearchseed==1) && ~strcmp(vfoptions.exoticpreferences,'None')
    error('vfoptions.localsearch=1 is not implemented with exoticpreferences')
end

if strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
    % V slot = Valt (Naive) / Vunderbar (Sophisticated); Policy = QH choice; drop QH-only extras
    [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_QH(VKron,n_d,n_a,n_z,N_j,d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return % QH variant already un-Krons Policy
end

%% If get to here then not using exoticpreferences nor StateDependentVariables_z
% N_z==0 is handled by a different command
if vfoptions.localsearchseed==1 && vfoptions.localsearch==0
    %% The SEED sweep: unrestricted, but returning a reference for vfoptions.localsearch
    % Set by Step1 on the standard sweeps of a localsearch solve -- iteration 1 and every verification
    % sweep. Every branch returns THE COARSE ARGMAX OF THE UNRESTRICTED SEARCH, per d where there is a
    % d, so the reference means one thing whichever sweep built it:
    %   no d, no GI: nothing is folded, so Policy IS the argmax.
    %   GI, with or without d: the GI raw's own midpoint, returned as its third output.
    %   with d, no GI: a per-d argmax, which no single max over a folded (d,aprime) dimension can
    %     give, so LSseed_raw unfolds it.
    % divideandconquer on the seed sweeps is not yet done here (InfHorz honours it); it is refused
    % where the option is validated, in TransitionPath_Case1_FHorz.
    if isscalar(n_a) && vfoptions.divideandconquer==0
        if vfoptions.gridinterplayer==0
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                aprimeReferencePolicyNew=reshape(PolicyKron,[1,N_a,N_z,N_j]);
            else
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_LSseed_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_d==0
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_GI1_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_GI1_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    else
        error('vfoptions.localsearch=1 is currently only implemented for one endogenous state, with z (no e, no semiz), without divideandconquer')
    end
elseif vfoptions.localsearch==1
    %% Local search: aprime restricted to a window around aprimeReferencePolicy
    % lowmemory and divideandconquer are IGNORED on a restricted sweep, silently and by design, as in
    % InfHorz: the window fits in memory whatever the full sweep needs, and the window IS the
    % restriction, so DC has nothing to add.
    vfoptions.divideandconquer=0;
    vfoptions.lowmemory=0;
    if isscalar(n_a)
        N_dref=max(N_d,1);
        if isempty(aprimeReferencePolicy)
            % Default reference: the current a index, so the window sits around staying put. Only
            % ValueFnOnTransPath_Case1_FHorz at localsearch=2 gets here; the TPath solvers always hand
            % in the previous sweep's answer, and Step1 refuses to start a restricted sweep without one.
            aprimeReferencePolicy=repmat((1:1:N_a),[N_dref,1,N_z,N_j]);
        elseif ~(size(aprimeReferencePolicy,1)==N_dref && size(aprimeReferencePolicy,2)==N_a && numel(aprimeReferencePolicy)==N_dref*N_a*N_z*N_j)
            % numel rather than size(...,3) because MATLAB drops trailing singletons
            error('aprimeReferencePolicy must hold one aprime index per state the search conditions on: [1,N_a,N_z,N_j] with no d, [N_d,N_a,N_z,N_j] with d')
        end
        if N_d==0
            if vfoptions.gridinterplayer==0
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_LS1_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions);
            else
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_LS1_GI1_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions);
            end
        else
            if vfoptions.gridinterplayer==0
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_LS1_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions);
            else
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_LS1_GI1_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions);
            end
        end
    else
        error('vfoptions.localsearch=1 is currently only implemented for one endogenous state, with z (no e, no semiz)')
    end
    % Both kinds of sweep return Policy in the same Kron form as the unrestricted raws, so the UnKron
    % below applies unchanged.
elseif vfoptions.divideandconquer==0
    if vfoptions.gridinterplayer==0
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else % vfoptions.gridinterplayer==1
        if isscalar(n_a)
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_GI1_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_GI1_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else % 2 endogenous states
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_GI2A_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_GI2A_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    end
else % vfoptions.divideandconquer==1
    if ~isscalar(n_a) && length(vfoptions.level1n)>1
        if vfoptions.level1n(2)>=n_a(2)
            vfoptions.level1n=vfoptions.level1n(1); % DC2A: only first endo dim is divide-and-conquered
        else
            error('With two endogenous states, can only do divide-and-conquer in the first endogenous state (not in both)')
        end
    end
    if vfoptions.gridinterplayer==0
        if isscalar(n_a)
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_DC1_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_DC1_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else % 2 endogenous states
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_DC2A_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_DC2A_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    else % vfoptions.gridinterplayer==1
        if isscalar(n_a)
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_DC1_GI1_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_DC1_GI1_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else % 2 endogenous states
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_DC2A_GI2A_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_DC2A_GI2A_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    end
end

%% Policy in transition paths
if isscalar(n_a)
    if N_d==0
        PolicyKron=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_a,N_a,N_z,N_j,vfoptions);
    else
        if vfoptions.gridinterplayer==0
            PolicyKron=UnKronPolicyIndexes1_FHorz_z(PolicyKron,[n_d,n_a],N_a,N_z,N_j,vfoptions);
        else
            PolicyKron=UnKronPolicyIndexes2_FHorz_z(PolicyKron,n_d,n_a,N_a,N_z,N_j,vfoptions);
        end
    end
else
    n_a1=n_a(1);
    n_a2=n_a(2:end);
    if N_d==0
        if vfoptions.gridinterplayer==0
            PolicyKron=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_a,N_a,N_z,N_j,vfoptions);
        else
            PolicyKron=UnKronPolicyIndexes2_FHorz_z(PolicyKron,n_a1,n_a2,N_a,N_z,N_j,vfoptions);
        end
    else
        if vfoptions.gridinterplayer==0
            PolicyKron=UnKronPolicyIndexes1_FHorz_z(PolicyKron,[n_d,n_a],N_a,N_z,N_j,vfoptions);
        else
            PolicyKron=UnKronPolicyIndexes3_FHorz_z(PolicyKron,n_d,n_a1,n_a2,N_a,N_z,N_j,vfoptions);
        end
    end
end

end
