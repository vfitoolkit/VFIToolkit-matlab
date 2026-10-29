function [VKron, PolicyKron, aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_noz_e(VKron,n_d,n_a,n_e,N_j,d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% The VKron input is next period value fn, the VKron output is this period.
aprimeReferencePolicyNew=[]; % the third output, filled only by local search and seed sweeps (see ValueFnIter_FHorz_TPath_SingleStep)

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

if strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
    % V slot = Valt (Naive) / Vunderbar (Sophisticated); Policy = QH choice; drop QH-only extras
    [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_QH_noz_e(VKron,n_d,n_a,n_e,N_j,d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return % QH variant already un-Krons Policy
end

%% If get to here then not using exoticpreferences nor StateDependentVariables_z
% N_z==0 is handled by a different command
% Local search: see ValueFnIter_FHorz_TPath_SingleStep, which this mirrors for the model without z and with e.
% The reference holds one aprime index per (d,a,e,j), in the raws' own Policy order after the leading N_dref.
if ~isfield(vfoptions,'localsearch')
    vfoptions.localsearch=0;
end
if ~isfield(vfoptions,'localsearchseed')
    vfoptions.localsearchseed=0;
end
if vfoptions.localsearch~=0 && vfoptions.localsearch~=1
    error('vfoptions.localsearch must be 0 or 1 here; the value 2 is only meaningful to ValueFnOnTransPath_Case1_FHorz, which uses it to force a restricted search it would otherwise ignore')
end
if (vfoptions.localsearch==1 || vfoptions.localsearchseed==1) && ~strcmp(vfoptions.exoticpreferences,'None')
    error('vfoptions.localsearch=1 is not implemented with exoticpreferences')
end
if (vfoptions.localsearch==1 || vfoptions.localsearchseed==1) && ~isscalar(n_a)
    error('vfoptions.localsearch=1 is currently only implemented for one endogenous state')
end
aprimeReferencePolicyNew=[];
N_dref=max(N_d,1);
if vfoptions.localsearchseed==1 && vfoptions.localsearch==0
    %% The SEED sweep: unrestricted, honouring divideandconquer, returning the coarse argmax per d
    if vfoptions.divideandconquer==0
        if vfoptions.gridinterplayer==0
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_nod_noz_e_raw(VKron,n_a,n_e, N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                aprimeReferencePolicyNew=reshape(PolicyKron,1,[]); % no d, no GI: Policy IS the argmax
            else
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_LSseed_noz_e_raw(VKron,n_d,n_a,n_e, N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_d==0
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_GI1_nod_noz_e_raw(VKron,n_a,n_e, N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_GI1_noz_e_raw(VKron,n_d,n_a,n_e, N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    else
        if vfoptions.gridinterplayer==0
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_DC1_nod_noz_e_raw(VKron,n_a,n_e, N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                aprimeReferencePolicyNew=reshape(PolicyKron,1,[]); % no d, no GI: Policy IS the argmax
            else
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_DC1_noz_e_raw(VKron,n_d,n_a,n_e, N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_d==0
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_DC1_GI1_nod_noz_e_raw(VKron,n_a,n_e, N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_DC1_GI1_noz_e_raw(VKron,n_d,n_a,n_e, N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    end
elseif vfoptions.localsearch==1
    %% Local search: aprime restricted to a window around aprimeReferencePolicy
    % lowmemory and divideandconquer are ignored on a restricted sweep, silently and by design
    vfoptions.divideandconquer=0;
    vfoptions.lowmemory=0;
    if isempty(aprimeReferencePolicy)
        % Default reference: stay put. Only ValueFnOnTransPath_Case1_FHorz at localsearch=2 gets here.
        aprimeReferencePolicy=repmat((1:1:N_a),[N_dref,1,N_e*N_j]);
    elseif ~(size(aprimeReferencePolicy,1)==N_dref && numel(aprimeReferencePolicy)==N_dref*N_a*N_e*N_j)
        error('aprimeReferencePolicy must hold one aprime index per state the search conditions on: N_dref-by-(N_a*N_e*N_j), N_dref=1 with no d and N_d with d')
    end
    if N_d==0
        if vfoptions.gridinterplayer==0
            [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_LS1_nod_noz_e_raw(VKron,n_a,n_e, N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions);
        else
            [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_LS1_GI1_nod_noz_e_raw(VKron,n_a,n_e, N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions);
        end
    else
        if vfoptions.gridinterplayer==0
            [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_LS1_noz_e_raw(VKron,n_d,n_a,n_e, N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions);
        else
            [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_LS1_GI1_noz_e_raw(VKron,n_d,n_a,n_e, N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions);
        end
    end
    % Both kinds of sweep return Policy in the same Kron form as the unrestricted raws
elseif vfoptions.divideandconquer==0
    if vfoptions.gridinterplayer==0
        if N_d==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_nod_noz_e_raw(VKron,n_a,n_e, N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_noz_e_raw(VKron,n_d,n_a,n_e, N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else % vfoptions.gridinterplayer==1
        if isscalar(n_a)
            if N_d==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_GI1_nod_noz_e_raw(VKron,n_a,n_e, N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_GI1_noz_e_raw(VKron,n_d,n_a,n_e, N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else % 2 endogenous states
            if N_d==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_GI2A_nod_noz_e_raw(VKron,n_a,n_e, N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_GI2A_noz_e_raw(VKron,n_d,n_a,n_e, N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    end
else % vfoptions.divideandconquer==1
    if ~isscalar(n_a) && length(vfoptions.level1n)>1
        if vfoptions.level1n(2)>=n_a(2)
            vfoptions.level1n=vfoptions.level1n(1);
        else
            error('With two endogenous states, can only do divide-and-conquer in the first endogenous state (not in both)')
        end
    end
    if vfoptions.gridinterplayer==0
        if isscalar(n_a)
            if N_d==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_DC1_nod_noz_e_raw(VKron,n_a,n_e, N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_DC1_noz_e_raw(VKron,n_d,n_a,n_e, N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else % 2 endogenous states
            if N_d==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_DC2A_nod_noz_e_raw(VKron,n_a,n_e, N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_DC2A_noz_e_raw(VKron,n_d,n_a,n_e, N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    else % vfoptions.gridinterplayer==1
        if isscalar(n_a)
            if N_d==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_DC1_GI1_nod_noz_e_raw(VKron,n_a,n_e, N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_DC1_GI1_noz_e_raw(VKron,n_d,n_a,n_e, N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else % 2 endogenous states
            if N_d==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_DC2A_GI2A_nod_noz_e_raw(VKron,n_a,n_e, N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_DC2A_GI2A_noz_e_raw(VKron,n_d,n_a,n_e, N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    end
end

%% Policy in transition paths
% Treat the e-dim as the z-dim (no z) and use the _FHorz_z UnKron family
if isscalar(n_a)
    if N_d==0
        PolicyKron=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_a,N_a,N_e,N_j,vfoptions);
    else
        if vfoptions.gridinterplayer==0
            PolicyKron=UnKronPolicyIndexes1_FHorz_z(PolicyKron,[n_d,n_a],N_a,N_e,N_j,vfoptions);
        else
            PolicyKron=UnKronPolicyIndexes2_FHorz_z(PolicyKron,n_d,n_a,N_a,N_e,N_j,vfoptions);
        end
    end
else
    n_a1=n_a(1);
    n_a2=n_a(2:end);
    if N_d==0
        if vfoptions.gridinterplayer==0
            PolicyKron=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_a,N_a,N_e,N_j,vfoptions);
        else
            PolicyKron=UnKronPolicyIndexes2_FHorz_z(PolicyKron,n_a1,n_a2,N_a,N_e,N_j,vfoptions);
        end
    else
        if vfoptions.gridinterplayer==0
            PolicyKron=UnKronPolicyIndexes1_FHorz_z(PolicyKron,[n_d,n_a],N_a,N_e,N_j,vfoptions);
        else
            PolicyKron=UnKronPolicyIndexes3_FHorz_z(PolicyKron,n_d,n_a1,n_a2,N_a,N_e,N_j,vfoptions);
        end
    end
end

end
