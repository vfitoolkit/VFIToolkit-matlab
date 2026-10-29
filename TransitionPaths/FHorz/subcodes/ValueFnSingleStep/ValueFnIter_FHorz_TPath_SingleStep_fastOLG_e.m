function [VKron, PolicyKron, aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_e(VKron,n_d,n_a,n_z,n_e,N_j,d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
aprimeReferencePolicyNew=[]; % the third output, filled only by local search and seed sweeps (see ValueFnIter_FHorz_TPath_SingleStep)
% vfoptions.EVpre=1 (the Matched Expectations Path) is only implemented in the divide-and-conquer raws for
% this model: the other raws always build the EVpre==0 expectation (append a zero j=N_j slice) and would
% silently solve with the wrong EV. Refused rather than ignored. The Matched Expectations Path itself only
% calls the fastOLG dispatcher with z and no e, so nothing currently gets here with EVpre=1.
if isfield(vfoptions,'EVpre')
    if vfoptions.EVpre==1 && vfoptions.divideandconquer==0
        error('vfoptions.EVpre=1 (Matched Expectations Path) is only implemented with vfoptions.divideandconquer=1 for this model')
    end
end
% The VKron input is next period value fn, the VKron output is this period.
% 'fastOLG' just means parallelizing across all of the "ages" (j) at once.

% V is done as (a,j)-by-z-by-e [this form makes the expectations easier]
% Policy is done as a-by-j-by-z-by-e [this form is easier later, and easier for handling DC1]
% (fastOLG requires swapping order of j and z)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);
% N_z=0 is handled elsewhere
% N_e=0 is handled elsewhere

% Local search: see ValueFnIter_FHorz_TPath_SingleStep, which this mirrors for fastOLG with z and e.
% The reference holds one aprime index per (d,a,z,e,j), in the raws' own Policy order after the leading N_dref.
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
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_nod_e_raw(VKron,n_a, n_z,n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                aprimeReferencePolicyNew=reshape(PolicyKron,1,[]); % no d, no GI: Policy IS the argmax
            else
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LSseed_e_raw(VKron,n_d,n_a,n_z,n_e, N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_d==0
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI1_nod_e_raw(VKron,n_a, n_z,n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI1_e_raw(VKron,n_d,n_a,n_z,n_e, N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    else
        if vfoptions.gridinterplayer==0
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_nod_e_raw(VKron,n_a, n_z,n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                aprimeReferencePolicyNew=reshape(PolicyKron,1,[]); % no d, no GI: Policy IS the argmax
            else
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_e_raw(VKron,n_d,n_a,n_z,n_e, N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_d==0
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_GI1_nod_e_raw(VKron,n_a, n_z,n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_GI1_e_raw(VKron,n_d,n_a,n_z,n_e, N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
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
        aprimeReferencePolicy=repmat((1:1:N_a),[N_dref,1,N_z*N_e*N_j]);
    elseif ~(size(aprimeReferencePolicy,1)==N_dref && numel(aprimeReferencePolicy)==N_dref*N_a*N_z*N_e*N_j)
        error('aprimeReferencePolicy must hold one aprime index per state the search conditions on: N_dref-by-(N_a*N_z*N_e*N_j), N_dref=1 with no d and N_d with d')
    end
    if N_d==0
        if vfoptions.gridinterplayer==0
            [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LS1_nod_e_raw(VKron,n_a, n_z,n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions);
        else
            [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LS1_GI1_nod_e_raw(VKron,n_a, n_z,n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions);
        end
    else
        if vfoptions.gridinterplayer==0
            [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LS1_e_raw(VKron,n_d,n_a,n_z,n_e, N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions);
        else
            [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_LS1_GI1_e_raw(VKron,n_d,n_a,n_z,n_e, N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions);
        end
    end
    % Both kinds of sweep return Policy in the same Kron form as the unrestricted raws
elseif strcmp(vfoptions.exoticpreferences,'None')
    if vfoptions.divideandconquer==0
        if vfoptions.gridinterplayer==0
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_nod_e_raw(VKron,n_a, n_z, n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_e_raw(VKron,n_d,n_a,n_z, n_e, N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else % vfoptions.gridinterplayer==1
            if isscalar(n_a)
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI1_nod_e_raw(VKron,n_a, n_z, n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI1_e_raw(VKron,n_d,n_a,n_z, n_e, N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else % 2 endogenous states
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI2A_nod_e_raw(VKron,n_a, n_z, n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_GI2A_e_raw(VKron,n_d,n_a,n_z, n_e, N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
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
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_nod_e_raw(VKron,n_a, n_z, n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_e_raw(VKron,n_d,n_a,n_z, n_e, N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else % 2 endogenous states
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC2A_nod_e_raw(VKron,n_a, n_z, n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC2A_e_raw(VKron,n_d,n_a,n_z, n_e, N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
        else % vfoptions.gridinterplayer==1
            if isscalar(n_a)
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_GI1_nod_e_raw(VKron,n_a, n_z, n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC1_GI1_e_raw(VKron,n_d,n_a,n_z, n_e, N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else % 2 endogenous states
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC2A_GI2A_nod_e_raw(VKron,n_a, n_z, n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_DC2A_GI2A_e_raw(VKron,n_d,n_a,n_z, n_e, N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
        end
    end
elseif strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
    % V slot = Valt (Naive) / Vunderbar (Sophisticated); Policy = QH choice; drop QH-only extras
    [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_QH_fastOLG_e(VKron,n_d,n_a,n_z,n_e,N_j,d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return % QH variant already un-Krons Policy
else
    error('Not yet implemented exoticpreferences for transtion paths (email me :)')
end



%% Policy in transition paths
% Note: The actual ordering of N_z,N_e,N_j is not relevant to how this command works, so can just mix them up. [as long as N_e not n_e, and N_z not n_z]
if isscalar(n_a)
    if N_d==0
        PolicyKron=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron,n_a,N_a,N_j,N_z,N_e,vfoptions);
    else
        if vfoptions.gridinterplayer==0
            PolicyKron=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron,[n_d,n_a],N_a,N_j,N_z,N_e,vfoptions);
        else
            PolicyKron=UnKronPolicyIndexes2_FHorz_z_e(PolicyKron,n_d,n_a,N_a,N_j,N_z,N_e,vfoptions);
        end
    end
else
    n_a1=n_a(1);
    n_a2=n_a(2:end);
    if N_d==0
        if vfoptions.gridinterplayer==0
            PolicyKron=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron,n_a,N_a,N_j,N_z,N_e,vfoptions);
        else
            PolicyKron=UnKronPolicyIndexes2_FHorz_z_e(PolicyKron,n_a1,n_a2,N_a,N_j,N_z,N_e,vfoptions);
        end
    else
        if vfoptions.gridinterplayer==0
            PolicyKron=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron,[n_d,n_a],N_a,N_j,N_z,N_e,vfoptions);
        else
            PolicyKron=UnKronPolicyIndexes3_FHorz_z_e(PolicyKron,n_d,n_a1,n_a2,N_a,N_j,N_z,N_e,vfoptions);
        end
    end
end


end
