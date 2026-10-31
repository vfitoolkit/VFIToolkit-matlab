function [VKron, PolicyKron, aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_noz_e(VKron,n_d,n_a,n_e,d_gridvals, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% The VKron input is next period value fn, the VKron output is this period.
% No z, with e: VKron is [N_a,N_e]. PolicyKron is [l_d+l_a,N_a,N_e] (+2 rows, L2 and L2flag, with the grid interpolation layer)
%
% aprimeReferencePolicyNew, the third output, is the local search reference, [N_dref,N_a,N_e] with N_dref=max(N_d,1): one
% aprime index per state the search conditions on, so per current e. Only the seed and the restricted sweeps produce
% it; on the standard path it comes back empty. See ValueFnIter_InfHorz_TPath_SingleStep for the z version.
aprimeReferencePolicyNew=[];

% vfoptions must be already fully set up (this command is for internal use only so it should be)

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

%%
if strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
    dbstack
    error('QuasiHyperbolic Preferences Not yet supported')
elseif strcmp(vfoptions.exoticpreferences,'EpsteinZin')
    dbstack
    error('EpsteinZin Preferences Not yet supported')
end

% localsearchseed is INTERNAL (Step1 sets it), so it is defaulted here, above its first read, as in the z dispatcher
if ~isfield(vfoptions,'localsearchseed')
    vfoptions.localsearchseed=0;
end
if ~isfield(vfoptions,'localsearch')
    vfoptions.localsearch=0;
end

%% The SEED sweep: unrestricted, but returning a reference for vfoptions.localsearch
% As in the z dispatcher: it honours divideandconquer and lowmemory, and every branch returns THE COARSE ARGMAX OF
% THE UNRESTRICTED SEARCH (Policy's aprime channel with no d and no GI; the raw's own clamped midpoint with GI; the
% per-d argmax with d and no GI, from LSseed_noz_e_raw or the split max in DC1_noz_e_raw).
if vfoptions.localsearchseed==1
    if ~isscalar(n_a)
        error('vfoptions.localsearch=1 is currently only implemented for one endogenous state')
    end
    if vfoptions.gridinterplayer==0
        if N_d==0
            if vfoptions.divideandconquer==0
                [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_nod_noz_e_raw(VKron,n_a,n_e, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_nod_noz_e_raw(VKron,n_a,n_e, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
            aprimeReferencePolicyNew=reshape(PolicyKron(1,:,:),[1,N_a,N_e]);
        else
            if vfoptions.divideandconquer==0
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LSseed_noz_e_raw(VKron,n_d,n_a,n_e, d_gridvals, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_noz_e_raw(VKron,n_d,n_a,n_e, d_gridvals, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    else
        if N_d==0
            if vfoptions.divideandconquer==0
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_GI1_nod_noz_e_raw(VKron,n_a,n_e, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_GI1_nod_noz_e_raw(VKron,n_a,n_e, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if vfoptions.divideandconquer==0
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_GI1_noz_e_raw(VKron,n_d,n_a,n_e, d_gridvals, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_GI1_noz_e_raw(VKron,n_d,n_a,n_e, d_gridvals, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    end
    return
end

%% Local search: aprime restricted to a window around aprimeReferencePolicy
% Only 0 and 1 are meaningful here; 2 is ValueFnOnTransPath_InfHorz's 'force it' value and is refused by name, as in
% the z dispatcher (it would otherwise fall through to an unrestricted search in silence).
if vfoptions.localsearch~=0 && vfoptions.localsearch~=1
    error('vfoptions.localsearch must be 0 or 1 here; the value 2 is only meaningful to ValueFnOnTransPath_InfHorz, which uses it to force a restricted search it would otherwise ignore')
end

if vfoptions.localsearch==1
    if ~isscalar(n_a)
        error('vfoptions.localsearch=1 is currently only implemented for one endogenous state')
    end
    % Divide-and-conquer and lowmemory are ignored on a restricted sweep, silently and by design: the window is the
    % restriction and it fits in memory. The seed sweeps are the full sweeps and they honour both.
    vfoptions.divideandconquer=0;
    vfoptions.lowmemory=0;
    % One reference per state the search conditions on: (a,e) with no d, (d,a,e) with one
    N_dref=max(N_d,1);
    if isempty(aprimeReferencePolicy)
        % Default reference: the current a index, so the window sits around staying put
        aprimeReferencePolicy=repmat((1:1:N_a),[N_dref,1,N_e]);
    elseif ~(size(aprimeReferencePolicy,1)==N_dref && size(aprimeReferencePolicy,2)==N_a && numel(aprimeReferencePolicy)==N_dref*N_a*N_e)
        % numel rather than size(...,3) because MATLAB drops a trailing singleton
        error('aprimeReferencePolicy must hold one aprime index per state the search conditions on: [1,N_a,N_e] with no d, [N_d,N_a,N_e] with d')
    end
    if N_d==0
        if vfoptions.gridinterplayer==0
            [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LS1_nod_noz_e_raw(VKron,n_a,n_e, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions);
        else
            [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LS1_GI1_nod_noz_e_raw(VKron,n_a,n_e, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions);
        end
    else
        if vfoptions.gridinterplayer==0
            [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LS1_noz_e_raw(VKron,n_d,n_a,n_e, d_gridvals, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions);
        else
            [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LS1_GI1_noz_e_raw(VKron,n_d,n_a,n_e, d_gridvals, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions);
        end
    end
    return
end

%% Solve the standard problem
if vfoptions.gridinterplayer==0
    if vfoptions.divideandconquer==0
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_nod_noz_e_raw(VKron,n_a,n_e, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_noz_e_raw(VKron,n_d,n_a,n_e, d_gridvals, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    elseif vfoptions.divideandconquer==1
        if isscalar(n_a)
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_nod_noz_e_raw(VKron,n_a,n_e, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_noz_e_raw(VKron,n_d,n_a,n_e, d_gridvals, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            error('Transition paths without z (with e): divide-and-conquer is only implemented for one endogenous state')
        end
    end
else % vfoptions.gridinterplayer==1
    if ~isscalar(n_a)
        error('Transition paths without z (with e): the grid interpolation layer is only implemented for one endogenous state')
    end
    if vfoptions.divideandconquer==0
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_GI1_nod_noz_e_raw(VKron,n_a,n_e, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_GI1_noz_e_raw(VKron,n_d,n_a,n_e, d_gridvals, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    elseif vfoptions.divideandconquer==1
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_GI1_nod_noz_e_raw(VKron,n_a,n_e, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_GI1_noz_e_raw(VKron,n_d,n_a,n_e, d_gridvals, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
end


end
