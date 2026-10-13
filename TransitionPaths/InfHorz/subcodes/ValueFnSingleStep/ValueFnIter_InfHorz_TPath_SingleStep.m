function [VKron, PolicyKron, aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep(VKron,n_d,n_a,n_z,d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions)
% The VKron input is next period value fn, the VKron output is this period.
%
% aprimeReferencePolicyNew, the third output, is the reference a later call could be given: the
% aprime index chosen for each state the search conditions on. Only the local search produces it,
% because only the local search computes an aprime optimum PER d; on every other path it comes back
% empty. That is not a gap to be filled by the unrestricted raws -- a chain of restricted steps can
% be seeded by one wide-window local search pass, which does produce it.
aprimeReferencePolicyNew=[];

% vfoptions must be already fully set up (this command is for internal use only so it should be)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

%%
if strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
    dbstack
    error('QuasiHyperbolic Preferences Not yet supported')
elseif strcmp(vfoptions.exoticpreferences,'EpsteinZin')
    dbstack
    error('EpsteinZin Preferences Not yet supported')
end

% localsearchseed is INTERNAL: Step1 sets it, no user does, so it is defaulted here rather than in
% the commands, and it is defaulted ABOVE ITS FIRST READ. The seed block below is the first read, and
% MATLAB evaluates it before any later default could run, so a caller that does not set the field --
% every caller that does not come through Step1, the test bank included -- errors on a missing field
% name instead of taking the ordinary path. The same ordering mistake was made once before with
% transpathoptions.localsearchdiagnostic in TransitionPath_InfHorz_shooting.
if ~isfield(vfoptions,'localsearchseed')
    vfoptions.localsearchseed=0;
end

%% The SEED sweep: unrestricted, but returning a reference for vfoptions.localsearch
% Set by Step1 on the standard sweeps of a localsearch solve -- iteration 1 and every verification
% sweep -- never by a user. Those sweeps have to leave behind a reference for the NEXT restricted
% sweep, and with a d that means one coarse aprime per (d,a,z) and not just for the d they chose.
%
% IT HONOURS vfoptions.divideandconquer. This was first built ignoring it, on the argument that a
% per-d reference has to come from the raw and the DC family would be too many raws to touch. That
% argument was wrong twice over: DC is a MEMORY option here, not just a speed one, so ignoring it
% means a model that needs DC to be solvable at all cannot use local search -- and the DC raws turned
% out to be nearly free, because DC1_GI1 and DC1_GI1_nod already build a per-d midpoint over every a
% for their own interpolation layer, DC1_nod has nothing folded so its Policy already is the argmax,
% and only DC1_raw needed real work.
%
% Each branch returns a reference built the same way: THE COARSE ARGMAX OF THE UNRESTRICTED SEARCH.
% Without the interpolation layer that is just the chosen aprime (per d, where there is a d); with
% it, it is the standard GI raw's own midpoint, which is the unrestricted version of exactly what
% the restricted GI raw's first layer produces. So the reference means one thing whichever sweep
% built it, which it did not when the standard sweep's reference was converted from Policy by
% rounding a fine answer to the nearest coarse point.
if vfoptions.localsearchseed==1
    if isscalar(n_a) && N_z>0
        % IT HONOURS vfoptions.divideandconquer, which matters because DC is a MEMORY option here and
        % not just a speed one. The seed sweep is the UNRESTRICTED one: with a d, no interpolation
        % layer, N_d=51, N_a=1001 and N_z=5 its return matrix is [51,1001,1001,5], about 2 GB, where
        % DC at level1n=31 makes it [51,1001,31,5], about 63 MB. Ignoring DC here would mean a model
        % that needs DC to be solvable at all cannot use local search -- which is exactly the size of
        % model local search is for.
        % Every branch returns the same object, THE COARSE ARGMAX OF THE UNRESTRICTED SEARCH, so the
        % reference means one thing whichever raw produced it:
        %   no d, no GI: nothing is folded and there is no interpolation layer, so Policy's aprime
        %     channel IS the argmax. True of the DC raw as well, so neither needed changing.
        %   no d, GI and with d, GI: the raw's own midpoint, clamped to [2,N_a-1]. The DC+GI raws
        %     already built one over every a for their interpolation layer, so they were nearly free.
        %   with d, no GI: a per-d argmax, which no single max over a folded (d,aprime) dimension can
        %     give. LSseed_raw unfolds it; DC1_raw splits its level-2 max into one over aprime and
        %     then one over d, which also cut a reduction it was doing twice.
        if vfoptions.gridinterplayer==0
            if N_d==0
                if vfoptions.divideandconquer==0
                    [VKron,Policy]=ValueFnIter_InfHorz_TPath_SingleStep_nod_raw(VKron,n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron,Policy]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_nod_raw(VKron,n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
                PolicyKron=reshape(Policy,[size(Policy,1),N_a,N_z]);
                aprimeReferencePolicyNew=PolicyKron(1,:,:);
            else
                if vfoptions.divideandconquer==0
                    [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LSseed_raw(VKron,n_d,n_a,n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_raw(VKron,n_d,n_a,n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
        else
            if N_d==0
                if vfoptions.divideandconquer==0
                    [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_GI1_nod_raw(VKron,n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_GI1_nod_raw(VKron,n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else
                if vfoptions.divideandconquer==0
                    [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_GI1_raw(VKron,n_d,n_a,n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_GI1_raw(VKron,n_d,n_a,n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
        end
    else
        error('vfoptions.localsearch=1 is currently only implemented for one endogenous state, with z (no e, no semiz)')
    end
    return
end

%% Local search: aprime restricted to a window around aprimeReferencePolicy
% Sits ahead of the gridinterplayer and divideandconquer branches because it is an
% alternative to them, not a tier on top. Divide-and-conquer restricts the search a
% different way, so the combination is refused.
% The grid interpolation layer is NOT a tier on top either: the LS1_GI1 raw does the whole thing
% itself, maxing over the window's coarse points, clamping that argmax to the window interior, and
% running the interpolation layer one coarse cell either side -- which covers exactly the window.
% Its coarse layer also hands back the new reference directly, with no fine answer to compress.
% ONLY 0 AND 1 ARE MEANINGFUL HERE. ValueFnOnTransPath_InfHorz reads a THIRD value: it ignores
% localsearch=1 with a warning, because it has no verification sweep to certify a restricted answer
% against, and takes localsearch=2 as "force it anyway". A vfoptions struct carrying that 2 can
% easily reach this command instead -- the test bank hands related structs to both -- and 2 would
% otherwise fall through every ==1 test below, running an unrestricted search AND skipping the whole
% localsearch validation block: no nlocalsearch requirement, no grid-size check, none of the
% refusals. Silently. So it is refused by name.
if vfoptions.localsearch~=0 && vfoptions.localsearch~=1
    error('vfoptions.localsearch must be 0 or 1 here; the value 2 is only meaningful to ValueFnOnTransPath_InfHorz, which uses it to force a restricted search it would otherwise ignore')
end

if vfoptions.localsearch==1
    % DIVIDE-AND-CONQUER IS IGNORED ON A RESTRICTED SWEEP, silently, exactly as lowmemory is. This
    % used to be an ERROR, which made the whole combination unusable: the SEED sweeps honour DC
    % because they are the full sweeps and DC is what keeps them in memory, so a user who sets
    % divideandconquer=1 for that reason -- the only reason to set it on a big model -- hit the
    % refusal on the first restricted sweep and could not solve at all. The two options do not
    % compose on a restricted sweep (the window IS the restriction, and DC restricts the same search
    % a different way), but that is a reason to ignore DC here, not to refuse the solve.
    % So on a localsearch solve: full sweeps use DC and lowmemory as given, restricted sweeps use
    % neither. Both are ignored in silence because that is the intended behaviour and a warning per
    % period would be noise; it is documented on the options instead.
    vfoptions.divideandconquer=0;
    if isscalar(n_a) && N_z>0
        % The reference holds the aprime index at the centre of each window, indexed by everything
        % the search conditions on: (a,z) with no d, and (d,a,z) with one. N_dref is 1 in the first
        % case and N_d in the second, which is the whole difference between the two shapes.
        N_dref=max(N_d,1);
        if isempty(aprimeReferencePolicy)
            % Default reference: the current a index, so the window sits around staying put. With d
            % that is the same aprime whatever d is being considered, which is as much as a default
            % can say without having solved something first.
            aprimeReferencePolicy=repmat((1:1:N_a),[N_dref,1,N_z]);
        elseif ~(size(aprimeReferencePolicy,1)==N_dref && size(aprimeReferencePolicy,2)==N_a && numel(aprimeReferencePolicy)==N_dref*N_a*N_z)
            % numel rather than size(...,3) because MATLAB drops a trailing singleton when N_z==1
            error('aprimeReferencePolicy must hold one aprime index per state the search conditions on: [1,N_a,N_z] with no d, [N_d,N_a,N_z] with d')
        end
        % LOWMEMORY IS IGNORED ON A RESTRICTED SWEEP TOO, SILENTLY AND BY DESIGN. lowmemory exists because
        % the unrestricted return matrix does not fit; the restricted one is a window and does fit, so
        % looping over z there would cost launches for nothing. The SEED sweeps above are the full
        % sweeps and they honour it, which is the point: a model that needs lowmemory to solve at all
        % can now use local search, where before the raws refused the combination outright.
        % THE ONE CASE THIS GETS WRONG is left to fail. The ratchet's cap is floor((N_a-1)/2), where
        % 2*nlocalsearch+1 reaches N_a and the restricted matrix is the same size as the unrestricted
        % one, so a model that needed lowmemory will run out of memory there. The cap warning in
        % TransitionPath_InfHorz_shooting already says local search is strictly worse than not using
        % it at the cap; this just means the user finds out by running out of memory rather than by
        % reading it.
        vfoptions.lowmemory=0;
        if N_d==0
            if vfoptions.gridinterplayer==0
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LS1_nod_raw(VKron,n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions);
            else
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LS1_GI1_nod_raw(VKron,n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions);
            end
        else
            if vfoptions.gridinterplayer==0
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LS1_raw(VKron,n_d,n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions);
            else
                [VKron,PolicyKron,aprimeReferencePolicyNew]=ValueFnIter_InfHorz_TPath_SingleStep_LS1_GI1_raw(VKron,n_d,n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeReferencePolicy, vfoptions);
            end
        end
    else
        error('vfoptions.localsearch=1 is currently only implemented for one endogenous state, with z (no e, no semiz)')
    end
    return
end

%% Solve the standard problem
% Note: being infinite horizon, I don't imagine anyone will come here without z variable
if vfoptions.gridinterplayer==0
    if vfoptions.divideandconquer==0
        if N_d==0
            [VKron,Policy]=ValueFnIter_InfHorz_TPath_SingleStep_nod_raw(VKron,n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            PolicyKron=reshape(Policy,[size(Policy,1),N_a,N_z]);
        else
            [VKron, PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_raw(VKron,n_d,n_a,n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    elseif vfoptions.divideandconquer==1
        if isscalar(n_a)
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_nod_raw(VKron,n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_raw(VKron,n_d,n_a,n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        elseif length(n_a)>1
            if vfoptions.level1n(2)==n_a(2) % Don't bother with divide-and-conquer on the endogenous states after the first
                vfoptions.level1n=vfoptions.level1n(1); % Only first one is relevant for DC2A
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_DC2A_nod_raw(VKron,n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_DC2A_raw(VKron,n_d,n_a,n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else % Do divide-and-conquer for more than just the first endogenous state
                error('With more than one endogenous state, can only do divide-and-conquer in the first endogenous state')
            end
        end
    end
else % vfoptions.gridinterplayer==1
    if vfoptions.divideandconquer==0
        if isscalar(n_a)
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_GI1_nod_raw(VKron,n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_GI1_raw(VKron,n_d,n_a,n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        elseif length(n_a)>1
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_GI2A_nod_raw(VKron,n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_GI2A_raw(VKron,n_d,n_a,n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    elseif vfoptions.divideandconquer==1
        if isscalar(n_a)
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_GI1_nod_raw(VKron,n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_GI1_raw(VKron,n_d,n_a,n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        elseif length(n_a)>1
            if vfoptions.level1n(2)==n_a(2) % Don't bother with divide-and-conquer on the endogenous states after the first
                vfoptions.level1n=vfoptions.level1n(1); % Only first one is relevant for DC2A
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_DC2A_GI2A_nod_raw(VKron,n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_DC2A_GI2A_raw(VKron,n_d,n_a,n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else
                error('With more than one endogenous state, can only do divide-and-conquer in the first endogenous state')
            end
        end
    end
end



% if strcmp(vfoptions.solnmethod,'purediscretization_refinement')
%     % COMMENT: testing a transition in model of Pijoan-Mas (2006) it
%     % seems refinement is slower for transitions, so this is never
%     % really used for anything.
%     [VKron, PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_Refine_raw(VKron,n_d,n_a,n_z, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
% end

end
