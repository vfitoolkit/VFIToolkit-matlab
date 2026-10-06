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

%% Local search: aprime restricted to a window around aprimeReferencePolicy
% Sits ahead of the gridinterplayer and divideandconquer branches because it is an
% alternative to them, not a tier on top. Divide-and-conquer restricts the search a
% different way, so the combination is refused. The grid interpolation layer is NOT a
% tier on top either: the LS1_GI1 raw does the whole thing itself, putting the
% ngridinterp points between each consecutive pair of the window's coarse points and
% taking one max over them, with no coarse layer at all.
if vfoptions.localsearch==1
    if vfoptions.divideandconquer==1
        error('vfoptions.localsearch=1 cannot yet be combined with vfoptions.divideandconquer=1')
    end
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
