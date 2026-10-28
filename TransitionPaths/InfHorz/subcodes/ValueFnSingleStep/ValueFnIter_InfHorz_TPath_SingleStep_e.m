function [VKron, PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_e(VKron,n_d,n_a,n_z,n_e,d_gridvals, a_grid, z_gridvals, e_gridvals, pi_z, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The VKron input is next period value fn, the VKron output is this period.
% With z and e: VKron is [N_a,N_z,N_e]. PolicyKron is [l_d+l_a,N_a,N_z,N_e] (+2 rows, L2 and L2flag, with the grid interpolation layer)

% vfoptions must be already fully set up (this command is for internal use only so it should be)

N_d=prod(n_d);

%%
if strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
    dbstack
    error('QuasiHyperbolic Preferences Not yet supported')
elseif strcmp(vfoptions.exoticpreferences,'EpsteinZin')
    dbstack
    error('EpsteinZin Preferences Not yet supported')
end
% Local search is refused upstream (TransitionPath_InfHorz and ValueFnOnTransPath_InfHorz) for anything but (a,z),
% so reaching here with it on is a bug in the caller, not a user error
if isfield(vfoptions,'localsearch') && vfoptions.localsearch~=0
    error('vfoptions.localsearch is only implemented for one endogenous state, with z (no e, no semiz)')
end

%% Solve the standard problem
if vfoptions.gridinterplayer==0
    if vfoptions.divideandconquer==0
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_nod_e_raw(VKron,n_a,n_z,n_e, a_grid, z_gridvals, e_gridvals, pi_z, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_e_raw(VKron,n_d,n_a,n_z,n_e, d_gridvals, a_grid, z_gridvals, e_gridvals, pi_z, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    elseif vfoptions.divideandconquer==1
        if isscalar(n_a)
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_nod_e_raw(VKron,n_a,n_z,n_e, a_grid, z_gridvals, e_gridvals, pi_z, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_e_raw(VKron,n_d,n_a,n_z,n_e, d_gridvals, a_grid, z_gridvals, e_gridvals, pi_z, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            error('Transition paths with z and e: divide-and-conquer is only implemented for one endogenous state')
        end
    end
else % vfoptions.gridinterplayer==1
    if ~isscalar(n_a)
        error('Transition paths with z and e: the grid interpolation layer is only implemented for one endogenous state')
    end
    if vfoptions.divideandconquer==0
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_GI1_nod_e_raw(VKron,n_a,n_z,n_e, a_grid, z_gridvals, e_gridvals, pi_z, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_GI1_e_raw(VKron,n_d,n_a,n_z,n_e, d_gridvals, a_grid, z_gridvals, e_gridvals, pi_z, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    elseif vfoptions.divideandconquer==1
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_GI1_nod_e_raw(VKron,n_a,n_z,n_e, a_grid, z_gridvals, e_gridvals, pi_z, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_GI1_e_raw(VKron,n_d,n_a,n_z,n_e, d_gridvals, a_grid, z_gridvals, e_gridvals, pi_z, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
end


end
