function [VKron, PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep(VKron,n_d,n_a,n_z,d_grid, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The VKron input is next period value fn, the VKron output is this period.

% vfoptions must be already fully set up (this command is for internal use only so it should be)

N_d=prod(n_d);
% N_a=prod(n_a);
% N_z=prod(n_z);

%% 
if strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
    dbstack
    error('QuasiHyperbolic Preferences Not yet supported')
elseif strcmp(vfoptions.exoticpreferences,'EpsteinZin')
    dbstack
    error('EpsteinZin Preferences Not yet supported')
end

%% Solve the standard problem
% Note: being infinite horizon, I don't imagine anyone will come here without z variable
if vfoptions.gridinterplayer==0
    if vfoptions.divideandconquer==0
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_nod_raw(VKron,n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_raw(VKron,n_d,n_a,n_z, d_grid, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    elseif vfoptions.divideandconquer==1
        if isscalar(n_a)
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_nod_raw(VKron,n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_raw(VKron,n_d,n_a,n_z, d_grid, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        elseif length(n_a)==2
            if vfoptions.level1n(2)==n_a(2) % Don't bother with divide-and-conquer on the second endogenous state        vfoptions.level1n=vfoptions.level1n(1); % Only first one is relevant for DC2B
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_DC2B_nod_raw(VKron,n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_DC2B_raw(VKron,n_d,n_a,n_z, d_grid, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else % Do divide-and-conquer for both endogenous states
                error('With two endogenous states, can only do divide-and-conquer in the first endogenous state (not in both)')
                % if N_d==0
                %     [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_DC2_nod_raw(VKron,n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                % else
                %     [VKron, PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_DC2_raw(VKron,n_d,n_a,n_z, d_grid, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                % end
            end
        else
            error('Cannot use vfoptions.divideandconquer with more than two endogenous states (you have length(n_a)>2)')
        end
    end
else % vfoptions.gridinterplayer==1
    if vfoptions.divideandconquer==0
        if N_d==0
            error('Have not yet implemented combo of vfoptions.gridinterplayer=1 with vfoptions.divideandconquer=0')
            % [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_GI_nod_raw(VKron,n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            error('Have not yet implemented combo of vfoptions.gridinterplayer=1 with vfoptions.divideandconquer=0')
            % [VKron, PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_GI_raw(VKron,n_d,n_a,n_z, d_grid, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    elseif vfoptions.divideandconquer==1
        if isscalar(n_a)
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_GI_nod_raw(VKron,n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_GI_raw(VKron,n_d,n_a,n_z, d_grid, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        elseif length(n_a)==2
            if vfoptions.level1n(2)==n_a(2) % Don't bother with divide-and-conquer on the second endogenous state
                vfoptions.level1n=vfoptions.level1n(1); % Only first one is relevant for DC2B
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_DC2B_GI_nod_raw(VKron,n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_DC2B_GI_raw(VKron,n_d,n_a,n_z, d_grid, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else
                error('With two endogenous states, can only do divide-and-conquer in the first endogenous state (not in both)')
            end
        end
    end
end



% if strcmp(vfoptions.solnmethod,'purediscretization_refinement')
%     % COMMENT: testing a transition in model of Pijoan-Mas (2006) it
%     % seems refirement is slower for transtions, so this is never
%     % really used for anything.
%     [VKron, PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_Refine_raw(VKron,n_d,n_a,n_z, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
% end

end
