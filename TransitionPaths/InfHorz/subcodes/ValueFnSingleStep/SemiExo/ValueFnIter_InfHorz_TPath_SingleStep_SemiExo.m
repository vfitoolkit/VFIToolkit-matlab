function [VKron, PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_SemiExo(VKron,n_d,n_a,n_z,n_semiz,d_gridvals, a_grid, z_gridvals, semiz_gridvals, pi_z, pi_semiz, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% One backward step of the value fn on an InfHorz transition path, with semi-exogenous state semiz and markov z (no e).
% The VKron input is next period value fn, the VKron output is this period.
% VKron is [N_a,N_bothz], bothz=(semiz,z) with semiz indexing fastest.
% d_gridvals is the joint grid on d=(d1,d2), d1 varying first; the last vfoptions.l_dsemiz decision variables are d2,
% which drive the semiz transitions pi_semiz(semiz,semiz',d2) [N_semiz,N_semiz,N_d2] (this period's t->t+1 transition).
% semiz_gridvals is [N_semiz,l_semiz].
% PolicyKron is [l_d1+l_d2+l_a,N_a,N_bothz] (d1 rows, then d2 rows, then aprime), +2 rows (L2 index, L2flag) with the
% grid interpolation layer.

% vfoptions must be already fully set up (this command is for internal use only so it should be)

%%
if strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
    dbstack
    error('QuasiHyperbolic Preferences Not yet supported with semi-exogenous states on a transition path')
elseif strcmp(vfoptions.exoticpreferences,'EpsteinZin')
    dbstack
    error('EpsteinZin Preferences Not yet supported with semi-exogenous states on a transition path')
end
if ~isscalar(n_a)
    error('Transition paths with semi-exogenous states only allow a single endogenous state (cannot have length(n_a)>1)')
end
% Local search is refused upstream (TransitionPath_InfHorz and ValueFnOnTransPath_InfHorz) for anything but (a,z),
% so reaching here with it on is a bug in the caller, not a user error
if isfield(vfoptions,'localsearch') && vfoptions.localsearch~=0
    error('vfoptions.localsearch is not implemented with semi-exogenous states')
end
if ~isfield(vfoptions,'l_dsemiz')
    vfoptions.l_dsemiz=1; % by default, only one decision variable influences the semi-exogenous state
end

%% Split the decision variables into d1 and d2 (d2 are the last vfoptions.l_dsemiz)
l_d2=vfoptions.l_dsemiz;
n_d2=n_d(end-l_d2+1:end);
if length(n_d)>l_d2
    n_d1=n_d(1:end-l_d2);
else
    n_d1=0;
end
N_d1=prod(n_d1);
l_d1=length(n_d1);
if N_d1==0
    d1_gridvals=[];
    d2_gridvals=d_gridvals;
else
    d1_gridvals=d_gridvals(1:N_d1,1:l_d1);
    d2_gridvals=d_gridvals(1:N_d1:end,l_d1+1:end);
end

N_a=prod(n_a);
N_bothz=prod(n_semiz)*prod(n_z);

%% Solve
if vfoptions.divideandconquer==0
    if vfoptions.gridinterplayer==0
        if N_d1==0
            [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_SemiExo_nod1_raw(VKron,n_d2,n_a,n_z,n_semiz, d2_gridvals, a_grid, z_gridvals, semiz_gridvals, pi_z, pi_semiz, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_SemiExo_raw(VKron,n_d1,n_d2,n_a,n_z,n_semiz, d1_gridvals, d2_gridvals, a_grid, z_gridvals, semiz_gridvals, pi_z, pi_semiz, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else % vfoptions.gridinterplayer==1
        if N_d1==0
            [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_SemiExo_GI1_nod1_raw(VKron,n_d2,n_a,n_z,n_semiz, d2_gridvals, a_grid, z_gridvals, semiz_gridvals, pi_z, pi_semiz, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_SemiExo_GI1_raw(VKron,n_d1,n_d2,n_a,n_z,n_semiz, d1_gridvals, d2_gridvals, a_grid, z_gridvals, semiz_gridvals, pi_z, pi_semiz, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
else % vfoptions.divideandconquer==1
    if vfoptions.gridinterplayer==0
        if N_d1==0
            [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_SemiExo_DC1_nod1_raw(VKron,n_d2,n_a,n_z,n_semiz, d2_gridvals, a_grid, z_gridvals, semiz_gridvals, pi_z, pi_semiz, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_SemiExo_DC1_raw(VKron,n_d1,n_d2,n_a,n_z,n_semiz, d1_gridvals, d2_gridvals, a_grid, z_gridvals, semiz_gridvals, pi_z, pi_semiz, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else % vfoptions.gridinterplayer==1
        if N_d1==0
            [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_SemiExo_DC1_GI1_nod1_raw(VKron,n_d2,n_a,n_z,n_semiz, d2_gridvals, a_grid, z_gridvals, semiz_gridvals, pi_z, pi_semiz, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_InfHorz_TPath_SingleStep_SemiExo_DC1_GI1_raw(VKron,n_d1,n_d2,n_a,n_z,n_semiz, d1_gridvals, d2_gridvals, a_grid, z_gridvals, semiz_gridvals, pi_z, pi_semiz, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
end

%% UnKron Policy to rows
% The raws return Kron rows: [d2; aprime] or [d1; d2; aprime] (+ [L2; L2flag] with the grid interpolation layer).
% Because of how we have (semiz,z) together, use the _z commands with N_bothz (they pass the L2 and L2flag rows through).
if N_d1==0
    PolicyKron=UnKronPolicyIndexes2_z(PolicyKron, n_d2, n_a, N_a, N_bothz, vfoptions);
else
    PolicyKron=UnKronPolicyIndexes3_z(PolicyKron, n_d1, n_d2, n_a, N_a, N_bothz, vfoptions);
end

end
