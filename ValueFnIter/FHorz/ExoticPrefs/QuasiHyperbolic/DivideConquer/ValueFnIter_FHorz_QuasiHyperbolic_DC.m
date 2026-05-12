function [V1, Policy,Valt]=ValueFnIter_FHorz_QuasiHyperbolic_DC(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Quasi-hyperbolic discounting variant of ValueFnIter_FHorz_DC1_raw.
% No d (decision) variables. Uses divide-and-conquer on a (GPU, parallel==2 only).
%
% Interpretation of output differs by Naive/Sophisticated
% Naive:         {Vtilde, Policy, V}
% Sophisticated: {Vhat,  Policy, Vunderbar}
%
% DiscountFactorParamNames is the standard discount factor beta
% vfoptions.QHadditionaldiscount.gives the name of the beta_0 is the additional discount factor parameter

N_d=prod(n_d);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

if ~isfield(vfoptions,'level1n')
    if isscalar(n_a)
        vfoptions.level1n=max(ceil(n_a(1)/50),5);
        if n_a(1)<5
            error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
        end
    elseif length(n_a)==2
        vfoptions.level1n=[max(ceil(sqrt(n_a(1))),5),n_a(2)];
        if n_a(1)<5
            error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
        end
    end
    if vfoptions.verbose==1
        fprintf('Suggestion: When using vfoptions.divideandconquer it will be faster or slower if you set different values of vfoptions.level1n (for smaller models 7 or 9 is good, but for larger models something 15 or 21 can be better) \n')
    end
else
    if ~isscalar(n_a) && isscalar(vfoptions.level1n)
        vfoptions.level1n=[vfoptions.level1n,n_a(2:end)];
    end
end
vfoptions.level1n=min(vfoptions.level1n,n_a);

%%
if vfoptions.gridinterplayer==1
    [V1, Policy,Valt]==ValueFnIter_FHorz_QuasiHyperbolic_DC_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return
end


%%
if isscalar(n_a)
    if strcmp(vfoptions.quasi_hyperbolic,'Naive') % Output: [Vtilde,Policy,V]
        if N_e==0
            if N_z==0
                if N_d==0
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicN_DC1_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V1Kron, PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicN_DC1_noz_raw(n_d,n_a, N_j, d_gridvals, a_grid, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else
                if N_d==0
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicN_DC1_nod_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V1Kron, PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicN_DC1_raw(n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
            % Policy without d
            PolicyKron=shiftdim(PolicyKron,-1);
        else
            if N_z==0
                if N_d==0
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicN_DC1_nod_noz_e_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicN_DC1_noz_e_raw(n_d,n_a, vfoptions.n_e, N_j, d_gridvals, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else
                if N_d==0
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicN_DC1_nod_e_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicN_DC1_e_raw(n_d,n_a, n_z, vfoptions.n_e, N_j, d_gridvals, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
        end
    elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated') % Output: [Vunderbar,Policy,Vhat]
        if N_e==0
            if N_z==0
                if N_d==0
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicS_DC1_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V1Kron, PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicS_DC1_noz_raw(n_d,n_a, N_j, d_gridvals, a_grid, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else
                if N_d==0
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicS_DC1_nod_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V1Kron, PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicS_DC1_raw(n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
            % Policy without d
            PolicyKron=shiftdim(PolicyKron,-1);
        else
            if N_z==0
                if N_d==0
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicS_DC1_nod_noz_e_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicS_DC1_noz_e_raw(n_d,n_a, vfoptions.n_e, N_j, d_gridvals, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else
                if N_d==0
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicS_DC1_nod_e_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicS_DC1_e_raw(n_d,n_a, n_z, vfoptions.n_e, N_j, d_gridvals, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
        end
    end
else
    error('ValueFnIter_FHorz_DC_QuasiHyperbolic currently only supports scalar n_a (one endogenous state)')
end

%% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if vfoptions.outputkron==0
    if N_z==0
        V1=reshape(V1Kron,[n_a,N_j]);
        Policy=UnKronPolicyIndexes_Case1_FHorz_noz(PolicyKron, n_d, n_a, N_j, vfoptions);
        Valt=reshape(ValtKron,[n_a,N_j]);
    else
        V1=reshape(V1Kron,[n_a,n_z,N_j]);
        Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, n_z, N_j, vfoptions);
        Valt=reshape(ValtKron,[n_a,n_z,N_j]);
    end
else
    V1=V1Kron;
    Policy=PolicyKron;
    Valt=ValtKron;
end


end
