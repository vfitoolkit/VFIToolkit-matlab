function varargout=ValueFnOnTransPath_FHorz_QuasiHyperbolic(PricePath, ParamPath, T, V_final, Policy_final, Parameters, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, pi_z, DiscountFactorParamNames, ReturnFn, transpathoptions, vfoptions)
% Quasi-Hyperbolic discounting on FHorz transition path (compute-only, no GE).
%
% Output convention (mirrors ValueFnIter_FHorz_QuasiHyperbolic):
%   Naive:         varargout = {VPath, PolicyPath, ValtPath, PolicyaltPath}
%     VPath         = Vtilde-path      (QH-discounted value the agent sees each period)
%     PolicyPath    = QH-optimal choice (argmax of Vtilde-step)
%     ValtPath      = Valt-path        (exp-discounter continuation value carried backwards)
%     PolicyaltPath = exp-discounter argmax (needed by ValueFnFromPolicy for Naive)
%   Sophisticated: varargout = {VPath, PolicyPath, ValtPath}
%     VPath         = Vhat-path        (QH-discounted from current self's perspective)
%     PolicyPath    = equilibrium choice (argmax of Vhat-step)
%     ValtPath      = Vunderbar-path   (realised continuation under future selves' QH choices)
%
% V_final input semantics: Naive => Valt_final; Sophisticated => Vunderbar_final.
% Both are the 3rd output of ValueFnIter_Case1_FHorz with vfoptions.exoticpreferences='QuasiHyperbolic'.

error('ValueFnOnTransPath_FHorz_QuasiHyperbolic: not yet implemented (Phase 1 plumbing only; QH single-step raws and per-config dispatch land in Phase 4/5)')

end
