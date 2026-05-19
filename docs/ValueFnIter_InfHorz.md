---
title: ValueFnIter_InfHorz
sidebar_label: ValueFnIter_InfHorz
description: Solve an infinite-horizon value function iteration problem and return the value function and policy.
---

# ValueFnIter_InfHorz

> Top-level dispatcher that solves an infinite-horizon value function iteration problem, returning the value function `V` and the policy index array `Policy` (typically `varargout=[V, Policy]`).

## Signature

```matlab
varargout = ValueFnIter_InfHorz( ...
    n_d, n_a, n_z, ...
    d_grid, a_grid, z_grid, pi_z, ...
    ReturnFn, Parameters, ...
    DiscountFactorParamNames, ReturnFnParamNames, ...
    vfoptions)
```

## Description

`ValueFnIter_InfHorz` is the entry point for solving infinite-horizon
value function iteration problems. It is built around decision variables `d`,
endogenous states `a`, and markov exogenous states `z`.
It accepts as inputs the grid sizes (`n_d`, `n_a`, `n_z`), the corresponding grids, the
exogenous Markov transition matrix `pi_z`, a return function `ReturnFn`, and parameter information, and iterates the Bellman operator to a fixed point, returning the value function and policy.

Other exogenous states: semi-endogenous shocks where `pi_z` depends on the current endogenous state (`SemiEndogShockFn`). i.i.d. shocks `e` are accepted as `vfoptions.n_e`/`e_grid`/`pi_e` by the experience-asset and inheritance-asset branches.

Alternative endogenous states: experience assets (`experienceasset`), inheritance assets (`inheritanceasset`), incremental endogenous states (`incrementaltype`).

Can handle exotic preferences: `QuasiHyperbolic`, `EpsteinZin`, and state-dependent discounting (`exoticpreferences=3`, discount factor depends on next-period `z`).

Alternative model setups: endogenous exit (`endogenousexit`), separable return function (`separableReturnFn`).

Many codes can be sped up using Howards improvement (controlled via `howards`, `howardsgreedy`, `howardssparse`), refinement on `d` (default when `n_d>0`), and a grid-interpolation layer (`gridinterplayer`). Divide-and-conquer (`divideandconquer`) is implemented but typically slow for infinite-horizon problems.

The `endogenousexit` branch returns additional outputs alongside (`V`, `Policy`): `ExitPolicy` when `endogenousexit=1`, and `PolicyWhenExit`, `ExitPolicy` when `endogenousexit=2`. All other branches return two outputs (`V`, `Policy`).

## Inputs

| Name | Type | Description |
|---|---|---|
| `n_d` | row vector (or `0`) | Grid size per decision variable. Set `n_d=0` if there are no decision variables. |
| `n_a` | row vector | Grid size per endogenous state variable. |
| `n_z` | row vector (or `0`) | Grid size per exogenous (Markov) shock. Set `n_z=0` if there are none. |
| `d_grid` | column vector, `sum(n_d)`-by-1 | Stacked decision grids. Joint-grid form (`prod(n_d)`-by-`length(n_d)`) is also accepted. |
| `a_grid` | column vector, `sum(n_a)`-by-1 | Stacked endogenous state grids. |
| `z_grid` | column vector or matrix | Exogenous shock grid. Accepted shapes: `sum(n_z)`-by-1 stacked vector, or `prod(n_z)`-by-`length(n_z)` joint grid. May also be a function handle (paired with `vfoptions.ExogShockFn`). |
| `pi_z` | matrix | Markov transition matrix, `N_z`-by-`N_z`. |
| `ReturnFn` | function handle | Return function. Inputs begin with `(d, aprime, a, z, ...)` followed by parameters; the exact prefix depends on whether `e`, multiple `d`, or experience/inheritance assets are used. |
| `Parameters` | struct | Struct holding all model parameters by name. |
| `DiscountFactorParamNames` | cell array of strings | Names (in `Parameters`) of the discount-factor parameters, in order. |
| `ReturnFnParamNames` | cell array of strings (may be empty) | User should just input [] (empty), is used internally to speed up repeat calls. |
| `vfoptions` | struct (optional) | Solver options. See below. |

## Outputs

| Name | Type | Description |
|---|---|---|
| `V` | multidimensional array | Value function. Shape is `[n_a, n_z]` in the standard case; `[n_a]` if `N_z=0`. If `vfoptions.outputkron=1`, returned in Kronecker form (`N_a`-by-`N_z`). |
| `Policy` | multidimensional array | Policy indices. The leading dimension indexes `(d, aprime)` jointly when `n_d>0`, or just `aprime` when `n_d=0`; remaining dimensions match `V`. Returned in Kronecker form when `vfoptions.outputkron=1`. |
| `ExitPolicy` | array | (`endogenousexit=1` or `2`) Binary decision to exit (`1` is exit, `0` is 'not exit'). |
| `PolicyWhenExit` | array | (`endogenousexit=2` only) Current-period decisions of agents who exit at end of period. |

## Options (vfoptions)

The function reads the following fields from `vfoptions`. All are optional unless noted.

| Field | Default | Description |
|---|---|---|
| `verbose` | `0` | If `1`, print feedback on what is happening internally (also prints `vfoptions` itself when `>=1`). |
| `tolerance` | `10^(-9)` | Convergence tolerance for `||V_n - V_{n-1}||`. |
| `maxiter` | `10^4` | Maximum number of value-function iterations before stopping. |
| `V0` | (none) | Optional initial guess for the value function, of shape `[n_a, n_z]` or `[N_a, N_z]`. Defaults to zeros. |
| `solnmethod` | `'purediscretization_refinement'` | Solver algorithm. One of `'purediscretization_refinement'` (default when `n_d>0`; presolves for `dstar(aprime,a,z)`), `'purediscretization'` (used when `n_d=0`), `'purediscretization_relativeVFI'` (can be unstable). `'purediscretization_endogenousVFI'` is not yet working. |
| `divideandconquer` | `0` | If `1`, use the divide-and-conquer algorithm to exploit monotonicity. Tends to be slow in infinite horizon; cannot be combined with refinement. Automatically disabled when `n_a` is scalar or when `gridinterplayer=1`. |
| `gridinterplayer` | `0` | If `1`, interpolate between grid points. Requires `vfoptions.ngridinterp`. |
| `ngridinterp` | (none) | Number of interpolation points between consecutive `a_grid` points. Required if `gridinterplayer=1`. |
| `preGI` | `0` | Used only with `gridinterplayer=1`; post-GI is typically faster. |
| `lowmemory` | `0` | If `>0`, use more loops and less parallelisation to reduce memory at the cost of speed. |
| **Howards improvement** | | |
| `howardsgreedy` | `0` | `0` for iterated Howards (modified policy-function iteration), `1` for greedy Howards (policy-function iteration). Greedy is faster for small models but cannot handle `V` taking `-Inf`, and is incompatible with `gridinterplayer=1`. |
| `howards` | `150` | Number of Howards iterations per Bellman step (when `howardsgreedy=0`). Based on tests, 80–150 is typically fastest. |
| `maxhowards` | `500` | Turn Howards off after this many outer iterations (safety guard against bad convergence). |
| `howardssparse` | auto | If `1`, do Howards iteration using a sparse matrix; only faster for larger models. Default is `1` when `N_a>1200` and `N_z>100`, else `0`. |
| **More exogenous states** | | |
| `ExogShockFn` | (no default) | Function handle producing `z_grid`/`pi_z`. If supplied, replaces `z_grid`/`pi_z` validation. |
| `SemiEndogShockFn` | (no default) | Function handle (or matrix) producing a state-dependent transition `pi_z(a,z,zprime)`. Triggers the semi-endogenous solver. Requires `SemiEndogShockFnParamNames`. |
| `SemiEndogShockFnParamNames` | (required with `SemiEndogShockFn`) | Names (in `Parameters`) of the parameters used by `SemiEndogShockFn`. |
| `n_e` | `0` | Grid size(s) for i.i.d. shocks `e`. Used by the experience-asset and inheritance-asset branches; requires `vfoptions.e_grid` and `vfoptions.pi_e` (or `vfoptions.EiidShockFn`). |
| `e_grid` | (none; required if `n_e>0`) | Grid for i.i.d. shocks. Either stacked `sum(n_e)`-by-1 or joint `prod(n_e)`-by-`length(n_e)`. |
| `pi_e` | (none; required if `n_e>0`) | Probabilities for i.i.d. shocks `e`, `prod(n_e)`-by-1. |
| `EiidShockFn` | (no default) | Function handle producing `e_grid`/`pi_e`. If supplied, replaces validation of `vfoptions.e_grid`/`pi_e`. |
| `n_semiz` | `0` | Semi-exogenous shocks (depends on `d`) are not yet implemented for infinite-horizon and will throw an error if set. |
| **Alternative endogenous states** | | |
| `experienceasset` | `0` | If `1`, treat the last endogenous state as an experience asset with `aprime(d,a)`. Requires `vfoptions.aprimeFn`. |
| `inheritanceasset` | `0` | If `1`, treat the last endogenous state as an inheritance asset. Requires `vfoptions.aprimeFn`. |
| `incrementaltype` | `0` | Vector flagging endogenous states that are incremental (`aprime` either equals `a` or is one grid point higher). Requires `solnmethod='purediscretization'`. |
| `aprimeFn` | (required by `experienceasset`/`inheritanceasset`) | Function handle giving the deterministic transition `aprime` for the experience/inheritance asset. |
| **Alternative model setup** | | |
| `endogenousexit` | `0` | `0`: no endogenous exit. `1`: agents may exit; requires `vfoptions.ReturnToExitFn`. `2`: mixture of endogenous and exogenous exit (also returns `PolicyWhenExit`). |
| `ReturnToExitFn` | (required if `endogenousexit>=1`) | Function handle giving the one-shot return received on exit. |
| `keeppolicyonexit` | `0` | If `1`, retain the policy choice for agents who exit (relevant to how `Policy` is reported on exit states). |
| `separableReturnFn` | `0` | Advanced option to split `ReturnFn` into two parts (`ReturnFn.R1` and `ReturnFn.R2`). |
| `exoticpreferences` | `'None'` | One of `'None'`, `'QuasiHyperbolic'`, `'EpsteinZin'`, or numeric `3` (state-dependent discounting, where the discount factor depends on next-period `z`). Selects the corresponding solver branch. |
| `quasi_hyperbolic` | `'Naive'` | Only used with `exoticpreferences='QuasiHyperbolic'`. Either `'Naive'` or `'Sophisticated'`. |
| `QHadditionaldiscount` | (required with `'QuasiHyperbolic'`) | Name (in `Parameters`) of the additional present-bias discount factor (`beta0`). |
| **Other** | | |
| `piz_strictonrowsaddingtoone` | `0` | If `1`, require rows of `pi_z` to sum to exactly `1`; otherwise allow tolerance of `10^(-13)`. |
| **Intended for internal use only** | | |
| `outputkron` | `0` | If `1`, return `V` and `Policy` in Kronecker form rather than reshaped to `[n_a, n_z]`. |
| `parallel` | `1 + (gpuDeviceCount>0)` | `2` for GPU, `1` for CPU. CPU only works for basic problem and is really just for illustrative purposes, it is slow. |
| `alreadygridvals` | `0` | Internal flag set when `z_grid` has already been pre-converted to gridvals form by a caller. |
| `alreadygridvals_semiexo` | `0` | Internal flag set when semi-exogenous gridvals have already been pre-computed by a caller. |


See the full [vfoptions reference](../options-reference/vfoptions.md) for fields shared with other functions.

## Example

See Intro to Infinite-Horizon Models for numerous examples.

See [Heterogeneous Firm Entry and Exit Models](https://www.vfitoolkit.com/updates-blog/2020/entry-exit-example-based-on-hopenhayn-rogerson-1993/) for examples with Firm Entry/Exit.

## Notes

- Use this function for infinite-horizon problems. The finite-horizon counterpart is [`ValueFnIter_Case1_FHorz`](ValueFnIter_Case1_FHorz.md).
- CPU mode (`vfoptions.parallel<2`) supports only the basic case via a dedicated `ValueFnIter_InfHorz_CPU` routine: no refinement, no exotic preferences, only the standard endogenous state.
- The default solver path is refinement (`purediscretization_refinement`) when `n_d>0`. With `n_d=0`, the solver automatically falls back to `purediscretization`.
- Internally, exogenous grids are reshaped into joint-grid form (`z_gridvals`, and `e_gridvals`/`pi_e` when used). This is done by `ExogShockSetup_InfHorz` (and `SemiExogShockSetup_InfHorz` for semi-exogenous setups).
- For Howards improvement, choose between iterated (`howardsgreedy=0`, more robust to `-Inf` values) and greedy (`howardsgreedy=1`, faster on small models). The `howardssparse` toggle becomes worthwhile only on larger state spaces.

## Source

[`ValueFnIter/InfHorz/ValueFnIter_InfHorz.m`](https://github.com/vfitoolkit/VFIToolkit-matlab/blob/master/ValueFnIter/InfHorz/ValueFnIter_InfHorz.m)
