---
title: ValueFnIter_Case1_FHorz
sidebar_label: ValueFnIter_Case1_FHorz
description: Solve a Case 1 finite-horizon value function iteration problem and return the value function and policy.
---

# ValueFnIter_Case1_FHorz

> Top-level dispatcher that solves a Case 1 finite-horizon value function iteration problem, returning the value function `V` and the policy index array `Policy` (typically `varargout=[V, Policy]`).

## Signature

```matlab
varargout = ValueFnIter_Case1_FHorz( ...
    n_d, n_a, n_z, N_j, ...
    d_grid, a_grid, z_grid, pi_z, ...
    ReturnFn, Parameters, ...
    DiscountFactorParamNames, ReturnFnParamNames, ...
    vfoptions)
```

## Description

`ValueFnIter_Case1_FHorz` is the entry point for solving Case 1 finite-horizon
value function iteration problems. It is built around decision variables `d`, 
endogenous states `a`, and markov exogenous states `z`.
It accepts as inputs the grid sizes (`n_d`, `n_a`, `n_z`), the number of horizon periods `N_j`, the corresponding grids, the
exogenous Markov transition matrix `pi_z`, a return function `ReturnFn`, and parameter information, and returns the value function and policy.

Other exogenous states: i.i.d. shocks `e`, semi-exogenous shocks `n_semiz`. Can set up `z` and `e` as parameterized functions.

Alternative endogenous states: experience assets (`experienceasset`, `experienceassetu`), risky assets (`riskyasset`),
residual assets (`residualasset`), incremental endogenous states (`incrementaltype`).

Can handle various exotic preferences: `QuasiHyperbolic`, `EpsteinZin`, `GulPesendorfer`, `AmbiguityAversion`.

Alternative model setups:  dynasty problems (`dynasty`), input period J+1 value function `V_Jplus1`.

Many codes can be sped up using divide-and-conquer (`divideandconquer`; requires conditional monotonicity of the policy fn), 
and grid-interpolation layer (`gridinterplayer`).

The `QuasiHyperbolic` exotic-preferences branch returns three outputs: (`V1`, `Policy`, `Valt`); all other branches return two (`V`, `Policy`).

## Inputs

| Name | Type | Description |
|---|---|---|
| `n_d` | row vector (or `0`) | Grid size per decision variable. Set `n_d=0` if there are no decision variables. |
| `n_a` | row vector | Grid size per endogenous state variable. |
| `n_z` | row vector (or `0`) | Grid size per exogenous (Markov) shock. Set `n_z=0` if there are none. |
| `N_j` | scalar (positive integer) | Number of finite-horizon periods. |
| `d_grid` | column vector, `sum(n_d)`-by-1 | Stacked decision grids. |
| `a_grid` | column vector, `sum(n_a)`-by-1 | Stacked endogenous state grids. |
| `z_grid` | column vector or matrix | Exogenous shock grid. Accepted shapes: `sum(n_z)`-by-1 stacked vector; `prod(n_z)`-by-`length(n_z)` joint grid; `sum(n_z)`-by-`N_j` age-dependent stacked vector; `prod(n_z)`-by-`length(n_z)`-by-`N_j` age-dependent joint grid. May also be a function handle (paired with `vfoptions.ExogShockFn`). |
| `pi_z` | matrix or 3-D array | Markov transition matrix. Either `N_z`-by-`N_z` or `N_z`-by-`N_z`-by-`N_j` (age-dependent). |
| `ReturnFn` | function handle | Return function. Inputs begin with `(d, aprime, a, z, ...)` followed by parameters; the exact prefix depends on whether `e`, `semiz`, multiple `d`, `riskyasset`, or experience assets are used. |
| `Parameters` | struct | Struct holding all model parameters by name. |
| `DiscountFactorParamNames` | cell array of strings | Names (in `Parameters`) of the discount-factor parameters, in order. |
| `ReturnFnParamNames` | cell array of strings (may be empty) | User should just input [] (empty), is used internally to speed up repeat calls. |
| `vfoptions` | struct (optional) | Solver options. See below. |

## Outputs

| Name | Type | Description |
|---|---|---|
| `V` | multidimensional array | Value function. Shape is `[n_a, n_z, N_j]` in the standard case; `[n_a, N_j]` if `N_z=0`; `[n_a, n_z, n_e, N_j]` (or `[n_a, n_e, N_j]`) when `vfoptions.n_e` is used. If `vfoptions.outputkron=1`, returned in Kronecker form (`N_a`-by-`N_z`-by-`N_j`). |
| `Policy` | multidimensional array | Policy indices. The leading dimension indexes `(d, aprime)` jointly when `n_d>0`, or just `aprime` when `n_d=0`; remaining dimensions match `V`. Returned in Kronecker form when `vfoptions.outputkron=1`. |
| `Valt` | array | (`QuasiHyperbolic` exotic preferences only) Alternative value function returned alongside `V` and `Policy`. |

## Options (vfoptions)

The function reads the following fields from `vfoptions`. All are optional unless noted.

| Field | Default | Description |
|---|---|---|
| `verbose` | `0` | If `1`, print feedback on what is happening internally (also prints `vfoptions` itself when `>=1`). |
| `divideandconquer` | `0` | If `1`, use the divide-and-conquer algorithm to exploit monotonicity (GPU only). |
| `gridinterplayer` | `0` | If `1`, interpolate between grid points. Requires `vfoptions.ngridinterp` to also be set. |
| `ngridinterp` | (none) | Number of interpolation points between consecutive `a_grid` points. Required if `gridinterplayer=1`. |
| **More exogenous states** | | |
| `lowmemory` | `0` | If `>0`, use more loops and less parallelisation to reduce memory at the cost of speed. |
| `n_e` | `0` | Grid size(s) for i.i.d. shocks `e`. If `>0`, also requires `vfoptions.e_grid` and `vfoptions.pi_e` (or `vfoptions.EiidShockFn`). |
| `e_grid` | (none; required if `n_e>0`) | Grid for i.i.d. shocks. Either stacked-grid `sum(n_e)`-by-1 or joint-grid `prod(n_e)`-by-`length(n_e)`. Can be age-dependent in which case  `sum(n_e)`-by-`N_j` or `prod(n_e)`-by-`length(n_e)`-by-`N_j`. |
| `pi_e` | (none; required if `n_e>0`) | Probabilities for i.i.d. shocks `e`. Either `prod(n_e)`-by-1 or `prod(n_e)`-by-`N_j`. |
| `n_semiz` | `0` | Grid size(s) for semi-exogenous shocks. If `>0`, triggers the semi-exogenous setup and solver. |
| `l_dsemiz` | `1` | Number of decision variables that influence the semi-exogenous transition. |
| `ExogShockFn` | (no default) | Function handle producing `z_grid`/`pi_z`. If supplied, replaces `z_grid`/`pi_z` validation. |
| `EiidShockFn` | (no default) | Function handle producing `e_grid`/`pi_e`. If supplied, replaces validation of `vfoptions.e_grid`/`pi_e`. |
| `n_u` | (none; required if `riskyasset=1`) | `u` is between-period i.i.d. shock. Grid size for the shock `u`. |
| `u_grid` | (none; required if `riskyasset=1`) | `u` is between-period i.i.d. shock. Grid for the shock. |
| `pi_u` | (none; required if `riskyasset=1`) | `u` is between-period i.i.d. shock. Probabilities for the shock. |
| **Alternative endogenous states** | | |
| `experienceasset` | `0` | If `1`, treat the last endogenous state as an experience asset with `aprime(d,a)`. |
| `l_dexperienceasset` | `1` | Number of decision variables that influence the experience asset. Only used when `experienceasset=1`. |
| `experienceassetu` | `0` | If `1`, treat the last endogenous state as an experience asset with shock, `aprime(d,a,u)`. |
| `l_dexperienceassetu` | `1` | Number of decision variables that influence the (shocked) experience asset. Only used when `experienceassetu=1`. |
| `riskyasset` | `0` | If `1`, treat the last endogenous state as a risky asset; requires `aprimeFn(d,u)`, `n_u`, `u_grid`, `pi_u`. |
| `refine_d` | (no default) | Needed when using `riskyasset`, see `riskyasset` examples. |
| `residualasset` | `0` | If `1`, treat the last endogenous state as a residual asset. |
| `incrementaltype` | `0` | Vector flagging endogenous states that are incremental (`aprime` either equals `a` or is one grid point higher). Any non-zero element triggers the `Increment` solver. |
| **Alternative model setup** | | |
| `exoticpreferences` | `'None'` | One of `'None'`, `'QuasiHyperbolic'`, `'EpsteinZin'`, `'GulPesendorfer'`, `'AmbiguityAversion'`. Selects the corresponding solver branch. |
| `n_ambiguity` | `0` | Grid size for ambiguity dimension (used with the `AmbiguityAversion` branch). |
| `WarmGlowBequestsFn` | (no default) | Function handle for warm-glow-of-bequests term. Only used by Epstein–Zin preferences. Inputs begin with `aprime`. |
| `dynasty` | `0` | If `1`, solve a dynasty (overlapping-generations-style) problem; sets `tolerance` default if not provided. |
| `tolerance` | `10^(-9)` | Only used by `dynasty`. Convergence tolerance. |
| `V_Jplus1` | (no default) | Optional terminal value function at age `N_j+1`. If absent, the terminal-period return is maximised on its own. |
| **Intended for internal use only** | | |
| `outputkron` | `0` | If `1`, return `V` and `Policy` in Kronecker form rather than reshaped to `[n_a, n_z, ...]`. |
| `parallel` | `1 + (gpuDeviceCount>0)` | `2` for GPU, `1` for CPU. CPU only works for basic problem and is really just for illustrative purposes, it is slow. |


See the full [vfoptions reference](../options-reference/vfoptions.md) for fields shared with other functions.

## Example

See [Intro to Life-Cycle Models](https://www.vfitoolkit.com/updates-blog/2021/an-introduction-to-life-cycle-models/) for numerous examples.

## Notes

- Use this function for finite-horizon Case 1 problems. The infinite-horizon counterpart is `ValueFnIter_InfHorz`.
- CPU mode (`vfoptions.parallel<2`) supports only the basic case: no `e` shocks, no exotic preferences, only standard endogenous state.
- Internally, exogenous grids are always reshaped into age-dependent joint-grid form (`z_gridvals_J`, `pi_z_J`, `e_gridvals_J`, `pi_e_J`, `semiz_gridvals_J`, `pi_semiz_J`). This is done by the `ExogShockSetup_FHorz` and `SemiExogShockSetup_FHorz` internal commands, see them for details.

## Source

[`ValueFnIter/FHorz/ValueFnIter_Case1_FHorz.m`](https://github.com/vfitoolkit/VFIToolkit-matlab/blob/master/ValueFnIter/FHorz/ValueFnIter_Case1_FHorz.m)
