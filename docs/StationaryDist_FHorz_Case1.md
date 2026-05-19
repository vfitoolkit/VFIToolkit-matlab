---
title: StationaryDist_FHorz_Case1
sidebar_label: StationaryDist_FHorz_Case1
description: Compute the stationary (agent) distribution for a finite-horizon model given a policy function.
---

# StationaryDist_FHorz_Case1

> Top-level dispatcher that computes the agent distribution over `(a, z, j)` for a Case 1 finite-horizon model, given the policy returned by [`ValueFnIter_Case1_FHorz`](ValueFnIter_Case1_FHorz.md) and the age-one distribution.

## Signature

```matlab
StationaryDist = StationaryDist_FHorz_Case1( ...
    jequaloneDist, AgeWeightParamNames, Policy, ...
    n_d, n_a, n_z, N_j, pi_z, Parameters, simoptions)
```

## Description

`StationaryDist_FHorz_Case1` iterates the age-`j` distribution forward to age `j+1` using `Policy` and the (possibly age-dependent) Markov transition `pi_z`, starting from the user-supplied age-one distribution `jequaloneDist`. The age-`j` slices are then weighted by the age weights named in `AgeWeightParamNames` so that the returned distribution integrates to one across `(a, z, j)`.

Iteration uses the Tan (2020) improvement throughout.

Other exogenous states: i.i.d. shocks `e` (via `simoptions.n_e`, `e_grid`, `pi_e` or `EiidShockFn`) and semi-exogenous shocks (`simoptions.n_semiz`) are dispatched to dedicated subroutines.

Alternative endogenous states: experience assets (`experienceasset`, `experienceassetu`), risky assets (`riskyasset`), and residual assets (`residualasset`) are handled by dedicated dispatchers, including combinations with semi-exogenous shocks.

The age-one distribution `jequaloneDist` can be passed as either an array or a function handle. When a function handle, it is called as `jequaloneDistFn(a_grid, z_grid, n_a, n_z, ...)` with any trailing inputs filled from `Parameters`; in that case `simoptions.a_grid` and `simoptions.z_grid` must be supplied.

## Inputs

| Name | Type | Description |
|---|---|---|
| `jequaloneDist` | array or function handle | Distribution of agents at age `j=1`. Shape `[n_a, n_z]` (or the appropriate variant when there is no `z`, or when `e` is used). Must integrate to one. If a function handle, it is evaluated as `f(a_grid, z_grid, n_a, n_z, ...)` with trailing parameter inputs filled from `Parameters`. |
| `AgeWeightParamNames` | cell array of strings (single entry) | Name (in `Parameters`) of the age-weights row vector. The vector must have length `N_j` and sum to one (a warning is issued otherwise). |
| `Policy` | multidimensional array | Policy index array returned by `ValueFnIter_Case1_FHorz`. The leading dimension indexes `(d, aprime)` jointly when `n_d>0` (or just `aprime` when `n_d=0`); remaining dimensions are `[n_a, n_z, N_j]` (or appropriate variants). |
| `n_d` | row vector (or `0`) | Grid size per decision variable. Set `n_d=0` if there are no decision variables. |
| `n_a` | row vector | Grid size per endogenous state variable. Currently supports `length(n_a)<=4`. |
| `n_z` | row vector (or `0`) | Grid size per exogenous (Markov) shock. Set `n_z=0` if there are none. |
| `N_j` | scalar (positive integer) | Number of finite-horizon periods. |
| `pi_z` | matrix or 3-D array | Markov transition matrix. Either `N_z`-by-`N_z` or `N_z`-by-`N_z`-by-`N_j` (age-dependent). |
| `Parameters` | struct | Struct holding all model parameters by name (including the age-weight vector and any `ExogShockFn` / `EiidShockFn` / asset-function parameters). |
| `simoptions` | struct (optional) | Solver options. See below. |

## Outputs

| Name | Type | Description |
|---|---|---|
| `StationaryDist` | multidimensional array | Agent distribution. Shape is `[n_a, n_z, N_j]` in the standard case; `[n_a, N_j]` if `N_z=0`; `[n_a, n_z, n_e, N_j]` (or `[n_a, n_e, N_j]`) when `simoptions.n_e` is used. If `simoptions.outputkron=1`, returned in Kronecker form. Integrates to one. |

## Options (simoptions)

The function reads the following fields from `simoptions`. All are optional unless noted.

| Field | Default | Description |
|---|---|---|
| `gridinterplayer` | `0` | If `1`, interpolate between grid points (must match `vfoptions.gridinterplayer`). Requires `simoptions.ngridinterp`. |
| `ngridinterp` | (required if `gridinterplayer=1`) | Number of interpolation points between consecutive `a_grid` points. |
| **More exogenous states** | | |
| `ExogShockFn` | (no default) | Function handle producing `z_grid`/`pi_z`. If supplied, replaces `z_grid`/`pi_z` validation. |
| `EiidShockFn` | (no default) | Function handle producing `e_grid`/`pi_e`. If supplied, replaces validation of `simoptions.e_grid`/`pi_e`. |
| `n_e` | `0` | Grid size(s) for i.i.d. shocks `e`. If `>0`, also requires `simoptions.e_grid` and `simoptions.pi_e` (or `simoptions.EiidShockFn`). |
| `e_grid` | (required if `n_e>0`) | Grid for i.i.d. shocks. Either stacked `sum(n_e)`-by-1 or joint `prod(n_e)`-by-`length(n_e)`. Can be age-dependent (`sum(n_e)`-by-`N_j` or `prod(n_e)`-by-`length(n_e)`-by-`N_j`). |
| `pi_e` | (required if `n_e>0`) | Probabilities for i.i.d. shocks `e`. Either `prod(n_e)`-by-1 or `prod(n_e)`-by-`N_j`. |
| `n_semiz` | `0` | Grid size(s) for semi-exogenous shocks. If `>0`, triggers the semi-exogenous setup and dispatcher. |
| `l_dsemiz` | `1` | Number of decision variables that influence the semi-exogenous transition. Used when `n_semiz>0`. |
| **Alternative endogenous states** | | |
| `experienceasset` | `0` | If `1`, treat the last endogenous state as an experience asset; iteration uses `StationaryDist_FHorz_ExpAsset` (or `_ExpAssetSemiExo` when also using `n_semiz`). |
| `l_dexperienceasset` | `1` | Number of decision variables that influence the experience asset. Used when `experienceasset=1`. |
| `experienceassetu` | `0` | If `1`, treat the last endogenous state as an experience asset with shock. |
| `l_dexperienceassetu` | `1` | Number of decision variables that influence the (shocked) experience asset. Used when `experienceassetu=1`. |
| `riskyasset` | `0` | If `1`, treat the last endogenous state as a risky asset. Strongly recommended to also set `simoptions.refine_d` (matching the value-fn solver). |
| `refine_d` | (no default) | Refinement flag for the risky-asset path; see the `riskyasset` examples. |
| `residualasset` | `0` | If `1`, treat the last endogenous state as a residual asset; dispatches to `StationaryDist_FHorz_ResidAsset`. |
| **Required for some setups** | | |
| `a_grid` | (required when `jequaloneDist` is a function handle, and for some asset/semi-exo branches) | Endogenous-state grid. |
| `z_grid` | (required when `jequaloneDist` is a function handle, and for some semi-exo branches) | Exogenous-state grid. |
| `d_grid` | (required when `n_semiz>0` and on some asset branches) | Decision-variable grid. |
| **Intended for internal use only** | | |
| `outputkron` | `0` | If `1`, return the distribution in Kronecker form rather than reshaped to `[n_a, n_z, N_j]`. |
| `parallel` | `1 + (gpuDeviceCount>0)` | `2` for GPU, `1` for CPU. CPU only works for basic problem and is really just for illustrative purposes, it is slow. |
| `alreadygridvals` | `0` | Internal flag set when `z_grid` has already been pre-converted to gridvals form by a caller. |
| `alreadygridvals_semiexo` | `0` | Internal flag set when semi-exogenous gridvals have already been pre-computed by a caller. |


See the full [simoptions reference](../options-reference/simoptions.md) for fields shared with other functions.

## Example

See [Intro to Life-Cycle Models](https://www.vfitoolkit.com/updates-blog/2021/an-introduction-to-life-cycle-models/) for numerous examples.

## Notes

- Use this function for finite-horizon problems. The infinite-horizon counterpart is [`StationaryDist_InfHorz`](StationaryDist_InfHorz.md).
- The age weights named by `AgeWeightParamNames` must be a row vector of length `N_j` summing to one; a column vector is auto-transposed.
- The age-one distribution `jequaloneDist` must integrate to exactly one (within `10^(-9)`), otherwise the function errors out.
- Internally, exogenous grids are reshaped into age-dependent joint-grid form (`pi_z_J`, `e_gridvals_J`, `pi_e_J`, `semiz_gridvals_J`, `pi_semiz_J`) by `ExogShockSetup_FHorz` and `SemiExogShockSetup_FHorz`.
- The standard endogenous-state branch currently supports up to `length(n_a)<=4`.

## Source

[`StationaryDist/FHorz/StationaryDist_FHorz_Case1.m`](https://github.com/vfitoolkit/VFIToolkit-matlab/blob/master/StationaryDist/FHorz/StationaryDist_FHorz_Case1.m)
