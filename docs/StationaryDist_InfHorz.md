---
title: StationaryDist_InfHorz
sidebar_label: StationaryDist_InfHorz
description: Compute the stationary (agent) distribution for an infinite-horizon model given a policy function.
---

# StationaryDist_InfHorz

> Top-level dispatcher that computes the stationary distribution of agents over `(a, z)` for an infinite-horizon model, given the policy returned by [`ValueFnIter_InfHorz`](ValueFnIter_InfHorz.md).

## Signature

```matlab
StationaryDist = StationaryDist_InfHorz( ...
    Policy, n_d, n_a, n_z, pi_z, ...
    simoptions, Parameters, EntryExitParamNames)
```

## Description

`StationaryDist_InfHorz` iterates the transition operator induced by `Policy` and `pi_z` to a fixed point, returning the stationary distribution over endogenous states `a` and Markov exogenous states `z`.

By default the iteration uses the Tan (2020) improvement (`tanimprovement=1`) starting either from `simoptions.initialdist` (if provided) or from a basic internal guess. The legacy simulation-based approach (`iterate=0`) and an eigenvector-based approach (`eigenvector=1`) are retained for demonstration but not recommended in practice.

Alternative endogenous states: experience assets (`experienceasset`) and inheritance assets (`inheritanceasset`) are handled by dedicated dispatchers. The grid-interpolation layer (`gridinterplayer`) is also supported.

Alternative model setups: entry and exit of agents (`agententryandexit=1` for endogenous/exogenous exit and mass adjustment via a new-agents distribution; `agententryandexit=2` for trivial entry-exit that preserves total mass). Semi-endogenous shocks (`SemiEndogShockFn`) where the transition matrix depends on the current endogenous state are supported via the eigenvector method only.

When using entry and exit, the output `StationaryDist` is a struct with fields `pdf` and `mass`; otherwise it is a plain array.

## Inputs

| Name | Type | Description |
|---|---|---|
| `Policy` | multidimensional array | Policy index array returned by `ValueFnIter_InfHorz`. Shape matches that produced by the value function solver. |
| `n_d` | row vector (or `0`) | Grid size per decision variable. Set `n_d=0` if there are no decision variables. |
| `n_a` | row vector | Grid size per endogenous state variable. |
| `n_z` | row vector (or `0`) | Grid size per exogenous (Markov) shock. Set `n_z=0` if there are none. |
| `pi_z` | matrix | Markov transition matrix, `N_z`-by-`N_z`. |
| `simoptions` | struct (optional) | Solver options. See below. |
| `Parameters` | struct (optional) | Struct holding model parameters by name. Required when using `ExogShockFn`, `SemiEndogShockFn`, `experienceasset`, `inheritanceasset`, or `agententryandexit>=1`. |
| `EntryExitParamNames` | struct (optional) | Names (in `Parameters`) of the entry/exit-related fields. Required when `simoptions.agententryandexit>=1`. See below. |

### `EntryExitParamNames` fields

| Field | Used when | Description |
|---|---|---|
| `DistOfNewAgents` | `agententryandexit=1` or `2` | Name (in `Parameters`) of the distribution of newly entering agents over `(a, z)`. |
| `MassOfNewAgents` | `agententryandexit=1` | Name of the mass of newly entering agents per period. |
| `CondlProbOfSurvival` | `agententryandexit=1` | Name of the conditional probability of survival (period-by-period non-exit probability), which may depend on `(a, z)`. |
| `CondlEntryDecisions` | optional, with `agententryandexit=1` | Name of conditional entry decisions used to filter the new-agents distribution. |
| `ProbOfDeath` | `agententryandexit=2` | Name of the (scalar) exogenous exit probability. |

## Outputs

| Name | Type | Description |
|---|---|---|
| `StationaryDist` | array (or struct) | Stationary distribution. Shape is `[n_a, n_z]` in the standard case; `[n_a]` if `N_z=0`. With `n_e>0` (where supported), trailing `n_e` dimension is appended. With entry and exit (`agententryandexit=1`), `StationaryDist` is a struct: `StationaryDist.pdf` contains the pdf (shape `[n_a, n_z]`) and `StationaryDist.mass` contains the total mass. If `simoptions.outputkron=1`, returned in Kronecker form (`N_a`-by-`N_z`). |

## Options (simoptions)

The function reads the following fields from `simoptions`. All are optional unless noted.

| Field | Default | Description |
|---|---|---|
| `verbose` | `0` | If `1`, print feedback on what is happening internally. |
| `tolerance` | `10^(-6)` | Convergence tolerance (L-infinity norm of the change in the distribution between checks). |
| `maxit` | `10^6` | Maximum number of iterations before stopping. |
| `initialdist` | (none) | Optional initial guess for the distribution (shape `[n_a, n_z]` or `N_a*N_z`-by-1). Used as the starting point for iteration. |
| `gridinterplayer` | `0` | If `1`, use the grid-interpolation layer (matches the value-function solver's `gridinterplayer`). Requires `simoptions.ngridinterp`. |
| `ngridinterp` | (required if `gridinterplayer=1`) | Number of interpolation points between consecutive `a_grid` points. |
| **Solution method (not recommended users alter these)** | | |
| `iterate` | `1` | If `1`, iterate the transition operator (recommended). If `0`, use the legacy simulation method (not recommended). |
| `tanimprovement` | `1` | If `1`, use the Tan (2020) two-step decomposition of the transition operator (faster). Only the baseline `z`-only case still supports `tanimprovement=0`. |
| `eigenvector` | `0` | If `1`, compute the stationary distribution as the left eigenvector of the transition matrix. Fast but not robust; falls back to iteration if it fails. Required when using `SemiEndogShockFn`. |
| `multiiter` | `50` | Number of iteration steps between tolerance checks. |
| **Simulation method (only used when `iterate=0`)** | | |
| `ncores` | `1` | Number of cores to use during simulation. |
| `seedpoint` | `[ceil(N_a/2), ceil(N_z/2)]` | Starting point for simulation. |
| `simperiods` | `10^6` | Number of simulation periods. |
| `burnin` | `10^3` | Number of burn-in periods discarded before recording. |
| **More exogenous states** | | |
| `ExogShockFn` | (no default) | Function handle producing `z_grid`/`pi_z`. If supplied, requires the `Parameters` input. |
| `SemiEndogShockFn` | (no default) | Function handle (or matrix) giving a state-dependent transition `pi_z(a, z, zprime)`. Triggers the semi-endogenous branch; only `eigenvector=1` is implemented for this case. Cannot be combined with entry and exit. |
| `SemiEndogShockFnParamNames` | (required with `SemiEndogShockFn` as a function handle) | Names (in `Parameters`) of the parameters used by `SemiEndogShockFn`. |
| `n_e` | `0` | Grid size(s) for i.i.d. shocks `e`. **Not yet implemented for InfHorz stationary distribution** — setting `n_e>0` errors out. |
| `n_semiz` | `0` | Semi-exogenous shocks (depends on `d`) are not yet implemented for the infinite-horizon stationary distribution and will throw an error if set. |
| **Alternative endogenous states** | | |
| `experienceasset` | `0` | If `1`, treat the last endogenous state as an experience asset; iteration uses the Tan improvement via `StationaryDist_InfHorz_ExpAsset`. Requires the `Parameters` input. |
| `inheritanceasset` | `0` | If `1`, treat the last endogenous state as an inheritance asset; iteration via `StationaryDist_InfHorz_InheritAsset`. Requires the `Parameters` input. |
| **Alternative model setup** | | |
| `agententryandexit` | `0` | `0`: no entry and exit. `1`: full entry and exit with endogenous mass adjustment; requires `EntryExitParamNames.DistOfNewAgents`, `MassOfNewAgents`, `CondlProbOfSurvival`. `2`: exogenous entry and exit of trivial nature (total mass unaffected); requires `DistOfNewAgents` and `ProbOfDeath`. |
| `endogenousexit` | `0` | Used internally when `agententryandexit=1` to match the value-function solver's `endogenousexit` setting. |
| **Intended for internal use only** | | |
| `outputkron` | `0` | If `1`, return the distribution in Kronecker form rather than reshaped to `[n_a, n_z]`. |
| `parallel` | `1 + (gpuDeviceCount>0)` | `2` for GPU, `1` for CPU. CPU only works for basic problem and is really just for illustrative purposes, it is slow. |
| `alreadygridvals` | `0` | Internal flag set when `z_grid` has already been pre-converted to gridvals form by a caller. |
| `alreadygridvals_semiexo` | `0` | Internal flag set when semi-exogenous gridvals have already been pre-computed by a caller. |


See the full [simoptions reference](../options-reference/simoptions.md) for fields shared with other functions.

## Example

See Intro to Infinite-Horizon Models for numerous examples.

See [Heterogeneous Firm Entry and Exit Models](https://www.vfitoolkit.com/updates-blog/2020/entry-exit-example-based-on-hopenhayn-rogerson-1993/) for examples with Firm Entry/Exit.

## Notes

- Use this function for infinite-horizon problems. The finite-horizon counterpart is `StationaryDist_FHorz_Case1`.
- The default and recommended path is iteration with the Tan (2020) improvement (`iterate=1`, `tanimprovement=1`). The simulation (`iterate=0`) and eigenvector (`eigenvector=1`) paths are retained mainly for legacy/demonstration.
- A good initial guess via `simoptions.initialdist` (e.g. the previous solution from an outer loop) substantially reduces iteration count compared to the default internal guess.
- Internally, `Policy` is converted to Kronecker form via `KronPolicyIndexes_Case1` (or its `noz` variant) before iterating.

## Source

[`StationaryDist/InfHorz/StationaryDist_InfHorz.m`](https://github.com/vfitoolkit/VFIToolkit-matlab/blob/master/StationaryDist/InfHorz/StationaryDist_InfHorz.m)
