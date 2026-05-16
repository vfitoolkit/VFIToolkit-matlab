---
title: EvalFnOnAgentDist_AllStats_InfHorz
sidebar_label: EvalFnOnAgentDist_AllStats_InfHorz
description: Compute a wide range of summary statistics (mean, median, variance, Lorenz curve, Gini, quantiles, etc.) for functions evaluated on the agent distribution of an infinite-horizon model.
---

# EvalFnOnAgentDist_AllStats_InfHorz

> Evaluates each function in `FnsToEvaluate` on the `(a, z)` grid and returns a struct of summary statistics (mean, median, variance, std deviation, Lorenz curve, Gini coefficient, quantile cutoffs and means, plus inequality measures) computed against the stationary distribution from [`StationaryDist_InfHorz`](StationaryDist_InfHorz.md).

## Signature

```matlab
AllStats = EvalFnOnAgentDist_AllStats_InfHorz( ...
    StationaryDist, Policy, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, ...
    n_d, n_a, n_z, d_grid, a_grid, z_grid, simoptions)
```

## Description

`EvalFnOnAgentDist_AllStats_InfHorz` turns `Policy` (indices) into `PolicyValues`, evaluates each user-supplied function on the joint grid of `(d, aprime, a, z, ...)`, and then summarises the resulting values against the stationary distribution `StationaryDist`. It is the standard way to extract aggregate moments and inequality statistics from an infinite-horizon model.

The stats computed for each function include `Mean`, `Median`, `RatioMeanToMedian`, `Variance`, `StdDeviation`, `LorenzCurve` (`npoints`-by-1), `Gini`, `QuantileCutoffs` (`nquantiles+1`-by-1, including min and max), `QuantileMeans` (`nquantiles`-by-1), and a set of additional inequality measures. Individual groups can be turned off via `simoptions.whichstats` to reduce runtime.

Conditional restrictions (`simoptions.conditionalrestrictions`) allow computing the same stats on the sub-population for which a user-supplied restriction function is non-zero. The output then also reports the `RestrictedSampleMass` for each restriction.

## Inputs

| Name | Type | Description |
|---|---|---|
| `StationaryDist` | array | Stationary distribution from `StationaryDist_InfHorz`. Shape `[n_a, n_z]` (or appropriate variant when there is no `z`, or with `e`/`semiz`). Must integrate to one. |
| `Policy` | multidimensional array | Policy index array returned by `ValueFnIter_InfHorz`. The leading dimension indexes `(d, aprime)` jointly when `n_d>0`, or just `aprime` when `n_d=0`. |
| `FnsToEvaluate` | struct or cell array of function handles | Functions to evaluate. As a struct, each field is one function and the field name labels the output; the function signature begins with `(d, aprime, a, z, ...)` followed by parameters. As a cell array, the output fields are named generically. |
| `Parameters` | struct | Struct holding all model parameters by name (used to fill the trailing parameter inputs of each function and any restriction function). |
| `FnsToEvaluateParamNames` | cell array of structs (may be empty) | Names of the parameter inputs for each function. Auto-derived from each function's signature when `FnsToEvaluate` is a struct; supply `[]` in that case. |
| `n_d` | row vector (or `0`) | Grid size per decision variable. Set `n_d=0` if there are no decision variables. |
| `n_a` | row vector | Grid size per endogenous state variable. |
| `n_z` | row vector (or `0`) | Grid size per exogenous (Markov) shock. Set `n_z=0` if there are none. |
| `d_grid` | column vector | Decision-variable grid (stacked or joint form). |
| `a_grid` | column vector | Endogenous-state grid (stacked). |
| `z_grid` | column vector or matrix | Exogenous-shock grid (stacked or joint form). |
| `simoptions` | struct (optional) | Solver options. See below. |

## Outputs

| Name | Type | Description |
|---|---|---|
| `AllStats` | struct | One field per function in `FnsToEvaluate` (named after the struct field, or generically when `FnsToEvaluate` is a cell array). Each contains: `Mean`, `Median`, `RatioMeanToMedian`, `Variance`, `StdDeviation`, `LorenzCurve` (`npoints`-by-1), `Gini`, `QuantileCutoffs` (`nquantiles+1`-by-1), `QuantileMeans` (`nquantiles`-by-1), plus additional inequality measures. Groups disabled in `simoptions.whichstats` are omitted. If `simoptions.conditionalrestrictions` is set, `AllStats` also contains one sub-struct per restriction name, holding `RestrictedSampleMass` and a copy of the same per-function stats computed against the restricted (renormalised) distribution. |

## Options (simoptions)

The function reads the following fields from `simoptions`. All are optional unless noted.

| Field | Default | Description |
|---|---|---|
| `npoints` | `100` | Number of points reported on the Lorenz curve. |
| `nquantiles` | `20` | Number of quantile groups (e.g. `20` for ventiles, `5` for quintiles). The output `QuantileCutoffs` has `nquantiles+1` entries (includes min and max). |
| `whichstats` | `ones(7,1)` | Length-7 indicator vector selecting which stat groups to compute: `(1)` mean, `(2)` median, `(3)` std dev and variance, `(4)` Lorenz curve and Gini, `(5)` min/max, `(6)` quantiles, `(7)` more inequality. Zeros skip the group to save time. For `(4)` and `(6)`, setting the entry to `2` switches to a faster but more memory-intensive routine. `RatioMeanToMedian` is reported whenever both mean and median are computed. |
| `tolerance` | `10^(-12)` | Numerical tolerance used when calculating min and max values. |
| `conditionalrestrictions` | (no default) | Struct of restriction function handles. Each field name labels the restriction and each handle has the same input signature as `FnsToEvaluate`. Stats are reported on the sub-distribution where the restriction returns a non-zero value (renormalised to mass one), and `RestrictedSampleMass` is included for each restriction. |
| `gridinterplayer` | `0` | If `1`, `Policy`'s leading dimension carries one extra slot used by the grid-interpolation layer (must match the value-function solver). |
| **Exogenous states folded into `z`** | | |
| `n_e` | `0` | Grid size(s) for i.i.d. shocks `e`. If `>0`, `e` is folded into `z` for the purposes of stat computation; the matching `e_grid`/`pi_e` (or `EiidShockFn`) must be in `simoptions`. |
| `n_semiz` | `0` | Grid size(s) for semi-exogenous shocks. If `>0`, `semiz` is folded into `z` for stat computation; the matching `semiz_grid`/`pi_semiz` setup must be in `simoptions`. |
| **Intended for internal use only** | | |
| `alreadygridvals` | `0` | Internal flag set when `z_grid` has already been pre-converted to gridvals form by a caller. |
| `alreadygridvals_semiexo` | `0` | Internal flag set when semi-exogenous gridvals have already been pre-computed by a caller. |


See the full [simoptions reference](../options-reference/simoptions.md) for fields shared with other functions.

## Example

See Intro to Infinite-Horizon Models for numerous examples.

## Notes

- This function requires a GPU; it errors out otherwise.
- `EvalFnOnAgentDist_AggVars_InfHorz` is used internally to produce the same output as `AllStats` reports in `Mean`, as it is faster since it is lighter and hard-coded. But as an end user it is recommended you just use `AllStats` as it is much more powerful.
- Use this function for infinite-horizon problems. The finite-horizon counterpart is `EvalFnOnAgentDist_AllStats_FHorz_Case1`.
- For very large state spaces or many functions, disable unneeded stat groups via `simoptions.whichstats` to cut runtime substantially.
- The `LorenzCurve` is informative for variables that are non-negative; for variables that can be negative, the Gini coefficient and Lorenz curve interpretations may be non-standard.
- Internally, `e` and `semiz` shocks are folded into `z` (via `CreateGridvals_FnsToEvaluate_InfHorz`) so that the same kernel evaluates the functions over the full state space.

## Source

[`EvaluateFnOnAgentDist/InfHorz/EvalFnOnAgentDist_AllStats_InfHorz.m`](https://github.com/vfitoolkit/VFIToolkit-matlab/blob/master/EvaluateFnOnAgentDist/InfHorz/EvalFnOnAgentDist_AllStats_InfHorz.m)
