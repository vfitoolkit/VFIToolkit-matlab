# Action Space and State Space

Almost every shape in the VFI Toolkit — the value function `V`, the agent distribution `AgentDist`, the policy `Policy`, and the input list of your `ReturnFn` and `FnsToEvaluate` — is a direct consequence of two things you declare at the top of your script: the **action space** and the **state space**. This document sets out that mapping.

Nothing here is an option you set. You declare `n_d`, `n_a`, `n_z` (and `vfoptions.n_semiz`, `vfoptions.n_e`, `N_j`), and everything below follows.

---

## Notation

For a block of variables `x` (one of `d`, `a`, `z`, `semiz`, `e`):

| symbol | meaning |
|---|---|
| `n_x` | row vector of grid sizes, one entry per variable in the block |
| `l_x` | `length(n_x)` — the **number of variables** in the block |
| `N_x` | `prod(n_x)` — the **number of points** in the block's (joint) grid |

So `n_z=[7,5]` means `l_z=2` markov variables and `N_z=35` joint markov states. To switch a block off, set it to `0` (`n_d=0`, `n_z=0`).

Age/period is `j=1..N_j` in finite horizon; infinite horizon has no `j`.

---

## The two spaces

**The action space** is the decision variables `d`, plus the next-period endogenous states `aprime`. These are what the agent chooses; they are the arguments the Bellman operator maximises over.

**The state space** is everything the agent conditions on when choosing: the endogenous states `a`, the exogenous states `semiz`, `z`, `e`, and (finite horizon) the age `j`.

The toolkit's canonical ordering, used everywhere, is

```
   action:   d ,  aprime
   state:    a ,  semiz ,  z ,  e   [ , j ]
```

`j` is bracketed because it behaves differently from the rest: it is part of the state space and it shapes every array below, but it is never an argument of `ReturnFn` or `FnsToEvaluate`. See [Age `j` and transition period `t` are not inputs](#age-j-and-transition-period-t-are-not-inputs).

Within a block, variables are in the order they appear in `n_d`/`n_a`/`n_z`/`n_semiz`/`n_e`. See [ExogenousShocks.md](ExogenousShocks.md) for what `semiz`, `z` and `e` are and why they sit in that order.

Two remarks that matter for the shapes below:

- `aprime` is *not* automatically the same size as `a`. In the standard case it is (every endogenous state is chosen), but the alternative asset types (`experienceasset`, `riskyasset`, `residualasset`, `inheritanceasset`, …) make one endogenous state evolve through a law of motion `aprimeFn` rather than being chosen. That state is in the state space but not in the action space.
- The between-period shock `u` (used by `riskyasset`/`experienceassetu`) is neither: it is realised after the decision and is integrated out, so it appears in no shape below.

---

## State space determines the size of `V` and `AgentDist`

`V` and `AgentDist` are both functions of the state alone. They therefore have **exactly the same shape**, one entry per point of the state space.

| model | `V` and `AgentDist` shape |
|---|---|
| infinite horizon | `[n_a, n_z, n_e]` |
| finite horizon | `[n_a, n_semiz, n_z, n_e, N_j]` |

(`semiz` is finite-horizon only — `ValueFnIter_InfHorz` errors if `vfoptions.n_semiz>0`.)

Blocks that are switched off are simply absent — with no `semiz` and no `e` this is the familiar `[n_a,n_z]` / `[n_a,n_z,N_j]`. With `n_z=0` and no other shock it collapses to `[n_a]` / `[n_a,N_j]`.

The **number of elements** is

```
numel(V) = N_a * N_semiz * N_z * N_e ( * N_j )
```

> **Worked example.** `n_a=501`, `n_z=[7,5]`, `n_e=11`, `N_j=81`. Then `N_a=501`, `N_z=35`, `N_e=11`, giving `501*35*11*81 = 15,623,685` elements. `V` is ~125MB, `AgentDist` another ~125MB.

### `AgentDist` is always on the coarse grid

`gridinterplayer` refines where `aprime` may land, not where agents may be. `AgentDist` stays on `n_a`, with the interpolated `aprime` split as linear interpolation probabilities over the two neighbouring `a_grid` points.

---

## Action space and state space together determine the size of `Policy`

`Policy` records, for each point of the state space, the chosen point of the action space. So it is the state-space shape with **one extra leading dimension** holding the action:

```
Policy is [ l_d + l_aprime , n_a, n_semiz, n_z, n_e (, N_j) ]
```

The leading dimension is *not* the number of action grid points; it is the number of action *variables*. Each row holds a grid **index** into that variable's own grid (`Policy(1,...)` indexes `d_grid`'s first variable, and so on). The rows are in canonical order:

```
row 1 .. l_d          : d1, d2, ...    (indices into d_grid)
row l_d+1 .. l_d+l_aprime : a1prime, a2prime, ...  (indices into a_grid)
```

Some concrete cases:

| setup | leading dimension | rows |
|---|---|---|
| `n_d=0`, one asset | `1` | `aprime` |
| one `d`, one asset | `2` | `d`, `aprime` |
| `n_d=[nd1,nd2]`, `n_a=[na1,na2]` | `4` | `d1`, `d2`, `a1prime`, `a2prime` |
| semi-exogenous (`d` splits into `d1`,`d2`) | `l_d1+l_d2+l_a` | `d1`s, then `d2`s, then `aprime`s |
| `experienceasset` with `n_a=[na1,na2]` | `l_d + 1` | `d`s, then `a1prime` only — the experience asset `a2` has no chosen `a2prime` |
| `riskyasset` / `residualasset` / `inheritanceasset` | `l_d + l_a - 1` | as above; the special asset's next value comes from `aprimeFn` |

Note the leading dimension is kept even when it is 1: with `n_d=0` and a single asset, `Policy` is `[1, n_a, n_z, N_j]`, not `[n_a, n_z, N_j]`.

### `gridinterplayer` adds two rows

With `vfoptions.gridinterplayer=1`, `Policy` gains **two extra trailing rows**:

```
Policy is [ l_d + l_aprime + 2 , n_a, ... ]
```

- second-to-last row: `L2`, the index of the interpolation point between the lower `a_grid` point (held in the `a1prime` row) and the next one up;
- last row: `L2flag`, an override used when forming the distribution (`1` = put all weight on the lower grid point, `2` = usual interpolation, `3` = all weight on the upper).

Only the **first** endogenous asset is interpolated. `PolicyInd2Val` collapses these two rows back into a single interpolated `aprime` value, which is why everything downstream sees the plain `l_d+l_aprime` rows again.

### From indices to values

`Policy` holds indices. `PolicyInd2Val_FHorz` / `PolicyInd2Val_InfHorz` convert it to `PolicyValues`, same shape but with the actual grid values, with the `gridinterplayer` rows already collapsed and the alternative-asset rows already absent. `size(PolicyValues,1)` is therefore `l_d+l_aprime` — which is exactly the count used for `FnsToEvaluate` below.

---

## `ReturnFn` and `FnsToEvaluate` take the action and state space as their leading inputs

Both are anonymous functions, and both use the **same positional convention**: the action space and the state space come first, in canonical order, and **every input after them is a parameter**.

```matlab
ReturnFn      = @(d..., aprime..., a..., semiz..., z..., e..., <params>) ...
FnsToEvaluate = @(d..., aprime..., a..., semiz..., z..., e..., <params>) ...
```

Two examples:

```matlab
% one d (labour), one asset, one markov shock; r, w, sigma are parameters
ReturnFn = @(h, aprime, a, z, r, w, sigma) ...

% two assets, no d, markov z and iid e; agej and Jr are parameters
ReturnFn = @(a1prime, a2prime, a1, a2, z, e, agej, Jr) ...
```

The names you give the leading inputs are irrelevant — only their *position* and *count* matter. The names of the trailing inputs matter a great deal: each is looked up **by name** in the `Parameters` struct.

### Age `j` and transition period `t` are not inputs

Age `j` is part of the state space, and it is the reason `V`, `Policy` and `AgentDist` all carry a trailing `N_j` dimension. The transition period `t` does the same job on a transition path, adding a trailing `T` dimension. **Neither is ever an input to `ReturnFn` or `FnsToEvaluate`.** Notice that the prefix count derived below — `l_d + l_aprime + l_a + l_semizze` — has no term for `j` and no term for `t`.

The reason is that the toolkit solves one period at a time. When it builds the return matrix for period `j`, it has already selected that period's grids (`z_gridvals_J(:,:,j)` and so on) and that period's parameter values, so the function being evaluated is already the age-`j` return function. Handing it `j` as well would be redundant. The same holds on a transition path: the period-`t` entries of `ParamPath` and `PricePath` are written into `Parameters` before period `t` is solved, so `t` reaches your function purely through the values its parameters take.

Age- and time-dependence therefore enter through the **parameters**, in two ways:

- **Implicitly.** Any parameter of size `[1,N_j]` or `[N_j,1]` is indexed by the current `j` automatically, so `@(aprime,a,z,kappa_j)` sees the age-`j` value of `kappa_j` without you doing anything. This covers most age-dependence — survival probabilities, deterministic earnings profiles, age-varying tax rates.
- **Explicitly.** When you need the age index *itself* — a retirement condition, say — put it in `Parameters` as an ordinary age-dependent parameter and pass it like any other: `Params.agej=1:1:N_j`, then `@(aprime,a,z,agej,Jr) ...` with a branch on `agej>=Jr` inside. This is the `agej` in the second example above. It is a parameter, positioned after all the states, not a state.

A corollary worth keeping in mind: because a single `ReturnFn` is used for every age, it cannot change its own argument list with `j`. A model where some ages have a decision variable and others do not still declares one `n_d` covering all ages, and switches behaviour on an age-dependent parameter inside the function.

### The parameter inputs

Everything after the leading block is matched against `Parameters`:

- the name must be a field of `Parameters`, otherwise you get `Cannot find the parameter X in the Parameters structure (it is needed as an input to the ReturnFn)`;
- the value must be either a **scalar** or an **age-dependent vector** of size `[1,N_j]` or `[N_j,1]`. Anything else errors. A scalar is broadcast across ages; an age-dependent vector is indexed by `j` automatically.
- ordering among the parameters is free — they are matched by name, not position.

`FnsToEvaluate` parameters follow the same rule (scalar or age-dependent).

> **The commonest mistake.** Because the split is purely positional, getting the leading count wrong does not produce a helpful message — it shifts the boundary. Omit one state variable and the toolkit treats that state as a parameter name and reports `Cannot find the parameter z`. Add one input too many and a genuine parameter gets silently consumed as a state. If you see a "cannot find the parameter" error naming something that is obviously a state variable, the leading input list is wrong.

### Other functions with their own prefix

The same "leading inputs are positional, the rest are parameters" rule applies to the toolkit's other user-supplied functions, but each has its own prefix:

| function | leading inputs |
|---|---|
| `aprimeFn` for `experienceasset` | `(d2..., a2...)` — the decisions driving the asset, then the asset's current value |
| `aprimeFn` for `riskyasset` | `(d2..., d3..., u...)` — the decisions driving the asset, then the between-period shock |
| `SemiExoStateFn` | `(semiz, semizprime, dsemiz)` |
| `WarmGlowBequestsFn` (Epstein–Zin) | `(aprime, ...)` |

Here `d2`/`d3` are the decision blocks identified by `vfoptions.l_dexperienceasset` / `vfoptions.refine_d`; see below.

---

## How the toolkit works out the two spaces

You never tell the toolkit how many leading inputs your functions have. It **counts** them, from the grid-size inputs and the `vfoptions` flags, and then splits your function's argument list at that count.

The count is built up as follows.

**Decision variables.**

```
l_d = length(n_d),  or 0 if n_d(1)==0
```

**Endogenous states.** The current states and the chosen next-period states are counted separately, because they can differ:

```
l_a      = length(n_a),  or 0 if n_a(1)==0
l_aprime = l_a
```

then `l_aprime` is decremented by one for **each** alternative-asset flag that is set — `experienceasset`, `experienceassetz`, `experienceassete`, `experienceassetze`, `experienceassetu`, `experienceassetsemiz`, `riskyasset`, `residualasset`, `inheritanceasset`. Each such asset is a state whose next-period value is produced by `aprimeFn`, so it drops out of the action space while staying in the state space.

**Exogenous states.** The three shock blocks are counted into one number:

```
l_z       = length(n_z),  or 0 if prod(n_z)==0
l_semizze = l_z + length(vfoptions.n_semiz) + length(vfoptions.n_e)
```

with the `n_semiz`/`n_e` terms only added when those blocks are non-empty. Note the *count* is a sum, but the *order* in the argument list is `semiz`, then `z`, then `e`.

**`refine_d`.** For `riskyasset`, `vfoptions.refine_d` splits the decision variables by role: `d1` enter the `ReturnFn` only, `d2` enter the `aprimeFn` only, `d3` enter both (and a fourth block `d4` is the semi-exogenous decision, when present). Since `d2` never appears in the return function, the count is reduced:

```
l_d = l_d - vfoptions.refine_d(2)
```

so `ReturnFn` takes `(d1, d3, d4, ...)`. Correspondingly `aprimeFn` takes `(d2, d3, u, ...)`, i.e. `l_d - refine_d(1) - refine_d(4)` decisions.

**The split.** With those in hand:

```
prefix = l_d + l_aprime + l_a + l_semizze
ReturnFnParamNames = inputs (prefix+1) .. end
```

and each of those names is then checked against `Parameters` for existence and for being scalar or age-dependent.

**`FnsToEvaluate` is counted slightly differently.** Rather than re-deriving `l_d` and `l_aprime` from the flags, the evaluation commands read the action-space size straight off the policy:

```
l_daprime = size(PolicyValues,1)          % = l_d + l_aprime
prefix    = l_daprime + l_a + l_z         % l_z here already includes semiz and e
```

This is why `FnsToEvaluate` and `ReturnFn` normally have the same leading inputs: both counts come out to `l_d+l_aprime+l_a+l_semizze`. The one case where they differ is `refine_d`: `Policy` retains the `d2` rows, so `FnsToEvaluate` sees them even though `ReturnFn` does not.

---

## Variants

**Permanent types.** With `N_i` permanent types, `V`, `Policy` and `AgentDist` become **structs** with one field per permanent type (named by the type's name, or `ptype001`, `ptype002`, … if unnamed). Each field holds an array of exactly the shape described above, computed with that type's own `n_a`/`n_z`/`N_j` — which may differ across types.

**Transition paths.** `VPath`, `PolicyPath` and `AgentDistPath` append a transition-period dimension `T` after the shapes above.

**Exotic preferences.** `QuasiHyperbolic` in finite horizon returns a third output `Valt` alongside `V` and `Policy`; `endogenousexit` in infinite horizon returns `ExitPolicy` (and `PolicyWhenExit` when `endogenousexit=2`). All have state-space shape.

---

## See also

- [ExogenousShocks.md](ExogenousShocks.md) — what `z`, `e` and `semiz` are, their grid shapes, timing, and the `lowmemory` levels the combination allows.
