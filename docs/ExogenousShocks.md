# Exogenous Shocks

The VFI Toolkit supports three kinds of exogenous shock. This document describes how to set each one up, how to combine them, and the `lowmemory` options that the combination determines.

| Type | Persistence | Enters `Policy`? | Transition input |
|------|-------------|------------------|------------------|
| **z** (markov) | persistent | yes | `pi_z` (Markov matrix) |
| **e** (iid) | iid, *within-period* | yes | `pi_e` (distribution) |
| **semiz** (semi-exogenous) | persistent, **decision-dependent** | yes | `SemiExoStateFn` → `pi_semiz` |

All three are "start-of-period" states that the agent sees before choosing, so `Policy` depends on them.

> **Not to be confused with `u`.** The between-period iid shock `u` used by `riskyasset` / `experienceassetu` is *not* one of these. `u` is realised *after* the decision (it perturbs next period's asset via `aprimeFn(d,...,u)`), so `Policy` does **not** depend on `u`; it is integrated out by `pi_u`. See the riskyasset / experienceassetu command docs.

---

## z — markov shock

A persistent shock with a Markov transition matrix. You provide `n_z`, `z_grid`, and `pi_z`.

### `z_grid`: stacked column vs joint

With **more than one** markov variable (`n_z = [n_z1, n_z2, ...]`), `z_grid` can be given in either of two forms:

- **Stacked column grid** — shape `[sum(n_z), 1]`. Each univariate grid written one beneath the next in a single column. Compact; the joint state space is only implicit.
  Example, `n_z = [3,2]`: a `5x1` column = the 3 values of `z1` followed by the 2 values of `z2`.

- **Joint grid** — shape `[prod(n_z), length(n_z)]`. Every point of the product space listed explicitly, one per row, each variable in its own column.
  Example, `n_z = [3,2]`: a `6x2` matrix; each row pairs one `z1` value with one `z2` value, covering all 6 combinations.

With a single markov variable the two coincide (`[n_z,1]`). Internally the toolkit always converts to the joint form (`z_gridvals_J`, shape `[prod(n_z), length(n_z), N_j]`).

### Age-dependent `z`

Both grid and transition can vary with age `j`:

- `z_grid`: `[sum(n_z), N_j]` (stacked column per age) or `[prod(n_z), length(n_z), N_j]` (joint per age).
- `pi_z`: `[prod(n_z), prod(n_z)]` (age-independent) or `[prod(n_z), prod(n_z), N_j]` (per age).

Alternatively set `vfoptions.ExogShockFn` (and the same on `simoptions`): a function called once per age `j` returning that age's `[z_grid, pi_z]` in the age-independent shapes. When supplied, the raw `z_grid`/`pi_z` inputs are ignored.

### Discretizing an AR(1) / VAR(1) into `(z_grid, pi_z)`

The toolkit ships discretization routines that return a `(grid, pi)` pair ready to pass in:

- **Single AR(1):** `discretizeAR1_FarmerToda`, `discretizeAR1_Rouwenhorst`, `discretizeAR1_Tauchen`, `discretizeAR1_TauchenHussey`.
- **AR(1) extensions:** `discretizeAR1wGM_FarmerToda` (Gaussian-mixture innovations), `discretizeAR1wSV_FarmerToda` (stochastic volatility).
- **Age-dependent (life-cycle) AR(1):** `discretizeLifeCycleAR1_FellaGallipoliPan`, `discretizeLifeCycleAR1_FellaGallipoliPanTauchen`, `discretizeLifeCycleAR1_KFTT`, `discretizeLifeCycleAR1wGM_KFTT`.
- **VAR(1):** `discretizeVAR1_FarmerToda`, `discretizeVAR1_Tauchen`, `discretizeLifeCycleVAR1_Tauchen`.

VAR / life-cycle routines naturally produce the multivariate / age-dependent grids described above.

---

## e — iid shock

An iid shock drawn fresh each period (within-period). Because it has no persistence, its "transition" is just a distribution.

Set on `vfoptions` (and `simoptions`):

- `n_e`, `e_grid`, `pi_e`.
- `e_grid` accepts the same **stacked column** (`[sum(n_e),1]`) and **joint** (`[prod(n_e), length(n_e)]`) forms as `z_grid`, plus their age-dependent variants (`…, N_j`).
- `pi_e` is a distribution: `[prod(n_e), 1]`, or `[prod(n_e), N_j]` for an age-dependent iid distribution.
- `vfoptions.EiidShockFn` is the iid analogue of `ExogShockFn`: called per age to return `[e_grid, pi_e]`; raw inputs then ignored.

Use `e` (rather than `z`) when a shock genuinely has no persistence — it is cheaper because no full Markov matrix is stored and it can be integrated out more aggressively (see `lowmemory`).

---

## semiz — semi-exogenous shock

A persistent state whose transition probabilities **depend on a decision variable** (the "semi-exogenous decision"). Its transition is built from a user function rather than a fixed matrix.

Set on `vfoptions` (and `simoptions`):

- `n_semiz`, `semiz_grid` — `semiz_grid` must be a **column vector** (stacked, `[sum(n_semiz),1]`).
- One of:
  - `SemiExoStateFn` — `prob = SemiExoStateFn(semiz, semizprime, dsemiz, <params>)`, giving the probability of moving from `semiz` to `semizprime` given the decision `dsemiz`; the toolkit assembles `pi_semiz_J`. Extra scalar parameters are passed by name.
  - or a precomputed `pi_semiz` directly.
- `l_dsemiz` (default `1`) — how many decision variables drive the semiz transition. These are the **last** `l_dsemiz` decision variables in `n_d`.

The setup routine `SemiExogShockSetup_FHorz` produces `options.semiz_gridvals_J` and `options.pi_semiz_J` (shape `[prod(n_semiz), prod(n_semiz), prod(n_dsemiz), N_j]`).

For asset types that split `d` by role (e.g. `riskyasset` with `vfoptions.refine_d`), the semiz decision is category `d4` — the last block of `d`.

---

## Combining shocks

Any subset of `{z, e, semiz}` may be used together. In the state space the order is `(a, semiz, z, e)` — semiz sits directly after the endogenous states, then z, then e.

When **semiz and z** are both present they are handled as a combined markov `bothz`:

```
pi_bothz = kron(pi_z, pi_semiz)
```

so **semiz is the inner (fast) index and z is the outer (slow) index**: `N_bothz = N_semiz * N_z`, and the semiz block belonging to a given `z_c` is `(z_c-1)*N_semiz + (1:N_semiz)`. `e`, being iid, is layered on top of whatever markov/semiz structure exists.

---

## `lowmemory` levels

`vfoptions.lowmemory` / `simoptions.lowmemory` trade GPU memory for looping: a higher level loops over more of the exogenous dimensions (building the return matrix in smaller pieces) and gives an **identical result** to level 0, just using less memory. **Which levels are valid is determined by which of `{z, e, semiz}` the model has** — do not request a level above what the shock combination allows. Higher level = more looping = less memory.

"Parallel over X" means X is vectorised; "loop over X" means it is looped. When `semiz` and `z` are both present they form the combined markov `bothz` (see [Combining shocks](#combining-shocks)), which can either be split (parallel semiz / loop z) or looped jointly as one index.

| shocks present | valid `lowmemory` | what each level does |
|---|---|---|
| none | `0` | everything vectorised |
| one of `z` / `e` / `semiz` | `0, 1` | `=1`: loop that shock |
| `z` & `e` | `0, 1, 2` | `=1`: loop e. `=2`: outer loop z, inner loop e |
| `semiz` & `e` | `0, 1, 2` | `=1`: loop e. `=2`: outer loop semiz, inner loop e |
| `z` & `semiz` | `0, 1, 2` | `=1`: parallel over semiz, loop over z. `=2`: single joint loop over semiz & z (`bothz`) |
| `z` & `semiz` & `e` | `0, 1, 2, 3` | `=1`: parallel over semiz & z, loop over e. `=2`: parallel over semiz, outer loop over z, inner loop over e. `=3`: joint outer loop over semiz & z (`bothz`), inner loop over e |

`lowmemory=0` always vectorises everything; it is the default and the fastest when memory permits.

---

## See also

- `ValueFnIter_Case1_FHorz.md`, `StationaryDist_FHorz_Case1.md` — where these options are consumed.
- Command docs for `riskyasset` / `experienceassetu` — for the between-period `u` shock, which is distinct from the three shocks above.
