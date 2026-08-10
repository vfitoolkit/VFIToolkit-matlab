# Project: Exogenous-shock timing compliance (pi_e "Reading A")

**Status: in progress.** Scoped 2026-08-10. **Categories A, E, D, B and F complete 2026-08-10** (A: three ExogShockSetup files + two SemiExogShockSetup files, see category A for the TPath "Option B" trimming decision; E: all 16 sim raws; D: all 8 dist iterations; B: all 515 VFI raws in 35 batches, 1029 sites — see the ticked checklist; F: all 39 ValueFnFromPolicy files, 61 sites). **Category C complete 2026-08-10** (C1-C7; see the staged checklist). **Residual audits complete 2026-08-10:** FieldExp builds full-size arrays (supersets of what the raws read — compatible); no wrapper size-checks reject the new shapes; the 4 wrappers' stale size comments corrected (`N_j-1` / `V_Jplus1` clauses); Ambiguity audited and deferred to the user's planned overhaul session. **All functional code editing is DONE. Remaining: GPU validation** (see plan below; caveats: fixed-seed sims won't reproduce old paths; MEP must be run only against the completed C5 tree; results differ from old code only for age-dependent pi_e).

**Follow-ups surfaced during category B (2026-08-10):**
1. **Pre-existing bug FIXED 2026-08-10 (user approved):** the five recent `ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeN_{DC2A, DC2A_GI2A, DC2A_GI2A_nod1, GI2A, GI2A_nod1}_e_raw.m` files never integrated the e-dimension of `vfoptions.V_Jplus1` in their final-period branch — the branch would in fact have crashed (the downstream `reshape` had an element-count mismatch by a factor of `N_e`, so the path was untested/unrunnable). Each got the sibling-pattern one-liner: `EVpre=squeeze(sum(reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3));` replacing the bare reshape.
2. **AmbiguityAversion — audited 2026-08-10, deliberately left alone:** `ambiguity_pi_e_J` is user-supplied (the wrapper only broadcasts `ambiguity_pi_z` to `N_j` slices, compatible with the guarded reads); the raws' consumption indices were fixed in category B. Consequence of the convention: with `V_Jplus1`, user-supplied age-dependent `ambiguity_pi_e_J` needs `N_j+1` age-columns per prior (fails loudly otherwise). **User plans an ambiguity overhaul in a separate session — documenting/adjusting this belongs there; do not touch ambiguity in this project.**

**Implementation rules:**
- **Never use helper functions** (local subfunctions): write the logic inline in each branch/file, even when this duplicates code. This is the toolkit's house style.
- Do not add guards or special-casing for `N_j<3` edge cases; let them fail silently/naturally.

The timing conventions in [docs/ExogenousShocks.md](docs/ExogenousShocks.md) ("Timing" section, written 2026-08-10) are the statement of intent. The doc is written as if already true; this project makes the code comply. Do **not** "fix" code back to the old timing, and do not edit the doc's Timing section to match old code.

## The convention (target state)

- **Grids** (z, e, semiz alike): slice `j` = values realized **in** period `j`. *(Code already complies — no changes.)*
- **`pi_z_J(:,:,j)` / `pi_semiz_J(:,:,:,j)`**: transition **from `j` to `j+1`**. *(Code already complies — only the input-shape flexibility below is new.)*
- **`pi_e_J(:,j)`**: distribution of the e realized **in period `j`** ("Reading A"). Column 1 is typically ignored (period-1 e comes from `jequaloneDist`). With `vfoptions.V_Jplus1`, `pi_e_J` needs `N_j+1` columns.
- `pi_z` is accepted as `[N_z,N_z,N_j]` **or** `[N_z,N_z,N_j-1]`; `pi_e` as `[N_e,1]`, `[N_e,N_j]` or `[N_e,N_j+1]`.
- **No padding — trim instead (decided 2026-08-10):** the setup routines *output* only what can be read. `pi_z_J` is emitted with `N_j-1` slices, or `N_j` slices when `isfield(options,'V_Jplus1')`; `pi_e_J` with `N_j` columns, or `N_j+1` with `V_Jplus1`; semiz transitions analogously (`N_j-1` vs `N_j`). Any never-readable final slice/column in the input is dropped; any out-of-range read downstream fails loudly as an index error. (`options` is simoptions in the dist paths and never has `V_Jplus1`, which is fine: slice `N_j` / column `N_j+1` are only ever needed in value-function contexts, where `options` is vfoptions.)

## What the code currently does ("Reading B")

Everywhere except two files (see category G), consumption sites use `pi_e_J(:,jj)` at age `jj` to weight/draw the e realized at age `jj+1`. Concretely: the age-`jj+1` e is currently distributed `pi_e_J(:,jj)`, and `pi_e_J(:,1)` **is** used (it governs the age-2 draw) while `pi_e_J(:,N_j)` is only used against `V_Jplus1`.

The toolkit is internally **inconsistent**: the newer P-transition-matrix code (`SubCodes/Ptransitionmatrix/CreatePTransitionMatrix_J.m`, `EvaluateFnOnAgentDist/FHorz/EvalFnOnAgentDist_AutoCorrTransProbs_FHorz.m`) already implements Reading A (`pi_e_J(:,jj+1)`, with comments saying so).

**Behavioral impact of the fix:** results are *identical* whenever `pi_e` is age-constant (the common case). Only age-dependent `pi_e` users see a change (a one-period shift, i.e. the bug fix).

## The core fix recipe

At a consumption site for the e realized in period `t+1`, change column index `t` → `t+1`:

```matlab
% VFI raws (expectation over V at age jj+1), before → after:
EV=sum(EV.*pi_e_J(1,1,:,jj),3);   →   EV=sum(EV.*pi_e_J(1,1,:,jj+1),3);
% final period against V_Jplus1:
EV=sum(EV.*pi_e_J(1,1,:,N_j),3);  →   EV=sum(EV.*pi_e_J(1,1,:,N_j+1),3);

% Dist iterations (populate age jj+1), before → after:
pi_e=sparse(gather(pi_e_J(:,jj)));  →  pi_e=sparse(gather(pi_e_J(:,jj+1)));
```

**Construction sites are already correct** under Reading A (`pi_e_J(:,jj)=...` storing the period-`jj` distribution from age-`jj` parameters) — do not touch them. Files that only construct: `SubCodes/ExoShocks/ExogShockSetup_FHorz*.m` loops, `StationaryDist/FHorz/FieldExp/StationaryDist_FHorz_FieldExp_Treatment.m`.

## Work items by category

### A. Input handling / shape checks — `SubCodes/ExoShocks/`

Files: `ExogShockSetup_FHorz.m` **[DONE 2026-08-10]**, `ExogShockSetup_FHorz_PType.m` **[DONE 2026-08-10; also adds warnings when the last-dim-equals-N_i ptype detection is ambiguous with an age-dependent shape (N_i==N_j, N_j-1 for pi_z, N_j+1 for pi_e), stating the ptype interpretation is applied]**, `ExogShockSetup_FHorz_TPath.m` **[DONE 2026-08-10, "Option B" — see below]**, plus the semiz analogues `SubCodes/SemiExoShocks/SemiExogShockSetup_FHorz.m` **[DONE 2026-08-10]** and `SemiExogShockSetup_FHorz_PType.m` **[DONE 2026-08-10]** (`pi_semiz_J` output is `[N_semiz,N_semiz,N_dsemiz,N_j-1]`, or `N_j` slices with `V_Jplus1`; inputs accept the `N_j-1` form; `SemiExoStateFn` is evaluated for `jj=1:N_j-1` only, +`N_j` with `V_Jplus1`; rows-sum-to-one check loops over existing slices; `SemiExogShockSetup_InfHorz.m` unaffected). Note: the PType semiz variant's last-dim-equals-`N_i` slicing has the same ptype-vs-age ambiguity class as the z/e PType setup but no warnings were added there (they would fire once per ptype inside the loop) — flag if wanted.

**Category A is complete.** Docs updated in step: `docs/ExogenousShocks.md` semiz section now states the `N_j-1` output shape.

**TPath "Option B" decision (2026-08-10):** the TPath setup trims too — no structural padding slot anywhere. New TPath output contract: `pi_z_J` is `[N_z,N_z,N_j-1]` (fastOLG=0, (z,z',j)) / `[N_j-1,N_z,N_z]` (fastOLG=1, (j,z',z)); `.pi_z_J_T`/`.pi_z_J_alt`/`.pi_z_J_T_alt` likewise have `N_j-1` age slots; `pi_z_J_sim` is built directly from the trimmed `pi_z_J`. `pi_e_J` keeps its `[N_e,N_j]` / `[N_a*N_j,(1,)N_e]` shapes with column 1 (the j=1 block) never read; `pi_e_J_sim`/`.pi_e_J_sim_T` block `jj` now holds `pi_e_J(:,jj+1)` (was `(:,jj)` — the Reading-B `temp(1:end-N_a,:)` / `pi_e_J(:,1:end-1)` selections became `temp(N_a+1:end,:)` / `pi_e_J(:,2:end)`). Inputs additionally accept `[N_z,N_z,N_j-1]` and `[N_z,N_z,N_j-1,T]` (a 3-D third dim of N_j-1 resolves as age-dependent-trimmed with a warning, not as T=N_j-1; no `[N_e,N_j+1]` form in TPath since V_Jplus1 does not arise on a path). **Consequence for category C:** every fastOLG/slowOLG consumer of the TPath pi_z objects must be reworked to the `N_j-1` age-axis contract (array algebra, not just index shifts), and consumers of `pi_e_J` must drop their `[1,1:end-1]`-style shifts and treat the j=1 block as unused.

The pattern, as implemented in `ExogShockSetup_FHorz.m` (replicate in the others, inline, no helper functions):

1. **Sizes defined once up front:** `N_jpiz = N_j-1` (`N_j` if `isfield(options,'V_Jplus1')`); `N_jpie = N_j` (`N_j+1` with `V_Jplus1`). Semiz analogue: `N_j-1` vs `N_j`.
2. **`pi_z`:** accept `[N_z,N_z]` (broadcast), `[N_z,N_z,N_j]` (trim to `N_jpiz` slices), `[N_z,N_z,N_j-1]` (pass through; **error** if `V_Jplus1` since slice `N_j` is then needed). No padding anywhere.
3. **`pi_e`:** accept `[N_e,1]` (broadcast to `N_jpie` columns), `[N_e,N_j]` (**error** if `V_Jplus1`), `[N_e,N_j+1]` (trim to `N_jpie` columns).
4. **Shock-fn interactions:** `EiidShockFn` + `V_Jplus1` **errors** (period-`N_j+1` parameters are unobtainable); `ExogShockFn` + `V_Jplus1` is **allowed** (slice `N_j` = `ExogShockFn` at age-`N_j` parameters). `ExogShockFn` loops store `pi_z_J` slices only for `jj<=N_jpiz`.
5. **Bundled latent-bug fix:** `pi_z_J`/`pi_e_J` construction is decoupled from the grid-shape branches (previously an age-independent `pi_z`/`pi_e` combined with an age-dependent grid passed through un-broadcast, giving a 2-D `pi_z_J` / `[N_e,1]` `pi_e_J`).
6. `ExogShockSetup_FHorz_TPath.m` also *precomputes* `pi_e_J_sim` / `pi_e_J_sim_T` blocks whose comments state Reading-B timing ("block jj holds pi_e_J(:,jj), the e' distribution when moving jj->jj+1", lines ~803-902): re-derive these blocks under Reading A (block `jj` should hold `pi_e_J(:,jj+1)`) and fix the comments.
7. Update the shape documentation in file headers, and the shape-check warnings in the top-level wrappers (`ValueFnIter_Case1_FHorz.m` etc.) for the new output sizes.

**Sequencing consequence:** because output slices/columns shrink, category A must land together with the consumer updates. In particular the category-E simulation loop fix is now a **prerequisite** (its discarded final-iteration draw reads `cumsumpi_z_J(:,:,N_j)`, which no longer exists without `V_Jplus1`), and any internal code that pre-builds `pi_z_J`/`pi_e_J` (PType wrappers, FieldExp, `vfoptions.alreadygridvals==1` users) must emit/supply the new sizes. Re-grep for `pi_z_J(:,:,N_j)`-without-`V_Jplus1` readers (e.g. `kron`-built `pi_bothz`, PType/TPath plumbing) before landing.

### B. ValueFnIter FHorz raws — 515 files (batch plan below)

**Batch plan (2026-08-10).** The 515 files are processed in 35 batches of at most 16 files. Each batch is a union of *whole* directories under `ValueFnIter/FHorz/` (no directory is split), so the table below fully determines membership: a batch = all category-B files in its listed directories, in sorted order. Families are kept together; the irregular solver families (paralleld2, EpsteinZin, AmbiguityAversion, GulPesendorfer) are isolated in the final two batches for hand review.

**Per-batch procedure:**
1. **Dry run:** list every `pi_e_J`-referencing line in the batch. Classify each as (a) pure reshape/self-assignment with no age index (e.g. `pi_e_J=shiftdim(pi_e_J,-2);`) — leave untouched; (b) consumption whose *final* index inside `pi_e_J(...)` is `jj` or `N_j` — transform; (c) anything else (already `jj+1`, an unexpected index variable, an assignment into `pi_e_J`) — **halt the batch and review by hand**.
2. **Transform:** within the `pi_e_J(...)` index list only, final index `jj` → `jj+1` and `N_j` → `N_j+1` (the `N_j` sites are exactly the `V_Jplus1` branches; the setup supplies column `N_j+1` in that case). No other tokens on the line change.
3. **Verify:** (i) scripted char-level check that every git-diff changed line differs from its old counterpart only by the `+1` insertions inside `pi_e_J(...)`; (ii) changed-line count equals the dry-run type-(b) site count; (iii) zero remaining type-(b) sites in the batch.
4. **Record:** tick the batch in the checklist with its site count.

Batches run consecutively without per-batch approval once the plan is approved, halting for review only on a type-(c) flag.

**Batch checklist** (`[dirs (files)]`):
- [x] B01 [16]: (root) (4); DivideConquer (4); DivideConquer/DC2A (4); DivideConquerGridInterpLayer (4)
- [x] B02 [16]: DivideConquerGridInterpLayer/DC2A (4); GridInterpLayer (4); GridInterpLayer/GI2A (4); SemiExo (4)
- [x] B03 [16]: SemiExo/DivideConquer (4); SemiExo/DivideConquer/DC2A (4); SemiExo/DivideConquerGridInterpLayer (4); SemiExo/DivideConquerGridInterpLayer/DC2A (4)
- [x] B04 [16]: SemiExo/GridInterpLayer (4); SemiExo/GridInterpLayer/GI2A (4); ExperienceAsset (8)
- [x] B05 [16]: ExperienceAsset/DivideConquer (8); ExperienceAsset/DivideConquerGridInterpLayer (8)
- [x] B06 [16]: ExperienceAsset/ExpAssetSemiExo (8); ExperienceAsset/ExpAssetSemiExo/DivideConquer (8)
- [x] B07 [16]: ExperienceAsset/ExpAssetSemiExo/DivideConquerGridInterpLayer (8); ExperienceAsset/ExpAssetSemiExo/GridInterpLayer (8)
- [x] B08 [16]: ExperienceAsset/GridInterpLayer (8); ExperienceAssete (8)
- [x] B09 [16]: ExperienceAssete/DivideConquer (4); ExperienceAssete/DivideConquerGridInterpLayer (4); ExperienceAssete/ExpAsseteSemiExo (4); ExperienceAssete/ExpAsseteSemiExo/DivideConquer (4)
- [x] B10 [16]: ExperienceAssete/ExpAsseteSemiExo/DivideConquerGridInterpLayer (4); ExperienceAssete/ExpAsseteSemiExo/GridInterpLayer (4); ExperienceAssete/GridInterpLayer (4); ExperienceAssetsemiz (4)
- [x] B11 [12]: ExperienceAssetsemiz/DivideConquer (4); ExperienceAssetsemiz/DivideConquerGridInterpLayer (4); ExperienceAssetsemiz/GridInterpLayer (4)
- [x] B12 [16]: ExperienceAssetu (8); ExperienceAssetu/DivideConquer (8)
- [x] B13 [16]: ExperienceAssetu/DivideConquerGridInterpLayer (8); ExperienceAssetu/ExpAssetuSemiExo (8)
- [x] B14 [16]: ExperienceAssetu/ExpAssetuSemiExo/DivideConquer (8); ExperienceAssetu/ExpAssetuSemiExo/DivideConquerGridInterpLayer (8)
- [x] B15 [16]: ExperienceAssetu/ExpAssetuSemiExo/GridInterpLayer (8); ExperienceAssetu/GridInterpLayer (8)
- [x] B16 [16]: ExperienceAssetz (4); ExperienceAssetz/DivideConquer (4); ExperienceAssetz/DivideConquerGridInterpLayer (4); ExperienceAssetz/ExpAssetzSemiExo (2); ExperienceAssetz/ExpAssetzSemiExo/DivideConquer (2)
- [x] B17 [16]: ExperienceAssetz/ExpAssetzSemiExo/DivideConquerGridInterpLayer (2); ExperienceAssetz/ExpAssetzSemiExo/GridInterpLayer (2); ExperienceAssetz/GridInterpLayer (4); ExperienceAssetz/QuasiHyperbolic (4); ExperienceAssetz/QuasiHyperbolic/DivideConquer (4)
- [x] B18 [16]: ExperienceAssetz/QuasiHyperbolic/DivideConquerGridInterpLayer (4); ExperienceAssetz/QuasiHyperbolic/GridInterpLayer (4); ExperienceAssetze (4); ExperienceAssetze/DivideConquer (4)
- [x] B19 [14]: ExperienceAssetze/DivideConquerGridInterpLayer (4); ExperienceAssetze/ExpAssetzeSemiExo (2); ExperienceAssetze/ExpAssetzeSemiExo/DivideConquer (4); ExperienceAssetze/ExpAssetzeSemiExo/DivideConquerGridInterpLayer (4)
- [x] B20 [16]: ExperienceAssetze/ExpAssetzeSemiExo/GridInterpLayer (4); ExperienceAssetze/GridInterpLayer (4); ExperienceAssetze/QuasiHyperbolic (8)
- [x] B21 [11]: ExperienceAssetze/QuasiHyperbolic/DivideConquer (11)
- [x] B22 [9]: ExperienceAssetze/QuasiHyperbolic/DivideConquerGridInterpLayer (9)
- [x] B23 [16]: ExperienceAssetze/QuasiHyperbolic/GridInterpLayer (12); RiskyAsset/DivideConquer (4)
- [x] B24 [16]: RiskyAsset/DivideConquerGridInterpLayer (4); RiskyAsset/GridInterpLayer (4); RiskyAsset/Raw (8)
- [x] B25 [16]: RiskyAsset/RiskyAssetSemiExo (8); RiskyAsset/RiskyAssetSemiExo/DivideConquer (4); RiskyAsset/RiskyAssetSemiExo/DivideConquerGridInterpLayer (4)
- [x] B26 [12]: RiskyAsset/RiskyAssetSemiExo/GridInterpLayer (4); ExoticPrefs/QuasiHyperbolic (8)
- [x] B27 [16]: ExoticPrefs/QuasiHyperbolic/DivideConquer (16)
- [x] B28 [16]: ExoticPrefs/QuasiHyperbolic/DivideConquerGridInterpLayer (16)
- [x] B29 [16]: ExoticPrefs/QuasiHyperbolic/GridInterpLayer (16)
- [x] B30 [8]: ExoticPrefs/QuasiHyperbolic/QuasiHyperbolicSemiExo (8)
- [x] B31 [16]: ExoticPrefs/QuasiHyperbolic/QuasiHyperbolicSemiExo/DivideConquer (16)
- [x] B32 [16]: ExoticPrefs/QuasiHyperbolic/QuasiHyperbolicSemiExo/DivideConquerGridInterpLayer (16)
- [x] B33 [16]: ExoticPrefs/QuasiHyperbolic/QuasiHyperbolicSemiExo/GridInterpLayer (16)
- [x] B34 [14] *(hand review)*: SemiExo/paralleld2 (1); SemiExo/paralleld2/DivideConquer (1); RiskyAsset/EpsteinZin (4); ExoticPrefs/AmbiguityAversion (4); ExoticPrefs/EpsteinZin (4)
- [x] B35 [3] *(hand review)*: ExoticPrefs/GulPesendorfer (3)

All `*_e_raw.m` (and QuasiHyperbolic/Epstein-Zin/GulPesendorfer etc. variants that take expectations over e) under `ValueFnIter/FHorz/`. Apply the core recipe: `jj` → `jj+1` in the backward-induction expectation; `N_j` → `N_j+1` in the `V_Jplus1` branch. Note most raws `shiftdim` `pi_e_J` up front — the column index is then in the 4th (or higher) position; the recipe is the same.

Plumbing: the wrappers pass `vfoptions.pi_e_J` straight through, so once the setup (category A) emits the `N_j+1`-column form when `V_Jplus1` is present, no wrapper changes should be needed beyond the shape checks.

### C. TransitionPaths FHorz — re-scoped 2026-08-10: ~450 files in 7 stages

True scope (the original 108 was only the files visible to the age-indexed-`pi_e_J` grep): 218 fastOLG + 200 slowOLG `ValueFnSingleStep` files + 32 `AgentDistSingleStep` files; many need more than one fix. Stages:

- [x] **C1 (60 files) [DONE 2026-08-10]:** slowOLG vectorized e-shift — `pi_e_J(:,[1,1:end-1])` / `pi_e_J(1,1,:,[1,1:end-1])` → plain `pi_e_J` (Reading A aligns naturally); the 60 "first column is padding" comments rewritten. One site per file; verified zero leftovers.
- [x] **C2 (32 files) [DONE 2026-08-10]:** per-age `pi_e_J(:,jj)` / `pi_e_J(1,1,:,jj)` sites → `jj+1` (category-B driver: 32 sites, zero flags, zero residual). The earlier "92 files" estimate double-counted the 60 C1 files, which matched the scoping grep only via their (since rewritten) comments.
- [x] **C3 (100 files) [DONE 2026-08-10]:** fastOLG e-block selections — `pi_e_J(1:end-N_a,...)` → `pi_e_J(N_a+1:end,...)` (keep age blocks `2..N_j`, mirroring the `V(N_a+1:end,...)` selection); exactly one site per file (38 `,:)` + 62 `,:,:)` variants), zero leftovers. Selection-only change; the adjacent "I use zeros in j=N_j" comments are left for C4, which restructures those lines. The MEP branches' e-multiply (`sum(reshape(V,...).*pi_e_J,3)`) weights each age slot's own e-dim by its own block and is already correct under Reading A — no change.
- [~] **C4 (128 files) — "Recipe X" adopted 2026-08-10:** instead of restructuring each multiply (pad-after), each fastOLG consumer appends one explicit zero transition row for `j=N_j` at the point where its pi array is age-first — `pi_z_J=cat(1,pi_z_J,zeros(1,N_z,N_z,'gpuArray'));` — a meaningful zero ("no continuation past the final period") that participates in real arithmetic; all downstream algebra unchanged and numerically identical (the `j=N_j` slot's product was zero via V-side padding before, and is zero via the zero row now).
  - [x] **Group (i) [DONE 2026-08-10]: plain fastOLG + QuasiHyperbolic/fastOLG, 60 files** (13 further `noz` files match only via comments — untouched). One append after `N_z=prod(n_z);` per file + 34 header comments updated; all code-level `pi_z_J` uses verified to be the two known multiply forms (`shiftdim -2`, DC2A `-4`); exemplars hand-checked. **The MEP `EVpre==1` branches in these files are now dimension-conforming with zero final-period continuation — provisional pending the C5 audit.**
  - [x] **Group (ii) [DONE 2026-08-10]: ExpAsset families, 32 files** (`ExpAsset`/`ExpAssetz`/`ExpAssetze` fastOLG) — one append per file after `N_z=prod(n_z);` + 32 header notes; all code-level `pi_z_J` sites verified to be exactly `EV=EV.*shiftdim(pi_z_J,-2);` (32) or `EV=EV.*shiftdim(permute(pi_z_J,[1,3,2]),-2);` (32) — the permuted form keeps the zero row, so the gather-fusion V-side pads are untouched. Same provisional-MEP flag as group (i).
  - [x] **Group (iii) [DONE 2026-08-10]: SemiExo families, 32 files** — semiz append (`cat(1,pi_semiz_J,zeros(1,N_semiz,N_semiz,N_d2,'gpuArray'))`) inserted after the in-file `permute(pi_semiz_J,[4,2,1,3])` in all 32; the 16 with-z files additionally got the `pi_z_J` append (their hard-coded `[N_j,...]` `pi_bothz` reshapes then run unchanged, and `pi_bothz` inherits a zero final row). All code-level `pi_semiz_J`/`pi_z_J` lines verified against the known forms (never-fired halt = value-fn side of the C7 bothz audit passed in the process). Same provisional-MEP flag.

**C4 is complete** (60+32+32 = 124 files; the ~4-file gap vs the original 128 estimate is the noz comment-only matches). **Follow-up spawned to a separate session (2026-08-10):** restructure TPath so `pi_semiz_J` is created up-front alongside `pi_z_J`/`pi_e_J` (as in the non-TPath FHorz flow where `ExogShockSetup_FHorz` and `SemiExogShockSetup_FHorz` run back-to-back) and passed into the SingleStep raws in fastOLG orientation, instead of each raw permuting/appending in-file.
- [~] **C5 (in progress 2026-08-10):** MEP `EVpre==1` requires a NON-ZERO `j=N_j` transition row (user decision; MEP is the InfHorz-shooting machinery reusing FHorz fastOLG, the age axis is time along the path). **Audit done:** MEP builds its own time-indexed `pi_z_T_fastOLG` (`[T,N_z',N_z]`, all rows genuine, final row = transition into the continuing future) in `RecursiveGeneralEqmWithAggShocks_InfHorz.m` and passes it to the shared SingleStep dispatchers with `N_j=T`; `vfoptions.EVpre` is set explicitly on all 7 entry paths (`=1` only by RecursiveGE); MEP is z-only (errors on e). **Implemented (104 of 124 appended files):** the C4 `cat()` appends were MOVED inside each raw's existing `if vfoptions.EVpre==0` branch (first line(s) of the branch body), with an explanatory comment in the `elseif vfoptions.EVpre==1` branch. Contract: `EVpre==0` callers (TPath setup) supply `N_j-1` slices and the raw appends; `EVpre==1` callers (MEP) supply full-`N_j` genuine arrays untouched. In SemiExo files the shared post-branch `pi_bothz` code works under both branches (appended-to-`N_j` vs caller-supplied-`N_j`). **The 20 fastOLG e-raws with no EVpre branch** (4 plain + 8 QH + 8 SemiExo, the plain/GI1 e-variants; MEP does not support e, and these raws never read `vfoptions.EVpre` — the only "EVpre" occurrences are a local variable name) keep their top-of-file unconditional append, which is correct as-is; optional annotation not yet requested.
- [x] **C6 (32 files) [DONE 2026-08-10]:** AgentDistSingleStep — 8 files' `pi_e_J(:,jj)` → `jj+1` (reverse-order loop populates age `jj+1`, same as category D; verified 10 files ±10 lines exactly); sim-object consumers (`pi_z_J_sim`/`pi_e_J_sim`) shape-compatible unchanged (contracts fixed at construction in the TPath setup); 2 stale commented-out `pi_z_J(:,:,1:end-1)` reference lines updated to the trimmed contract. The SemiExo `pi_semiz_J_short` age-linear-indexing is deferred to the C7 audit.
- [x] **C7 [DONE 2026-08-10] — audits all passed; comments-only fix (31 lines, 16 files).** Findings: (1) VFI semiz families, 774 files checked — every `pi_z_J`/`pi_semiz_J` final-slice (`N_j`) read is `V_Jplus1`-guarded (zero unguarded), no whole-array/hard-coded-`N_j` reshapes; (2) StationaryDist + TPath AgentDist `pi_semiz_J_short` machinery (incl. IterFast all-age index constructions): age offsets max at `N_j-2` and age is the last dim (stride-invariant), so trimmed arrays work unchanged; (3) `CreatePTransitionMatrix(_J).m` and the AutoCorrTransProbs caller: loops `1:N_j-1`, `pi_e_J(:,jj+1)`, fully compliant. The only change: 31 stale `[N_semiz,N_semizshort,N_dsemiz,N_j]` shape comments corrected to `N_j-1]`.

Original scoping notes (superseded by the staging above):

Three sub-patterns under `TransitionPaths/FHorz/subcodes/`:

1. **ValueFnSingleStep slowOLG "vectorised" files** use a column-shift trick with an explicit Reading-B comment:
   ```matlab
   Vnext=sum(V.*shiftdim(pi_e_J(:,[1,1:end-1]),-2),3);  % first column is padding, never read
   ```
   Under Reading A the shift disappears entirely: `Vnext=sum(V.*shiftdim(pi_e_J,-2),3);` (column `jj+1` naturally aligns with `V(...,jj+1)`). Delete the stale comments.
2. **Other ValueFnSingleStep raws** (per-age loop style): core recipe, `pi_e_J(:,jj)` → `pi_e_J(:,jj+1)`.
3. **AgentDistSingleStep iterations**: core recipe, same as category D.

Also note `ValueFnIter_FHorz_TPath_SingleStep_e_raw.m:71` has a commented-out line that already uses `jj+1` — evidence the intended timing was Reading A all along.

### D. StationaryDist FHorz iterations — 8 files **[DONE 2026-08-10]**

All 8 `StationaryDist_FHorz_Iteration*_e_raw.m` (incl. `SemiExo/`, `nProbs/`) now use `pi_e_J(:,jj+1)` when populating age `jj+1`. Verified: no other `pi_e_J(:,jj)` consumption remains under `StationaryDist/` (FieldExp's two hits are construction, correct under Reading A); all `pi_z_J`/`pi_semiz_J` reads in this category use slices `1..N_j-1`, compatible with the trimmed arrays unchanged.

Original scoping notes:

`StationaryDist_FHorz_Iteration*_e_raw.m` under `StationaryDist/FHorz/` (incl. `SemiExo/`, `nProbs/`): core recipe (`kron(pi_e_J(:,jj),...)` / `sparse(...pi_e_J(:,jj))` → `jj+1`). The loop runs `jj=1:N_j-1` populating age `jj+1`, so the new indices are columns `2..N_j` — fits the existing `[N_e,N_j]` shape; column 1 becomes genuinely unused, as the convention states.

(`FieldExp/StationaryDist_FHorz_FieldExp_Treatment.m` appears in the grep but is construction-only — no change.)

### E. SimulateTimeSeries — 16 raws **[DONE 2026-08-10]**

All 16 `SimLifeCycleIndexes_FHorz*_raw.m` (incl. z-only, semiz, and PolicyProbs variants) now wrap the advance block (policy lookups + shock draws) in `if jj<periods`, so the discarded final-iteration draws are gone (they would have read out of bounds on the trimmed pi arrays), and the e draw reads `cumsumpi_e_J(:,jj+initialage+1)` (the e realized in period t+1, Reading A). z/semiz draws keep slice `jj+initialage` (that is the t→t+1 transition). The `SimPanelIndexes_FHorz*` wrappers only build `cumsum` over the pi arrays and are unchanged. **Note: the RNG stream shifts** (fewer rand calls per simulated life-cycle), so fixed-seed simulations will not reproduce pre-change paths bit-for-bit even with age-constant `pi_e`; statistically nothing changes. Three of the 16 files pre-datedly omit the function-closing `end` (legal single-function style) — left as found.

Original scoping notes:

`SimLifeCycleIndexes_FHorz*_e_raw.m` under `SimulateTimeSeries/FHorz/SimPanelIndexes/SimLifeCycleIndexes/`. At period `t = jj+initialage` these draw next period's e via `cumsumpi_e_J(:,t)` → change to `t+1`. **Boundary gotcha (now a prerequisite for category A's trimmed outputs):** the sim loop's final iteration draws states for period `t+1` that are never recorded; with trimmed `pi_z_J` (`N_j-1` slices) the existing `cumsumpi_z_J(:,:,N_j)` read is already out of bounds, and the new e index `t+1=N_j+1` would be too. Restructure the loop to skip the (unused) draws on the final iteration. This applies to the z-only sim files as well, not just the `_e_` ones, **and to the semiz sim files** (`SimLifeCycleIndexes_FHorz*semiz*`): with the trimmed `pi_semiz_J` (`N_j-1` slices in the 4th dimension), a final-iteration draw from `cumsum(pi_semiz_J(...,N_j))` is likewise out of bounds.

### F. ValueFnFromPolicy — 39 files **[DONE 2026-08-10]**

All 39 files transformed (61 sites, scripted with dry-run classification + post-scan; zero flags; the 8-site `ValueFnFromPolicy_FHorz_QuasiHyperbolic.m` hand-inspected — naive `Valt` and sophisticated `Vunderbar` continuation values across noz/z branches all correct; one stale comment fixed). Verified before applying: `ValueFnFromPolicy/` contains no `V_Jplus1` usage and no final-slice `pi_z_J`/`pi_semiz_J` reads, so it is fully compatible with the trimmed arrays with only the e-shift.

Original scoping notes:

All FHorz variants under `ValueFnFromPolicy/` that integrate over e (`... .* shiftdim(vfoptions.pi_e_J(:,jj),-2)` weighting `V(...,jj+1)`): core recipe, `jj` → `jj+1`.

### G. Already compliant — no change (they set the pattern)

- `SubCodes/Ptransitionmatrix/CreatePTransitionMatrix_J.m`
- `EvaluateFnOnAgentDist/FHorz/EvalFnOnAgentDist_AutoCorrTransProbs_FHorz.m`

### H. Discretization routines — verified compliant (no change)

All five life-cycle discretize commands already return `pi_z_J` in the target timing — each explicitly re-indexes so slice `jj` is the transition from period `jj` to `jj+1` (rows on the period-`jj` grid, columns on the period-`jj+1` grid), with `z_grid_J(:,j)` the period-`j` grid and `jequaloneDistz` the period-1 distribution:

- `discretizeLifeCycleAR1_FellaGallipoliPan` (shift at line ~120), `discretizeLifeCycleAR1_FellaGallipoliPanTauchen` (~218), `discretizeLifeCycleAR1_KFTT` (~388), `discretizeLifeCycleAR1wGM_KFTT` (~371): final slice `J` is filled with a **uniform** matrix as padding.
- `discretizeLifeCycleVAR1_Tauchen`: builds slices `1..J-1` directly in target timing; final slice is left as **zeros** padding.

**Caveat:** the padded final slice is meaningless, so output from these routines must not be combined with `vfoptions.V_Jplus1` without overwriting `pi_z_J(:,:,J)` with the true period-`J`-to-`J+1` transition. (Minor optional item: make VAR1_Tauchen's padding uniform for consistency with the others.)

The non-life-cycle discretize routines (`discretizeAR1_*`, `discretizeVAR1_*`) return age-independent `(grid, pi)` and are unaffected.

### Not affected

- InfHorz code (no age dimension).
- All z / semiz consumption sites (already comply); grids everywhere.
- PType/wrapper files that merely pass `pi_e_J` through (except the shape-check text in category A).

## Validation plan (run on the GPU machine)

1. **Age-constant `pi_e` regression:** any existing e-shock test model must produce bit-identical `V`, `Policy`, `StationaryDist` before vs after (columns of `pi_e_J` are identical, so index shifts change nothing).
2. **Age-dependent `pi_e` correctness:** small model (`N_j=3`, `N_e=2`) with distinctly different `pi_e_J` columns; hand-compute the age-1 expectation over the age-2 e (must use column 2) and the age-2 e marginal of the distribution (must equal column 2 exactly).
3. **Internal consistency:** dist evolved by `StationaryDist_FHorz_Iteration_e_raw` must match evolution via `CreatePTransitionMatrix_J` (currently they disagree for age-dependent `pi_e`; after the fix they must agree).
4. **`V_Jplus1` path:** solve an `N_j+1`-period model, then re-solve the first `N_j` periods passing `V_Jplus1` and the `[N_e,N_j+1]` `pi_e_J`; results for periods `1..N_j` must match.
5. **Shape flexibility:** `pi_z` as `[N_z,N_z,N_j-1]` runs without error and matches the `[N_z,N_z,N_j]` run; combining it with `V_Jplus1` errors informatively.
6. TPath: a transition path with (age-constant) e must be unchanged; slowOLG vs non-slowOLG paths must agree.

## Regenerating the file list

```bash
grep -rn "pi_e_J(" --include="*.m" . \
  | grep -v "worktrees" \
  | grep -E "pi_e_J\([^)]*(jj|N_j|,j\)|,j,|,tt|,ii)"
```

Consumption vs construction: assignments (`pi_e_J(...)=`) are construction (leave alone); everything else is consumption (apply recipe). Audit each match rather than blind-sed: the TPath column-shift trick, `cumsumpi_e_J`, and shiftdim'd index positions vary.

## Appendix: full file list (684 files)

Counts by area: ValueFnIter/FHorz 515 · TransitionPaths/FHorz 108 · ValueFnFromPolicy 39 · StationaryDist/FHorz 9 · SimulateTimeSeries/FHorz 8 · SubCodes 4 · EvaluateFnOnAgentDist 1.


<details><summary>ValueFnIter/FHorz (0 files)</summary>

```
```

</details>

<details><summary>TransitionPaths/FHorz (0 files)</summary>

```
```

</details>

<details><summary>ValueFnFromPolicy (0 files)</summary>

```
```

</details>

<details><summary>StationaryDist/FHorz (0 files)</summary>

```
```

</details>

<details><summary>SimulateTimeSeries/FHorz (0 files)</summary>

```
```

</details>

<details><summary>SubCodes (0 files)</summary>

```
```

</details>

<details><summary>EvaluateFnOnAgentDist (0 files)</summary>

```
```

</details>

