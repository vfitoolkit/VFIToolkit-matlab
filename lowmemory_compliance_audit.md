
### GPU run 2026-08-10 — FIRST FULL END-TO-END RUN (no errors). Figs 17–24 green; figs 25–32 have 2 real failures
**No errors — the suite ran to completion** (the semiz-2A cross-tests at the very end are present and green:
`Test 1 (d2-driven semiz)`, `Test 2/3 (collapse)`, `Test 4 (d1 ignored)` all 0.00000000). Capture is truncated
at the START (no `>> CoreFHorzQHTests` header), so figs 1–~16 partly scrolled off.

**Figs 17–24 (plain 2A): still green** — no failing assertion anywhere before the semiz-2A section.

**Figs 25–32 (QH-SemiExo 2A): solvers green, two failure classes, both GI-specific.**
Passing in every one of the 8 subcodes: `Divide-and-conquer` (V/Policy/Valt), `lowmemory=1`,
`lowmemory=1 (with DC)`, non-GI `ValueFnFromPolicy`, `Divide-and-conquer (with Grid Interp Layer)`,
`lowmemory=1 (with GI)`, `lowmemory=1 (with DC+GI)`, and the first 10 of the 12 `beta0=1` checks.

FAILURE A — `QH with beta0=1: should give zero: 6.00000000`, ×2 per subcode (16 total).
These are subcode lines 343/344: `max(abs(Policy3a-Policy3b))` and `max(abs(Policy3a-Policy3c))` — the
**vfoptions3 (GI) Policy** comparisons, exp-vs-QH-Naive and exp-vs-QH-Sophisticated at beta0=1.
Note: the corresponding **V** comparisons (lines 339–342) are **exactly 0**, and the non-GI Policy
comparisons (lines 326/327) are **exactly 0**. So values agree; only the GI Policy encoding differs.
`6 == n2short+1` with `ngridinterp=5`.

FAILURE B — `ValueFnFromPolicy (GI, Naive)` / `(GI, Sophisticated)` ≈ 38–57, ×2 per subcode (16 total).
Only the GI variants; the non-GI `ValueFnFromPolicy` is exactly 0 in the same subcodes.

**Diagnosis so far (not yet root-caused).** A normalized line-by-line diff of the new
`QuasiHyperbolicSemiExoN_GI2A_nod1_noz_raw` **Vtilde pass** against the exponential
`SemiExo_GI2A_nod1_noz_raw` shows them **identical** — the only differing lines are the function signature,
`beta=prod(...)` vs `DiscountFactorParamsVec=prod(...)`, and the continuation `V_next=Valt(:,:,jj+1)` (correct
for Naive). The terminal-period block is character-identical too. So at beta0=1 the QH Vtilde pass should
reproduce the exponential exactly, and the solver code does not obviously explain a 6-unit Policy difference.
Two candidate explanations remain, distinguishable only by inspecting WHICH channel differs:
 (i) an equivalent-but-different (midpoint,L2) encoding of the same fine-grid point — note
     `(a1lower=m, L2raw=13) ≡ (a1lower=m+1, L2raw=7)` since the layer-2 windows overlap, and after `adjust`
     that shows up as a difference of exactly `n2short+1=6` in the L2 channel plus 1 in the a1 channel; or
 (ii) a **d2 tie** broken differently between the two solvers (V identical, d2 channel differs by up to N_d2-1).
Because `ValueFnFromPolicy` decodes Policy, FAILURE B is plausibly downstream of FAILURE A rather than an
independent bug in the new `extract_gi_indices` GI2A fold — but that is NOT yet established.
Related prior art: this family already has a **documented pre-existing Policy-encoding anomaly** (semiz
cross-test 3: V 0.0, Dist 0.0, but Policy 12.0 = 2*(n2short+1)) — same flavour, so a pre-existing semiz
Policy-encoding quirk is a live possibility and an exp-vs-exp control is needed to rule it in or out.

**Decisive next step (needs GPU).** In `QHDFHorz_nod1_noz_noe_semiz_with2A`, after line 344, report the
per-channel discrepancy instead of the aggregate:
```matlab
for cc=1:size(Policy3a,1)
    d=max(abs(reshape(Policy3a(cc,:),[],1)-reshape(Policy3b(cc,:),[],1)));
    fprintf('  channel %d: max|diff| = %g\n',cc,d);
end
```
Channels are `[d2, a1lower, a2prime, L2a1, L2flag]`. If ch4 shows 6 and ch2 shows 1 → explanation (i)
(equivalent encoding). If ch1 shows 6 and the rest 0 → explanation (ii) (d2 tie). That single result
determines whether anything actually needs fixing in the solvers, or only in ValueFnFromPolicy's decode.

### ROOT CAUSE FOUND + FIXED 2026-08-12 — QH-SemiExo GI/DC_GI dispatchers used the 1A UnKron for 2A
The channel diagnostic added to `QHDFHorz_nod1_noz_noe_semiz_with2A` settled it:
```
  channel 1 (d2):      Naive 0   Sophisticated 0
  channel 2 (a1lower): Naive 0   Sophisticated 0
  channel 3 (a2prime): Naive 3   Sophisticated 3
  channel 4 (L2a1):    Naive 6   Sophisticated 6
  channel 5 (L2flag):  Naive 5   Sophisticated 5
  fine a1prime index:  Naive 6   Sophisticated 6      (0 would mean same point, different encoding)
```
This **refuted both** earlier hypotheses: not a d2 tie (ch1=0) and not an equivalent (midpoint,L2) encoding
(fine index != 0). The giveaway is **ch5 = 5**: `L2flag` is built as `2 + (...) - (...)` so it can only be
1/2/3 — a difference of 5 is impossible for a well-formed flag, i.e. the channels were not lining up.

**Root cause (my error, introduced 2026-08-08).** When adding the 2A branches to the QH-SemiExo dispatchers I
checked the UnKron section of the exponential `ValueFnIter_FHorz_SemiExo_DC.m`, found it generic (same call
for 1A and 2A), and wrote in this audit "No UnKron changes needed" — generalising that to the GI and DC_GI
dispatchers **without checking them**. It does not generalise:
- **DC**: Policy's aprime is a single kron'd index, so `UnKronPolicyIndexes2/3_FHorz_z(...,n_a,n_a,...)` is
  correct for both 1A and 2A. (QH DC dispatcher was, and remains, correct.)
- **GI and DC_GI**: the grid-interp layer keeps **a1prime and a2prime as separate Policy channels**, so the
  exponential dispatchers split on `length(n_a)==2` and use `n_a1,n_a2` with UnKron **3 instead of 2** and
  **4 instead of 3** (`ValueFnIter_FHorz_SemiExo_GI.m:113-131`, `..._SemiExo_DC_GI.m:130-147`). My QH GI and
  DC_GI dispatchers always used the 1A form, so the 2A Policy was un-Kron'd with the wrong channel split.

**Fix:** wrapped the UnKron section of `QuasiHyperbolicSemiExo_GI.m` and `QuasiHyperbolicSemiExo_DC_GI.m` in
`if isscalar(n_a) … else (n_a1,n_a2) … end`, with `UnKronPolicyIndexes2→3` and `3→4` and the argument change
`n_a,n_a` → `n_a1,n_a2,n_a`, applied to `Policy` and (for Naive) `Policyalt`. Verified argument-for-argument
against the exponential dispatchers; both files balanced.

**Why the symptoms looked the way they did:**
- `V` comparisons were exactly 0 — V never goes through UnKron, so the solvers were right all along.
- The DC-only column passed — the DC dispatcher's generic UnKron is correct.
- `Divide-and-conquer (with Grid Interp Layer)` (V3 vs V4) passed — GI and DC_GI shared the *same* wrong
  UnKron, so they agreed with each other.
- `ValueFnFromPolicy (GI, …)` ≈ 38–57 — it decodes Policy, so it inherited the mis-split channels. This is
  therefore expected to clear with the dispatcher fix; the `extract_gi_indices` GI2A fold added on 2026-08-10
  is *not* implicated (its `a1lower` fold is applied to a correctly-split channel once UnKron is right).

**Verify next:** re-run. The temporary channel diagnostic is still in place and should now print 0 on every
channel and 0 for the fine index; the 16 `beta0=1 = 6.0` and 16 `ValueFnFromPolicy (GI, …)` failures should
go to 0.00000000. Delete the diagnostic block once confirmed.

## ★★ GPU run 2026-08-12 — ENTIRE QH FHorz TEST SUITE GREEN (figs 1–32) ★★
**2352 exact-zero assertions, ALL `0.00000000`. Zero failed assertions. Zero errors.**
Coverage confirmed complete: 384 `QH with beta0=1` lines / 12 per subcode = **32 subcodes** = figs 1–32.
All 32 `ValueFnFromPolicy (GI, Naive)` and all 32 `(GI, Sophisticated)` are zero. The only non-exact numbers
are the `close to zero` tolerance lines (max 0.05), which are grid-interp / simulation noise as designed.

The channel diagnostic confirmed the UnKron fix outright — every channel and the fine index now 0:
```
  channel 1..5: Naive 0, Sophisticated 0     fine a1prime index: Naive 0, Sophisticated 0
```
The temporary diagnostic block has been **removed** from `QHDFHorz_nod1_noz_noe_semiz_with2A.m`.

### Workstream complete
The QH two-endogenous-asset layer is done and GPU-validated end to end:
- **96 new solver raws**: plain-QH (48) + QH-SemiExo (48), each = {DC2A, GI2A, DC2A_GI2A} × {Naive,
  Sophisticated} × 8 d/z/e (or d1/z/e) variants, every one carrying its exponential sibling's lowmemory ladder.
- **6 dispatcher files** given `length(n_a)==2` branches: plain `QuasiHyperbolic_{DC,GI,DC_GI}` and
  `QuasiHyperbolicSemiExo_{DC,GI,DC_GI}`, incl. the `level1n` scalar-collapse guard (DC/DC_GI) and the
  `isscalar(ngridinterp)` guard (GI).
- **2 ValueFnFromPolicy files** taught GI2A with the QH dual: `ValueFnFromPolicy_FHorz_QuasiHyperbolic.m`
  (8 guarded `alower+n_a1*(a2prime-1)` folds) and `.../SemiExo/..._QuasiHyperbolic_SemiExo.m` (the same fold
  inside `extract_gi_indices`).

### The three real bugs this workstream produced, and how each was caught
1. **Missing `level1n` collapse guard** in the first plain `_DC.m` edit — DC2A raws need a SCALAR `level1n`.
   Caught by static comparison against the exponential dispatcher, before any run.
2. **Stale `ReturnMatrix_ii` reads** in `N_DC2A_GI2A_{,nod_e,e}_raw` terminal blocks (level-2 matrix renamed
   to `_dc` while the following `max(...)`/`(linidx)` probes still read the level-1 matrix; 16 dead
   assignments). Flagged by a sub-agent, verified independently, fixed; an automated scanner was then run over
   all 96 raws every batch thereafter and stayed clean.
3. **1A UnKron used for 2A** in `QuasiHyperbolicSemiExo_{GI,DC_GI}` — the actual cause of the only failures
   that ever reached the GPU. Root: I verified "UnKron is generic" on the exponential *DC* dispatcher and
   generalised it to GI/DC_GI without checking them; GI keeps a1prime/a2prime as separate Policy channels and
   needs `n_a1,n_a2` with `UnKronPolicyIndexes` 3-instead-of-2 and 4-instead-of-3. Caught by a per-channel
   Policy diagnostic, whose `L2flag` difference of 5 (impossible for a 1/2/3 flag) proved the channels were
   misaligned rather than the values being wrong.

**Lesson worth keeping:** per-family verification. The exponential DC / GI / DC_GI dispatchers do NOT share
one UnKron convention, and assuming they did cost a full GPU round-trip. Check each family's exponential
counterpart individually.

## QH ExpAssetze `noa1` tier — toolkit side — BUILT 2026-08-12 (awaiting GPU validation)
Completes the noa1 tier begun on the test side (QH ExpAssetze bank figs 1-4, prepended to mirror the
baseline bank's ordering; existing figs renumbered 1-8 -> 5-12).

**8 new raws**, all in `ValueFnIter/FHorz/ExperienceAssetze/QuasiHyperbolic/`:
`QuasiHyperbolicExpAssetze{N,S}_{,nod1_}noa1_e_raw` (lm {0,1,2}, 265-280 lines) and
`QuasiHyperbolicExpAssetzeSemiExo{N,S}_{,nod1_}noa1_e_raw` (lm {0,1,2,3}, 474-511 lines).
Splice = exponential-noa1 structure + QH dual, in 2 batches of 4 (nosemiz, then semiz).
Signature transform is exactly **drop `n_a1` and `a1_gridvals`**; with no a1 the Policy joint index is over
d only, and a2prime comes from the `aprimeFn` lottery (that machinery preserved character-for-character).

**4 dispatcher edits:**
- `QuasiHyperbolicExpAssetze.m:139/147` — both `error('noa1 variant not yet implemented')` replaced with
  N_d1-forked calls to the 4 nosemiz raws (Naive and Sophisticated).
- `QuasiHyperbolicExpAssetze.m` UnKron — `n_a=[n_a1,n_a2]` -> `n_a=n_a2` and `n_daprime=[n_d,n_a1]` ->
  `n_daprime=n_d` when `N_a1==0`, mirroring exp `ValueFnIter_FHorz_ExpAssetze.m:82-88`.
- `QuasiHyperbolicExpAssetzeSemiExo.m:88` — `error('experienceassetze+semiz requires a standard asset a1')`
  replaced with routing to the 4 semiz raws.
- `QuasiHyperbolicExpAssetzeSemiExo.m` UnKron — **added the `N_a1==0` branch** (`UnKron2/3` instead of
  `UnKron3/4`, dropping `n_a1`) for both `Policy` and `Policyalt`, mirroring exp
  `ValueFnIter_FHorz_ExpAssetzeSemiExo.m:97-107`.

**The UnKron branch was found by applying the 2026-08-12 lesson** ([[exp-dispatcher-unkron-differs-by-family]]):
the exp *nosemiz* dispatcher is generic but the exp *SemiExo* one has a distinct noa1 UnKron branch — checked
per-family this time instead of generalising, so the same bug class that cost a GPU round-trip on
QuasiHyperbolicSemiExo_{GI,DC_GI} was caught before any run.

**Two things checked rather than assumed:**
- The exp noa1 raws assign `V(:,:,:,N_j)=Vtemp;` (no `shiftdim`) in the terminal no-`V_Jplus1` branch but
  `shiftdim(Vtemp,1)` in the `V_Jplus1` branch — internally inconsistent. NOT a bug: the GPU-validated
  `ValueFnIter_FHorz_QuasiHyperbolicN_e_raw.m:41` uses the no-`shiftdim` form, so MATLAB accepts the leading
  singleton. The new raws use the `shiftdim` form (matching their QH siblings); both are equivalent.
- The Sophisticated semiz raws read `Vunderbar` off `V_ford3_under` at the SAME `d3maxindex_lin` that sets
  `Policy`, with the continuation from `Vunderbar(...,jj+1)` — verified directly, not taken on report.

**Static verification:** all 8 present and matching the dispatcher call sites token-for-token (18/20 args
nosemiz, 23/25 semiz); names match filenames; Naive 4-output / Sophisticated 3-output; zero
`n_a1`/`a1_gridvals`/`N_a1` references; zero Naive tokens in the 4 Sophisticated files; `N_bothz` defined
before use in all semiz raws; block openers balance ends; stale-`ReturnMatrix` scan clean across all 8;
lowmemory ladders match the exp noa1 siblings exactly. Both dispatchers' opener/`end` gap is unchanged from
HEAD (the gap of 2 is two single-line `if isNaive, Policyalt=PolicyaltKron; end` statements).

**Verify next:** run `CoreFHorzQHExpAssetzeTests` — figs 1-4 (noa1) should now execute rather than error, and
figs 5-12 become reachable for the first time since the test-side change. Note this bank has never had a
diary, so figs 5-12 are themselves unvalidated; and the ExpAssetze 2A1+semiz work is still awaiting its own
GPU run (now QH figs 11/12).

## QH test banks for ExpAsset / ExpAssetz / ExpAssete — BUILT 2026-08-12 (TEST-FIRST, will error)
Three QH test banks now mirror their baselines one-for-one, in the same figure order. **96 subcodes**
(88 new), written ahead of the toolkit code on purpose — most figures error today.

| bank | before | after | new subcodes |
|---|---|---|---|
| ExpAssetz | 8 of 24 (nosemiz, withA1+with2A1 only) | **24/24** | 16 (noa1 tier ×8, semiz halves of withA1/with2A1 ×8) |
| ExpAssete | none | **24/24** (new bank + script) | 24 |
| ExpAsset  | none | **48/48** (new bank + script) | 48 |

Baseline grids mirrored exactly (the families differ only in which shocks `aprimeFn` forces):
`ExpAsset {d1,nod1}×{z,noz}×{e,noe}×{noa1,withA1,with2A1}×{nosemiz,semiz}`; ExpAssetz forces z; ExpAssete
forces e. Figure order is identical to each bank's own baseline script (verified call-by-call, 24/24, 24/24,
48/48). Naming follows each bank's own convention (they differ) with `QH` inserted after `CoreFHorz`.

**Subcode shape** (from the completed QH ExpAssetze bank): baseline setup verbatim → `exoticpreferences=
'QuasiHyperbolic'` → Naive (4 outputs) → Sophisticated (3 outputs) → exponential cross-tests
((i) Naive continuation==exponential at the real beta0; (ii)/(iii) beta0=1 ⇒ V and Valt ==exponential).
Every method the baseline runs, the full lowmemory ladder on each, and a ValueFnFromPolicy oracle
(Naive passes `Policyalt`, Sophisticated passes `[]`). Moments/figure code omitted throughout, per the
convention in the existing QH banks. No new ReturnFns — all reuse the baselines'.

**lowmemory ladders** were read off each baseline counterpart, never inferred, and match the shock-count
rule: none→{0}; one of {z}/{e}/{semiz}→{0,1}; two→{0,1,2}; all three→{0,1,2,3}. ExpAsset has genuine
**zero-shock** combos (`noz_noe_nosemiz`) that correctly get {0} and no sweep.

**The ExpAsset script was generated from its baseline** by name substitution rather than re-derived, so all
48 argument lists (which vary by tier: `n_a_justexpasset`, `n_a`, `n_a_2A1`…) are exact. Cross-test calls
dropped; addpaths repointed; diary → `../TestOutput/CoreFHorzQHExpAssetTestsdiary.txt`.

**Checked rather than assumed:** the ExpAsset baseline re-runs `CoreFHorzExpAsset_setup` four times between
tiers. That would wipe `Params.beta0`/`QHadditionaldiscount` if setup reset its structs — it does not (it only
assigns fields), so the single QH preamble at the top is safe.

**Deviations from baseline, deliberate:** two baseline files carry an apparent debugging leftover
`ngridinterp=1; % 5` where all siblings use 5 (`Semiz_subcodes/CoreFHorzExpAsset_nod1_z_noe_semiz.m` and
`With2A1_subcodes/Semiz_subcodes/CoreFHorzExpAsset_nod1_z_noe_semiz_with2A1.m`); the QH counterparts use 5.
Worth fixing in the baselines separately. Also, several ExpAsset with2A1 baselines truncate the ladder on
some methods; the QH versions run the full valid ladder uniformly on all methods.

**Verification (static, no MATLAB/Octave):** 96/96 subcodes — function name == filename, correct 16-arg
signature, both QH modes, beta0=1 cross-tests, ladder == baseline's, no moments/figure code, referenced
ReturnFns exist; block balance 0 on every file *and* every script.
NB: balance checking MUST strip quoted strings before `%` comments — MATLAB format strings contain `%s`/
`%2.8f`, and naive comment-stripping truncates the line and reports false imbalances (this produced a false
"8 unclosed blocks" alarm mid-session before being corrected).

**Expected on first run:** ExpAssetz figs 1-8 and 13-16/21-24 error (no QH ExpAssetz noa1 or semiz support);
ExpAssete and ExpAsset error at fig 1 (no QH branch for those families at all — `ValueFnIter_Case1_FHorz`
falls through to the exponential solver and returns 2 outputs).

**Toolkit phase next** (deliberately deferred): QH ExpAssetz semiz + noa1 (~40 raws + a
`QuasiHyperbolicExpAssetzSemiExo` dispatcher); QH ExpAssete from scratch (~32-40 raws, 4 dispatchers, router
branch, VFP file); QH ExpAsset from scratch (~60-80 raws, ditto). Plus the separate silent-QH-ignored guard
for experienceasset/u/e in `ValueFnIter_Case1_FHorz` (user deferred it).

### Convention unified 2026-08-13 — `qhcase` loops unrolled; all QH subcodes now write Naive/Sophisticated in full
The QH banks had **two conventions**: `CoreFHorzQHTests` (43 files, the GPU-validated one) and the TPath bank
write Naive and Sophisticated out as two explicit blocks, while the ExpAsset* family used a
`for qhcase=1:2` loop with `if qhcase==1` branches wherever Naive's 4-output signature differs from
Sophisticated's 3-output. The loop was pre-existing (6 ExpAssetze files) and had been propagated to the
three banks built this session.

Unified toward **written-out-in-full**, matching `CoreFHorzQHTests` and the [[never-use-helper-functions]]
house preference for inlining per branch over shared control flow.

**66 files unrolled** (ExpAssetze 6, ExpAssetz 12, ExpAssete 16, ExpAsset 32) by a source transform that
loop-unrolls `for qhcase=1:2` and constant-propagates `qhcase`:
- single-line `if qhcase==1, STMT; end` → `STMT;` in the Naive copy, dropped in the Sophisticated copy
- block `if qhcase==1 … [else …] end` → the corresponding branch, nesting-aware
- each copy prefixed `%% Naive` / `%% Sophisticated`, body dedented one level
Semantics are preserved exactly (a faithful unroll with constant folding): the Sophisticated pass never had
a `Policyalt` field on its `vfoptionsVFP` in the original either, since that was set inside `if qhcase==1`.

**Verification:** every output validated for block balance and `qhcase`-freedom BEFORE being written.
Across all 5 QH banks — **150 subcodes, 0 remaining `qhcase`, 0 unbalanced, 0 missing a Naive block.**
Spot-checked a 4-method semiz file: 4 Naive 4-output solves, 4 Sophisticated 3-output solves, 4 `Policyalt`
wirings, and **zero `Policyalt` references in the Sophisticated section** (no cross-contamination).
QH test-bank total now 41348 lines.

Note: the first pass wrote only 6 of 66 — the guard counted `%% Naive` as a substring and tripped on the
pre-existing descriptive header `%% Naive then Sophisticated: base / DC1 / GI1 / DC1_GI1`. Guard fixed to
match whole lines; transform itself was never at fault.

## BUG FOUND + FIXED 2026-08-13 — missing `e_c` subscript in 5 exponential DC1_e raws
Found while splicing the QH ExpAssete DC1 batch (the sub-agent declined to copy the line; verified independently).

```matlab
V=zeros(N_a,N_z,N_e,N_j,'gpuArray');            % V is 4-D
...
V(curraindex,z_c,N_j)=shiftdim(Vtempii,1);      % BUG: only 3 subscripts
Policy(curraindex,z_c,e_c,N_j)=...;             % adjacent line correctly has 4
```
With 3 subscripts on a 4-D array MATLAB collapses the trailing dims, so `N_j` indexes the flattened
(e,j) space: the write lands in the wrong (e,j) slot and the intended entry stays zero. Every other `V`
write in the block uses 4 subscripts, including the `else` branch of the very same `if maxgap(ii)>0`.

Reached only on: terminal period + `lowmemory==2` + `maxgap(ii)>0` + no-`V_Jplus1`. That narrowness is why
it survived; it also means the 2026-08-13 green ExpAssetze QH run did not touch it (that bank's DC1 columns
go through the *SemiExo* raws, a different file).

**Same line, copied across 5 families — all fixed:**
```
ExperienceAsset/DivideConquer/ValueFnIter_FHorz_ExpAsset_DC1_e_raw.m:172
ExperienceAssete/DivideConquer/ValueFnIter_FHorz_ExpAssete_DC1_e_raw.m:173
ExperienceAssetu/DivideConquer/ValueFnIter_FHorz_ExpAssetu_DC1_e_raw.m:175
ExperienceAssetz/DivideConquer/ValueFnIter_FHorz_ExpAssetz_DC1_e_raw.m:172
ExperienceAssetze/DivideConquer/ValueFnIter_FHorz_ExpAssetze_DC1_e_raw.m:172
```
`ExpAssetsemiz_DC1_e_raw` is NOT affected (declares `V` with `N_bothz`; all its writes already use 4 subscripts).
Verified after: zero 3-subscript writes remain in any raw declaring a 4-D `V`; block balance unchanged in all 5.

**Note for future runs:** this changes exponential output on that path, so the affected banks' `lowmemory=2`
DC1_e columns will move. The new QH ExpAssete DC1 raws already carry the corrected form, so QH-vs-exponential
parity now holds (before the fix they would have disagreed, with the exponential side wrong).

### QH ExpAssete (nosemiz) build progress — 2026-08-13
4 dispatchers DONE and verified (main + DC + GI + DC_GI, mirroring the exponential layout). UnKron parity
checked per-family against each exponential counterpart — they legitimately differ (main `UnKron1`; DC 1A
`UnKron1`/2A `UnKron3`; GI and DC_GI 1A `UnKron2`/2A `UnKron3`) — with a `Policyalt` twin for every call.
Raws: **24 of 64** done (base 8, noa1 8, DC1 8), each verified for name/signature/arg-list-vs-exponential/
ladder/Sophisticated-purity/stale-ReturnMatrix/balance. Remaining: GI1, DC1_GI1, DC2A, GI2A, DC2A_GI2A (40),
then the router branch in `ValueFnIter_Case1_FHorz` and `ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssete`.

## ★ QH ExpAssete (nosemiz) toolkit implementation — COMPLETE 2026-08-13/14 (awaiting GPU) ★
| piece | count |
|---|---|
| QH raws | **64/64** (8 families × 4 variants × {Naive, Sophisticated}) |
| dispatchers | 4 (main + DC + GI + DC_GI, mirroring the exponential layout) |
| router branch | `ValueFnIter_Case1_FHorz` `experienceassete` + `QuasiHyperbolic` |
| ValueFnFromPolicy | `..._QuasiHyperbolic_ExpAssete.m` (221 L) + `..._GI.m` (301 L) + router route |

Families: base, noa1, DC1, GI1, DC1_GI1, DC2A, GI2A, DC2A_GI2A. Variants `{,nod1} × {,noz}` (ExpAssete forces
e, makes z optional). Built in 8 batches of 8, each verified before moving on.

**Dual templates by family:** base/noa1/DC2A/GI2A/DC2A_GI2A ← the GPU-validated (776/776, 2026-08-13)
ExpAssetze QH raws; DC1/GI1/DC1_GI1 ← the plain-core QH raws (2352 assertions) since no nosemiz ExpAsset*-family
DC1/GI1 QH raw existed. Exponential ExpAssete raws supplied the structure in every case.

**Verified across all 64:** name==filename; Naive 4-output / Sophisticated 3-output; **arg lists byte-identical
to the exponential counterpart** (all 32 pairs); lowmemory ladders equal to the exponential's; Sophisticated
files free of Naive tokens (`Vtilde`/`Valt`/`Policyalt`/`PolicyL2flagalt`/`maxgap_V`); stale-`ReturnMatrix`
scan clean; block balance 0. Every raw the dispatchers reference exists. Dispatcher UnKron parity was checked
per-family against each exponential counterpart (they legitimately differ: main `UnKron1`; DC 1A `UnKron1` /
2A `UnKron3`; GI and DC_GI 1A `UnKron2` / 2A `UnKron3`), each with a `Policyalt` twin.

**Conventions settled in this family:** undiscounted `EV`/`entireEV`/`entireEVinterp` with `beta`/`beta0beta`
applied at use sites (so one `interp1` serves both Naive passes and the Sophisticated gather); Naive uses
`maxgap_V` for the beta pass and plain `maxgap` for the beta0beta pass, sharing the level-1 return matrix;
Sophisticated gathers from the interpolated `EVfine` with the stride written as `size(EVfine,1)` so it cannot
drift from the reshape above it. The a2/a3 lottery is resolved BEFORE the max — `EV` is indexed by the choice,
so re-reading the RHS at the beta0beta-argmax yields `R(policy)+beta*E[V(policy)]` with no separate lottery
handling in the gather.

**Out of scope / known gaps:** semiz (the `Semiz_subcodes` tier of the QH ExpAssete test bank) — needs SemiExo
raws plus an `experienceassete` branch in `ValueFnFromPolicy_FHorz_QuasiHyperbolic_SemiExo.m`, which currently
falls through silently. Also still open: the silent-QH-ignored fallthrough for `experienceasset`/`experienceassetu`
in `ValueFnIter_Case1_FHorz` (user deferred).

**Verify next:** run `CoreFHorzQHExpAsseteTests` — figs 1-4 (noa1), 9-12 (withA1), 17-20 (with2A1) are the
nosemiz ones now implemented; the semiz figs (5-8, 13-16, 21-24) will still error by design.

## QH ExpAssete + SemiExo (semiz tier) — stage 0, batch 1, VFP  [2026-08-14]

Closes the gap that stopped `CoreFHorzQHExpAsseteTests` at fig 5 (`varargout{3}` not assigned:
the router sent experienceassete+semiz to the exponential 2-output `ExpAsseteSemiExo`).

Scope: 64 QH raws (32 exp variants x Naive/Sophisticated) + 4 dispatchers + router + VFP.

Done so far:
- **Router** `ValueFnIter_Case1_FHorz.m`: added `prod(n_semiz)>0 && QuasiHyperbolic` branch to the
  `experienceassete` arm, mirroring the `experienceassetze` arm exactly. Exponential path untouched.
- **4 dispatchers** (`ExperienceAssete/QuasiHyperbolic/...SemiExo{,_DC,_GI,_DC_GI}.m`): spliced from
  exp `ExpAsseteSemiExo` structure + QH ze-SemiExo varargout idiom. Between them they reference
  exactly 64 raw names (16 base + 8 each of DC1/GI1/DC1_GI1/DC2A/GI2A/DC2A_GI2A; 32 N / 32 S).
  `_DC`/`_DC_GI` carry the level1n collapse guard; DC2A/GI2A UnKron copied from this family's own
  exp dispatchers (`UnKronPolicyIndexes4_FHorz_z_e(...,nDPolicyChannel,n_d3,n_a1DC,n_a1fold,...)`),
  NOT generalised from another family.
- **Batch 1 — 8 noa1 base raws** (`{N,S}_{,nod1}_noa1{,_noz}_e_raw`): generated from the QH ze-SemiExo
  raws via the ze->e delta, then the z->noz delta. Verified: numeric core is line-identical to the exp
  `ExpAsseteSemiExo` source except the QH continuation (`EVpre` from `Valt` for Naive, `Vunderbar` for
  Sophisticated); block balance 0; function name == filename; output/arg arity matches every
  dispatcher call site.
- **ValueFnFromPolicy**: `..._QuasiHyperbolic_SemiExo.m` gained the `experienceassete` branch (it was
  falling through silently); new `ExpAssete/ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssete_SemiExo.m`
  adapted from the ze SemiExo VFP. Unlike ze, z is optional -> `N_zeff=max(N_z,1)` and the z-integration
  step is skipped when there is no z.

Key structural facts recorded:
- ze->e differs by aprime tensor ORDER: e computes `[N_d2,N_a2,N_e,N_bothz]` then
  `permute(...,[1,2,4,3])`; post-permute the shape equals ze's `entireEV`, so everything downstream is
  unchanged. This is what makes the splice safe.
- The with-a1 e variant uses different names again (`EV_byzcur`, `aprimeProbs_d2a1a2e`, no
  `bothz_offset`/`_full` arrays) and `CreateExperienceAsseteFnMatrix(...,2)` rather than `...,1`.
  Do not reuse the noa1 patterns for batch 2.

- **Batch 2 - 8 with-a1 base raws** (`{N,S}_{,nod1}{,_noz}_e_raw`): same two-step generation. The
  with-a1 EV block exists in TWO styles in the QH ze source (spaced `lin_lower`/`lin_upper` at
  lowmemory==0; compact direct-index at lowmemory 1/2/3) - both patterns are required, 8 sites per file.
  The ze `entireEVpart=repelem(entireEV,1,N_a1,1,1)` scaffolding line is left in place: e's post-permute
  `entireEV` has the same shape as ze's, so it stays valid.

**Base tier complete: 16/64 raws.** All 16 pass: block balance 0, function name == filename,
output/arg arity matches every dispatcher call site, lowmemory ladder {0,1,2,3} with z and {0,1,2}
for noz. Numeric core is line-identical to the exp `ExpAsseteSemiExo` source except (a) the QH
continuation (`Valt` for Naive, `Vunderbar` for Sophisticated) and (b) the local temp name
`pi_semiz_d3` - the exp sources used to disagree here (11 files carried a `pi_semi_d3` spelling, the
other 62 `pi_semiz_d3`; the split was per-file copy-paste ancestry, not per-tier). Those 11 have since
been renamed, so `pi_semiz_d3` is now the single spelling everywhere, including all 16 here.

Traps hit and recorded:
- Regex patterns must tolerate trailing comments (`[^\n]*\n`, not `\n`). A first with-a1 pass
  silently half-transformed 4 files; the leftover-identifier check caught it and they were deleted.
  Every generator now refuses to write unless the site counts are exact AND no ze-only identifier
  survives.
- Renaming `ExpAssetze`->`ExpAssete` does NOT catch prose `ExperienceAssetze` (different substring).

- **Batch 3 - 8 DC1 raws** (`{N,S}_DC1_{,nod1}{,_noz}_e_raw`). The DC1 aprime setup and EV block are
  IDENTICAL to the with-a1 base tier, so the batch-2 generator applied unchanged except that the
  `bothz_offset` declaration has no preceding comment in the DC1 files (pattern made comment-optional).
  Validated against exp `ExpAsseteSemiExo_DC1_*`: the only extras are the QH continuation and
  `entireEV=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e])` x2 - correct QH scaffolding, since QH must
  keep entireEV UNdiscounted and apply beta / beta0beta separately (exp folds the discount into
  `DiscountedEV=DiscountFactorParamsVec*reshape(...)`).
- ORDERING TRAP for the noz transform: the exp-e DC1 setup block is
  `if lowmemory==2 ... elseif lowmemory==3 ... else ... end`, so the generic
  `elseif lowmemory==3 -> ==2` renumber would corrupt it. The setup replacement must run BEFORE the
  body drop/renumber (it does). The QH-ze DC1 source happens to use the simpler
  `if lowmemory>1 / special_n_bothz` setup, so this did not bite here - but it will if any later batch
  is sourced from an exp-e file.

**24/64 raws built, 24/24 pass** (block balance, name==filename, arity vs dispatcher call sites,
lowmemory ladder). Reusable generators saved: `gen_witha1.py` (ze->e), `z2noz.py` (z->noz),
`check.py` (verifier).

- **Guard fix (the magic-3 bug).** The z->noz guard asserted `drop==3 and renum==3`. That count is
  NOT the invariant: DC1_GI1/DC2A/GI2A/DC2A_GI2A each carry a FOURTH lowmemory ladder in the setup
  region (the `midpoint` / 2A preallocation), so all 16 of their noz raws were falsely held. Replaced
  with `drop==renum and drop>=3` plus a new structural validator `ladder_faults()` that walks every
  `if/elseif vfoptions.lowmemory==K` ladder and rejects duplicate or non-increasing labels, and any
  surviving `==3`. This directly detects the `if ==2 ... elseif ==2` corruption (dead branch ->
  `special_n_semiz` never assigned) regardless of which pattern produced it; previously that was
  caught only coincidentally via `lm_setup==0`. Regression: the 12 noz raws already written are
  12/12 clean, content unchanged.
- **Batch 4 (GI1) + Batch 5 (DC1_GI1) - 16 raws.** NEAR-MISS worth recording: the GI tiers KEEP
  `EV_2D`/`bothz_offset`/`lin_lower`/`EV1` (same names as ze), unlike base+DC1 which use
  `EV_byzcur`/`Vlower`. Running the base-tier generator on them would have rewritten those blocks into
  the EV_byzcur form AND DELETED `bothz_offset`, which the GI tiers still need. It was held only
  because the aprime pattern also failed (`A=0`) and left residue. GI needs its own transform
  (`gen_gi.py`): the e side drops the `*_full` arrays and instead reshapes to a SINGLETON bothz slot
  `[N_d2*N_a1,N_a2,1,N_e]` and broadcasts, so `bothz_offset` is retained and
  `aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,N_bothz,1)`.
  Also: noz renames `bothz_offset`->`semiz_offset` (exp-e noz uses that name).
  Two accepted cosmetic deltas vs exp-e: (a) 4 of 8 blocks inline `EV1=EV_2D(aprimeIndex+bothz_offset)`
  where exp-e always splits out `lin_lower` first - same computation; (b) ze declares `bothz_offset`
  twice (once inside each period section) where exp-e hoists it once - identical constant, kept
  faithful to the GPU-validated ze source.

- **Cleanup applied to the GI tiers (approved).** Two redundancies removed, both verified against the
  exp-e sources and both numerically inert:
  (a) *offset hoist* - the ze DC1_GI1 sources declare `bothz_offset` TWICE, once in the V_Jplus1 branch
  and once INSIDE the backward age loop, so a loop-invariant GPU array was rebuilt N_j-1 times per
  solve. GI1 (same family) already hoists it once. Collapsed to a single top-level declaration before
  `%% j=N_j` in all 8 DC1_GI1 raws. My earlier note called this "harmless/faithful" - that was wrong
  about the in-loop copy.
  (b) *duplicate index precomputes* - the `bothzind->semizind` / `bothzBind->semizBind` renames in the
  noz transform collided with existing `semizind`/`semizBind` lines, leaving identical duplicates in 8
  files (4 DC1_GI1 noz, 4 GI1 noz). exp-e noz carries exactly one of each. Removed.
  BOTH fixes were folded back into the generators (`gen_gi.py` hoist, `z2noz.py` dedupe), and
  regeneration now reproduces the on-disk files byte-for-byte (8/8 and 8/8) - so the manual edits and
  the generators agree and future batches will not reintroduce either.
  NOT touched: the same duplicate-offset pattern exists in 8 ze QH files (DC1_GI1 + DC2A_GI2A). Those
  are GPU-validated and belong to another family; left alone deliberately.

**40/64 raws built, 40/40 pass; 0 duplicate top-level assignments across all 40.** Of the 16 previously
blocked noz raws, DC1_GI1's 4 are now built.

- **DC2A tier - 8 raws** (`gen_2a.py`, a THIRD transform). 2A naming: `n_a3` is the experience asset,
  a1 DC'd, a2 folded; `a3primeIndex` + `a1_col`/`a2_col` linear-index construction. ze uses
  `EV_2D`+`bothz_offset`+`*_full`; exp-e uses `EV_byzcur`+direct indexing+`permute`. As with every
  tier so far, post-permute `entireEV` matches ze's shape, so the `DiscountedEV_alt/_tilde` lines
  following it stay valid untouched.
  KNOWN TRAP CHECKED: the DC2A `maxgap` else-branch reshapes DO carry `*N_a2`
  (`vfoptions.level1n*N_a2*N_a3`), matching exp-e exactly - the bug fixed earlier in the exp files is
  not present here.
  `a3_gridvals` (defined once via CreateGridvals) is a ze-family convention where exp-e passes
  `a3_grid` directly; my files define it, so they are self-consistent. Not a defect.
- **z2noz extensions needed for DC2A**: spaced-comma signatures (`n_a3, n_z, n_semiz, n_e, N_j`),
  `N_semiz*N_z`->`N_semiz`, and `ones(1,length(n_semiz)+length(n_z))`->`ones(1,length(n_semiz))`.
- **Guard relaxed again**: dropped the `lm_setup==1` requirement. DC2A carries its setup as a 4-branch
  lowmemory ladder that the generic drop+renumber converts correctly, so no dedicated setup pattern
  fires and `lm_setup` is legitimately 0. `residual()` + `ladder_faults()` are strictly stronger -
  an unconverted setup would still mention `n_z`/`N_bothz`/`special_n_bothz`.
- **`residual()` now scans CODE ONLY** (comments stripped). A stale word in a comment was falsely
  gating correctness.
- **ORDERING BUG I introduced and the guard caught**: the new `length(n_semiz)+length(n_z)` collapse
  was placed BEFORE the noa1 setup pattern, which matches on that original text - so the pattern
  stopped firing and 4 noa1 files failed to regenerate (`drop=3` vs `renum=4`, plus leftover
  `special_n_semiz=[n_semiz,ones(1,length(n_z))]`). Moved the collapse AFTER the setup patterns.
  All 24 noz raws now regenerate cleanly, so generator/disk reproducibility is restored.

**48/64 raws built, 48/48 pass.** Remaining `bothz` strings in noz raws are cosmetic only, of two kinds:
(a) comments, and (b) compound local variable names like `d2a1primea2bothze`, which are defined and used
within the same file (the name says bothz but it indexes semiz). Neither is a live reference.
CAVEAT on the checker: `residual()` tests `\bbothz`, which does NOT match `pi_bothz` (underscore is a
word char), so it could in principle miss a genuinely undefined z-era variable. Backstopped with an
explicit scan for undefined z-era names (`pi_bothz`, `bothz_gridvals_J`, `*_full`, `special_n_bothz`,
`bothzind/Bind`, `aprimeProbs_d2a1a2ze`) across all 48 raws: **0 undefined-variable problems**.

### The last two tiers split into two DIFFERENT methods (measured, not assumed)

The e family supports only a 1-dim experience asset: exp-e GI2A and DC2A_GI2A have ZERO
`length(n_a3)` branches, while the ze QH sources carry an `l_a3==2` bilinear branch
(GI2A: 70 lines / 6%; DC2A_GI2A: 170 lines / 11%) that has no e-side counterpart.

Vocabulary overlap decides which route is available:

| identifier        | DC2A_GI2A ze / exp-e | GI2A ze / exp-e |
|-------------------|----------------------|-----------------|
| `aprimeIndex_full`| 10 / 10              | 0 / 0           |
| `aprimeProbs_full`| 10 / 10              | 0 / 0           |
| `EV1=`            | 16 / 16              | 0 / 16          |
| `EV_2D`           | 56 / 24              | 0 / 24          |
| `Vlower=`         | 0 / 0                | 4 / 0           |
| `EV_aprime`       | -                    | 14 / 0          |

- **DC2A_GI2A: ze->e IS feasible.** Names line up exactly; the work is (a) collapse the
  `if length(n_a3)==1 ... else ... end` blocks to the if-body, (b) the usual aprime/z conversion,
  (c) e builds `_full` as `repmat(reshape(...,[...,1,N_e]),1,1,N_bothz,1)` where ze uses
  `repelem(...,1,1,N_semiz,1)`. exp-e KEEPS `bothz_offset` here (18 uses), so it must NOT be deleted.
- **GI2A: ze->e is NOT feasible.** The vocabularies are disjoint - ze indexes `EVpre` directly into a
  5-D `[fold,a3,bothz,e,bothz]` result (`Vlower`/`EV_aprime`), exp-e uses the leaner singleton-bothz +
  `bothz_offset` broadcast (`EV_2D`/`EV1`). A ze->e transform would produce a DIFFERENT implementation
  with no reference to validate against. GI2A must instead be built as **exp-e GI2A + the QH dual**,
  using the dual pattern from the already-validated QH-e GI1 pair (measured expansion ratio 1.31x:
  740 -> 971 lines). That dual is a structural rewrite, not a regex transform: every RHS is duplicated
  (alt=beta, tilde=beta0beta), `DiscountedEV`->`DiscountedEV_alt/_tilde`, per-d3 arrays doubled, and
  the argmax/gather logic duplicated.

- **DC2A_GI2A - 8 raws DONE** (`gen_dc2agi2a.py`). The ze source has exactly TWO distinct
  `if length(n_a3)==1` bodies repeated across 10 blocks: the aprime setup (2 blocks, indent 4) and the
  EV block (8 blocks, indent 12). The EV body already matches exp-e verbatim
  (`EV1=EV_2D(aprimeIndex_full+bothz_offset)`), so only the setup needed converting; `bothz_offset` is
  RETAINED here (exp-e uses it 18x). Collapse keeps the if-body, de-indents by 4, drops the 170-line
  `l_a3==2` else-branches. Validation vs exp-e: ZERO core differences; `EVpre` correctly uses `Valt`
  (Naive) / `Vunderbar` (Sophisticated).
- Guard bug of my own: the DC2A_GI2A BAD-list wrongly included `N_z`/`z_gridvals_J`, which are
  LEGITIMATE in the with-z variants (exp-e uses `N_z` 6x). Corrected - those are only forbidden in noz,
  where `z2noz.residual()` already checks them.
- Two checkers were reading COMMENTS as code and false-flagging: `residual()` (fixed earlier) and
  `check.py`'s lowmemory-ladder extraction, which read a header doc line
  (`% ... lowmemory=3 loop bothz + e`) as a live ladder level and failed 2 noz files that were
  actually correct. Both now strip comments first; the stale noz header doc is also rewritten.

**56/64 raws built, 56/56 pass.** Of the 16 originally-blocked noz raws, 12 are done (DC1_GI1, DC2A,
DC2A_GI2A); the last 4 are GI2A noz.

- **GI2A - 8 raws DONE, and my earlier "cannot go ze->e" call was WRONG.** The vocabularies are indeed
  disjoint, but that was a surface reading. Structurally ze GI2A and exp-e GI2A meet at an IDENTICAL
  seam: both hand an 8-D `[N_d2,N_a1,N_a2,1,1,N_a3,N_bothz,N_e]` discounted EV to `interp1`. They differ
  only in ORDERING - ze interpolates the a3 lottery once per period then contracts with pi_bothz inside
  the d3 loop; exp-e contracts then interpolates. Both are linear, so the orderings are equivalent.
  Because the e aprime lottery is z-independent, ze's current-bothz slot collapses to a SINGLETON and
  broadcasts against pi_bothz. That makes GI2A a small targeted edit (`gen_gi2a.py`) that PRESERVES the
  GPU-validated QH dual scaffolding (`EVbase` -> `DiscountedEV_alt/_tilde`), instead of a ~1.31x
  hand-written rewrite. Trade-off accepted knowingly: the result does NOT diff line-for-line against
  exp-e (different loop order), so that particular cross-check is unavailable for this tier; the seam
  shape, the singleton slot, and the whole-file invariants below are checked instead.
- **REAL BUG caught by the backstop, exactly as predicted.** The 4 GI2A noz raws used `pi_bothz` 6x
  while defining only `pi_semiz_d3` - an undefined variable that would have failed at runtime. Cause:
  `z2noz` rewrote the definition and the `pi_bothz'` transpose use, but GI2A uses
  `reshape(pi_bothz,[...])`, which the transpose-only pattern missed. `residual()` could not see it
  because `\bbothz` does not match after an underscore - the exact limitation flagged one step earlier.
  Fixed by a blanket `pi_bothz`->`pi_semiz_d3` rename plus a new SUBSTRING residual list
  (`RESID_SUB=['pi_bothz','bothz_gridvals_J','bothz_offset']`; `_full` deliberately excluded because it
  is legitimate in DC2A_GI2A).

## ALL 64 RAWS BUILT - final whole-set verification

| check | result |
|---|---|
| dispatcher coverage | 64 referenced / 64 built, 64/64 PASS |
| block balance, name==filename, output/arg arity vs call sites | 64/64 |
| lowmemory ladders | {0,1,2,3} with z, {0,1,2} noz |
| undefined z-era / l_a3 / tier-local variables (25 names, code-only scan) | 0 problems |
| QH continuation: Naive driver == Valt | 0 problems (32/32) |
| QH continuation: Sophisticated driver == 3rd output, is the GATHERED beta-RHS (not re-maximised) | 0 problems (32/32) |

NOTE on Sophisticated naming (inherited from the ze sources, NOT introduced here): the base/DC/2A tiers
call the recursion driver `Vunderbar`, the GI tiers call it `Valt` and document `Valt (=Vunderbar)`.
Semantics are identical in both - V/Policy come from the beta0*beta argmax and the driver is the
beta-RHS gathered at that same argmax. The dispatchers take 3 outputs positionally, so the internal
name does not matter. A first pass of my checker wrongly flagged the 8 GI-tier S files for this.
| output signature (N=4, S=3) | 0 problems |
| noz code-level z residue | 0 |

### ValueFnFromPolicy GI subfn - DONE (the last gap)

`ValueFnFromPolicy/ExpAssete/ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssete_SemiExo_GI.m` written by
mirroring the ze SemiExo GI VFP. A structural diff against that source shows ONLY the intended e-deltas:
`N_e`-only error check, `N_zeff=max(N_z,1)`, aprimeFnParamNames without `l_z`, conditional
`n_shocks`/`joint_gridvals_J`/z-integration/final reshape, and `N_z`->`N_zeff` throughout. The 2x2
corner interpolation (a1 low/up x a2 low/up) and the QH dual are carried over verbatim.
The non-GI subfn's `gridinterplayer==1` error stub was replaced with the real call.

Checks: block balance 0; function name == filename; `CreateaprimePolicyExperienceAssete` call arity
14 == definition arity 14 (both call sites); no mutual recursion (the GI subfn sets
`vfoptions.gridinterplayer=0` BEFORE falling back to the non-GI subfn for the noa1 case); the whole
VFP chain (`_ExpAssete`, `_ExpAssete_GI`, `_ExpAssete_SemiExo`, `_ExpAssete_SemiExo_GI`) resolves with
no dangling references.

Also corrected a now-false message: the NO-semiz `_QuasiHyperbolic_ExpAssete` carried
`error('...experienceassete+SemiExo not yet implemented')`. It is unreachable (ValueFnFromPolicy_FHorz
dispatches on n_semiz>0 first) and the guard itself is right, but the text was stale. Reworded to say
where semiz actually belongs.

### GPU run 2026-08-18 15:36 (CoreFHorzQHExpAsseteTests, BEFORE the GI subfn landed)

12 subcodes / figs 1-12 completed, 500 assertions, 498 exactly zero. Stopped at fig 13 on the then-still
-stubbed GI VFP, exactly as predicted. **Figs 5-8 passed** - that is GPU confirmation of the 8 noa1
SemiExo raws, the router branch, and the new non-GI SemiExo VFP.
Two values at 1e-8, both `Sophisticated ValueFnFromPolicy (..., Valt)`. Isolated to the one quantity
computed by different arithmetic routes (solver GATHERS a precomputed F+beta*EV at the beta0beta-argmax;
the VFP recomputes F and adds beta*EV_at_policy). Naive Valt is exact 13/13 because both routes evaluate
identically at Policyalt; Sophisticated V is exact 12/12. 1e-8 is one unit in the last place of %2.8f and
appears in 2 of 12 figs, consistent with rounding rather than a systematic error. To settle it, print
that one comparison at %2.14f.

Still to do: the they use the 2A naming (`n_a3` is the experience
asset, `a3primeIndex`, `a1_col`/`a2_col` linear-index construction) and branch on `length(n_a3)==1` vs 2.
The exp-e `DC2A_*` sources exist to copy the e-side blocks from. And the
`_ExpAssete_SemiExo_GI` VFP (currently an explicit not-implemented error, needed for figs 13-16/21-24).

## QH ExpAssetz — stages 0 and 1  [2026-08-18]

**Corrected scope** (an earlier count was wrong): the two families are SYMMETRIC — 32 variants in every
quadrant (ExpAssete nosemiz/semiz, ExpAssetz nosemiz/semiz). ExpAssete doubles on the optional-z axis
with e required; ExpAssetz doubles on the optional-e axis with z required. Both total 128 QH raws.
ExpAssetz has 32 built, so **96 remain** (not 94 — two earlier miscounts, both from a script that
mangled the bare no-suffix variant when stripping `_raw`).

### Stage 0 — DONE
- **Router** `ValueFnIter_Case1_FHorz.m`: the `experienceassetz` arm had the SAME ordering bug that
  broke ExpAssete (`if prod(n_semiz)>0` matched before the QuasiHyperbolic test, sending semiz+QH to
  the exponential 2-output call -> `varargout{3}` not assigned). Hoisted a
  `prod(n_semiz)>0 && QuasiHyperbolic` branch ahead of it.
- **`ValueFnIter_FHorz_QuasiHyperbolicExpAssetzSemiExo.m`** written (semiz main dispatcher). Mirrors the
  exp `ExpAssetzSemiExo` dispatcher plus the QH varargout idiom. Because e is OPTIONAL here, the UnKron
  tail needs the `_z` vs `_z_e` split that ExpAssete did not. References 16/16 base-tier raw names
  (14 via the two-underscore pattern + the 2 bare `SemiExoN_raw`/`SemiExoS_raw`). Balance 0.

### Stage 1 — IN PROGRESS, nothing written (gates held all 4 outputs)
Source choice, decided by measurement not assumption: **QH-Z with-a1 + the exp a1->noa1 delta**, all
in-family, validated against `exp ExpAssetz *_noa1_*`. Rejected the ze route because QH-ZE carries 2
`length(n_a2)` bilinear branches that the z family does not have (0) — the same trap as GI2A.
**Stage 1 COMPLETE: 8 noa1 raws + the 4 dispatcher guards replaced.**
`gen_z_noa1.py`, 12 gated rules, all 8 written clean. The two gaps that initially held it:
1. **ReturnMatrix builder swap** `CreateReturnFnMatrix_ExpAsset_Disc_e` -> `_Case2_Disc_e` (drops the
   a1prime args, drops the trailing `,0,0`). 9 call sites per file (paren-aware count), but my regex
   matched only 3: the arg lists contain nested parens like `z_gridvals_J(:,:,N_j)`, so `[^)]*`-style
   patterns truncate. Needs a paren-aware matcher.
2. **d1-variant differences**: signature carries `n_d1`/`d_gridvals`, and the EV expansion differs.
   Confirmed against exp: noa1 nod1 -> `entireEV=EV`; noa1 d1 -> `entireEV=repelem(EV,N_d1,1)`
   (expand over d1, not over a1).

Two further things the gates caught mid-build, both mine:
- The **Sophisticated gather stride**: `maxindexfull` uses `N_d2*N_a1` (or `N_d*N_a1`) as the RHS row
  stride, and it appears in EVERY term of the index expression, not just the leading one. My first rule
  anchored on `maxindexfull=maxindex+` and missed 6 of 12 sites per file; the `N_a1` residue check
  held the files. Made global. (Naive raws have no gather, so it fires 0 times there.)
- An **off-by-one** when I generalised the ReturnMatrix rewriter to the no-e builder: `a1_gridvals` sits
  at index 9/10 with e and 8/9 without, not 8/9 and 7/8. The positional assertions caught it on all 8.

Two accepted equivalences vs the exp reference, both verified against the helper source rather than
assumed:
- **`aprimeFnParamsVec` flag 2 vs exp's 1.** Per `CreateExperienceAssetzFnMatrix` lines 170-178:
  1 -> column, 3 -> `[N_d*N_a2,N_z]`, else (2) -> `[N_d,N_a2,N_z]`; `a2primeProbs` is reshaped to
  `[N_d,N_a2,N_z]` UNCONDITIONALLY. My raws keep flag 2 and index with `a2primeIndex(:)`, whose linear
  order equals the flag-1 column (both reshapes of the same underlying vector). Equivalent.
- **`EVpre` intermediate** where exp inlines `V(a2primeIndex,:,jj+1)`. Required for QH anyway, since the
  continuation is `Valt` (Naive) / `Vunderbar` (Sophisticated), not `V`.

Verification: 8/8 balance 0, name==filename, no `N_a1`/`n_a1`/`a1_gridvals`/`aprimeIndex`/`aprimeProbs`/
`ExpAsset_Disc` residue; lowmemory ladders {0,1,2} with e and {0,1} without (z only); the nosemiz
dispatcher now references 40 raws and all 40 exist.

**Figs 1-4 of CoreFHorzQHExpAssetzTests should now run.** Remaining: 88 raws (24 nosemiz DC1/GI1/DC1_GI1,
64 semiz), 6 dispatchers, and the `_SemiExo_GI` VFP.

### Stage 1 follow-up: the UnKron also had to become noa1-aware

First GPU run after stage 1 failed at fig 1 with `Error using reshape ... V1=reshape(V1Kron,[n_a,n_z,N_j])`.
Cause: replacing the four `noa1 not yet implemented` guards wired the correct RAW calls, but the
dispatcher's downstream UnKron still assumed a1 exists. Two things were wrong for noa1:
- `n_a=[n_a1,n_a2];` at the two BASELINE UnKron sites -> must be `n_a=n_a2` when `N_a1==0`.
  (The two DC2A UnKron sites are left alone: divide-and-conquer requires a standard endogenous asset.)
- `n_daprime=[n_d,n_a1];` -> `n_daprime=n_d` when `N_a1==0`, since Policy carries no a1prime channel.
  This mirrors `ValueFnIter_FHorz_ExpAssetz` lines 98-104, which does `if n_a1>0, n_d=[n_d,n_a1]; ... end`.

Fixed at 2 + 2 sites. `N_a1` is assigned at line 26, well before all four. Dispatcher balance 0;
40 raws referenced and all 40 present; 0 guards remaining.

LESSON (generalises to stages 2-6): replacing a dimension guard is not just about the raw call — the
UnKron/reshape tail encodes the same dimensional assumption and has to be updated in the same pass.

### Stage 2 (QH ExpAssetz semiz base noa1, 8 raws) — DONE

Unlike stages so far, no near-isomorphic QH source exists. Three candidate routes, measured:

| route | differing code lines | verdict |
|---|---|---|
| ze-semiz -> z-semiz | — | REJECT: the ze and z families organise their lowmemory internals differently |
| z-nosemiz -> z-semiz | 335 | REJECT: adds the entire d3/semiz layer |
| **e-semiz -> z-semiz** | **81** | **CHOSEN** |

ROUTE CORRECTED after closer inspection (the e-semiz -> z-semiz swap above is REJECTED too):
the e-family and z-family differ not just in the EV *idiom* but in the lowmemory 2/3 *structure* —
e computes a full `entireEV` then slices `(:,:,:,e_c)` / `(:,:,semizblock,:)`, whereas z computes
`entireEV_z` per z-block (lm2) and per-bothz `EV_z` (lm3). Restructuring that by hand is where errors
would hide, and it is the one part with no reference to check against.

**CHOSEN ROUTE (B): exp ExpAssetzSemiExo -> QH, i.e. apply the QH dual to the authoritative z source.**
Rationale: the EV machinery (the hard, family-specific part) stays untouched from the exp source, and
the QH dual is a UNIFORM, local duplication. Template for the dual: the e-family pair
`exp ExpAsseteSemiExo_nod1_noa1_e` -> `QH ExpAsseteSemiExoN_nod1_noa1_e` (383 -> 462 lines, ratio 1.21),
which I built and which is GPU-validated at figs 5-8.

Exact block texts (both extracted):
- e-idiom: `Vlower=reshape(EV(a2primeIndex,:),[N_d2,N_a2,N_e,N_bothz])` ... `entireEV=permute(...,[1,2,4,3])`
- z-idiom: `EV_2D=reshape(EV,[N_a,N_bothz])`; `EV1=EV_2D(aprimeIndex_full+bothz_offset)` ...
  `entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3)`
- aprime setup z-side adds `aprimeIndex_full=repelem(aprimeIndex,1,1,N_semiz)` etc. and needs a
  top-level `bothz_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]);`

KEY SHAPE DIFFERENCE: with z-dependent aprime the lottery does NOT depend on e, so the z-idiom
`entireEV` is 3-D `[N_d2,N_a2,N_bothz]` and broadcasts against the ReturnMatrix's e axis, whereas the
e-idiom `entireEV` is 4-D `[N_d2,N_a2,N_bothz,N_e]`. Any downstream `entireEV(:,:,:,e_c)` slice in the
e-source must therefore be dropped, not carried over.

COMPLETE SPEC for route B (all units located in `exp ExpAssetzSemiExo_nod1_noa1_e_raw`):
1. signature `[V,Policy2]` -> `[Vtilde,Policy,Valt,Policyalt]` (N) / `[Vhat,Policy,Vunderbar]` (S)
2. declarations: `V`->`Valt` + new `Vtilde`; `Policy2`->`Policyalt` + new `Policy`;
   `V_ford3_jj`->`V_ford3_alt` + new `V_ford3_tilde`; same for `Policy_ford3_jj`
3. `DiscountFactorParamsVec=prod(...)` -> `beta=prod(...); beta0=CreateVectorFromParams(...,QHadditionaldiscount,IDX); beta0beta=beta0*beta;`
4. `EVpre=...V(:,:,:,jj+1)...` -> `Valt` (Naive) / `Vunderbar` (Sophisticated)
5. terminal (no V_Jplus1, no EV): writes become `Valt`/`Policyalt`, then append
   `Vtilde(:,:,:,N_j)=Valt(:,:,:,N_j); Policy(:,:,:,:,N_j)=Policyalt(:,:,:,:,N_j);`
6. **8 RHS+max groups** at lines 133,162,195,228,291,320,353,386 — each 4 lines
   (`entireRHS_X=RM+DiscountFactorParamsVec*EV;` / `[Vtemp,maxindex]=max(...)` / `V_ford3_jj(idx)=` /
   `Policy_ford3_jj(idx)=`) duplicated into an alt pass (`beta`, `_alt`) and a tilde pass
   (`beta0beta`, `_tilde`)
7. **2 d3-resolution blocks** at lines 239, 396 — duplicated for alt and tilde
For Sophisticated the RHS+max groups become max-at-beta0beta plus a GATHER of the beta-RHS at that
argmax (not a second max), per the established S convention.

**Stage 2 COMPLETE (8/8).** Route B executed via `gen_zsemiz.py` (`naive()` + `soph()`).
Key design point that paid off: because the QH dual is applied TO the exponential z source, the
family-specific EV machinery is carried over untouched — verified as **0 differences vs the exp source**
on all 8 files (comparing `CreateExperienceAssetzFnMatrix`/`aprimeIndex_full`/`EV_2D`/`EV1`/`EV2`/
`skipinterp`/`entireEV`/`EV_z`/`pi_bothz`/`aprimeProbs_z`), once the legitimately-duplicated RHS lines
are excluded.

Implementation notes:
- The regex matchers had to be replaced with LINE-BASED block consumers. The four variants differ in
  ways regexes kept missing: `Policy2` vs `Policy3` (2 vs 3 channels), `max(...,[],3)` vs `[],4`,
  with/without `N_e` dims, and the d1 EV term being `repelem(entireEV,N_d1,1,1)` rather than a bare name.
- Sophisticated gather stride is DERIVED from each block's `V_ford3` write index (one term per
  non-scalar dim after the a-dim), giving the 4 canonical forms x {N_d2 (nod1), N_d12 (d1)} = 8.
  `Nrow=N_d2` / `N_d12` matches the e-family template.
- BUG CAUGHT BY VERIFICATION: the Naive continuation rename initially only handled the 4-index
  `V(:,:,:,jj+1)` form, so the no-e raws kept `V(:,:,jj+1)` — an UNDEFINED variable (V had been renamed
  to Valt). Fixed to cover both index forms. The Sophisticated path already had both.

Verification: 8/8 balance 0, name==filename, output/arg arity matches every dispatcher call site
(N=4 outputs, S=3); continuation driver Valt (Naive) / Vunderbar (Sophisticated) in all 8;
0 undefined-variable problems. Semiz dispatcher: 16 base raws referenced, 8 present (the other 8 are
the with-a1 base variants = stage 4).

**Figs 5-8 of CoreFHorzQHExpAssetzTests should now run.**

### Stage 3 (QH ExpAssetz) — (a) SPLIT DONE, (b) raws still to do

Two independent pieces, both larger than stages 1-2:

**(a) The dispatcher split** (agreed to happen at the start of stage 3). Exact boundaries in
`ValueFnIter_FHorz_QuasiHyperbolicExpAssetz.m` (362 lines):

| region | lines |
|---|---|
| e-block 2A prologue (n_a1DC/n_a1fold/level1n) | 49-59 |
| e-block method branches: DC / GI / DC_GI | 60-73 / 74-87 / 88-102 |
| e-block shared UnKron (+return) | 103-127 |
| e-block 1A guard (error) | 131-133 |
| no-e 2A prologue | 200-210 |
| no-e method branches: DC / GI / DC_GI | 211-224 / 225-238 / 239-252 |
| no-e shared UnKron (+return) | 254-284 |
| no-e 1A guard (error) | 289-291 |

Target shape (mirrors exp `ValueFnIter_FHorz_ExpAssetz_DC.m`): each sub-dispatcher takes
`(n_d1,n_d2,n_a1,n_a2,n_z,N_j,d_gridvals,d2_gridvals,a1_gridvals,a2_grid,z_gridvals_J,pi_z_J,ReturnFn,
aprimeFn,Parameters,DiscountFactorParamNames,ReturnFnParamNames,aprimeFnParamNames,vfoptions)`,
does its own level1n setup, branches `length(n_a1)>1` (2A, moved verbatim) vs scalar (1A, new), and
UnKrons its own results. Main dispatcher keeps baseline only + early-return dispatch.
IMPORTANT: keep the 1A guards erroring verbatim until (b) lands, so the split is purely
behaviour-preserving — figs 9-12 currently PASS on these 2A paths.

**(b) The 24 nosemiz DC1/GI1/DC1_GI1 raws.** NOT like the base tier: the QH dual for DC/GI is a
WHOLE-BLOCK duplication (the level1 max, maxgap, and level2 max all re-run for beta and beta0beta),
not the local 4-line unit that `gen_zsemiz.py` handles. Measured on this family's own 2A pair:
exp `ExpAssetz_DC2A_nod1_e_raw` 631 lines -> QH 921 lines, ratio 1.46.
Source route (same logic as route B): exp `ExpAssetz_DC1*`/`GI1*`/`DC1_GI1*` + the QH dual, with the
existing QH `*_DC2A_*` raws as the in-family template for what that dual looks like in DC/GI code.

**Stage 3(a) COMPLETE — dispatcher split.** The 362-line monolith is now 207 lines (baseline only)
plus three 120-line sub-dispatchers under `DivideConquer/`, `GridInterpLayer/`,
`DivideConquerGridInterpLayer/`, matching the exponential `ValueFnIter_FHorz_ExpAssetz_*` layout.
Method blocks were moved VERBATIM (extract + re-indent, no rewriting), so this is behaviour-preserving:
figs 9-12 pass on those 2A paths and must continue to.
The 1A guards are retained inside each sub-dispatcher (`if length(n_a1)<=1 -> error`), with the original
error text, so behaviour for scalar n_a1 + DC/GI is unchanged until stage 3(b) lands.
Verified: main references 16 base raws and ZERO 2A raws; each sub-dispatcher references exactly its own
8 (2A x {e,no-e} x {nod1,d1} x {N,S}); union = 40, all present on disk; all four balance 0; names match
filenames; call-site arity matches every sub-dispatcher declaration; every argument passed at the
dispatch site is defined by that point.

### SCOPE CORRECTION found while starting stage 3(b)

Checked which subcodes actually set `divideandconquer=1` / `gridinterplayer=1`:

| figure group | DC/GI tested? |
|---|---|
| noa1 (figs 1-8, nosemiz+semiz) | NO |
| **nosemiz withA1 (figs 9-12)** | **NO** |
| semiz withA1 (figs 13-16) | YES |
| nosemiz with2A1 (figs 17-20) | YES (2A raws, already exist) |
| semiz with2A1 (figs 21-24) | YES |

So the 24 **nosemiz** DC1/GI1/DC1_GI1 raws I had scoped for stage 3(b) are NEVER CALLED by this bank —
figs 9-12 pass on base methods alone, which is why they passed with no DC1/GI1 raws present.
The 1A DC/GI methods are exercised only by the SEMIZ withA1 figures. "The 24" is therefore naturally the
**semiz** DC1/GI1/DC1_GI1 set: 3 tiers x {,nod1} x {,_e} x {N,S} = 24.

### Fig 13 (`nod1_z_noe_semiz_withA1`) needs: 8 raws + 3 semiz sub-dispatchers
- base semiz with-a1 nod1 no-e: **DONE** (`SemiExoN/S_nod1_raw`), via `gen_zsemiz`. One fix needed: the
  with-a1 sources put a COMMENT LINE between the `V=zeros` and `Policy3=zeros` declarations, so the
  adjacent-lines decl pattern had to tolerate `(?:%[^\n]*\n)?`.
- `SemiExoN/S_{DC1,GI1,DC1_GI1}_nod1_raw`: 6 raws, still to do.
- the 3 semiz sub-dispatchers (`..._SemiExo_DC/_GI/_DC_GI`), referenced by the semiz main dispatcher
  since stage 0 but never created.

### The DC/GI QH dual, characterised from the in-family template
`gen_zsemiz` handles d3/beta/decl/pre/sig on the DC sources but `rhs=0`: the DC RHS is not a local unit.
Pattern (from `QuasiHyperbolicExpAssetzN_DC2A_nod1_raw`, lines ~171-218) is TWO PARALLEL DC PASSES:
1. `EVbase` once, then `DiscountedEV_alt=beta*EVbase` and `DiscountedEV=beta0beta*EVbase`;
2. level-1 run twice -> `maxindex1_alt/maxindex2_alt` and `maxindex1/maxindex2`; both Valt/Policyalt and
   Vtilde/Policy written;
3. `maxgap_alt` AND `maxgap` computed separately;
4. the level-2 `for ii` loop carries separate narrow bands per pass (own `loweredge`,
   `a1primeindexes`, `ReturnMatrix_ii_alt`), with `curraindex` shared.
So the transform is: duplicate everything from `DiscountedEV` through the level-2 loop, `_alt`-suffixing
one copy. For Sophisticated, the under-value must instead be GATHERED at the hat argmax at every write
site inside the DC structure.

### Fig-13 progress: 3 semiz sub-dispatchers DONE, base pair DONE, 6 DC/GI raws remain

Created `ValueFnIter_FHorz_QuasiHyperbolicExpAssetzSemiExo_{DC,GI,DC_GI}.m` (101 lines each, balance 0),
mirroring `ValueFnIter_FHorz_ExpAssetzSemiExo_{DC,GI,DC_GI}`. Each references exactly its 8 raws
(1A x {,nod1} x {,_e} x {N,S}) -> 24 total, which IS "the 24" after the scope correction. The 2A branch
errors explicitly (semiz 2A raws are stage 6). The semiz main dispatcher has called these since stage 0.

### DESIGN FINDING for the 6 DC/GI raws: level-1 shares, level-2 cannot

Structure of each EV-bearing DC block in `exp ExpAssetzSemiExo_DC1_nod1_raw` (6 such blocks:
3 lowmemory levels x 2 period sections), from `DiscountedEV=DiscountFactorParamsVec*...` to the `end`
of the `for ii=1:(level1n-1)` loop:

- **Level 1 CAN share** `ReturnMatrix_ii`: both passes form their RHS from the same matrix
  (`entireRHS_alt=ReturnMatrix_ii+DiscountedEV_alt`, `entireRHS=ReturnMatrix_ii+DiscountedEV`).
- **Level 2 CANNOT share anything**: the narrow band comes from `loweredge` <- `maxindex1`, which
  DIFFERS between the beta and beta0beta argmaxes. So `maxgap`, `loweredge`, `a1primeindexes` and the
  level-2 `ReturnMatrix` are all per-pass. This is exactly what the in-family template
  `QuasiHyperbolicExpAssetzN_DC2A_nod1_raw` does (`maxgap_alt` AND `maxgap`; `ReturnMatrix_ii_alt`).

Consequence: a naive "duplicate the whole block sequentially" transform is CORRECT but recomputes the
level-1 ReturnFn matrix, doubling that work in the hot loop. The interleaved form (share level-1, suffix
everything derived from the argmax with `_alt`) matches the family and avoids it. Recommend interleaved.

For Sophisticated the same structure applies, but the under-value is GATHERED at the hat argmax at every
write site inside the DC structure (level-1 and both level-2 branches).

## QH ExpAssetz semiz — Fig 13 raw set (DC1 / GI1 / DC1_GI1, nod1, no-e)

Six raws written, completing everything Fig 13 (`nod1_z_noe_semiz_withA1`) needs.

| tier | N | S |
|---|---|---|
| DC1_nod1 | 967 | 744 |
| GI1_nod1 | 804 | 624 |
| DC1_GI1_nod1 | 1125 | 820 |

**The QH dual per tier.** Naive always needs two *independent* passes; Sophisticated always needs
one beta0beta pass plus a **gather** of the beta-RHS at that argmax (never a second max).

- **DC1** — Level 1 CAN share `ReturnMatrix_ii` (both passes build their RHS from the same matrix).
  Level 2 CANNOT share anything: the narrow band comes from `loweredge` <- `maxindex1`, which differs
  between the beta and beta0beta argmaxes, so `maxgap`, `loweredge`, `a1primeindexes` and the level-2
  `ReturnMatrix` are all per-pass. Emitted interleaved (level-1 shared, then two narrow-band loops).
- **GI1** — coarse argmax -> `midpoint` -> fine interpolated argmax. Only the COARSE `ReturnMatrix_d3`
  is shared; the fine one follows `midpoint` and is per-pass. Needs `DiscountedEVinterp` per pass too.
- **DC1_GI1** — the DC stage only builds `midpoint` (no V/Policy writes inside `for ii`), then a single
  fine GI stage writes V/Policy3/flag. `ReturnMatrix_ii_d3` is *reassigned* three times under one name
  (coarse / narrow / fine), so sharing the coarse one is not cleanly separable — full duplication is
  used instead, which is the faithful and lowest-risk choice.

**Three traps found while generating these (all fixed):**
1. `DiscountedEV*` names vary by lowmemory (`DiscountedEV` vs `DiscountedEV_z`), as do the RHS names
   (`entireRHS_ii_d3` / `entireRHS_ii_z`) and index names (`d2aprimez` / `d2aprime`). Anchors must be
   name-agnostic — hardcoding the lowmemory-0 names silently skips the lowmemory-2 blocks.
2. At GI lowmemory 2, `DiscountedEV`/`DiscountedEVinterp` are computed OUTSIDE `for z_c` and sliced
   inside. Duplicating the block wholesale duplicates the loop opener without its `end`. The dual must
   split into head (pre-loop) and tail (per-state) parts.
3. **`midpoint` is preallocated** (`midpoint=zeros(...)`, one per lowmemory branch) in DC1_GI1. Renaming
   the declarations to `midpoint_alt`/`midpoint_tilde` without renaming the terminal region's uses left
   20 bare `midpoint` uses indexed-assigning an undefined variable.

**Gate.** `gate.py` checks block balance, un-dualled leftovers, define-before-use for `_alt/_tilde/_hat/_under`
names, and — added after trap 3 — *indexed assignment that precedes any preallocation of that name*.
That last check is the only one that catches trap 3; verified by deleting a preallocation line from a
good file and confirming FAIL. All six pass; the gate also passes the GPU-validated base and DC2A raws.

Still to write for the other 18 of "the 24": the with-d1 and with-e variants of these three tiers.

### Fig 13 run 2: the with-a1 under-gather stride bug

323 assertions, 13 nonzero — all confined to the Sophisticated **base** raw. Every one of the 27
DC1 / GI1 / DC1_GI1 Sophisticated checks passed, as did all Naive checks.

Diagnosis: `ValueFnFromPolicy (DC1)` was 0 while the base `ValueFnFromPolicy` was Inf, i.e. base's V
disagreed with base's *own* policy. So base was wrong and `DC1 vs base` failed only because base is
the reference. The Inf/Policy-mismatch combination is the signature of a corrupted `Vunderbar`: it
drives the recursion, so a bad gather poisons next period's EV, which then moves both Vhat and the
argmax, and differently in each lowmemory branch.

Root cause: in `..._QuasiHyperbolicExpAssetzSemiExoS_nod1_raw`, the under-gather strode by `N_d2`,
but dim 1 of the in-loop RHS is **`N_d2*N_a1`** whenever there is a standard endogenous state — the
Policy decode proves it (`rem(d2a1prime_ind-1,N_d2)+1` for d2, `ceil(d2a1prime_ind/N_d2)` for a1prime).
Six sites fixed. The bug came from `gen_zsemiz._stride`, whose `Nrow` was `N_d12`-or-`N_d2`: right for
the noa1 raws (dim 1 is the d block alone), wrong for every with-a1 raw. Generator patched; it now
reproduces the hand-fix byte-for-byte.

This was latent from the base pair written earlier in the session — it only surfaced once the
ValueFnFromPolicy GI blocker was removed and the run could reach the Sophisticated section.

**The gate does not catch this class.** It is a semantic stride error, not a structural one: the code
is well-formed, every name is defined, and the arithmetic is only wrong against the array's true dim-1
size. A repo-wide scan for the bare `maxindexfull=maxindex+N_dX*(0:` signature found no other affected
file — the four apparent hits in the GPU-validated ExpAssete DC1 raws are the `maxgap==0` narrow-band
branch, where the band width is 1 and dim 1 genuinely is `N_d`.

### Fig 13 GREEN; the no-e semiz set completed (16 raws)

Fig 13: **323 assertions, 0 nonzero.** The stride fix cleared all 13 failures.

All 32 semiz raws (4 tiers x nod1/with-d1 x no-e/with-e x N/S) now generate. Landed the **16 no-e**
ones — 8 complete N/S pairs across base, DC1, GI1, DC1_GI1. Crucially, the 8 fig-13 raws regenerate
**byte-identical** to the GPU-validated files after all the generator changes below, which is the
regression check that makes the other 8 trustworthy enough to run.

Generator fixes needed to reach the with-d1 variants:
- **`DISC` must tolerate a trailing comment.** The with-d1 `DiscountedEV=...; % (d2,a1prime,...)`
  line failed the `;\s*$` anchor, so an entire EV block was silently skipped — it surfaced only as
  one leftover `V_ford3_jj` in the gate. Same class of bug as the earlier expression-extraction fix;
  MATLAB comments end statements, and every anchor in these generators has to allow for them.
- **Index-variable names carry the d-block width**: `d12a1primea2bothz` with d1 vs `d2a1primea2bothz`
  without. The narrow pattern silently left them shared across passes — wrong for Naive, and invisible
  to the gate because the code stays well-formed. Patterns are now `d\d*a1primea2\w*` / `d\d*aprimez?\w*`.
- **Policy accumulator has 4 channels with d1** (`Policy4_ford3_jj`) vs 3 without.
- **Policy L2 tail channel** is `Policy(5,...)` with d1 vs `Policy(4,...)`; the d3 collapse reduces
  dim 4 with e vs dim 3 without; the level-1 reshape is 4-dim with e. All three now derived, not assumed.
- **Base with-d1 raws are `..._SemiExo_raw`** with no middle segment, so the `_(\w+)_raw` signature
  pattern never fired and the outputs stayed exponential.

**Outstanding: the 14 with-e raws**, all still failing the gate, in four groups:
1. base `_e` / `_nod1_e` (N+S): the RHS block anchor does not match the with-e block shape
   (`V_ford3_jj`/`Policy_ford3_jj` left x6 each).
2. `N_DC1_e`, `N_DC1_nod1_e`: a `DiscountedEV` reference survives unsuffixed (the
   `repelem(DiscountedEV,N_d1,1,1,1,1)` broadcast form).
3. GI1/DC1_GI1 with-e, Naive: `block balance 2` — the head/tail split mis-handles the with-e loop nesting.
4. GI1/DC1_GI1 with-e, Soph: `bare V( left`, and `DiscountedEVinterp_z_under` never assigned in `S_GI1_e`.
(`S_DC1_e` and `S_DC1_nod1_e` pass but were held back — their N partners fail, and landing half a pair
is not useful.)

### The d1 in-place broadcast (fig 14)

Run reached 347 assertions / 0 nonzero (fig 14 base + DC1 green), then died with
`Unrecognized function or variable 'DiscountedEV'` in the with-d1 GI1 Naive raw.

Cause: the with-d1 GI sources expand the discounted EV in place for the d1 block --
`DiscountedEV=repelem(DiscountedEV,N_d1,1);` (and the same for `DiscountedEVinterp`) -- on lines
sitting between the discount line and the tail start. The head/tail split covered neither, so the
lines survived unsuffixed while their only definitions became `_alt`/`_tilde`.

Fix: the head now runs from the discount line to the tail start **or to the first control-flow line,
whichever comes first**. That boundary matters: at lowmemory 2 a `for z_c` opens between the discount
line and the per-state slices, so extending the head blindly to the tail start duplicates a loop
opener without its `end` (seen as `block balance 2`). Putting the broadcast in the *tail* instead has
the same defect from the other side.

Gate gained a matching check: `x=f(x)` as the FIRST assignment of `x`, with no earlier definition, is
use-before-assignment. The earlier "used but never assigned" test could not see this, because the
offending line *is* an assignment to that very name. Verified it fires on the landed buggy file.

All 16 no-e raws now pass the gate. The 4 fig-13 GI/DC1_GI1 raws changed only by 6 blank lines each
(**0 code-line differences**), so the GPU-validated behaviour is preserved.

### The with-e semiz raws (16) — completed

Worked in three independent stages, each a distinct root cause. All 32 semiz raws now generate and
pass the gate; the 16 no-e ones remain **code-identical** to the GPU-validated files.

**Stage 1 — GI/DC1_GI1 with-e (8 raws).** Three separate defects:
- With e, `for e_c` opens *inside* the tail range but its `end` falls after the flag write, so
  duplicating the tail unbalanced the file. The tail range is now extended forward until it is
  self-balancing (`_balance`). Note the head has the opposite rule — it *stops* at the first control
  flow. Head and tail are not symmetric and cannot share a boundary rule.
- The continuation read is `V(:,:,:,jj+1)` with e and `V(:,:,jj+1)` without; the GI scaffold only
  rewrote the 3-index form. `gen_zsemiz` already carried a comment warning about exactly this in the
  mirror direction — the warning was there, the second scaffold just didn't inherit it.
- The under-twin emitter required the RHS to begin with `DiscountedEV...`, but with d1 it is wrapped:
  `repelem(DiscountedEVinterp_hat(...),N_d1,1)`. Now any `DiscountedEV*_hat=` assignment in the tail
  gets a twin, whatever the RHS shape.

**Stage 2 — DC1 with-e (4 raws).** At lowmemory 2/3 the discounted EV is sliced per state inside the
loop (`DiscountedEV_z=DiscountedEV(:,:,:,:,z_c)`). That line sits in the *shared* prefix, emitted once,
so both passes read one slice. Naive now expands it into an alt/tilde pair in place; Sophisticated
emits an under twin.

The Sophisticated case was a **latent wrong-number bug the gate passed**: `DiscountedEV_z_under` was
used in one lowmemory branch while its only assignment lived in a different, mutually exclusive branch.
"Used but never assigned" is not flow-sensitive, so it saw a definition and was satisfied. Added a
pairing invariant instead: every `_hat` discounted-EV name must have an `_under` partner (and `_alt`
an `_tilde`), which is branch-independent and catches this directly.

**Stage 3 — base with-e (4 raws).** With e, higher lowmemory pre-computes
`DiscountedEV=DiscountFactorParamsVec*...` and then writes `entireRHS_d3e=ReturnMatrix_d3e+DiscountedEV`,
so the inline `+DiscountFactorParamsVec*` anchor never fired and six write sites per file were left
untransformed. Both generators now recognise a standalone discount line (split into `EVbase_qh` plus
the two pass copies) and accept either RHS form; for the pre-computed form the EV term is *already*
discounted, so the hat/under RHS uses `DiscountedEV_hat`/`_under` directly rather than multiplying
by beta again.

All 40 on-disk semiz raws pass the gate and every dispatcher reference resolves. Figs 17+ are unrun.

### BUG A 2026-08-19 — Naive GI1 with-e, lowmemory 3: the shared coarse ReturnMatrix was lifted out of the e loop

Figs 15/16 reported the Naive GI1 with-e raws disagreeing with themselves: lowmemory 1 and 2 exactly
zero, lowmemory 3 off by 6.03 (V), 8.39 (Valt), 35/34 differing Policy entries. Sophisticated passed
at every level; DC1_GI1 passed for both duals. Files:
`...SemiExoN_GI1_e_raw.m` and `...SemiExoN_GI1_nod1_e_raw.m`.

**Cause.** `naive_gi` emits the coarse `ReturnMatrix_d3ze` once (it is pass-independent) ahead of the
two passes, by pulling that one line out of the tail and putting it at the top of the new tail. That
is safe only when the tail *starts inside* the e loop, which is the shape at lowmemory 2: the tail
anchor `TAILSTART` matches the `ReturnMatrix_d3\w*=` line itself, so the `for e_c` opener and `e_val=`
stay in the shared prefix and the line never moves.

At lowmemory 3 the per-state EV slices `DiscountedEV_z=DiscountedEV(:,:,:,:,z_c)` sit **above** the
`for e_c`, so `TAILSTART` fires on `DiscountedEV\w*_z=` instead and the tail swallows the loop opener.
Hoisting the ReturnMatrix line to the top of that tail moved it **out of the e loop**, where `e_val`
still holds the last e of the previous z iteration. Both passes then took their coarse argmax — hence
the midpoint, hence the fine window — from a stale-e return matrix, while the fine `ReturnMatrix_ii`
inside the loop used the correct `e_val`. A max over the wrong sub-window: small V drift plus a few
dozen Policy entries. This is why only lowmemory 3 broke and why the Sophisticated twin was fine —
`soph_gi` never reorders the tail.

**Fix.** `naive_gi` now checks whether a control-flow opener lies between the tail start and the
ReturnMatrix line. If not, behaviour is unchanged (lowmemory 0/1/2, and every no-e raw). If one does,
the tail is split three ways instead: the pre-loop per-state slices are duplicated per pass, the loop
opener + `e_val` + the coarse ReturnMatrix are emitted **once inside** the loop, both passes run in
that one loop body, and the loop's closers are re-appended at the end. The result is exactly the
lowmemory-2 shape and matches the Sophisticated twin line-for-line outside the dual.

**Regression check.** With the generator fixed, all 32 QH ExpAssetz semiz raws regenerate and only
these 2 files change; the other 30 — including all 16 GPU-validated no-e raws — are **byte-identical**
to their on-disk versions. A toolkit-wide scan for `e_val` read with no enclosing `for e_c` returned
exactly 4 hits before the fix (the 2 files x their 2 lowmemory-3 blocks) and 0 after.

**Gate gained the matching invariant (3e):** a loop-scoped value (`e_val`, `z_val`, `e_c`) read where
its loop is not open. Nothing already in the gate could see this — the file stays balanced, every name
is assigned, and no dual leftover survives; the code is simply *legal and wrong*. Verified it fires on
a reconstruction of the pre-fix output, at exactly the two offending lines, and on nothing else.

**Note, not a bug:** `d12a1primea2` is still unsuffixed in the with-d1 GI raws (`gen_zgi.PASS` carries
the narrow `d2a1primea2\w*` pattern, which cannot match it). It is assigned immediately before its one
use in each pass, so the passes cannot share it. Widening the pattern to `d\d*a1primea2\w*` as
`gen_zsemiz` does would rewrite the 16 GPU-validated no-e raws for no behavioural gain, so it was left
alone deliberately.

Awaiting the GPU run of CoreFHorzQHExpAssetzTests, figs 15 and 16.

**Figure numbers corrected.** This section originally said figs 17/19. Wrong: figs 17-20 are with2A1
*nosemiz*, which has neither semiz nor a lowmemory 3, and they run DC2A/GI2A rather than GI1. The only
two subcodes that print `lowmemory=3 (GI1)` are `CoreFHorzQHExpAssetz_nod1_z_e_semiz_withA1`
(figure_c=15) and `CoreFHorzQHExpAssetz_d1_z_e_semiz_withA1` (figure_c=16) — withA1 + semiz + e, the
only shape with all four rungs of the z+e+semiz ladder.

**Re-verified independently 2026-08-19 (post-commit c716314f), no code change needed.** Four checks:
(1) the invariant-3e scan over all of `ValueFnIter/FHorz/ExperienceAssetz` returns 0 hits; (2) the gate
passes on all 32 semiz raws; (3) `build24.py` regenerates all 32 **byte-identical** to the on-disk
committed files, so the generators and the toolkit are in step and the 16 no-e raws are provably
untouched; (4) both lowmemory-3 blocks (the `V_Jplus1` branch and the jj loop) of each named file were
diffed line-for-line against the GPU-validated exponential source
`ValueFnIter_FHorz_ExpAssetzSemiExo_GI1[_nod1]_e_raw.m` with the `_alt`/`_tilde` suffixes stripped —
both passes reproduce the source exactly, and the only lines outside the pass bodies are the correct
dual expansion (`EVbase_qh` split into `beta`/`beta0beta` copies, per-pass per-state slices, and the
two d3 collapses). Check (4) matters because a suffix-stripping diff cannot see a *missing* suffix; a
separate read confirms every pass-local name in those blocks is suffixed, the sole exception being
`d12a1primea2`, which is assigned immediately before its one use in each pass (see the note above).

One generator drift was found and fixed while doing (3): `gen_zgiscaf` hardcoded the no-e widths in the
terminal-period copy (`Vtilde(:,:,N_j)=Valt(:,:,N_j)` etc.), so it under-copied by one dimension on the
8 with-e GI-family raws — it would have copied only the first e slice. The on-disk files already carry
the correct 4-index form, so this was generator-only drift, not a toolkit bug. `_slices()` now reads the
colon counts off the exponential source's own `V=`/`Policy=` declarations, the way `gen_zdcscaf` and
`gen_zsemiz` already did. After the patch all 32 regenerate byte-identical.
