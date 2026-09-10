# Proposal: replace the EV broadcast with a matrix product (EVmatmul)

STATUS 2026-09-04: decision settled on **SCOPED, NO GUARD** — the matmul is the only path
in the converted files, and the scope is the plain-z and SemiExo tiers, leaving the other
~6,000 sites entirely untouched (still on the broadcast).

A size guard (threshold 1e7 elements, on the measured crossover) was built first and then
dropped. It costs nothing at runtime — 0.25ns per branch test, 0.02us per core solve, 0.4us
for a SemiExo solve at N_d2=20 — but it would have added ~1,000 lines of duplicated
two-path code for wave 1A alone, and, decisively, the matmul path would never have been
executed by any test bank: every bank grid is ~1e4 elements against a 1e7 threshold, so the
new code's first real execution would have been Guvenen at n_z=3965 in production.
Unguarded, every bank run and every model exercises it. Price accepted: ~1% runtime on small
models (a fixed ~3ms per solve) plus ULP-level V differences and rare Policy tie-flips in
the banks. §4's option analysis is superseded by this; §5's wave order still stands.

**Wave 1A APPLIED 2026-09-04** (unrun, uncommitted): 48 files / 128 sites —
FHorz/{root,DivideConquer,GridInterpLayer,DivideConquerGridInterpLayer} and
SemiExo/{root,DC,GI,DCGI}. All 128 use the same V1 recipe. Awaiting the GPU bank run before
wave 2A.

Measurements that settled it (RTX 4000 Ada, EVtest.m Test 4 + EVruntime.m):
- Crossover at ~2e6 elements of broadcast transient, drifting with N_a — 2.94e6 / 2.12e6 /
  1.69e6 at N_a=101/201/501 (spread 1.74x). In N_z terms it moved 170/103/58, so the guard
  keys on `N_a*N_z^2`, never on N_z alone.
- Largest bank grid that runs a value function iteration is n_a=451, n_z=5 = 1.1e4 elements,
  ~150x below the crossover and ~900x below the guard. No bank changes behaviour.
- Below the crossover the matmul costs a fixed ~30us per EV call (~3ms per 80-age solve,
  0.3–1.3% of total VFI runtime). Above it: 4.8x at N_z=525, 14.3x at 1000, 27.9x at 1500,
  and the broadcast OOMs from ~1500–2000 (the exact point depends on free GPU memory).
- The exact--Inf-restoration pipeline is GPU-validated (EVtest 6/6): identical -Inf pattern,
  entries exactly -Inf, no NaN, non-vacuous populations, including the gpuArray logical matmul.
- Policy tie-flips confirmed rare and immaterial (EVpolicydiag.m): exactly one entry in
  341,901 and one in 2,069,631, each one fine-grid step, V identical to 15 digits.

2026-09-03. Motivated by the Guvenen (2007) replication: the belief chain there needs
N_z=3965, at which the toolkit's EV "broadcast" computation is a 70.9GB transient — it OOMs
a 20GB GPU from about N_z=1500 (with N_a=501) and hits MATLAB's gpuArray pmaxsize limit at
N_z~4000. Stage (c) of Guvenen2007 (`baseline/RunBaseline.m`) is blocked on
`ValueFnIter_FHorz_DC1_nod_raw.m`.

Toolkit work all on master, no branches. Nothing here is committed until GPU-green.

## 1. The two computations

Current (the "broadcast", canonical 3-line form; `EV` is `[N_a,N_z]`, a copy of next-period V):

```matlab
EV=EV.*shiftdim(pi_z_J(:,:,jj)',-1);
EV(isnan(EV))=0; % multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=sum(EV,2); % sum over z', leaving a singular second dimension
```

This materializes an `[N_a,N_z,N_z]` double plus an equally-sized logical mask (~9
bytes/element). The isnan line implements one semantic rule: a zero-probability transition
into a `-Inf` continuation contributes exactly 0.

Proposed (the "matmul", cleaned pipeline with exact -Inf restoration — reference
implementation and tests in `~/Dropbox/Matlab_Codes/Replications/Guvenen2007/EVtest.m`):

```matlab
EVinf=(EV==-Inf);
EV(EVinf)=-1e250;                    % stop -Inf*0 -> NaN inside the product
EV=EV*pi_z_J(:,:,jj)';               % sum over z'
EV(EVinf*(pi_z_J(:,:,jj)'>0)>0)=-Inf; % exact -Inf restoration
EV=reshape(EV,[N_a,1,N_z]);
```

The fourth line is an indicator matmul: `(EV==-Inf)*(pi'>0)` counts positive-probability
transitions into `-Inf` continuations. Counts of 0/1 doubles are exact integers, so `>0` is
not a magnitude threshold. Verified this session on CPU MATLAB: `logical*logical` mtimes is
legal and returns double (gpuArray case is covered by the pending EVtest GPU rerun; if it
ever errors on GPU the fix is `double(EVinf)*double(pi'>0)`).

### Semantics: no behavioural difference

- **-Inf pattern is IDENTICAL to the broadcast.** Wherever any positive-probability
  transition leads to a `-Inf` continuation, the output is exactly `-Inf` (restoration);
  wherever all `-Inf` continuations have probability exactly 0, the clamped `-1e250` enters
  the product with weight exactly 0 and contributes exactly 0, so finite entries carry no
  clamp artifact. Identical argmax behaviour, including at fully-infeasible states.
- **Finite entries differ only by floating-point summation order** (broadcast: MATLAB `sum`
  along dim 2; matmul: BLAS dot products). Measured: max rel diff 3.331e-16 on GPU
  (old-pipeline EVtest run), 2.2e-16 on CPU (cleaned pipeline). This is the ONLY difference,
  and it is what drives the design decision in §4.
- **No NaN either way** on the populations that occur in practice (finite, `-Inf`,
  `-Inf`×zero-prob all exercised in EVtest Test 2 with all three populations asserted
  non-empty).
- Overflow safety: `-1e250` entries enter only with weight 0; the restored `-Inf` never
  feeds arithmetic that the broadcast's `-Inf` didn't. realmax is 1.8e308, so even the
  pre-restoration intermediate `sum_z' p*(-1e250) >= -1e250` cannot overflow.

Two edge populations where behaviour differs, both only reachable on already-broken inputs:
genuine NaN in V (a user ReturnFn bug) is silently zeroed by the broadcast's isnan line but
propagates under the matmul (arguably the safer failure mode); `+Inf` in V (does not occur in
the covered families — see §3 safety column) would give `+Inf*0=NaN` inside the product.
Sites where NaN *legitimately* pre-exists get a scrub line — recipe N in §2.

### Evidence (EVtest.m + EVtest_diary.txt, RTX 4000 Ada, N_a=501)

| N_z | broadcast | matmul | broadcast transient | matmul transient |
|---|---|---|---|---|
| 525 | 0.0081s | 0.0009s | 1.24 GB | 0.004 GB |
| 1000 | 0.0823s | 0.0030s | 4.51 GB | 0.008 GB |
| 1500 | OOM | 0.0063s | 10.15 GB | 0.012 GB |
| 2000 | OOM | 0.0107s | 18.04 GB | 0.016 GB |
| 3965 | OOM (pmaxsize) | 0.0416s | 70.89 GB | 0.032 GB |

Correctness: max rel diff 3.331e-16 at N_z=525/1000, all-finite and -Inf-laden inputs, no
NaN either way. Timing table is from the pre-restoration pipeline run; the cleaned pipeline
adds one indicator matmul (~2x the EV cost, still ~5-15x faster than the broadcast and
MB-scale). **A GPU rerun of the updated EVtest.m (strict -Inf equality + timings including
restoration) is pending and should precede Wave 0.**

## 2. Rewrite recipes (per shape variant)

Every in-scope site is the canonical 3-line form (verified mechanically by the survey:
~6,600 sites, 100% have the isnan line immediately after the `.*`, and the sum dimension is
always the pi-prime dimension). Recipes, to be instantiated per-file with that file's own
variable names and target reshape (house style: inline, replicated per file, no helpers):

**V1 — 2-D source, shift −1, sum dim 2** (the bulk: core, SemiExo `pi_bothz`/`pi_semiz`,
QH, GP, AA, RiskyAsset z-sites, ExpAsset semiz/ze contraction-first sites). Source
`[M,N]`, output `[M,1,N]` (or `[M,N]` where the file follows the sum with that reshape):

```matlab
EVinf=(EV==-Inf);
EV(EVinf)=-1e250;
EV=EV*pi';
EV(EVinf*(pi'>0)>0)=-Inf;
EV=reshape(EV,[M,1,N]);      % only if the file's broadcast left [M,1,N]
```

**V2 — 3-D source, shift −2, sum dim 3** (ExpAsset/ExpAssetu plain + `EV_aprime` SemiExo
DC/GI sites, DC2, ResidualAsset). Source `[M1,M2,N]`: fold to 2-D, apply V1, unfold:

```matlab
EV=reshape(EV,[M1*M2,N]);
% ... V1 body ...
EV=reshape(EV,[M1,M2,N]);    % matching the file's squeeze(sum(...,3)) result
```

**V3 — 4-D source, shift −3, sum dim 4** (ExpAssete plain): fold `[M1*M2*M3,N]`, V1,
unfold, keep the file's trailing `permute` unchanged.

**N — NaN-scrub prefix**, prepended to V1/V2/V3 wherever the source array is a
probability-weighted interpolation blend (`aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper`
with `skipinterp` zeroing, or the ResidualAsset `rprimeProbs` blend): there,
`0*(-Inf)=NaN` legitimately pre-exists in the source and the broadcast's isnan line was
doing double duty. Prepend:

```matlab
EV(isnan(EV))=0; % interpolation 0*(-Inf) NaNs contribute zero, matching the broadcast's isnan-fix
```

This reproduces the broadcast exactly (a pre-existing NaN row contributed 0 to every z
under the broadcast). Decision rule for implementers: scrub iff the source is an
interpolation blend; never when the source is V/V_Jplus1/an e-integrated V directly.

**Guard** (under the recommended option A, §4): each site becomes

```matlab
if M*N*N>1e8 % broadcast transient too large; use matmul path (same answer, see EVmatmul_proposal.md)
    % recipe
else
    % existing 3 lines, byte-identical
end
```

where `M*N*N` is that site's broadcast transient element count (`N_a*N_z^2`,
`N_a*N_bothz^2`, `N_d2a1a2*N_a3*N_bothz^2`, ...). 1e8 elements = 0.9GB transient; every
existing test-bank grid is orders of magnitude below (bank N_z ~ 10-30), Guvenen is 7.9e12.

## 3. Inventory

Method: grep `\.\*shiftdim\(pi_` across ValueFnIter/ plus the spellings that grep misses —
`ambiguity_pi_z_J` (AA), `reshape(pi_z_J...)`/`reshape(pi_bothz...)` broadcasts (ze
family), hoisted `pi_z_alt`/`pi_z_alt2`/`Epi_z` (InfHorz), and non-shiftdim row-slice
spellings (`ones(N_a,1)*pi(z_c,:)`, `EV.*pi(z_c,:)`). Five parallel survey agents read
representative files per (subdir x variant); per-file shape verification remains an
implementation-time step per the standing rule (never generalise across families blindly).

~1,637 files carry at least one in-scope site. Sites, by family:

| Family (ValueFnIter/FHorz/...) | in-scope sites | recipe | clamp-safe? | notes |
|---|---|---|---|---|
| core (root, DC, DC2, DC2A, GI, GI2A, DCGI, DCGI-DC2A) | 58 | V1 (DC2: V2) | YES — plain V, no guards/transforms | 2 sites/file: V_Jplus1 branch + jj loop |
| SemiExo (7 subdirs) | 168 | V1 | YES | `pi_bothz=kron(pi_z,pi_semiz)` rebuilt inside the d2 loop; transient is N_bothz², the worst in the core trees |
| ExperienceAsset plain (+u; roots/DC/GI/DCGI) | 128 | V2+N | YES | interp-before-expectation: NaN scrub needed |
| ExperienceAssete plain | 32 | V3+N | YES | trailing permute stays |
| ExpAsset*SemiExo trees (incl. u, e, ze, semiz) | 1,516 | V1 (contraction-first) / V2+N (`EV_aprime`) | YES | inside d3 loops; `pi_bothz_d3`/`pi_semiz_d3` are naming variants of the same slices |
| QuasiHyperbolic, all 7 trees | 3,752 | V1 / V2+N / V3+N per donor family | YES — one expectation per site, two discount scalings after; zero transforms | largest family |
| GulPesendorfer (incl. SemiExo, 2A) | 228 | V1 | YES — plain V; no temptation-twin expectation exists | |
| AmbiguityAversion (incl. AA RiskyAsset) | 110 | V1 | YES — plain V per prior, min after | uses `ambiguity_pi_z_J(:,:,jj,amb_c)` |
| RiskyAsset non-EZ (Raw/DC/GI/DCGI/SemiExo) | 305 | V1 | YES — z-expectation precedes the u/aprimeProbs machinery (which is untouched) | DC/GI variants already reshape to `[N_a,N_z]` after the sum: matmul lands there with no reshape |
| ResidualAsset (+SemiExo) | 8 | V2+N | YES | transient is `N_d*N_a^2*N_z^2` — worst per-site payoff in the repo |
| **subtotal, convertible now** | **~6,305** | | | |
| EpsteinZin-flavoured (EZ 132, RiskyAsset-EZ 48, RA-EZ-SemiExo 151) | 331 | — | **DEFERRED** | see below |
| InfHorz (hoisted `pi_z_alt` 47, `pi_z_alt2` 32, `Epi_z` 13, +2 inline) | 94 | V1/V2 after per-file check | YES (plain V) | different conventions: `pi_z_alt2` is UN-transposed, sums dim 4; verify each shape |

**Excluded (not matmul candidates):**

- **"Diagonal" sites, 240** (ExperienceAssetz plain 64 + its QH 128 + ExperienceAssetze
  plain 32 + one ze-SemiExo GI subdir 16): a2/a3prime depends on *current* z, so EV is
  already `(...,z_cur,z_prime)`-shaped before the broadcast. The contraction is a per-page
  weighted sum (`pagemtimes` territory), not `EV*pi'` — and the broadcast adds no extra
  N_z factor there (transient ≈ 2x EV, which is already N_z²-sized). No memory win from
  this proposal's recipe; leave unchanged.
- **InfHorz/InheritAsset_noa1_raw (1 site)**: EV is already 3-D `(d2,z',z)` with
  z-dependent gathers; same reason.
- **Row-slice sites (~1,550, incl. non-shiftdim spellings)**: lowmemory loops over today's
  z; transient is `[N_a,N]`, no blowup. Out of scope — and deliberately untouched under
  option A so lowmemory tiers stay bit-identical.
- **iid-vector sites (~2,850)**: `pi_e_J`/`pi_u`/`ambiguity_pi_u` broadcasts; the
  transient equals `size(EV)`, nothing squares. Out of scope.

**Why EZ is deferred**: the EZ-transformed `temp` entering the broadcast deliberately
carries infinities through the `ezc*` power transforms, and (in
`ValueFnIter_FHorz_EpsteinZin_raw.m`, uniquely) is sign-flipped by `ezc4`, so **+Inf can
enter the broadcast**. `+Inf*0=NaN` inside a matmul is not handled by a `-Inf`-only
clamp/restore, and a state with mixed ±Inf continuations sums to NaN under the broadcast
itself. With exact -Inf restoration the *downstream* hazards (the `isfinite`/`becareful`
masks, the `~isinf` mask in RA-EZ, the load-bearing `-Inf*0=NaN` in `EV1.*aprimeProbs`) all
disappear — downstream sees an identical pattern — so the open question is purely the
upstream ±Inf population. A dedicated later wave should establish, per EZ parameter regime,
whether `+Inf` actually occurs; if it can, the recipe needs a symmetric `+Inf`
clamp/restore plus a mixed-sign rule. Until then EZ keeps the broadcast (EZ models also
rarely need huge N_z).

**Future scope (not this proposal)**: TransitionPaths/ has 143 sites in 126 subcodes,
ValueFnFromPolicy/ has 9 (incl. the QH/GP/EZ FromPolicy files). Same recipes will apply;
they should get their own pass once ValueFnIter is done, since TPath subcodes are copies of
these raws and should follow, not lead.

## 4. Design decision (for the user)

**(a) Guarded path only — RECOMMENDED.** Each in-scope site gets
`if <transient elements> > 1e8` matmul else the existing 3 lines byte-identical.

- Pros: bit-identical behaviour for every existing run and every test-bank grid (all far
  below threshold), so **no re-baselining and no mandatory bank reruns**; every bank's
  exact-zero discipline survives untouched. Unblocks Guvenen immediately. Rollout can pause
  after any wave with the toolkit in a consistent state. The speed win at small N_z that
  option (b) would claim is microseconds/age — the real speed and memory wins live
  above the threshold, which the guard captures.
- Cons: two code paths per site (~6,300 sites) — copy-paste drift surface, the known
  V_Jplus1-drift class. Mitigated by: the else-branch is byte-identical to today's code
  (checked mechanically post-edit), and the matmul branch is one of three recipes.
  Threshold-crossing changes results at ULP between two runs of *different* sizes — not
  observable within any run.

**(b) Matmul as the new default everywhere.** Single code path, strictly faster/lighter.
But: the lowmemory row-slice branches are out of scope and stay on the broadcast, and every
bank cross-checks lowmemory tiers against lowmemory=0 as exact zeros — those checks WILL
break to ~1e-16-level diffs. Same for any cross-tier check where only one side's shape
variant changed. Consequence: re-baselining (or tolerance-converting) essentially **all**
GPU-green banks: CoreFHorzTests + withQH/withEZ/withGP/withAA subbanks, all six
CoreFHorzExpAsset*Tests, CoreFHorzRiskyAssetTests, CoreInfHorzTests, and eventually the
TPath banks. That is a large GPU programme and abandons the exact-zero discipline the banks
were built on.

**(c) vfoptions.EVmatmul switch.** All of (a)'s code plus defaults plumbing through ~30
dispatchers, to gain only a manual override nobody needs (the guard already picks correctly
by size). If a forced override is ever wanted for debugging, promote the inline constant
then. Not recommended.

Bank impact per option: (a) none mandatory (per-wave smoke runs recommended, §5);
(b) all of them; (c) as (a) plus dispatcher-default tests.

## 5. Rollout order and validation

### CPU instruments (this sandbox, `matlab -singleCompThread -batch`; raws themselves are
gpuArray-only so per-file execution happens on the GPU machine)

1. **Pattern unit tests** — one per recipe (V1, V1-bothz, V2, V3, V2+N, V3+N), EVtest
   Test-2 style: random source with fully-infeasible rows + scattered -Inf (+NaN population
   for the N recipes), ~70% structural zeros in pi, both pipelines, then assert: identical
   non-finite masks (`isequal`), all non-finite entries exactly `-Inf`, finite entries
   ≤1e-12 rel, and **all populations non-empty** (finite, -Inf, -Inf×zero-prob counts
   asserted >0 — no vacuous passes).
2. **Mutation tests of the checker** (per the measured three-instruments rule): break the
   matmul deliberately — drop the transpose, wrong fold order in V2/V3, omit the
   restoration line, omit the clamp, scrub NaN in a non-N recipe reference — and require
   the pattern test to FAIL each mutant.
3. **Static sweeps after each wave**: site-count reconciliation against this inventory;
   else-branch byte-identity against git HEAD; guard expression uses the file's own dims;
   checkcode clean on every edited file.

### GPU protocol (user, per wave)

- **Precondition for Wave 0**: rerun the updated EVtest.m (strict -Inf equality; also
  confirms gpuArray logical-mtimes).
- Per wave: (i) rerun that family's bank — expect **unchanged** output (guard inactive at
  bank sizes; any diff is a bug, not a re-baseline); (ii) one **window A/B**: a model with
  the transient in [2e8, 8e8] elements (e.g. N_a=501, N_z≈700-1200 — above the guard,
  below OOM), run on pre-edit vs post-edit toolkit, require Policy identical and V ≤1e-12
  rel.
- Wave 0 extra: Guvenen2007 `baseline/RunBaseline.m` stage (c) at N_z=3965 — the payoff test.

### Waves

| Wave | Files | Sites | GPU bank |
|---|---|---|---|
| 0 | `ValueFnIter_FHorz_DC1_nod_raw.m` | 2 | EVtest rerun + Guvenen stage (c) + CoreFHorzTests smoke |
| 1 | FHorz core: root quartet, DC, DC2, DC2A, GI, GI2A, DCGI(+DC2A) | 56 | CoreFHorzTests |
| 2 | FHorz/SemiExo (7 subdirs) | 168 | CoreFHorzTests semiz figs |
| 3 | ExpAsset plain families (minus diagonal exclusions) | ~1,676 | CoreFHorzExpAsset{,e,z,ze,U,semiz}Tests |
| 4 | QuasiHyperbolic trees | ~3,752 | withQuasiHyperbolicDiscounting + ExpAsset QH tiers |
| 5 | GulPesendorfer + AmbiguityAversion | 338 | withGulPesendorferPrefs, withAmbiguityAversion |
| 6 | RiskyAsset non-EZ + ResidualAsset | 313 | CoreFHorzRiskyAssetTests (+ResidAsset when its bank runs) |
| 7 | InfHorz (incl. hoisted-variable sites, per-file shape check) | 94 | CoreInfHorzTests |
| later | EZ-flavoured (needs ±Inf analysis); TPath + ValueFnFromPolicy (own proposal) | 331 + 152 | |

Each wave: edit → CPU pattern tests + static sweeps green → hand to GPU → user commits
after green. Waves 1-2 are hand-edited; waves 3-6 are agent-swept with per-file shape
verification and the static sweeps as the gate.

## 6. Decision needed

Approve one of (a)/(b)/(c) in §4 (recommendation: (a), threshold 1e8 transient elements),
and Wave 0 as the first edit.
