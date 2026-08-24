# Design notes

This document records *why* radiarandom is built the way it is, including the
places where the obvious approach turns out to be wrong. If you are only trying
to use the thing, read [README.md](README.md) instead.

---

## 1. What the device actually gives you

The RadiaCode 103 is a CsI(Tl) scintillator read out by a silicon
photomultiplier, with a 1024-channel pulse-height analyser. Over USB it speaks
a vendor protocol with several observables. Only one of them is any good as an
entropy source.

| Observable | Verdict |
|---|---|
| `VS_DATA_BUF` → `RealTimeData.count_rate` | **Unusable.** A smoothed fixed-point average (values land on multiples of 1/256). A filtered statistic, not a sample. |
| `VS_DATA_BUF` → `RawData.count_rate` | Also a float average, and only ~1 Hz. Derived from the same counts, so it carries no independent entropy. |
| `VS_DATA_BUF` → `AccelData`, `RareData.temperature` | Environmental, not quantum, and low-entropy. Mixed into the pool uncredited; never counted. |
| `VS_DATA_BUF` → `eid=1` sample arrays (gid 1/2/3) | Undecoded in the library and **never emitted** during normal operation — 12 polls over 12 s produced none. Presumably a calibration or oscilloscope mode. |
| `VS_SPECTRUM` | Cumulative since the last spectrum *reset*. Usable but redundant. |
| `VS_SPEC_DIFF` (0x206) | A device-side delta, sparse and compact (56–68 B versus a full 1024-channel read). Same information, same 2 Hz refresh — measured minimum inter-refresh gap 0.508 s. **Rejected: the read is destructive.** Each read clears the delta, so a dropped or failed USB read loses those photons permanently. |
| `VS_SPEC_ACCUM` | **This one.** A monotonic, cumulative, 1024-bin pulse-height histogram. |

Differencing two reads of `VS_SPEC_ACCUM` yields, exactly and without loss or
double-counting, the multiset of pulse-height channels of every photon detected
in between. That is strictly more information than any other endpoint offers,
and — the decisive property over `SPEC_DIFF` — the monotonic counter makes reads
*idempotent*: a dropped, delayed, or failed poll costs nothing, because the next
read recovers everything. For an entropy source, a transport hiccup that
silently discards samples is much worse than one that merely delays them.

`CPS_FILTER` reads 0 on this firmware, so the device is not applying its own
count-rate smoothing; `RAW_FILTER` is not a valid register here.

### Measurements on RC-103-013128 (firmware 4.14)

| Quantity | Value | How |
|---|---|---|
| Device refresh interval | **500 ms** | Polled at 276 Hz; counts appear in batches at exactly 0.5 s intervals |
| USB read latency | 3.6 ms median | 300 timed reads |
| Background count rate | 4.4 counts/s | 25-minute soak |
| Channel min-entropy `H∞` | **5.19 bits/photon** | Lifetime spectrum, 49,403,203 counts over 88 days; `p_max = 0.0274` at channel 1 |
| Channel Shannon entropy | 7.14 bits/photon | Same spectrum |
| Energy calibration | `E = 4.094 + 2.372·ch + 3.62e-4·ch²` keV | Device-reported |

The 2 Hz refresh means arrival times are only ever resolved to 500 ms, and
polling faster is cheap but tells you nothing new. That sounds like a serious
limitation and, at these count rates, it turns out to cost exactly nothing —
§2 works out why, and quantifies where it *would* start to matter.

---

## 2. The entropy model, and the one I got wrong first

### The wrong model

The first implementation reasoned like this: condition on the photon count `n`
in a batch; given `n`, the channels are i.i.d. draws worth `H_ch` bits each; the
device hands us a multiset rather than a sequence, and a multiset of size `n`
has at most `n!` preimages, so bank

```
n · H_ch − log2(n!)
```

This is wrong in **both** directions, and the test suite caught it:

- **It over-credits at low rates.** Min-entropy is fixed by the single most
  likely outcome. At 4.4 counts/s over a 500 ms window the most likely outcome
  is *no photons at all*, with probability `exp(−2.2) = 0.111`. The window is
  therefore worth `−log2(0.111) = 3.17` bits — not the 9.6 bits the formula
  claimed. You may condition on `n`, or you may decline to pay for `n`, but not
  both.

- **It collapses at high rates.** With a check source at 200 counts/s a batch
  holds ~100 photons, and `100 · 3.6 − log2(100!) = 360 − 525 < 0`. The formula
  credits *zero* entropy exactly when the detector is supplying the most. Since
  raising the count rate is the main lever a user has, this was fatal.

### The right model

Radioactive decay is a Poisson process, and each detected photon independently
lands in channel `i` with probability `p_i`. By the **Poisson splitting
theorem** the per-channel counts over a fixed window are *independent* Poisson
variables:

```
N_i ~ Poisson(μ_i),   μ_i = rate · p_i · window
```

Independence means min-entropies add, so a window is worth

```
H∞(window) = Σ_i H∞(Poisson(μ_i)),   H∞(Poisson(μ)) = −log2( P(N = ⌊μ⌋) )
```

This is exact under the model rather than a bound, and it behaves correctly at
both ends:

- At low rates every `μ_i ≪ 1`, the mode of each channel is 0, and the sum
  reduces to `log2(e) · rate · window` — **1.443 bits per photon**.
- At high rates the busy channels saturate individually (their min-entropy grows
  like `½log2(2πeμ)`), so the total keeps rising, sub-linearly and correctly.

**What we bank at background:** 4.4 counts/s × 1.443 = 6.35 bits/s, × 0.9 safety
= **5.7 bits/s ≈ 0.71 bytes/s**. A 256-bit seed costs 320 banked bits, so about
56 seconds.

Note that this is roughly *half* what the broken formula claimed. The generator
got slower and more honest at the same time.

### Why finer resolution buys nothing, and why buckets still matter

A natural question is whether the design is leaving entropy on the table by
only resolving arrival times to the device's 500 ms refresh — and whether the
pulse-height channel is even the right observable, versus timing.

The model answers both, and the answer is counterintuitive. Computed against
the measured spectra:

| | background, 4.4 c/s | with Am-241, 16.2 c/s |
|---|---|---|
| count only, no channel resolution | 1.899 bits/window | 2.842 bits/window |
| **channel-resolved (what we use)** | **3.174** | **11.686** |
| gain from resolving channels | **1.67×** | **4.11×** |
| linear ceiling, `rate·T·log2(e)` | 3.174 | 11.686 |
| **we are at** | **100.0% of ceiling** | **100.0% of ceiling** |

Splitting the 500 ms window into finer sub-windows — 250 ms, 50 ms, 5 ms,
0.5 ms — changes the total by **+0.00%** at both rates.

The reason: min-entropy is fixed by the single most likely outcome, and once
every cell is sparse (`μ_i ≪ 1`) that outcome is *"nothing arrived anywhere"*,
with probability `exp(−λT)` — which does not depend on how finely you subdivide
the cells. Subdividing time multiplies the number of cells and divides `μ` per
cell, leaving `Σ_i H∞(Poisson(μ_i)) = λT·log2(e)` exactly unchanged. The budget
is **rate-limited, not resolution-limited**.

Channel resolution is not exempt from that argument — it is just that going
from 1 bin to 1024 bins is what moved us *onto* the ceiling in the first place.
Counting alone sits well below it (the mode of a Poisson is not its zero), and
resolving channels closes the gap. Having closed it, further resolution of any
kind — finer time, more channels — is worth nothing.

Finer timing only starts to pay once the rate is high enough that cells stop
being sparse:

| count rate | 500 ms window | 0.5 ms sub-windows | gain | % of ceiling at 500 ms |
|---|---|---|---|---|
| 16 /s | 11.5 bits | 11.5 | 0.0% | 100.0% |
| 100 /s | 49.2 | 72.1 | +47% | 68.2% |
| 1 000 /s | 202.6 | 721.3 | +256% | 28.1% |
| 10 000 /s | 562.6 | 7 213.5 | +1182% | 7.8% |

So the crossover is somewhere near 50–100 counts/s. Below it, per-event
timestamps would be worth precisely nothing; above it they become the dominant
term. Since the device caps at 2 Hz regardless, the practical conclusion is
that **the only lever that increases entropy at these rates is the count rate
itself** — which is why the docs push check sources rather than faster polling.

### What is deliberately not counted

Photon arrival time within a batch, the Poisson fluctuation of `n` as a separate
quantity, and host timing jitter are all mixed into the pool but credited **zero
bits**. They can only help; the budget does not depend on them.

### Guarding the inputs to the model

The budget scales with the count rate, so an over-estimated rate over-credits.
Therefore:

- The rate is a **99% lower confidence bound**, `(k − 2.576√k)/t`, not the point
  estimate. It is measured on its own clock, which advances on every poll:
  sharing the credit clock (which only advances when the device demonstrably
  moved) re-counts the same interval whenever a poll finds nothing new, and
  under-reported the rate by 1.8× on real hardware — 8.97 photons/s against an
  actual 16.21.
- The spectrum used is whichever of the *reference* (device lifetime) and *live*
  distributions yields the **smaller** budget, so a spectrum that narrows earns
  less automatically.
- Entropy accrues only for time in which the device demonstrably advanced (new
  photons, or its own accumulation clock ticking), and any single gap is capped
  at 2 seconds, so a stalled detector cannot bank its way through the outage.
- Before the rate has been measured at all, the assessment credits **zero**.
  Entropy is never assumed, only observed.

---

## 3. Health tests, and why they are not the textbook ones

NIST SP 800-90B §4.4 specifies two continuous tests — the Repetition Count Test
and the Adaptive Proportion Test. Both are defined on an **ordered** sequence of
samples. The RadiaCode does not give us one: differencing the histogram yields a
multiset, and the order is gone.

Expanding that multiset into a sorted list and running the textbook tests on it
is wrong, and wrong in the worst possible way — it fails hardest when the
detector is working best:

- **Sorting manufactures runs.** Two photons in the same channel become adjacent
  by construction. At 4.4 counts/s this is rare; under a check source at a few
  hundred counts/s the RCT fires on every single batch of a perfectly healthy
  detector. This was observed, not theorised: `test_healthy_high_rate_stream_passes`
  exists because the first implementation failed it.
- **Sorting biases the APT.** The test keys on the *first* sample of each
  window. In ascending order that is systematically a low channel — precisely
  where the probability mass is concentrated — so match counts are inflated.

### What is used instead

**Proportion test** (replaces APT). Over a sliding window of 512 photons, no
single channel may account for more than `C` of them. This is the APT statistic
with the window's arbitrary first sample replaced by the *most common* channel,
which is strictly more sensitive. Taking a maximum over `K = 1024` channels
requires a multiple-comparison correction, so `C` is chosen by union bound
against `α/K = 2^−30` rather than `α = 2^−20`. Order-free by construction.

**Repetition test** (replaces RCT). Fires when consecutive batches have
identical, non-empty channel multisets. A live detector essentially never
repeats itself; a wedged device replaying a buffer does nothing else. Applied to
whole batches, where order is not involved.

### Calibrating to the detector instead of to an assumption

Both the proportion test and the spectral-shape test learn their baseline from
the first `calibrate_photons` of the run, not from a constant. Two live failures
forced this, and both are instructive:

**The shape test's reference was the device's lifetime spectrum.** That covers
88 days of wherever the detector happened to be, at whatever temperature, near
whatever. Today's background looks nothing like it, so the test warned every
four minutes — χ² ≈ 2980 against a threshold of 360 — with nothing wrong. A
warning that always fires trains the operator to ignore warnings, which is worse
than having no test. It now calibrates on the session and watches for drift from
*that*, with a single informational note if the session differs from the
lifetime spectrum.

**The proportion cutoff came from an assumed 4 bits/photon.** With an Am-241
check source on the detector, channel 25 held 71 of every 512 photons against a
cutoff of exactly 71, and capture halted after 100 seconds. This is backwards:
adding a source is the recommended way to raise the entropy rate, and the
Poisson budget already handles the narrower spectrum by crediting less per
photon. The cutoff is now fitted to the calibrated baseline (in that run: 107,
versus 71 assumed), never tighter than the assumed value and never looser than
a hard ceiling of half the window. During calibration only the ceiling applies —
safe, because calibration completes no later than the start-up test, which gates
every output path.

The general principle: a health test should detect *change* in the detector, not
disagreement with a guess about detectors in general.

### Detector-specific tests

A physically dead detector emits *no* samples rather than bad ones, so it would
sail past any test that only inspects the samples it does emit.

- **Stall.** Under Poisson the chance of silence lasting `t` is `exp(−rate·t)`;
  the limit is set where that reaches ~`2^−20`, floored by a grace period.
  Subtlety: the rate estimate feeding this must be measured **up to the last
  photon**, not up to now. Measuring to "now" lets a stall deflate the very rate
  estimate that sets the stall threshold, so the longer the detector stayed dead
  the more patient the test became — and it would never fire. That was a real
  bug, caught by `test_stall_detected_when_the_detector_goes_silent`.
- **Rate excursion.** A ±12σ Poisson deviation from baseline warns.
- **Spectral collapse.** χ² of the recent coarse spectrum against the reference,
  with a deliberately loose threshold (20× the degrees of freedom). Looking for
  a gain or bias-voltage fault, not for ordinary drift.

**Failures latch.** Once unhealthy, the generator stops rather than degrading
silently. For something that may be seeding keys, that is the only defensible
behaviour.

---

## 4. Conditioning, expansion, and what is actually guaranteed

```
photons ──▶ EntropyPool ──┬──▶ 256-bit blocks           "physical" mode
            HMAC-SHA-512  │    (320 banked bits each)
            = WHITENING   │
                          └──▶ HMAC_DRBG(SHA-512) ────▶ bulk output
                               = EXPANSION              "drbg" mode
```

### The DRBG is not the whitener

This is the single most common misreading of the design, so it is worth being
blunt: **the HMAC-SHA-512 pool does the whitening; the DRBG only does rate
expansion.** They are different jobs and they are separable, which is why
physical mode exists and works on its own.

The evidence is direct. Running the portable battery over the three streams:

| Stream | What it is | Result |
|---|---|---|
| `data/*.channels.u16` | raw channel values, no conditioning | **11 of 11 tests FAIL**, every one at p = 0.000000 |
| `data/*.physical.bin` | pool output, **no DRBG anywhere** | **10 of 10 tests pass** |
| DRBG output | pool output expanded by HMAC_DRBG | 10 of 10 tests pass |

The raw stream fails catastrophically and could not possibly be used directly:
it is a sorted sequence of 10-bit channel values drawn from a steeply falling
spectrum, so it is biased, autocorrelated, and monotone within each batch. The
conditioner fixes that. The DRBG, statistically speaking, changes nothing —
its input already passes.

### So what is the DRBG for?

Exactly one thing: **rate**. Physical output is bounded by radioactive decay at
0.7–2.5 bytes/s. Filling a 1 MB file would take four to sixteen days; the
Dieharder battery consumes gigabytes. HMAC_DRBG turns a 256-bit seed into
30 MB/s.

That is a real service, and it is the *only* service. Specifically, the DRBG:

- **does not improve statistical quality** — its input already passes;
- **does not add entropy** — no deterministic function can;
- **weakens the guarantee**, from "full entropy per SP 800-90C" to
  "computationally indistinguishable from random, at a 256-bit security
  strength, assuming HMAC-SHA-512 is a PRF".

Which is why the two modes are kept rigidly apart and every command says which
one it is using. If you are seeding a key, use `--physical` and pay the
seconds. If you are filling a disk, use the default and understand that you are
getting a stream cryptographically derived from a much smaller amount of
physical entropy.

What the DRBG *does* contribute beyond throughput is state-compromise
resilience: HMAC_DRBG updates its state after every generate call
(backtracking resistance), and continuous reseeding from fresh physical blocks
gives forward and backward secrecy across a compromise. Handing out raw pool
output at high rate would provide neither.

### The guarantee ladder, stated precisely

Being exact about this matters more than it looks, and my own first draft of
this document got it wrong.

**Level 1 — the entropy estimate.** `H∞(window) = Σ_i H∞(Poisson(μ_i))` is
*exact* given the rate and spectrum, by the Poisson splitting theorem. The
assumptions are physical: that decays are Poissonian (extremely well
established), that channel assignment is independent per photon (good, modulo
detector dead time), and that the measured rate and spectrum are accurate. The
0.9 safety factor and the 99% lower-bounded rate cover the last two.

**Level 2 — the conditioner.** SP 800-90C permits treating a *vetted
conditioning function*'s output as "full entropy" when the input min-entropy
exceeds the output length by at least 64 bits. Hence 320 banked bits per
256-bit block. **This is a standards-body construction, not a theorem.**
HMAC-SHA-512 under a fixed, published key is not a proven randomness extractor;
the claim rests on SHA-512 behaving like a random oracle. It is the same
assumption every SP 800-90-compliant RNG makes, and it is a conventional and
well-tested one — but it is computational, and physical mode is therefore *not*
unconditionally secure.

**Level 3 — the DRBG.** Computational security at a 256-bit strength, assuming
HMAC-SHA-512 is a PRF. Seeded with 256 bits of entropy input plus a 256-bit
nonce, against SP 800-90A minima of 256 and 128 respectively.

So the honest one-line summary: *no mode of this generator is
information-theoretically secure*. Physical mode is one assumption away
(SHA-512 as a good conditioner); DRBG mode is two (that, plus SHA-512 as a PRF
over a long output stream).

### What an unconditional guarantee would require

The gap at level 2 is closable, and it is worth knowing how, because the
quantum-RNG literature closes it as a matter of routine.

Replace the HMAC conditioner with a **2-universal hash family** — in practice a
Toeplitz matrix over GF(2) — and the **Leftover Hash Lemma** gives a theorem
rather than an assumption: if the input has `k` bits of min-entropy and you
output `m` bits, the result is ε-close to uniform in statistical distance with

```
ε = ½ · 2^−(k−m)/2
```

Setting `k − m = 64`, the same overhead this design already pays, yields
ε ≤ 2⁻³³ — **provably**, with no assumption about any hash function. A Toeplitz
matrix is defined by `n + m − 1` seed bits and applied as `m` parities, which is
cheap at these sizes: a few milliseconds per 256-bit block.

The catch, and the reason it is not the default:

- **The seed must be uniform and independent of the source.** A strong
  extractor's seed may be public, so it need not be kept secret, but it cannot
  be derived by hashing a constant without reintroducing exactly the assumption
  we were trying to remove. The correct construction is to generate the seed
  once from the device itself, before it extracts anything, and store it.
- **It adds an artefact to manage.** A seed file that must not be lost, must
  not be regenerated mid-stream, and must be distinguished from key material
  even though it is not secret.
- **It buys nothing against any realistic adversary.** An attacker who can
  break SHA-512 as a conditioner has already broken far more interesting things
  than this RNG.

The trade is therefore real but narrow: it converts one conventional assumption
into a theorem, at the cost of a stateful public seed. Worth doing for a
device whose entire selling point is a physical entropy source, and a natural
next step — but a deliberate choice rather than an oversight, and it should not
be presented as though the current build already provides it.

### Other options considered

| Option | Why not (or not yet) |
|---|---|
| **Von Neumann debiasing** | The classic choice for radioactive RNGs, and information-theoretic with no assumptions. But it is defined for i.i.d. *bits*; our samples are 10-bit symbols from a steeply non-uniform distribution, delivered as an unordered multiset. Applying it to the bit expansion throws away most of the entropy — the low channels dominate, so pairs are overwhelmingly equal and get discarded. Peres' iterated variant recovers more but still assumes i.i.d. |
| **Elias / arithmetic-coding extractor** | Asymptotically optimal and information-theoretic, but requires the source distribution to be known exactly. Ours drifts with temperature, geometry, and whatever source is nearby — the very thing the health tests exist to notice. Fragile in precisely the conditions this device operates in. |
| **XOR of two independent detectors** | Genuinely strong: two-source extraction needs far weaker assumptions, and an attacker must defeat both. Requires a second RadiaCode. Worth it for anyone who has one. |
| **XOR-folding one detector against itself** (temporal interleave) | **Measured and rejected — see below.** Halves the rate *and* destroys entropy, and is fragile to exactly the serial correlation we worry about. |
| **AES-CTR-DRBG instead of HMAC_DRBG** | Faster with AES-NI, and equally standard. HMAC_DRBG was chosen because it needs no cipher primitive beyond `hashlib`, keeping the package pure-Python with no build step — which matters more here than throughput that is already 30 MB/s. |
| **ChaCha20 expansion (the Linux kernel's choice)** | Same reasoning. No advantage over HMAC_DRBG at this scale, and less directly covered by SP 800-90A. |
| **No DRBG at all; physical only** | The purest option and still available: `--physical`. Rejected as the *default* because 0.7 bytes/s makes `gen -n 1048576` take days, and a tool whose default mode is unusable pushes people toward worse alternatives. |
| **Feed the OS pool and let the kernel expand** | This is exactly what `radiarandom feed` does on Linux, and it is the best answer where it is available: the kernel's DRBG is better reviewed than mine. It is not a substitute for the CLI, because Windows offers no equivalent (see §5) and because seeding a specific application often wants bytes in hand rather than a system-wide contribution. |
| **Skip conditioning, hand raw samples to the caller** | Available as `radiarandom raw`, for entropy assessment. Never as generator output: the raw stream fails all eleven statistical tests, and anyone using it as randomness would be badly wrong. |

### Why not XOR-fold the detector against itself

Interleaving the stream temporally and XORing the halves is an appealing idea:
halve the rate, get "better" randomness. It does make the output *look* better,
and it is still the wrong trade. There is a clean closed form for what it does.

For `X, Y` i.i.d. from the channel distribution `p`, the XOR distribution is
`q(z) = Σ_x p(x)p(x⊕z)`, whose largest term is `q(0) = Σ_x p(x)²`. So

```
H∞(X ⊕ Y)  =  H₂(X)      the Rényi-2 (collision) entropy of a single draw
```

Measured against the real spectra — the prediction is exact, not approximate:

| | background | with Am-241 |
|---|---|---|
| single photon `H∞` | 5.773 | 3.369 |
| single photon `H₂` (predicted XOR result) | 6.679 | 4.455 |
| **measured `H∞(X⊕Y)`** | **6.679** | **4.455** |
| consumed: `2 × H∞` | 11.546 | 6.738 |
| produced | 6.679 | 4.455 |
| **efficiency** | **57.8%** | **66.1%** |
| entropy destroyed per pair | 4.867 bits | 2.283 bits |

So yes, each *output symbol* carries more min-entropy than each input symbol
(6.68 > 5.77) — that is the "improvement". But two inputs were spent to make
one output, and 42% of the source entropy was thrown away. Since `H∞ ≤ H₂`
always, this is guaranteed to lose whenever the source is not already uniform:
XOR-folding is a fixed, distribution-dependent, lossy extractor.

Compare with hashing. The HMAC pool pays a *fixed* 64-bit overhead regardless
of block size, so its efficiency is `m/(m+64)` — 80% for a 256-bit block and
98.4% for a 4096-bit one, versus XOR's 58–66% that never improves. Hashing wins
on efficiency and gives a stronger guarantee. Iterating the XOR makes it worse,
not better: `k`-fold XOR drives the bias down geometrically but consumes `k`
symbols per output, so efficiency → 0.

The decisive objection is different, though. XOR-folding assumes the two halves
are **independent**, and it fails catastrophically when they are not — at the
limit `X = Y` gives `X ⊕ Y = 0`, entropy zero. Measured on real data at
various lags:

| lag between XORed samples | background | with Am-241 |
|---|---|---|
| 1 | 6.140 | **2.746** |
| 2 | 6.426 | 4.238 |
| 8 | 6.408 | 4.204 |
| 64 | 6.504 | 4.418 |
| 512 | 6.390 | 4.410 |
| i.i.d. prediction | 6.679 | 4.455 |

At lag 1 with a source present, XOR-folding yields **2.746 bits — less than the
3.369 bits a single un-folded photon carries.** Folding adjacent samples
actively destroys entropy, because adjacent samples are correlated (here
largely by the within-batch sort, but detector dead time and afterpulsing would
do the same). Correlation costs a hash function a little; it costs XOR
everything.

The general lesson: XOR is a good way to combine *independent sources* and a
bad way to condition *one* source. Two detectors XORed together buy real
insurance — an attacker must defeat both. A single detector XORed against
itself buys none of that, since one failing detector fails both halves at once,
while paying half the rate and a third of the entropy for the privilege.

### Banking: the pool cannot hold what the reservoir can

The pool is a running HMAC-SHA-512, so its accumulated state *is* a 512-bit
chaining value and it cannot carry more than 512 bits of min-entropy no matter
how much is absorbed. An earlier version counted credited bits without any
limit and was observed claiming 2240 banked bits in a 512-bit state. That claim
was simply false, and credit now saturates at the state size.

Accumulating a real reserve is therefore a separate job. Blocks are *extracted*
promptly into a bounded buffer -- 4 KiB by default -- where every byte is
genuine conditioned output rather than a claim about a hash state. Draining
eagerly also stops the pool from saturating and throwing the surplus away.

The reserve is deliberately finite. Its purpose is to absorb bursts (a run of
dice rolls, a handful of seeds), not to accumulate forever, and an unbounded
counter would misreport how much is actually available.

Serving small reads from the reserve matters more than it sounds. A one-byte
draw used to take a whole 32-byte block and discard 31 bytes, so a coin flip
cost the full 320 banked bits -- sixteen seconds of detector time at 20 bits/s
for eight bits of output. It also made the GUI's auto-repeat look broken: every
click drained a block, so the gauge read full one moment and empty the next.

### Continuous re-calibration

Every baseline adapts, because a fixed one is the "warns forever" bug waiting
for the next legitimate change -- a source added or removed, the detector
moved, the room warming up.

* The **spectral baseline** blends each completed window into the reference.
  A window that triggered a warning blends at 0.6 rather than 0.25: once the
  transition has been reported there is nothing to gain from reporting it again
  for the next half hour. Measured on a 300-channel shift: chi2 26426 -> 7302
  -> 1428 over three windows, then silence.
* The **proportion cutoff** is re-derived from a decaying channel histogram
  every 4096 photons, and immediately on an excursion.
* The **rate baseline** tracks the observed rate, more slowly during an
  excursion so a genuine collapse is reported for several windows first.
* The **live spectrum estimate** feeding the entropy budget decays with a
  20-minute half-life, so the budget prices the detector as it is rather than
  as it was averaged over the whole run.

Adaptation must not be able to hide degradation, so two things never move:

* the **proportion hard ceiling** -- one channel taking half the window is a
  stuck ADC whatever the baseline has learned, and that stays fatal;
* the **entropy budget**, which is recomputed from the live spectrum and simply
  credits less as the detector gets worse. That is the real protection against
  being boiled slowly: a degraded detector does not get waved through, it earns
  less and the output rate falls.

This is also why exceeding the *adaptive* cutoff is a warning rather than a
failure. A check source legitimately puts 20% of counts in one channel; failing
there would reject the configuration the documentation recommends, while the
budget already prices the narrower spectrum correctly.

### Reseeding

The mandatory reseed interval is 2²⁰ requests — loose on purpose. An earlier
value of 1024 meant blocking for ~26 s of radioactive decay every 64 MB of
output, which bought nothing: HMAC_DRBG's security does not decay with output
volume (SP 800-90A permits up to 2⁴⁸ requests). Reseeding still happens
*opportunistically* every time a fresh block is available, which is what
actually provides the state-compromise resilience described above.

### Why DRBG mode needs a pump thread

The DRBG generates at ~30 MB/s while the device is polled at 5 Hz. Without a
background thread reading the device, the pool would never refill and reseeds
would stall. `radiarandom gen` starts one in DRBG mode and pumps inline in
physical mode.

### Pool mechanics

The pool is a running HMAC-SHA-512. Extraction finalises it, splits the 512-bit
digest into a 256-bit output block and a 256-bit chaining value, and rekeys from
the chaining value so no absorbed entropy is discarded. The 64 bits of
min-entropy above the output length are not re-credited to the next block — they
are spent, which is the conservative choice.

---

## 5. Platform integration

**Linux** has `RNDADDENTROPY` on `/dev/random`: mix a buffer in *and* credit a
specific number of entropy bits, visible in
`/proc/sys/kernel/random/entropy_avail` and benefiting every `getrandom(2)`
caller. This is the real thing, and it is what `radiarandom feed` does. The
credit is the assessed min-entropy, never the buffer length — over-crediting the
kernel is worse than not contributing at all, because the kernel will then hand
out bytes believing it has entropy it does not. Requires `CAP_SYS_ADMIN`; without
it the daemon falls back to plain uncredited writes, which still stir the pool
and are honest about contributing nothing countable.

**Windows** has no supported equivalent, and this project says so rather than
inventing one. `BCRYPT_RNG_USE_ENTROPY_IN_BUFFER` is documented as *"ignored in
Windows 8 and later"*; `CryptGenRandom` is deprecated; the
`HKLM\...\Cryptography\RNG\Seed` registry value is a legacy artefact; `cng.sys`
exposes no user-mode contribution interface. So Windows gets an opt-in
named-pipe service instead. See [packaging/windows/README.md](packaging/windows/README.md).

---

## 6. Validation strategy

Two questions, two tools, and they are not interchangeable.

**"Does the output stream look uniform?"** → Dieharder. But note carefully what
this can and cannot establish. The generator's normal output passes through
HMAC_DRBG, and *a correctly implemented DRBG passes Dieharder whether or not its
seed was any good*. A clean Dieharder sweep validates the plumbing — framing,
byte order, conditioning, reseeding, the whole path end to end — and nothing at
all about the physics. It is necessary and it is not sufficient. Anyone who
shows you a hardware RNG "validated by Dieharder" and nothing else has shown you
that their DRBG works.

**"How much unpredictability does the physical source actually supply?"** → the
SP 800-90B estimators in `validation/sp800_90b.py`, run on the *raw* channel
stream before any conditioning. These need ~10⁶ samples rather than gigabytes,
which is achievable at 4.4 counts/s in about 63 hours (or far less with a check
source).

The physical-mode stream is the one whose statistics genuinely reflect
radioactive decay, but at ~0.7 bytes/s a full Dieharder battery would take
years. Raising the count rate with a check source is the only practical route to
a real physical-mode Dieharder run; entropy rate scales roughly linearly with
count rate until the low channels start to saturate.

### Estimator implementation notes

The estimators were validated against known-answer sources before being trusted:

| Source | True `H∞` | Reported minimum |
|---|---|---|
| `os.urandom` bytes | 8.0 | 6.43 (conservative) |
| Biased bits, `p = 0.9` | 0.152 | 0.147 |
| Uniform 4-bit symbols | 4.0 | 2.95 |
| Period-8 counter | 0.0 | **0.000** |

The compression estimator reads low on high-entropy sources (0.78 bits/bit on
ideal data at 60 k samples, rising to 0.88 at 1 M). That is the documented
conservatism of the statistic, not a defect: its expected value is very flat in
`p` near uniform, so the 99% confidence subtraction costs a lot when samples are
few. It was checked against theory — the `G(z)` function predicts a mean of
5.2177 for a uniform 6-bit source where the observed mean was 5.2170 and Monte
Carlo gives 5.2198.

One aggregation subtlety worth stating: an estimator that returns exactly `0.0`
must not be filtered out as falsy when taking the minimum. A zero is the single
most important result the battery can produce.

---

## 7. Known limitations

- **No mode is information-theoretically secure.** Physical mode's "full
  entropy" is the SP 800-90C construction and rests on HMAC-SHA-512 being a good
  conditioning function; DRBG mode adds a PRF assumption on top. §4 sets out
  what a 2-universal extractor plus the Leftover Hash Lemma would buy and what
  it would cost. If you need an unconditional guarantee, this build does not
  give you one.
- **It is slow.** ~0.7 bytes/s of true entropy on indoor background. This is a
  seeding device, not a bulk source. Bulk output is DRBG-expanded and says so.
- **Timing entropy is left on the table.** The 2 Hz device refresh caps arrival
  resolution at 500 ms, and we bank none of it. A device exposing per-event
  timestamps would do much better.
- **The i.i.d. assumption is not verified online.** Detector dead time and
  afterpulsing introduce short-range correlations. The 0.9 safety factor is a
  blunt instrument covering this; the SP 800-90B predictors are the real check
  and they run offline.
- **Single detector.** There is no cross-check against a second independent
  source, so a subtly compromised device would be caught only by the health
  tests and the offline estimators.
- **The pure-Python estimators are a cross-check, not a certification.** For a
  citable assessment run NIST's own
  [SP800-90B_EntropyAssessment](https://github.com/usnistgov/SP800-90B_EntropyAssessment)
  against the 8-bit export (`--export-nist`).
