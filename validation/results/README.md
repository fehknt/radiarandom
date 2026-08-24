# Validation results

Hardware: **RadiaCode 103, serial RC-103-013128, firmware 4.14**
Host: Windows 11 Pro for Workstations 26200, Python 3.14
Dieharder: 3.31.1 (Debian, under WSL2 kernel 6.18)
Date: 2026-08-23

Two kinds of evidence, kept apart on purpose. Dieharder speaks to the
**pipeline**; the SP 800-90B estimators speak to the **physics**. A report that
blurs them is worse than no report, because a correct DRBG passes Dieharder
regardless of whether its seed was any good.

---

## 1. Dieharder

### Complete battery

| | |
|---|---|
| Transcript | [`dieharder-drbg-wsl.txt`](dieharder-drbg-wsl.txt) |
| Command | `python3 validation/wsl_stream.py data/soak.physical.bin \| dieharder -a -g 200` |
| Tests run | **114** (the full `-a` battery) |
| PASSED | **111** |
| WEAK | 3 |
| **FAILED** | **0** |

The three weak results are all high-side p-values:

| Test | ntup | p |
|---|---|---|
| `diehard_craps` | 0 | 0.99732 |
| `sts_serial` | 6 | 0.99730 |
| `sts_serial` | 7 | 0.99983 |

Three flags across 114 tests is within expectation — dieharder marks WEAK
outside roughly `[0.005, 0.995]`, so about one per hundred tests is normal
noise. The two `sts_serial` entries are adjacent variants computed over the
same data and are not independent observations, so this reads as one mild
excursion plus one, not a pattern. Nothing failed.

The generator was fed from a real detector seed (`data/soak.physical.bin`,
captured by `radiarandom raw`) and expanded through the same HMAC_DRBG the CLI
uses. It was run inside WSL to remove a pipe bottleneck; the code path is
identical.

### Live-detector run

`radiarandom gen --stream` on Windows, piped into `dieharder -a -g 200` in WSL,
with the generator reseeding continuously from the detector throughout.

| | |
|---|---|
| Transcript | [`dieharder-drbg-full.txt`](dieharder-drbg-full.txt) |
| Generator log | [`dieharder-drbg-live-generator.log`](dieharder-drbg-live-generator.log) |
| Tests completed | **87** |
| PASSED | 86 |
| WEAK | 1 (`rgb_lagged_sum` ntup=5, p=0.99707) |
| **FAILED** | **0** |

This run did not finish the battery, for two reasons worth stating plainly:

1. **Throughput.** The Windows→WSL pipe capped dieharder at ~1.7e5 rands/s. The
   later `rgb_lagged_sum` variants consume 400 MB each, so the remainder would
   have taken more than twelve hours. The bottleneck is the pipe, not the
   generator, which does ~30 MB/s.

2. **It tripped its own health test — and that was the bug.** After 37 minutes
   the run halted with `rct: channel 24 repeated 7 times (cutoff 7)`. This was
   a **false positive on a healthy detector**: the textbook SP 800-90B
   Repetition Count Test is defined on an *ordered* sample sequence, but the
   device reports an unordered histogram, and expanding that into a sorted list
   manufactures runs. The generator halting was correct behaviour given its
   rules; the rules were wrong. See [DESIGN.md §3](../../DESIGN.md).

   The order-sensitive RCT and APT have been replaced with order-free
   equivalents carrying the same `α = 2⁻²⁰` discipline, and a subsequent live
   `selftest` passed cleanly.

### What these runs establish, and what they do not

They establish that the output path is sound end to end: framing, byte order,
conditioning, DRBG construction, reseeding, and binary output all behave. They
establish **nothing** about the detector, because HMAC_DRBG would launder a
constant seed into a stream that passes just as cleanly. For the physics, see
section 2.

Reproduce either run with [`../run_dieharder.sh`](../run_dieharder.sh).

---

## 2. SP 800-90B entropy estimation

Run on the **raw channel stream**, before any conditioning
(`python validation/sp800_90b.py data/<capture>.channels.u16`):

| Capture | Photons | Measured H∞ | Binding estimator | Banked | Margin |
|---|---|---|---|---|---|
| Background | 7,054 | **3.69 bits/photon** | compression | 1.30 | 2.8× |
| With Am-241 source | 31,090 | **1.80 bits/photon** | t-Tuple | ~1.23 | 1.5× |

Both are conservative, but note the second row: a strong low-energy source
raises the count rate while *narrowing* the spectrum, so the per-photon entropy
falls and the margin tightens. It remains positive, and the Poisson budget
tracks it automatically because it is computed from the measured spectrum
rather than a constant.

Per-estimator breakdown for the with-source capture
([`sp800_90b-with-source.json`](sp800_90b-with-source.json)):

| Estimator | H∞ (bits/photon) |
|---|---|
| most common value | 3.31 |
| t-Tuple | **1.87** |
| LRS | 2.78 |
| MultiMCW | 3.31 |
| Lag | 2.70 |
| MultiMMC | 2.21 |
| LZ78Y | 2.21 |
| collision (×10 bits) | 3.77 |
| Markov (×10 bits) | 4.46 |
| compression (×10 bits) | 2.80 |

The four prediction estimators (MultiMCW, Lag, MultiMMC, LZ78Y) are the ones
that probe serial structure — exactly the correlation that detector dead time
and afterpulsing would introduce, and the main risk to the model's
independence assumption. All sit comfortably above the banked figure.

SP 800-90B asks for 10⁶ samples for the non-IID track; these captures are
smaller, so the bounds are wider than the standard intends. Collect more with
`radiarandom raw --duration 86400` before treating any of this as a formal
assessment, and cross-check with NIST's reference tool via `--export-nist`.

### The estimators were themselves validated

They must never *over*-report. Checked against sources whose min-entropy is
known exactly:

| Source | True H∞ | Reported | |
|---|---|---|---|
| `os.urandom` bytes | 8.000 | 6.431 | conservative |
| Uniform 4-bit symbols | 4.000 | 2.955 | conservative |
| Biased bits, p = 0.9 | 0.152 | 0.147 | conservative, close |
| Period-8 counter | 0.000 | **0.000** | correctly detects zero |

The compression estimator reads low on high-entropy sources — 0.78 bits/bit on
ideal data at 60 k samples, 0.83 at 250 k, 0.88 at 1 M. That is the documented
conservatism of the statistic, not a defect: its expected value is very flat in
`p` near uniform, so the 99% confidence subtraction costs a lot when samples
are few. The `G(z)` implementation was checked against theory and Monte Carlo
for a uniform 6-bit source:

| | mean log₂ distance |
|---|---|
| `G(z)` prediction | 5.2177 |
| Observed | 5.2170 |
| Monte Carlo | 5.2198 |

These known-answer checks run in CI (`.github/workflows/ci.yml`, job
`estimators`).

---

## 3. Measured detector characteristics

Two regimes on the same detector on the same day — a useful demonstration that
the entropy model tracks the hardware rather than a hardcoded constant.

| | Background | With a check source |
|---|---|---|
| Count rate | 4.4 photons/s | **16.2 photons/s** |
| Channel min-entropy | 5.77 bits/photon | **3.26 bits/photon** |
| Modal channel | 37 (~92 keV) | **25 (~64 keV)** |
| Counts below channel 64 | 67.7% | **90.4%** |
| **Banked entropy rate** | 5.7 bits/s (0.71 B/s) | **~20 bits/s (2.5 B/s)** |

A source raises the rate 3.7× but narrows the spectrum, nearly halving the
per-photon min-entropy. A naive "bits per photon × rate" model would get this
wrong in one direction or the other; the Poisson model absorbs both effects and
reports a ~3.5× net gain.

The mode at channel 25 corresponds to ~63.6 keV on this device's calibration
(`E = 4.094 + 2.372·ch + 3.62e-4·ch²`), consistent with the **59.5 keV Am-241
line** from a smoke-detector source.

Lifetime reference spectrum (49,403,203 counts over 88 days): `p_max = 0.0274`
at channel 1 → **H∞ = 5.19 bits/photon**, Shannon 7.14.

### Health tests under a check source

A 35-minute capture with the source in place, verbose:

```
[info] shape: spectral baseline calibrated on 2051 photons
[info] proportion: proportion cutoff set to 107 of 512 (assumed 71, fitted 107,
       ceiling 256; busiest channel holds 9.7%, was 256 during calibration)
[warning] shape: session baseline differs from the device lifetime spectrum
       (chi2=1339, dof=13). Expected if the detector has moved or a source is
       present; monitoring drift from the session baseline instead
```

31,090 photons, zero health failures. Before the calibration fix this same
configuration halted after 100 seconds.

---

## 4. Kernel pool contribution (Linux)

Verified against a real kernel (WSL2 6.18), as root:

```
can_credit    : True
bits credited : 2048
entropy_avail : 256 (before and after)
over-credit 9999 bits on a 4-byte buffer -> clamped to 32
```

`entropy_avail` does not visibly rise because kernels 5.6 and later keep a
fixed 256-bit "initialised" pool rather than a depleting counter. The ioctl is
accepted; the contribution is real. The full feed daemon was also exercised end
to end against a simulated detector on that kernel: 2,579,968 bits credited over
a six-second run.

Windows has no equivalent and none is claimed — see
[packaging/windows/README.md](../../packaging/windows/README.md).

---

## 5. Which stage actually does the whitening

Run to settle the question directly, since it is easy to assume the DRBG is
what makes the output look random:

| Stream | What it is | Bytes | Result |
|---|---|---|---|
| `data/soak2.channels.u16` | raw channel values, **no conditioning** | 62,180 | **11 of 11 tests FAIL**, every one at p = 0.000000 |
| `data/soak2.physical.bin` | pool output, **no DRBG anywhere** | 3,584 | **10 of 10 tests pass** |
| HMAC_DRBG output | pool output, expanded | 3,584 | 10 of 10 tests pass |

Reproduce with:

```bash
python validation/localtests.py data/soak2.channels.u16 --quick
python validation/localtests.py data/soak2.physical.bin --quick
```

So: the **HMAC-SHA-512 pool is the whitener**, and it works on its own. The
raw stream is unusable as randomness — sorted, biased, autocorrelated, and it
fails everything. The DRBG changes nothing statistically because its input
already passes; it exists purely to convert 0.7–2.5 bytes/s into 30 MB/s, and
in doing so it trades "full entropy" for computational security. See
[DESIGN.md §4](../../DESIGN.md) for the full guarantee ladder and the
alternatives considered.

---

## 6. Portable battery

```
python validation/localtests.py --self-test
```

Thirteen classical tests, verified against known-good and known-bad streams:

| Stream | Expected | Result |
|---|---|---|
| `os.urandom`, 1 MiB | pass | all 13 passed |
| Byte counter, 1 MiB | fail | 6 failures |
| LSB-stuck stream, 512 KiB | fail | 12 failures, incl. `bit_position_bias` |

---

## 7. Bugs this validation exercise found

Recorded because they are the argument for doing the work rather than asserting
the result. Every one is now pinned by a test.

| Bug | Found by |
|---|---|
| Entropy formula over-credited at low rates and collapsed to zero at high rates | unit test on batch-entropy monotonicity |
| Order-sensitive RCT false-positived on healthy data | 37-minute live dieharder run |
| Proportion cutoff rejected a legitimate check source after 100 s | live soak capture |
| Shape test warned forever, comparing today against an 88-day lifetime average | live generator log |
| Stall detector could never fire — the stall deflated its own threshold | unit test |
| Rate estimator double-counted elapsed time, under-reporting the rate by 1.8× | live `bench` vs `info` disagreement |
| Rate-excursion test warned constantly (circular baseline) | CLI test log |
| Seed files corrupted on Windows: `os.open` defaults to text mode, 0x0A → 0x0D 0x0A | intermittent test failure |
| `gen` leaked a pump thread per invocation, spinning on a closed device | CLI suite slowing to a crawl |
| Raw capture buffered without flushing, losing hours on interrupt | live soak restart |
| `--format uuid4` advertised but unimplemented | format-coverage test |
| `TcpEntropyServer.stop()` deadlocked if never served | pool test hang |
| `pools.linux` unimportable on Windows (`import fcntl` at module top) | cross-platform test run |
