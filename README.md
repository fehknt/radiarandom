# radiarandom

A hardware random number generator built on a **RadiaCode 103** gamma
spectrometer.

Entropy comes from radioactive decay. Every count the detector reports is one
gamma photon depositing energy in its CsI(Tl) scintillator, and both *when* a
nucleus decays and *how much* energy the photon carries are quantum
indeterminate. The generator reads the device's channel-resolved event counter,
models the result as independent Poisson processes, banks a conservative
min-entropy budget, and feeds it through a vetted conditioner.

It runs on **Linux**, where it can contribute credited entropy to the kernel
pool, and on **Windows**, where — as explained below — no operating system can
accept such a contribution, so it serves entropy to applications instead. On
both it works as a standalone tool that writes random numbers to a file or the
terminal.

---

## Quick start

```bash
pip install -e ".[usb]"
radiarandom info
```

```bash
radiarandom gen -n 32 --format hex
```

```bash
radiarandom gen -n 1048576 --format bin -o random.bin
```

```bash
radiarandom int --min 1 --max 6 --count 5
```

> **First output takes a few minutes.** The start-up test requires 1024 photons
> before anything is emitted, which is about four minutes on indoor background,
> and the DRBG then needs two full-entropy blocks to seed. Every command shows a
> progress bar on stderr while it waits. `--startup-samples N` shortens it, at
> the cost of a weaker guarantee.

---

## The numbers, honestly

Measured on RC-103-013128, firmware 4.14, in both regimes on the same day:

| | Indoor background | With a check source |
|---|---|---|
| Count rate | 4.4 photons/s | **16.2 photons/s** |
| Channel min-entropy | 5.77 bits/photon | 3.26 bits/photon |
| Modal channel | 37 (≈92 keV) | 25 (≈64 keV) |
| **True entropy rate** | **5.7 bits/s ≈ 0.71 B/s** | **≈20 bits/s ≈ 2.5 B/s** |
| Time per 256-bit seed | ≈ 56 s | ≈ 16 s |

Fixed properties: device refresh 2 Hz (so arrival times resolve no finer than
500 ms), USB read latency 3.6 ms, DRBG throughput ≈ 30 MB/s. The lifetime
reference spectrum (49.4 M counts over 88 days) gives H∞ = 5.19 bits/photon,
Shannon 7.14.

**This is a seeding device, not a bulk source.** A couple of bytes per second
is what physics gives you at these count rates. Everything faster is
DRBG-expanded, and the tool never blurs the two.

### Making it faster

Entropy rate scales with count rate, so a check source parked against the
detector helps a lot: a thoriated lantern mantle or welding rod, uranium glass,
the americium pellet from a smoke detector, or potassium-chloride salt
substitute. `radiarandom info` prints what 10× and 100× would give you on your
device.

Note the second column above, though — it is not a pure win. A source raises
the rate 3.7× but *narrows* the spectrum, nearly halving the per-photon
min-entropy, because most counts now pile into one photopeak. (The mode at
channel 25 ≈ 64 keV is the 59.5 keV Am-241 line.) A naive "bits per photon ×
rate" model would get this wrong in one direction or the other. The Poisson
model uses the measured spectrum, absorbs both effects, and still reports a
~3.5× net gain. The health tests calibrate to the peak rather than tripping
over it.

---

## Two output modes, never confused

| | `--physical` | default (DRBG) |
|---|---|---|
| Rate | ~0.7–2.5 bytes/s | ~30 MB/s |
| Bytes out per byte of physical entropy | 1 : 1.25 | unbounded |
| Guarantee | "full entropy" per SP 800-90C — see below | computational, 256-bit strength |
| Mechanism | 256-bit blocks from the conditioner, released only as fast as the detector supplies min-entropy | HMAC_DRBG(SHA-512), seeded and continuously reseeded from those blocks |
| Use it for | seeding, key material, anything where "how much entropy" is the question | filling files, test batteries, anything where "how many bytes" is the question |

**Being precise about "full entropy."** Physical mode emits 256 bits per 320
banked bits of min-entropy, which is the SP 800-90C rule for treating a *vetted
conditioning function*'s output as full entropy. That is a standards-body
construction, not an information-theoretic theorem: HMAC-SHA-512 with a fixed,
published key is not a proven randomness extractor, and the claim rests on
SHA-512 behaving like a random oracle. It is a strong, conventional assumption —
the same one every SP 800-90 compliant RNG makes — but it is a computational
assumption, and physical mode is therefore *not* unconditionally secure.
[DESIGN.md §4](DESIGN.md) sets out what an unconditional guarantee would require
(a 2-universal extractor with the Leftover Hash Lemma) and why that is not the
default here.

```bash
radiarandom gen -n 32 --physical --format hex     # slow, full entropy
radiarandom gen -n 32 --format hex                # fast, DRBG
```

---

## How it works

```
RadiaCode 103
  │  VS_SPEC_ACCUM: cumulative 1024-bin pulse-height histogram, refreshed at 2 Hz
  ▼
difference consecutive reads
  │  → the multiset of pulse-height channels of every photon since the last read
  ▼
health tests ──── proportion · repetition · stall · rate · spectral shape
  │               (failure latches; output stops rather than degrading)
  ▼
entropy budget ── Σ_i H∞(Poisson(rate · p_i · window)) × safety
  │               credited only for confirmed-live detector time
  ▼
EntropyPool ───── HMAC-SHA-512, 320 banked bits per 256-bit block
  │
  ├──▶ physical mode: full-entropy blocks
  └──▶ HMAC_DRBG(SHA-512) ──▶ bulk output
```

The entropy model treats per-channel counts as independent Poisson variables —
which they provably are, by the Poisson splitting theorem — and sums their
min-entropies. [DESIGN.md](DESIGN.md) explains why, including the two earlier
formulations that were wrong and how the test suite caught them.

Deliberately **not** counted toward the budget: photon arrival timing, host
jitter, and the count fluctuation itself. They are mixed into the pool anyway,
where they can only help.

---

## Linux: contributing to the kernel entropy pool

```bash
sudo radiarandom feed
```

This uses `RNDADDENTROPY` on `/dev/random` — the same interface `rngd` uses — to
mix in each 256-bit block **and credit exactly 256 bits**. The contribution is
visible in `/proc/sys/kernel/random/entropy_avail` and benefits every consumer
of `getrandom(2)`.

The credit is always the assessed min-entropy, never the buffer length.
Over-crediting the kernel is worse than not contributing at all, because the
kernel will then hand out bytes believing it holds entropy it does not.

Needs `CAP_SYS_ADMIN`. Without it the daemon falls back to plain uncredited
writes, which still stir the pool and are honest about counting for nothing.

Install as a service:

```bash
sudo cp packaging/linux/99-radiacode.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo cp packaging/linux/radiarandom.service /etc/systemd/system/
sudo systemctl enable --now radiarandom
```

The unit runs unprivileged apart from `CAP_SYS_ADMIN`, with `ProtectSystem=strict`
and a `@system-service` syscall filter.

---

## Windows: there is no OS pool to feed

**Windows has no supported API for adding entropy to the operating system's
RNG,** and this project says so instead of pretending otherwise:

- `BCryptGenRandom`'s `BCRYPT_RNG_USE_ENTROPY_IN_BUFFER` flag is documented by
  Microsoft as *"ignored in Windows 8 and later"*.
  ([reference](https://learn.microsoft.com/en-us/windows/win32/api/bcrypt/nf-bcrypt-bcryptgenrandom))
- `CryptGenRandom` is deprecated and sits on the same CNG DRBG.
- `HKLM\SOFTWARE\Microsoft\Cryptography\RNG\Seed` is a legacy artefact, not a
  modern entropy input.
- `cng.sys` exposes no user-mode contribution interface.

So there is no `rngd` for Windows, from this project or anyone else. What you
get instead is an opt-in service:

```powershell
radiarandom serve --transport pipe          # \\.\pipe\radiarandom
radiarandom serve --transport tcp --port 7373
radiarandom seed-file C:\ProgramData\radiarandom\seed.bin -n 64
```

The named pipe is created with `PIPE_REJECT_REMOTE_CLIENTS`, so it is local-only.
TCP refuses to bind off-loopback without `--allow-remote`. Full details,
including running it as a Windows service and the libusb DLL trap, are in
[packaging/windows/README.md](packaging/windows/README.md).

---

## Validation

Two questions, two tools. They are not interchangeable, and conflating them is
the most common way hardware RNG claims go wrong.

### 1. Does the output stream look uniform? → Dieharder

```bash
./validation/run_dieharder.sh --mode drbg
```

This streams the generator straight into `dieharder -a -g 200`, so no test is
ever starved of data and nothing is rewound. Results from the run on this
hardware are in [`validation/results/`](validation/results/).

**What a clean sweep does and does not prove.** The generator's normal output
passes through HMAC_DRBG, and a correctly implemented DRBG passes Dieharder
whether or not its seed was any good. A clean sweep validates the *plumbing* —
framing, byte order, conditioning, reseeding, the whole path end to end — and
says nothing about the physics. It is necessary and it is not sufficient.

On Windows, use WSL, a Linux host, or the provided container:

```bash
docker build -t radiarandom-dieharder validation/
radiarandom gen --stream -f bin -o - --quiet | docker run --rm -i radiarandom-dieharder dieharder -a -g 200
```

### 2. How much entropy does the physics actually supply? → SP 800-90B

```bash
radiarandom raw --duration 3600 --prefix data/soak
python validation/sp800_90b.py data/soak.channels.u16 --compare 5.19
```

This runs the full non-IID track of SP 800-90B §6.3 — Most Common Value,
Collision, Markov, Compression, t-Tuple, LRS, MultiMCW, Lag, MultiMMC and LZ78Y
— on the **raw** channel stream, before any conditioning, and reports the
minimum across estimators. It needs ~10⁶ samples rather than gigabytes, which is
about 63 hours at background or a couple of hours with a check source.

The estimators were themselves validated against known-answer sources:

| Source | True H∞ | Reported |
|---|---|---|
| `os.urandom` | 8.0 | 6.43 (conservative) |
| Biased bits, p=0.9 | 0.152 | 0.147 |
| Uniform 4-bit | 4.0 | 2.95 |
| Period-8 counter | 0.0 | **0.000** |

For a citable assessment, export for NIST's own reference tool:

```bash
python validation/sp800_90b.py data/soak.channels.u16 --export-nist data/nist8.bin
ea_non_iid -i -a -v data/nist8.bin 8
```

### 3. Quick pre-flight → portable battery

Runs anywhere Python does, in seconds, with no dieharder needed:

```bash
python validation/localtests.py random.bin
python validation/localtests.py --self-test
```

Thirteen classical tests with real p-values (monobit, block frequency, runs,
longest run, binary matrix rank, cumulative sums, approximate entropy, serial
correlation, byte χ², poker, per-bit-position bias, birthday spacings). The
`--self-test` mode checks the battery itself against known-good and known-bad
streams.

---

## Health tests

Run continuously; a failure latches and stops output.

| Test | Catches |
|---|---|
| **Proportion** | any channel exceeding its cutoff in a sliding 512-photon window — a stuck ADC or collapsed gain |
| **Repetition** | consecutive identical batches — a wedged device replaying a buffer |
| **Stall** | silence far longer than Poisson permits — a dead or unplugged detector |
| **Rate excursion** | count rate collapsing or saturating, over a bounded 60 s window |
| **Spectral shape** | drift away from the session baseline — gain or bias-voltage faults |
| **Start-up** | 1024 photons must pass before any output |

The proportion and shape tests **calibrate to your detector** during the
start-up window rather than to a hardcoded spectrum, then watch for departures
from that baseline. This matters in practice: a check source puts a sharp
photopeak in the spectrum, and a fixed cutoff would reject the very setup that
makes the generator faster. The calibrated cutoff is never allowed past a hard
ceiling of half the window, so a genuinely stuck channel is still caught.

These are *not* the textbook SP 800-90B §4.4 tests, and that is deliberate: the
device hands us an unordered multiset, and the RCT and APT are both defined on
ordered sequences. Running them on a sorted expansion manufactures runs and
biases the window statistic, failing hardest on a healthy detector under a check
source. The replacements are order-free with the same `α = 2⁻²⁰` discipline.
[DESIGN.md §3](DESIGN.md) has the full argument.

```bash
radiarandom selftest
```

---

## The GUI

```bash
radiarandom gui
```

```
┌──────────────────────────────────────────────┐
│ RC-103-013128 — firmware 4.14                │
│ 16.2 counts/s · 19.2 bits/s entropy          │
│ ████████████████████████████████████████████ │
├──────────────────────────────────────────────┤
│                                              │
│                     79                       │
│                                              │
├─ Quick picks ────────────────────────────────┤
│ [Coin flip] [D6] [1 – 10] [1 – 100]          │
├─ Range ──────────────────────────────────────┤
│ Min [1]   Max [100]   How many [1]           │
├──────────────────────────────────────────────┤
│ [              Generate              ]       │
│ ☐ Repeat automatically  no faster than [2.0] │
├─ Source ─────────────────────────────────────┤
│ ◉ Fast — DRBG seeded by the detector         │
│ ○ True entropy — detector rate only (slow)   │
├─ History ────────────────────────────────────┤
│ 79 · Heads · 2 · 10 · ...                    │
│ [Copy last] [Copy all]              [Clear]  │
└──────────────────────────────────────────────┘
```

Quick picks for **coin flip**, **D6**, **1–10** and **1–100**, plus arbitrary
min/max and a "how many" box for rolling several at once. Every draw goes
through the same unbiased rejection sampling the CLI uses, so a d6 really is a
d6.

Two controls worth explaining:

- **"no faster than N/sec"** caps the draw rate. It applies to auto-repeat and
  to rapid clicking alike, and when a click is throttled the button says
  `Rate limited — 0.4s` and then fires, rather than silently doing nothing.
- **Source** picks fast DRBG output or true detector-rate entropy. In physical
  mode the progress bar becomes a pool gauge and a draw genuinely takes
  seconds — 16 s was typical on this hardware — so the button tells you it is
  waiting for photons.

It is Tkinter, which ships with CPython on Windows and macOS. On Linux:

```bash
sudo apt install python3-tk     # or: sudo dnf install python3-tkinter
```

The window opens immediately and shows start-up progress rather than freezing,
because the SP 800-90B start-up test takes a minute or more. If another
`radiarandom` process is holding the detector it says so in words — only one
process can have it at a time.

---

## Command reference

| Command | Purpose |
|---|---|
| `gui` | graphical front end: presets, min/max, rate limit, history |
| `info` | device identity, measured rate, entropy budget, platform capability |
| `bench` | measure the achievable entropy rate |
| `selftest` | run the health tests and exercise both output modes |
| `gen` | random bytes → stdout or a file (`hex`, `bin`, `base64`, `dec`, `bits`, `c`) |
| `int` / `float` / `uuid` / `password` | typed output, unbiased by rejection sampling |
| `raw` | capture the unprocessed noise source for offline analysis |
| `feed` | contribute credited entropy to the Linux kernel pool |
| `serve` | serve entropy over a named pipe, FIFO, or loopback TCP |
| `seed-file` | write a mode-0600 full-entropy seed file |

Every command accepts `--serial`, `--poll-interval`, `--safety`,
`--startup-samples`, `--json`, and `-q`.

---

## Library use

```python
from radiarandom import open_generator

with open_generator() as gen:
    gen.wait_for_startup()
    seed = gen.physical_block()       # 32 bytes, full entropy, ~56 s
    bulk = gen.read(1024 * 1024)      # DRBG output, immediate
    print(gen.stats())
```

---

## Limitations

- **No mode is information-theoretically secure.** Physical mode's "full
  entropy" is the SP 800-90C construction, which assumes HMAC-SHA-512 is a good
  conditioning function; DRBG mode adds a PRF assumption. See
  [DESIGN.md §4](DESIGN.md) for what an unconditional guarantee would take.
- **Slow.** ~0.7 bytes/s of true entropy on background. By design and by physics.
- **Timing entropy is left on the table.** The 2 Hz device refresh caps arrival
  resolution at 500 ms and none of it is banked.
- **The i.i.d. assumption is not verified online.** Detector dead time and
  afterpulsing introduce short-range correlation; the 0.9 safety factor is a
  blunt cover, and the SP 800-90B predictors are the real check, offline.
- **Single detector.** No cross-check against an independent source.
- **The bundled estimators are a cross-check, not a certification.** Use NIST's
  reference implementation for anything citable.
- **Not audited.** This is a carefully built tool, not a certified one. Do not
  put it on the critical path of something that matters without reviewing it
  yourself.

---

## Development

```bash
pip install -e ".[usb,dev]"
python -m pytest tests/ -q
```

The test suite runs without hardware — `tests/conftest.py` provides a simulated
detector that can also be made to stall, collapse, or reset on demand. Several
of the tests exist because they caught real bugs; those are commented as such.

## Licence

MIT.
