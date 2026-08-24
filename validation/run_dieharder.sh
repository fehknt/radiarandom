#!/usr/bin/env bash
# Run the Dieharder battery against radiarandom output.
#
#   https://github.com/seehuhn/dieharder
#
# Two things get tested, and they answer different questions:
#
#   --mode drbg      The generator's normal output: HMAC_DRBG(SHA-512) seeded
#                    and reseeded from the detector. Streams as fast as you
#                    like, so the *full* battery runs with no file rewinding.
#                    This validates the plumbing -- framing, byte order,
#                    conditioning, reseeding -- end to end. It does not, and
#                    cannot, validate the physics: a correctly implemented
#                    DRBG passes Dieharder whether or not its seed was any
#                    good.
#
#   --mode physical  Conditioned output emitted at the detector's true entropy
#                    rate, roughly 1.5 bytes/second. This is the stream whose
#                    statistics actually reflect radioactive decay. Collecting
#                    enough for a full battery is a matter of weeks, so this
#                    mode runs whatever subset the available data supports and
#                    says plainly which tests were underpowered.
#
#   --mode file      Run against a file you already have.
#
# Usage:
#   ./run_dieharder.sh --mode drbg [--size 8G] [--out results/]
#   ./run_dieharder.sh --mode physical --file data/soak.physical.bin
#   ./run_dieharder.sh --mode file --file some.bin [--tests all|quick]
#
# Dieharder needs a lot of data. With -g 200 (stdin) it consumes as much as it
# is given; with -g 201 (file) it rewinds when it runs out and prints
# "ks_test_p" warnings, which invalidate the run. Budget ~8 GiB for a clean
# full battery.

set -euo pipefail

MODE=""
SIZE="8G"
OUTDIR="results"
FILE=""
TESTS="all"
PSAMPLES=""
RADIARANDOM="${RADIARANDOM:-radiarandom}"

usage() { sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)      MODE="$2"; shift 2 ;;
    --size)      SIZE="$2"; shift 2 ;;
    --out)       OUTDIR="$2"; shift 2 ;;
    --file)      FILE="$2"; shift 2 ;;
    --tests)     TESTS="$2"; shift 2 ;;
    --psamples)  PSAMPLES="$2"; shift 2 ;;
    -h|--help)   usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 1 ;;
  esac
done

[[ -n "$MODE" ]] || { echo "--mode is required" >&2; usage 1; }

if ! command -v dieharder >/dev/null 2>&1; then
  cat >&2 <<'EOF'
dieharder is not installed.

  Debian/Ubuntu : sudo apt-get install dieharder
  Fedora        : sudo dnf install dieharder
  macOS         : brew install dieharder
  From source   : git clone https://github.com/seehuhn/dieharder
                  cd dieharder && ./autogen.sh && ./configure && make && sudo make install
                  (needs the GNU Scientific Library)

On Windows there is no native build; use WSL, a container, or a Linux host.
The Dockerfile next to this script builds a ready-to-run image:

  docker build -t radiarandom-dieharder validation/
  docker run --rm -v "$PWD:/work" radiarandom-dieharder \
      dieharder -a -g 201 -f /work/data/output.bin
EOF
  exit 127
fi

mkdir -p "$OUTDIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT="$OUTDIR/dieharder-$MODE-$STAMP.txt"
META="$OUTDIR/dieharder-$MODE-$STAMP.meta.txt"

{
  echo "# radiarandom dieharder run"
  echo "date_utc:  $(date -u +%FT%TZ)"
  echo "mode:      $MODE"
  echo "host:      $(uname -a)"
  echo "dieharder: $(dieharder -h 2>&1 | head -1 || true)"
  echo "version:   $($RADIARANDOM --version 2>/dev/null || echo 'n/a')"
} | tee "$META"

dh_args=(-g 201)
[[ "$TESTS" == "all" ]] && dh_args=(-a "${dh_args[@]}")
[[ -n "$PSAMPLES" ]] && dh_args+=(-p "$PSAMPLES")

case "$MODE" in
  drbg)
    echo "streaming DRBG output straight into dieharder (no file, no rewinding)"
    echo "this runs until the battery finishes; expect several hours."
    # -g 200 reads an unbounded raw binary stream from stdin, which is exactly
    # what the generator produces, so no test is ever starved of data.
    "$RADIARANDOM" gen --stream --format bin --output - --quiet \
      | dieharder -a -g 200 ${PSAMPLES:+-p "$PSAMPLES"} | tee "$RESULT"
    ;;

  physical)
    [[ -n "$FILE" ]] || { echo "--mode physical needs --file" >&2; exit 1; }
    [[ -f "$FILE" ]] || { echo "no such file: $FILE" >&2; exit 1; }
    BYTES=$(stat -c%s "$FILE" 2>/dev/null || stat -f%z "$FILE")
    echo "physical-mode sample: $BYTES bytes" | tee -a "$META"
    if (( BYTES < 1048576 )); then
      cat <<EOF | tee -a "$META"

WARNING: $BYTES bytes is far below what Dieharder needs for a full battery.
Dieharder will rewind the file and its p-values will not be trustworthy;
treat this run as a smoke test, not as validation. At ~1.5 bytes/s you need
about 8 days per 1 MB. Raising the count rate with a check source (a thoriated
lantern mantle, uranium glass, a smoke-detector americium source) is the
practical way to make a real physical-mode run feasible: entropy rate scales
linearly with the count rate.

For a rigorous assessment of this stream, run the SP 800-90B estimators
instead -- they are designed for exactly this sample size:
  python validation/sp800_90b.py data/soak.channels.u16
EOF
    fi
    dieharder -a -g 201 -f "$FILE" ${PSAMPLES:+-p "$PSAMPLES"} | tee "$RESULT"
    ;;

  file)
    [[ -n "$FILE" ]] || { echo "--mode file needs --file" >&2; exit 1; }
    dieharder "${dh_args[@]}" -f "$FILE" | tee "$RESULT"
    ;;

  *)
    echo "unknown mode: $MODE" >&2; exit 1 ;;
esac

echo
echo "=================== summary ==================="
awk '
  /PASSED|WEAK|FAILED/ {
    if ($0 ~ /PASSED/) pass++
    else if ($0 ~ /WEAK/) weak++
    else if ($0 ~ /FAILED/) fail++
  }
  END {
    printf "PASSED: %d\nWEAK:   %d\nFAILED: %d\n", pass, weak, fail
    if (fail > 0)
      print "\nA FAILED result is meaningful. Investigate before using this build."
    else if (weak > 0)
      print "\nWEAK results are expected: with ~114 tests at alpha=0.05 a handful\nof p-values in the tails is normal. Re-run to see whether they persist."
    else
      print "\nClean sweep."
  }
' "$RESULT" | tee -a "$META"

echo
echo "results: $RESULT"
echo "metadata: $META"
