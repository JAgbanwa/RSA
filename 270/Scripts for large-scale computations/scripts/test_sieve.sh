#!/usr/bin/env bash
# Test sieving: sample special-q across the planned range on one GPU, and
# estimate relations per special-q, GPU seconds per special-q, and the GPU
# time for the whole run.
#
#   scripts/test_sieve.sh [NQ_PER_POINT=8] [TARGET_RELATIONS=16e9]
#
# Compare settings by editing params/rsa270.env (or exporting overrides) and
# re-running; keep the set with the lowest estimated GPU-days.
# Generated data goes to $WORK_ROOT (default ~/rsa270-work): outside the git
# checkout, and in a path without spaces, which CADO-NFS work directories need.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
. "$here/params/rsa270.env"
nq=${1:-8}
target=${2:-16e9}
bin=${GSIEVE:-$here/gsieve}
out=${WORK_ROOT:-$HOME/rsa270-work}/test_sieve
mkdir -p "$out"
cd "$here/params"
summary=$out/summary.txt
: > "$summary"
# sample points spread over [QMIN, QMAX)
for frac in 0.0 0.1 0.25 0.5 0.75 0.95; do
    q0=$(python3 -c "print(int($QMIN + $frac * ($QMAX - $QMIN)))")
    f=$out/q$q0.rels
    "$bin" -poly "$POLY" -fb1 "$FB1" -lim0 "$LIM0" -lim1 "$LIM1" -lpb0 "$LPB0" -lpb1 "$LPB1" \
        -mfb0 "$MFB0" -mfb1 "$MFB1" -A "$A" -sqside "$SQSIDE" -logreg "$LOGREG" \
        -q0 "$q0" -q1 "$((q0 + 100000000))" -nq "$nq" -out "$f" -v 2> "$f.log"
    rels=$(grep -c -v '^#' "$f" || true)
    secq=$(sed -n 's/^# Total sieve time .*(\([0-9.]*\)s\/special-q)/\1/p' "$f")
    echo "$q0 $rels $nq $secq" | tee -a "$summary"
done
python3 - "$summary" "$QMIN" "$QMAX" "$target" <<'EOF'
import sys, math
rows = [tuple(map(float, l.split())) for l in open(sys.argv[1]) if l.strip()]
qmin, qmax, target = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
# special-q ideals in [qmin, qmax) ~ number of primes (about one root per prime on average)
nideals = qmax / math.log(qmax) - qmin / math.log(qmin)
rpq = sum(r[1] for r in rows) / sum(r[2] for r in rows)
spq = sum(r[3] for r in rows) / len(rows)
print(f"average {rpq:.2f} relations/special-q, {spq:.3f} GPU-s/special-q")
print(f"[{qmin:.3g}, {qmax:.3g}) holds ~{nideals:.3g} special-q -> ~{nideals * rpq:.3g} raw relations")
need = target / rpq
print(f"for {target:.3g} raw relations: ~{need:.3g} special-q, ~{need * spq / 86400:.0f} GPU-days")
EOF
