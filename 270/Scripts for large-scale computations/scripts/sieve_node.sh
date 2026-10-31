#!/usr/bin/env bash
# Sieve a special-q range on every GPU of this machine. Work units of QCHUNK
# special-q go to GPUs round-robin; each writes relations/q<start>.rels.gz.
# Restartable: finished work units (with a "# Total" line) are skipped.
#
#   scripts/sieve_node.sh Q_START Q_END [OUTDIR]   (default OUTDIR: ~/rsa270-work/relations)
#
# Run one copy per machine with disjoint [Q_START, Q_END) ranges.
# Generated data goes to $WORK_ROOT (default ~/rsa270-work): outside the git
# checkout, and in a path without spaces, which CADO-NFS work directories need.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
. "$here/params/rsa270.env"
qs=${1:?Q_START}; qe=${2:?Q_END}
outdir=${3:-${WORK_ROOT:-$HOME/rsa270-work}/relations}
bin=${GSIEVE:-$here/gsieve}
mkdir -p "$outdir"
ngpu=$(nvidia-smi -L | wc -l)
ncpu=$(nproc)
tpg=$(( ncpu / ngpu > 1 ? ncpu / ngpu : 1 ))   # cofactorisation threads per GPU
echo "sieving [$qs, $qe) on $ngpu GPUs, $tpg CPU threads each -> $outdir"
cd "$here/params"
run_gpu() {
    local g=$1 k=0 q0 q1 f
    for (( q0 = qs; q0 < qe; q0 += QCHUNK, k++ )); do
        (( k % ngpu == g )) || continue
        q1=$(( q0 + QCHUNK < qe ? q0 + QCHUNK : qe ))
        f=$outdir/q$q0.rels.gz
        if [ -s "$f" ] && zcat "$f" 2>/dev/null | tail -1 | grep -q '^# Total'; then continue; fi
        CUDA_VISIBLE_DEVICES=$g "$bin" -backend gpu -poly "$POLY" -fb1 "$FB1" \
            -lim0 "$LIM0" -lim1 "$LIM1" -lpb0 "$LPB0" -lpb1 "$LPB1" -mfb0 "$MFB0" -mfb1 "$MFB1" \
            -A "$A" -sqside "$SQSIDE" -logreg "$LOGREG" -q0 "$q0" -q1 "$q1" -t "$tpg" \
            2>> "$outdir/gpu$g.log" | gzip -1 > "$f.tmp" && mv "$f.tmp" "$f"
    done
}
for (( g = 0; g < ngpu; g++ )); do run_gpu "$g" & done
wait
echo "done"
