#!/usr/bin/env bash
# Polynomial selection for RSA-270 with CADO-NFS (CPU). Writes params/rsa270.poly.
#
#   CADO=/path/to/cado-nfs scripts/polyselect.sh [extra cado-nfs.py options]
#
# This is a large CPU job (RSA-260's selection took hundreds of GPU-days in its
# GPU form). Run it on as many cores as you can, e.g. with CADO's
# client/server mode across machines (see CADO's README, "--server").
# Generated data goes to $WORK_ROOT (default ~/rsa270-work): outside the git
# checkout, and in a path without spaces, which CADO-NFS work directories need.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
: "${CADO:?set CADO to your cado-nfs checkout (built with make)}"
work=${WORK:-${WORK_ROOT:-$HOME/rsa270-work}/polyselect}
mkdir -p "$work"
# tasks.sieve.run=false stops cado-nfs.py once a polynomial has been chosen
"$CADO/cado-nfs.py" "$here/params/rsa270.cado-params" \
    tasks.workdir="$work" tasks.sieve.run=false "$@" || true
poly=$(ls -t "$work"/*.poly 2>/dev/null | head -1 || true)
[ -n "$poly" ] || { echo "no polynomial produced in $work" >&2; exit 1; }
cp "$poly" "$here/params/rsa270.poly"
echo "polynomial: $here/params/rsa270.poly"
