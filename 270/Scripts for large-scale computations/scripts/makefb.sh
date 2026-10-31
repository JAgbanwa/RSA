#!/usr/bin/env bash
# Algebraic factor base for RSA-270 with CADO-NFS makefb (CPU, ~minutes for lim 2^31).
#   CADO=/path/to/cado-nfs scripts/makefb.sh
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
. "$here/params/rsa270.env"
: "${CADO:?set CADO to your cado-nfs checkout (built with make)}"
mf=$(find "$CADO/build" -type f -name makefb -perm -u+x | head -1)
[ -n "$mf" ] || { echo "makefb not found under $CADO/build (run make there)" >&2; exit 1; }
cd "$here/params"
"$mf" -poly "$POLY" -lim "$LIM1" -maxbits "$LOGREG" -side 1 -t "$(nproc)" -out "$FB1"
echo "factor base: $here/params/$FB1"
