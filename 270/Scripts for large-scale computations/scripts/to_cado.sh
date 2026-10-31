#!/usr/bin/env bash
# Hand gsieve relations to CADO-NFS for filtering, linear algebra and square root.
#
#   CADO=/path/to/cado-nfs scripts/to_cado.sh RELDIR [extra cado-nfs.py options]
#
# cado-nfs.py imports the polynomial and every relation file, then runs
# dup/purge/merge, block Wiedemann and sqrt. If filtering reports that more
# relations are needed, it will start its own CPU siever (las) for the rest;
# add tasks.sieve.run=false to stop instead and sieve more with gsieve.
#
# Scale warning: for RSA-270 the matrix will be several hundred million rows
# (RSA-260: 656M x 656M, 98B non-zeros). Linear algebra needs a large
# multi-node cluster or a GPU block Wiedemann; plan for it separately.
# Generated data goes to $WORK_ROOT (default ~/rsa270-work): outside the git
# checkout, and in a path without spaces, which CADO-NFS work directories need.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
: "${CADO:?set CADO to your cado-nfs checkout (built with make)}"
reldir=${1:?directory with q*.rels.gz}
shift
work=${WORK:-${WORK_ROOT:-$HOME/rsa270-work}/cado}
mkdir -p "$work"
list=$work/relation_files.txt
ls "$(cd "$reldir" && pwd)"/q*.rels.gz > "$list"          # cado-nfs.py takes "@listfile"
nrels=$(zcat "$reldir"/q*.rels.gz | grep -c -v '^#')
echo "importing $nrels relations from $(wc -l < "$list") files"
"$CADO/cado-nfs.py" "$here/params/rsa270.cado-params" tasks.workdir="$work" \
    tasks.polyselect.import="$here/params/rsa270.poly" \
    tasks.sieve.import="@$list" tasks.sieve.rels_wanted="$nrels" "$@"
