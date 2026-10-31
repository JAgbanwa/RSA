#!/usr/bin/env python3
"""Independent relation checker (pure Python, no gsieve code).

For every relation "a,b:p,..:p,.." it checks that b > 0, gcd(a,b) = 1, every
listed prime is prime and at most 2^lpb on its side, and that the product of
the listed primes equals |F(a,b)| exactly on both sides (full factorisation).

usage: check_rels.py POLY RELS [-lpb0 B0 -lpb1 B1] [--compare OTHER_RELS]
"""
import gzip
import math
import random
import sys


def read_poly(path):
    c, y, n = {}, {}, None
    for line in open(path):
        line = line.split("#")[0].strip()
        if ":" not in line:
            continue
        k, v = (t.strip() for t in line.split(":", 1))
        if not v:
            continue
        if k == "n":
            n = int(v)
        elif len(k) == 2 and k[0] == "c" and k[1].isdigit():
            c[int(k[1])] = int(v)
        elif len(k) == 2 and k[0] == "Y" and k[1].isdigit():
            y[int(k[1])] = int(v)
    f0 = [y.get(i, 0) for i in range(max(y) + 1)]
    f1 = [c.get(i, 0) for i in range(max(c) + 1)]
    return n, [f0, f1]


def homog(f, a, b):
    d = len(f) - 1
    return sum(f[k] * a**k * b**(d - k) for k in range(d + 1))


def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] if n < 3317044064679887385961981 else \
        [random.randrange(2, n - 1) for _ in range(30)]
    for a in bases:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def read_rels(path):
    op = gzip.open if path.endswith(".gz") else open
    rels = []
    with op(path, "rt") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            ab, s0, s1 = line.strip().split(":")
            a, b = (int(t) for t in ab.split(","))
            p0 = [int(t, 16) for t in s0.split(",") if t]
            p1 = [int(t, 16) for t in s1.split(",") if t]
            rels.append((a, b, p0, p1))
    return rels


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        sys.exit(2)
    poly, relfile = args[0], args[1]
    lpb = [64, 64]
    other = None
    i = 2
    while i < len(args):
        if args[i] == "-lpb0":
            lpb[0] = int(args[i + 1]); i += 2
        elif args[i] == "-lpb1":
            lpb[1] = int(args[i + 1]); i += 2
        elif args[i] == "--compare":
            other = args[i + 1]; i += 2
        else:
            sys.exit("unknown option " + args[i])
    n, f = read_poly(poly)
    rels = read_rels(relfile)
    bad = 0
    primecache = {}
    for (a, b, p0, p1) in rels:
        why = None
        if b <= 0 or math.gcd(a, b) != 1:
            why = "b<=0 or gcd(a,b)!=1"
        for side, ps in ((0, p0), (1, p1)):
            if why:
                break
            v = abs(homog(f[side], a, b))
            prod = 1
            for p in ps:
                if p not in primecache:
                    primecache[p] = is_prime(p)
                if not primecache[p]:
                    why = f"side {side}: {p} not prime"
                    break
                if p.bit_length() > lpb[side]:
                    why = f"side {side}: {p} exceeds 2^{lpb[side]}"
                    break
                prod *= p
            if not why and prod != v:
                why = f"side {side}: product of primes != |F(a,b)|"
        if why:
            bad += 1
            if bad <= 10:
                print(f"BAD {a},{b}: {why}")
    keys = {(a, b) for (a, b, _, _) in rels}
    print(f"{relfile}: {len(rels)} relations, {len(keys)} distinct (a,b), {bad} invalid")
    if other:
        okeys = {(a, b) for (a, b, _, _) in read_rels(other)}
        both = keys & okeys
        print(f"compare with {other}: {len(okeys)} distinct there; common {len(both)}, "
              f"only here {len(keys - okeys)}, only there {len(okeys - keys)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
