// ecm128.h -- small, fast ECM for cofactors below 2^125 (host side).
//
// Montgomery curves with Suyama's parametrisation, x-only arithmetic in
// Montgomery representation modulo n with two 64-bit limbs (R = 2^128).
// Stage 1 multiplies by every prime power <= B1 with a Montgomery ladder;
// stage 2 is a baby-step giant-step continuation over primes in (B1, B2].
// This is ~20x cheaper per curve than calling GMP-ECM on 2-limb numbers,
// which matters because the host has to keep up with the GPU siever.
#pragma once
#include <gmpxx.h>
#include <stdint.h>

// Try up to 'ncurves' curves with B1 starting at B1base and ramping up.
// Returns true and sets f to a proper factor of n (n odd, composite, 2^20 < n < 2^125).
bool ecm128_factor(const mpz_class& n, double B1base, int ncurves, mpz_class& f);
