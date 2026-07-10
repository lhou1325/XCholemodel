# Smooth moment-conserving GGA correlation holes

This research companion audits the PBE/GGA correlation-hole construction in
`holemodel.py` and provides smooth replacements for its sharp real-space cutoff.
The production script is not changed by this branch.

## Problem in the conventional model

The current implementation forms the dimensionless local kernel

```text
h_c(v) = A_c(r_s,zeta;v) + t^2 B_c(r_s,zeta;v)
```

and multiplies the complete kernel by a step function at the last zero of the
cumulative particle moment. This makes

```text
integral_0^infinity v^2 h_c(v) dv = 0
```

when a crossing is found, but it does not impose the independent energy moment

```text
2 pi phi^3 integral_0^infinity v h_c(v) dv = epsilon_c^PBE.
```

The hard step also produces a finite-order discontinuity at the local cutoff.
On a finite radial grid, a missing crossing is represented by `v_c = 0`, so the
local PBE correlation hole can collapse to zero.

## Flux-closure construction

Keep the PW92/LSDA kernel `A_c` unchanged. Let

```text
b(v) = t^2 B_c(v)
F(v) = integral_0^v s^2 b(s) ds
```

and choose a smooth closure `D(x)` satisfying `D(0)=1` and `D(infinity)=0`.
Define

```text
Delta_D(v;L) = (1/v^2) d/dv [D(v/L) F(v)]
h_c^D(v) = A_c(v) + Delta_D(v;L).
```

The particle sum is then a boundary term:

```text
integral v^2 Delta_D(v;L) dv = [D(v/L) F(v)]_0^infinity = 0.
```

The remaining scale `L > 0` is obtained from the scalar equation

```text
integral_0^infinity D(v/L) F(v)/v^2 dv
    = H_PBE(r_s,zeta,t)/(2 pi phi^3),
```

which makes the local hole recover the analytical PBE correlation energy.

Implemented closures:

- Gaussian: `D(x)=exp(-x^2)`
- quartic super-Gaussian: `D(x)=exp(-x^4)`
- sextic super-Gaussian: `D(x)=exp(-x^6)`
- compact C-infinity bump: `D(x)=exp[-x^4/(1-x^4)]` for `x<1`, zero otherwise

All four are C-infinity. The quartic closure is selected as the default because
it is the minimum-order closure that leaves both leading nonzero short-range
coefficients of `t^2 B_c` unchanged. Among the closures with that property, it
has the smallest median compensating negative lobe in the benchmark.

## Constraints retained

The smooth construction has the following properties:

1. exact zero particle moment, analytically;
2. exact PBE correlation-energy moment, up to scalar root/quadrature tolerance;
3. exact LSDA limit for `t=0`;
4. unchanged PW92/LSDA on-top value and cusp;
5. spin-reversal invariance, because the input dependence is even in `zeta`;
6. C-infinity separation dependence and no hard cutoff;
7. preservation of the parent `B_c` short-range expansion through order `v^4`
   for the quartic closure.

Exact one-electron self-correlation cancellation is deliberately not claimed.
It is incompatible with exact recovery of the PBE correlation energy because
PBE itself is not one-electron self-correlation free.

## Numerical audit

The local benchmark contains 100 environments:

```text
r_s  = 0.5, 1, 2, 5, 10
zeta = 0, 0.3, 0.7, 0.9
 t   = 0.2, 0.5, 1, 2, 4
```

Results for the conventional sharp cutoff:

- median absolute energy-moment error: 7.500 mHa/electron;
- maximum absolute energy-moment error: 22.986 mHa/electron.

For the smooth closures, over the same grid:

- maximum absolute energy error: 6.041e-8 Ha/electron;
- maximum numerical particle-moment residual: 2.094e-6.

A controlled atom-like test uses spherical Slater-screened hydrogenic densities
for He, Li, Be, N, and Ne. These are analytic validation densities, not CCSD
reference densities. The sharp-cutoff energy errors span -2.862 to +6.598
mHa/electron; the smooth construction recovers the PBE target by definition.

## Files

- `gga_c_hole_models.py`: compact companion implementation;
- `test_gga_c_hole_models.py`: constraint and regression tests;
- `STUDY.md`: derivation, selection logic, and benchmark summary.

The complete reproducibility bundle additionally contains the full parameter
sweep, CSV tables, vector figures, figure-generation script, and the compiled
LaTeX manuscript.

## Test

From the repository root:

```bash
PYTHONPATH=research/gga_c_hole \
python -m unittest research/gga_c_hole/test_gga_c_hole_models.py -v
```

The companion module requires NumPy and SciPy, already used by XCholemodel.
