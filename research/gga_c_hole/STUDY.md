# Derivation and benchmark

## 1. Audit of the conventional correlation hole

The PBE correlation path in `holemodel.py` evaluates the PW92/LSDA kernel
`A_c`, adds the second-order gradient correction `t^2 B_c`, integrates
`v^2(A_c+t^2 B_c)` cumulatively, locates the last sign crossing, and then uses

```text
h_c^RSC(v) = [A_c(v)+t^2 B_c(v)] Theta(v_c-v).
```

This real-space-cutoff construction has two independent limitations.

First, the cutoff is fixed only by the particle sum rule.  The correlation
energy is a different radial moment and is therefore not guaranteed to equal
the analytical PBE energy.  Second, the Heaviside factor is not differentiable
at `v_c`.  The branch also exposes a finite-grid failure mode: a local
environment without a detected crossing receives `v_c=0` and hence a zero GGA
correlation hole.

The relevant dimensionless moments are

```text
N_c = integral_0^infinity v^2 h_c(v) dv,
E_c = 2 pi phi^3 integral_0^infinity v h_c(v) dv.
```

A physically usable replacement must control both moments, preserve the PW92
short-range structure, remove the hard boundary, and reduce to LSDA at `t=0`.

## 2. Moment-conserving flux closure

Write

```text
h_c(v) = A_c(v) + Delta_c(v)
```

and leave `A_c` unchanged.  For the uncut PBE gradient correction

```text
b(v) = t^2 B_c(v),
F(v) = integral_0^v s^2 b(s) ds.
```

For a positive scale `L` and a dimensionless closure `D`, define

```text
Delta_D(v;L) = (1/v^2) d/dv [D(v/L) F(v)].
```

Because `F(0)=0`, `D(0)=1`, and `D(infinity)=0`,

```text
integral_0^infinity v^2 Delta_D(v;L) dv
  = [D(v/L)F(v)]_0^infinity
  = 0.
```

Thus the correction has exactly zero particle number for every positive `L`.
The correlation-energy contribution follows by integration by parts:

```text
integral_0^infinity v Delta_D(v;L) dv
  = integral_0^infinity D(v/L)F(v)/v^2 dv.
```

The analytical PBE increment is

```text
H = gamma phi^3 ln{1 + (beta/gamma)t^2
    (1+A t^2)/(1+A t^2+A^2 t^4)},

A = (beta/gamma)/[exp(-epsilon_c^LSDA/(gamma phi^3))-1].
```

The scale is the positive root of

```text
integral_0^infinity D(v/L)F(v)/v^2 dv = H/(2 pi phi^3).
```

The closure therefore satisfies both local moments without altering the LSDA
kernel.

## 3. Candidate closures

Four physically interpretable C-infinity closures were tested.

| model | closure D(x) | short-range expansion |
|---|---|---|
| Gaussian | exp(-x^2) | 1-x^2+O(x^4) |
| quartic | exp(-x^4) | 1-x^4+O(x^8) |
| sextic | exp(-x^6) | 1-x^6+O(x^12) |
| compact | exp[-x^4/(1-x^4)] for x<1; 0 otherwise | 1-x^4+O(x^8) |

The parent PBE correction starts as `b(v)=b_2 v^2+b_4 v^4+...`, so
`F(v)=O(v^5)`.  Gaussian damping modifies the closed correction at order
`v^4`.  Quartic, sextic, and compact closures preserve both the `v^2` and
`v^4` coefficients.  Quartic is the minimum-order noncompact choice with that
property.

## 4. Model-selection rule

The selection is not based on a free fit to atomic energies.  Every smooth
closure is forced to the same exact PBE energy and particle moments.  The
remaining comparison measures how much the closure distorts the parent
short-range GEA kernel and how large a compensating negative lobe is required.

Aggregate results over 100 local environments:

| closure | median short-range distortion | median negative compensation ratio | maximum energy error (Ha/e) |
|---|---:|---:|---:|
| Gaussian | 1.2721e-2 | 0.034882 | 6.0405e-8 |
| quartic | 6.7034e-3 | 0.060219 | 6.0407e-8 |
| sextic | 7.3910e-4 | 0.103390 | 6.0407e-8 |
| compact | 6.0069e-3 | 0.686110 | 6.0407e-8 |

Gaussian gives the smallest compensation but changes a leading short-range
coefficient.  Sextic gives the smallest local distortion but requires a much
deeper compensating lobe.  The compact bump is excessively stiff.  Quartic is
therefore selected as the minimum-intervention closure: it is the lowest-order
model preserving the first two nonzero parent coefficients, and among the
models with that property it has the smallest median compensation ratio.

## 5. Local benchmark

The benchmark grid is the Cartesian product

```text
r_s  in {0.5, 1, 2, 5, 10}
zeta in {0, 0.3, 0.7, 0.9}
t    in {0.2, 0.5, 1, 2, 4}.
```

The spin factor is calculated as

```text
phi(zeta) = [(1+zeta)^(2/3)+(1-zeta)^(2/3)]/2.
```

Across these 100 environments, the conventional sharp cutoff has a median
absolute correlation-energy error of 7.500 mHa/electron and a maximum absolute
error of 22.986 mHa/electron.  The smooth models reduce the maximum numerical
energy error to 6.041e-8 Ha/electron.  The maximum finite-grid particle
residual is 2.094e-6; analytically it is zero by construction.

Representative conventional-cutoff errors are:

| case | r_s | zeta | phi | t | PBE target (Ha/e) | sharp cutoff (Ha/e) | error (mHa/e) |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense unpolarized | 0.5 | 0.0 | 1.0000 | 0.5 | -0.0803720 | -0.0757410 | +4.631 |
| valence unpolarized | 2.0 | 0.0 | 1.0000 | 1.0 | -0.0447595 | -0.0508433 | -6.084 |
| diffuse unpolarized | 5.0 | 0.0 | 1.0000 | 2.0 | -0.0159549 | -0.0260588 | -10.104 |
| moderate spin | 2.0 | 0.5 | 0.9710 | 1.0 | -0.0403227 | -0.0453097 | -4.987 |
| strong spin | 2.0 | 0.9 | 0.8747 | 1.0 | -0.0352589 | -0.0338692 | +1.390 |

## 6. Controlled atomic-density comparison

Spherical atom-like references were generated from normalized Slater-screened
hydrogenic shells for He, Li, Be, N, and Ne.  They provide conventional radial
core/valence/tail variation and spin polarization while remaining completely
reproducible.  They are not correlated wave-function reference holes.

| atom | quartic/PBE target (Ha/e) | conventional sharp cutoff (Ha/e) | error (mHa/e) |
|---|---:|---:|---:|
| He | -0.0401421 | -0.0430043 | -2.862 |
| Li | -0.0482670 | -0.0495371 | -1.270 |
| Be | -0.0515313 | -0.0519564 | -0.425 |
| N  | -0.0565533 | -0.0524704 | +4.083 |
| Ne | -0.0624743 | -0.0558762 | +6.598 |

No local environment in these five analytic atoms lacked a conventional
particle-moment crossing.  The reported error is therefore the intrinsic
mismatch between the particle-selected cutoff and the PBE energy moment, not a
missing-crossing artifact.

## 7. Exact constraints and limitations

Satisfied:

- zero correlation-hole particle sum;
- exact analytical PBE correlation energy;
- LSDA limit at `t=0`;
- spin reversal invariance;
- unchanged PW92/LSDA on-top value and cusp;
- C-infinity smoothness in separation;
- quartic preservation of the parent short-range expansion through `v^4`.

Not claimed:

- positivity or monotonicity of every local correlation-hole segment;
- exact one-electron self-correlation cancellation;
- validation against system- and angle-resolved correlated holes.

The one-electron condition cannot be imposed simultaneously with exact PBE
energy recovery without changing the target functional, because PBE itself
has nonzero one-electron correlation.  The next validation stage should use
CCSD or explicitly correlated pair densities for atoms and small molecules and
should test the density dependence of the implicitly determined scale `L`.
