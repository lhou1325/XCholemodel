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

## Correlation-only flux closure

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
Gaussian, quartic, sextic, and compact C-infinity closures are included. The
quartic closure is the minimum-order flux closure that preserves the parent
short-range expansion through `v^4`.

## LDA-regressive algebraic-tail coupled extension

The correlation-only model does not control the sign or asymptotic cancellation
of the sum with a separately constructed GGA exchange hole. The additional
coupled construction in `LDA_REGRESSIVE_ALGEBRAIC_TAIL.md` imposes those
properties while retaining the exact PW92/LDA correlation profile at `t=0`.

Write the exchange hole as `n_x=-w_x`, with a nonnegative exchange weight, and
let `c_L` be the physical PW92/LDA correlation hole. A smooth exchange
majorant is constructed so that `w_x>=c_L` wherever `c_L` is positive while the
exchange particle sum and PBE exchange energy remain exact. Define

```text
g(u) = w_x(u) - c_L(u) >= 0
n_xc(u) = -g(u) F_m(u)
n_c(u)  =  w_x(u) - g(u) F_m(u).
```

For `x=k_F u`,

```text
tau(t) = t/sqrt(1+t^2)
F_m = T_m(tau*x) exp(theta_in*f_in(x)+theta_out*f_out(x))
f_in  = x^2/(0.5^2+x^2)
f_out = x^2/(4^2+x^2).
```

The algebraic gates are `T_m(z)=1/(1+z^m)` for `m=2,4,6`. The two coefficients
are fixed by the correlation particle sum and the analytic PBE correlation
energy. Since `g>=0` and `F_m>0`, the combined hole is nonpositive without
clipping.

At `t=0`, `tau=0` and both fitted coefficients are exactly zero, so

```text
n_c(u;t=0) = c_L(u)
```

pointwise. The gate and basis functions have zero first radial derivative at
the origin. Consequently the PW92/PBE correlation on-top value and cusp are
unchanged for every `t`.

For the selected quartic gate, an exchange tail `n_x~-C_x/u^4` gives

```text
n_c  ~  C_x/u^4 - D/u^8 > 0
n_xc ~             -D/u^8 < 0.
```

The positive correlation tail cancels the leading exchange `u^-4` term while a
smaller negative coupled tail remains. Quadratic and sextic gates give `u^-6`
and `u^-10` coupled tails, respectively.

## Compatibility boundary

Exact pointwise LDA regression, exact exchange moments, and `n_xc<=0` cannot be
imposed simultaneously for every possible low-density local descriptor. A
necessary condition is

```text
4*pi*integral u^2 max(c_L,0) du <= 1
2*pi*integral u   max(c_L,0) du <= -epsilon_x^PBE.
```

When either positive-lobe budget is exceeded, no nonnegative exchange weight
with the required exchange moments can majorize the fixed LDA correlation
profile. The reference implementation raises a dedicated incompatibility error
instead of silently clipping a hole or changing an energy. The audited atomic
and molecular descriptor domain is inside the compatible region; the boundary
appears only in extremely low-density local states.

## Scaled-LDA diagnostic

The particle-sum-preserving scaled form

```text
c_L^lambda(u) = lambda^3 c_L(lambda*u)
```

has correlation energy proportional to `lambda`, but its on-top value and cusp
scale as `lambda^3` and `lambda^4`. It is therefore useful as a shape diagnostic
or auxiliary ingredient, but cannot by itself recover a non-LDA PBE energy and
preserve both short-range anchors.

## Constraints retained

The complete research construction provides:

1. zero correlation particle moment;
2. exact analytic PBE correlation-energy moment;
3. exact pointwise PW92/LDA correlation-hole limit at `t=0` in the compatible domain;
4. unchanged PW92/PBE correlation on-top value and cusp;
5. pointwise nonpositive coupled hole by construction;
6. a positive correlation tail that cancels the exchange `u^-4` term;
7. spin-reversal invariance;
8. C-infinity separation dependence and no hard radial cutoff;
9. explicit failure rather than silent constraint violation outside the compatibility domain.

Exact one-electron self-correlation cancellation is deliberately not claimed.
It is incompatible with exact recovery of the parent PBE correlation energy
because PBE itself is not one-electron self-correlation free.

## Numerical audit

The original correlation-only benchmark contains 100 environments:

```text
r_s  = 0.5, 1, 2, 5, 10
zeta = 0, 0.3, 0.7, 0.9
t    = 0.2, 0.5, 1, 2, 4.
```

For those smooth flux closures, the maximum absolute PBE correlation-energy
error is `6.041e-8 Ha/electron`, and the maximum numerical particle residual is
`2.094e-6`.

The algebraic-tail extension was audited on 48 train/validation states spanning
`r_s=0.25` to `8`, `|zeta|<=0.95`, and reduced gradients through `4`. Across
all 144 quadratic, quartic, and sextic profiles:

- maximum sampled positive combined hole: `0.0`;
- maximum pointwise `t=0` LDA-profile error: `1.110e-16`;
- selected quartic maximum correlation-particle residual: `1.166e-14`;
- selected quartic maximum PBE correlation-energy error: `1.582e-15 Ha/electron`;
- selected quartic fitted coupled-tail power: `7.975`, consistent with `u^-8`;
- all ten constraint and regression test groups pass.

A controlled helium-like analytic density is included only as a reproducible
system-average contraction test, not as a correlated reference pair density.

## Files

- `gga_c_hole_models.py`: compact correlation-only companion implementation;
- `test_gga_c_hole_models.py`: correlation-only constraint tests;
- `STUDY.md`: original derivation and benchmark summary;
- `COUPLED_XC_EXTENSION.md`: first coupled-screen design;
- `LDA_REGRESSIVE_ALGEBRAIC_TAIL.md`: exact-LDA, anchored, algebraic-tail construction and compatibility theorem.

The complete reproducibility bundle is delivered separately. It contains the
full reference implementation, XCholemodel grid adapter, train/validation CSV
data, vector figures, figure-generation script, ten regression-test groups,
LaTeX source, and compiled manuscript.

## Test

From the repository root:

```bash
PYTHONPATH=research/gga_c_hole \
python -m unittest research/gga_c_hole/test_gga_c_hole_models.py -v
```

The companion modules require NumPy and SciPy, already used by XCholemodel.
