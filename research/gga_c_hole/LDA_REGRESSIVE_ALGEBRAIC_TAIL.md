# LDA-regressive, cusp-preserving algebraic-tail coupled hole

This note supersedes the first positive-screen construction when the correlation model must recover an LDA correlation-hole profile at `t=0`, preserve the PBE on-top value and cusp, and cancel a `u^-4` exchange tail without making the combined model hole positive.

## Local moment convention

For an angle-averaged local hole,

```text
N_alpha       = 4*pi*integral_0^infinity u^2 n_alpha(u) du
Epsilon_alpha = 2*pi*integral_0^infinity u   n_alpha(u) du.
```

The targets are

```text
N_x=-1, N_c=0, N_xc=-1,
Epsilon_x=epsilon_x^PBE,
Epsilon_c=epsilon_c^PBE.
```

The `2*pi` energy prefactor contains the pair-counting factor. The `4*pi` prefactor belongs to the particle moment.

Pointwise `n_xc<=0` is imposed as a model-admissibility condition. It is not asserted to be a universal theorem for every exact spatially resolved hole.

## 1. Constraint-exact LDA reference

The rounded constants in the conventional PW92/LDA correlation-hole representation leave small numerical particle- and energy-moment drifts. Define

```text
h_c^L(v) = h_c^PW92(v)
         + alpha v^2 exp(-a v^2)
         + beta  v^2 exp(-b v^2),
a=p/2, b=2p.
```

`alpha` and `beta` are obtained from the two linear equations

```text
integral v^2 Delta h_c^L(v) dv = - integral v^2 h_c^PW92(v) dv,
2*pi*phi^3 integral v Delta h_c^L(v) dv
    = epsilon_c^PW92 - 2*pi*phi^3 integral v h_c^PW92(v) dv.
```

The correction starts at `O(v^2)` and is exponentially localized. It therefore leaves the correlation on-top value, one-sided electron-electron cusp, and positive `v^-4` tail unchanged. The finite-gradient model has an explicit `t=0` branch that returns this LDA reference point by point.

## 2. Why exchange must be repaired jointly

Let the parent exchange hole be `n_x^parent=-w_parent`. The fixed parent LDA exchange profile is not pointwise large enough to dominate the positive part of the fixed LDA correlation profile for every separation. Consequently, exact LDA correlation regression and pointwise `n_xc<=0` cannot both be obtained by changing correlation alone.

A smooth positive LDA-correlation majorant is used:

```text
c_+(u) = 0.5*[c_L(u) + sqrt(c_L(u)^2 + (eta*w_L(u))^2)],
eta = 1.0e-3.
```

It is nonnegative and is never smaller than `max(c_L,0)`. The LDA exchange weight `w_L` supplies the correct algebraic radial scale, so the smoothing does not introduce a constant tail.

The repaired exchange weight is

```text
w_x(u) = c_+(u)
       + A_x w_parent(u) exp[-(k_F*u/L_x)^4].
```

The positive parameters `A_x` and `L_x` are solved from

```text
4*pi*integral u^2 w_x(u) du = 1,
2*pi*integral u   w_x(u) du = -epsilon_x^PBE.
```

Thus `n_x=-w_x` remains nonpositive and has the exact exchange particle and energy moments. The exchange modification is a material part of the requested strict-sign model and is reported explicitly rather than hidden as a numerical patch.

## 3. Exact LDA screen

Define

```text
Q_L(u) = 1 - c_L(u)/w_x(u).
```

The majorant guarantees `Q_L(u)>0`, and

```text
c_L(u) = w_x(u)[1-Q_L(u)].
```

This identity is the LDA-regression anchor.

## 4. Finite-gradient algebraic gate

For `x=k_F*u`, define

```text
tau(t) = t/sqrt(1+t^2),
G_m(x,t) = 1/[1+(tau*x)^m],  m=2,4,6.
```

The moment-transport basis functions are

```text
b_s(x) = (e/4)  x^2 exp[-(x/2)^2],
b_m(x) = (e/36) x^2 exp[-(x/6)^2].
```

Both are bounded, vanish with zero first derivative at the origin, and vanish at infinity. The positive screen is

```text
Q(u) = Q_L(u) G_m(k_F*u,t)
       [1 + theta_s b_s(k_F*u) + theta_m b_m(k_F*u)].
```

The bracket is required to remain strictly positive. A state is rejected if the positivity certificate fails; no clipping is used.

The final holes are

```text
n_x(u)  = -w_x(u),
n_c(u)  =  w_x(u)[1-Q(u)],
n_xc(u) = -w_x(u)Q(u).
```

Therefore `n_x+n_c=n_xc` pointwise and `n_xc<0` wherever `w_x Q` is nonzero.

## 5. Moment equations

The two coefficients are obtained from the linear weighted-moment equations

```text
4*pi*integral u^2 w_x(u) Q(u) du = 1,
2*pi*integral u   w_x(u) Q(u) du
    = -(epsilon_x^PBE+epsilon_c^PBE).
```

With the exchange moments already fixed, these equations imply

```text
N_c=0,
Epsilon_c=epsilon_c^PBE.
```

## 6. LDA limit, on-top value, and cusp

At `t=0`, the implementation returns `n_c=c_L` directly. Hence LDA regression is exact point by point rather than only in its moments.

For nonzero `t`, `G_m(0,t)=1` and `G_m'(0,t)=0`; the two basis functions start at `O(u^2)`. The constant and linear radial terms of `w_x(1-Q)` are therefore identical to those of `c_L`. The PW92/PBE correlation on-top value and one-sided cusp are unchanged.

A nonzero cusp is incompatible with a globally `C-infinity` even extension through `u=0`. The physically compatible regularity is the exact one-sided cusp at the origin and `C-infinity` dependence for every `u>0`, without a finite-radius cutoff.

## 7. Tail cancellation

For the selected quartic gate, suppose

```text
n_x(u) = -C_x/u^4 + o(u^-4).
```

The LDA reference screen tends to a finite positive algebraic scale, while

```text
G_4(k_F*u,t) ~ const/u^4.
```

Consequently,

```text
n_c(u)  = +C_x/u^4 - D/u^8 + o(u^-8) > 0,
n_xc(u) =             -D/u^8 + o(u^-8) < 0.
```

The correlation tail cancels the leading exchange `u^-4` term while leaving a smaller negative combined tail. The quadratic and sextic gates produce `u^-6` and `u^-10` combined tails. The quartic gate is selected because it gives the requested `u^-8` residual and does not alter the gate itself through quadratic order at the origin.

## 8. Scaled-LDA diagnostic

The particle-preserving scaled form

```text
c_lambda(u)=lambda^3 c_L(lambda*u)
```

has

```text
Epsilon[c_lambda] = lambda Epsilon[c_L],
c_lambda(0)       = lambda^3 c_L(0),
c_lambda'(0)      = lambda^4 c_L'(0).
```

Choosing `lambda=epsilon_c^PBE/epsilon_c^LDA` matches the PBE correlation energy but violates the on-top and cusp constraints unless `lambda=1`. Scaled LDA is therefore retained as a diagnostic ingredient, not as the complete model.

## 9. Current validation artifact

The separate reproducibility package contains the complete reference implementation, XCholemodel grid adapter, generated data, ten vector figures, nine constraint-test groups, LaTeX source, and a 14-page compiled manuscript.

The compact descriptor audit contains eight finite-gradient states over three distinct `(r_s,zeta,phi)` environments. For the selected quartic model:

- maximum correlation-particle residual: `5.551115e-16`;
- maximum PBE correlation-energy residual in the construction equations: `6.938894e-18 Ha/electron`;
- maximum independently represented PBE exchange-energy residual: `1.004363e-12 Ha/electron`;
- maximum sampled positive combined hole: `0`;
- representative tail-ratio error `|n_c/(-n_x)-1|`: `7.072121e-14`.

Independent segmented infinite-range quadrature for `(r_s,zeta,s,t)=(2,0.4,1.2,1)` gives correlation-particle error `1.836313e-15` and exchange/correlation energy errors of approximately `1.0e-12 Ha/electron` with opposite signs. All nine regression-test groups pass.