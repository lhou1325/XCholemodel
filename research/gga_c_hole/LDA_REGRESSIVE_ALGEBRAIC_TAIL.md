# LDA-regressive algebraic-tail coupled hole

This note supersedes the unconstrained `n_c=w_x(1-Q)` construction when exact pointwise recovery of the PW92/LDA correlation hole at `t=0` is required.

## Required local moments

For an angle-averaged local hole,

```text
N_alpha = 4*pi*integral u^2 n_alpha(u) du
E_alpha = 2*pi*integral u   n_alpha(u) du
```

The target values are

```text
N_x=-1, N_c=0, N_xc=-1
E_x=epsilon_x^PBE, E_c=epsilon_c^PBE.
```

The model also preserves the PW92/PBE correlation on-top value and cusp, uses no radial step cutoff, and imposes `n_xc<=0` as a model-admissibility condition.

## Compatibility theorem

Write `n_x=-w_x` with `w_x>=0`. Exact pointwise LDA regression and `n_xc<=0` at `t=0` require

```text
w_x(u) >= n_c^LDA(u)
```

where the LDA correlation hole is positive. Therefore every possible exchange companion must satisfy

```text
P_plus = 4*pi*integral u^2 max(n_c^LDA,0) du <= 1
X_plus = 2*pi*integral u   max(n_c^LDA,0) du <= -epsilon_x^PBE.
```

If either inequality fails, exact PW92/LDA regression, exact exchange moments, and pointwise `n_xc<=0` cannot coexist, independent of parameterization. The reference implementation raises a dedicated incompatibility error in that low-density domain rather than silently changing an energy or clipping a profile.

## Sign-safe exchange companion

Inside the compatible domain, form a smooth positive-lobe majorant

```text
b_delta(u) = 0.5*[c_L(u) + sqrt(c_L(u)^2 + delta^2*w_L(u)^2*S(k_F*u)^2)]
S(x) = exp[-(x_a/x)^4] for x>0, S(0)=0.
```

The flat activation leaves the exchange on-top behavior unchanged. Add a scaled, short-range parent GGA exchange core

```text
r_0(u) = w_x^parent(u) exp[-(k_F*u/L_x)^4]
r(u)   = A_x r_0(B_x u)
w_x    = b_delta + r.
```

`A_x` and `B_x` are analytic functions of the residual particle and PBE exchange-energy budgets. The resulting `w_x` is nonnegative, majorizes the exact LDA correlation profile, and has exact exchange particle and energy moments.

## Exact-LDA coupled weight

Define

```text
g(u) = w_x(u) - n_c^LDA(u) >= 0.
```

For `x=k_F*u`, use

```text
tau(t) = t/sqrt(1+t^2)
f_in(x)  = x^2/(0.5^2+x^2)
f_out(x) = x^2/(4^2+x^2)
F_m(x;t) = T_m(tau*x) exp[theta_in*f_in + theta_out*f_out].
```

Algebraic gates are

```text
T_2(z)=1/(1+z^2)
T_4(z)=1/(1+z^4)
T_6(z)=1/(1+z^6).
```

The final profiles are

```text
n_x  = -w_x
n_xc = -g F_m
n_c  =  w_x - g F_m.
```

Because `g>=0` and `F_m>0`, `n_xc<=0` is exact without clipping.

## Exact LDA limit and short range

At `t=0`, set `theta_in=theta_out=0`; then `tau=0`, `T_m=1`, and

```text
n_c(u;0) = w_x(u) - g(u) = n_c^LDA(u)
```

pointwise. Both basis functions are `O(u^2)`, and every gate has `T_m(0)=1`, `T_m'(0)=0`. Since the exchange weight has zero radial derivative at the origin, the correlation on-top value and cusp are exactly the PW92/PBE values for every `t`.

## Moment equations

The two coefficients are determined from

```text
4*pi*integral u^2 g(u) F_m(u) du = 1
2*pi*integral u   g(u) F_m(u) du = -(epsilon_x^PBE+epsilon_c^PBE).
```

The implementation calibrates these equations to the exact ungated moments and quadratures only `g(F_m-1)`. This removes the finite-grid LDA-tail offset and makes both coefficients vanish continuously as `t->0`.

## Tail cancellation

For the selected quartic gate,

```text
w_x ~ C_x/u^4
g   ~ C_g/u^4
F_4 ~ const/u^4
```

and therefore

```text
n_x  ~ -C_x/u^4
n_c  ~  C_x/u^4 - D/u^8 > 0
n_xc ~             -D/u^8 < 0.
```

Thus the positive correlation tail cancels the leading exchange `u^-4` term while leaving a smaller negative coupled tail. The quadratic and sextic gates produce `u^-6` and `u^-10` coupled tails. The quartic gate is selected because it is the lowest even gate that does not alter the gate itself through `O(u^2)` and gives the requested `u^-8` residual.

## Scaled-LDA diagnostic

The particle-sum-preserving scaled LDA form

```text
n_c^lambda(u)=lambda^3 n_c^LDA(lambda*u)
```

has energy, on-top value, and cusp proportional to `lambda`, `lambda^3`, and `lambda^4`. Choosing `lambda=epsilon_c^PBE/epsilon_c^LDA` recovers the PBE correlation energy but violates both short-range anchors unless `lambda=1`. It can be used as an auxiliary shape ingredient, not as the complete constrained model.

## Validation artifact

The separate reproducibility package contains the full reference implementation, XCholemodel grid adapter, 48-state train/validation audit, three algebraic families, ten regression-test groups, figures, generated CSV data, and the compiled manuscript. The selected quartic model gives maximum correlation-particle residual `1.166e-14`, maximum PBE correlation-energy error `1.582e-15 Ha/e`, zero sampled positive combined hole, and maximum pointwise `t=0` LDA-profile error `1.110e-16` over the audited domain.
