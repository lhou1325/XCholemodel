# Constraint-coupled GGA exchange-correlation-hole extension

## Reason for coupling exchange and correlation

An independently constructed GGA correlation hole can satisfy its own particle and energy moments while making the sum with the GGA exchange hole positive over a finite radial interval. Correlation-only moment conservation therefore does not enforce a pointwise condition on the combined hole.

The extension models the combined profile directly. Let the repaired exchange hole be

```text
n_x(u) = -w_x(u),  w_x(u) >= 0.
```

Choose a positive smooth screening factor `Q(u)` and define

```text
n_c(u)  =  w_x(u) [1 - Q(u)]
n_xc(u) = -w_x(u) Q(u).
```

This gives the pointwise identity `n_x + n_c = n_xc`. For every noncompact screen, `Q(u)>0` at finite separation, so `n_xc(u)<0` wherever the exchange profile is nonzero. Numerically the tested condition is `n_xc<=0` because extreme-tail underflow can produce signed zero.

Pointwise nonpositivity of the total modeled hole is imposed here as a model-admissibility condition. It is not claimed as a universal theorem for every exact pair density.

## Exchange moment repair

For the local XCholemodel/Ernzerhof-Perdew exchange profile `n_x^(0)(u)<=0`, define

```text
P_0 = -4 pi integral u^2 n_x^(0)(u) du
X_0 = -2 pi integral u   n_x^(0)(u) du
X_t = -epsilon_x^PBE.
```

Apply a positive amplitude and radial dilation,

```text
b = P_0 X_t / X_0
a = b^3 / P_0
n_x(u) = a n_x^(0)(b u).
```

The transformation preserves sign and topology while enforcing

```text
4 pi integral u^2 n_x(u) du = -1
2 pi integral u n_x(u) du = epsilon_x^PBE.
```

Across the 96-state audit, the amplitude lies in `[0.9863,1.0129]` and the dilation lies in `[0.9954,1.0043]`.

## Positive correlation tail and exchange-tail cancellation

Use

```text
Q(u) = A D(k_F u/L),
```

where `D(0)=1`, `D>0`, and `D->0`. For

```text
I_j(L) = integral u^j w_x(u) D(k_F u/L) du,
```

the correlation-particle sum fixes

```text
A(L) = 1 / [4 pi I_2(L)].
```

The single positive length `L` is fixed from

```text
epsilon_c^PBE = -epsilon_x^PBE - 2 pi A(L) I_1(L).
```

Because `A>1`, the correlation hole starts negative and crosses at `Q=1`. Beyond that crossing it is positive. Since `Q->0`,

```text
n_c/(-n_x) -> 1
n_xc/n_x -> 0.
```

The long-range correlation component therefore cancels the GGA exchange tail in the asymptotic ratio sense.

## Screen families and selection

Five positive `C-infinity` screens were audited:

```text
exp(-z)
exp(-z^2)
exp(-z^4)
(1+z^4)^(-2)
compact C-infinity bump
```

The noncompact quartic super-Gaussian `D(z)=exp(-z^4)` is the recommended default. It is strictly positive at finite separation, flat through third derivative at the origin, and reaches 99% exchange-tail cancellation substantially earlier than the Gaussian and exponential screens.

## Numerical audit

The audit uses 48 training states and 48 disjoint validation states over `r_s`, spin polarization, exchange reduced gradients, and PBE correlation reduced gradients. Five screen families produce 480 coupled profiles.

For the selected quartic super-Gaussian on the validation set:

```text
maximum sampled positive n_xc                 0.000000e+00
maximum absolute correlation particle error  1.580241e-14
maximum absolute correlation energy error    2.692291e-15 Ha/e
maximum absolute exchange particle error     1.554312e-14
maximum absolute exchange energy error        1.332268e-15 Ha/e
median dimensionless 99% cancellation range  1.441107e+01
```

The previous independently smoothed correlation model makes the combined hole positive in all 48 validation states.

## Implementation status

The complete reproducibility artifact contains:

```text
coupled_gga_xc_hole.py
xcholemodel_adapter.py
gga_c_hole_model.py
benchmark_coupled_models.py
generate_figures.py
tests/test_coupled_constraints.py
manuscript.tex
manuscript.pdf
```

The local model accepts explicit `s_up` and `s_down` values, matching the spin-resolved gradient data already derived in `holemodel.py`. A reference grid adapter contracts local profiles with the repository convention

```text
<n_alpha(u)> = N^(-1) integral n(r) n_alpha(r,u) dr.
```

The production path should tabulate or interpolate the two screening parameters over semilocal descriptor space rather than solve a scalar root independently at every real-space integration point.

The regression suite contains nine passing test groups. The compiled 16-page manuscript passes PDF preflight and renders correctly with PDFium and Poppler.