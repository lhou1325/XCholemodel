"""C-infinity moment-conserving closures for PBE-type correlation holes.

The conventional real-space-cutoff model multiplies the complete GGA
correlation kernel by a step function.  This module instead modifies only the
gradient correction through a smooth flux closure.  The construction enforces
zero particle moment analytically and fixes one positive length scale from the
PBE correlation-energy moment.

All equations use the dimensionless separation v = phi * k_s * u.  If

    nbar_c(r,u) = phi**5 * k_s**2 * h_c(v),

then the local constraints are

    integral v**2 h_c(v) dv = 0,
    2*pi*phi**3 * integral v*h_c(v) dv = epsilon_c.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

Array = np.ndarray
ClosureName = Literal["gaussian", "quartic", "sextic", "compact"]

PBE_GAMMA = 0.031091
PBE_BETA = 0.066725
TINY = 1.0e-15


@dataclass(frozen=True)
class ClosureResult:
    """Result of closing one local gradient-correction kernel."""

    v: Array
    lda_kernel: Array
    parent_correction: Array
    closed_correction: Array
    closed_kernel: Array
    flux: Array
    damping: Array
    scale: float
    closure: str
    target_increment: float
    particle_residual: float
    energy_residual: float


def _as_grid(values: Array, name: str) -> Array:
    out = np.asarray(values, dtype=float)
    if out.ndim != 1 or out.size < 4:
        raise ValueError(f"{name} must be a one-dimensional array with at least four points")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values")
    return out


def _trapz(y: Array, x: Array) -> float:
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(y, x=x))


def pbe_correlation_increment(
    epsilon_c_lda: float | Array,
    phi: float | Array,
    t: float | Array,
    *,
    beta: float = PBE_BETA,
    gamma: float = PBE_GAMMA,
) -> Array:
    """Return the PBE correlation increment H in Hartree per electron.

    Parameters
    ----------
    epsilon_c_lda
        Spin-interpolated uniform-gas correlation energy per electron.  It is
        negative for physical densities.
    phi
        PBE spin-scaling factor
        ``[(1+zeta)^(2/3)+(1-zeta)^(2/3)]/2``.
    t
        Reduced density gradient used by PBE correlation.
    """

    eps, phi, t = np.broadcast_arrays(
        np.asarray(epsilon_c_lda, dtype=float),
        np.asarray(phi, dtype=float),
        np.asarray(t, dtype=float),
    )
    if np.any(phi <= 0.0):
        raise ValueError("phi must be positive")
    if np.any(t < 0.0):
        raise ValueError("t must be non-negative")

    exponent = np.clip(-eps / (gamma * phi**3), -700.0, 700.0)
    denominator = np.expm1(exponent)
    a_value = np.divide(
        beta / gamma,
        denominator,
        out=np.zeros_like(denominator),
        where=np.abs(denominator) > TINY,
    )
    t2 = t * t
    rational = np.divide(
        1.0 + a_value * t2,
        1.0 + a_value * t2 + a_value * a_value * t2 * t2,
    )
    return gamma * phi**3 * np.log1p((beta / gamma) * t2 * rational)


def closure_value_and_derivative(x: Array, closure: ClosureName) -> tuple[Array, Array]:
    """Return D(x) and dD/dx for one C-infinity closure."""

    x = np.asarray(x, dtype=float)
    if np.any(x < 0.0):
        raise ValueError("closure coordinate must be non-negative")

    if closure == "gaussian":
        d = np.exp(-(x**2))
        return d, -2.0 * x * d
    if closure == "quartic":
        d = np.exp(-(x**4))
        return d, -4.0 * x**3 * d
    if closure == "sextic":
        d = np.exp(-(x**6))
        return d, -6.0 * x**5 * d
    if closure == "compact":
        d = np.zeros_like(x)
        derivative = np.zeros_like(x)
        inside = x < 1.0
        if np.any(inside):
            xi = x[inside]
            one_minus = 1.0 - xi**4
            di = np.exp(-(xi**4) / one_minus)
            d[inside] = di
            derivative[inside] = -4.0 * xi**3 * di / one_minus**2
        return d, derivative
    raise ValueError(f"unknown closure: {closure}")


def correction_flux(v: Array, parent_correction: Array) -> Array:
    """Return F(v)=integral_0^v s^2 b(s) ds on the supplied grid."""

    v = _as_grid(v, "v")
    b = _as_grid(parent_correction, "parent_correction")
    if b.shape != v.shape:
        raise ValueError("v and parent_correction must have identical shapes")
    if np.any(v < 0.0) or np.any(np.diff(v) <= 0.0):
        raise ValueError("v must be non-negative and strictly increasing")
    return cumulative_trapezoid(v * v * b, v, initial=0.0)


def _energy_increment_from_flux(v: Array, flux: Array, scale: float, closure: ClosureName) -> float:
    damping, _ = closure_value_and_derivative(v / scale, closure)
    integrand = np.zeros_like(v)
    positive = v > 0.0
    integrand[positive] = damping[positive] * flux[positive] / v[positive] ** 2
    return _trapz(integrand, v)


def solve_closure_scale(
    v: Array,
    flux: Array,
    target_moment: float,
    closure: ClosureName = "quartic",
    *,
    rtol: float = 1.0e-11,
    max_expand: int = 80,
) -> float:
    """Solve the positive closure length from the energy-moment equation.

    `target_moment` is the dimensionless correction moment
    `H/(2*pi*phi**3)`.  The supplied radial interval must be long enough that
    the target is bracketed.  Failure to bracket is reported rather than
    silently producing a non-energy-conserving hole.
    """

    v = _as_grid(v, "v")
    flux = _as_grid(flux, "flux")
    if flux.shape != v.shape:
        raise ValueError("v and flux must have identical shapes")
    if target_moment < -1.0e-14:
        raise ValueError("target correction moment must be non-negative")
    if abs(target_moment) <= 1.0e-14:
        return 0.0

    first_positive = v[np.nonzero(v > 0.0)[0][0]]
    lower = max(first_positive * 1.0e-6, 1.0e-12)
    upper = max(v[-1] / 8.0, first_positive)

    def residual(scale: float) -> float:
        return _energy_increment_from_flux(v, flux, scale, closure) - target_moment

    f_lower = residual(lower)
    f_upper = residual(upper)
    for _ in range(max_expand):
        if f_lower == 0.0:
            return lower
        if f_upper == 0.0:
            return upper
        if f_lower * f_upper < 0.0:
            return float(brentq(residual, lower, upper, rtol=rtol, xtol=1.0e-14, maxiter=300))
        upper *= 2.0
        f_upper = residual(upper)

    limiting = _energy_increment_from_flux(v, flux, upper, closure)
    raise RuntimeError(
        "unable to bracket the closure scale; extend v_max or inspect the parent correction "
        f"(target={target_moment:.12e}, sampled_limit={limiting:.12e})"
    )


def close_gradient_correction(
    v: Array,
    lda_kernel: Array,
    parent_correction: Array,
    *,
    phi: float,
    target_increment: float,
    closure: ClosureName = "quartic",
) -> ClosureResult:
    """Construct a smooth local GGA correlation kernel.

    Parameters
    ----------
    v
        Dimensionless radial grid.  A zero first point is allowed.
    lda_kernel
        PW92/LSDA correlation kernel A_c(v).
    parent_correction
        Uncut GGA correction b(v)=t^2 B_c(v).
    phi
        PBE spin-scaling factor.
    target_increment
        Analytical PBE correlation increment H in Hartree per electron.
    closure
        One of ``gaussian``, ``quartic``, ``sextic``, or ``compact``.
    """

    v = _as_grid(v, "v")
    lda = _as_grid(lda_kernel, "lda_kernel")
    parent = _as_grid(parent_correction, "parent_correction")
    if lda.shape != v.shape or parent.shape != v.shape:
        raise ValueError("v, lda_kernel, and parent_correction must have identical shapes")
    if phi <= 0.0:
        raise ValueError("phi must be positive")
    if target_increment < -1.0e-14:
        raise ValueError("PBE correlation increment must be non-negative")

    flux = correction_flux(v, parent)
    target_moment = target_increment / (2.0 * np.pi * phi**3)

    if abs(target_moment) <= 1.0e-14:
        closed = np.zeros_like(parent)
        damping = np.ones_like(parent)
        scale = 0.0
    else:
        scale = solve_closure_scale(v, flux, target_moment, closure)
        damping, damping_prime = closure_value_and_derivative(v / scale, closure)
        closed = damping * parent
        positive = v > 0.0
        closed[positive] += (
            damping_prime[positive]
            * flux[positive]
            / (scale * v[positive] ** 2)
        )
        if not positive[0]:
            closed[0] = parent[0]

    kernel = lda + closed
    particle_residual = _trapz(v * v * closed, v)
    energy_increment = 2.0 * np.pi * phi**3 * _trapz(v * closed, v)
    energy_residual = energy_increment - target_increment

    return ClosureResult(
        v=v.copy(),
        lda_kernel=lda.copy(),
        parent_correction=parent.copy(),
        closed_correction=closed,
        closed_kernel=kernel,
        flux=flux,
        damping=damping,
        scale=float(scale),
        closure=closure,
        target_increment=float(target_increment),
        particle_residual=float(particle_residual),
        energy_residual=float(energy_residual),
    )


def close_parent_gga_kernel(
    v: Array,
    lda_kernel: Array,
    parent_gga_kernel: Array,
    *,
    epsilon_c_lda: float,
    phi: float,
    t: float,
    closure: ClosureName = "quartic",
) -> ClosureResult:
    """Convenience adapter for an LSDA kernel and an uncut GGA kernel."""

    lda = np.asarray(lda_kernel, dtype=float)
    parent = np.asarray(parent_gga_kernel, dtype=float)
    increment = float(pbe_correlation_increment(epsilon_c_lda, phi, t))
    return close_gradient_correction(
        v,
        lda,
        parent - lda,
        phi=phi,
        target_increment=increment,
        closure=closure,
    )


def particle_moment(v: Array, kernel: Array) -> float:
    """Return integral v^2 h(v) dv."""

    v = _as_grid(v, "v")
    kernel = _as_grid(kernel, "kernel")
    return _trapz(v * v * kernel, v)


def correlation_energy(v: Array, kernel: Array, phi: float) -> float:
    """Return 2*pi*phi^3 integral v h(v) dv in Hartree per electron."""

    v = _as_grid(v, "v")
    kernel = _as_grid(kernel, "kernel")
    return 2.0 * np.pi * phi**3 * _trapz(v * kernel, v)


__all__ = [
    "ClosureResult",
    "close_gradient_correction",
    "close_parent_gga_kernel",
    "closure_value_and_derivative",
    "correction_flux",
    "correlation_energy",
    "particle_moment",
    "pbe_correlation_increment",
    "solve_closure_scale",
]
