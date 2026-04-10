"""Compute LDA and PBE exchange-correlation hole models from HDF5 grid data.

Expected input datasets:
- ``rho``: spin densities with shape ``(n_grid, 2)``
- ``grd``: spin gradients with shape ``(n_grid, 7)``
- ``xyz``: grid coordinates and weights with shape ``(n_grid, 4)``

For each input file, the script writes a text energy summary and an HDF5
``.plot`` file containing the radial hole profiles and cumulative energies.
"""

from dataclasses import dataclass
import math
import os
import sys
import time

import h5py
import numpy as np
from scipy import integrate
from scipy.special import erfcx, exp1


CUMTRAPZ = (
    integrate.cumulative_trapezoid
    if hasattr(integrate, "cumulative_trapezoid")
    else integrate.cumtrapz
)
TRAPEZOID = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# Numerical safety constants.
RHO_FLOOR = 1e-18
DENOM_FLOOR = 1e-30
RADIAL_FLOOR = 1e-14
MAX_NEGEXP_ARGUMENT = 700.0
PBE_S_REGULARIZATION_START = 8.0
PBE_S_REGULARIZATION_LIMIT = 11.0

# Radial grid settings.
RADIAL_GRID_POINTS = 4001
RADIAL_GRID_SPACING = 0.0125
EXCHANGE_U_ZERO = 1e-6
OUTPUT_U_ZERO = 1e-10

# Exchange-hole coefficients.
EXCHANGE_H_COEFFS = (0.00979681, 0.041083, 0.187440, 0.00120824, 0.0347188)
EXCHANGE_KERNEL_COEFFS = (1.0161144, -0.37170836, -0.077215461, 0.57786348, -0.051955731)

# Correlation-hole coefficients.
CORRELATION_RATIONAL_NUMERATOR = (-0.1244, 0.027032, 0.0024317)
CORRELATION_RATIONAL_DENOMINATOR = (0.2199, 0.086664, 0.012858, 0.0020)
CORRELATION_ALPHA = 0.193
CORRELATION_BETA = 0.525
CORRELATION_GAMMA = 0.3393
CORRELATION_DELTA = 0.9
CORRELATION_EPSILON = 0.10161
SPIN_INTERPOLATION_NORMALIZER = 1.709921
CORRELATION_D_BASE = 0.305
CORRELATION_D_ZETA_COEFF = 0.136
PW92_UNPOLARIZED = (0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, 1)
PW92_FULLY_POLARIZED = (0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517, 1)
PW92_ALPHA_C = (0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671, 1)

# Plot dataset order must stay stable for downstream consumers.
PLOT_DATASET_ORDER = (
    "u_axis",
    "LDA_X",
    "LDA_C",
    "LDA_XC",
    "PBE_X",
    "PBE_C",
    "PBE_XC",
    "LDA_EX",
    "LDA_EC",
    "LDA_EXC",
    "PBE_EX",
    "PBE_EC",
    "PBE_EXC",
)

RUN_STEP_DEFINITIONS = (
    ("Load input grid data", "Reading densities, gradients, and integration weights from the HDF5 file."),
    ("Build radial grid", "Constructing the u-axis used for hole profiles and cumulative energies."),
    ("Derive density fields", "Preparing reduced gradients and spin-polarization quantities."),
    ("Compute exchange holes", "Building the LDA and PBE exchange-hole profiles."),
    ("Compute correlation holes", "Building the LDA and PBE correlation-hole profiles."),
    ("Summarize energies and sum rules", "Combining hole profiles into energies, sum rules, and cusp values."),
    ("Write text report", "Saving the human-readable energy summary file."),
    ("Write plot file", "Saving the HDF5 plot datasets for downstream analysis."),
)


@dataclass(frozen=True)
class GridData:
    path: str
    rho_up: np.ndarray
    rho_down: np.ndarray
    grad_up: np.ndarray
    grad_down: np.ndarray
    grid_weights: np.ndarray
    density_total: np.ndarray
    electron_count_up: float
    electron_count_down: float
    electron_count_total: float
    normalizer: float


@dataclass(frozen=True)
class RadialGrid:
    npts: int
    delta_u: float
    energy_u: np.ndarray
    exchange_u: np.ndarray
    u_axis: np.ndarray


@dataclass(frozen=True)
class DerivedFields:
    active_total: np.ndarray
    active_up: np.ndarray
    active_down: np.ndarray
    density_safe: np.ndarray
    rho_up_safe: np.ndarray
    rho_down_safe: np.ndarray
    grad_up_sq: np.ndarray
    grad_down_sq: np.ndarray
    grad_cross: np.ndarray
    grad_total_sq: np.ndarray
    kf: np.ndarray
    ks: np.ndarray
    rs: np.ndarray
    zeta: np.ndarray
    phi: np.ndarray
    s: np.ndarray
    s_up: np.ndarray
    s_down: np.ndarray
    correlation_d: np.ndarray
    p: np.ndarray
    t: np.ndarray


@dataclass(frozen=True)
class ModelReport:
    hx_lda: np.ndarray
    hc_lda: np.ndarray
    hxc_lda: np.ndarray
    hx_pbe: np.ndarray
    hc_pbe: np.ndarray
    hxc_pbe: np.ndarray
    ex_lda: np.ndarray
    ec_lda: np.ndarray
    exc_lda: np.ndarray
    ex_pbe: np.ndarray
    ec_pbe: np.ndarray
    exc_pbe: np.ndarray
    sumx_lda: float
    sumc_lda: float
    sumx_pbe: float
    sumc_pbe: float
    ontop_x: float
    ontop_c: float
    ontop_xc: float
    cusp_x: float
    cusp_c: float
    cusp_xc: float


@dataclass
class RunStatus:
    input_path: str
    output_stem: str
    total_steps: int
    run_start: float
    current_step_number: int = 0
    current_step_name: str = ""
    current_step_start: float = 0.0


def safe_clip_density(rho):
    """Clip densities to the non-negative domain."""
    return np.clip(rho, 0.0, None)


def finite_or_zero(values):
    """Replace NaN and Inf values with zeros."""
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def safe_divide(numerator, denominator, fill=0.0, where=None, min_abs_denominator=DENOM_FLOOR):
    """Divide while masking small denominators and non-finite outputs."""
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, dtype=float),
        np.asarray(denominator, dtype=float),
    )
    result = np.full(numerator.shape, fill, dtype=float)
    valid = np.abs(denominator) > min_abs_denominator
    if where is not None:
        valid &= np.broadcast_to(where, numerator.shape)
    np.divide(numerator, denominator, out=result, where=valid)
    return finite_or_zero(result)


def safe_inverse_square(x, where=None, min_abs_x=RADIAL_FLOOR):
    """Compute ``1 / x**2`` only where the radius is safely away from zero."""
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x, dtype=float)
    valid = np.abs(x) > min_abs_x
    if where is not None:
        valid &= np.broadcast_to(where, x.shape)
    np.divide(1.0, x * x, out=result, where=valid)
    return finite_or_zero(result)


def safe_negexp(argument):
    """Evaluate ``exp(-x)`` with clipping that avoids overflow."""
    return np.exp(-np.clip(np.asarray(argument, dtype=float), 0.0, MAX_NEGEXP_ARGUMENT))


def trapz_integral(y, x):
    """Integrate a one-dimensional profile with finite cleanup."""
    return float(finite_or_zero(TRAPEZOID(finite_or_zero(y), x=x)))


def cumulative_integral(y, x):
    """Compute a cumulative trapezoid integral with finite cleanup."""
    return finite_or_zero(CUMTRAPZ(finite_or_zero(y), x, initial=0))


def regularize_reduced_gradient(s, s1=PBE_S_REGULARIZATION_START, s2=PBE_S_REGULARIZATION_LIMIT):
    """Smooth very large reduced gradients to the stable PBE hole regime."""
    s = np.asarray(s, dtype=float)
    out = np.clip(s, 0.0, None).copy()
    mask = out > s1
    if np.any(mask):
        delta = out[mask] - s1
        out[mask] = out[mask] - delta * np.exp(-(s2 - s1) / delta)
        out[mask] = np.minimum(out[mask], np.nextafter(s2, 0.0))
    return out


def scaled_exp1(x):
    """Evaluate the stabilized ``x * exp(x) * E1(x)`` helper."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    zero_mask = x <= 0
    large_mask = x > 50
    small_mask = ~(zero_mask | large_mask)

    if np.any(zero_mask):
        out[zero_mask] = 0.0

    if np.any(small_mask):
        x_small = x[small_mask]
        out[small_mask] = x_small * np.exp(x_small) * exp1(x_small)

    if np.any(large_mask):
        inv_x = 1 / x[large_mask]
        out[large_mask] = 1 - inv_x + 2 * inv_x**2 - 6 * inv_x**3

    return finite_or_zero(out)


def _format_seconds(seconds):
    return f"{seconds:.2f} s"


def _make_run_status(path):
    input_path = os.path.abspath(path)
    return RunStatus(
        input_path=input_path,
        output_stem=_output_stem(input_path),
        total_steps=len(RUN_STEP_DEFINITIONS),
        run_start=time.monotonic(),
    )


def _log_run_header(status):
    print("XCholemodel progress")
    print(f"Input file: {status.input_path}")
    print(f"Planned steps: {status.total_steps}")


def _start_step(status, step_name, detail):
    status.current_step_number += 1
    status.current_step_name = step_name
    status.current_step_start = time.monotonic()
    print("")
    print(f"[{status.current_step_number}/{status.total_steps}] {step_name}...")
    print(f"    {detail}")


def _log_substep(_, message):
    print(f"    - {message}")


def _finish_step(status, detail=None):
    elapsed = time.monotonic() - status.current_step_start
    message = f"    Done in {_format_seconds(elapsed)}"
    if detail:
        message += f" | {detail}"
    print(message)


def _fail_step(status, exc):
    elapsed = time.monotonic() - status.current_step_start
    print(f"    FAILED after {_format_seconds(elapsed)} during {status.current_step_name}: {exc}")


def _render_step_detail(detail, result):
    if detail is None:
        return None
    if callable(detail):
        return detail(result)
    return detail


def _run_step(status, step_name, detail, func, *args, completion_detail=None):
    _start_step(status, step_name, detail)
    try:
        result = func(*args)
    except Exception as exc:
        _fail_step(status, exc)
        raise
    _finish_step(status, _render_step_detail(completion_detail, result))
    return result


def _log_final_summary(status, report, text_report_path, plot_file_path):
    total_runtime = time.monotonic() - status.run_start
    print("")
    print("Final summary")
    print(
        "  LDA: "
        f"Ex={report.ex_lda[-1]: .12f}  "
        f"Ec={report.ec_lda[-1]: .12f}  "
        f"Exc={report.exc_lda[-1]: .12f}"
    )
    print(
        "  PBE: "
        f"Ex={report.ex_pbe[-1]: .12f}  "
        f"Ec={report.ec_pbe[-1]: .12f}  "
        f"Exc={report.exc_pbe[-1]: .12f}"
    )
    print(
        "  Sum rules: "
        f"LDA Sumx={report.sumx_lda: .12f}, "
        f"LDA Sumc={report.sumc_lda: .12f}, "
        f"PBE Sumx={report.sumx_pbe: .12f}, "
        f"PBE Sumc={report.sumc_pbe: .12f}"
    )
    print(f"  Text report: {text_report_path}")
    print(f"  Plot file: {plot_file_path}")
    print(f"  Total runtime: {_format_seconds(total_runtime)}")


def _load_grid_data(path):
    density_file = h5py.File(path, "r")
    try:
        rho_up = safe_clip_density(np.asarray(density_file["rho"][:, 0], dtype=float))
        rho_down = safe_clip_density(np.asarray(density_file["rho"][:, 1], dtype=float))
        grad_up = np.asarray(density_file["grd"][:, 0:3], dtype=float)
        grad_down = np.asarray(density_file["grd"][:, 4:7], dtype=float)
        grid_weights = np.asarray(density_file["xyz"][:, 3], dtype=float)
    finally:
        density_file.close()

    if not (rho_up.shape == rho_down.shape == grid_weights.shape):
        raise ValueError("rho and xyz datasets must share the same leading dimension.")
    if grad_up.shape[0] != rho_up.shape[0] or grad_down.shape[0] != rho_up.shape[0]:
        raise ValueError("grd dataset must align with rho along the grid dimension.")

    density_total = rho_up + rho_down
    electron_count_up = float(np.dot(grid_weights, rho_up))
    electron_count_down = float(np.dot(grid_weights, rho_down))
    electron_count_total = float(np.dot(grid_weights, density_total))
    normalizer = electron_count_total if electron_count_total > RHO_FLOOR else 1.0

    return GridData(
        path=os.path.abspath(path),
        rho_up=rho_up,
        rho_down=rho_down,
        grad_up=grad_up,
        grad_down=grad_down,
        grid_weights=grid_weights,
        density_total=density_total,
        electron_count_up=electron_count_up,
        electron_count_down=electron_count_down,
        electron_count_total=electron_count_total,
        normalizer=normalizer,
    )


def _build_radial_grid():
    energy_u = np.linspace(0, (RADIAL_GRID_POINTS - 1) * RADIAL_GRID_SPACING, RADIAL_GRID_POINTS)
    exchange_u = energy_u.copy()
    exchange_u[0] = EXCHANGE_U_ZERO
    u_axis = energy_u.copy()
    u_axis[0] = OUTPUT_U_ZERO
    return RadialGrid(
        npts=RADIAL_GRID_POINTS,
        delta_u=RADIAL_GRID_SPACING,
        energy_u=energy_u,
        exchange_u=exchange_u,
        u_axis=u_axis,
    )

def _derive_fields(grid, radial):
    grad_up_sq = np.einsum("ij,ij->i", grid.grad_up, grid.grad_up)
    grad_down_sq = np.einsum("ij,ij->i", grid.grad_down, grid.grad_down)
    grad_cross = np.einsum("ij,ij->i", grid.grad_up, grid.grad_down)
    grad_total_sq = grad_up_sq + grad_down_sq + 2 * grad_cross

    active_total = grid.density_total > RHO_FLOOR
    active_up = grid.rho_up > RHO_FLOOR
    active_down = grid.rho_down > RHO_FLOOR

    density_safe = np.where(active_total, grid.density_total, RHO_FLOOR)
    rho_up_safe = np.where(active_up, grid.rho_up, RHO_FLOOR)
    rho_down_safe = np.where(active_down, grid.rho_down, RHO_FLOOR)

    sqrt_grad_total = np.sqrt(np.clip(grad_total_sq, 0.0, None))
    sqrt_grad_up = np.sqrt(np.clip(grad_up_sq, 0.0, None))
    sqrt_grad_down = np.sqrt(np.clip(grad_down_sq, 0.0, None))

    # Canonical density-derived quantities used throughout the model.
    kf = (3 * math.pi**2 * density_safe) ** (1 / 3)
    rs = (3 / (4 * np.pi * density_safe)) ** (1 / 3)
    zeta = np.clip(
        safe_divide(grid.rho_up - grid.rho_down, grid.density_total, where=active_total),
        -1.0,
        1.0,
    )
    ks = np.sqrt(4 * kf / np.pi)
    phi = 0.5 * ((1 + zeta) ** (2 / 3) + (1 - zeta) ** (2 / 3))

    s = regularize_reduced_gradient(
        safe_divide(sqrt_grad_total, 2 * kf * grid.density_total, where=active_total)
    )
    spin_kf_up = (3 * math.pi**2 * (2 * rho_up_safe)) ** (1 / 3)
    spin_kf_down = (3 * math.pi**2 * (2 * rho_down_safe)) ** (1 / 3)
    s_up = regularize_reduced_gradient(
        safe_divide(sqrt_grad_up, spin_kf_up * (2 * grid.rho_up), where=active_up)
    )
    s_down = regularize_reduced_gradient(
        safe_divide(sqrt_grad_down, spin_kf_down * (2 * grid.rho_down), where=active_down)
    )

    correlation_d = CORRELATION_D_BASE - CORRELATION_D_ZETA_COEFF * zeta * zeta
    p = safe_divide(np.pi * kf * correlation_d, 4 * phi**4)
    t = safe_divide(sqrt_grad_total, 2 * ks * phi * grid.density_total, where=active_total)

    return DerivedFields(
        active_total=active_total,
        active_up=active_up,
        active_down=active_down,
        density_safe=density_safe,
        rho_up_safe=rho_up_safe,
        rho_down_safe=rho_down_safe,
        grad_up_sq=grad_up_sq,
        grad_down_sq=grad_down_sq,
        grad_cross=grad_cross,
        grad_total_sq=grad_total_sq,
        kf=kf,
        ks=ks,
        rs=rs,
        zeta=zeta,
        phi=phi,
        s=s,
        s_up=s_up,
        s_down=s_down,
        correlation_d=correlation_d,
        p=p,
        t=t,
    )


def _gga_h_function(s):
    a1, a2, a3, a4, a5 = EXCHANGE_H_COEFFS
    numerator = a1 * s**2 + a2 * s**4
    denominator = 1 + a3 * s**4 + a4 * s**5 + a5 * s**6
    return safe_divide(numerator, denominator)


def _gga_f_function(h):
    return 6.475 * h + 0.4797


def _exchange_constant_a(h, f_value, s):
    a_gga, b_gga, c_gga, d_gga, e_gga = EXCHANGE_KERNEL_COEFFS
    h = np.clip(h, 0.0, None)
    denom_term = d_gga + h * s**2
    first_term = 15 * e_gga + 6 * c_gga * (1 + f_value * s**2) * denom_term
    second_term = 4 * b_gga * denom_term**2 + 8 * a_gga * denom_term**3
    first_value = np.sqrt(math.pi) * (first_term + second_term)
    third_term = safe_divide(1.0, 16 * denom_term ** (7 / 2))
    erfcx_arg = (3 * s / 2) * np.sqrt(h / a_gga)
    fourth_term = (3 * math.pi * np.sqrt(a_gga) / 4) * erfcx(erfcx_arg)
    return finite_or_zero(first_value * third_term - fourth_term)


def _exchange_constant_b(s, h):
    _, _, _, d_gga, _ = EXCHANGE_KERNEL_COEFFS
    h = np.clip(h, 0.0, None)
    denom_term = np.maximum(d_gga + h * s**2, DENOM_FLOOR)
    numerator = 15 * np.sqrt(math.pi) * s**2
    denominator = 16 * denom_term ** (7 / 2)
    return safe_divide(numerator, denominator)


def _exchange_g_function(v1, v2):
    _, _, _, _, e_gga = EXCHANGE_KERNEL_COEFFS
    numerator = 0.75 * math.pi + v1
    denominator = v2 * e_gga
    return safe_divide(-numerator, denominator)


def _j_lda_kernel(x):
    a_gga, b_gga, c_gga, d_gga, e_gga = EXCHANGE_KERNEL_COEFFS
    x = np.asarray(x, dtype=float)
    output = np.zeros_like(x, dtype=float)
    valid = np.isfinite(x) & (np.abs(x) > RADIAL_FLOOR)
    if not np.any(valid):
        return output

    x_valid = x[valid]
    x_sq = x_valid**2
    inv_x_sq = safe_inverse_square(x_valid)
    output[valid] = finite_or_zero(
        (-a_gga * inv_x_sq) * safe_divide(1.0, 1 + (4 / 9) * a_gga * x_sq)
        + ((a_gga * inv_x_sq) + b_gga + c_gga * x_sq + e_gga * x_sq**2) * safe_negexp(d_gga * x_sq)
    )
    return output


def _j_gga_kernel(s, x):
    a_gga, b_gga, c_gga, d_gga, e_gga = EXCHANGE_KERNEL_COEFFS
    s, x = np.broadcast_arrays(np.asarray(s, dtype=float), np.asarray(x, dtype=float))
    output = np.zeros_like(x, dtype=float)
    valid = np.isfinite(s) & np.isfinite(x) & (np.abs(x) > RADIAL_FLOOR)
    if not np.any(valid):
        return output

    lda_mask = valid & np.isclose(s, 0.0)
    if np.any(lda_mask):
        output[lda_mask] = _j_lda_kernel(x[lda_mask])

    gga_mask = valid & ~lda_mask
    if np.any(gga_mask):
        s_valid = s[gga_mask]
        x_valid = x[gga_mask]
        h_gga = _gga_h_function(s_valid)
        f_gga = _gga_f_function(h_gga)
        a_value = _exchange_constant_a(h_gga, f_gga, s_valid)
        b_value = _exchange_constant_b(s_valid, h_gga)
        g_gga = _exchange_g_function(a_value, b_value)
        x_sq = x_valid**2
        inv_x_sq = safe_inverse_square(x_valid)
        attenuation = safe_negexp(s_valid**2 * h_gga * x_sq)
        damped = safe_negexp(d_gga * x_sq) * attenuation
        output[gga_mask] = finite_or_zero(
            (-a_gga * inv_x_sq) * safe_divide(1.0, 1 + (4 / 9) * a_gga * x_sq) * attenuation
            + (
                (a_gga * inv_x_sq)
                + b_gga
                + c_gga * (1 + s_valid**2 * f_gga) * x_sq
                + e_gga * (1 + s_valid**2 * g_gga) * x_sq**2
            )
            * damped
        )

    return finite_or_zero(output)


def _compute_exchange_holes(grid, radial, fields, status=None):
    if status is not None:
        _log_substep(status, "Preparing exchange kernels and spin-scaled radii.")
    spin_scale_up = (1 + fields.zeta) ** (1 / 3) * fields.kf
    spin_scale_down = (1 - fields.zeta) ** (1 / 3) * fields.kf
    x_up = np.outer(radial.exchange_u, spin_scale_up)
    x_down = np.outer(radial.exchange_u, spin_scale_down)
    spin_density_up = (2 * grid.rho_up) ** 2
    spin_density_down = (2 * grid.rho_down) ** 2

    if status is not None:
        _log_substep(status, "Contracting weighted exchange profiles over the grid.")
    hx_lda = finite_or_zero(
        (
            (_j_lda_kernel(x_up) * spin_density_up + _j_lda_kernel(x_down) * spin_density_down)
            @ grid.grid_weights
        )
        / (2 * grid.normalizer)
    )
    hx_pbe = finite_or_zero(
        (
            (
                _j_gga_kernel(fields.s_up[np.newaxis, :], x_up) * spin_density_up
                + _j_gga_kernel(fields.s_down[np.newaxis, :], x_down) * spin_density_down
            )
            @ grid.grid_weights
        )
        / (2 * grid.normalizer)
    )
    return hx_lda, hx_pbe


def spin_interpolation(zeta):
    return (((1 + zeta) ** (4 / 3) + (1 - zeta) ** (4 / 3) - 2) / (2 ** (4 / 3) - 2))


def pw92_correlation_energy(rs, a_value, alpha1, beta1, beta2, beta3, beta4, power):
    denominator = 2 * a_value * (
        beta1 * rs ** (1 / 2)
        + beta2 * rs
        + beta3 * rs ** (3 / 2)
        + beta4 * rs ** (power + 1)
    )
    log_term = 1 + safe_divide(1.0, denominator)
    return finite_or_zero(-2 * a_value * (1 + alpha1 * rs) * np.log(log_term))


def _lda_correlation_kernel(v, v_sq, exp_factor, fields, correlation_energy, active_grid):
    a1, a2, a3 = CORRELATION_RATIONAL_NUMERATOR
    b1, b2, b3, b4 = CORRELATION_RATIONAL_DENOMINATOR

    c1 = (
        -0.0012529
        + 0.1244 * fields.p
        + 0.61386
        * (1 - fields.zeta**2)
        / (fields.phi**5 * fields.rs**2)
        * (
            safe_divide(
                1 + CORRELATION_ALPHA * fields.rs,
                1 + CORRELATION_BETA * fields.rs + CORRELATION_ALPHA * CORRELATION_BETA * fields.rs**2,
            )
            - 1
        )
    )
    c2 = (
        0.0033894
        - 0.054388 * fields.p
        + 0.39270
        * (1 - fields.zeta**2)
        / (fields.phi**6 * fields.rs**1.5)
        * safe_divide(
            1 + CORRELATION_GAMMA * fields.rs,
            2 + CORRELATION_DELTA * fields.rs + CORRELATION_EPSILON * fields.rs**2,
        )
        * safe_divide(
            1 + CORRELATION_ALPHA * fields.rs,
            1 + CORRELATION_BETA * fields.rs + CORRELATION_ALPHA * CORRELATION_BETA * fields.rs**2,
        )
    )
    c3 = (
        0.10847 * fields.p**2.5
        + 1.4604 * fields.p**2
        + 0.51749 * fields.p**1.5
        - 3.5297 * c1 * fields.p
        - 1.9030 * c2 * fields.p**0.5
        + 1.0685 * fields.p**2 * np.log(np.maximum(fields.p, RHO_FLOOR))
        + 34.356 * fields.phi**(-3) * correlation_energy * fields.p**2
    )
    c4 = (
        -0.081596 * fields.p**3
        - 1.0810 * fields.p**2.5
        - 0.31677 * fields.p**2
        + 1.9030 * c1 * fields.p**1.5
        + 0.76485 * c2 * fields.p
        - 0.71019 * fields.p**2.5 * np.log(np.maximum(fields.p, RHO_FLOOR))
        - 22.836 * fields.phi**(-3) * correlation_energy * fields.p**2.5
    )
    c1 = finite_or_zero(c1)
    c2 = finite_or_zero(c2)
    c3 = finite_or_zero(c3)
    c4 = finite_or_zero(c4)

    rational_term = safe_divide(
        a1 + a2 * v + a3 * v_sq,
        1 + b1 * v + b2 * v_sq + b3 * v**3 + b4 * v**4,
        where=active_grid,
    )
    exponential_term = (
        -a1
        - (a2 - a1 * b1) * v
        + c1 * v_sq
        + c2 * v**3
        + c3 * v**4
        + c4 * v**5
    ) * exp_factor
    lda_correlation_kernel = safe_divide(rational_term + exponential_term, 4 * np.pi * v_sq, where=active_grid)
    lda_correlation_kernel[:, ~fields.active_total] = 0.0
    return finite_or_zero(lda_correlation_kernel)


def _gga_correction_kernel(v_sq, fields, active_grid):
    beta_rs = 2 * fields.p**2 / (3 * np.pi**3) * (1 - scaled_exp1(12 * fields.p))
    bclm = safe_divide(1.0, 18 * np.pi**3 * (1 + v_sq / 12) ** 2, where=active_grid)
    decay = safe_negexp(fields.p[np.newaxis, :] * v_sq)
    gga_correction_kernel = bclm * (1 - decay) + beta_rs[np.newaxis, :] * v_sq * decay
    gga_correction_kernel[:, ~fields.active_total] = 0.0
    return finite_or_zero(gga_correction_kernel)


def _interpolate_cutoff(v, cumulative_values):
    sign_change = np.isfinite(cumulative_values[1:]) & np.isfinite(cumulative_values[:-1])
    sign_change &= cumulative_values[1:] * cumulative_values[:-1] < 0
    candidate_rows = np.arange(1, v.shape[0])[:, np.newaxis]
    last_crossing = np.max(np.where(sign_change, candidate_rows, 0), axis=0)

    vc = np.zeros(v.shape[1], dtype=float)
    crossing_columns = np.nonzero(last_crossing)[0]
    if not crossing_columns.size:
        return vc

    right_index = last_crossing[crossing_columns]
    left_index = right_index - 1
    v0 = v[left_index, crossing_columns]
    v1 = v[right_index, crossing_columns]
    f0 = cumulative_values[left_index, crossing_columns]
    f1 = cumulative_values[right_index, crossing_columns]
    denominator = f1 - f0
    good_bracket = np.isfinite(v0) & np.isfinite(v1) & np.isfinite(f0) & np.isfinite(f1)
    correction = safe_divide(f0 * (v1 - v0), denominator, where=good_bracket)
    interpolated = np.where(
        good_bracket & (np.abs(denominator) > DENOM_FLOOR),
        v0 - correction,
        v1,
    )
    vc[crossing_columns] = finite_or_zero(interpolated)
    return vc


def _compute_correlation_holes(grid, radial, fields, status=None):
    if status is not None:
        _log_substep(status, "Evaluating PW92/LDA correlation ingredients.")
    ec_0 = pw92_correlation_energy(fields.rs, *PW92_UNPOLARIZED)
    ec_1 = pw92_correlation_energy(fields.rs, *PW92_FULLY_POLARIZED)
    alpha_c = -pw92_correlation_energy(fields.rs, *PW92_ALPHA_C)
    spin_factor = spin_interpolation(fields.zeta)
    correlation_energy = finite_or_zero(
        ec_0
        + alpha_c * spin_factor / SPIN_INTERPOLATION_NORMALIZER * (1 - fields.zeta**4)
        + (ec_1 - ec_0) * spin_factor * fields.zeta**4
    )

    v = np.outer(radial.u_axis, fields.phi * fields.ks)
    v_sq = v**2
    radial_scale = np.outer(radial.u_axis, safe_divide(fields.kf, fields.phi))
    exp_factor = safe_negexp(fields.correlation_d[np.newaxis, :] * radial_scale**2)
    active_grid = fields.active_total[np.newaxis, :]

    lda_correlation_kernel = _lda_correlation_kernel(
        v,
        v_sq,
        exp_factor,
        fields,
        correlation_energy,
        active_grid,
    )
    phi_ks_factor = (fields.phi**5 * fields.ks**2)[np.newaxis, :]
    nc_lsd = finite_or_zero(phi_ks_factor * lda_correlation_kernel)
    hc_lda = finite_or_zero(((nc_lsd * grid.density_total[np.newaxis, :]) @ grid.grid_weights) / grid.normalizer)

    if status is not None:
        _log_substep(status, "Building the GGA correction and radial cutoff.")
    gga_correction_kernel = _gga_correction_kernel(v_sq, fields, active_grid)
    gga_hole = finite_or_zero(lda_correlation_kernel + (fields.t**2)[np.newaxis, :] * gga_correction_kernel)
    nc_gea = finite_or_zero(phi_ks_factor * gga_hole)

    cumulative_hole = finite_or_zero(CUMTRAPZ(finite_or_zero(v_sq * gga_hole), v, axis=0, initial=0))
    vc = _interpolate_cutoff(v, cumulative_hole)

    nc_gga = np.where(v <= vc[np.newaxis, :], nc_gea, 0.0)
    nc_gga[:, ~fields.active_total] = 0.0
    if status is not None:
        _log_substep(status, "Contracting correlation profiles over the grid.")
    hc_pbe = finite_or_zero(((nc_gga * grid.density_total[np.newaxis, :]) @ grid.grid_weights) / grid.normalizer)
    return hc_lda, hc_pbe


def _summarize_results(radial, hx_lda, hx_pbe, hc_lda, hc_pbe):
    hxc_lda = finite_or_zero(hx_lda + hc_lda)
    hxc_pbe = finite_or_zero(hx_pbe + hc_pbe)

    delta_u0 = radial.u_axis[1] - radial.u_axis[0]
    cusp_x = float(safe_divide(hx_pbe[1] - hx_pbe[0], delta_u0))
    cusp_c = float(safe_divide(hc_pbe[1] - hc_pbe[0], delta_u0))
    cusp_xc = float(safe_divide(hxc_pbe[1] - hxc_pbe[0], delta_u0))

    sumx_lda = trapz_integral(4 * np.pi * radial.u_axis**2 * hx_lda, radial.u_axis)
    sumx_pbe = trapz_integral(4 * np.pi * radial.u_axis**2 * hx_pbe, radial.u_axis)
    sumc_lda = trapz_integral(4 * np.pi * radial.u_axis**2 * hc_lda, radial.u_axis)
    sumc_pbe = trapz_integral(4 * np.pi * radial.u_axis**2 * hc_pbe, radial.u_axis)

    ex_lda = cumulative_integral(4 * np.pi * hx_lda * radial.u_axis, radial.u_axis)
    ec_lda = cumulative_integral(4 * np.pi * hc_lda * radial.u_axis, radial.u_axis)
    exc_lda = cumulative_integral(4 * np.pi * hxc_lda * radial.u_axis, radial.u_axis)
    ex_pbe = cumulative_integral(4 * np.pi * hx_pbe * radial.u_axis, radial.u_axis)
    ec_pbe = cumulative_integral(4 * np.pi * hc_pbe * radial.u_axis, radial.u_axis)
    exc_pbe = cumulative_integral(4 * np.pi * hxc_pbe * radial.u_axis, radial.u_axis)

    return ModelReport(
        hx_lda=hx_lda,
        hc_lda=hc_lda,
        hxc_lda=hxc_lda,
        hx_pbe=hx_pbe,
        hc_pbe=hc_pbe,
        hxc_pbe=hxc_pbe,
        ex_lda=ex_lda,
        ec_lda=ec_lda,
        exc_lda=exc_lda,
        ex_pbe=ex_pbe,
        ec_pbe=ec_pbe,
        exc_pbe=exc_pbe,
        sumx_lda=sumx_lda,
        sumc_lda=sumc_lda,
        sumx_pbe=sumx_pbe,
        sumc_pbe=sumc_pbe,
        ontop_x=float(hx_lda[0]),
        ontop_c=float(hc_lda[0]),
        ontop_xc=float(hxc_lda[0]),
        cusp_x=cusp_x,
        cusp_c=cusp_c,
        cusp_xc=cusp_xc,
    )


def _output_stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _write_text_report(output_stem, report):
    report_lines = [
        "",
        f"LDA:Ex  = {report.ex_lda[-1]: 16.12f}  ",
        f"LDA:Ec  = {report.ec_lda[-1]: 16.12f}  ",
        f"LDA:Exc = {report.exc_lda[-1]: 16.12f}  ",
        "",
        f"PBE:Ex  = {report.ex_pbe[-1]: 16.12f}  ",
        f"PBE:Ec  = {report.ec_pbe[-1]: 16.12f}  ",
        f"PBE:Exc = {report.exc_pbe[-1]: 16.12f}  ",
        "",
        f"LDA:Sumx  = {report.sumx_lda: 16.12f}  ",
        f"LDA:Sumc  = {report.sumc_lda: 16.12f}  ",
        "",
        f"PBE:Sumx  = {report.sumx_pbe: 16.12f}  ",
        f"PBE:Sumc  = {report.sumc_pbe: 16.12f}  ",
        "",
        f"ontop x  = {report.ontop_x: 16.12f}  ",
        f"ontop c  = {report.ontop_c: 16.12f}  ",
        f"ontop xc = {report.ontop_xc: 16.12f}  ",
        "",
        f"cusp x  = {report.cusp_x: 16.12f}  ",
        f"cusp c  = {report.cusp_c: 16.12f}  ",
        f"cusp xc = {report.cusp_xc: 16.12f}  ",
    ]
    report_path = os.path.abspath(f"XChole_energy_{output_stem}.txt")
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(report_lines) + "\n")
    return report_path


def _write_plot_file(output_stem, radial, report):
    dataset_values = {
        "u_axis": radial.u_axis,
        "LDA_X": report.hx_lda,
        "LDA_C": report.hc_lda,
        "LDA_XC": report.hxc_lda,
        "PBE_X": report.hx_pbe,
        "PBE_C": report.hc_pbe,
        "PBE_XC": report.hxc_pbe,
        "LDA_EX": report.ex_lda,
        "LDA_EC": report.ec_lda,
        "LDA_EXC": report.exc_lda,
        "PBE_EX": report.ex_pbe,
        "PBE_EC": report.ec_pbe,
        "PBE_EXC": report.exc_pbe,
    }

    plot_path = os.path.abspath(f"XCholemodel_{output_stem}.plot")
    plot_file = h5py.File(plot_path, "w")
    try:
        for dataset_name in PLOT_DATASET_ORDER:
            values = np.asarray(dataset_values[dataset_name], dtype=float)
            plot_file.create_dataset(dataset_name, values.shape, dtype="f8", compression="gzip")
            plot_file[dataset_name][:] = values
    finally:
        plot_file.close()
    return plot_path


def DFThxcmodel(path):
    status = _make_run_status(path)
    _log_run_header(status)

    grid = _run_step(
        status,
        *RUN_STEP_DEFINITIONS[0],
        _load_grid_data,
        path,
        completion_detail=lambda loaded_grid: (
            f"{loaded_grid.density_total.size} grid points | "
            f"N_up={loaded_grid.electron_count_up:.6f} | "
            f"N_down={loaded_grid.electron_count_down:.6f} | "
            f"N_tot={loaded_grid.electron_count_total:.6f}"
        ),
    )
    radial = _run_step(
        status,
        *RUN_STEP_DEFINITIONS[1],
        _build_radial_grid,
        completion_detail=lambda built_radial: (
            f"{built_radial.npts} radial points | u_max={built_radial.energy_u[-1]:.4f}"
        ),
    )
    fields = _run_step(
        status,
        *RUN_STEP_DEFINITIONS[2],
        _derive_fields,
        grid,
        radial,
        completion_detail="Prepared reduced gradients and spin-polarization quantities.",
    )
    hx_lda, hx_pbe = _run_step(
        status,
        *RUN_STEP_DEFINITIONS[3],
        _compute_exchange_holes,
        grid,
        radial,
        fields,
        status,
        completion_detail="Generated LDA and PBE exchange-hole profiles.",
    )
    hc_lda, hc_pbe = _run_step(
        status,
        *RUN_STEP_DEFINITIONS[4],
        _compute_correlation_holes,
        grid,
        radial,
        fields,
        status,
        completion_detail="Generated LDA and PBE correlation-hole profiles.",
    )
    report = _run_step(
        status,
        *RUN_STEP_DEFINITIONS[5],
        _summarize_results,
        radial,
        hx_lda,
        hx_pbe,
        hc_lda,
        hc_pbe,
        completion_detail="Computed energies, sum rules, and cusp values.",
    )
    text_report_path = _run_step(
        status,
        *RUN_STEP_DEFINITIONS[6],
        _write_text_report,
        status.output_stem,
        report,
        completion_detail=lambda report_path: f"Saved {report_path}",
    )
    plot_file_path = _run_step(
        status,
        *RUN_STEP_DEFINITIONS[7],
        _write_plot_file,
        status.output_stem,
        radial,
        report,
        completion_detail=lambda plot_path: f"Saved {plot_path}",
    )
    _log_final_summary(status, report, text_report_path, plot_file_path)


def main():
    if len(sys.argv) < 2:
        print("Usage: python holemodel.py <input_path>")
        return 1

    input_path = sys.argv[1]
    DFThxcmodel(input_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
