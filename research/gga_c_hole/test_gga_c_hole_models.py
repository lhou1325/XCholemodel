"""Tests for the smooth GGA correlation-hole companion module."""

import math
import unittest

import numpy as np
from scipy.integrate import cumulative_trapezoid

from gga_c_hole_models import (
    close_gradient_correction,
    closure_value_and_derivative,
    correlation_energy,
    particle_moment,
    pbe_correlation_increment,
)


TRAPEZOID = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


class SmoothCorrelationHoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v = np.linspace(0.0, 40.0, 40001)
        cls.parent = cls.v**2 * np.exp(-cls.v)
        cls.lda = -0.06 * np.exp(-0.7 * cls.v**2) * (1.0 - 0.35 * cls.v**2)
        cls.phi = 0.91

    def target_from_known_scale(self, closure, scale):
        flux = cumulative_trapezoid(self.v**2 * self.parent, self.v, initial=0.0)
        damping, _ = closure_value_and_derivative(self.v / scale, closure)
        integrand = np.zeros_like(self.v)
        integrand[1:] = damping[1:] * flux[1:] / self.v[1:] ** 2
        moment = float(TRAPEZOID(integrand, x=self.v))
        return 2.0 * math.pi * self.phi**3 * moment

    def test_all_closures_recover_known_scale_and_moments(self):
        for closure, known_scale in (
            ("gaussian", 2.0),
            ("quartic", 2.5),
            ("sextic", 3.0),
            ("compact", 6.0),
        ):
            with self.subTest(closure=closure):
                target = self.target_from_known_scale(closure, known_scale)
                result = close_gradient_correction(
                    self.v,
                    self.lda,
                    self.parent,
                    phi=self.phi,
                    target_increment=target,
                    closure=closure,
                )
                self.assertTrue(np.isfinite(result.closed_kernel).all())
                self.assertAlmostEqual(result.scale, known_scale, delta=2.0e-7)
                self.assertLess(abs(result.energy_residual), 2.0e-7)
                self.assertLess(abs(result.particle_residual), 2.0e-7)

    def test_total_particle_and_energy_constraints(self):
        # Make an LSDA-like baseline with exactly zero numerical particle moment
        # by adding a smooth compensating Gaussian.
        lda = -np.exp(-self.v**2)
        basis = np.exp(-0.25 * self.v**2)
        coefficient = -particle_moment(self.v, lda) / particle_moment(self.v, basis)
        lda = lda + coefficient * basis
        eps_lda = correlation_energy(self.v, lda, self.phi)

        target_increment = self.target_from_known_scale("quartic", 2.8)
        result = close_gradient_correction(
            self.v,
            lda,
            self.parent,
            phi=self.phi,
            target_increment=target_increment,
            closure="quartic",
        )
        self.assertLess(abs(particle_moment(self.v, result.closed_kernel)), 3.0e-7)
        self.assertLess(
            abs(correlation_energy(self.v, result.closed_kernel, self.phi) - (eps_lda + target_increment)),
            3.0e-7,
        )

    def test_lsda_limit(self):
        zeros = np.zeros_like(self.v)
        result = close_gradient_correction(
            self.v,
            self.lda,
            zeros,
            phi=1.0,
            target_increment=0.0,
            closure="quartic",
        )
        self.assertEqual(result.scale, 0.0)
        self.assertTrue(np.array_equal(result.closed_kernel, self.lda))
        self.assertTrue(np.array_equal(result.closed_correction, zeros))

    def test_quartic_preserves_parent_through_v4(self):
        # D_4(x)=1-x^4+O(x^8).  Because F(v)=O(v^5) when b(v)=O(v^2),
        # the closure term first changes b at order v^6.
        x = np.array([0.0, 1.0e-4, 2.0e-4, 4.0e-4])
        d, derivative = closure_value_and_derivative(x, "quartic")
        self.assertLess(np.max(np.abs(d - (1.0 - x**4))), 1.0e-15)
        self.assertLess(np.max(np.abs(derivative + 4.0 * x**3)), 1.0e-14)

    def test_compact_closure_is_flat_at_boundary(self):
        x = np.array([0.999, 0.9999, 1.0, 1.001])
        d, derivative = closure_value_and_derivative(x, "compact")
        self.assertEqual(d[2], 0.0)
        self.assertEqual(d[3], 0.0)
        self.assertEqual(derivative[2], 0.0)
        self.assertEqual(derivative[3], 0.0)
        self.assertLess(d[1], d[0])

    def test_pbe_increment_is_nonnegative_and_has_lsda_limit(self):
        eps = np.array([-0.08, -0.04, -0.02])
        phi = np.array([1.0, 0.95, 0.85])
        zero = pbe_correlation_increment(eps, phi, np.zeros(3))
        finite_t = pbe_correlation_increment(eps, phi, np.ones(3))
        self.assertTrue(np.allclose(zero, 0.0))
        self.assertTrue(np.all(finite_t > 0.0))

    def test_spin_reversal_invariance_of_phi_input(self):
        for zeta in (0.0, 0.3, 0.7, 0.95):
            phi_plus = 0.5 * ((1.0 + zeta) ** (2.0 / 3.0) + (1.0 - zeta) ** (2.0 / 3.0))
            phi_minus = 0.5 * ((1.0 - zeta) ** (2.0 / 3.0) + (1.0 + zeta) ** (2.0 / 3.0))
            h_plus = pbe_correlation_increment(-0.04, phi_plus, 1.2)
            h_minus = pbe_correlation_increment(-0.04, phi_minus, 1.2)
            self.assertAlmostEqual(float(h_plus), float(h_minus), places=15)


if __name__ == "__main__":
    unittest.main()
