import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = REPO_ROOT / "holemodel.py"
EXPECTED_PLOT_DATASET_NAMES = (
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
EXPECTED_PATHOLOGICAL_SCALARS = {
    "LDA:Ex": -1.538978557638,
    "LDA:Ec": -0.118648198565,
    "LDA:Exc": -1.657626756203,
    "PBE:Ex": -1.853333483585,
    "PBE:Ec": 0.0,
    "PBE:Exc": -1.853333483585,
    "LDA:Sumx": -0.993888570335,
    "LDA:Sumc": -0.006260059501,
    "PBE:Sumx": -0.995690140678,
    "PBE:Sumc": 0.0,
}
EXPECTED_PROGRESS_STEPS = (
    "[1/8] Load input grid data...",
    "[2/8] Build radial grid...",
    "[3/8] Derive density fields...",
    "[4/8] Compute exchange holes...",
    "[5/8] Compute correlation holes...",
    "[6/8] Summarize energies and sum rules...",
    "[7/8] Write text report...",
    "[8/8] Write plot file...",
)


def load_holemodel_module(store):
    class FakeH5File:
        def __init__(self, path, mode="r"):
            self.path = os.path.abspath(str(path))
            self.mode = mode
            if "r" in mode:
                self.datasets = store[self.path]
            else:
                self.datasets = {}
                store[self.path] = self.datasets

        def __getitem__(self, key):
            return self.datasets[key]

        def create_dataset(self, name, shape, dtype=None, compression=None):
            dataset = np.zeros(shape, dtype=dtype)
            self.datasets[name] = dataset
            return dataset

        def close(self):
            return None

    fake_h5py = types.ModuleType("h5py")
    fake_h5py.File = FakeH5File

    module_name = f"holemodel_test_{id(store)}"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"h5py": fake_h5py}):
        spec.loader.exec_module(module)
    return module


def pathological_input():
    return {
        "rho": np.array(
            [
                [0.8, 0.7],
                [0.0, 0.5],
                [-0.1, 0.0],
                [0.3, 1e-20],
                [1e-24, 0.2],
                [0.1, 0.09],
            ],
            dtype=float,
        ),
        "grd": np.array(
            [
                [1e-2, 2e-2, 1e-2, 0.0, 8e-3, 1e-2, 2e-2],
                [1e2, 5e1, 1e2, 0.0, 5e1, 2e1, 4e1],
                [1e5, 2e5, 1e5, 0.0, 2e5, 1e5, 3e5],
                [1e-6, 2e-6, 1e-6, 0.0, 3e-6, 1e-6, 2e-6],
                [5e3, 1e4, 8e3, 0.0, 1e4, 7e3, 9e3],
                [2e-2, 1e-2, 2e-2, 0.0, 1e-2, 1e-2, 2e-2],
            ],
            dtype=float,
        ),
        "xyz": np.array(
            [
                [0.0, 0.0, 0.0, 0.20],
                [0.1, 0.0, 0.0, 0.20],
                [0.2, 0.0, 0.0, 0.20],
                [0.3, 0.0, 0.0, 0.15],
                [0.4, 0.0, 0.0, 0.15],
                [0.5, 0.0, 0.0, 0.10],
            ],
            dtype=float,
        ),
    }


def vacuum_input():
    return {
        "rho": np.zeros((4, 2), dtype=float),
        "grd": np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0, 0.0, 0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0],
            ],
            dtype=float,
        ),
        "xyz": np.array(
            [
                [0.0, 0.0, 0.0, 0.25],
                [0.1, 0.0, 0.0, 0.25],
                [0.2, 0.0, 0.0, 0.25],
                [0.3, 0.0, 0.0, 0.25],
            ],
            dtype=float,
        ),
    }


def parse_text_report(text_path):
    metrics = {}
    for raw_line in text_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        label, value = line.split("=", 1)
        metrics[label.strip()] = float(value)
    return metrics


def get_store_entry(store, suffix):
    for key, value in store.items():
        if key.endswith(suffix):
            return value
    raise KeyError(suffix)


def run_model(module, store, fixture_name, datasets):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.abspath(os.path.join(tmpdir, f"{fixture_name}.plot"))
        store[input_path] = datasets

        captured_stdout = io.StringIO()
        cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            with contextlib.redirect_stdout(captured_stdout):
                module.DFThxcmodel(input_path)
        finally:
            os.chdir(cwd)

        text_path = Path(tmpdir) / f"XChole_energy_{fixture_name}.txt"
        text_metrics = parse_text_report(text_path)
        text_body = text_path.read_text()
        console_output = captured_stdout.getvalue()
        plot_output = get_store_entry(store, f"XCholemodel_{fixture_name}.plot")
        copied_plot_output = {name: np.array(values, copy=True) for name, values in plot_output.items()}
    return text_metrics, text_body, copied_plot_output, console_output


class HoleModelRobustnessTests(unittest.TestCase):
    def assert_metrics_close(self, actual_metrics, expected_metrics):
        for label, expected_value in expected_metrics.items():
            self.assertIn(label, actual_metrics)
            actual_value = actual_metrics[label]
            self.assertTrue(
                np.isclose(actual_value, expected_value, rtol=1e-10, atol=1e-12),
                msg=f"{label}: expected {expected_value}, got {actual_value}",
            )

    def assert_valid_plot_output(self, plot_output):
        self.assertEqual(tuple(plot_output.keys()), EXPECTED_PLOT_DATASET_NAMES)
        for dataset_name, values in plot_output.items():
            self.assertTrue(np.isfinite(values).all(), msg=dataset_name)

    def assert_progress_output(self, console_output):
        self.assertIn("XCholemodel progress", console_output)
        self.assertIn("Input file:", console_output)
        self.assertIn("Planned steps: 8", console_output)
        self.assertIn("Final summary", console_output)
        self.assertIn("Total runtime:", console_output)
        self.assertEqual(console_output.count("Done in "), 8)
        self.assertIn("Preparing exchange kernels and spin-scaled radii.", console_output)
        self.assertIn("Contracting weighted exchange profiles over the grid.", console_output)
        self.assertIn("Evaluating PW92/LDA correlation ingredients.", console_output)
        self.assertIn("Building the GGA correction and radial cutoff.", console_output)
        self.assertIn("Contracting correlation profiles over the grid.", console_output)
        self.assertNotIn("array(", console_output)
        self.assertNotIn("hx_lda finish", console_output)

        last_position = -1
        for step_line in EXPECTED_PROGRESS_STEPS:
            position = console_output.find(step_line)
            self.assertNotEqual(position, -1, msg=step_line)
            self.assertGreater(position, last_position, msg=step_line)
            last_position = position

    def test_safe_helpers_return_finite_values(self):
        module = load_holemodel_module({})
        divided = module.safe_divide(np.array([1.0, 2.0]), np.array([0.0, 4.0]))
        inverse_square = module.safe_inverse_square(np.array([0.0, 2.0]))

        self.assertTrue(np.isfinite(divided).all())
        self.assertTrue(np.isfinite(inverse_square).all())
        self.assertEqual(divided[0], 0.0)
        self.assertEqual(inverse_square[0], 0.0)
        self.assertAlmostEqual(divided[1], 0.5)
        self.assertAlmostEqual(inverse_square[1], 0.25)

    def test_regularized_gradient_is_bounded(self):
        module = load_holemodel_module({})
        values = np.array([0.0, 2.5, 25.0, 250.0])
        regularized = module.regularize_reduced_gradient(values)

        self.assertTrue(np.isfinite(regularized).all())
        self.assertAlmostEqual(regularized[1], 2.5)
        self.assertGreaterEqual(regularized[-1], module.PBE_S_REGULARIZATION_START)
        self.assertLess(regularized[-1], module.PBE_S_REGULARIZATION_LIMIT)

    def test_load_grid_data_validates_dataset_shapes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = {}
            module = load_holemodel_module(store)

            bad_xyz_path = os.path.abspath(os.path.join(tmpdir, "bad_xyz.plot"))
            store[bad_xyz_path] = {
                "rho": np.zeros((3, 2), dtype=float),
                "grd": np.zeros((3, 7), dtype=float),
                "xyz": np.zeros((2, 4), dtype=float),
            }
            with self.assertRaisesRegex(ValueError, "rho and xyz"):
                module._load_grid_data(bad_xyz_path)

            bad_grd_path = os.path.abspath(os.path.join(tmpdir, "bad_grd.plot"))
            store[bad_grd_path] = {
                "rho": np.zeros((3, 2), dtype=float),
                "grd": np.zeros((2, 7), dtype=float),
                "xyz": np.zeros((3, 4), dtype=float),
            }
            with self.assertRaisesRegex(ValueError, "grd dataset"):
                module._load_grid_data(bad_grd_path)

    def test_failure_path_reports_the_active_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = {}
            module = load_holemodel_module(store)
            bad_path = os.path.abspath(os.path.join(tmpdir, "broken.plot"))
            store[bad_path] = {
                "rho": np.zeros((3, 2), dtype=float),
                "grd": np.zeros((3, 7), dtype=float),
                "xyz": np.zeros((2, 4), dtype=float),
            }

            captured_stdout = io.StringIO()
            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with contextlib.redirect_stdout(captured_stdout):
                    with self.assertRaisesRegex(ValueError, "rho and xyz"):
                        module.DFThxcmodel(bad_path)
            finally:
                os.chdir(cwd)

            console_output = captured_stdout.getvalue()
            self.assertIn("[1/8] Load input grid data...", console_output)
            self.assertIn("FAILED after ", console_output)
            self.assertIn("during Load input grid data", console_output)

    def test_pathological_fixture_preserves_scalar_outputs(self):
        store = {}
        module = load_holemodel_module(store)
        text_metrics, text_body, plot_output, console_output = run_model(
            module,
            store,
            "pathological",
            pathological_input(),
        )

        self.assertNotIn("nan", text_body.lower())
        self.assertNotIn("inf", text_body.lower())
        self.assert_metrics_close(text_metrics, EXPECTED_PATHOLOGICAL_SCALARS)
        self.assert_valid_plot_output(plot_output)
        self.assert_progress_output(console_output)

    def test_output_schema_and_finiteness_for_fixture_outputs(self):
        fixtures = {
            "pathological": pathological_input,
            "vacuum": vacuum_input,
        }

        for fixture_name, factory in fixtures.items():
            with self.subTest(fixture=fixture_name):
                store = {}
                module = load_holemodel_module(store)
                text_metrics, text_body, plot_output, console_output = run_model(
                    module,
                    store,
                    fixture_name,
                    factory(),
                )

                self.assertTrue(all(np.isfinite(value) for value in text_metrics.values()))
                self.assertNotIn("nan", text_body.lower())
                self.assertNotIn("inf", text_body.lower())
                self.assert_valid_plot_output(plot_output)
                self.assertIn("Final summary", console_output)


if __name__ == "__main__":
    unittest.main()
