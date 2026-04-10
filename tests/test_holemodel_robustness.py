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

    fake_matplotlib = types.ModuleType("matplotlib")
    fake_pyplot = types.ModuleType("matplotlib.pyplot")
    fake_matplotlib.pyplot = fake_pyplot

    fake_mpl_toolkits = types.ModuleType("mpl_toolkits")
    fake_mplot3d = types.ModuleType("mpl_toolkits.mplot3d")
    fake_mplot3d.Axes3D = object
    fake_mpl_toolkits.mplot3d = fake_mplot3d

    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.trange = range

    module_name = f"holemodel_test_{id(store)}"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "h5py": fake_h5py,
            "matplotlib": fake_matplotlib,
            "matplotlib.pyplot": fake_pyplot,
            "mpl_toolkits": fake_mpl_toolkits,
            "mpl_toolkits.mplot3d": fake_mplot3d,
            "tqdm": fake_tqdm,
        },
    ):
        spec.loader.exec_module(module)
    return module


class HoleModelRobustnessTests(unittest.TestCase):
    @staticmethod
    def _get_store_entry(store, suffix):
        for key, value in store.items():
            if key.endswith(suffix):
                return value
        raise KeyError(suffix)

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

    def test_dfthxcmodel_pathological_input_outputs_are_finite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = {}
            module = load_holemodel_module(store)
            input_path = os.path.abspath(os.path.join(tmpdir, "pathological.plot"))
            store[input_path] = {
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

            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with contextlib.redirect_stdout(io.StringIO()):
                    module.DFThxcmodel(input_path)
            finally:
                os.chdir(cwd)

            text_output = Path(tmpdir) / "XChole_energy_pathological.txt"
            self.assertTrue(text_output.exists())
            text = text_output.read_text().lower()
            self.assertNotIn("nan", text)
            self.assertNotIn("inf", text)

            plot_output = self._get_store_entry(store, "XCholemodel_pathological.plot")
            for dataset in plot_output.values():
                self.assertTrue(np.isfinite(dataset).all())

    def test_dfthxcmodel_vacuum_input_outputs_are_finite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = {}
            module = load_holemodel_module(store)
            input_path = os.path.abspath(os.path.join(tmpdir, "vacuum.plot"))
            store[input_path] = {
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

            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with contextlib.redirect_stdout(io.StringIO()):
                    module.DFThxcmodel(input_path)
            finally:
                os.chdir(cwd)

            plot_output = self._get_store_entry(store, "XCholemodel_vacuum.plot")
            for dataset in plot_output.values():
                self.assertTrue(np.isfinite(dataset).all())


if __name__ == "__main__":
    unittest.main()
