# XCholemodel

Author: Lin Hou

`XCholemodel` is a research-oriented Python script for building exchange and correlation hole models from spin-resolved density data stored in an HDF5-compatible file. It reads density, gradient, and grid-weight information, evaluates LDA and PBE hole models, and writes both human-readable energy summaries and machine-readable output curves for downstream analysis.

## Overview

The script computes:

- LDA exchange hole
- PBE exchange hole
- LDA correlation hole
- PBE correlation hole
- LDA and PBE exchange-correlation energies
- radial output curves suitable for plotting or post-processing

The core numerical path in `holemodel.py` has been vectorized with NumPy to reduce time spent in Python loops during the main model evaluation.

## Repository Contents

- `holemodel.py`: main executable script
- `requirements.txt`: Python dependencies
- `LICENSE`: project license
- `README.md`: documentation

## Installation

Create a fresh Python environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you prefer not to use a virtual environment:

```bash
pip install -r requirements.txt
```

## Requirements

The script depends on:

- `numpy`
- `scipy`
- `h5py`
- `matplotlib`
- `tqdm`

`matplotlib` is currently imported by the script even though the plotting section is commented out.

## Quick Start

Run the model from the repository directory with one input file:

```bash
python holemodel.py /path/to/input.plot
```

Example:

```bash
python holemodel.py dens_ccsd_H2.plot
```

The script expects exactly one input path from the command line. If no input path is provided, it prints an argument reminder and exits without doing useful work.

## Input File Format

The input file must be readable with `h5py.File(path, "r")` and must contain the following datasets:

### `rho`

Spin densities with shape `(ngrid, 2)`:

- `rho[:, 0]`: spin-up density
- `rho[:, 1]`: spin-down density

### `grd`

Gradient information with at least 7 columns:

- `grd[:, 0:3]`: spin-up density gradient
- `grd[:, 4:7]`: spin-down density gradient

### `xyz`

Grid data with at least 4 columns:

- `xyz[:, 3]`: integration weights

## What The Script Produces

For an input file named `sample.plot`, the script writes:

### 1. Text summary

`XChole_energy_sample.txt`

This file contains:

- LDA exchange, correlation, and exchange-correlation energies
- PBE exchange, correlation, and exchange-correlation energies
- LDA and PBE sum rules
- on-top values
- cusp estimates

### 2. HDF5-style curve output

`XCholemodel_sample.plot`

This file stores the radial axis and model results as compressed datasets, including:

- `u_axis`
- `LDA_X`
- `LDA_C`
- `LDA_XC`
- `PBE_X`
- `PBE_C`
- `PBE_XC`
- `LDA_EX`
- `LDA_EC`
- `LDA_EXC`
- `PBE_EX`
- `PBE_EC`
- `PBE_EXC`

## Console Output

During execution the script prints diagnostic information such as:

- integrated spin populations
- total density normalization
- timing for reading the density file
- exchange and correlation sum rules
- final LDA and PBE energies

This is useful for quick sanity checks when running large calculations.

## Numerical Notes

- The radial grid is built internally with `npts = 4001`.
- The radial spacing is `delta_u = 0.0125`.
- Exchange and correlation quantities are integrated on that internal grid.
- The script computes both spin-scaled exchange terms and correlation-hole quantities from the supplied density and gradient data.

## Usage Notes

- Output filenames are derived from the input filename.
- Output files are written to the current working directory.
- The script currently uses a single-file command-line interface rather than named arguments.
- The plotting code is present but commented out, so the current release focuses on numerical output rather than automatic figure generation.

## Troubleshooting

- If you see `ModuleNotFoundError` for `h5py`, `scipy`, `numpy`, `matplotlib`, or `tqdm`, reinstall the dependencies with `pip install -r requirements.txt`.
- If the script fails while reading the input file, check that the file is HDF5-compatible and that the datasets `rho`, `grd`, and `xyz` exist with the expected column layout.
- If output files are not appearing where you expect, remember that they are written to the current working directory, not necessarily next to the input file.
- If you want plots, note that the plotting section is currently commented out in `holemodel.py`.

## Limitations

This repository is ready for release as a research script, but users should be aware of the current assumptions:

- the input file format is fixed and not validated beyond expected dataset names and indexing
- there is no built-in schema checker for malformed input files
- there is no batch-processing CLI yet
- error handling is minimal if required datasets are missing
- the code is distributed as a script, not an installable Python package

## Reproducibility Tips

For cleaner runs and easier sharing:

- keep a copy of the exact input `.plot` file used for each run
- record the Python environment used to run the script
- version-control any downstream analysis notebooks separately from generated output
- avoid committing generated `XChole_energy_*.txt` and `XCholemodel_*.plot` files unless they are intended release artifacts

## Recommended Release Checklist

Before publishing a release, it is a good idea to confirm:

- `pip install -r requirements.txt` works in a fresh environment
- `python holemodel.py your_input.plot` runs successfully on a known-good input
- the generated energy summary and `.plot` output look reasonable for a reference system
- the license and README match the intended public release

## Citation And Attribution

If you use this code in research or derivative work, please credit:

Lin Hou

If you plan to publish results based on this implementation, consider citing the associated scientific work or internal project notes that define the model and input data generation procedure.

## License

This project is distributed under the terms of the license included in [LICENSE](LICENSE).
