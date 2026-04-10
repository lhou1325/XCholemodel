# XCholemodel

Author: Lin Hou

`XCholemodel` computes exchange and correlation hole model data from an input density file and writes both text summaries and HDF5-style output for later analysis.

## Files

- `holemodel.py`: main script
- `LICENSE`: repository license

## Requirements

Install the Python packages used by the script:

```bash
pip install numpy scipy h5py matplotlib tqdm
```

## Input Format

The script reads one HDF5-compatible input file. The file is expected to contain these datasets:

- `rho`: density array with two columns
  - `rho[:, 0]`: spin-up density
  - `rho[:, 1]`: spin-down density
- `grd`: gradient array with at least 7 columns
  - `grd[:, 0:3]`: spin-up density gradient
  - `grd[:, 4:7]`: spin-down density gradient
- `xyz`: grid information with at least 4 columns
  - `xyz[:, 3]`: integration weights

## How To Run

Run the script from the repository directory:

```bash
python holemodel.py /path/to/input.plot
```

Example:

```bash
python holemodel.py dens_ccsd_H2.plot
```

If no input file is provided, the script prints a message asking for a command-line argument.

## Output Files

For an input file named `sample.plot`, the script writes:

- `XChole_energy_sample.txt`
  - text summary of LDA and PBE exchange/correlation energies
  - sum rules and cusp-related values
- `XCholemodel_sample.plot`
  - HDF5-style output containing the computed radial axis and model curves
  - includes datasets such as `LDA_X`, `LDA_C`, `LDA_XC`, `PBE_X`, `PBE_C`, `PBE_XC`, and the corresponding integrated energies

## What The Script Computes

The model evaluates:

- LDA exchange hole
- PBE exchange hole
- LDA correlation hole
- PBE correlation hole
- integrated exchange-correlation energies based on those hole models

## Notes

- The script currently uses the first command-line argument as the input path.
- Output filenames are generated from the input filename.
- The plotting section in `holemodel.py` is currently commented out, so the main outputs are the text summary and the `.plot` data file.
