# Environment Verification

Verified locally on 2026-11-24 in Windows PowerShell with the repository-local Conda environment at `./env`.

## Required workflow

Create the environment from the repository root:

```powershell
conda env create --prefix .\env --file environment.yml
conda activate .\env
```

The exact package-version audit export captured from the verified environment is
[environment-lock.txt](../../environment-lock.txt). It is used to compare an
environment created from `environment.yml`; the supported installation
mechanism remains Conda because pyFMI and Assimulo require native binaries.

For an existing environment:

```powershell
conda env update --prefix .\env --file environment.yml --prune
conda activate .\env
```

Keep the environment active for all imports, tests, smoke checks, and controller commands. The FMU path requires the Windows-native pyFMI/Assimulo stack installed in this environment. A non-interactive equivalent is:

```powershell
conda run --prefix .\env python <command>
```

## Verified results

- Python: 3.10.20
- `pip check`: passed with no broken requirements.
- Core package imports: NumPy, SciPy, pandas, CasADi, PyTorch, GPyTorch, pyFMI, InfluxDB client, OPC UA, APScheduler, and scikit-learn imported.
- Project module imports: passed using `./env/python.exe`.
- `python smoke_test.py`: passed using `./env/python.exe`.
- Offline invariant tests: passed through the standard-library unittest entry point using `./env/python.exe tests/test_offline_invariants.py`.
- Python compilation: passed for the project source and tests.

`openpyxl` is intentionally optional and is not part of the core environment; it is used only by the private Excel metadata conversion utility.

This record verifies environment setup and imports only. It does not validate FMU trajectories, numerical article reproduction, or live plant integrations.

## Validation update: 2026-11-24

The publication validation was rerun in the repository-local environment:

- `python tests/validate_publication.py`: passed Python compilation, the extended
  smoke test, and all 14 unit tests.
- `python tests/validate_fmu_baseline.py`: passed the versioned asset hashes and the
  reduced GP baseline for solver status, first optimized control, one controlled
  FMU interval, and cost.
- `python tests/validate_strategy_baseline.py`: passed the versioned asset hashes,
  recorded package versions, and reduced NNARX, GP, and BLL trajectory baselines.

The FMU baseline is an implementation-level regression only. It does not
validate a complete trajectory, other model strategies, exact article results,
or a live plant integration.
