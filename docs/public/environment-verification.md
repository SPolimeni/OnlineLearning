# Environment Verification

Verified locally on 2026-10-14 in Windows PowerShell with the repository-local Conda environment at `./env`.

## Required workflow

Create the environment from the repository root:

```powershell
conda env create --prefix .\env --file environment.yml
conda activate .\env
```

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
