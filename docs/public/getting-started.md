# Getting Started

Run all commands from the repository root in Windows PowerShell.

## 1. Create the Conda environment

```powershell
conda env create --prefix .\env --file environment.yml
conda activate .\env
```

For an existing environment:

```powershell
conda env update --prefix .\env --file environment.yml --prune
conda activate .\env
```

The FMU path depends on the Windows-native pyFMI/Assimulo stack. The repository-local `env/` directory is a local artifact and is not a substitute for installation from [environment.yml](../../environment.yml). See the preserved [environment audit export](environment-lock.txt).

## 2. Inspect the command line

[Main_OnlineLearningMPC.py](../../Main_OnlineLearningMPC.py) accepts:

| Option | Values | Meaning |
| --- | --- | --- |
| `--Model` | `NNARX`, `GP`, `BLL` | Select the strategy; default: `NNARX`. |
| `--Debug` | flag | Select FMU simulation explicitly; debug is already the default. |
| `--Live` | flag | Rejected before runtime construction; public live execution is disabled. |

`BLL` is the implementation name for the Bayesian neural network exploration (BNNExp) strategy.

## 3. Run the smoke test

**Smoke test:**

```powershell
python smoke_test.py
```

This imports principal modules and validates the representative GP configuration, bundled files, MAT metadata, required dependencies, and IPOPT availability. It does not run the controller loop.

## 4. Run tests

**Test:**

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

**Publication validation:**

```powershell
python tests/validate_publication.py
```

The publication command compiles Python files, runs the smoke test, and runs unit tests. It does not run an FMU trajectory.

The two `validate_*_baseline.py` scripts under `tests/` require baseline and asset-manifest JSON files that are not present. Do not use them as current validation commands.

## 5. Run a debug simulation

**Debug simulation:**

```powershell
python Main_OnlineLearningMPC.py --Debug --Model NNARX
```

Use `GP` or `BLL` in place of `NNARX` for another strategy. The runner loads `FMUs/DHN_sim.fmu`, `NNmodels/NARX_net.mat`, and the two CSV files under `src/utils/data/`. It writes generated outputs under `Solutions/<Model>/<ddmm>/` and exceptions under `LogsFolder/`.

A completed run exercises the implementation and FMU integration. It does not establish agreement with an article, physical validity, or numerical repeatability on another environment.

## 6. Reproducibility boundary

The current tree supports source inspection, smoke testing, unit testing, and Windows FMU debug simulation. Exact article reproduction is not established because article-specific configuration, provenance, hashes, baseline files, and approved tolerances are missing.

The public entry point does not support plant-connected operation. An explicit `--Live` request raises an error before connector construction.

## Troubleshooting

- For import failures, confirm that `.\env` is active and update it from `environment.yml`.
- For preflight failures, verify `NNmodels/NARX_net.mat`, `FMUs/DHN_sim.fmu`, and both CSV inputs.
- For FMU failures, verify the Windows pyFMI/Assimulo installation.
- For missing outputs, check write permissions and `LogsFolder/Exceptions.log`.

See [execution modes](execution-modes.md) for claim boundaries and [architecture](architecture.md) for runtime flow.
