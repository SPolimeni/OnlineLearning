# OnlineLearning MPC

## Overview

OnlineLearning is a research-oriented model-predictive control (MPC) application for a thermal or district-heating plant. It predicts plant behavior over a receding horizon, solves a constrained optimization problem, applies the first control move, and repeats the cycle.

## Scientific purpose

The controller evaluates online-learning model strategies for thermal control. The repository contains an offline/debug route based on a functional mock-up unit (FMU) and a live route that reads plant measurements and writes setpoints through deployment-specific services.

## Methodology

At each control step, the runtime reads the current state, optionally updates the Bayesian model state, solves the MPC problem with CasADi and IPOPT, records predictions and diagnostics, and applies the first control action. The horizon uses the configured prediction length and move blocking. The parameter values, equations, constraints, units, timing, and model assets are scientific inputs and must not be inferred from this guide alone; see [public architecture documentation](docs/public/architecture.md) and the source configuration.

## Repository architecture

- [Main_OnlineLearningMPC.py](Main_OnlineLearningMPC.py) is the command-line entry point.
- [src/config/parameters.py](src/config/parameters.py) builds the parameter dictionary and loads bundled CSV inputs.
- [src/runtime/controller_runner.py](src/runtime/controller_runner.py) owns the control loop, debug/live boundary, I/O, persistence, and exception logging.
- [src/class_MPC.py](src/class_MPC.py) builds and solves the optimization problem.
- [src/model_strategies.py](src/model_strategies.py) implements model-specific behavior.
- [NNmodels](NNmodels) contains MAT model files; [FMUs](FMUs) contains debug FMUs.
- `Solutions/` and `LogsFolder/` are local-only output locations and are excluded from the public repository.

For the detailed data and control flow, see the [public architecture documentation](docs/public/architecture.md).

## Supported model strategies

The `--Model` option accepts `NNARX`, `GP`, or `BLL`; the default is `GP`.

- `NNARX` uses the neural-network autoregressive prediction strategy.
- `GP` adds a Gaussian-process correction based on buffered residuals.
- `BLL` updates a Bayesian last layer and includes uncertainty-aware exploration behavior.

## Requirements

The declared environment requires Conda, Windows-oriented FMU support, Python 3.10, NumPy, SciPy, pandas, CasADi, PyTorch, GPyTorch, APScheduler, pyFMI, InfluxDB client support, and OPC UA support. Exact core package declarations are in [environment.yml](environment.yml). `openpyxl` is optional and is used only by the private Excel-to-JSON metadata conversion utility. FMU execution depends on native pyFMI/Assimulo components and is not claimed to be platform independent.

## Installation

Run these commands from the repository root in Windows PowerShell:

```powershell
conda env create --prefix .\env --file environment.yml
conda activate .\env
```

For an existing environment:

```powershell
conda env update --prefix .\env --file environment.yml --prune
```

Select `env\python.exe` in VS Code when using the repository-local environment. The checked-in `env/` directory is a local environment artifact, not a substitute for a clean installation or a record of redistribution rights.

## Quick start

From the repository root, run the import smoke test:

```powershell
python smoke_test.py
```

To select a strategy, pass `--Model NNARX`, `--Model GP`, or `--Model BLL`. Debug mode is the default; `--Debug` may be supplied explicitly.

The terms **test**, **debug**, and **live** refer to distinct paths. See the
[execution-mode reference](docs/public/execution-modes.md) for the authoritative
commands, external-I/O boundaries, and validation claims for each case.

## Debug simulation

The debug command is an FMU simulation and does not intentionally use the live InfluxDB or OPC UA path:

```powershell
python Main_OnlineLearningMPC.py --Model GP
```

The runner loads `FMUs/DHN_sim.fmu`, seeds startup buffers, executes the controller loop, and feeds generated controls back to the FMU. pyFMI support is Windows-dependent in the current implementation. This command is a debug simulation, not proof of numerical reproduction of published results; implementation-level asset hashes and reduced strategy baselines are documented under `docs/public/`.

## Live integration boundary

The retained live integration can be inspected in the source, but public live execution is disabled. An explicit live request is rejected:

```powershell
python Main_OnlineLearningMPC.py --Live --Model GP
```

The command raises an error stating that live integration is illustrative and read-only and is not enabled for public execution. The source retains the InfluxDB and OPC UA integration boundary to document the applied architecture, but it does not permit public connections or writes.

## Input data and trained models

The parameter builder loads `src/utils/data/DayAhead_ZonalPrices.csv` and `src/utils/data/DHN_ground_truth.csv`. The default neural model is `NNmodels/NARX_net.mat`; additional MAT files and model directories are present in `NNmodels/`. The selected model, input ordering, normalization, units, and asset provenance must be recorded with any scientific run. Redistribution rights and checksums for the data, MAT files, and FMUs have not been established in this repository.

## Outputs and result files

Local runs append CSV and pickle files below `Solutions/<Model>/<ddmm>/`, including measured and predicted outputs, recurrent predictions, control actions, boiler predictions, slacks, exploration, cost, computation time, feasibility status, configuration, and Bayesian `theta`/`sigma` state. Exceptions are appended to `LogsFolder/Exceptions.log`. These generated locations are excluded from the public repository.

## Reproducing the published results

Three reproducibility levels are currently distinguishable:

1. **Source and import inspection:** read the documented architecture and inspect source, models, FMUs, and bundled inputs. This requires no runtime integration.
2. **Smoke test:** run `python smoke_test.py` in the declared environment. This checks imports and project wiring; it does not validate optimization results or FMU trajectories.
3. **FMU debug simulation:** run the `--Debug` command on a supported Windows environment with the required FMU and dependencies. This exercises the controller and simulation path; implementation-level fixtures, asset provenance, environment capture, and reduced strategy baselines are recorded, but exact article reproduction remains unverified.

Live plant execution is not a reproducibility level. It requires private infrastructure and must be validated only in an approved isolated deployment.

## Tests and validation

Run the unit tests separately from the smoke test:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

The smoke test validates imports, the representative GP configuration, and the
example FMU path; the unit tests cover the assertions currently encoded under
`tests/`. Run compilation, smoke validation, and all unit tests with:

```powershell
python tests/validate_publication.py
```

Validate the separate reduced GP/FMU numerical baseline with:

```powershell
python tests/validate_fmu_baseline.py
```

Validate the reduced trajectory and strategy comparison baseline with:

```powershell
python tests/validate_strategy_baseline.py
```

These are implementation-level regressions, not article-level validation.
They do not certify live operation, arbitrary horizons, or exact article
reproduction.

## Known limitations

The application is plant-specific, uses source-controlled runtime parameters from the central configuration builder, and depends on mutable model state for GP and BLL. The live path is deployment-specific. See the [operational guide](docs/public/operations.md) for the authorized deployment procedure, output meanings, failure diagnosis, and strong plant/model assumptions. Licensing, provenance, redistribution rights, and anonymization status for several artifacts remain unresolved.

## Security and configuration notes

Never commit credentials or private endpoints. Review live connector configuration, node maps, logs, generated solutions, model files, FMUs, and bundled data before redistribution. Debug execution should be used for local validation; do not run the live command against an unknown or unapproved plant.

## Citation

Citation metadata, authorship, DOI, and a `CITATION.cff` file are not yet provided. Add owner-approved bibliographic metadata before public release.

## License

No repository license has been identified. The project owner must select and publish a license, and separately confirm redistribution rights for models, FMUs, data, and generated results.

## Contributing

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md). Contributions that affect scientific behavior require review of timing, units, equations, constraints, model ordering, solver behavior, and reproducibility evidence.

## Contact

No public maintainer or contact address is specified in the repository. Add an owner-approved contact before publication.
