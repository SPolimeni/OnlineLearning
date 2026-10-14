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
- [src/runtime/controller_runner.py](src/runtime/controller_runner.py) owns the control loop, debug/live selection, I/O, persistence, and exception logging.
- [src/class_MPC.py](src/class_MPC.py) builds and solves the optimization problem.
- [src/model_strategies.py](src/model_strategies.py) implements model-specific behavior.
- [NNmodels](NNmodels) contains MAT model files; [FMUs](FMUs) contains debug FMUs.
- [Solutions](Solutions) receives run outputs. `LogsFolder/` is created for exception logs.

For the detailed data and control flow, see the [public architecture documentation](docs/public/architecture.md).

## Supported model strategies

The `--Model` option accepts `NNARX`, `GP`, or `BNNExp`; the default is `GP`.

- `NNARX` uses the neural-network autoregressive prediction strategy.
- `GP` adds a Gaussian-process correction based on buffered residuals.
- `BNNExp` updates a Bayesian last layer and includes uncertainty-aware exploration behavior.

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

To select a strategy, pass `--Model NNARX`, `--Model GP`, or `--Model BNNExp`. Use `--Debug` for the FMU route described below.

## Debug simulation

The debug command is an FMU simulation and does not intentionally use the live InfluxDB or OPC UA path:

```powershell
python Main_OnlineLearningMPC.py --Debug --Model GP
```

The runner loads `FMUs/DHN_MC_V3_FMU_EB_60.fmu`, seeds startup buffers, executes the controller loop, and feeds generated controls back to the FMU. pyFMI support is Windows-dependent in the current implementation. This command is a debug simulation, not proof of numerical reproduction of published results; no fixed trajectory baseline, FMU hash, or full provenance record is currently documented.

## Live plant execution

Without `--Debug`, the process enters the live runtime path and remains active until interrupted:

```powershell
python Main_OnlineLearningMPC.py --Model GP
```

The live path reads status through InfluxDB and writes control setpoints through OPC UA. It requires deployment-specific credentials, endpoints, node mappings, network access, and plant approval. Do not publish credentials, tokens, internal hostnames, certificates, node identifiers, or other infrastructure metadata. No public installation can be considered live-ready without an approved deployment configuration.

## Input data and trained models

The parameter builder loads bundled CSV files from `src/utils/data/`, including electricity-price and heat-load data. The default neural model is `NNmodels/net_1020.mat`; additional MAT files and model directories are present in `NNmodels/`. The selected model, input ordering, normalization, units, and asset provenance must be recorded with any scientific run. Redistribution rights and checksums for the data, MAT files, and FMUs have not been established in this repository.

## Outputs and result files

Runs append CSV and pickle files below `Solutions/<Model>/<ddmm>/`, including measured and predicted outputs, recurrent predictions, control actions, boiler predictions, slacks, exploration, cost, computation time, feasibility status, configuration, and Bayesian `theta`/`sigma` state. The run timestamp is included in filenames. Exceptions are appended to `LogsFolder/Exceptions.log`. These repository locations are resolved from the source tree; launch commands from the repository root are still recommended because the entry-point import path is configured there.

## Reproducing the published results

Three reproducibility levels are currently distinguishable:

1. **Source and import inspection:** read the documented architecture and inspect source, models, FMUs, and bundled inputs. This requires no runtime integration.
2. **Smoke test:** run `python smoke_test.py` in the declared environment. This checks imports and project wiring; it does not validate optimization results or FMU trajectories.
3. **FMU debug simulation:** run the `--Debug` command on a supported Windows environment with the required FMU and dependencies. This exercises the controller and simulation path, but exact article reproduction remains unverified because fixed fixtures, asset provenance, environment capture, and expected numerical baselines are missing.

Live plant execution is not a reproducibility level. It requires private infrastructure and must be validated only in an approved isolated deployment.

## Tests and validation

The available `smoke_test.py` is an import check. The repository does not currently establish tests for optimization feasibility, first control moves, fallback status, strategy-specific behavior, Bayesian updates, exploration, or FMU trajectories. A publication release should add owner-approved fixtures, tolerances, environment records, asset hashes, and representative debug baselines before claiming article-level reproduction.

## Known limitations

The application is plant-specific, uses hard-coded runtime parameters, and depends on mutable model state for GP and BNNExp. The live path is deployment-specific. Licensing, provenance, redistribution rights, and anonymization status for several artifacts remain unresolved.

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
