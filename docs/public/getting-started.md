# Getting Started

This guide describes installation and the two runtime modes currently implemented. Run commands from the repository root because the entry point adds the relative `src/` directory to the import path. Model, FMU, result, and log paths are resolved from the repository tree by [src/project_paths.py](../../src/project_paths.py).

## 1. Install the environment

Use Windows PowerShell and Conda. The declared environment is pinned to Python 3.10 and includes CasADi/IPOPT support, numerical libraries, PyTorch/GPyTorch, APScheduler, pyFMI, InfluxDB client support, and OPC UA support.

`openpyxl` is optional. Install it only when using the private `src/utils/InfluxDB/DBVarListToJson.py` utility to convert a private Excel metadata workbook into a JSON registry. It is not required for the public controller, smoke test, or debug workflow.

```powershell
conda env create --prefix .\env --file environment.yml
conda activate .\env
```

Keep this Conda environment active for every import, smoke-test, and controller command. In particular, the FMU path requires the Windows-native pyFMI and Assimulo libraries supplied by this environment. An equivalent non-interactive form is `conda run --prefix .\env python <command>`.

The verified local workflow and results are recorded in [environment-verification.md](environment-verification.md). The distinction between source/import reproducibility, implementation-level debug simulation, and exact article-result reproduction is recorded in [reproducibility-matrix.md](reproducibility-matrix.md).

Update an existing environment with:

```powershell
conda env update --prefix .\env --file environment.yml --prune
```

FMU execution requires the native pyFMI/Assimulo stack and is Windows-dependent in the current implementation. In VS Code, select `env\python.exe` as the interpreter. The local `env/` directory does not establish public redistribution rights or a reproducible environment by itself.

## 2. Inspect the command line

The entry point is [Main_OnlineLearningMPC.py](../../Main_OnlineLearningMPC.py). It builds parameters through [src/config/parameters.py](../../src/config/parameters.py) and accepts:

| Option | Values | Meaning |
| --- | --- | --- |
| `--Model` | `NNARX`, `GP`, `BNNExp` | Selects the model strategy; default: `GP`. |
| `--Debug` | flag | Uses the FMU simulation path instead of live plant I/O. |

The runtime parameter dictionary includes dimensions, timing, operating limits, the default MAT model path, and bundled electricity-price and heat-load arrays. These values are source configuration, not user-editable deployment settings exposed by a configuration file.

## 3. Run the smoke test

This is an import and wiring check. It does not start the controller, load the FMU, contact InfluxDB or OPC UA, or establish numerical reproducibility.

```powershell
python smoke_test.py
```

## 4. Run a debug simulation

Use the debug flag for local controller work:

```powershell
python Main_OnlineLearningMPC.py --Debug --Model GP
```

The runner loads `FMUs/DHN_MC_V3_FMU_EB_60.fmu`, initializes startup buffers, solves successive MPC steps, and sends generated controls back to the FMU. Use `NNARX` or `BNNExp` in place of `GP` to exercise the other strategy implementations. Debug execution is not a live-plant test and should not be described as exact article reproduction without a fixed baseline, asset hash, environment record, and owner-approved tolerances.

## 5. Understand the live path

Omitting `--Debug` selects live mode:

```powershell
python Main_OnlineLearningMPC.py --Model GP
```

The runtime schedules status reads through InfluxDB and writes setpoints through OPC UA. Live operation requires deployment-specific credentials, endpoints, node mappings, network access, and plant authorization. These values must remain private and must not be added to public documentation or source control. Do not run this command unless the deployment owner has supplied and approved the complete configuration.

## 6. Locate inputs and outputs

Bundled CSV inputs are loaded from `src/utils/data/`. The default neural model is `NNmodels/net_1020.mat`; FMU files are under `FMUs/`. During execution, `Solutions/<Model>/<ddmm>/` receives timestamped CSV files for measurements, predictions, controls, boiler predictions, slacks, exploration, costs, computation time, feasibility, and configuration, plus a `ThetaAndSigma` pickle. Runtime exceptions are appended to `LogsFolder/Exceptions.log`.

## 7. Reproducibility levels

1. **Inspection:** verify source structure, parameters, bundled input paths, model assets, and architecture without executing plant integrations.
2. **Smoke test:** run `python smoke_test.py` in the declared Conda environment. This verifies imports only.
3. **FMU debug simulation:** run the Windows debug command with the required FMU and dependencies. This exercises the offline control path, but exact published-result reproduction is not currently verified.

Live execution is a deployment activity, not a reproducibility level. The repository does not yet provide fixed numerical fixtures, expected trajectories, complete asset provenance, or a publication-ready dependency and license record.

## Troubleshooting

For import failures, confirm that the intended Conda environment is active and update it from [environment.yml](../../environment.yml). For FMU failures, confirm the FMU exists and the Windows pyFMI dependencies are available. For missing outputs, check write permissions and inspect `LogsFolder/Exceptions.log`.

Do not troubleshoot live failures by publishing credentials, internal endpoints, node identifiers, or plant data. Live connector validation belongs in an approved isolated deployment.

## Further documentation

See the repository [README](../../README.md) for publication-readiness limitations and [architecture.md](architecture.md) for the control loop, MPC core, strategy behavior, FMU path, live path, and data flow.
