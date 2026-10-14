# Architecture Overview

This document explains the repository structure and runtime architecture as it appears in the current source code.

## System Overview

The system is organized around a model-predictive control loop. At a high level, it reads the current plant or simulation state, updates internal prediction and control state, solves a constrained optimization problem, applies the first control move, and repeats the cycle.

The code is split into three layers: a thin command-line entry point, a runtime layer that owns the control loop and plant/FMU I/O, and a controller core that builds and solves the optimization problem.

The most important files are:

- [Main_OnlineLearningMPC.py](../../Main_OnlineLearningMPC.py): command-line entry point; parses options and starts the runtime
- [src/config/parameters.py](../../src/config/parameters.py): builds the controller parameter dictionary (`build_parameters`)
- [src/runtime/controller_runner.py](../../src/runtime/controller_runner.py): `ControllerRunner`, which owns the control loop, buffers, plant/FMU I/O and error logging
- [src/class_MPC.py](../../src/class_MPC.py): `MPC`, the core optimization and prediction implementation
- [src/model_strategies.py](../../src/model_strategies.py): per-model strategies (NNARX, GP, BNNExp)
- [src/project_paths.py](../../src/project_paths.py): repository-stable paths to bundled input data
- [src/utils](../../src/utils): support code for data saving, state reading (InfluxDB) and setpoint writing (OPC UA)
- [NNmodels](../../NNmodels): trained neural-network model files (`.mat`)
- [FMUs](../../FMUs): simulation models used in the debug path

## Execution Flow

The execution flow begins in [Main_OnlineLearningMPC.py](../../Main_OnlineLearningMPC.py), which adds `src/` to the import path, parses `--Model {NNARX,GP,BNNExp}` (default `GP`) and `--Debug`, calls `build_parameters(model)`, creates a `ControllerRunner(parameters, debug=...)`, and calls `run()`.

`ControllerRunner.run()` selects one of two execution paths:

- **debug mode** (`--Debug`): loads the FMU, seeds buffers with the warm-start trajectory, and calls `solve_step()` in a plain loop
- **live mode**: configures the OPC UA setpoint writer and starts scheduled InfluxDB reads and MPC solves; the process runs until interrupted

Model, FMU, output, and log paths are resolved through [src/project_paths.py](../../src/project_paths.py) from the repository tree. The entry point is intended to be launched from the repository root because it adds the relative `src/` directory to the import path. CSV input data is also resolved through `project_paths.data_file`.

## Configuration

All controller parameters are produced by `build_parameters(model)` in [src/config/parameters.py](../../src/config/parameters.py) as a single dictionary. It contains problem dimensions and timing, operational limits, economic and tuning parameters, the selected model and NNARX `.mat` path, and time series loaded from [src/utils/data](../../src/utils/data).

`MPC._assign_parameters` copies every entry onto the controller as an attribute, so parameter names are used directly as `self.<name>` inside the controller core.

## Control Step

Each call to `ControllerRunner.solve_step()` performs one receding-horizon iteration:

1. **Read state.** `read_status()` fills `y_dict` with supply temperature, return temperature and flow. Live mode reads InfluxDB; debug mode uses startup fixtures and then FMU outputs.
2. **Bayesian update.** For BNNExp after `T_C0`, `MPC.bayesian_correction` updates the last-layer weights stored in `theta_history`.
3. **Optimize.** `MPC.mpc_controller(..., flag=1)` solves with exploration enabled. If no exploration slack is active, it re-solves with `flag=0`; the chosen flag is stored in `exploration`.
4. **Store results.** The first control move, slacks, predictions, boiler powers, uncertainty, and step cost are stored in buffers.
5. **Persist and apply.** `SaveAggregatedData` appends the current step to CSV files; live mode sends setpoints through OPC UA and debug mode advances the FMU.

Exceptions are caught, printed, and appended to `LogsFolder/Exceptions.log` by `_report_exception`, so a single failing step does not stop the loop.

## MPC Workflow

`MPC.mpc_controller` in [src/class_MPC.py](../../src/class_MPC.py) is the core of the control loop. Before `T_C0` it reuses the previous nominal action without solving. Otherwise it creates a CasADi `Opti` problem, adds variables, constraints and cost, warm-starts from the previous feasible trajectory, and solves with IPOPT. On failure it retries with relaxed tolerances and otherwise falls back to the last feasible trajectory. The outcome is reported as `sol_feas`: `1` solved, `2` solved on the relaxed retry, `3` fallback, and `0` during startup. The horizon has `N` steps, while the control sequence has `N / Nb` free moves (move blocking).

## Model Strategies

Behavior dependent on `Param["Model"]` is delegated to a strategy object created in `MPC._assign_parameters` from `_STRATEGY_FACTORY`. The base `ModelStrategy` implements NNARX behavior; other strategies override model-specific hooks.

- **NNARXStrategy:** neural autoregressive predictions with a temperature slack in the cost.
- **GPStrategy:** a Gaussian-process correction on top of the NNARX prediction, trained online from buffered residuals.
- **BNNExpStrategy:** a Bayesian last layer using updated `theta`, uncertainty-aware output constraints, exploration slacks, and `bayesian_correction`.

## FMU Interactions

In debug mode, `ControllerRunner` loads the FMU with pyFMI, which is Windows-dependent in the current implementation. It builds the input table from `u_out`, sets initial outputs, simulates from `T_C0` to the current step, reads `Ts_L4`, `T_return`, and `mf_supply`, writes them into the next `y_out` column, and resets the FMU for the next call.

This is an offline simulated-plant path. It does not validate live integrations or establish exact article-result reproduction.

## Neural-Network Interactions

The controller loads the MAT model named in `nnarx_mat` from [NNmodels](../../NNmodels) and extracts last-layer weights and output scaling used to denormalize predictions. `MPC.model_rnn`, `simulate_dynamics`, and `activation_fun` roll the network forward over the horizon inside the CasADi problem; the active strategy selects weights and post-processes outputs.

## Component Responsibilities

- **Entry point:** [Main_OnlineLearningMPC.py](../../Main_OnlineLearningMPC.py) parses options and starts the runner.
- **Configuration:** [src/config](../../src/config) builds parameters and loads input time series.
- **Runtime:** [src/runtime](../../src/runtime) allocates buffers, selects debug/live execution, handles I/O, invokes optimization and updates, and logs exceptions.
- **Controller core:** [src/class_MPC.py](../../src/class_MPC.py) loads model data, builds variables/constraints/cost, solves with IPOPT, and rolls out the model.
- **Model strategies:** [src/model_strategies.py](../../src/model_strategies.py) contains NNARX/GP/BNNExp-specific behavior.
- **Utilities:** [src/utils](../../src/utils) contains InfluxDB reads, OPC UA writes, result persistence, and bundled input data.

## Data Flow

1. Plant measurements are read from InfluxDB (live) or the FMU (debug).
2. The runner stores state in `y_out`.
3. BNNExp optionally updates its Bayesian last layer.
4. The active model strategy predicts future behavior.
5. The optimization routine computes feasible control moves.
6. The first action is stored and sent to the plant or FMU.
7. Results are appended under `Solutions/<Model>/<ddmm>/`; exceptions go to `LogsFolder/Exceptions.log`.
