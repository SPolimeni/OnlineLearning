# Architecture Overview

This document explains the repository structure and the runtime architecture as it appears in the current source code.

## System Overview

The system is organized around a model-predictive control loop. At a high level, it does the following:

1. reads the current plant or simulation state
2. updates internal prediction and control state
3. solves a constrained optimization problem
4. applies the first control move
5. repeats the cycle

The code is split into three layers: a thin command-line entry point, a runtime layer that owns the control loop and the plant/FMU I/O, and a controller core that builds and solves the optimization problem.

The most important files are:

- [Main_OnlineLearningMPC.py](../Main_OnlineLearningMPC.py): command-line entry point; parses options and starts the runtime
- [src/config/parameters.py](../src/config/parameters.py): builds the controller parameter dictionary (`build_parameters`)
- [src/runtime/controller_runner.py](../src/runtime/controller_runner.py): `ControllerRunner`, which owns the control loop, buffers, plant/FMU I/O and error logging
- [src/class_MPC.py](../src/class_MPC.py): `MPC`, the core optimization and prediction implementation
- [src/model_strategies.py](../src/model_strategies.py): per-model strategies (NNARX, GP, BNNExp)
- [src/project_paths.py](../src/project_paths.py): repository-stable paths to the bundled input data
- [src/utils](../src/utils): support code for data saving, state reading (InfluxDB) and setpoint writing (OPC UA)
- [NNmodels](../NNmodels): trained neural-network model files (`.mat`)
- [FMUs](../FMUs): simulation models used in the debug path

## Execution Flow

The execution flow begins in [Main_OnlineLearningMPC.py](../Main_OnlineLearningMPC.py), which only:

- adds `src/` to the import path, so the script runs from the repository root without installation
- parses the command-line arguments `--Model {NNARX,GP,BNNExp}` (default `GP`) and `--Debug`
- calls `build_parameters(model)` from [src/config/parameters.py](../src/config/parameters.py)
- creates a `ControllerRunner(parameters, debug=...)` and calls `run()`

`ControllerRunner.__init__` creates the `MPC` instance, sets the timing metadata (`total_time`, `time_step`, `num_steps`), allocates the history buffers via `initialize_buffers` and stamps the run with `InitialDate`.

`ControllerRunner.run()` then selects one of two execution paths:

- **debug mode** (`--Debug`): loads the FMU (`_configure_fmu`), seeds the buffers with the warm-start trajectory, and calls `solve_step()` in a plain loop for `num_steps - 1` iterations
- **live mode**: configures the OPC UA setpoint writer and its node map (`_configure_live_plant`), then starts an APScheduler `BackgroundScheduler` with two jobs: `read_status` every 10 s and `solve_step` every `Ts` seconds. The process runs until interrupted with Ctrl+C.

Model and FMU paths (`NNmodels/net_1020.mat`, `FMUs/DHN_MC_V3_FMU_EB_60.fmu`) and the output folders (`Solutions/`, `LogsFolder/`) are resolved relative to the working directory, so the script is expected to be launched from the repository root. The CSV input data is resolved through `project_paths.data_file` and does not depend on the working directory.

## Configuration

All controller parameters are produced by `build_parameters(model)` in [src/config/parameters.py](../src/config/parameters.py) as a single dictionary. It contains:

- problem dimensions and timing: `N` (horizon), `Nb` (control block size), `Ts`, `tot_hours`, `n_out`, `n_inp`, `n_x`, `n_slack`, `T_C0` (step at which the optimization becomes active)
- operational limits: supply/return temperatures, flow, boiler powers
- economic and tuning parameters: gas price, COP/efficiencies, slack and exploration weights, Bayesian prior parameters (`beta0_*`, `epsilon0`, `L_prev0`, `sigma_epsilon0_2`)
- the selected `Model` and the path to the NNARX `.mat` file
- time series loaded from [src/utils/data](../src/utils/data): zonal electricity prices (`c_el`) and the heat-load profiles (`Potenza`)

`MPC._assign_parameters` copies every entry onto the controller as an attribute, so parameter names are used directly as `self.<name>` inside the controller core.

## Control Step

Each call to `ControllerRunner.solve_step()` performs one receding-horizon iteration:

1. **read state** — `read_status()` fills `y_dict` with supply temperature, return temperature and flow. In live mode it calls `main_status_readings()` (InfluxDB); in debug mode it uses a fixed fixture for the first `T_C0` steps and the FMU outputs afterwards. The values are written into `y_out`.
2. **Bayesian update** (BNNExp only, after `T_C0`) — `MPC.bayesian_correction` updates the last-layer weights `theta`, stored in `theta_history`.
3. **optimize** — `MPC.mpc_controller(..., flag=1)` solves the problem with exploration enabled. If none of the exploration slacks is active, the problem is re-solved with `flag=0` (exploration off). The chosen flag is stored in `exploration`. For NNARX and GP the exploration slack is always 0, so the first solve is kept.
4. **store results** — the first control move goes into `u_out`; slacks, predictions, boiler powers, uncertainty (`sigma`) and the step cost are stored in their buffers.
5. **persist and apply** — `SaveAggregatedData` appends the current step to CSV files; then setpoints are sent to the plant via OPC UA (live mode, every `T_C0 - 1` steps) or the control history is simulated on the FMU (debug mode, from `T_C0` on).

Any exception in these phases is caught, printed and appended to `LogsFolder/Exceptions.log` by `_report_exception`, so a single failing step does not stop the loop.

## MPC Workflow

`MPC.mpc_controller` in [src/class_MPC.py](../src/class_MPC.py) is the core of the control loop:

- before `T_C0` the controller is in startup mode and reuses the previous nominal action without solving
- otherwise it creates a new CasADi `Opti` problem (`SetOptimizationVariables`), adds the constraints (`SetConstraints`) and the cost (`SetCostFunction`), and warm-starts the variables from the previous feasible trajectory
- the problem is solved with IPOPT; on failure it retries with relaxed tolerances, and if that also fails it falls back to the last feasible trajectory
- the outcome is reported as `sol_feas`: `1` solved, `2` solved on the relaxed retry, `3` fallback to the previous solution (`0` during startup)
- the optimized trajectory is kept as warm start for the next step, and only the first move is returned

The horizon has `N` steps, while the control sequence has `N / Nb` free moves (move blocking).

## Model Strategies

All behavior that depends on `Param["Model"]` is delegated to a strategy object created in `MPC._assign_parameters` from `_STRATEGY_FACTORY`. The base class `ModelStrategy` in [src/model_strategies.py](../src/model_strategies.py) implements the NNARX behavior, which is the common baseline; the other strategies override only the hooks where they differ.

| Hook | Purpose |
|---|---|
| `on_init` | initialize strategy-owned state after the controller is built |
| `declare_extra_optimization_variables` | add model-specific CasADi variables |
| `feedback_constraint` | tie the optimized output trajectory to the model prediction |
| `output_constraints` | comfort/operational envelope on the outputs |
| `cost_expression` | define the cost `J` |
| `warm_start_extra_variables` | initial guess for the extra variables |
| `extract_solution` / `fallback_slack_explo` | read slacks and uncertainty from the solution, or the fallback values |
| `select_theta` | last-layer weights used in the recurrent activation |
| `predict_outputs` | turn the network outputs into physical predictions |

- **NNARXStrategy**: the plain neural model; outputs are the denormalized NNARX predictions, with a temperature slack `T_slack` in the cost.
- **GPStrategy**: adds a Gaussian-process (GPyTorch) correction on top of the NNARX prediction, trained online from buffered residuals.
- **BNNExpStrategy**: Bayesian last layer; uses the updated `theta` in the prediction, tightens the output constraints by the predicted uncertainty `sigmaK`, and adds exploration slacks to the cost. It also implements `bayesian_correction`.

## FMU Interactions

In debug mode, `ControllerRunner` loads the FMU with pyFMI (Windows only) and uses it as the plant. In `set_points_to_fmu` it:

- builds the input table from the control history `u_out` (four heat loads and the two boiler temperature references)
- sets the initial outputs and simulates the FMU from `T_C0` to the current step
- reads back `Ts_L4`, `T_return`, `mf_supply` and writes them into the next column of `y_out`
- resets the FMU for the next call

This allows the optimization logic to be tested on a simulated plant without the live system.

## Neural-Network Interactions

The controller loads the MAT model file named in `nnarx_mat` from [NNmodels](../NNmodels) and extracts:

- the last-layer weights `U0`, `b0`, which form the initial Bayesian parameters `theta`
- the output scaler (`out_bias`, `out_scale`) used to denormalize predictions

`MPC.model_rnn`, `simulate_dynamics` and `activation_fun` roll the network forward over the horizon inside the CasADi problem; the active strategy decides which weights are used and how the outputs are post-processed. This is the bridge between the machine-learning model and the control system.

## Component Responsibilities

### Entry point
[Main_OnlineLearningMPC.py](../Main_OnlineLearningMPC.py): parse options, build the parameters, start the runner.

### Configuration layer
[src/config](../src/config): build the parameter dictionary and load the input time series.

### Runtime layer
[src/runtime](../src/runtime):

- allocating the history buffers
- selecting the debug or live execution path and scheduling the loop
- reading plant status and writing setpoints (OPC UA) or driving the FMU
- invoking the optimization and the Bayesian update
- logging exceptions

### Controller core
[src/class_MPC.py](../src/class_MPC.py):

- loading model data and initializing the warm-start trajectory
- building variables, constraints and cost of the optimization problem
- solving with IPOPT and handling solver failures
- rolling out the neural model over the horizon

### Model strategies
[src/model_strategies.py](../src/model_strategies.py): all NNARX/GP/BNNExp-specific variables, constraints, costs and prediction logic.

### Utility layer
[src/utils](../src/utils):

- [DERTF_StatesDefinition.py](../src/utils/DERTF_StatesDefinition.py): reading the plant status from InfluxDB
- [OPCUA/DERTF_SetPointsWriter.py](../src/utils/OPCUA/DERTF_SetPointsWriter.py): writing setpoints over OPC UA
- [DataManagement.py](../src/utils/DataManagement.py): saving results to the solution folders
- [InfluxDB](../src/utils/InfluxDB): InfluxDB/OPC UA connection helpers and variable configuration
- [data](../src/utils/data): bundled input data (prices, load profiles)

### Supporting scripts
- [PostProcessing_Solutions.py](../PostProcessing_Solutions.py): loads and plots the generated solutions
- [smoke_test.py](../smoke_test.py): import check of the main project modules

## Data Flow Description

The repository data flow is approximately as follows:

1. plant measurements are read from InfluxDB (live) or from the FMU simulation (debug)
2. the runner stores the state into `y_out`
3. for BNNExp, the Bayesian last layer is updated with the latest data
4. the neural model and the active strategy predict the future behavior
5. the optimization routine computes feasible control moves
6. the first control action is stored and sent to the plant (OPC UA) or the FMU
7. the step results are appended to CSV files under `Solutions/<Model>/<ddmm>/`, named with the run's `InitialDate`; exceptions go to `LogsFolder/Exceptions.log`
