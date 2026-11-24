# Architecture

## Scope

The public runtime is an offline model-predictive control (MPC) simulation. It combines a learned prediction model, a CasADi/IPOPT optimization problem, and a functional mock-up unit (FMU). Live plant execution is rejected by the public entry point.

## Main components

- [Main_OnlineLearningMPC.py](../../Main_OnlineLearningMPC.py): parses command-line arguments and starts the runtime.
- [src/config/parameters.py](../../src/config/parameters.py): builds and validates the parameter dictionary.
- [src/runtime/controller_runner.py](../../src/runtime/controller_runner.py): owns buffers, the control loop, persistence, FMU calls, and exception logging.
- [src/class_MPC.py](../../src/class_MPC.py): loads model metadata and constructs and solves the MPC problem.
- [src/model_strategies.py](../../src/model_strategies.py): implements NNARX, GP, and BNNExp behavior; the code and CLI call BNNExp `BLL`.
- [src/project_paths.py](../../src/project_paths.py): resolves data, model, FMU, output, and log paths from the repository root.
- [src/utils/DataManagement.py](../../src/utils/DataManagement.py): appends result files.

The source tree still contains InfluxDB and OPC UA connector modules, but no public command can construct the non-debug runner. They are not a supported execution path.

## Command-line and configuration flow

`Main_OnlineLearningMPC.py` accepts `--Model {NNARX,GP,BLL}`, `--Debug`, and `--Live`. Debug is the default. A `--Live` request raises `RuntimeError` before parameters or connectors are constructed.

For debug execution:

1. `build_parameters(model)` loads the two CSV inputs and returns dimensions, timing, bounds, costs, solver settings, FMU settings, initial values, and the path to `NNmodels/NARX_net.mat`.
2. `validate_parameters` checks dimensions, data, required model and FMU files, MAT metadata, required imports, and IPOPT availability.
3. `ControllerRunner(parameters, debug=True)` creates `MPC` and allocates result buffers.
4. `ControllerRunner.run()` configures the FMU and executes `num_steps - 1` control steps.

Paths are repository-stable, but running commands from the repository root remains the documented workflow.

## Runtime control loop

Each `solve_step()`:

1. reads a startup fixture or the latest FMU output;
2. updates the Bayesian last layer for `BLL` after the startup boundary;
3. calls `MPC.mpc_controller`;
4. may repeat the solve without exploration according to the exploration slack;
5. stores the first control move, predictions, slacks, uncertainty, cost, time, and solver status;
6. appends result files; and
7. after startup, sends the accumulated control history to the FMU and stores its next output.

Exceptions in individual stages are logged to `LogsFolder/Exceptions.log`. Some caught failures allow the loop to continue with retained values, so completion alone does not prove that every optimization succeeded.

## MPC core and CasADi/IPOPT lifecycle

Before `T_C0`, `MPC.mpc_controller` returns the nominal previous action without solving. At and after `T_C0`, it creates a new CasADi `Opti` problem, declares variables, adds dynamics, constraints, and cost terms, and warm-starts from retained trajectories.

The first IPOPT attempt uses the primary tolerances. If it fails, the code changes the objective scaling and retries with relaxed settings. If both attempts fail, it reuses retained plan data. `sol_feas` records `0` for startup, `1` for the first solve, `2` for the relaxed retry, and `3` for fallback. Only the first control move is applied.

The configured horizon is `N=12` samples with move block `Nb=3`, producing four free control moves. These values are scientific configuration, not general guarantees.

## Model strategies

- **NNARX:** rolls the neural-network autoregressive model over the horizon and uses a temperature slack in the MPC cost.
- **GP:** starts from NNARX predictions and trains separate Gaussian-process residual corrections from rolling buffers. This state changes during a run.
- **BNNExp (`BLL`):** uses a Bayesian last layer, updates `theta`, propagates output uncertainty, tightens output constraints, and introduces exploration slacks. `BLL` is the only accepted CLI spelling.

All strategies use the single bundled `NNmodels/NARX_net.mat` asset for network weights and output scaling.

## FMU debug path

On Windows, `_configure_fmu()` loads `FMUs/DHN_sim.fmu` through pyFMI and sets CVode relative and absolute tolerances from the parameter dictionary. The runner seeds startup state and control columns from the MPC warm start.

After each controlled step, `set_points_to_fmu()` supplies:

`PowerLoad1`, `PowerLoad2`, `PowerLoad3`, `PowerLoad4`, `Tref_GB`, and `Tref_EB`.

It reads:

`Ts_L4`, `T_return`, and `mf_supply`.

The FMU is reset after each simulation call. This path is a debug simulation, not evidence of physical validity or exact article reproduction.

## Data, model, and result flow

1. `src/utils/data/DayAhead_ZonalPrices.csv` and `DHN_ground_truth.csv` enter the parameter dictionary.
2. `NNmodels/NARX_net.mat` supplies network parameters and output scaling.
3. Startup fixtures and then `FMUs/DHN_sim.fmu` supply measured states.
4. The selected strategy generates horizon predictions.
5. CasADi/IPOPT computes a control plan; the runner retains and applies its first move.
6. New results are appended under `Solutions/<Model>/<ddmm>/`; exceptions go to `LogsFolder/Exceptions.log`.
7. `Solutions_experimental/` is a separate retained archive and is not read by the controller.

The current tree contains no asset manifest or numerical baseline JSON files. Therefore, the baseline validation scripts cannot currently establish numerical regression results without separately restoring their required fixtures.
