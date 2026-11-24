# Operational Guide

This guide separates the public offline workflow from the private procedure
required for an authorized live deployment. It does not enable live execution,
provide credentials, or replace plant-owner approval.

## Public debug operation

Run the debug path from the repository root in the verified Windows/Conda
environment:

```powershell
python Main_OnlineLearningMPC.py --Debug --Model GP
```

`NNARX` and `BLL` can be selected with `--Model`. Debug mode reads the bundled
CSV inputs and `FMUs/DHN_sim.fmu`, advances the local FMU, and writes generated
results below `Solutions/<Model>/<ddmm>/`. It does not connect to InfluxDB or
OPC UA.

The public `--Live` command is intentionally rejected before parameters,
connectors, or the controller are constructed. Environment variables alone
cannot enable it.

## Authorized live deployment procedure

Only a deployment owner may execute this procedure in a private, approved
environment:

1. Obtain written authorization covering the plant, operating window, model
   version, setpoint limits, operator on call, rollback action, and data
   retention.
2. Verify the approved FMU/model baseline, package environment, asset hashes,
   parameter configuration, and solver tolerances before connecting to any
   service.
3. Use a private deployment wrapper or approved deployment branch. Do not
   bypass the public `--Live` guard by editing the public entry point.
4. Provide connector settings through a private secret store or process
   environment: `ONLINELEARNING_OPCUA_SERVER_URL`,
   `ONLINELEARNING_INFLUXDB_URL`, `ONLINELEARNING_INFLUXDB_TOKEN`,
   `ONLINELEARNING_INFLUXDB_ORG`, and `ONLINELEARNING_INFLUXDB_BUCKET`.
   Never commit these values, certificates, node identifiers, or plant tags.
5. Confirm that InfluxDB reads return fresh values with the expected tags and
   units, and that OPC UA mappings resolve to the approved control nodes.
6. Start in an observation or read-only mode when the deployment permits it.
   Check state freshness, temperatures, flow, solver status, constraints,
   alarms, and log output before authorizing writes.
7. Enable setpoint writes only within the approved operating window. Keep an
   operator present and monitor the first control intervals.
8. Stop the run and restore the plant's last approved control policy if any
   state is stale, a constraint is violated, the solver falls back, a
   communication retry is exhausted, or measured behavior leaves the validated
   envelope.
9. Preserve the approved run configuration, solver statuses, logs, and result
   hashes in the private deployment record. Do not copy plant-derived files
   into the public repository.

The public repository does not contain a live-start command. A private
deployment must add its own reviewable entry point and safety interlocks
without changing the public debug contract.

## Output files and meanings

`SaveAggregatedData` appends one row per control step. Each CSV row starts with
the local timestamp and `t_step`, followed by the recorded values:

| File pattern | Meaning |
| --- | --- |
| `y_measured_<run>.csv` | Measured/simulated state: delivery temperature, return temperature, flow |
| `y_pred_<run>.csv` | Optimizer-predicted output trajectory at the current stored horizon index |
| `y_rnn_<run>.csv` | Neural-network output prediction |
| `controlLaw_<run>.csv` | Six applied/control variables: four branch powers and two supply-temperature commands |
| `Pb_<run>.csv` | Predicted gas and electric boiler powers |
| `slack_<run>.csv` | Constraint slack values; BLL stores output-sized slack |
| `exploration_<run>.csv` | Exploration flag selected for the step |
| `final_cost_<run>.csv` | Stored economic objective value for the step |
| `computational_time_<run>.csv` | Controller computation time for the step |
| `sol_feas<run>.csv` | Solver outcome code for the step |
| `config_<run>.csv` | Parameter dictionary captured on the first step |
| `ThetaAndSigma_<run>.pkl` | BLL parameter history and uncertainty history |

The exact ordering, scaling, and units are defined by
[parameters.py](../../src/config/parameters.py), not by the filenames alone.
`LogsFolder/Exceptions.log` contains timestamped exception location and
message records. `Solutions/` and `LogsFolder/` are generated, local-only
directories and must remain excluded from a public release.

## Diagnosing failures

### Startup and data failures

The command stops before controller construction when required files,
dependencies, model metadata, dimensions, parameter bounds, or finite data
checks fail. Treat `RuntimeError: Configuration preflight failed` as a
deployment-blocking error. Do not replace missing data with zeros or bypass
the validation.

Typical causes include:

- missing `NARX_net.mat`, `DHN_sim.fmu`, or either CSV input;
- non-numeric or non-finite CSV values;
- invalid model fields or output-scaler shapes;
- unavailable CasADi IPOPT, pyFMI, Torch, or GPyTorch dependencies;
- insufficient heat-load rows for the configured horizon.

### Solver failures and fallback

The `sol_feas` output records the control outcome:

| Code | Meaning | Required action |
| ---: | --- | --- |
| `0` | Startup step before optimization | Expected only during initialization |
| `1` | Direct IPOPT solve succeeded | Continue while state and constraints remain valid |
| `2` | Relaxed retry succeeded | Record and review; the nominal solve was not accepted |
| `3` | Both solves failed; last feasible trajectory reused | Treat as an alarm in live operation and apply the approved rollback policy |

The runtime prints solver messages and records exceptions in
`LogsFolder/Exceptions.log`. A fallback is not a successful fresh
optimization: it reuses the previous feasible plan and must be investigated.

### Runtime data and communication failures

Invalid runtime dimensions or non-finite states are errors. In the retained
live boundary, status-read failures are logged and the previous state may be
kept; this behavior is not a substitute for a plant safety system. OPC UA
write failures are retried according to the configured attempt count and
delay. Exhausted retries require operator intervention and rollback.

## Strong assumptions

- The state ordering is `(T_delivery, T_return, flow)` and the controller
  receives values in the configured physical units.
- The control ordering is four branch power inputs followed by `T_out_GB` and
  `T_out_EB`; external mappings must preserve this order and sign convention.
- The heat-load CSV has four usable load columns and enough future rows for the
  configured horizon; the electricity-price CSV has the expected price
  column and sampling relationship.
- The MAT model's last-layer weights and output scalers match `n_x`, `n_out`,
  the feature ordering, normalization, and the trained model's intended units.
- `N` and `Nb` define valid move blocking, and the fixed initial state/control
  and debug startup fixture are representative for the validated run.
- The FMU exposes `Ts_L4`, `T_return`, `mf_supply`, and the six input names
  used by `ControllerRunner`; changing those names or semantics invalidates
  the debug baseline.
- IPOPT and the FMU solver are deterministic enough under the recorded
  Windows/package configuration for the implementation-level baseline. This
  does not establish exact article-result reproducibility.
- The configured operating bounds, fallback values, scaling factors, COP,
  efficiency, and economic coefficients describe the intended plant. They
  are not independently inferred or safety-certified by this repository.

For architecture and data-flow details, see
[architecture.md](architecture.md). For environment and numerical validation,
see [environment-verification.md](environment-verification.md) and
[strategy-baseline.json](strategy-baseline.json).
