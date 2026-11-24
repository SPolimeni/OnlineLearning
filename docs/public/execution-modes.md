# Execution Modes

This document is the source of truth for the terms **test**, **debug**, and
**live** in this repository. They identify different execution paths and must
not be used interchangeably.

| Case | Purpose | Command | External I/O | Current public status |
| --- | --- | --- | --- | --- |
| Test | Validate imports, configuration guards, data parsing, metadata, path handling, strategy selection, solver fallback, and public live-mode safety. | `python tests/validate_publication.py` | No plant connection, setpoint write, or FMU trajectory. | Available; compilation, smoke validation, and 14 unit tests pass. |
| Smoke test | Check principal imports, load and validate the GP configuration, and resolve the example FMU path. | `python smoke_test.py` | No plant connection, setpoint write, FMU load, or controller loop. | Available; this is narrower than the unit-test suite. |
| Debug | Run the controller loop against the bundled FMU and feed computed controls back to that simulation. | `python Main_OnlineLearningMPC.py --Debug --Model GP` | Local FMU, bundled model/data inputs, and local result/log files only. It does not intentionally use InfluxDB or OPC UA. | Available on the supported Windows/Conda stack; the reduced GP path has a numerical regression baseline. |
| Live | Read plant state and write setpoints through deployment-specific InfluxDB/OPC UA integrations. | `python Main_OnlineLearningMPC.py --Live --Model GP` | Would require private services, mappings, credentials, and deployment approval. | Disabled. The command raises `RuntimeError` before parameters, connectors, or the controller are constructed. |

## Selection rules

- Omitting both mode flags selects **debug**. `--Debug` is the explicit,
  equivalent spelling.
- `--Model NNARX`, `--Model GP`, and `--Model BLL` select a model strategy;
  they do not select an execution mode.
- Do not combine `--Debug` and `--Live`. A live request takes precedence and
  is rejected.
- Environment variables described in
  [live-deployment-configuration.md](../live-deployment-configuration.md) do
  not enable live execution. They only describe the retained private connector
  boundary.

## What each case proves

Passing the smoke test proves only that the listed modules import. Passing the
unit tests proves the assertions currently encoded under `tests/`; it does not
prove optimization quality or a complete FMU trajectory. The automated
publication validation compiles the source, runs the extended smoke test, and
runs 14 unit tests, including a forced two-attempt solver failure that verifies
reuse of the last feasible plan.

A completed debug run exercises the offline controller and FMU integration.
It is not a live-plant test and is not evidence of exact article-result
reproduction without fixed inputs, asset hashes, expected outputs, and approved
tolerances. The separate command `python tests/validate_fmu_baseline.py` validates one
controlled GP interval against [fmu-baseline.json](fmu-baseline.json), including
asset hashes, solver status, first control, FMU output, cost, and tolerances.
The command `python tests/validate_strategy_baseline.py` validates the reduced
measured/predicted trajectories, costs, solver statuses, and first controls for
all three strategies against [strategy-baseline.json](strategy-baseline.json).

Live execution is an operational deployment activity, not a test or a
reproducibility level. The public repository intentionally provides no command
that connects to or controls a plant.
