# Execution Modes

The current public repository supports inspection, tests, and offline FMU debug simulation. It does not support live plant execution.

| Case | Command | What it does | What it does not prove |
| --- | --- | --- | --- |
| Inspection | None | Reviews source, parameters, and bundled assets. | Installability or numerical behavior. |
| Smoke test | `python smoke_test.py` | Imports principal modules and validates the GP configuration, required assets, and IPOPT availability. | An FMU trajectory or article result. |
| Test | `python -m unittest discover -s tests -p "test_*.py"` | Runs the assertions currently encoded in the unit tests. | General controller correctness or physical validity. |
| Publication validation | `python tests/validate_publication.py` | Compiles Python files, runs the smoke test, and runs unit tests. | A complete debug trajectory or article reproduction. |
| Debug simulation | `python Main_OnlineLearningMPC.py --Debug --Model GP` | Runs the controller against `FMUs/DHN_sim.fmu` and writes local outputs. | Live operation or exact article reproduction. |

## Selection rules

- Debug simulation is the default; `--Debug` is optional.
- `--Model NNARX`, `--Model GP`, and `--Model BLL` select strategies.
- The Bayesian neural network exploration (BNNExp) strategy is named `BLL` by the CLI and source.
- Run commands from the repository root.

## Disabled live request

`--Live` is not an execution mode available to public users. The entry point rejects it before parameter, controller, InfluxDB, or OPC UA construction. No operational procedure is included.

## Removed numerical fixtures

`tests/validate_fmu_baseline.py` and `tests/validate_strategy_baseline.py` reference `fmu-baseline.json`, `strategy-baseline.json`, and `asset-manifest.json`. Those JSON files are not present in the current tree, so the two scripts are not runnable as documented validation commands. They are also not invoked by `tests/validate_publication.py`.

Until fixed fixtures, hashes, environment metadata, and owner-approved tolerances are supplied, describe debug runs as implementation exercises rather than numerical regressions or article reproductions.
