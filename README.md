# Project Overview

This repository contains a model-predictive control (MPC) application for a thermal/control system. Based on the source code, it appears to target a district-heating or thermal-energy context, with a control loop that reads current measurements, predicts future behavior using a neural-network-based model, and selects control actions over a receding horizon.

The project combines several elements that are visible in the codebase:

- a real-time controller loop in [Main_OnlineLearningMPC.py](Main_OnlineLearningMPC.py)
- the main controller implementation in [src/class_MPC.py](src/class_MPC.py)
- support utilities in [src/utils](src/utils)
- an FMU simulation path for debugging and validation
- live measurement and actuation integrations via InfluxDB and OPC UA
- output files written to the [Solutions](Solutions) directory

This appears to be a research-oriented control application rather than a general-purpose library. The code is structured around a highly specific plant model and operating scenario, and the current repository layout reflects an internal/prototype workflow more than a fully polished public package.

## Architecture Overview

### Main_OnlineLearningMPC.py
This file acts as the runtime entry point and orchestration layer. It:

- configures the operating parameters for the control system
- parses command-line options such as model type and debug mode
- initializes the controller instance
- reads plant measurements from the live system or from debug data
- runs the optimization loop and writes setpoints to the plant or simulation layer

The main block is the place where the project decides whether it is running in debug mode or live deployment mode.

### src/class_MPC.py
This file contains the main controller logic. It loads the model, initializes prediction buffers, exposes the optimization-related state, and defines the control logic used to compute a feasible action sequence. This is the core of the system.

### src/utils
The [src/utils](src/utils) directory contains support modules for:

- reading operational state variables
- writing setpoints to the target system
- saving aggregated results and controller outputs
- managing integration with external data sources such as InfluxDB

### FMU layer
The code includes a debug path that loads a Functional Mock-up Unit (FMU) using pyFMI. This is used to simulate plant behavior without using the live industrial process.

### Neural network layer
The project loads a MAT file from [NNmodels](NNmodels) and extracts hyperparameters from the stored model. The code then uses those values to inform the predicted plant response and the optimization logic.

### Optimization layer
The system repeatedly solves a constrained optimization problem during each controller cycle. The results are used to decide the current control move while respecting plant operating limits and model predictions.

## Repository Structure

The repository currently contains both source code and research/runtime artifacts. The main folders and files are:

The runnable entry points stay at the repository root; all importable source code
lives under `src/`, which the entry points add to the import path.

- [Main_OnlineLearningMPC.py](Main_OnlineLearningMPC.py): main controller application entry point
- [PostProcessing_Solutions.py](PostProcessing_Solutions.py): post-processing utilities for generated solution outputs
- [smoke_test.py](smoke_test.py): lightweight import check of the project modules
- [src](src): the project source code
  - [class_MPC.py](src/class_MPC.py): core MPC implementation
  - [model_strategies.py](src/model_strategies.py): per-model strategies (NNARX, GP, BNNExp)
  - [project_paths.py](src/project_paths.py): repository-stable paths to runtime data
  - [utils](src/utils): integration and helper modules, including the bundled input data
- [NNmodels](NNmodels): trained or stored neural-network model files
- [FMUs](FMUs): FMU model files used in simulation/debug mode
- [Solutions](Solutions): output directory used to store result files generated during execution
- [docs](docs): project documentation
- [env](env): local environment/dependency directory, typically not intended for public source control

`LogsFolder` is created at runtime for exception logs and is not tracked.

## Prerequisites

This project is a research/industrial control application and depends on a specific scientific stack. Before running it, create the Conda environment defined in [environment.yml](environment.yml), which includes:

- Python 3.10+
- NumPy, SciPy
- CasADi
- PyTorch / GPyTorch (depending on the model path)
- pyFMI for FMU-based simulation
- OPC UA / InfluxDB support if using the live-plant path

The repository also assumes a local Windows-oriented setup for the industrial deployment path.

## Installation

From PowerShell, create the local Conda environment and activate it:

```powershell
conda env create --prefix .\env --file environment.yml
conda activate .\env
```

The Conda environment must be active before running scripts that use PyFMI. PyFMI relies on native dependencies, including FMILibrary and Assimulo, supplied by `conda-forge`. In VS Code, select `env\python.exe` as the Python interpreter.

To update an existing local environment after `environment.yml` changes:

```powershell
conda env update --prefix .\env --file environment.yml --prune
```

Further installation notes are available in [docs/Istruzioni_installazione.txt](docs/Istruzioni_installazione.txt).

## Quick Start

The minimum steps to run the application are:

1. Create the Conda environment from `environment.yml`.
2. Activate it with `conda activate .\env`.
3. Run the main controller script:

```powershell
python Main_OnlineLearningMPC.py --Model GP
```

4. If you want to run the debug/simulation route:

```powershell
python Main_OnlineLearningMPC.py --Debug --Model GP
```

The command-line interface in [Main_OnlineLearningMPC.py](Main_OnlineLearningMPC.py) supports the `--Model` and `--Debug` switches.

## Fast smoke validation

Before a longer run, use the lightweight repository check:

```powershell
python smoke_test.py
```

This validates the project root and key Python files without starting the controller loop.

## Execution Modes

### Debug mode
The debug mode path is enabled with the `--Debug` option. In this mode, the code loads an FMU and runs a simulated control loop. This mode appears intended for testing the controller behavior without interacting with the live plant.

### FMU simulation mode
This is effectively the same simulation path used in debug mode. The FMU receives the control actions generated by the optimization loop and returns system outputs that are used to update the controller state.

### Live deployment mode
When debug mode is not active, the project appears to connect to a live system through:

- InfluxDB for obtaining measurements
- OPC UA for writing setpoints to the plant

This mode is more operational and depends on external infrastructure that is not fully abstracted away from the code.

## Usage Example

A typical execution workflow is:

```powershell
conda env create --prefix .\env --file environment.yml
conda activate .\env
python Main_OnlineLearningMPC.py --Model GP
```

For a local simulated loop:

```powershell
python Main_OnlineLearningMPC.py --Debug --Model GP
```

The repository writes output files under [Solutions](Solutions) during execution, which is consistent with the data management utilities in [src/utils/DataManagement.py](src/utils/DataManagement.py).

## Contributing

Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance, expected workflows, branch practices, and coding/documentation expectations.

## License

A license has not yet been selected for this repository. Before public publication, a project owner should choose an appropriate open-source license and add it to the repository root.

---

This README reflects the repository as it exists in source code today. Where the repository does not yet provide a fully public-ready workflow, the text clearly distinguishes the confirmed behavior from the operational assumptions inferred from the code.
