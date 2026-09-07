# Getting Started

This guide provides the minimum setup steps required to understand and run the repository in its current form.

## Prerequisites

The project currently expects a Python environment and a Windows-based development flow, as shown in [docs/Istruzioni_installazione.txt](Istruzioni_installazione.txt). The repository also depends on packages listed in [requirements.txt](../requirements.txt).

The code references:

- pyFMI for FMU support
- InfluxDB connectivity for live measurements
- OPC UA connectivity for real setpoint writing
- numerical and optimization libraries such as NumPy, CasADi, and related packages

These dependencies are implied by the source code and should be installed according to the project’s existing requirements file.

## Installation Steps

From a Windows PowerShell session:

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

This matches the repository’s installation instruction file and is the primary setup path currently documented in the project.

## Configuration Steps

The runtime settings are defined in the main script [Main_OnlineLearningMPC.py](../Main_OnlineLearningMPC.py). The configuration dictionary includes values such as:

- plant model parameters
- control horizon information
- thermal operating limits
- NN model path
- electricity price data
- runtime model type

Most of the configuration is hard-coded in the script. This is an important repository characteristic and should be treated as an internal/prototype setup pattern rather than a generic user-facing configuration model.

## First Execution Example

Run the script in the default model configuration:

```powershell
python Main_OnlineLearningMPC.py --Model GP
```

Run the debug/simulation version:

```powershell
python Main_OnlineLearningMPC.py --Debug --Model GP
```

The script accepts:

- `--Model` to select the control model variant
- `--Debug` to enable the FMU-based simulation path

## Execution Behavior

The repository supports more than one execution mode:

### Default runtime mode
- reads live status values
- writes control setpoints to the plant
- uses external connectivity and live data sources

### Debug/FMU mode
- loads a simulation model instead of the live system
- runs the control loop against the FMU outputs
- allows the controller to be exercised in isolation from the live installation

This distinction appears clearly in [Main_OnlineLearningMPC.py](../Main_OnlineLearningMPC.py).

## Troubleshooting

### Dependency issues
If the environment cannot import a required package, reinstall the requirements file and verify that the virtual environment is active.

### FMU errors
If FMU simulation fails, verify that:

- the FMU file is present under [FMUs](../FMUs)
- the correct Python environment is active
- pyFMI and supporting runtime dependencies are installed

### Live deployment issues
If the runtime fails in non-debug mode, check the following:

- InfluxDB connectivity and credentials
- OPC UA server availability
- plant node IDs and expected data types
- whether the configured plant model matches the live system

### Output issues
The project stores result files and logs under [Solutions](../Solutions) and [LogsFolder](../LogsFolder). If the outputs are missing, check whether the execution path reached the save logic and whether the runtime folder permissions permit file creation.

## Assumptions and Limitations

This guide reflects the repository as it exists in source form. Some operational assumptions remain implicit in the code, especially around:

- the physical plant layout
- the exact live integration endpoints
- the target deployment environment
- the exact release process for public use

These items should be clarified in a later publication-oriented documentation phase.
