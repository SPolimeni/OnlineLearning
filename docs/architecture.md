# Architecture Overview

This document explains the repository structure and the runtime architecture as it appears in the current source code.

## System Overview

The system is organized around a model-predictive control loop. At a high level, it does the following:

1. reads the current plant or simulation state
2. updates internal prediction and control state
3. solves a constrained optimization problem
4. applies the first control move
5. repeats the cycle

The most important files are:

- [Main_OnlineLearningMPC.py](../Main_OnlineLearningMPC.py): runtime entry point and orchestration logic
- [class_MPC.py](../class_MPC.py): core control and optimization implementation
- [utils](../utils): support code for data handling, state definition, and integration
- [NNmodels](../NNmodels): trained or stored model files
- [FMUs](../FMUs): simulation models used in the debug path

## Execution Flow

The execution flow begins in [Main_OnlineLearningMPC.py](../Main_OnlineLearningMPC.py). The main block:

- builds the configuration dictionary
- parses command-line arguments
- creates the controller instance
- initializes arrays and timing metadata
- selects the operating mode

From there, the runtime proceeds in one of two directions:

- live deployment mode, where the code reads values from the operational system and writes setpoints to the plant
- debug mode, where the code relies on an FMU to simulate the plant behavior

The actual control cycle is triggered by the scheduled tasks or by the debug loop.

## MPC Workflow

The MPC workflow is centered on a repeated optimization process:

- the current state is read
- the system writes the measured state into the output history
- the model updates its forecast or correction terms when applicable
- the optimizer computes a feasible sequence of control actions
- the current action is stored and applied
- the loop advances one time step

This is a standard receding-horizon approach: only the first move of the optimized sequence is used before the next cycle recalculates a new plan.

## FMU Interactions

In debug mode, the controller loads an FMU and uses it as the plant model. The interaction pattern is:

- prepare input data using current controller outputs
- feed the input data to the FMU
- simulate the system over the relevant horizon
- read back the relevant outputs
- write the simulated values into the controller’s output history

This allows the optimization logic to be tested on a model-based approximation of the plant without requiring the live system.

## Neural-Network Interactions

The project loads a MAT model file from [NNmodels](../NNmodels) and extracts model parameters such as weights and scaling data. These values are then used by the control algorithm to estimate or predict system responses.

In the code, the model is treated as part of the prediction layer used during optimization. This is the bridge between the machine-learning model and the control system.

## Optimization Loop

The optimization is the central decision-making step in the system. Each cycle, the controller attempts to find a valid control action sequence that keeps the system within the relevant limits while minimizing the objective. The loop depends on:

- current measurements
- internal prediction arrays
- model-based estimates
- operational constraints and slack variables
- system costs and feasibility information

This loop is where the controller decides which setpoint values should be applied next.

## Component Responsibilities

### Runtime controller
Responsible for:

- orchestrating the loop
- evaluating runtime status
- selecting debug or live execution path
- invoking the optimization routine

### Controller core
Responsible for:

- loading model and configuration data
- maintaining state arrays and prediction buffers
- evaluating the system response
- calculating the optimized control action

### Utility layer
Responsible for:

- reading plant status
- writing setpoints
- saving results to the solution folders
- supporting external connections such as InfluxDB or OPC UA

### FMU layer
Responsible for:

- providing simulated plant behavior in debug mode
- calculating outputs based on the current control inputs

### Neural model layer
Responsible for:

- representing the predictive part of the system
- supplying model-based estimates for optimization

## Data Flow Description

The repository data flow is approximately as follows:

1. plant measurements are read from InfluxDB or a simulation model
2. the controller stores the state into the current output arrays
3. the neural model and prediction layer estimate the future behavior
4. the optimization routine computes feasible control moves
5. the system applies the first control action
6. the results are stored under [Solutions](../Solutions) and the relevant logs are updated

This architecture is consistent with the source code and preserves the operational logic currently used by the project.
