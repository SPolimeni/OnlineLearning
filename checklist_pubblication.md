# Repository publication and handoff checklist

## 1) Base release
- [ ] Add an appropriate open-source license, such as MIT / Apache 2.0
- [ ] Create minimal public documentation:
  - project overview
  - installation
  - prerequisites
  - architecture
  - execution workflow
  - known limitations
- [ ] Verify that the README does not contain overly specific references to the local setup
- [ ] Clearly separate:
  - source code
  - data / models / examples
  - generated artifacts
  - utility and support files

## 2) Configuration and paths
- [ ] Centralize configuration in one location
- [ ] Use a robust path resolver based on the repository root or configuration
- [ ] Remove dependencies on non-standard local folders where possible
- [ ] Add initial checks for:
  - required files being present
  - the model being loaded
  - data being available
  - the solver / dependencies being accessible
- [ ] Clearly document live vs debug vs test cases

## 3) Code stability
- [ ] Always validate the dimensions of state/input/output vectors
- [ ] Add guard checks for:
  - parameter bounds
  - optimization constraints
  - problem size
  - missing data
- [ ] Verify that the solver has clear fallbacks and readable errors
- [ ] Avoid hidden logic in overly "magical" execution scripts

## 4) Testing and validation
- [ ] Add a minimal smoke test for:
  - importing the main modules
  - compiling critical Python files
  - loading the configuration
  - validating an example path
- [ ] Add tests for:
  - data parsing
  - path resolution
  - input/metadata validation
  - solver fallback behavior
- [ ] Run a syntax regression check on the main files

## 5) Environment and reproducibility
- [ ] Create a pinned environment with known dependencies
- [ ] Include the minimum prerequisites for:
  - CasADi
  - PyTorch / GPyTorch / other model dependencies
  - FMU / OPC UA / solver
- [ ] Clearly document system requirements
- [ ] Specify which components are required and which are optional

## 6) Repository hygiene
- [ ] Exclude generated artifacts from version control
- [ ] Perform a final cleanup of legacy or duplicate files
- [ ] Maintain a single "source of truth" for example data and models
- [ ] Verify that root files do not contain unorganized runtime data
- [ ] Separate experimental results from project assets

## 7) Operational documentation
- [ ] Write a clear 5–10 minute quickstart
- [ ] Write a "Known limitations" section
- [ ] Document:
  - how to start a live run
  - how to use debug mode
  - how to interpret outputs / logs
  - how to identify solver failures or invalid data
- [ ] Explicitly identify where plant assumptions are strong

## 8) Publication / final handoff
- [ ] Verify that the repository can be presented to a colleague without the original setup
- [ ] Add an onboarding document for a new maintainer
- [ ] Add a minimal changelog
- [ ] Prepare a "what this project is / is not" note
- [ ] Perform a final check: build / import / compile / smoke test

## Realistic priority
For a minimal but useful transition, the first four tasks are:

1. License
2. Quickstart + prerequisites
3. Centralized configuration/paths
4. Smoke test / py_compile / import check

These provide the greatest benefit without risking changes to the control logic.
