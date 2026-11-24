# Public Documentation

This folder is the external-reader documentation set for OnlineLearning, a research-oriented model-predictive control (MPC) application for a thermal or district-heating plant.

## Documents

- [Getting started and installation](getting-started.md): requirements, environment setup, smoke test, debug simulation, and a warning about live execution.
- [Execution modes](execution-modes.md): authoritative definitions, commands, I/O boundaries, and claims for test, debug, and live cases.
- [FMU baseline](fmu-baseline.json): versioned assets, configuration, expected first controlled GP interval, and numerical tolerances.
- [Strategy baseline](strategy-baseline.json): fixed reduced trajectories, costs, solver statuses, first controls, and comparison tolerances for NNARX, GP, and BLL.
- [Environment lock](../../environment-lock.txt): exact package-version audit export from the verified local environment.
- [Architecture](architecture.md): command-line entry point, configuration flow, runtime loop, MPC core, model strategies, FMU debug path, live interfaces, and data flow.
- [Environment verification](environment-verification.md): the locally verified Windows/Conda/Python environment and checks that were performed.
- [Reproducibility matrix](reproducibility-matrix.md): what can currently be inspected, smoke-tested, or exercised through debug simulation, and what is still required for exact article-result reproduction.
- [Asset provenance](asset-provenance.md): measured hashes for local FMUs, MAT models, and CSV inputs, with their unresolved redistribution status.

## Scope and limitations

The documents preserve the repository's current distinction between inspection, import smoke testing, offline FMU debug simulation, and live plant execution. Live execution requires private InfluxDB and OPC UA deployment details and is not a public installation or validation procedure.

The local model, FMU, and data files are not automatically cleared for redistribution. Their hashes identify local bytes only; they do not establish ownership, licensing, provenance, anonymization, or publication permission. The strategy baseline is an implementation-level regression and does not claim exact reproduction of published results.

For project context, see the repository [README](../../README.md). Source links in the documents below point to repository paths and are provided for inspection; they do not imply that the referenced assets or services are publicly distributable or available.

## Owner metadata still required

- Repository license and asset-specific redistribution decisions.
- Citation metadata, authors, DOI, and public maintainer contact.
- Training-data provenance and publication approval for model and input files.
- Article configuration, initial conditions, solver records, expected outputs, and approved tolerances for exact-result reproduction.
