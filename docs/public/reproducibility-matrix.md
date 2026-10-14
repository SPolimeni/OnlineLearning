# Reproducibility Matrix

Verified locally on 2026-10-14. This matrix distinguishes what an external reader can reproduce from what still requires article-specific evidence.

| Reproducibility level | What the reader can do | Required environment/assets | Evidence | Current status |
| --- | --- | --- | --- | --- |
| Source inspection and imports | Inspect the architecture, configuration, MPC core, model strategies, synthetic live-interface fixtures, and import the project modules. | Windows; Conda; Python 3.10; repository source; core packages in `environment.yml`. | `smoke_test.py`, project import matrix, [environment-verification.md](environment-verification.md). | **Verified** with the repository-local environment. |
| Implementation-level debug simulation | Execute the offline controller path against the bundled FMU and bundled model/data assets. This demonstrates how the implementation operates, but does not claim generalizability or exact article-result reproduction. | Supported Windows environment; activated Conda environment; pyFMI/Assimulo; CasADi/IPOPT; PyTorch/GPyTorch; bundled FMU, MAT model, and CSV inputs. | CLI/debug documentation, [architecture.md](architecture.md), [asset-provenance.md](asset-provenance.md). | **Available but FMU execution and trajectory validation remain pending.** |
| Exact article results | Reproduce the article's reported trajectories, costs, solver statuses, first control moves, uncertainty, and strategy comparisons within approved tolerances. | All debug assets plus article configuration, initial conditions, dependency versions, random-state policy, solver records, expected outputs, and owner-approved tolerances. | Required: fixed fixtures, baseline outputs, configuration manifest, and comparison report. | **Not yet established.** |

## Supported environment matrix

The verified local target is Windows with Conda and Python 3.10.20. The repository-local environment was checked using `pip check`, which reported no broken requirements.

| Component | Declared/verified version | Role | Status |
| --- | --- | --- | --- |
| Operating system | Windows | Native FMU/pyFMI support | Verified local target |
| Python | 3.10 / verified 3.10.20 | Runtime | Verified |
| NumPy | 1.26.4 | Numerical arrays | Imported |
| SciPy | 1.15.3 local; `1.15` declaration | Numerical support | Imported |
| pandas | 2.3.3 | Data handling | Imported |
| CasADi | 3.7.2 | Optimization/model expressions | Imported |
| PyTorch | 2.9.0 | Neural model runtime | Imported |
| GPyTorch | 1.14.2 | GP strategy | Imported |
| pyFMI | 2.21.0 local | FMU interface | Imported; FMU run pending |
| Assimulo | 3.7.3 local | FMU numerical support | Installed; FMU run pending |
| APScheduler | 3.11.2 | Periodic runtime scheduling | Imported |
| InfluxDB client | 1.49.0 | Retained live integration boundary | Imported; no connection allowed |
| OPC UA | 0.98.13 | Retained live integration boundary | Imported; no connection allowed |
| scikit-learn | 1.7.2 | Supporting utilities | Imported |
| openpyxl | Optional, not in core environment | Private Excel metadata conversion only | Not required for public workflow |

This matrix records the environment observed locally; it is not a lockfile and does not certify other operating systems or future dependency releases.

## Asset hashes

The complete local hash table is maintained in [asset-provenance.md](asset-provenance.md). Required asset classes are:

- FMUs under `FMUs/`;
- MAT models under `NNmodels/`;
- CSV inputs under `src/utils/data/`.

The hashes identify local bytes only. They do not establish ownership, redistribution rights, model-training provenance, or permission to publish the assets. Those decisions remain separate release gates.

## Required exact-result record

Before claiming exact article reproduction, record the selected model and MAT hash, FMU hash, input CSV hashes, complete parameter configuration, initial conditions, Python/package versions, solver status sequence, output baseline, and approved absolute/relative comparison tolerances. No FMU or live-service execution is implied by this documentation.
