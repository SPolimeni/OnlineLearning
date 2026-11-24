# Asset Provenance Record

This record is rendered from the single-source manifest
[asset-manifest.json](asset-manifest.json), generated from the working tree on
2026-11-24. The hashes identify local files; they do not establish ownership,
licensing, or permission to redistribute them.

| Asset | Role | Bytes | SHA-256 | Local provenance | Redistribution status |
| --- | --- | ---: | --- | --- | --- |
| `FMUs/DHN_sim.fmu` | Debug FMU | 1969459 | `D3B4FBB695AF633C6F8717E77CF62641F15E9F31A5B2214C4D1992E7858DBBC9` | Present in working tree; upstream provenance not supplied | Review required |
| `NNmodels/NARX_net.mat` | Default NNARX model | 29592 | `A270D186C0D379DB0E5F1D4B2D6FD187CF641AAAFC6128FC04D5F00209379228` | Present in working tree; training provenance not supplied | Review required |
| `src/utils/data/DayAhead_ZonalPrices.csv` | Electricity-price input | 513 | `E401C5E3E6641A894609A3D363DF99B1D382B01E208A0A072B215FEDBA3EC712` | Present in working tree; source and redistribution permission not supplied | Review required |
| `src/utils/data/DHN_ground_truth.csv` | Heat-load input | 40570 | `8434044EE106888B9260636E374B9CFD1E209C3D4F180E5187D7E3D7CF342E89` | Present in working tree; source and redistribution permission not supplied | Review required |

No binary asset was modified by this change.

## Generated plant-derived results

The tracked `Solutions/` tree is not part of the intended public release. It contains generated controller outputs and potentially plant-derived values; ownership, anonymization, and redistribution approval are not established. It is excluded by `.gitignore` and must remain outside the future public repository unless specific artifacts are approved and anonymized.
