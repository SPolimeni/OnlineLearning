# Publication Asset Policy

`Solutions/` contains generated controller outputs and plant-derived result
artifacts. These files are retained locally for the current research workflow,
but are excluded from the public repository because their ownership,
anonymization, and redistribution status have not been approved.

`LogsFolder/` is also local-only and is ignored by Git. Do not publish runtime
logs, exception traces, timestamps, measurements, setpoints, node identifiers,
or other operational metadata.

The public release should contain only approved synthetic fixtures or
owner-approved, anonymized examples. The exclusion does not delete local
results or establish permission to redistribute them. Before creating the
future public repository, perform an independent release-tree scan and obtain
the required data-ownership and privacy decisions.

The runtime assets used by the public debug path are enumerated only in
[asset-manifest.json](public/asset-manifest.json). The manifest is the source
of truth for each asset's role, byte count, SHA-256 hash, provenance note, and
redistribution status; baseline fixtures refer to it rather than maintaining
independent asset lists.