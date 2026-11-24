# Legacy Utility Assessment

`SaveAggregatedData` is the runtime persistence function used by
`ControllerRunner`. `SaveData` remains as a compatibility utility but has no
references in the current Python source. It is not deleted in this checkpoint:
removal requires API ownership confirmation and regression coverage for any
external or historical caller.

`PostProcessing_Solutions.py` is an optional, non-runtime analysis utility. It
is not imported by `Main_OnlineLearningMPC.py` or the public debug validators
and consumes generated files under `Solutions/`; it must not be treated as a
runtime dependency or an asset source of truth.

`DBVarListToJson.py` is a private deployment utility for converting an
operator-provided Excel mapping. It is outside the public debug path and
requires the optional `openpyxl` dependency.