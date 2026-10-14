# Legacy Utility Assessment

`SaveAggregatedData` is the runtime persistence function used by
`ControllerRunner`. `SaveData` remains as a compatibility utility but has no
references in the current Python source. It is not deleted in this checkpoint:
removal requires API ownership confirmation and regression coverage for any
external or historical caller.