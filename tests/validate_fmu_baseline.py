"""Validate the reduced public GP debug path against its numerical baseline."""

import hashlib
import json
import os
import sys
import tempfile
from unittest.mock import patch

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
BASELINE_PATH = os.path.join(ROOT, "docs", "public", "fmu-baseline.json")
ASSET_MANIFEST_PATH = os.path.join(ROOT, "docs", "public", "asset-manifest.json")
sys.path.insert(0, SRC)

from config.parameters import build_parameters, validate_parameters
from runtime.controller_runner import ControllerRunner


def file_sha256(path):
    """Return the lowercase SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    """Run one controlled FMU interval and compare it with the baseline."""
    with open(BASELINE_PATH, encoding="utf-8") as handle:
        baseline = json.load(handle)

    with open(ASSET_MANIFEST_PATH, encoding="utf-8") as handle:
        asset_manifest = json.load(handle)

    for asset in asset_manifest["assets"]:
        relative_path = asset["path"]
        actual_hash = file_sha256(os.path.join(ROOT, *relative_path.split("/")))
        if actual_hash != asset["sha256"]:
            raise RuntimeError(
                f"Baseline asset hash mismatch for {relative_path}: "
                f"expected {asset['sha256']}, got {actual_hash}"
            )

    configuration = baseline["configuration"]
    parameters = build_parameters(configuration["model"])
    parameters["tot_hours"] = configuration["tot_hours"]
    parameters["seconds_per_hour"] = configuration["seconds_per_hour"]
    validate_parameters(parameters)

    original_directory = os.getcwd()
    with tempfile.TemporaryDirectory() as run_directory:
        try:
            os.chdir(run_directory)
            with patch("runtime.controller_runner.SaveAggregatedData"):
                runner = ControllerRunner(parameters, debug=True)
                runner.run()
        finally:
            os.chdir(original_directory)

    controller = runner.controller
    control = controller.u_out[:, configuration["controlled_step"]]
    output = controller.y_out[:, configuration["fmu_output_index"]]
    cost = controller.final_cost[configuration["controlled_step"]]
    solver_status = int(controller.sol_feas[configuration["controlled_step"]])
    expected = baseline["expected"]
    tolerances = baseline["tolerances"]

    if int(controller.num_steps) != configuration["num_steps"]:
        raise AssertionError(
            f"Expected {configuration['num_steps']} steps, got {controller.num_steps}"
        )
    if solver_status != expected["solver_status"]:
        raise AssertionError(
            f"Expected solver status {expected['solver_status']}, got {solver_status}"
        )
    np.testing.assert_allclose(
        control,
        expected["first_optimized_control"],
        atol=tolerances["control_absolute"],
        rtol=tolerances["control_relative"],
    )
    np.testing.assert_allclose(
        output,
        expected["fmu_output_after_first_control"],
        atol=tolerances["fmu_output_absolute"],
        rtol=tolerances["fmu_output_relative"],
    )
    np.testing.assert_allclose(
        cost,
        expected["final_cost"],
        atol=tolerances["cost_absolute"],
        rtol=tolerances["cost_relative"],
    )

    print(
        "FMU BASELINE PASSED: "
        + json.dumps(
            {
                "solver_status": solver_status,
                "first_optimized_control": control.tolist(),
                "fmu_output_after_first_control": output.tolist(),
                "final_cost": float(cost),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
