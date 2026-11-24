"""Validate the reduced debug baseline for all supported model strategies."""

import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import sys
import tempfile
from unittest.mock import patch

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
BASELINE_PATH = os.path.join(ROOT, "docs", "public", "strategy-baseline.json")
ASSET_MANIFEST_PATH = os.path.join(ROOT, "docs", "public", "asset-manifest.json")
sys.path.insert(0, SRC)

from config.parameters import build_parameters, validate_parameters
from runtime.controller_runner import ControllerRunner


def file_sha256(relative_path):
    """Return the lowercase SHA-256 digest for one repository asset."""
    digest = hashlib.sha256()
    with open(os.path.join(ROOT, relative_path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(package_names):
    """Return installed versions for the packages recorded in the baseline."""
    return {
        package: importlib.metadata.version(package)
        for package in package_names
    }


def run_strategy(model, configuration):
    """Run one reduced strategy regression without writing generated outputs."""
    parameters = build_parameters(model)
    parameters["tot_hours"] = configuration["tot_hours"]
    parameters["seconds_per_hour"] = configuration["seconds_per_hour"]
    validate_parameters(parameters)

    with tempfile.TemporaryDirectory() as run_directory:
        original_directory = os.getcwd()
        os.chdir(run_directory)
        try:
            with (
                patch("runtime.controller_runner.SaveAggregatedData"),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                runner = ControllerRunner(parameters, debug=True)
                runner.run()
        finally:
            os.chdir(original_directory)

    controller = runner.controller
    return {
        "solver_status_sequence": [int(value) for value in controller.sol_feas.tolist()],
        "control_trajectory": controller.u_out.tolist(),
        "measured_output_trajectory": controller.y_out.tolist(),
        "predicted_output_trajectory": controller.y_pred_out.tolist(),
        "cost_trajectory": controller.final_cost.tolist(),
        "first_optimized_control": controller.u_out[:, 4].tolist(),
        "fmu_output_after_first_control": controller.y_out[:, 5].tolist(),
        "controlled_step_cost": float(controller.final_cost[4]),
    }


def assert_baseline_close(actual, expected, tolerances):
    """Compare strategy trajectories and summary values against the fixture."""
    if actual["solver_status_sequence"] != expected["solver_status_sequence"]:
        raise AssertionError("Solver status sequence differs from the baseline")

    for key, absolute_key, relative_key in (
        ("control_trajectory", "control_absolute", "control_relative"),
        ("measured_output_trajectory", "output_absolute", "output_relative"),
        ("predicted_output_trajectory", "output_absolute", "output_relative"),
        ("cost_trajectory", "cost_absolute", "cost_relative"),
        ("first_optimized_control", "control_absolute", "control_relative"),
        ("fmu_output_after_first_control", "output_absolute", "output_relative"),
    ):
        np.testing.assert_allclose(
            actual[key],
            expected[key],
            atol=tolerances[absolute_key],
            rtol=tolerances[relative_key],
            err_msg=f"{key} differs from the baseline",
        )
    np.testing.assert_allclose(
        actual["controlled_step_cost"],
        expected["controlled_step_cost"],
        atol=tolerances["cost_absolute"],
        rtol=tolerances["cost_relative"],
        err_msg="controlled_step_cost differs from the baseline",
    )


def main():
    """Validate asset hashes, environment metadata, and all strategy outputs."""
    with open(BASELINE_PATH, encoding="utf-8") as handle:
        baseline = json.load(handle)

    with open(ASSET_MANIFEST_PATH, encoding="utf-8") as handle:
        asset_manifest = json.load(handle)

    for asset in asset_manifest["assets"]:
        relative_path = asset["path"]
        actual_hash = file_sha256(relative_path)
        if actual_hash != asset["sha256"]:
            raise RuntimeError(
                f"Baseline asset hash mismatch for {relative_path}: "
                f"expected {asset['sha256']}, got {actual_hash}"
            )

    if platform.platform() != baseline["platform"]["system"]:
        raise RuntimeError("Platform differs from the recorded strategy baseline")
    if platform.machine() != baseline["platform"]["machine"]:
        raise RuntimeError("Machine architecture differs from the recorded strategy baseline")
    if sys.version.split()[0] != baseline["platform"]["python"]:
        raise RuntimeError("Python version differs from the recorded strategy baseline")

    installed = package_versions(baseline["packages"])
    if installed != baseline["packages"]:
        raise RuntimeError(
            f"Package versions differ from the recorded strategy baseline: "
            f"{installed}"
        )

    configuration = baseline["configuration"]
    tolerances = baseline["comparison_tolerances"]
    for model in configuration["models"]:
        actual = run_strategy(model, configuration)
        assert_baseline_close(actual, baseline["strategies"][model], tolerances)

    print(
        "STRATEGY BASELINE PASSED: "
        + ", ".join(configuration["models"])
    )


if __name__ == "__main__":
    main()
