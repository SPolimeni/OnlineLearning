"""Runtime parameter construction for the online-learning MPC controller."""

import importlib
import os

import numpy as np
from scipy.io import loadmat

from project_paths import data_file, fmu_file, model_file


SUPPORTED_MODELS = ("NNARX", "GP", "BLL")


def _require_file(path, description):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required {description} was not found: {path}"
        )


def validate_parameter_guards(params):
    """Validate parameter bounds and the minimum problem/data dimensions."""
    errors = []

    positive_parameters = (
        "n_out", "n_x", "n_inp", "tot_hours", "seconds_per_hour",
        "Ts", "N", "Nb", "specific_heat_capacity", "q_eb",
        "COP", "eff_EB", "n_slack",
    )
    for name in positive_parameters:
        if params[name] <= 0:
            errors.append(f"Parameter '{name}' must be greater than zero")

    if params["N"] % params["Nb"] != 0:
        errors.append("Parameters 'N' and 'Nb' must be divisible")

    bounds = (
        ("Tr_min", "Tr_max"),
        ("Ts_min_eb", "Ts_max_eb"),
        ("Ts_min_gb", "Ts_max_gb"),
        ("Ts_min_supply", "Ts_max"),
        ("m_min", "m_max"),
        ("Pb_min", "Pb_max"),
        ("Pb_min_eb", "Pb_max_eb"),
    )
    for lower_name, upper_name in bounds:
        if params[lower_name] > params[upper_name]:
            errors.append(
                f"Parameter '{lower_name}' cannot be greater than '{upper_name}'"
            )

    total_time = params["tot_hours"] * params["seconds_per_hour"]
    num_steps = total_time // params["Ts"] + 1
    minimum_load_rows = num_steps + params["N"] - params["Nb"]
    load_data = np.asarray(params["Potenza"])
    if load_data.ndim != 2 or load_data.shape[1] != 4:
        errors.append("Parameter 'Potenza' must have shape (rows, 4)")
    elif load_data.shape[0] < minimum_load_rows:
        errors.append(
            f"Parameter 'Potenza' has {load_data.shape[0]} rows; "
            f"at least {minimum_load_rows} are required for the configured horizon"
        )

    for name, value in (("c_el", params["c_el"]), ("Potenza", params["Potenza"])):
        array = np.asarray(value)
        if array.size == 0:
            errors.append(f"Parameter '{name}' cannot be empty")
        elif not np.all(np.isfinite(array)):
            errors.append(f"Parameter '{name}' contains non-finite values")

    if errors:
        raise ValueError("Parameter guard validation failed:\n- " + "\n- ".join(errors))


def validate_parameters(params):
    """Validate required assets, loaded data, model structure, and dependencies."""
    errors = []

    try:
        validate_parameter_guards(params)
    except ValueError as error:
        errors.append(str(error))

    dimension_checks = (
        ("initial_state", params["initial_state"], (params["n_out"],)),
        ("initial_control", params["initial_control"], (params["n_inp"],)),
        ("debug_initial_state", params["debug_initial_state"], (params["n_out"], 4)),
    )
    for name, value, expected_shape in dimension_checks:
        actual_shape = np.asarray(value).shape
        if actual_shape != expected_shape:
            errors.append(
                f"Configured '{name}' has shape {actual_shape}; "
                f"expected {expected_shape}"
            )
    if params["N"] <= 0 or params["Nb"] <= 0 or params["N"] % params["Nb"] != 0:
        errors.append("Configured prediction horizon N must be positive and divisible by Nb")

    required_files = (
        (params["nnarx_mat"], "neural-network model"),
        (fmu_file(params["debug_fmu"]), "debug FMU"),
    )
    for path, description in required_files:
        if not os.path.isfile(path):
            errors.append(f"Required {description} was not found: {path}")

    for name, value in (
        ("c_el", params["c_el"]),
        ("Potenza", params["Potenza"]),
    ):
        array = np.asarray(value)
        if array.size == 0:
            errors.append(f"Configured data '{name}' is empty")
        elif not np.all(np.isfinite(array)):
            errors.append(f"Configured data '{name}' contains non-finite values")

    if np.asarray(params["c_el"]).ndim != 1:
        errors.append("Configured electricity-price data 'c_el' must be one-dimensional")
    if np.asarray(params["Potenza"]).ndim != 2 or np.asarray(params["Potenza"]).shape[1] != 4:
        errors.append("Configured load data 'Potenza' must have four columns")

    if params["Model"] not in SUPPORTED_MODELS:
        errors.append(
            f"Unsupported model '{params['Model']}'. "
            f"Supported models: {', '.join(SUPPORTED_MODELS)}"
        )

    if not errors:
        try:
            model_data = loadmat(params["nnarx_mat"])
        except (OSError, TypeError, ValueError) as error:
            errors.append(f"Neural-network model could not be loaded: {error}")
        else:
            for key in ("layers", "output_scaler"):
                if key not in model_data:
                    errors.append(f"Loaded neural-network model is missing '{key}'")

    for module_name in ("casadi", "torch", "gpytorch", "pyfmi"):
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            errors.append(f"Required dependency '{module_name}' is unavailable: {error}")
        else:
            if module_name == "casadi":
                try:
                    solver_available = module.has_nlpsol("ipopt")
                except (AttributeError, RuntimeError) as error:
                    errors.append(f"IPOPT solver availability could not be checked: {error}")
                else:
                    if not solver_available:
                        errors.append("CasADi IPOPT solver is not available")

    if errors:
        raise RuntimeError("Configuration preflight failed:\n- " + "\n- ".join(errors))


def build_parameters(model="GP"):
    """Build the controller parameter dictionary for the selected model.

    Args:
        model: Strategy name consumed by ``MPC``. Supported values are
            ``"NNARX"``, ``"GP"``, and ``"BLL"``.

    Returns:
        A dictionary containing controller dimensions, timing and operating
        limits, tuning values, model metadata, and the loaded electricity-price
        and heat-load arrays. The configured sample time ``Ts`` is 300 seconds,
        and the prediction horizon ``N`` is 12 steps.
    """
    
    price_data_path = data_file("20250501_20250501_MGP_PrezziZonali_Nord.csv")
    load_data_path = data_file("DHN_ground_truth_shifted.csv")
    _require_file(price_data_path, "electricity-price data")
    _require_file(load_data_path, "heat-load data")

    return {
        "Nb":               3,          # control updates using the block size Nb
        "n_out":            3,
        "n_x":              10,
        "n_inp":            6,
        "tot_hours":        5,
        "seconds_per_hour": 3600,
        "Ts":               300,
        "N":                12,
        "specific_heat_capacity": 4186,
        "q_eb":             3 / 3.6,
        "temperature_slack_max": 20,
        "temperature_delta_max": 5,
        "Tr_max":           66,
        "Tr_min":           45,
        "Ts_max":           75,
        "Ts_max_eb":        72,
        "Ts_min_supply":    68,
        "Ts_min_eb":        65,
        "Ts_max_gb":        75,
        "Ts_min_gb":        65,
        "m_max":            7,
        "m_min":            2,
        "Pb_max":           147e3,
        "Pb_min":           60e3,
        "Pb_max_eb":        38e3,
        "Pb_min_eb":        22e3,
        "beta0_1":          35,
        "beta0_2":          30,
        "beta0_3":          13,
        "epsilon0":         0.4,
        "L_prev0":          100,
        "sigma_epsilon0_2": 0.001,
        "gamma":            0.2,
        "T_ref":            70,
        "c_gas":            0.034,
        "COP":              0.8,
        "eff_EB":           0.8,
        "gamma_exp0":       5,
        "gamma_slack0":     5e3,
        "alpha_slack0":     5e3,
        "P_Shift":          5000,
        "P_Shift_mpc1":     0,
        "P_Shift_mpc2":     0,
        "P_Shift_mpc3":     0,
        "P_Shift_mpc4":     0,
        "Model":            model,
        "T_C0":             4,
        "n_slack":          1,
        "initial_state": np.array([70.2690912858482, 60.786245068179, 2.5]),
        "initial_control": np.array([-32000, -32000, -32000, -32000, 70.5, 70.0]),
        "initial_sigma":   0.05,
        "debug_initial_state": np.array([
            [71.06958, 70.96986, 71.104965, 71.12856],
            [60.12344, 60.224037, 60.187595, 60.183292],
            [2.7702346, 2.7731984, 2.7713318, 2.7692516],
        ]),
        "debug_fmu": "DHN_MC_V3_FMU_EB_60.fmu",
        "control_power_scale": -1e-3,
        "energy_power_scale": 1 / 1000,
        "exploration_slack_tolerance": 1e-3,
        "plant_power_minimum": 10,
        "plant_power_fallback": 32,
        "outlet_temperature_minimum": 45,
        "outlet_temperature_fallback": 72,
        "solver_max_iterations":  int(1e4),
        "solver_print_level":     3,
        "solver_relaxed_print_level": 0,
        "solver_complementarity_tolerance": 1e-5,
        "solver_relaxed_complementarity_tolerance": 1e-4,
        "solver_acceptable_tolerance": 1e-4,
        "solver_acceptable_iterations": 10,
        "solver_mu_strategy": "adaptive",
        "solver_relaxed_objective_scale": 0.01,
        "fmu_relative_tolerance": 1e-4,
        "fmu_absolute_tolerance": 1e-4,
        "fmu_initial_steps": 5,
        "opcua_write_attempts": 5,
        "opcua_retry_delay_seconds": 1,
        "minimum_slack": 1e-8,
        "exploration_min_slack": 1e-6,
        "exploration_max_slack": 1e1,
        "late_slack_weight": 1e-3,
        "bnn_slack_max": np.array([50, 50, 50]),
        "bnn_initial_slack_max": np.array([5, 1, 3]),
        "nnarx_mat": model_file("net_1020.mat"),
        "c_el": np.genfromtxt(
            price_data_path,
            delimiter=";",
            usecols=(2),
            skip_header=2,
        ),
        "Potenza": np.genfromtxt(
            load_data_path,
            delimiter=",",
            usecols=(3, 4, 5, 6),
            skip_header=28,
        ),
    }
