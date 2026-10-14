"""Runtime parameter construction for the online-learning MPC controller."""

import numpy as np

from project_paths import data_file, model_file


def build_parameters(model="GP"):
    """Build the controller parameter dictionary for the selected model.

    Args:
        model: Strategy name consumed by ``MPC``. Supported values are
            ``"NNARX"``, ``"GP"``, and ``"BNNExp"``.

    Returns:
        A dictionary containing controller dimensions, timing and operating
        limits, tuning values, model metadata, and the loaded electricity-price
        and heat-load arrays. The configured sample time ``Ts`` is 300 seconds,
        and the prediction horizon ``N`` is 12 steps.
    """
    
    return {
        "Nb":               3,          # control updates using the block size Nb
        "n_out":            3,
        "n_x":              10,
        "n_inp":            6,
        "tot_hours":        5,
        "Ts":               300,
        "N":                12,
        "q_eb":             3 / 3.6,
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
        "nnarx_mat": model_file("net_1020.mat"),
        "c_el": np.genfromtxt(
            data_file("20250501_20250501_MGP_PrezziZonali_Nord.csv"),
            delimiter=";",
            usecols=(2),
            skip_header=2,
        ),
        "Potenza": np.genfromtxt(
            data_file("DHN_ground_truth_shifted.csv"),
            delimiter=",",
            usecols=(3, 4, 5, 6),
            skip_header=28,
        ),
    }
