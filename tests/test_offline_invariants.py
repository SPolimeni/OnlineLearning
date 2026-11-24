import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from class_MPC import MPC, _STRATEGY_FACTORY
from config.parameters import (
    build_parameters,
    load_electricity_prices,
    load_heat_loads,
    validate_model_metadata,
    validate_parameter_guards,
    validate_parameters,
)
from project_paths import REPO_ROOT, fmu_file, model_file, solutions_dir


class OfflineInvariantsTest(unittest.TestCase):
    def test_parameter_dimensions_and_model_path(self):
        for model in ("NNARX", "GP", "BLL"):
            params = build_parameters(model)
            self.assertEqual(params["Model"], model)
            self.assertEqual(params["n_out"], 3)
            self.assertEqual(params["n_inp"], 6)
            self.assertEqual(params["n_x"], 10)
            self.assertEqual(params["N"] % params["Nb"], 0)
            self.assertEqual(params["nnarx_mat"], model_file("NARX_net.mat"))

    def test_repository_data_parsing(self):
        params = build_parameters("GP")
        self.assertEqual(params["c_el"].shape, (24,))
        self.assertAlmostEqual(params["c_el"][0], 106.20)
        self.assertEqual(params["Potenza"].shape, (262, 4))
        np.testing.assert_allclose(
            params["Potenza"][0],
            [-39788.454710909, -34563.2145594375, -38308.7160827516, -33255.2628171096],
        )

    def test_data_parsers_reject_non_numeric_values(self):
        with tempfile.TemporaryDirectory() as directory:
            price_path = os.path.join(directory, "prices.csv")
            with open(price_path, "w", encoding="utf-8") as handle:
                handle.write("Data;Ora;EUR/MWh\n01/05/2025;1;not-a-number\n")
            with self.assertRaisesRegex(ValueError, "finite numeric"):
                load_electricity_prices(price_path)

            load_path = os.path.join(directory, "loads.csv")
            with open(load_path, "w", encoding="utf-8") as handle:
                handle.writelines(["header\n"] * 28)
                handle.write("0,0,0,1,2,not-a-number,4\n")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_heat_loads(load_path)

    def test_strategy_selection_contract(self):
        self.assertEqual(set(_STRATEGY_FACTORY), {"NNARX", "GP", "BLL"})

    def test_configuration_preflight(self):
        validate_parameters(build_parameters("GP"))

    def test_input_and_model_metadata_validation(self):
        params = build_parameters("GP")
        invalid_state = params.copy()
        invalid_state["initial_state"] = np.zeros(2)
        with self.assertRaisesRegex(RuntimeError, "initial_state"):
            validate_parameters(invalid_state)

        with self.assertRaisesRegex(ValueError, "missing 'output_scaler'"):
            validate_model_metadata({"layers": np.empty((1, 1))}, 10, 3)

    def test_parameter_guards_reject_invalid_bounds_and_short_data(self):
        invalid_bounds = build_parameters("GP")
        invalid_bounds["Tr_min"] = invalid_bounds["Tr_max"] + 1
        with self.assertRaisesRegex(ValueError, "Tr_min"):
            validate_parameter_guards(invalid_bounds)

        short_data = build_parameters("GP")
        short_data["Potenza"] = short_data["Potenza"][:2, :]
        with self.assertRaisesRegex(ValueError, "at least"):
            validate_parameter_guards(short_data)

    def test_runtime_dimension_validation(self):
        controller = MPC(build_parameters("GP"))
        with self.assertRaisesRegex(ValueError, "state history xk"):
            controller.validate_runtime_dimensions(
                0,
                [[0.0], [0.0]],
                [[0.0] * 4] * 6,
                [[0.0] * 3],
            )

    def test_repository_paths_are_independent_of_cwd(self):
        self.assertTrue(model_file("NARX_net.mat").startswith(REPO_ROOT))
        self.assertTrue(fmu_file("DHN_sim.fmu").startswith(REPO_ROOT))
        self.assertEqual(
            solutions_dir("GP", "1410"),
            os.path.join(REPO_ROOT, "Solutions", "GP", "1410"),
        )

    def test_solver_failure_reuses_last_feasible_plan(self):
        controller = MPC(build_parameters("GP"))

        class FailingOpti:
            def __init__(self):
                self.solve_calls = 0

            def set_initial(self, *args):
                pass

            def minimize(self, *args):
                pass

            def solver(self, *args):
                pass

            def solve(self):
                self.solve_calls += 1
                raise RuntimeError("forced solver failure")

        failing_opti = FailingOpti()

        def configure_fake_variables():
            controller.opti = failing_opti
            controller.u = np.zeros((controller.n_inp, controller.n_uopt))
            controller.y = np.zeros((controller.n_out, controller.N))
            controller.y_nnarx = np.zeros((controller.n_out, controller.N))
            controller.Pb = np.zeros((2, controller.N))
            controller.J = 0.0
            controller.slack_y = np.zeros((controller.n_slack, 1))

        t = controller.T_C0
        xk = np.tile(controller.initial_state.reshape(-1, 1), (1, t + 1))
        uk = np.tile(controller.initial_control.reshape(-1, 1), (1, t + 1))
        y_rnn = np.tile(controller.initial_state, (t + 1, 1))
        expected_plan = controller.u_prec.copy()
        expected_plan[0:4, 0] = controller.Potenze[:, t] + controller.P_Shift_mpc1

        with patch.object(controller, "SetOptimizationVariables", configure_fake_variables), \
             patch.object(controller, "SetConstraints"), \
             patch.object(controller, "SetCostFunction"):
            result = controller.mpc_controller(t, controller.T_C0, xk, uk, y_rnn, flag=0)

        u_opt, slack, y_nnarx, _, sigma_pred, slack_explo, sol_feas = result
        self.assertEqual(failing_opti.solve_calls, 2)
        self.assertEqual(sol_feas, 3)
        np.testing.assert_allclose(u_opt, expected_plan[:, 0])
        np.testing.assert_allclose(controller.u_prec, expected_plan)
        np.testing.assert_allclose(slack, np.zeros((controller.n_slack, 1)))
        np.testing.assert_allclose(y_nnarx, controller.y_prec)
        np.testing.assert_allclose(sigma_pred, controller.sigma_prec)
        self.assertEqual(slack_explo, 0)


if __name__ == "__main__":
    unittest.main()