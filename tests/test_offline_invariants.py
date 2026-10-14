import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from class_MPC import _STRATEGY_FACTORY
from config.parameters import build_parameters
from project_paths import REPO_ROOT, fmu_file, model_file, solutions_dir


class OfflineInvariantsTest(unittest.TestCase):
    def test_parameter_dimensions_and_model_path(self):
        for model in ("NNARX", "GP", "BNNExp"):
            params = build_parameters(model)
            self.assertEqual(params["Model"], model)
            self.assertEqual(params["n_out"], 3)
            self.assertEqual(params["n_inp"], 6)
            self.assertEqual(params["n_x"], 10)
            self.assertEqual(params["N"] % params["Nb"], 0)
            self.assertEqual(params["nnarx_mat"], model_file("net_1020.mat"))

    def test_strategy_selection_contract(self):
        self.assertEqual(set(_STRATEGY_FACTORY), {"NNARX", "GP", "BNNExp"})

    def test_repository_paths_are_independent_of_cwd(self):
        self.assertTrue(model_file("net_1020.mat").startswith(REPO_ROOT))
        self.assertTrue(fmu_file("DHN_MC_V3_FMU_EB_60.fmu").startswith(REPO_ROOT))
        self.assertEqual(
            solutions_dir("GP", "1410"),
            os.path.join(REPO_ROOT, "Solutions", "GP", "1410"),
        )


if __name__ == "__main__":
    unittest.main()