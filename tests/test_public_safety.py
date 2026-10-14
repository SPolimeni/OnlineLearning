import json
import os
import sys
import unittest
from unittest.mock import patch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import Main_OnlineLearningMPC
from runtime.controller_runner import ControllerRunner


class PublicSafetyTest(unittest.TestCase):
    def test_cli_defaults_to_debug_and_preserves_model(self):
        args = Main_OnlineLearningMPC.parse_args(["--Model", "BNNExp"])
        self.assertTrue(args.Debug)
        self.assertFalse(args.Live)
        self.assertEqual(args.Model, "BNNExp")

    def test_explicit_live_request_is_rejected_before_building_parameters(self):
        with patch.object(Main_OnlineLearningMPC, "build_parameters") as build_parameters:
            with self.assertRaisesRegex(RuntimeError, "not enabled for public operation"):
                Main_OnlineLearningMPC.main(["--Live"])
        build_parameters.assert_not_called()

    def test_runner_rejects_live_before_constructing_mpc(self):
        with patch("runtime.controller_runner.MPC") as mpc:
            with self.assertRaisesRegex(RuntimeError, "not enabled for public operation"):
                ControllerRunner({}, debug=False)
        mpc.assert_not_called()

    def test_configuration_files_use_synthetic_identifiers(self):
        config_dir = os.path.join(ROOT, "src", "utils", "InfluxDB", "ConfigData")
        for filename in ("DBdataConfig.json", "Variabili_Database_Influx.json"):
            with open(os.path.join(config_dir, filename), encoding="utf-8") as handle:
                config = json.load(handle)
            for section in config.values():
                for node_id, metadata in section.items():
                    self.assertTrue(node_id.startswith("ns=0;s=PUBLIC.SYNTHETIC."))
                    self.assertTrue(metadata["Tag"].startswith("synthetic_tag_"))
                    self.assertEqual(metadata["Zone"], "synthetic_zone")
                    self.assertEqual(metadata["Measurement"], "synthetic_measurement")


if __name__ == "__main__":
    unittest.main()