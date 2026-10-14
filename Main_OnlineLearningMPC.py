"""Command-line entry point for the online-learning MPC controller."""

import os
import sys
import argparse


# Keep direct execution from the repository root working without installation.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from config.parameters import build_parameters
from runtime.controller_runner import ControllerRunner


def parse_args(argv=None):
    """Parse command-line options for the runtime entry point."""
    parser = argparse.ArgumentParser(description="Run script in debug or normal mode")
    parser.add_argument(
        "--Model",
        type=str,
        default="GP",
        choices=("NNARX", "GP", "BNNExp"),
        help="Set the model type: 'NNARX', 'GP', or 'BNNExp'",
    )
    parser.add_argument(
        "--Debug",
        action="store_true",
        default=False,
        help="Run the script in debug mode with simulated data with FMU",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Parse options, build parameters, and start the selected runtime."""
    args        = parse_args(argv)
    parameters  = build_parameters(args.Model)
    ControllerRunner(parameters, debug=args.Debug).run()


if __name__ == "__main__":
    main()
