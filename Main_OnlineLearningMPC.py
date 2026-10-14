"""Command-line entry point for the online-learning MPC controller."""

import os
import sys
import argparse


# Keep direct execution from the repository root working without installation.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from config.parameters import build_parameters
from runtime.controller_runner import ControllerRunner


def parse_args(argv=None):
    """Parse command-line options for the runtime entry point.

    Args:
        argv: Optional argument sequence. When omitted, ``argparse`` reads the
            process command line.

    Returns:
        The parsed model-selection and debug-mode options.
    """
    parser = argparse.ArgumentParser(description="Run the controller in offline debug mode")
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
        default=True,
        help="Run the script in debug mode with simulated FMU data (default)",
    )
    parser.add_argument(
        "--Live",
        action="store_true",
        help="Request the retained live integration boundary; public live operation is disabled",
    )
    args = parser.parse_args(argv)
    args.Debug = not args.Live
    return args


def main(argv=None):
    """Build the selected controller configuration and start its runtime.

    Debug mode is the safe public default. Live mode is retained only as an
    explicit, disabled implementation boundary.
    """
    args = parse_args(argv)
    if not args.Debug:
        raise RuntimeError(
            "Live integration is illustrative and read-only; it is not enabled "
            "for public execution."
        )
    parameters  = build_parameters(args.Model)
    ControllerRunner(parameters, debug=args.Debug).run()


if __name__ == "__main__":
    main()
