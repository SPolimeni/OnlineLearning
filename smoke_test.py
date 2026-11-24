import importlib
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# The project modules live under src/; mirror the bootstrap used by the
# runnable entry points so the modules below can be imported.
sys.path.insert(0, os.path.join(ROOT, 'src'))

PROJECT_FILES = [
    'class_MPC',
    'model_strategies',
    'project_paths',
    'Main_OnlineLearningMPC',
]


def validate_example_configuration():
    """Load and validate the representative public GP configuration."""
    from config.parameters import build_parameters, validate_parameters
    from project_paths import fmu_file

    parameters = build_parameters("GP")
    validate_parameters(parameters)
    example_fmu = fmu_file(parameters["debug_fmu"])
    if not os.path.isfile(example_fmu):
        raise FileNotFoundError(f"Example FMU path was not found: {example_fmu}")
    return parameters, example_fmu


def main():
    errors = []
    for module_name in PROJECT_FILES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")

    if errors:
        print('SMOKE TEST FAILED')
        for err in errors:
            print(err)
        raise SystemExit(1)

    try:
        parameters, example_fmu = validate_example_configuration()
    except Exception as exc:
        print("SMOKE TEST FAILED")
        print(f"configuration: {exc}")
        raise SystemExit(1)

    print(
        f"OK: imported {len(PROJECT_FILES)} modules; "
        f"validated {parameters['Model']} configuration and {example_fmu}"
    )


if __name__ == '__main__':
    main()
