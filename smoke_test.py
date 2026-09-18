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

    print(f'OK: imported {len(PROJECT_FILES)} modules from {ROOT}')


if __name__ == '__main__':
    main()
