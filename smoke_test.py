import importlib
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

PROJECT_FILES = [
    'class_MPC',
    'Main_OnlineLearningMPC',
    'MPC_BNNExp_TestParameters',
    'project_paths',
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
