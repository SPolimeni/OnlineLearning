"""Run the repository's publication validation checks."""

import compileall
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_command(label, *arguments):
    """Run one validation stage with the active Python interpreter."""
    print(f"== {label} ==")
    completed = subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main():
    """Compile sources, validate the smoke path, and run all unit tests."""
    print("== Python compilation ==")
    paths = [
        os.path.join(ROOT, "Main_OnlineLearningMPC.py"),
        os.path.join(ROOT, "smoke_test.py"),
        os.path.join(ROOT, "tests", "validate_fmu_baseline.py"),
        os.path.join(ROOT, "tests", "validate_publication.py"),
        os.path.join(ROOT, "src"),
        os.path.join(ROOT, "tests"),
    ]
    if not all(compileall.compile_dir(path, quiet=1) if os.path.isdir(path)
               else compileall.compile_file(path, quiet=1) for path in paths):
        raise SystemExit(1)

    run_command("Smoke test", "smoke_test.py")
    run_command(
        "Unit tests",
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_*.py",
    )
    print("PUBLICATION VALIDATION PASSED")


if __name__ == "__main__":
    main()
