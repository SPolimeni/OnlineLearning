import os

# This module lives at src/project_paths.py, so the source root is its own
# directory and the repository root is one level above it.
SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_ROOT)
DATA_DIR = os.path.join(SRC_ROOT, 'utils', 'data')
MODELS_DIR = os.path.join(REPO_ROOT, 'NNmodels')
FMUS_DIR = os.path.join(REPO_ROOT, 'FMUs')
SOLUTIONS_DIR = os.path.join(REPO_ROOT, 'Solutions')
LOGS_DIR = os.path.join(REPO_ROOT, 'LogsFolder')


def data_file(filename):
    """Return a repository-stable path to a runtime data artifact."""
    return os.path.join(DATA_DIR, filename)


def model_file(filename):
    """Return a repository-stable path to a bundled neural-network model."""
    return os.path.join(MODELS_DIR, filename)


def fmu_file(filename):
    """Return a repository-stable path to a bundled FMU."""
    return os.path.join(FMUS_DIR, filename)


def solutions_dir(model, date_folder):
    """Return the output directory for a controller run."""
    return os.path.join(SOLUTIONS_DIR, model, date_folder)
