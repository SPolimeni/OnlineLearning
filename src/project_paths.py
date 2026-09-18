import os

# This module lives at src/project_paths.py, so the source root is its own
# directory and the repository root is one level above it.
SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_ROOT)
DATA_DIR = os.path.join(SRC_ROOT, 'utils', 'data')


def data_file(filename):
    """Return a repository-stable path to a runtime data artifact."""
    return os.path.join(DATA_DIR, filename)
