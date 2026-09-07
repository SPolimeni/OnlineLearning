import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, 'utils', 'data')


def data_file(filename):
    """Return a repository-stable path to a runtime data artifact."""
    return os.path.join(DATA_DIR, filename)
