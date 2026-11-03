"""Read-only development features appended to edit_api inside the isolated jail."""
import json
from pathlib import Path
import numpy as np

FEATURES = json.loads(Path('/features/manifest.json').read_text())


def feature(name):
    if name not in FEATURES:
        raise ValueError('feature not allowlisted')
    return np.load('/features/'+FEATURES[name]['file'], allow_pickle=False).copy()
