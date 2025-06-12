from itertools import count
from pathlib import Path

def next_experiment_name(base: str, root: Path, model: str, seed: int) -> str:
    """
    Return the first experiment name of the form f"{base}_{k}"
    whose path `<root>/<model>/<name>/seed<seed>/` does **not** exist.
    """
    for k in count():                      # 0, 1, 2, …
        name = f"{base}_{k}"
        exp_dir = root / model / name / f"seed{seed}"
        if not exp_dir.exists():
            return name                    # found an unused slot