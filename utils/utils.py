from itertools import count
from pathlib import Path
from typing import Iterable


def build_bomb_mission_outcome(
    bomb_states: Iterable[str],
    bomb_phase_counts: Iterable[int],
    stop_reason: str,
) -> dict:
    """Build the stable mission-result payload written to results.json."""
    states = [str(state).lower() for state in bomb_states]
    phase_counts = [int(count) for count in bomb_phase_counts]
    total = len(states)
    defused = sum(state == "defused" for state in states)
    exploded = sum(state == "exploded" for state in states)

    if total > 0 and defused == total:
        success = True
        reason_code = "all_objectives_completed"
        reason = "All bombs were defused."
    else:
        success = False
        if exploded:
            reason_code = "bombs_exploded"
            reason = f"{exploded} bomb{'s' if exploded != 1 else ''} exploded."
        elif stop_reason == "mission_time_expired":
            reason_code = "mission_time_expired"
            reason = "Not all bombs were defused before mission time expired."
        elif stop_reason == "round_limit_reached":
            reason_code = "round_limit_reached"
            reason = "Not all bombs were defused before the round limit."
        else:
            reason_code = "incomplete_objectives"
            reason = "The mission ended before all bombs were defused."

    return {
        "success": success,
        "status": "accomplished" if success else "failed",
        "reason_code": reason_code,
        "reason": reason,
        "objectives_total": total,
        "objectives_completed": defused,
        "max_score": 10 * sum(phase_counts),
        "details": {
            "bombs_total": total,
            "bombs_defused": defused,
            "bombs_exploded": exploded,
            "phases_total": sum(phase_counts),
        },
    }

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
