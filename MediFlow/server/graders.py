"""
Deterministic graders for each task.
Each grader produces a score in [0.0, 1.0] from episode statistics.
"""


def compute_grader_score(stats: dict) -> float:
    """
    Compute a deterministic score in [0.0, 1.0].

    Components (weights):
      - treated_ratio  (35%): patients treated / total arrivals
      - wait_score     (25%): 1 - normalised avg wait time
      - critical_score (25%): critical patients treated / critical total
      - util_score     (15%): average resource utilization
    """
    total = max(stats.get("total_arrivals", 1), 1)
    treated = stats.get("patients_treated", 0)
    avg_wait = stats.get("avg_wait_time", 10.0)
    crit_treated = stats.get("critical_treated", 0)
    crit_total = max(stats.get("critical_total", 1), 1)
    utilization = stats.get("resource_utilization", 0.0)

    treated_ratio = treated / total
    wait_score = max(0.0, 1.0 - avg_wait / 10.0)
    critical_score = crit_treated / crit_total
    util_score = min(utilization, 1.0)

    score = (
        0.35 * treated_ratio
        + 0.25 * wait_score
        + 0.25 * critical_score
        + 0.15 * util_score
    )
    return round(min(max(score, 0.0), 1.0), 4)
