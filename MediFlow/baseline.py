"""
baseline.py — Heuristic baseline agent for Hospital Resource Management.

Uses a severity-first greedy scheduler. No LLM calls.
Emits structured [START], [STEP], [END] logs.
"""

import os
import sys
import json
import requests

ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")


def greedy_action(obs: dict) -> dict:
    """Severity-first greedy assignment."""
    assignments = []
    queue = sorted(
        obs.get("queue", []),
        key=lambda p: (-p["severity"], -p["wait_time"]),
    )
    avail_docs = [d for d in obs.get("doctors", []) if d.get("is_available")]
    avail_beds = [b for b in obs.get("beds", []) if not b.get("is_occupied")]
    avail_surgs = [s for s in obs.get("surgeons", []) if s.get("is_available")]

    used_docs, used_beds = set(), set()

    for p in queue:
        if not avail_docs or not avail_beds:
            break

        # Best-match doctor (specialization first, then any)
        doc = None
        for d in avail_docs:
            if d["id"] not in used_docs and d["specialization"] == p["specialization"]:
                doc = d
                break
        if doc is None:
            for d in avail_docs:
                if d["id"] not in used_docs:
                    doc = d
                    break
        if doc is None:
            continue

        bed = None
        for b in avail_beds:
            if b["id"] not in used_beds:
                bed = b
                break
        if bed is None:
            continue

        use_surg = p.get("requires_surgery", False)
        if use_surg and not avail_surgs:
            continue

        used_docs.add(doc["id"])
        used_beds.add(bed["id"])
        if use_surg:
            avail_surgs.pop(0)

        assignments.append({
            "patient_id": p["id"],
            "doctor_id": doc["id"],
            "bed_id": bed["id"],
            "use_surgery": use_surg,
        })

    return {"assignments": assignments, "reasoning": "greedy severity-first"}


def run_task(task_id: str, seed: int):
    print(f'[START] task_id={task_id}')

    resp = requests.post(
        f"{ENV_URL}/reset",
        json={"seed": seed, "episode_id": task_id},
        timeout=30,
    )
    obs = resp.json().get("observation", resp.json())
    done = obs.get("done", False)
    total_reward = 0.0
    step_num = 0

    while not done:
        action = greedy_action(obs)
        resp = requests.post(
            f"{ENV_URL}/step",
            json={"action": action},
            timeout=30,
        )
        step_data = resp.json()
        obs = step_data.get("observation", step_data)
        reward = step_data.get("reward", obs.get("reward", 0.0))
        done = step_data.get("done", obs.get("done", False))
        total_reward += reward
        step_num += 1

        print(
            f'[STEP] task_id={task_id} step={step_num} '
            f'reward={reward:.4f} total_reward={total_reward:.4f}'
        )

    try:
        score = requests.get(f"{ENV_URL}/score", timeout=10).json().get("score", 0.0)
    except Exception:
        score = 0.0

    print(f'[END] task_id={task_id} total_reward={total_reward:.4f} score={score:.4f}')
    return score


def main():
    print("=" * 60)
    print("Hospital Resource Management — Heuristic Baseline")
    print("=" * 60)

    tasks = [("easy", 42), ("medium", 123), ("hard", 7)]
    scores = {}

    for task_id, seed in tasks:
        try:
            scores[task_id] = run_task(task_id, seed)
        except Exception as e:
            print(f"[ERROR] {task_id}: {e}", file=sys.stderr)
            scores[task_id] = 0.0

    print("\n" + "=" * 60)
    print("BASELINE SCORES:")
    for tid, sc in scores.items():
        print(f"  {tid}: {sc:.4f}")
    print(f"  average: {sum(scores.values()) / len(scores):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
