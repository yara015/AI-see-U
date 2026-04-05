"""
inference.py — LLM-based agent for Hospital Resource Management.

Uses the OpenAI-compatible client pointed at a Hugging Face model.
Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables.
Emits structured [START], [STEP], [END] logs as required by the hackathon.
"""

import os
import sys
import json
import time
import requests

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "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")

# The environment server URL (your HF Space or local server)
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

# ---------------------------------------------------------------------------
# OpenAI client pointing at HF Inference API
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert hospital resource manager. Each turn you receive the current
state of a hospital (patient queue, doctor availability, bed availability, surgeon
availability, medicine inventory) and must decide which patients to assign to which
doctors and beds.

RULES:
1. PRIORITIZE critical patients (severity 4-5) — they must be treated ASAP.
2. MATCH doctor specialization to patient specialization when possible (faster treatment).
3. Only use surgery flag when the patient requires_surgery=true.
4. Check that doctor is_available=true, bed is_occupied=false before assigning.
5. Check surgeon is_available=true before assigning surgery cases.
6. You can make MULTIPLE assignments per step (assign several patients at once).
7. If no valid assignments possible, return empty assignments list.

OUTPUT FORMAT — you MUST return valid JSON and nothing else:
{
  "assignments": [
    {"patient_id": <int>, "doctor_id": <int>, "bed_id": <int>, "use_surgery": <bool>}
  ],
  "reasoning": "<brief explanation of your decisions>"
}

Be efficient. Treat as many patients as possible each step. Do NOT leave doctors
and beds idle when there are patients waiting.
"""


def format_observation(obs: dict) -> str:
    """Convert observation dict into a concise prompt for the LLM."""
    lines = []
    sr = obs.get("situation_report", "")
    if sr:
        lines.append(sr)
        lines.append("")

    q = obs.get("queue", [])
    if q:
        lines.append(f"PATIENT QUEUE ({len(q)} patients):")
        for p in q[:20]:  # limit to avoid token explosion
            surg = " [SURGERY]" if p.get("requires_surgery") else ""
            lines.append(
                f"  Patient {p['id']}: severity={p['severity']}, "
                f"spec={p['specialization']}, duration={p['treatment_duration']}, "
                f"wait={p['wait_time']}, med={p['medicine_needed']}{surg}"
            )
        if len(q) > 20:
            lines.append(f"  ... and {len(q) - 20} more")
    else:
        lines.append("PATIENT QUEUE: empty")

    lines.append("")
    docs = obs.get("doctors", [])
    lines.append("DOCTORS:")
    for d in docs:
        status = "AVAILABLE" if d.get("is_available") else "BUSY"
        lines.append(
            f"  Doctor {d['id']}: spec={d['specialization']}, "
            f"skill={d['skill_level']}, {status}"
        )

    surgs = obs.get("surgeons", [])
    lines.append("SURGEONS:")
    for s in surgs:
        status = "AVAILABLE" if s.get("is_available") else "BUSY"
        lines.append(f"  Surgeon {s['id']}: {status}")

    beds = obs.get("beds", [])
    avail_beds = [b for b in beds if not b.get("is_occupied")]
    lines.append(f"BEDS: {len(avail_beds)}/{len(beds)} available")
    if avail_beds:
        ids = [str(b["id"]) for b in avail_beds[:15]]
        lines.append(f"  Available IDs: {', '.join(ids)}")

    meds = obs.get("medicines", {})
    lines.append(f"MEDICINES: {json.dumps(meds)}")

    lines.append(f"\nStep {obs.get('current_step', 0)}/{obs.get('max_steps', 48)}")
    return "\n".join(lines)


def parse_llm_response(content: str) -> dict:
    """Parse LLM response to extract action JSON."""
    content = content.strip()
    # Try to extract JSON from markdown code blocks
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(content)
        return data
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass
    return {"assignments": [], "reasoning": "Failed to parse LLM response"}


def call_llm(observation: dict, retries: int = 2) -> dict:
    """Call the LLM with the observation and return parsed action."""
    prompt = format_observation(observation)
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            content = response.choices[0].message.content
            return parse_llm_response(content)
        except Exception as e:
            if attempt < retries:
                time.sleep(2)
                continue
            print(f"  LLM call failed after {retries + 1} attempts: {e}", file=sys.stderr)
            return {"assignments": [], "reasoning": f"LLM error: {e}"}


def run_heuristic_fallback(obs: dict) -> dict:
    """Simple greedy fallback if LLM fails."""
    assignments = []
    queue = obs.get("queue", [])
    doctors = obs.get("doctors", [])
    beds = obs.get("beds", [])
    surgeons = obs.get("surgeons", [])

    sorted_q = sorted(queue, key=lambda p: (-p["severity"], -p["wait_time"]))
    avail_docs = [d for d in doctors if d.get("is_available")]
    avail_beds = [b for b in beds if not b.get("is_occupied")]
    avail_surgs = [s for s in surgeons if s.get("is_available")]

    used_docs = set()
    used_beds = set()

    for p in sorted_q:
        if not avail_docs or not avail_beds:
            break
        # Find best doc
        best_doc = None
        for d in avail_docs:
            if d["id"] not in used_docs:
                if d["specialization"] == p["specialization"]:
                    best_doc = d
                    break
        if best_doc is None:
            for d in avail_docs:
                if d["id"] not in used_docs:
                    best_doc = d
                    break
        if best_doc is None:
            continue

        best_bed = None
        for b in avail_beds:
            if b["id"] not in used_beds:
                best_bed = b
                break
        if best_bed is None:
            continue

        use_surg = p.get("requires_surgery", False)
        if use_surg and not avail_surgs:
            continue

        used_docs.add(best_doc["id"])
        used_beds.add(best_bed["id"])
        if use_surg and avail_surgs:
            avail_surgs.pop(0)

        assignments.append({
            "patient_id": p["id"],
            "doctor_id": best_doc["id"],
            "bed_id": best_bed["id"],
            "use_surgery": use_surg,
        })

    return {"assignments": assignments, "reasoning": "heuristic fallback"}


def run_task(task_id: str, seed: int):
    """Run a single task and return the score."""
    print(f'[START] task_id={task_id}')

    # Reset environment
    resp = requests.post(
        f"{ENV_URL}/reset",
        json={"seed": seed, "episode_id": task_id},
        timeout=30,
    )
    reset_data = resp.json()
    obs = reset_data.get("observation", reset_data)
    done = obs.get("done", False)
    total_reward = 0.0
    step_num = 0

    while not done:
        # Get action from LLM
        action = call_llm(obs)

        # Validate — if LLM returned nothing useful, use heuristic
        if not action.get("assignments"):
            action = run_heuristic_fallback(obs)

        # Step
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
            f'reward={reward:.4f} total_reward={total_reward:.4f} '
            f'queue_size={len(obs.get("queue", []))} '
            f'treated={obs.get("stats", {}).get("patients_treated", 0)}'
        )

    # Get final score
    try:
        score_resp = requests.get(f"{ENV_URL}/score", timeout=10)
        score = score_resp.json().get("score", 0.0)
    except Exception:
        score = 0.0

    print(
        f'[END] task_id={task_id} total_reward={total_reward:.4f} '
        f'score={score:.4f} steps={step_num}'
    )
    return score


def main():
    print("=" * 60)
    print("Hospital Resource Management — LLM Agent Inference")
    print(f"Model: {MODEL_NAME}")
    print(f"Environment: {ENV_URL}")
    print("=" * 60)

    tasks = [
        ("easy", 42),
        ("medium", 123),
        ("hard", 7),
    ]

    scores = {}
    for task_id, seed in tasks:
        try:
            scores[task_id] = run_task(task_id, seed)
        except Exception as e:
            print(f"[ERROR] task_id={task_id} error={e}", file=sys.stderr)
            scores[task_id] = 0.0

    print("\n" + "=" * 60)
    print("FINAL SCORES:")
    for tid, sc in scores.items():
        print(f"  {tid}: {sc:.4f}")
    print(f"  average: {sum(scores.values()) / len(scores):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
