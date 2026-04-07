"""
inference.py — LLM-based agent for Hospital Resource Management.
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
HF_TOKEN = os.environ.get("HF_TOKEN")

ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
BENCHMARK_NAME = "Hospital-Resource-Management"

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

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

def wait_for_server(url: str, timeout: int = 60):
    """Wait for the environment server to become responsive to prevent connection crashes."""
    start_time = time.time()
    print(f"Waiting for server at {url}...", file=sys.stderr)
    while time.time() - start_time < timeout:
        try:
            # A simple GET request to check if the port is bound and responding
            requests.get(url, timeout=2)
            print("Server is up!", file=sys.stderr)
            return True
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(1)
    raise RuntimeError("Environment server did not start in time. Healthcheck failed.")

def format_observation(obs: dict) -> str:
    lines = []
    sr = obs.get("situation_report", "")
    if sr:
        lines.append(sr)
        lines.append("")

    q = obs.get("queue", [])
    if q:
        lines.append(f"PATIENT QUEUE ({len(q)} patients):")
        for p in q[:20]:
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
    content = content.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass
    return {"assignments": [], "reasoning": "Failed to parse LLM response"}


def call_llm(observation: dict, retries: int = 2) -> dict:
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


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def run_task(task_id: str, seed: int):
    # Print strictly compliant start log to STDOUT
    log_start(task=task_id, env=BENCHMARK_NAME, model=MODEL_NAME)

    try:
        resp = requests.post(f"{ENV_URL}/reset", json={"seed": seed, "episode_id": task_id}, timeout=30)
        resp.raise_for_status()
        reset_data = resp.json()
    except Exception as e:
        log_end(success=False, steps=0, score=0.0, rewards=[])
        raise RuntimeError(f"Failed to reset environment: {e}")

    obs = reset_data.get("observation", reset_data)
    done = obs.get("done", False)
    
    step_num = 0
    rewards_history = []
    error_msg = None

    while not done:
        step_num += 1
        
        action = call_llm(obs)
        if not action.get("assignments"):
            action = run_heuristic_fallback(obs)

        # Compress action into a single string for logging
        action_str = json.dumps(action).replace('\n', '').replace(' ', '')

        try:
            resp = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=30)
            resp.raise_for_status()
            step_data = resp.json()
            obs = step_data.get("observation", step_data)
            reward = step_data.get("reward", obs.get("reward", 0.0))
            done = step_data.get("done", obs.get("done", False))
        except Exception as e:
            error_msg = str(e).replace('\n', ' ')
            reward = 0.0
            done = True

        rewards_history.append(reward)
        log_step(step=step_num, action=action_str, reward=reward, done=done, error=error_msg)

        if error_msg:
            break

    try:
        score_resp = requests.get(f"{ENV_URL}/score", timeout=10)
        score = score_resp.json().get("score", 0.0)
    except Exception:
        score = 0.0

    # Determine success (customize this threshold if you know the benchmark rules)
    success = score > 0.0 
    
    log_end(success=success, steps=step_num, score=score, rewards=rewards_history)
    return score


def main():
    # Route decorative prints to stderr so they don't break stdout parsing
    print("=" * 60, file=sys.stderr)
    print("Hospital Resource Management — LLM Agent Inference", file=sys.stderr)
    print(f"Model: {MODEL_NAME}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # 1. Wait for environment to wake up
    wait_for_server(ENV_URL)

    tasks = [("easy", 42), ("medium", 123), ("hard", 7)]
    scores = {}

    for task_id, seed in tasks:
        try:
            scores[task_id] = run_task(task_id, seed)
        except Exception as e:
            print(f"[ERROR] task_id={task_id} error={e}", file=sys.stderr)
            scores[task_id] = 0.0

    print("\n" + "=" * 60, file=sys.stderr)
    print("FINAL SCORES:", file=sys.stderr)
    for tid, sc in scores.items():
        print(f"  {tid}: {sc:.4f}", file=sys.stderr)
    print(f"  average: {sum(scores.values()) / len(scores):.4f}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

if __name__ == "__main__":
    main()