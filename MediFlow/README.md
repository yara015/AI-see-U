---
title: Hospital Resource Management
emoji: 🏥
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
license: bsd-3-clause
---

# 🏥 Hospital Resource Management — OpenEnv RL Environment

An OpenEnv-compatible reinforcement learning environment that simulates hospital triage and resource scheduling. An agent must manage doctors, surgeons, beds, and medicine inventory to treat stochastically-arriving patients while minimizing wait times and prioritizing critical cases.

## Why This Matters

Hospital resource scheduling is a **$100B+ real-world problem**. Every emergency department performs manual triage daily. This environment models the core decision-making challenge: allocating scarce medical resources under uncertainty to maximize patient outcomes.

> ⚠️ **Disclaimer**: This is a SIMULATION for RL research. It does NOT provide real medical advice.

## Environment Overview

### Entities
| Entity | Description |
|--------|-------------|
| **Patients** | Arrive stochastically (Poisson). Have severity (1-5), specialization, surgery needs, medicine requirements |
| **Doctors** | Fixed pool with specializations and skill levels. Become busy during treatments |
| **Surgeons** | Scarce resource required for surgical cases only |
| **Beds** | Capacity-constrained. Required for every treatment |
| **Medicines** | 5 types with limited inventory, slow replenishment |

### Action Space
```json
{
  "assignments": [
    {"patient_id": 1, "doctor_id": 0, "bed_id": 3, "use_surgery": false}
  ],
  "reasoning": "Assigned critical cardiology patient to specialist"
}
```

### Observation Space
Full hospital state: patient queue, doctor/surgeon/bed availability, medicine inventory, episode statistics, and a natural-language situation report.

### Reward Function
Dense, multi-component reward signal:
- ✅ **+0.10** per patient treated
- ✅ **+0.05** speed bonus (faster treatment = higher reward)
- ✅ **+0.08** critical patient priority bonus
- ❌ **-0.01×severity×wait** waiting penalty (scaled by severity)
- ❌ **-0.15** critical patient waiting >3 steps
- ❌ **-0.05** invalid assignment
- ❌ **-0.01** idle resources when queue exists

## Tasks (3 Difficulty Levels)

| Task | Doctors | Surgeons | Beds | Arrival Rate | Critical % | Steps |
|------|---------|----------|------|-------------|-----------|-------|
| **Easy** | 6 | 3 | 15 | λ=1.5 | 10% | 24 |
| **Medium** | 5 | 2 | 10 | λ=2.5 | 25% | 48 |
| **Hard** | 4 | 1 | 8 | λ=4.0 | 40% | 48 |

## Grader

Deterministic scoring in [0.0, 1.0]:
- **35%** — Patients treated ratio
- **25%** — Wait time score (lower = better)
- **25%** — Critical patient response rate
- **15%** — Resource utilization

## Quick Start

### Run locally
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run baseline
```bash
python baseline.py
```

### Run LLM agent
```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/novita/v3/openai
export MODEL_NAME=deepseek-ai/DeepSeek-V3-0324
python inference.py
```

### Docker
```bash
docker build -t MediFlow .
docker run -p 7860:7860 MediFlow
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode. Body: `{"seed": 42, "episode_id": "easy"}` |
| `/step` | POST | Take action. Body: `{"action": {...}}` |
| `/state` | GET | Get current state |
| `/health` | GET | Health check |
| `/schema` | GET | Action/Observation JSON schemas |
| `/score` | GET | Get grader score for current episode |

## Project Structure
```
├── models.py              # Pydantic Action/Observation/State
├── server/
│   ├── app.py             # FastAPI server
│   ├── MediFlowironment.py  # Core simulation
│   ├── entities.py        # Internal dataclasses
│   ├── tasks.py           # Task configurations
│   └── graders.py         # Scoring graders
├── inference.py           # LLM agent
├── baseline.py            # Heuristic baseline
├── openenv.yaml           # OpenEnv manifest
├── Dockerfile             # Container definition
└── requirements.txt       # Dependencies
```

## License
BSD 3-Clause
