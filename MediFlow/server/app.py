"""
FastAPI server for Hospital Resource Management environment.

Exposes /reset, /step, /state, /health, /schema endpoints
compatible with the OpenEnv specification.
"""

import sys
import os

_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SERVER_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

from models import HospitalAction, HospitalObservation, HospitalState
from server.hospital_environment import HospitalEnvironment
from server.graders import compute_grader_score

app = FastAPI(
    title="Hospital Resource Management Environment",
    description="OpenEnv RL environment for hospital triage and resource scheduling",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = HospitalEnvironment()


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    episode_id: Optional[str] = None


class StepRequest(BaseModel):
    action: Dict[str, Any] = {}
    request_id: Optional[str] = None
    timeout_s: Optional[float] = None


@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()):
    task_id = request.episode_id or "easy"
    obs = env.reset(seed=request.seed, task_id=task_id)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }


@app.post("/step")
def step(request: StepRequest = StepRequest()):
    try:
        action = HospitalAction(**request.action)
    except Exception:
        action = HospitalAction()
    obs = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }


@app.get("/state")
def get_state():
    return env.state.model_dump()


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/schema")
def schema():
    return {
        "action": HospitalAction.model_json_schema(),
        "observation": HospitalObservation.model_json_schema(),
        "state": HospitalState.model_json_schema(),
    }


@app.get("/metadata")
def metadata():
    return {
        "name": "MediFlow",
        "description": "Hospital Resource Management RL Environment",
        "version": "1.0.0",
        "author": "AI-see-U",
    }


@app.get("/score")
def score():
    return {"score": env.get_grader_score()}



def main():
    import uvicorn
    # Make sure to run it with exactly "server.app:app" or just "app" depending on where it's run
    # For project.scripts, "server.app:app" works if server is a module
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()