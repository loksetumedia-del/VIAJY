"""
app.py — OpenEnv FastAPI Server
================================
Exposes the GridWorld-v1 environment via HTTP REST API:
  POST /reset        → resets env, returns initial observation
  POST /step         → takes action, returns (obs, reward, done, info)
  GET  /state        → returns full environment state
  GET  /             → health check

The checker calls POST /reset first — this MUST work.
"""

import os
import sys
import json
import time
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
from env import make

# ── Logging (START/STEP/END format) ───────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_event(event_type: str, data: dict):
    payload = {"type": event_type, "timestamp": time.time(), **data}
    print(json.dumps(payload), flush=True)

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="GridWorld-v1 OpenEnv",
    description="A 5x5 grid navigation environment implementing the OpenEnv API.",
    version="1.0.0",
)

# Global env instance (single-session)
_env = make("GridWorld-v1")
_env.reset()

# ── Request models ─────────────────────────────────────────────────────────────
class StepRequest(BaseModel):
    action: int  # 0=Up, 1=Right, 2=Down, 3=Left

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    """Health check."""
    return {"status": "ok", "env": "GridWorld-v1", "version": "1.0.0"}


@app.post("/reset")
def reset():
    """
    Reset the environment to its initial state.
    Returns the initial observation.
    """
    global _env
    _env = make("GridWorld-v1")
    obs = _env.reset()
    state = _env.state()

    log_event("START", {
        "env_id": "GridWorld-v1",
        "config": {
            "grid_size": state["grid_size"],
            "max_steps": state["max_steps"],
            "goal_pos": list(state["goal_pos"]),
            "hazards": [list(h) for h in state["hazards"]],
        }
    })

    return {
        "observation": list(obs),
        "state": {
            "agent_pos": list(state["agent_pos"]),
            "goal_pos": list(state["goal_pos"]),
            "hazards": [list(h) for h in state["hazards"]],
            "grid_size": state["grid_size"],
            "steps": state["steps"],
            "max_steps": state["max_steps"],
            "done": state["done"],
            "cumulative_reward": state["cumulative_reward"],
        }
    }


@app.post("/step")
def step(body: StepRequest):
    """
    Take one step in the environment.
    action: 0=Up, 1=Right, 2=Down, 3=Left
    """
    if body.action not in (0, 1, 2, 3):
        raise HTTPException(status_code=400, detail=f"Invalid action {body.action}. Must be 0-3.")

    if _env.state()["done"]:
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset first.")

    obs, reward, done, info = _env.step(body.action)

    log_event("STEP", {
        "step": info["steps"],
        "action": body.action,
        "obs": list(obs),
        "reward": reward,
        "done": done,
        "info": info,
    })

    if done:
        success = (tuple(obs) == tuple(_env.state()["goal_pos"]))
        log_event("END", {
            "total_reward": _env.state()["cumulative_reward"],
            "total_steps": info["steps"],
            "success": success,
            "message": "Goal reached!" if success else "Episode ended.",
        })

    return {
        "observation": list(obs),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    """Return the full current state of the environment."""
    s = _env.state()
    return {
        "agent_pos": list(s["agent_pos"]),
        "goal_pos": list(s["goal_pos"]),
        "hazards": [list(h) for h in s["hazards"]],
        "grid_size": s["grid_size"],
        "steps": s["steps"],
        "max_steps": s["max_steps"],
        "done": s["done"],
        "cumulative_reward": s["cumulative_reward"],
        "action_space": s["action_space"],
        "obs_space": list(s["obs_space"]),
    }


@app.get("/render")
def render():
    """Return ASCII render of the current grid."""
    return {"render": _env.render()}


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
