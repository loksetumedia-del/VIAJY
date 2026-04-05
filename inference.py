"""
inference.py  —  OpenEnv Competition Submission
================================================
GridWorld-v1: A 5x5 navigation environment where an AI agent
learns to reach a goal while avoiding hazards.

Logging format: START / STEP / END  (required by checker)
LLM calls     : OpenAI client via env vars (API_BASE_URL, MODEL_NAME)
"""

import os
import sys
import json
import time
import random
from openai import OpenAI

# ── Environment import ────────────────────────────────────────────────────────
from env import make

# ── OpenAI client setup (configured via environment variables) ────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")                  # no default for HF_TOKEN

# Optional Docker image name (for from_docker_image())
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "placeholder",
)


# ── Structured stdout logger ──────────────────────────────────────────────────

def log_start(env_id: str, config: dict):
    """Emit START log line."""
    payload = {
        "type"   : "START",
        "env_id" : env_id,
        "config" : config,
        "timestamp": time.time(),
    }
    print(json.dumps(payload), flush=True)


def log_step(step: int, action: int, obs, reward: float, done: bool, info: dict):
    """Emit STEP log line."""
    payload = {
        "type"  : "STEP",
        "step"  : step,
        "action": action,
        "obs"   : list(obs),
        "reward": reward,
        "done"  : done,
        "info"  : info,
    }
    print(json.dumps(payload), flush=True)


def log_end(total_reward: float, total_steps: int, success: bool, message: str = ""):
    """Emit END log line."""
    payload = {
        "type"        : "END",
        "total_reward": total_reward,
        "total_steps" : total_steps,
        "success"     : success,
        "message"     : message,
        "timestamp"   : time.time(),
    }
    print(json.dumps(payload), flush=True)


# ── LLM-based policy ─────────────────────────────────────────────────────────

def llm_select_action(obs: tuple, env_state: dict) -> int:
    """
    Call the LLM to select the best action given current state.
    Falls back to greedy heuristic if LLM call fails.
    """
    agent   = obs
    goal    = env_state["goal_pos"]
    hazards = env_state["hazards"]
    steps   = env_state["steps"]
    max_s   = env_state["max_steps"]

    prompt = (
        f"You are controlling an agent in a {env_state['grid_size']}x{env_state['grid_size']} grid.\n"
        f"Agent position : {agent}\n"
        f"Goal position  : {goal}\n"
        f"Hazard cells   : {hazards}\n"
        f"Steps taken    : {steps}/{max_s}\n\n"
        "Choose the BEST action to reach the goal while avoiding hazards.\n"
        "Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT\n"
        "Reply with ONLY the action number (0, 1, 2, or 3). No explanation."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        text   = response.choices[0].message.content.strip()
        action = int(text[0])
        if action in (0, 1, 2, 3):
            return action
    except Exception as e:
        pass  # fall through to heuristic

    return _greedy_action(agent, goal, hazards)


def _greedy_action(agent: tuple, goal: tuple, hazards: list) -> int:
    """Simple greedy heuristic: move toward goal, avoid hazards."""
    ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    ar, ac  = agent
    gr, gc  = goal
    best_action = 0
    best_dist   = float("inf")

    for a, (dr, dc) in enumerate(ACTIONS):
        nr = max(0, min(4, ar + dr))
        nc = max(0, min(4, ac + dc))
        if [nr, nc] in hazards or (nr, nc) in hazards:
            continue
        d = abs(nr - gr) + abs(nc - gc)
        if d < best_dist:
            best_dist   = d
            best_action = a

    return best_action


# ── Main episode runner ───────────────────────────────────────────────────────

def run_episode(use_llm: bool = True) -> dict:
    """Run one complete episode and return results."""
    env     = make("GridWorld-v1")
    obs     = env.reset()
    state   = env.state()

    log_start(
        env_id = "GridWorld-v1",
        config = {
            "grid_size": state["grid_size"],
            "max_steps": state["max_steps"],
            "goal_pos" : state["goal_pos"],
            "hazards"  : state["hazards"],
            "model"    : MODEL_NAME if use_llm else "greedy",
        },
    )

    done          = False
    total_reward  = 0.0
    step_count    = 0

    while not done:
        if use_llm:
            action = llm_select_action(obs, env.state())
        else:
            action = _greedy_action(obs, state["goal_pos"], list(state["hazards"]))

        obs, reward, done, info = env.step(action)
        step_count  += 1
        total_reward = round(total_reward + reward, 4)

        log_step(
            step   = step_count,
            action = action,
            obs    = obs,
            reward = reward,
            done   = done,
            info   = info,
        )

    success = (obs == state["goal_pos"])
    log_end(
        total_reward = total_reward,
        total_steps  = step_count,
        success      = success,
        message      = "Goal reached!" if success else "Episode ended without reaching goal.",
    )

    return {
        "total_reward": total_reward,
        "total_steps" : step_count,
        "success"     : success,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    use_llm = "--llm" in sys.argv
    result  = run_episode(use_llm=use_llm)
    sys.exit(0 if result["success"] else 1)
