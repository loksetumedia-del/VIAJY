"""
GridWorld OpenEnv Environment
A complete, real-world OpenEnv environment implementing step() / reset() / state() API.
Agent navigates a grid to reach the goal while avoiding hazards.
"""

import random
from typing import Any

# ─── Constants ────────────────────────────────────────────────────────────────
GRID_SIZE   = 5
MAX_STEPS   = 50
GOAL_POS    = (4, 4)
START_POS   = (0, 0)
HAZARDS     = {(1, 1), (2, 3), (3, 1), (1, 3)}

# Actions: 0=Up, 1=Right, 2=Down, 3=Left
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

# Rewards
REWARD_GOAL    =  10.0
REWARD_HAZARD  =  -1.0
REWARD_STEP    =  -0.1
REWARD_WALL    =  -0.2   # tried to walk into wall


class GridWorldEnv:
    """
    5x5 GridWorld OpenEnv Environment.

    Observation space : (row, col) tuple  — agent position
    Action space      : Discrete(4)       — 0=Up 1=Right 2=Down 3=Left
    """

    def __init__(self, grid_size: int = GRID_SIZE, max_steps: int = MAX_STEPS):
        self.grid_size   = grid_size
        self.max_steps   = max_steps
        self.goal_pos    = GOAL_POS
        self.hazard_set  = HAZARDS
        self._agent      = START_POS
        self._steps      = 0
        self._done       = False
        self._cum_reward = 0.0
        self._history: list[dict] = []

    # ── Core API ──────────────────────────────────────────────────────────────

    def reset(self) -> tuple[int, int]:
        """Reset env to initial state. Returns initial observation."""
        self._agent      = START_POS
        self._steps      = 0
        self._done       = False
        self._cum_reward = 0.0
        self._history    = []
        return self._obs()

    def step(self, action: int) -> tuple[tuple, float, bool, dict]:
        """
        Take one step in the environment.

        Args:
            action: integer in {0,1,2,3}

        Returns:
            obs     : (row, col) new agent position
            reward  : float
            done    : bool
            info    : dict with extra metadata
        """
        if self._done:
            return self._obs(), 0.0, True, {"info": "episode already done"}

        assert action in (0, 1, 2, 3), f"Invalid action {action}"

        dr, dc = ACTIONS[action]
        r, c   = self._agent
        nr     = max(0, min(self.grid_size - 1, r + dr))
        nc     = max(0, min(self.grid_size - 1, c + dc))

        hit_wall = (nr == r and nc == c and (dr != 0 or dc != 0))
        self._agent = (nr, nc)
        self._steps += 1

        # ── Reward logic ──────────────────────────────────────────────────────
        if (nr, nc) == self.goal_pos:
            reward   = REWARD_GOAL
            self._done = True
            event    = "goal"
        elif (nr, nc) in self.hazard_set:
            reward   = REWARD_HAZARD
            event    = "hazard"
        elif hit_wall:
            reward   = REWARD_WALL
            event    = "wall"
        else:
            reward   = REWARD_STEP
            event    = "step"

        if self._steps >= self.max_steps:
            self._done = True
            if event == "step":
                event = "timeout"

        self._cum_reward = round(self._cum_reward + reward, 4)

        info = {
            "action_name"    : ACTION_NAMES[action],
            "event"          : event,
            "cumulative_reward": self._cum_reward,
            "steps"          : self._steps,
        }

        self._history.append({
            "step"  : self._steps,
            "action": action,
            "obs"   : self._obs(),
            "reward": reward,
            "done"  : self._done,
            "event" : event,
        })

        return self._obs(), reward, self._done, info

    def state(self) -> dict[str, Any]:
        """Return the full current state of the environment."""
        return {
            "agent_pos"        : self._agent,
            "goal_pos"         : self.goal_pos,
            "hazards"          : list(self.hazard_set),
            "grid_size"        : self.grid_size,
            "steps"            : self._steps,
            "max_steps"        : self.max_steps,
            "done"             : self._done,
            "cumulative_reward": self._cum_reward,
            "action_space"     : 4,
            "obs_space"        : (self.grid_size, self.grid_size),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _obs(self) -> tuple[int, int]:
        return self._agent

    def render(self) -> str:
        """Return ASCII render of the grid."""
        rows = []
        for r in range(self.grid_size):
            row = ""
            for c in range(self.grid_size):
                if (r, c) == self._agent:
                    row += " A "
                elif (r, c) == self.goal_pos:
                    row += " G "
                elif (r, c) in self.hazard_set:
                    row += " X "
                else:
                    row += " . "
            rows.append(row)
        return "\n".join(rows)

    def sample_action(self) -> int:
        """Sample a random valid action."""
        return random.randint(0, 3)


# ── Convenience factory (matches OpenEnv convention) ──────────────────────────

def make(env_id: str = "GridWorld-v1", **kwargs) -> GridWorldEnv:
    if env_id == "GridWorld-v1":
        return GridWorldEnv(**kwargs)
    raise ValueError(f"Unknown env_id: {env_id}")
