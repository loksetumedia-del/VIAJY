---
title: My OpenEnv
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# GridWorld-v1 — OpenEnv

A 5×5 grid navigation environment implementing the OpenEnv HTTP API.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/reset` | Reset environment |
| `POST` | `/step` | Take action `{"action": 0-3}` |
| `GET` | `/state` | Full environment state |

## Actions
- `0` = Up
- `1` = Right  
- `2` = Down
- `3` = Left

## Example

```bash
# Reset
curl -X POST https://your-space.hf.space/reset

# Step
curl -X POST https://your-space.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": 1}'

# State
curl https://your-space.hf.space/state
```

## Environment Variables

| Variable | Required | Default |
|----------|----------|---------|
| `API_BASE_URL` | No | `https://api.openai.com/v1` |
| `MODEL_NAME` | No | `gpt-4o-mini` |
| `HF_TOKEN` | Yes (LLM) | — |
| `LOCAL_IMAGE_NAME` | No | — |

## Log Format (START/STEP/END)

```json
{"type": "START", "env_id": "GridWorld-v1", "config": {...}}
{"type": "STEP", "step": 1, "action": 1, "obs": [0,1], "reward": -0.1, "done": false}
{"type": "END", "total_reward": 9.3, "total_steps": 8, "success": true}
```
