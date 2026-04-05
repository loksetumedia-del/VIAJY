# GridWorld-v1 — OpenEnv Competition Submission

## Environment Description
A 5×5 grid navigation task where an AI agent must reach the goal `G` at `(4,4)` starting from `(0,0)`, while avoiding hazard tiles `X`.

## API
```python
from env import make

env = make("GridWorld-v1")
obs = env.reset()                          # → (0, 0)

while not done:
    action = policy.select(obs)            # 0=Up 1=Right 2=Down 3=Left
    obs, reward, done, info = env.step(action)

state = env.state()                        # full state dict anytime
```

## Reward Structure
| Event | Reward |
|-------|--------|
| Reach goal | +10.0 |
| Hit hazard | -1.0 |
| Wall bump | -0.2 |
| Normal step | -0.1 |

## Files
- `env.py` — environment (step / reset / state)
- `inference.py` — LLM agent with START/STEP/END logging
- `app.py` — Hugging Face Space (Gradio UI)
- `requirements.txt` — dependencies

## Running Locally
```bash
pip install -r requirements.txt

# Greedy agent (no LLM key needed)
python inference.py

# LLM agent
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_token_here
python inference.py --llm

# Launch HF Space locally
python app.py
```

## Environment Variables
| Variable | Required | Default |
|----------|----------|---------|
| `API_BASE_URL` | No | `https://api.openai.com/v1` |
| `MODEL_NAME` | No | `gpt-4o-mini` |
| `HF_TOKEN` | Yes (LLM mode) | — |
| `LOCAL_IMAGE_NAME` | No | — |

## Stdout Log Format (START/STEP/END)
```json
{"type": "START", "env_id": "GridWorld-v1", "config": {...}}
{"type": "STEP",  "step": 1, "action": 1, "obs": [0,1], "reward": -0.1, "done": false}
{"type": "END",   "total_reward": 9.5, "total_steps": 8, "success": true}
```
