"""
app.py  —  Hugging Face Space for GridWorld-v1 OpenEnv
======================================================
This Space hosts the environment and exposes a Gradio UI
so evaluators can run the agent interactively.
"""

import gradio as gr
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from env import make
from inference import run_episode, _greedy_action, llm_select_action

# ── Helper: render grid as HTML ───────────────────────────────────────────────

def render_html(env_obj, last_reward=None, last_event=None):
    s    = env_obj.state()
    ag   = s["agent_pos"]
    goal = s["goal_pos"]
    haz  = set(map(tuple, s["hazards"]))
    n    = s["grid_size"]

    cells = ""
    for r in range(n):
        cells += "<tr>"
        for c in range(n):
            pos = (r, c)
            if pos == ag:
                bg, txt = "#185FA5", "A"
            elif pos == goal:
                bg, txt = "#3B6D11", "G"
            elif pos in haz:
                bg, txt = "#A32D2D", "X"
            else:
                bg, txt = "#2a2a2a", "·"
            cells += f'<td style="width:60px;height:60px;text-align:center;font-size:22px;font-weight:bold;color:white;background:{bg};border-radius:6px;border:2px solid #111">{txt}</td>'
        cells += "</tr>"

    reward_info = ""
    if last_reward is not None:
        color = "lightgreen" if last_reward > 0 else ("tomato" if last_reward < -0.5 else "orange")
        reward_info = f'<p style="color:{color};font-family:monospace">Last reward: {last_reward}  |  Event: {last_event}</p>'

    return f"""
    <div style="font-family:monospace;background:#111;padding:16px;border-radius:8px;display:inline-block">
      <p style="color:#aaa;margin:0 0 8px">GridWorld-v1 — Step {s['steps']}/{s['max_steps']} | Cum. reward: {s['cumulative_reward']}</p>
      <table style="border-collapse:separate;border-spacing:4px">{cells}</table>
      {reward_info}
    </div>
    """


# ── Gradio state ──────────────────────────────────────────────────────────────
_env = [make()]

def reset_env():
    _env[0] = make()
    _env[0].reset()
    return render_html(_env[0]), "Episode reset! Use buttons or auto-run.", ""

def step_action(action_idx):
    env = _env[0]
    if env.state()["done"]:
        return render_html(env), "Episode done — press Reset!", ""
    obs, reward, done, info = env.step(int(action_idx))
    msg = f"Action: {info['action_name']} | Event: {info['event']}"
    log = json.dumps({"type":"STEP","step":info["steps"],"obs":list(obs),"reward":reward,"done":done}, indent=2)
    return render_html(env, reward, info["event"]), msg, log

def auto_run():
    env  = _env[0]
    if env.state()["done"]:
        env.reset()
    logs = []
    while not env.state()["done"]:
        s  = env.state()
        a  = _greedy_action(s["agent_pos"], s["goal_pos"], s["hazards"])
        obs, reward, done, info = env.step(a)
        logs.append(json.dumps({"step":info["steps"],"action":info["action_name"],"reward":reward,"done":done}))
    s   = env.state()
    msg = "✅ Goal reached!" if s["agent_pos"] == s["goal_pos"] else "❌ Timeout"
    return render_html(env), msg, "\n".join(logs)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="GridWorld-v1 OpenEnv", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🤖 GridWorld-v1 — OpenEnv\n**A=Agent · G=Goal · X=Hazard**  |  Actions: ↑↓←→")

    with gr.Row():
        with gr.Column(scale=2):
            grid_html = gr.HTML(label="Environment")
        with gr.Column(scale=1):
            status = gr.Textbox(label="Status", interactive=False)
            log_out = gr.Textbox(label="Step log (JSON)", lines=8, interactive=False)

    with gr.Row():
        btn_up    = gr.Button("↑ Up")
        btn_down  = gr.Button("↓ Down")
        btn_left  = gr.Button("← Left")
        btn_right = gr.Button("→ Right")

    with gr.Row():
        btn_reset = gr.Button("🔄 Reset", variant="secondary")
        btn_auto  = gr.Button("▶ Auto Run (Greedy)", variant="primary")

    btn_up.click   (lambda: step_action(0), outputs=[grid_html, status, log_out])
    btn_right.click(lambda: step_action(1), outputs=[grid_html, status, log_out])
    btn_down.click (lambda: step_action(2), outputs=[grid_html, status, log_out])
    btn_left.click (lambda: step_action(3), outputs=[grid_html, status, log_out])
    btn_reset.click(reset_env,              outputs=[grid_html, status, log_out])
    btn_auto.click (auto_run,               outputs=[grid_html, status, log_out])

    demo.load(reset_env, outputs=[grid_html, status, log_out])

if __name__ == "__main__":
    demo.launch()
