"""
VLM Debug HTML Visualizer

Converts the debug_full JSON + img + img_input under a single task_i directory into a
single-file HTML (images embedded as base64), saved to task_i/debug_full/debug.html
"""

import base64
import html
import json
import re
from pathlib import Path


# -- Utility functions -------------------------------------------------------

def _extract_prompt_text(prompt_data: dict) -> str:
    """Supports both the old and new prompt_data formats, extracting a readable prompt text

    Old (prompt_image_map / legacy gateway format): {"prompt": "..."}
    New (vlm_client OpenAI-compatible format): {"model", "messages": [{"role", "content"}]}
        content may be a str, or a multimodal list (text / image_url parts; image parts are skipped)
    """
    if not isinstance(prompt_data, dict):
        return ""
    if "prompt" in prompt_data:  # old format
        return prompt_data.get("prompt") or ""
    texts = []
    for m in prompt_data.get("messages") or []:
        c = m.get("content")
        if isinstance(c, str):
            texts.append(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
    return "\n".join(t for t in texts if t)


def _extract_task_instruction(prompt: str) -> str:
    """Extract the task instruction from the full step0 prompt"""
    match = re.search(r"This is your task:\s*(.+?)(?:\nUse the exact|\nAfter executing)", prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _clean_prompt(prompt: str, step_val) -> str:
    """Clean the prompt for display: keep step0 in full; strip the system prompt prefix from others"""
    if step_val != 0:
        prompt = re.sub(r"\[System prompt omitted for brevity\]\s*", "", prompt)
    prompt = re.sub(r"\n{3,}", "\n\n", prompt)
    return prompt.strip()


def _img_to_data_uri(img_path: Path) -> str | None:
    """Convert an image file into a base64 data URI"""
    if not img_path.exists():
        return None
    suffix = img_path.suffix.lower()
    mime = {
        "png": "image/png", "jpg": "image/jpeg",
        "jpeg": "image/jpeg", "gif": "image/gif", "webp": "image/webp"
    }.get(suffix.lstrip("."), "image/png")
    data = base64.b64encode(img_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _get_step_number(step_val) -> int | None:
    """Return the number for integer steps; None for grasp_N style steps"""
    if isinstance(step_val, int):
        return step_val
    if isinstance(step_val, str) and re.match(r"^\d+$", step_val):
        return int(step_val)
    return None


# -- HTML template -----------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{
    --bg: #f6f8fa;
    --card: #ffffff;
    --border: #d0d7de;
    --accent: #0969da;
    --action-bg: #ddf4ff;
    --action-border: #54aeff;
    --grasp-bg: #fff8c5;
    --grasp-border: #d4a72c;
    --code-bg: #f6f8fa;
    --text: #1f2328;
    --muted: #656d76;
    --instruction-bg: #dafbe1;
    --instruction-border: #2da44e;
    --radius: 8px;
    --shadow: 0 1px 3px rgba(0,0,0,.08);
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 32px 16px;
  }}
  .container {{ max-width: 900px; margin: 0 auto; }}

  .header-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 28px 32px;
    margin-bottom: 24px;
  }}
  .header-card h1 {{
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 12px;
  }}
  .instruction-box {{
    background: var(--instruction-bg);
    border-left: 4px solid var(--instruction-border);
    border-radius: 4px;
    padding: 12px 16px;
    font-size: 0.95rem;
  }}
  .instruction-box .label {{
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .05em;
    color: var(--instruction-border);
    margin-bottom: 4px;
  }}

  .step-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    margin-bottom: 20px;
    overflow: hidden;
  }}
  .step-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 14px 20px;
    border-bottom: 1px solid var(--border);
    background: var(--bg);
  }}
  .step-card.action .step-header {{ background: var(--action-bg); border-bottom-color: var(--action-border); }}
  .step-card.grasp  .step-header {{ background: var(--grasp-bg);  border-bottom-color: var(--grasp-border);  }}
  .step-num {{
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
  }}
  .badge {{
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: .04em;
  }}
  .badge-action {{ background: var(--action-border); color: #fff; }}
  .badge-grasp  {{ background: var(--grasp-border);  color: #fff; }}

  .step-body {{ padding: 20px; display: flex; flex-direction: column; gap: 16px; }}

  .section-label {{
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .06em;
    color: var(--muted);
    margin-bottom: 6px;
  }}
  pre {{
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 14px;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    font-size: 0.82rem;
    line-height: 1.55;
    white-space: pre-wrap;
    word-break: break-word;
    overflow-x: auto;
  }}
  .img-wrap {{
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    display: inline-block;
    max-width: 100%;
  }}
  .img-wrap img {{
    display: block;
    max-width: 100%;
    height: auto;
  }}
  .grasp-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: flex-start;
  }}
  .grasp-row .img-wrap {{
    flex: 1 1 0;
    min-width: 0;
  }}
  .grasp-row .img-wrap img {{
    width: 100%;
    height: auto;
  }}
  .wrist-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: flex-start;
  }}
  .wrist-item {{
    flex: 1 1 0;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }}
  .wrist-item img {{
    width: 100%;
    height: auto;
    display: block;
    border: 1px solid var(--border);
    border-radius: 6px;
  }}
  .img-caption {{
    font-size: 0.72rem;
    color: var(--muted);
    font-family: "SFMono-Regular", Consolas, monospace;
    text-align: center;
    word-break: break-all;
  }}

  @media (min-width: 1200px) {{
    body {{ padding: 32px; }}
  }}
</style>
</head>
<body>
<div class="container">
{body}
</div>
</body>
</html>
"""


# -- HTML builder functions --------------------------------------------------

def _build_header(task_name: str, task_idx: str, instruction: str) -> str:
    instr_html = ""
    if instruction:
        instr_html = f"""
    <div class="instruction-box">
      <div class="label">Instruction</div>
      <div>{html.escape(instruction)}</div>
    </div>"""
    return f"""<div class="header-card">
  <h1>Task: {html.escape(task_name)} — {html.escape(task_idx)}</h1>{instr_html}
</div>"""


def _build_step_card(step_val, step_type: str, prompt_text: str, response_text: str,
                     img_data_uri: str | None,
                     grasp_img_uris: list[str] | None = None,
                     wrist_img_uris: list[tuple[str, str]] | None = None,
                     place_img_uris: list[tuple[str, str]] | None = None) -> str:
    step_label = f"Step {step_val}"
    type_class = "grasp" if step_type == "grasp" else "action"
    badge_class = f"badge-{type_class}"

    img_html = ""
    if img_data_uri:
        img_html = f"""
    <div>
      <div class="section-label">Input Image</div>
      <div class="img-wrap"><img src="{img_data_uri}" alt="{html.escape(step_label)}"></div>
    </div>"""

    wrist_html = ""
    if wrist_img_uris:
        imgs = "".join(
            f'<div class="wrist-item"><img src="{uri}" alt="{html.escape(name)}"><div class="img-caption">{html.escape(name)}</div></div>'
            for uri, name in wrist_img_uris
        )
        wrist_html = f"""
    <div>
      <div class="section-label">Wrist Camera (Pick)</div>
      <div class="wrist-row">{imgs}</div>
    </div>"""

    place_html = ""
    if place_img_uris:
        imgs = "".join(
            f'<div class="wrist-item"><img src="{uri}" alt="{html.escape(name)}"><div class="img-caption">{html.escape(name)}</div></div>'
            for uri, name in place_img_uris
        )
        place_html = f"""
    <div>
      <div class="section-label">Head Camera (Place)</div>
      <div class="wrist-row">{imgs}</div>
    </div>"""

    grasp_html = ""
    if grasp_img_uris:
        imgs = "".join(
            f'<div class="img-wrap"><img src="{uri}" alt="grasp option {i + 1}"></div>'
            for i, uri in enumerate(grasp_img_uris)
        )
        grasp_html = f"""
    <div>
      <div class="section-label">Grasp Candidates</div>
      <div class="grasp-row">{imgs}</div>
    </div>"""

    return f"""<div class="step-card {type_class}">
  <div class="step-header">
    <span class="step-num">{html.escape(step_label)}</span>
    <span class="badge {badge_class}">{html.escape(step_type)}</span>
  </div>
  <div class="step-body">
    <div>
      <div class="section-label">Prompt</div>
      <pre>{html.escape(prompt_text)}</pre>
    </div>
    <div>
      <div class="section-label">Response</div>
      <pre>{html.escape(response_text)}</pre>
    </div>{img_html}{wrist_html}{place_html}{grasp_html}
  </div>
</div>"""


# -- Main class --------------------------------------------------------------

class DebugVisualizer:
    """
    Convert a single task_i directory into an HTML visualization file.

    Directory layout convention:
        task_i/
            debug_full/   <- *.json (prompt/response data)
            img_input/    <- step{N}.png (VLM input images)
            img/          <- img_{N}_wrist_pick_pts.png, grasp_viz_*_action{N}.png, etc.
    """

    def generate(self, task_dir: str | Path, task_name: str) -> Path | None:
        """
        Generate the HTML for a single task_i, saved to task_i/debug_full/debug.html.

        Args:
            task_dir:  Path to the task_i directory (e.g. .../cook_steak/task_1)
            task_name: Task name used in the HTML title (e.g. "cook_steak")

        Returns:
            The generated HTML file path, or None if there is no JSON in debug_full.
        """
        task_dir = Path(task_dir)
        debug_full_dir = task_dir / "debug_full"
        img_input_dir = task_dir / "img_input"
        img_dir = task_dir / "img"

        json_files = list(debug_full_dir.glob("*.json"))
        if not json_files:
            return None

        with open(json_files[0], "r", encoding="utf-8") as f:
            steps_data = json.load(f)

        parts = []

        # header: extract the task instruction from step0
        step0_entry = next((s for s in steps_data if s.get("step") == 0), None)
        instruction = ""
        if step0_entry:
            instruction = _extract_task_instruction(_extract_prompt_text(step0_entry["prompt_data"]))
        parts.append(_build_header(task_name, task_dir.name, instruction))

        # Generate cards step by step
        last_action_step_num = None
        for entry in steps_data:
            step_val = entry.get("step")
            step_type = entry.get("step_type", "action")
            raw_prompt = _extract_prompt_text(entry["prompt_data"])

            # Adapt response_data: it may be a string or a dict
            response_data = entry.get("response_data", "")
            if isinstance(response_data, str):
                response_text = response_data
            elif isinstance(response_data, dict):
                completions = response_data.get("completions", [])
                response_text = completions[0].get("text", "") if completions else ""
            else:
                response_text = str(response_data)

            display_prompt = _clean_prompt(raw_prompt, step_val)
            step_num = _get_step_number(step_val)

            # action step: attach img_input/step{N}.png
            img_uri = None
            if step_num is not None and step_type != "grasp":
                img_path = img_input_dir / f"step{step_num}.png"
                img_uri = _img_to_data_uri(img_path)

            # action step: attach the pick wrist bbox image
            wrist_uris = None
            if step_num is not None and step_type != "grasp" and img_dir.exists():
                pts_path = img_dir / f"img_{step_num}_wrist_pick_pts.png"
                if pts_path.exists():
                    uri = _img_to_data_uri(pts_path)
                    wrist_uris = [(uri, pts_path.name)] if uri else None

            # action step: attach the place head bbox image
            place_uris = None
            if step_num is not None and step_type != "grasp" and img_dir.exists():
                place_path = img_dir / f"img_{step_num}_head_place.png"
                if place_path.exists():
                    uri = _img_to_data_uri(place_path)
                    place_uris = [(uri, place_path.name)] if uri else None

            # grasp step: look up grasp_viz images using the previous action step number
            grasp_uris = None
            if step_type == "grasp" and last_action_step_num is not None and img_dir.exists():
                candidates = sorted(
                    img_dir.glob(f"grasp_viz_*_action{last_action_step_num}.png"),
                    key=lambda p: int(re.search(r"grasp_viz_(\d+)", p.name).group(1))
                )
                grasp_uris = [uri for p in candidates if (uri := _img_to_data_uri(p))]

            if step_type != "grasp" and step_num is not None:
                last_action_step_num = step_num

            parts.append(_build_step_card(
                step_val, step_type, display_prompt, response_text,
                img_uri, grasp_uris, wrist_uris, place_uris
            ))

        body_html = "\n\n".join(parts)
        final_html = _HTML_TEMPLATE.format(
            title=html.escape(f"{task_name} — {task_dir.name}"),
            body=body_html
        )

        output_path = debug_full_dir / "debug.html"
        output_path.write_text(final_html, encoding="utf-8")
        return output_path


# -- Command-line entry ------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Paste a task_i path here directly, or pass it via a command-line argument
    task_dir = "/Users/lh/work/vlabench_etc/VLABench/VLABench/vlm_evaluation/results/seed2d0/episode_0729_016/cook_steak_to_stove/task_1"

    # Command-line arguments take priority
    if len(sys.argv) > 1:
        task_dir = sys.argv[1]

    if not task_dir:
        print("Usage: python debug_visualizer.py <task_i_dir>")
        print("  或直接在脚本底部 task_dir 变量中赋值")
        sys.exit(1)

    p = Path(task_dir)
    if not p.is_dir():
        print(f"[Error] 目录不存在: {p}")
        sys.exit(1)

    # Automatically use the parent directory name as task_name (e.g. .../select_steak/task_1 -> "select_steak")
    task_name = p.parent.name

    viz = DebugVisualizer()
    result = viz.generate(task_dir=p, task_name=task_name)

    if result:
        print(f"✓ HTML 已生成: {result}")
    else:
        print(f"[Warning] 未在 {p / 'debug_full'} 中找到 JSON 文件，跳过生成。")
