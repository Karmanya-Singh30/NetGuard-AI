"""Shared UI helpers for the NetGuard AI Streamlit app.

The app only *reads* artifacts produced by `python -m src.train`; it never
trains a model and never shows a number that is not in those files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

# Two semantic signals only: ok = benign/passed, attack = flagged. Warn/info are for notes.
C = {"bg": "#0A1520", "panel": "#101E2C", "raise": "#152738", "line": "#1E3246",
     "text": "#DDE7F0", "muted": "#8B9DB1", "ok": "#3DDBA8", "attack": "#FF6F61",
     "warn": "#F2B544", "info": "#6CC6F5"}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Archivo, system-ui, sans-serif", color=C["text"], size=13),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor=C["line"], zerolinecolor=C["line"], linecolor=C["line"]),
    yaxis=dict(gridcolor=C["line"], zerolinecolor=C["line"], linecolor=C["line"]),
    colorway=[C["ok"], C["info"], "#A99BF5", C["warn"], C["attack"], "#8B9DB1"],
    hoverlabel=dict(bgcolor=C["raise"], bordercolor=C["line"],
                    font=dict(family="Archivo, sans-serif", color=C["text"], size=13)),
    hovermode="closest",
)


def inject_css():
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@100..125,400..800&display=swap');
:root {{
  --ng-bg: {C['bg']}; --ng-panel: {C['panel']}; --ng-raise: {C['raise']}; --ng-line: {C['line']};
  --ng-text: {C['text']}; --ng-muted: {C['muted']}; --ng-ok: {C['ok']}; --ng-attack: {C['attack']};
  --ng-warn: {C['warn']}; --ng-info: {C['info']};
  --ng-ease: cubic-bezier(.2,.7,.2,1);
}}
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {{
  font-family: 'Archivo', system-ui, sans-serif;
}}
.block-container {{ max-width: 1180px; padding-top: 2.4rem; }}
h1 {{ font-stretch: 115%; font-weight: 760; letter-spacing: -0.025em; font-size: 2.35rem;
      line-height: 1.08; }}
h2 {{ font-stretch: 110%; font-weight: 640; font-size: 1.28rem; letter-spacing: -0.01em;
      margin-top: 2rem; padding-top: 1.1rem; border-top: 1px solid var(--ng-line); }}
[data-testid="stMarkdownContainer"] h2 {{ font-size: 1.6rem; }}
h3 {{ font-stretch: 105%; font-weight: 620; font-size: 1.05rem; color: var(--ng-text); }}
p, li {{ line-height: 1.6; max-width: 78ch; }}
code {{ font-size: .86em; }}

/* Sidebar: active page gets a signal bar that slides in when the page changes */
[data-testid="stSidebar"] {{ border-right: 1px solid var(--ng-line); }}
[data-testid="stSidebarNavLink"] {{ position: relative; border-radius: 6px;
  transition: background-color .2s var(--ng-ease), color .2s var(--ng-ease); }}
[data-testid="stSidebarNavLink"]::before {{ content: ""; position: absolute; left: -.55rem;
  top: 22%; bottom: 22%; width: 3px; border-radius: 3px; background: var(--ng-ok);
  transform: scaleY(0); transition: transform .32s var(--ng-ease); }}
[data-testid="stSidebarNavLink"][aria-current="page"]::before {{ transform: scaleY(1); }}
[data-testid="stSidebarNavLink"]:hover {{ background: var(--ng-raise); }}
.ng-status {{ font-size: .85rem; color: var(--ng-muted); line-height: 1.5; }}
.ng-status b {{ color: var(--ng-text); font-weight: 600; display: block; margin-bottom: .35rem; }}
.ng-status div {{ display: flex; align-items: center; gap: .5rem; }}
.ng-dot {{ width: 7px; height: 7px; border-radius: 50%; flex: none; background: var(--ng-muted); }}
.ng-dot.on {{ background: var(--ng-ok); }}
.ng-dot.off {{ background: transparent; border: 1px solid var(--ng-warn); }}

/* Entrance: one staggered reveal when a page mounts; later reruns reuse the nodes */
@keyframes ng-rise {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: none; }} }}
[data-testid="stMainBlockContainer"] > div > [data-testid="stVerticalBlock"] > div {{
  animation: ng-rise .45s var(--ng-ease) both; }}
{"".join(f'[data-testid="stMainBlockContainer"] > div > [data-testid="stVerticalBlock"] > div:nth-child({i}) {{ animation-delay: {i * 40}ms; }}' for i in range(1, 11))}

/* Controls */
[data-testid^="stBaseButton"] {{ transition: transform .12s var(--ng-ease),
  background-color .2s var(--ng-ease), border-color .2s var(--ng-ease), box-shadow .2s var(--ng-ease); }}
[data-testid^="stBaseButton"]:hover {{ border-color: var(--ng-ok); }}
[data-testid^="stBaseButton"]:active {{ transform: translateY(1px) scale(.985); }}
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {{
  color: {C['bg']} !important; font-weight: 600; }}
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover {{
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--ng-ok) 25%, transparent); }}
[data-testid="stExpander"] details {{ border-color: var(--ng-line); transition: border-color .2s; }}
[data-testid="stExpander"] details:hover {{ border-color: color-mix(in srgb, var(--ng-ok) 45%, var(--ng-line)); }}
[data-baseweb="tab"] {{ transition: color .2s var(--ng-ease); }}
[data-testid="stFileUploaderDropzone"] {{ border: 1px dashed var(--ng-line); transition: border-color .2s; }}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color: var(--ng-ok); }}
:focus-visible {{ outline: 2px solid var(--ng-ok) !important; outline-offset: 2px; }}

/* Data */
[data-testid="stMetric"] {{ border-left: 2px solid var(--ng-line); padding-left: .9rem; }}
[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; font-weight: 640;
  font-stretch: 108%; }}
[data-testid="stMetricLabel"] p {{ color: var(--ng-muted); }}
[data-testid="stDataFrame"] {{ font-variant-numeric: tabular-nums; }}

.ng-lede {{ color: var(--ng-muted); font-size: 1.08rem; max-width: 66ch; margin-top: -0.3rem; }}
.ng-note, .ng-warn {{ border-left: 3px solid var(--ng-info); background: var(--ng-panel);
  padding: .7rem 1rem; border-radius: 0 6px 6px 0; color: var(--ng-text); margin: .6rem 0 1rem; }}
.ng-warn {{ border-left-color: var(--ng-warn); }}
.ng-attack {{ color: var(--ng-attack); }}
.ng-ok {{ color: var(--ng-ok); }}
.ng-kv {{ border-collapse: collapse; }}
.ng-kv td {{ padding: .4rem 1.2rem .4rem 0; vertical-align: top; border: 0;
  border-bottom: 1px solid var(--ng-line); }}
.ng-kv td:first-child {{ color: var(--ng-muted); white-space: nowrap; }}
.ng-kv a {{ color: var(--ng-info); }}

/* Triage ledger: the test set split into the four outcomes */
.ng-ledger {{ margin: .4rem 0 1.4rem; }}
.ng-ledger .head {{ font-size: 1.02rem; color: var(--ng-muted); margin-bottom: .7rem; }}
.ng-ledger .head b {{ color: var(--ng-text); font-stretch: 112%; font-size: 1.5rem;
  font-variant-numeric: tabular-nums; margin-right: .35rem; }}
.ng-bar {{ display: flex; height: 18px; gap: 2px; border-radius: 4px; overflow: hidden; }}
.ng-bar span {{ min-width: 2px; transform-origin: left; animation: ng-grow .9s var(--ng-ease) both; }}
.ng-bar span:nth-child(2) {{ animation-delay: .12s; }}
.ng-bar span:nth-child(3) {{ animation-delay: .24s; }}
.ng-bar span:nth-child(4) {{ animation-delay: .36s; }}
@keyframes ng-grow {{ from {{ transform: scaleX(0); }} to {{ transform: none; }} }}
.ng-cells {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1rem; }}
.ng-cell {{ border-top: 3px solid var(--sw); padding-top: .55rem; }}
.ng-cell .v {{ font-size: 1.55rem; font-weight: 680; font-stretch: 110%;
  font-variant-numeric: tabular-nums; line-height: 1.15; }}
.ng-cell .k {{ color: var(--ng-text); font-weight: 560; }}
.ng-cell .d {{ color: var(--ng-muted); font-size: .86rem; }}

/* Verdict for a classification run */
.ng-verdict {{ border: 1px solid var(--ng-line); border-radius: 10px; padding: 1.3rem 1.5rem 1.1rem;
  background: var(--ng-panel); margin: .8rem 0 1rem; animation: ng-rise .4s var(--ng-ease) both; }}
.ng-verdict .big {{ font-size: 2.3rem; font-weight: 720; font-stretch: 112%; letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums; line-height: 1.1; }}
.ng-verdict .sub {{ color: var(--ng-muted); margin-top: .35rem; }}
.ng-verdict .ng-bar {{ height: 8px; margin-top: 1rem; }}

/* Empty state */
.ng-empty {{ border: 1px dashed var(--ng-line); border-radius: 10px; padding: 1.4rem 1.5rem;
  margin: 1rem 0; }}
.ng-empty b {{ display: block; font-size: 1.05rem; font-stretch: 108%; margin-bottom: .3rem; }}
.ng-empty p {{ color: var(--ng-muted); margin: 0; }}

@media (max-width: 640px) {{
  .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
  h1 {{ font-size: 1.85rem; }}
  [data-testid="stMarkdownContainer"] h2 {{ font-size: 1.3rem; }}
  .ng-cells {{ grid-template-columns: repeat(2, 1fr); }}
  .ng-verdict .big {{ font-size: 1.7rem; }}
}}
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation-duration: .001ms !important; animation-delay: 0s !important;
    transition-duration: .001ms !important; scroll-behavior: auto !important; }}
}}
</style>""", unsafe_allow_html=True)


def note(text: str, kind: str = "note"):
    st.markdown(f'<div class="ng-{kind}">{text}</div>', unsafe_allow_html=True)


def kv_table(rows: list[tuple[str, str]]):
    body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    st.markdown(f'<table class="ng-kv">{body}</table>', unsafe_allow_html=True)


def ledger(m: dict):
    """Proportional bar of the four test-set outcomes, from the saved confusion counts."""
    n = m["tp"] + m["fn"] + m["fp"] + m["tn"]
    parts = [("tp", "Attacks caught", "true positives", C["ok"]),
             ("fn", "Attacks missed", "false negatives", C["attack"]),
             ("fp", "False alarms", "benign flagged", C["warn"]),
             ("tn", "Benign passed", "true negatives", "#3A5874")]
    bar = "".join(f'<span style="flex:{m[k]} 0 0;background:{c}" title="{lab}: {m[k]:,} '
                  f'({m[k] / n:.3%})"></span>' for k, lab, _, c in parts)
    cells = "".join(f'<div class="ng-cell" style="--sw:{c}"><div class="v">{m[k]:,}</div>'
                    f'<div class="k">{lab}</div><div class="d">{m[k] / n:.3%} · {d}</div></div>'
                    for k, lab, d, c in parts)
    label = ", ".join(f"{lab} {m[k]:,}" for k, lab, _, _ in parts)
    st.markdown(f'<div class="ng-ledger"><div class="head"><b>{n:,}</b>held-out test flows, '
                f'as the model sorted them</div><div class="ng-bar" role="img" aria-label="{label}">'
                f'{bar}</div><div class="ng-cells">{cells}</div></div>', unsafe_allow_html=True)


def empty_state(title: str, body: str, command: str | None = None):
    st.markdown(f'<div class="ng-empty"><b>{title}</b><p>{body}</p></div>', unsafe_allow_html=True)
    if command:
        st.code(command, language="bash")


def sidebar_status():
    """What is actually on disk. NetGuard never monitors traffic, so nothing here is 'live'."""
    rows = [("Detection model", config.FINAL_PIPELINE_PATH.exists()),
            ("Attack-family model", config.MULTICLASS_PIPELINE_PATH.exists()),
            ("Evaluation results", (config.RESULTS_DIR / "model_comparison.json").exists()),
            ("Demo sample", config.DEMO_SAMPLE_PATH.exists())]
    body = "".join(f'<div><span class="ng-dot {"on" if ok else "off"}"></span>{name}'
                   f'{"" if ok else " (missing)"}</div>' for name, ok in rows)
    st.sidebar.markdown(f'<div class="ng-status"><b>Artifacts on disk</b>{body}</div>',
                        unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Artifact access (cached; invalidated when files change)
# ----------------------------------------------------------------------------
def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime: float):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_json(path: Path):
    """Return parsed JSON, or None if the file does not exist yet."""
    if not path.exists():
        return None
    try:
        return _read_json(str(path), _mtime(path))
    except (json.JSONDecodeError, OSError):
        return None


def results(name: str):
    return read_json(config.RESULTS_DIR / name)


def metadata():
    return read_json(config.METADATA_PATH)


def schema():
    return read_json(config.SCHEMA_PATH)


@st.cache_resource(show_spinner="Loading the trained model…")
def _load_model(path: str, mtime: float):
    import joblib
    return joblib.load(path)   # only ever our own training output, never an upload


def model(kind: str = "binary"):
    p = config.FINAL_PIPELINE_PATH if kind == "binary" else config.MULTICLASS_PIPELINE_PATH
    if not p.exists():
        return None
    return _load_model(str(p), _mtime(p))


def figure(name: str, caption: str):
    p = config.FIGURES_DIR / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.caption(f"Figure `{name}` has not been generated yet.")


def require_artifacts() -> bool:
    """Show setup instructions and return False if training has not run."""
    if metadata() is None or schema() is None or not config.FINAL_PIPELINE_PATH.exists():
        empty_state("No trained model yet",
                    "This page reads the results written by the training script, and they are not "
                    "in <code>models/</code> and <code>reports/</code> yet. Run this from the project "
                    "folder, then reload. The README explains how to get the dataset.",
                    "python -m src.train")
        return False
    return True


def require_results(name: str, what: str):
    """Return the results JSON, or show why the section is empty and return None."""
    data = results(name)
    if data is None:
        empty_state(f"{what} not available",
                    f"<code>reports/results/{name}</code> is missing. The model files are present, "
                    "but the evaluation outputs were not copied or were deleted. Re-run training "
                    "to regenerate them.", "python -m src.train")
    return data


def reportability_banner():
    meta = metadata()
    if meta and not meta.get("reportable", False):
        note("<b>Smoke-test artifacts.</b> These results come from quick mode or a small/synthetic "
             "dataset and are only proof that the code runs. Do not quote them. Run "
             "<code>python -m src.train</code> on the full CICIDS2017 data.", "warn")


def pct(v, nd=2):
    return "n/a" if v is None else f"{100 * v:.{nd}f}%"


def num(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"
