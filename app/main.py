"""StarLens — Gemma 4 Night Sky Companion (Gradio UI)

Gemma 4 is the PROTAGONIST — every feature flows through Gemma intelligence.
Beautiful dark astronomy theme with the Gemma 4 brand front and center.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import tempfile

from datetime import datetime, timedelta, timezone
from pathlib import Path
from starlens.engine import StarLensEngine
from starlens.settings import settings

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr  # pylint: disable=wrong-import-position
from geopy.geocoders import Nominatim  # type: ignore[import-not-found,import-untyped]  # pylint: disable=wrong-import-position
from PIL import Image  # pylint: disable=wrong-import-position

import httpx  # pylint: disable=wrong-import-position
import ollama  # pylint: disable=wrong-import-position

logging.basicConfig(
    level=getattr(logging, settings.app.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Engine singleton ───────────────────────────────────────
_engine: StarLensEngine | None = None
_engine_host: str | None = None
_engine_model: str | None = None


def get_engine(
    ollama_host: str | None = None, model: str | None = None
) -> StarLensEngine:
    global _engine, _engine_host, _engine_model  # pylint: disable=global-statement

    host = ollama_host or settings.ollama.host
    data_dir = Path(__file__).parent / "data"
    if _engine is None or _engine_host != host or _engine_model != model:
        logger.info("Initializing engine: host=%s, model=%s", host, model)
        _engine = StarLensEngine(
            data_dir=data_dir, ollama_host=host, model=model
        )
        _engine_host = host
        _engine_model = model
    return _engine


def geocode(city: str) -> tuple[float, float]:
    try:
        logger.debug("Geocoding city: %s", city)
        loc = Nominatim(user_agent="starlens").geocode(city)
        if loc:
            logger.debug("Geocoded %s → (%.4f, %.4f)", city, loc.latitude, loc.longitude)
            return round(loc.latitude, 4), round(loc.longitude, 4)
    except Exception:  # pylint: disable=broad-exception-caught
        logger.warning("Geocoding failed for '%s', using fallback", city)
    return settings.app.fallback_lat, settings.app.fallback_lon


def bytes_to_pil(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b))


# ─── Gemma 4 brand logo ────────────────────────────────────
_LOGO_PATH = Path(__file__).parent / "assets" / "gemma_logo.png"
_LOGO_B64 = (
    base64.b64encode(_LOGO_PATH.read_bytes()).decode() if _LOGO_PATH.exists() else ""
)

# ─── Custom CSS ─────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global ─────────────────────────────────────────── */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    max-width: 1400px !important;
}

/* ── Header (light) ─────────────────────────────────── */
.starlens-header {
    text-align: center;
    padding: 20px 16px 16px;
    background: linear-gradient(135deg, #e8eaf6 0%, #ede7f6 50%, #e8eaf6 100%);
    border-radius: 12px;
    border: 1px solid rgba(100, 120, 200, 0.15);
    margin-bottom: 12px;
}
.starlens-header img {
    height: 56px;
    margin-bottom: 4px;
}
.starlens-header h1 {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 1.6em;
    background: linear-gradient(135deg, #3b6ce7, #7c3aed, #db2777);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 4px 0 2px;
    letter-spacing: -0.02em;
}
.starlens-header .subtitle {
    color: #5c6784;
    font-size: 0.88em;
    font-weight: 400;
    letter-spacing: 0.02em;
}
.gemma-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, rgba(66,133,244,0.1), rgba(155,114,203,0.1));
    border: 1px solid rgba(155,114,203,0.25);
    border-radius: 20px;
    padding: 4px 14px;
    margin-top: 8px;
    font-size: 0.75em;
    color: #7c3aed;
    font-weight: 500;
}

/* ── Header (dark) ──────────────────────────────────── */
.dark .starlens-header {
    background: linear-gradient(135deg, #0d1225 0%, #1a1040 50%, #0d1225 100%);
    border-color: rgba(100, 120, 200, 0.2);
}
.dark .starlens-header h1 {
    background: linear-gradient(135deg, #8ab4f8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.dark .starlens-header .subtitle { color: #8899bb; }
.dark .starlens-header img { filter: brightness(1.1); }
.dark .gemma-badge {
    background: linear-gradient(135deg, rgba(66,133,244,0.15), rgba(155,114,203,0.15));
    border-color: rgba(155,114,203,0.3);
    color: #c084fc;
}

/* ── Tab styling ────────────────────────────────────── */
.tab-nav button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.82em !important;
}

/* ── Buttons ──────────────────────────────────────────── */
button.primary, button.secondary {
    font-size: 0.85em !important;
    padding: 8px 16px !important;
}
button.lg {
    font-size: 0.9em !important;
    padding: 10px 20px !important;
}

/* ── Markdown headings inside tabs ───────────────────── */
.prose h2 { font-size: 1.15em !important; }
.prose h3 { font-size: 1.0em !important; }
.tab-nav button.selected {
    border-bottom: 2px solid #4285F4 !important;
}
.dark .tab-nav button.selected {
    background: linear-gradient(135deg, #1a2040, #201a40) !important;
    border-bottom: 2px solid #8ab4f8 !important;
}

/* ── Row alignment ───────────────────────────────────── */
.row-aligned {
    align-items: flex-end !important;
}

/* ── Footer ─────────────────────────────────────────── */
.starlens-footer {
    text-align: center;
    color: #778;
    font-size: 0.8em;
    padding: 16px;
    margin-top: 8px;
}
.starlens-footer a { color: #4285F4; }
.dark .starlens-footer { color: #556; }
.dark .starlens-footer a { color: #8ab4f8; }
"""

HEADER_HTML = f"""
<div class="starlens-header">
    <img src="data:image/png;base64,{_LOGO_B64}" alt="Gemma 4">
    <h1>StarLens</h1>
    <div class="subtitle">Your AI Night Sky Companion</div>
    <div class="gemma-badge">
        <span>⚡</span> Powered by <strong>Gemma 4</strong> — Google's Multimodal AI
    </div>
</div>
"""


# ─── Ollama error handling ──────────────────────────────────
_OLLAMA_ERRORS = (
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    httpx.ConnectError,
    httpx.TimeoutException,
    ollama.ResponseError,
    ConnectionError,
    TimeoutError,
)

_ERR_TIMEOUT = (
    "⏳ **Gemma 4 is taking too long to respond.** "
    "The model may be loading or the request is too complex. "
    "Please try again in a moment."
)

_ERR_CONNECTION = (
    "🔌 **Cannot connect to Ollama.** "
    "Make sure Ollama is running and the host address is correct.\n\n"
    "Start Ollama with: `ollama serve`"
)

_ERR_UNAUTHORIZED = (
    "🔑 **Ollama returned 401 Unauthorized.** "
    "The model may require an API key. Set `OLLAMA_API_KEY` "
    "in your environment or `.env` file."
)

_ERR_MODEL = (
    "⚠️ **Model error from Ollama.** "
    "The model may not be downloaded yet. "
    "Pull it with: `ollama pull {model}`"
)


def _ollama_error_msg(exc: Exception) -> str:
    """Return a user-friendly message for Ollama/network errors."""
    logger.error("Ollama error: %s: %s", type(exc).__name__, exc)
    if isinstance(exc, (httpx.ReadTimeout, httpx.ConnectTimeout, TimeoutError)):
        return _ERR_TIMEOUT
    if isinstance(exc, ollama.ResponseError):
        if exc.status_code == 401:  # type: ignore[union-attr]
            return _ERR_UNAUTHORIZED
        return _ERR_MODEL
    return _ERR_CONNECTION


# ═══════════════════════════════════════════════════════════
#  Tab 1 — Tonight's Sky
# ═══════════════════════════════════════════════════════════
def fn_tonights_sky(city: str, ollama_host: str, model: str):
    logger.info("Tonight's sky requested for %s", city)
    try:
        lat, lon = geocode(city)
        engine = get_engine(ollama_host, model)

        sky = engine.catalog.whats_up(lat, lon)
        chart_bytes = engine.render_chart(lat, lon)
        chart_img = bytes_to_pil(chart_bytes)

        # Build info text
        sun = sky.get("sun", {})
        moon = sky.get("moon", {})
        lines = [f"### Sky Conditions — {city}"]
        lines.append(
            f"**Sun:** {sun.get('condition', '?').replace('_', ' ').title()} "
            f"({sun.get('altitude', '?')}°)"
        )
        if moon.get("visible"):
            lines.append(
                f"**Moon:** {moon['phase_name']} — "
                f"{moon['phase_percent']}% illuminated, "
                f"{moon['direction']} at {moon['altitude']}°"
            )

        planets = sky.get("planets", [])
        if planets:
            planet_lines = "\n".join(
                f"- 🪐 **{p['name']}** — {p['direction']} at {p['altitude']}°"
                for p in planets
            )
            lines.append(f"\n**Planets ({len(planets)}):**\n{planet_lines}")

        stars = sky.get("bright_stars", [])
        if stars:
            star_lines = "\n".join(
                f"- ⭐ **{s['name']}** — mag {s['magnitude']}, "
                f"{s['direction']} {s['altitude']}°"
                for s in stars[:12]
            )
            lines.append(f"\n**Bright Stars ({len(stars)}):**\n{star_lines}")

        constellations = sky.get("constellations", [])
        if constellations:
            const_lines = "\n".join(
                f"- ✨ **{c['name']}** — {c['completeness']}% visible"
                for c in constellations[:12]
            )
            lines.append(
                f"\n**Constellations ({len(constellations)}):**\n{const_lines}"
            )

        info_md = "\n".join(lines)

        # Gemma narration
        sky_summary = json.dumps(sky, indent=2, default=str)
        narration = engine.gemma.narrate_sky(sky_summary)

        return chart_img, info_md, narration
    except _OLLAMA_ERRORS as exc:
        msg = _ollama_error_msg(exc)
        return None, msg, msg


def fn_why_object(object_name: str, city: str, ollama_host: str, model: str):
    if not object_name.strip():
        yield "Please enter an object name (e.g., Jupiter, Sirius, Moon)."
        return
    logger.info("Why object requested: %s from %s", object_name, city)
    try:
        lat, lon = geocode(city)
        engine = get_engine(ollama_host, model)
        accumulated = ""
        for chunk in engine.explain_why_stream(object_name.strip(), lat, lon):
            accumulated += chunk or ""
            yield accumulated
    except _OLLAMA_ERRORS as exc:
        yield _ollama_error_msg(exc)


def fn_analyze_chart(city: str, ollama_host: str, model: str):
    logger.info("Chart analysis requested for %s", city)
    try:
        lat, lon = geocode(city)
        engine = get_engine(ollama_host, model)
        accumulated = ""
        for chunk in engine.analyze_sky_chart_stream(lat, lon):
            accumulated += chunk or ""
            yield accumulated
    except _OLLAMA_ERRORS as exc:
        yield _ollama_error_msg(exc)


# ═══════════════════════════════════════════════════════════
#  Tab 2 — Sky Chat
# ═══════════════════════════════════════════════════════════
def fn_chat(
    user_msg: str,
    history: list,
    city: str,
    ollama_host: str,
    model: str,
):
    if not user_msg.strip():
        return history, ""

    logger.info("Chat message: %s", user_msg[:80])
    try:
        lat, lon = geocode(city)
        engine = get_engine(ollama_host, model)

        # Convert Gradio chat history to engine format
        engine_history = []
        for msg in history:
            content = msg["content"]
            # Gradio 6.x may send content as list of dicts
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            engine_history.append({"role": msg["role"], "content": content})

        response = engine.chat(user_msg, lat, lon, history=engine_history)
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": response})
    except _OLLAMA_ERRORS as exc:
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": _ollama_error_msg(exc)})
    return history, ""


# ═══════════════════════════════════════════════════════════
#  Tab 3 — Guided Tour
# ═══════════════════════════════════════════════════════════
def fn_tour_start(total_steps: int, city: str, ollama_host: str, model: str):
    logger.info("Guided tour started: %d steps from %s", total_steps, city)
    try:
        lat, lon = geocode(city)
        engine = get_engine(ollama_host, model)
        header = f"### 🚀 Stop 1 of {total_steps}\n\n"
        accumulated = header
        for chunk in engine.guided_tour_step_stream(
            lat, lon, step=0, total_steps=total_steps
        ):
            accumulated += chunk or ""
            yield accumulated, 1
    except _OLLAMA_ERRORS as exc:
        yield _ollama_error_msg(exc), 0


def fn_tour_next(
    current_step: int,
    total_steps: int,
    previous_md: str,
    city: str,
    ollama_host: str,
    model: str,
):
    if current_step >= total_steps:
        yield (
            previous_md + "\n\n---\n\n🎉 **Tour complete!** Go explore the sky!",
            current_step,
        )
        return
    try:
        lat, lon = geocode(city)
        engine = get_engine(ollama_host, model)
        header = f"\n\n---\n\n### 🚀 Stop {current_step + 1}" f" of {total_steps}\n\n"
        logger.debug("Tour step %d/%d", current_step + 1, total_steps)
        accumulated = previous_md + header
        for chunk in engine.guided_tour_step_stream(
            lat, lon, step=current_step, total_steps=total_steps
        ):
            accumulated += chunk or ""
            yield accumulated, current_step + 1
    except _OLLAMA_ERRORS as exc:
        yield previous_md + f"\n\n---\n\n{_ollama_error_msg(exc)}", current_step


# ═══════════════════════════════════════════════════════════
#  Tab 4 — Sky Comparison
# ═══════════════════════════════════════════════════════════
def fn_compare(hours: float, city: str, ollama_host: str, model: str):
    logger.info("Sky comparison: %s +%.1fh", city, hours)
    try:
        lat, lon = geocode(city)
        engine = get_engine(ollama_host, model)

        # Phase 1: charts + planet lists (no LLM, fast)
        chart_now = bytes_to_pil(engine.render_chart(lat, lon))
        future = datetime.now(timezone.utc) + timedelta(hours=hours)
        chart_later = bytes_to_pil(engine.render_chart(lat, lon, when=future))

        narration_acc = ""
        first = True
        for sky_now, sky_later, _td, chunk in engine.compare_skies_stream(
            lat, lon, hours_ahead=hours
        ):
            if first:
                # Build planet markdown once from first yield
                planets_now = sky_now.get("planets", [])
                now_md = f"**Planets:** {len(planets_now)}\n\n"
                now_md += "\n".join(
                    f"- 🪐 {p['name']}: {p['direction']} {p['altitude']}°"
                    for p in planets_now
                )
                planets_later = sky_later.get("planets", [])
                later_md = f"**Planets:** {len(planets_later)}\n\n"
                later_md += "\n".join(
                    f"- 🪐 {p['name']}: {p['direction']} {p['altitude']}°"
                    for p in planets_later
                )
                first = False
                yield (
                    chart_now,
                    now_md,
                    chart_later,
                    later_md,
                    "⏳ Gemma is narrating the transformation…",
                )

            # Phase 2: stream narration chunks
            narration_acc += chunk or ""
            if chunk:
                yield chart_now, now_md, chart_later, later_md, narration_acc

    except _OLLAMA_ERRORS as exc:
        msg = _ollama_error_msg(exc)
        yield None, msg, None, "", msg


# ═══════════════════════════════════════════════════════════
#  Tab 5 — Identify Photo
# ═══════════════════════════════════════════════════════════
def fn_identify(image, city: str, ollama_host: str, model: str):
    if image is None:
        return "Please upload a night sky photo first."

    try:
        lat, lon = geocode(city)
        engine = get_engine(ollama_host, model)
        logger.info("Identifying photo from %s", city)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            if isinstance(image, Image.Image):
                image.save(tmp, format="JPEG")
            else:
                tmp.write(image)
            tmp_path = tmp.name

        result = engine.identify_photo(tmp_path, lat=lat, lon=lon)
        gemma_id = result["gemma_identification"]

        lines = []
        constellations = gemma_id.get("constellations", [])
        if constellations:
            lines.append("### ✨ Constellations Found")
            for c in constellations:
                confidence = c.get("confidence", "?")
                emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(
                    confidence, "⚪"
                )
                lines.append(
                    f"{emoji} **{c.get('name', '?')}** ({confidence} confidence)"
                )

        stars = gemma_id.get("stars", [])
        if stars:
            lines.append("\n### ⭐ Stars Identified")
            for s in stars:
                lines.append(f"- **{s.get('name', '?')}** — {s.get('brightness', '')}")

        planets = gemma_id.get("planets", [])
        if planets:
            lines.append("\n### 🪐 Planets Spotted")
            for p in planets:
                lines.append(f"- **{p.get('name', '?')}** — {p.get('position', '')}")

        moon_data = gemma_id.get("moon", {})
        if moon_data.get("visible"):
            lines.append(f"\n### 🌙 Moon: {moon_data.get('phase', 'visible')}")

        validation = result.get("ephemeris_validation", {})
        if validation.get("confirmed"):
            lines.append("\n### ✅ Confirmed by Ephemeris")
            for item in validation["confirmed"]:
                lines.append(f"- {item}")

        if "raw_analysis" in gemma_id:
            lines.append(f"\n### Analysis\n{gemma_id['raw_analysis']}")

        return "\n".join(lines) if lines else "No objects identified."
    except _OLLAMA_ERRORS as exc:
        return _ollama_error_msg(exc)


# ═══════════════════════════════════════════════════════════
#  Tab 6 — Deep Dive
# ═══════════════════════════════════════════════════════════
def fn_explain(object_name: str, ollama_host: str, model: str):
    if not object_name.strip():
        yield "Please enter an object name."
        return
    logger.info("Deep dive requested: %s", object_name)
    try:
        engine = get_engine(ollama_host, model)
        accumulated = ""
        for chunk in engine.explain_stream(object_name.strip()):
            accumulated += chunk or ""
            yield accumulated
    except _OLLAMA_ERRORS as exc:
        yield _ollama_error_msg(exc)


# ═══════════════════════════════════════════════════════════
#  Tab 7 — Plan Session
# ═══════════════════════════════════════════════════════════
def fn_plan(preferences: str, city: str, ollama_host: str, model: str):
    logger.info("Session plan requested for %s", city)
    try:
        lat, lon = geocode(city)
        engine = get_engine(ollama_host, model)
        plan = engine.plan_session(lat, lon, preferences=preferences)
        chart = bytes_to_pil(engine.render_chart(lat, lon))
        return plan, chart
    except _OLLAMA_ERRORS as exc:
        return _ollama_error_msg(exc), None


# ═══════════════════════════════════════════════════════════
#  Build the Gradio app
# ═══════════════════════════════════════════════════════════
with gr.Blocks(
    title="StarLens — Gemma 4 Night Sky Companion",
    theme=gr.themes.Soft(  # type: ignore[attr-defined]
        primary_hue=gr.themes.colors.blue,  # type: ignore[attr-defined]
        secondary_hue=gr.themes.colors.purple,  # type: ignore[attr-defined]
        neutral_hue=gr.themes.colors.slate,  # type: ignore[attr-defined]
        font=[  # type: ignore[attr-defined]
            gr.themes.GoogleFont("Inter"),  # type: ignore[attr-defined]
            "system-ui",
            "sans-serif",
        ],
        font_mono=[
            gr.themes.GoogleFont("JetBrains Mono"),  # type: ignore[attr-defined]
            "monospace",
        ],
    ).set(
        # ── Light mode ──
        body_background_fill="#f8f9fc",
        block_background_fill="#ffffff",
        block_border_color="rgba(100,130,220,0.12)",
        block_label_text_color="#5c6784",
        block_title_text_color="#3b6ce7",
        input_background_fill="#f0f2f8",
        input_border_color="rgba(100,130,220,0.2)",
        button_primary_background_fill="linear-gradient(135deg, #4285F4, #6366f1)",
        button_primary_text_color="white",
        button_secondary_background_fill="#eef1f8",
        button_secondary_text_color="#4255d4",
        border_color_primary="rgba(100,130,220,0.15)",
        # ── Dark mode ──
        body_background_fill_dark="#0a0e1a",
        block_background_fill_dark="#0f1424",
        block_border_color_dark="rgba(100,130,220,0.12)",
        block_label_text_color_dark="#8899bb",
        block_title_text_color_dark="#8ab4f8",
        input_background_fill_dark="#111832",
        input_border_color_dark="rgba(100,130,220,0.2)",
        button_primary_background_fill_dark="linear-gradient(135deg, #4285F4, #6366f1)",
        button_primary_text_color_dark="white",
        button_secondary_background_fill_dark="#1a2040",
        button_secondary_text_color_dark="#8ab4f8",
        border_color_primary_dark="rgba(100,130,220,0.2)",
    ),
    css=CUSTOM_CSS,
) as app:

    # ── Header ──────────────────────────────────────────
    gr.HTML(HEADER_HTML)

    # ── Settings row (always visible) ───────────────────
    with gr.Row():
        with gr.Column(scale=2):
            city_input = gr.Textbox(
                label="📍 Location",
                value=settings.app.default_city,
                placeholder="City name, e.g. 'Tokyo, Japan'",
            )
        with gr.Column(scale=1):
            host_input = gr.Textbox(
                label="⚙️ Ollama Host",
                value=settings.ollama.host,
            )
        with gr.Column(scale=1):
            model_select = gr.Dropdown(
                label="🧠 Gemma 4 Model",
                choices=settings.ollama.available_models,
                value=settings.ollama.model_reason,
            )

    # ── Tabs ────────────────────────────────────────────
    with gr.Tabs():

        # ─── Tab 1: Tonight's Sky ───────────────────────
        with gr.TabItem("🌌 Tonight's Sky", id="sky"):
            gr.Markdown(
                "## What's Up Tonight?\n"
                "*Real-time sky data + Gemma 4 narration "
                "+ multimodal analysis*"
            )
            sky_btn = gr.Button(
                "🌟 Show me tonight's sky",
                variant="primary",
                size="lg",
            )

            with gr.Row():
                sky_chart = gr.Image(label="Sky Chart", type="pil", height=500)
                sky_info = gr.Markdown(label="Sky Data")

            sky_narration = gr.Markdown(label="🧠 Gemma 4 Narration")

            sky_btn.click(
                fn=fn_tonights_sky,
                inputs=[city_input, host_input, model_select],
                outputs=[sky_chart, sky_info, sky_narration],
            )

            gr.Markdown("---")
            gr.Markdown(
                "### ❓ Ask *Why?* About Any Object\n"
                "*Gemma 4 explains the orbital mechanics "
                "behind any object's position*"
            )
            why_input = gr.Textbox(
                label="Object name",
                placeholder="e.g., Jupiter, Sirius, Moon",
            )
            why_btn = gr.Button(
                "❓ Why is it there?",
                variant="secondary",
            )
            why_output = gr.Markdown(label="🧠 Gemma 4 Explains")

            why_btn.click(
                fn=fn_why_object,
                inputs=[
                    why_input,
                    city_input,
                    host_input,
                    model_select,
                ],
                outputs=[why_output],
                show_progress="full",
            )
            why_input.submit(
                fn=fn_why_object,
                inputs=[
                    why_input,
                    city_input,
                    host_input,
                    model_select,
                ],
                outputs=[why_output],
                show_progress="full",
            )

            gr.Markdown("---")
            gr.Markdown(
                "### 👁️ Multimodal Round-Trip\n"
                "*Gemma 4 analyzes its OWN rendered sky chart "
                "using vision*"
            )
            chart_analyze_btn = gr.Button(
                "🔬 Let Gemma analyze the chart",
                variant="secondary",
            )
            chart_analysis_output = gr.Markdown(label="🧠 Chart Analysis")

            chart_analyze_btn.click(
                fn=fn_analyze_chart,
                inputs=[city_input, host_input, model_select],
                outputs=[chart_analysis_output],
            )

        # ─── Tab 2: Sky Chat ────────────────────────────
        with gr.TabItem("💬 Sky Chat", id="chat"):
            gr.Markdown(
                "## 💬 Chat with Gemma 4 About the Sky\n"
                "*Ask anything — Gemma 4 has real-time ephemeris "
                "data loaded. It knows exactly what's above you "
                "right now.*"
            )
            chatbot = gr.Chatbot(
                label="StarLens Chat",
                height=500,
            )
            chat_input = gr.Textbox(
                label="Your question",
                placeholder=("What's that bright thing in the south?"),
                show_label=False,
            )
            with gr.Row():
                chat_send = gr.Button("Send", variant="primary", scale=3)
                chat_clear = gr.Button("🗑️ Clear", variant="secondary", scale=1)

            chat_send.click(
                fn=fn_chat,
                inputs=[
                    chat_input,
                    chatbot,
                    city_input,
                    host_input,
                    model_select,
                ],
                outputs=[chatbot, chat_input],
            )
            chat_input.submit(
                fn=fn_chat,
                inputs=[
                    chat_input,
                    chatbot,
                    city_input,
                    host_input,
                    model_select,
                ],
                outputs=[chatbot, chat_input],
            )
            chat_clear.click(fn=lambda: ([], ""), outputs=[chatbot, chat_input])

        # ─── Tab 3: Guided Tour ─────────────────────────
        with gr.TabItem("🚀 Guided Tour", id="tour"):
            gr.Markdown(
                "## 🚀 Gemma-Guided Sky Tour\n"
                "*Let Gemma 4 walk you through the sky "
                "step by step*"
            )

            tour_steps_slider = gr.Slider(
                label="Number of tour stops",
                minimum=3,
                maximum=8,
                value=5,
                step=1,
            )

            with gr.Row():
                tour_start_btn = gr.Button("🚀 Start Tour", variant="primary", scale=1)
                tour_next_btn = gr.Button("➡️ Next Stop", variant="secondary", scale=1)

            tour_output = gr.Markdown(
                label="Tour",
                value="*Press Start Tour to begin...*",
            )
            tour_step_state = gr.State(value=0)

            tour_start_btn.click(
                fn=fn_tour_start,
                inputs=[
                    tour_steps_slider,
                    city_input,
                    host_input,
                    model_select,
                ],
                outputs=[tour_output, tour_step_state],
            )
            tour_next_btn.click(
                fn=fn_tour_next,
                inputs=[
                    tour_step_state,
                    tour_steps_slider,
                    tour_output,
                    city_input,
                    host_input,
                    model_select,
                ],
                outputs=[tour_output, tour_step_state],
            )

        # ─── Tab 4: Sky Comparison ──────────────────────
        with gr.TabItem("🔄 Sky Comparison", id="compare"):
            gr.Markdown(
                "## 🔄 Sky Comparison\n"
                "*How will the sky change? Gemma 4 compares "
                "two moments and narrates the transformation.*"
            )

            compare_hours = gr.Slider(
                label="Hours ahead to compare",
                minimum=1,
                maximum=12,
                value=3,
                step=0.5,
            )
            compare_btn = gr.Button("🔄 Compare Sky", variant="primary", size="lg")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🕐 Sky Now")
                    compare_chart_now = gr.Image(label="Now", type="pil", height=350)
                    compare_info_now = gr.Markdown()
                with gr.Column():
                    gr.Markdown("### 🕐 Sky Later")
                    compare_chart_later = gr.Image(
                        label="Later", type="pil", height=350
                    )
                    compare_info_later = gr.Markdown()

            compare_narration = gr.Markdown(
                label="🧠 Gemma 4 Narrates the Transformation"
            )

            compare_btn.click(
                fn=fn_compare,
                inputs=[
                    compare_hours,
                    city_input,
                    host_input,
                    model_select,
                ],
                outputs=[
                    compare_chart_now,
                    compare_info_now,
                    compare_chart_later,
                    compare_info_later,
                    compare_narration,
                ],
            )

        # ─── Tab 5: Identify Photo ──────────────────────
        with gr.TabItem("📸 Identify Photo", id="identify"):
            gr.Markdown(
                "## 📸 Identify Your Night Sky Photo\n"
                "*Upload a photo and Gemma 4 will identify "
                "stars, constellations, and planets*"
            )

            with gr.Row():
                photo_input = gr.Image(
                    label="Upload a night sky photo",
                    type="pil",
                    height=400,
                )
                identify_output = gr.Markdown(
                    value=(
                        "### 🧠 Gemma 4 Identification\n\n"
                        "Upload a photo and click **Identify Objects** "
                        "to let Gemma 4 analyze your night sky."
                    ),
                    label="🧠 Gemma 4 Identification",
                )

            identify_btn = gr.Button(
                "🔍 Identify Objects",
                variant="primary",
                size="lg",
            )
            identify_btn.click(
                fn=fn_identify,
                inputs=[
                    photo_input,
                    city_input,
                    host_input,
                    model_select,
                ],
                outputs=[identify_output],
                show_progress="full",
            )

        # ─── Tab 6: Deep Dive ───────────────────────────
        with gr.TabItem("🔍 Deep Dive", id="explain"):
            gr.Markdown(
                "## 🔍 Deep Dive — Learn About Any Object\n"
                "*Gemma 4 explains the science, mythology, "
                "and observation tips*"
            )
            explain_input = gr.Textbox(
                label="What do you want to learn about?",
                placeholder=(
                    "e.g., Orion, Sirius, Jupiter, " "Andromeda Galaxy, Pleiades..."
                ),
            )
            explain_btn = gr.Button("🧠 Explain", variant="primary", size="lg")
            explain_output = gr.Markdown(label="🧠 Gemma 4 Deep Dive")

            explain_btn.click(
                fn=fn_explain,
                inputs=[explain_input, host_input, model_select],
                outputs=[explain_output],
            )
            explain_input.submit(
                fn=fn_explain,
                inputs=[explain_input, host_input, model_select],
                outputs=[explain_output],
            )

        # ─── Tab 7: Plan Session ────────────────────────
        with gr.TabItem("📋 Plan Session", id="plan"):
            gr.Markdown(
                "## 📋 Observation Planner\n"
                "*Gemma 4 creates an optimized stargazing "
                "plan for tonight*"
            )
            plan_prefs = gr.Textbox(
                label="Any preferences? (optional)",
                placeholder=(
                    "e.g., I have binoculars, I'm interested "
                    "in planets, I'm a beginner..."
                ),
                lines=3,
            )
            plan_btn = gr.Button(
                "📋 Plan My Session",
                variant="primary",
                size="lg",
            )
            plan_output = gr.Markdown(label="🧠 Your Observation Plan")
            plan_chart = gr.Image(
                label="Tonight's Sky Chart",
                type="pil",
                height=400,
            )

            plan_btn.click(
                fn=fn_plan,
                inputs=[
                    plan_prefs,
                    city_input,
                    host_input,
                    model_select,
                ],
                outputs=[plan_output, plan_chart],
            )

    # ── Footer ──────────────────────────────────────────
    gr.HTML(
        '<div class="starlens-footer">'
        "Built with ❤️ for the "
        '<a href="https://dev.to/devteam/join-the-gemma-4-challenge-'
        '3000-prize-pool-for-ten-winners-23in">'
        "DEV.to Gemma 4 Challenge</a>"
        " — Powered by "
        '<a href="https://ai.google.dev/gemma">Gemma 4</a>, '
        '<a href="https://rhodesmill.org/skyfield/">Skyfield</a>, '
        "and NASA/JPL ephemeris"
        "</div>"
    )


if __name__ == "__main__":
    app.launch(
        server_name=settings.app.host,
        server_port=settings.app.port,
        share=settings.app.share,
        favicon_path=str(_LOGO_PATH) if _LOGO_PATH.exists() else None,
    )
