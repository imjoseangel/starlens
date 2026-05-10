*This is a submission for the [Gemma 4 Challenge: Build with Gemma 4](https://dev.to/challenges/google-gemma-2026-05-06)*

## What I Built

**StarLens** — an AI-powered night sky companion where Gemma 4 is the protagonist, not a bolt-on.

StarLens computes the real-time positions of 118,000 stars, 8 planets, the Sun, and Moon using NASA/JPL ephemeris data (Skyfield + Hipparcos catalog), then feeds that data directly into Gemma 4 via the **Gemini API on Google AI Studio** — so every answer is grounded in actual science, not hallucination.

It features **7 interactive tabs**, each showcasing a different Gemma 4 capability:

- **🌌 Tonight's Sky** — Real-time sky map with Gemma narration. Tap any object and ask "Why is it there?" — Gemma explains the orbital mechanics behind its current position.
- **💬 Sky Chat** — Multi-turn conversation with the full ephemeris state injected as system context. Ask "What's that bright thing in the south?" and Gemma cross-references real computed positions to answer.
- **🚀 Guided Tour** — Gemma leads you through the sky step-by-step with exact directions, altitudes, and surprising facts — like a live planetarium show in your pocket.
- **🔄 Sky Comparison** — Side-by-side charts showing how the sky transforms over 1–12 hours, with Gemma narrating what rises, sets, and shifts.
- **📸 Photo Identification** — Upload a night sky photo and Gemma's native multimodal vision identifies constellations, stars, and planets — then the engine cross-validates against computed ephemeris positions.
- **🔍 Deep Dive** — Ask about any celestial object. Gemma receives the full Hipparcos star catalog via the **256K context window** for comprehensive explanations backed by real data.
- **📋 Observation Planner** — Gemma creates an optimized stargazing plan with best times, viewing order, equipment tips, and astrophotography settings.

The **multimodal round-trip** is the highlight: StarLens renders a sky chart from ephemeris data, then feeds it back to Gemma 4's vision model for analysis — proving Gemma can both produce and reason about astronomical visualizations in the same session.

## Demo

[🎬 Watch the full demo (2:30)](https://github.com/imjoseangel/starlens/blob/devel/docs/demo.m4v?raw=true)

🚀 **[Try it live on Hugging Face Spaces](https://huggingface.co/spaces/imjoseangel/starlens)**

## Code

{% github imjoseangel/starlens %}

## How I Used Gemma 4

StarLens exercises **7 distinct Gemma 4 capabilities** through 9 specialized methods, all accessed via the **Gemini API on Google AI Studio** — no local GPU required:

| Capability               | Method                              | Model                | Why                                                                       |
|--------------------------|-------------------------------------|----------------------|---------------------------------------------------------------------------|
| **Multimodal Vision**    | `identify_sky()`, `analyze_chart()` | `gemma-4-26b-a4b-it` | MoE activates only 4B params — fast image analysis with low latency       |
| **256K Context Window**  | `explain_object()`                  | `gemma-4-31b-it`     | Loads the entire Hipparcos star catalog for deep, data-grounded reasoning |
| **Multi-Turn Chat**      | `chat()`                            | `gemma-4-31b-it`     | Conversational sky Q&A with real ephemeris as system context              |
| **Structured Reasoning** | `explain_why()`                     | `gemma-4-31b-it`     | Orbital mechanics and celestial geometry explained intuitively            |
| **Narrative Generation** | `guided_tour()`, `narrate_sky()`    | `gemma-4-31b-it`     | Step-by-step tours and engaging sky descriptions with streaming output    |
| **Temporal Reasoning**   | `compare_skies()`                   | `gemma-4-31b-it`     | Narrates how the sky transforms between two computed sky states           |
| **Cross-Validation**     | `identify_sky()` + engine           | `gemma-4-26b-a4b-it` | AI identifications checked against real ephemeris positions               |

### Why these two models?

StarLens deliberately uses **two Gemma 4 variants** to match the right model to each task — intentional model selection is core to the architecture:

**`gemma-4-26b-a4b-it` (Mixture-of-Experts)** for all vision tasks: photo identification and the multimodal chart round-trip. The MoE architecture activates only 4B parameters per inference, which means much lower latency on the Gemini API — and for interactive vision tasks where the user is waiting, speed matters. The full 26B parameter knowledge is still encoded; you just pay the cost of 4B at runtime.

**`gemma-4-31b-it` (Dense 31B)** for all reasoning, planning, chat, and narrative tasks. The **256K context window** is the decisive factor: loading the complete Hipparcos star catalog (~118,000 entries), the full sky state, and multi-turn chat history simultaneously would be impossible with a smaller context. The dense architecture also produces richer, more nuanced astronomical explanations — orbital mechanics and constellation mythology benefit from depth, not just speed.

Users can switch models from the sidebar at runtime — all 9 AI methods respect the selection dynamically.

### Architecture: Context Injection

The core insight that makes StarLens work is **context injection**:

```sh
- Real Sky Data (Skyfield + JPL ephemeris)
- Engine layer serializes positions
- Injected as system context into every Gemma 4 call
- Gemma reasons from science, not training data
```

Every single Gemma call receives real computed astronomical data — planet altitudes, star magnitudes, constellation visibility percentages, moon phase — as context. Gemma never has to guess where Jupiter is. It's told, with arcsecond precision, and then asked to narrate, explain, or plan around that truth.

The engine layer (`engine.py`) orchestrates catalog computations and Gemma calls, cross-validates photo identifications against ephemeris, builds context strings, and streams results back to the Gradio UI.

### Code quality improvements in this iteration

The Gemma client (`gemma.py`) was refactored to eliminate the duplication that comes from maintaining both streaming and non-streaming versions of every method:

- **`_call()` / `_stream()` helpers** — shared base for all 9 methods; non-streaming and streaming variants differ by one line
- **Static prompt builders** (`_explain_prompt()`, `_tour_prompt()`, `_chart_prompt()`, `_why_prompt()`, `_compare_prompt()`) — prompts are defined once and reused
- **`_find_object_data()`** in `engine.py` — extracted from two identical 15-line blocks in `explain_why` / `explain_why_stream`

Result: ~40% less code in the AI layer, with no loss of functionality.

### No local setup needed

Thanks to the Gemini API, anyone can run StarLens with just:

```bash
pip install -e .
STARLENS_GEMINI_API_KEY=your_key python app/main.py
```

Or try it instantly — no install required — at **[huggingface.co/spaces/imjoseangel/starlens](https://huggingface.co/spaces/imjoseangel/starlens)**.

Get a free API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). No GPU, no model downloads, no local inference server.

### Tech stack

- **[Gemma 4](https://ai.google.dev/gemma)** via **[Google AI Studio](https://aistudio.google.com/)** (Gemini API) — the AI brain
- **[Skyfield](https://rhodesmill.org/skyfield/)** + JPL DE421 — astronomical ephemeris
- **[Hipparcos catalog](https://www.cosmos.esa.int/web/hipparcos)** — 118,000 real stars
- **[Gradio](https://www.gradio.app/)** — dark-themed interactive UI
- **[Matplotlib](https://matplotlib.org/)** — sky chart rendering
- **[pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)** — typed configuration
- **[Hugging Face Spaces](https://huggingface.co/spaces/imjoseangel/starlens)** — live deployment
- Optional: Redis caching, Docker deployment
