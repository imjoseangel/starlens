*This is a submission for the [Gemma 4 Challenge: Build with Gemma 4](https://dev.to/challenges/google-gemma-2026-05-06)*

## What I Built

**StarLens** — an AI-powered night sky companion where Gemma 4 is the protagonist, not a bolt-on.

StarLens computes the real-time positions of 118,000 stars, 8 planets, the Sun, and Moon using NASA/JPL ephemeris data (Skyfield + Hipparcos catalog), then feeds that data into Gemma 4 so every answer is grounded in actual science — not hallucination.

It features **7 interactive tabs**, each showcasing a different Gemma 4 capability:

- **🌌 Tonight's Sky** — Real-time sky map with Gemma narration. Tap any object and ask "Why is it there?" — Gemma explains the orbital mechanics.
- **💬 Sky Chat** — Multi-turn conversation with the full sky state injected as context. Ask "What's that bright thing in the south?" and Gemma looks up the real computed positions.
- **🚀 Guided Tour** — Gemma leads you through the sky step-by-step with exact directions, altitudes, and surprising facts — like a live planetarium show.
- **🔄 Sky Comparison** — Side-by-side charts showing how the sky transforms over 1-12 hours, with Gemma narrating what rises, sets, and shifts.
- **📸 Photo Identification** — Upload a night sky photo and Gemma's vision identifies constellations, stars, and planets — then the engine cross-validates against computed ephemeris positions.
- **🔍 Deep Dive** — Ask about any celestial object. Gemma receives the full star catalog via its 128K context window for comprehensive explanations.
- **📋 Observation Planner** — Gemma creates an optimized stargazing plan with best times, viewing order, and photography tips.

The **multimodal round-trip** is the highlight: StarLens renders a sky chart from ephemeris data, then feeds it back to Gemma 4's vision model for analysis — proving Gemma can both consume and reason about astronomical visualizations.

## Demo

![StarLens Demo](https://github.com/imjoseangel/starlens/blob/devel/docs/demo.gif?raw=true)

## Code

{% github imjoseangel/starlens %}

## How I Used Gemma 4

StarLens exercises **7 distinct Gemma 4 capabilities** through 9 specialized methods:

| Capability               | Method                              | Model              | Why                                                         |
|--------------------------|-------------------------------------|--------------------|-------------------------------------------------------------|
| **Multimodal Vision**    | `identify_sky()`, `analyze_chart()` | `gemma4:e4b`       | Fast image analysis for photo ID and chart round-trip       |
| **128K Context Window**  | `explain_object()`                  | `gemma4:31b-cloud` | Loads the entire Hipparcos star catalog for deep reasoning  |
| **Multi-Turn Chat**      | `chat()`                            | `gemma4:31b-cloud` | Conversational sky Q&A with ephemeris as system context     |
| **Structured Reasoning** | `explain_why()`                     | `gemma4:31b-cloud` | Orbital mechanics and celestial geometry explanations       |
| **Narrative Generation** | `guided_tour()`, `narrate_sky()`    | `gemma4:31b-cloud` | Step-by-step tours and engaging sky descriptions            |
| **Temporal Reasoning**   | `compare_skies()`                   | `gemma4:31b-cloud` | Narrates how the sky transforms over hours                  |
| **Cross-Validation**     | `identify_sky()` + engine           | `gemma4:e4b`       | AI identifications checked against real ephemeris positions |

### Why these models?

- **`gemma4:e4b`** for vision tasks — it's fast enough for interactive photo analysis and the multimodal round-trip where latency matters.
- **`gemma4:31b-cloud`** for reasoning — the 128K context window is essential for loading full star catalogs, and the deeper reasoning produces better tour narrations and orbital mechanics explanations.

Users can switch models from the sidebar based on their hardware — all 9 methods respect the selection dynamically.

### Architecture

The key insight is **context injection**: every Gemma call receives real computed astronomical data as context, so Gemma reasons from science, not training data. The engine layer orchestrates catalog computations and Gemma calls, cross-validates results, and builds the context that makes Gemma's answers accurate.

Built with Python, Gradio, Skyfield, Ollama, and pydantic-settings. Optional Redis caching and Docker deployment included.
