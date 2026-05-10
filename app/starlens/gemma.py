"""Gemma 4 client — the brain of StarLens.

Uses Google AI Studio (Gemini API) to access Gemma 4 natively:
- Multimodal: identifies celestial objects from night sky photos
- Reasoning: explains astronomical phenomena with chain-of-thought
- 256K context: processes full star catalogs for observation planning
"""

import base64
import json
import logging
from pathlib import Path

from google import genai
from google.genai import types

from . import cache
from .settings import settings

logger = logging.getLogger(__name__)

_cfg = settings.gemini
_opts = settings.options
_ttl_llm = settings.redis.ttl_llm

# ── System prompts ────────────────────────────────────────────────────────────
_SYS_ASTRONOMER = (
    "You are an expert astronomer who makes the cosmos accessible and exciting."
)

_SYS_GUIDE = (
    "You are an experienced stargazing guide."
    " Be precise with times, directions, and practical advice."
)

_SYS_TOUR_GUIDE = (
    "You are an enthusiastic, knowledgeable stargazing guide. Be specific with"
    " directions and altitudes. Write as if you're standing next to the person."
)

_SYS_ORBITAL = (
    "You are an astronomer who makes orbital mechanics and celestial geometry"
    " intuitive and exciting."
)

_SYS_CHAT = (
    "You are StarLens, an expert AI astronomer companion. You have access to "
    "real-time astronomical data computed from NASA/JPL ephemeris and the Hipparcos "
    "star catalog. Use this data to answer questions accurately.\n\n"
    "When the user asks about something in the sky, reference the real computed "
    "positions. Be conversational, enthusiastic, and precise.\n\n"
    "If asked 'what's that bright thing in the east?', look at the data for objects "
    "in the east with high altitude or low magnitude (bright)."
)

_TOUR_STEP_LABELS = [
    "most impressive",
    "second most interesting",
    "a hidden gem",
    "something unexpected",
    "the grand finale",
]


class GemmaClient:
    """Gemma 4 client via Google AI Studio for astronomical intelligence."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or _cfg.api_key
        self.client = genai.Client(api_key=self.api_key)
        self.default_model = model or _cfg.model_reason

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _config(
        self, temperature: float, system: str | None = None
    ) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=_cfg.max_output_tokens,
            system_instruction=system,
        )

    def _call(
        self,
        contents: str | list,
        *,
        system: str | None = None,
        temperature: float = 0.5,
        model: str | None = None,
    ) -> str:
        """Single non-streaming generation call."""
        response = self.client.models.generate_content(
            model=model or self.default_model,
            contents=contents,
            config=self._config(temperature, system),
        )
        return response.text or ""

    def _stream(
        self,
        contents: str | list,
        *,
        system: str | None = None,
        temperature: float = 0.5,
        model: str | None = None,
    ):
        """Streaming generation — yields text chunks."""
        for chunk in self.client.models.generate_content_stream(
            model=model or self.default_model,
            contents=contents,
            config=self._config(temperature, system),
        ):
            yield chunk.text or ""

    # ── Prompt builders ───────────────────────────────────────────────────────

    @staticmethod
    def _explain_prompt(object_name: str, catalog_context: str) -> str:
        prompt = (
            f"Tell me everything fascinating about **{object_name}**:\n\n"
            "1. **What it is** — type, distance, physical properties\n"
            "2. **Mythology & history** — stories from different cultures\n"
            "3. **How to find it** — practical observation tips\n"
            "4. **Why it matters** — scientific significance\n"
            "5. **Fun fact** — something surprising most people don't know\n\n"
            "Be engaging, accurate, and inspiring. Write for a curious beginner."
        )
        if catalog_context:
            prompt = f"## Reference Data\n\n{catalog_context}\n\n{prompt}"
        return prompt

    @staticmethod
    def _tour_prompt(sky_context: str, step: int, total_steps: int) -> str:
        label = _TOUR_STEP_LABELS[min(step, len(_TOUR_STEP_LABELS) - 1)]
        intro = (
            "Start with the most impressive object visible right now."
            if step == 0
            else ""
        )
        outro = (
            "This is the final stop — end with something inspiring."
            if step == total_steps - 1
            else ""
        )
        return (
            f"You are a stargazing guide leading a live sky tour. "
            f"This is step {step + 1} of {total_steps}.\n\n"
            f"## Tonight's Sky Data\n{sky_context}\n\n"
            "Generate ONLY this one tour stop. Include:\n"
            "- **Direction to face** (cardinal direction)\n"
            "- **Where to look** (altitude in degrees — 'halfway up' or 'near the horizon')\n"
            "- **What you'll see** and why it's interesting\n"
            "- **A surprising fact** about this object\n"
            "- **Transition** — a teaser for the next stop\n\n"
            f"{intro}{outro}\n"
            f"Step {step + 1}: pick the {label} object."
        )

    @staticmethod
    def _chart_prompt(sky_context: str) -> str:
        return (
            "You are looking at a sky chart rendered from real ephemeris data. "
            "Analyze this chart and provide:\n\n"
            "1. **What's most striking** — the standout objects or patterns\n"
            "2. **Constellation highlights** — which constellations are well-placed\n"
            "3. **Planet positions** — identify the orange dots (planets)\n"
            "4. **Best targets** — what should an observer focus on first\n"
            "5. **Hidden treasures** — deep-sky objects near visible constellations\n\n"
            f"Cross-reference with this computed data:\n{sky_context}\n\n"
            "Be specific and observational — describe what you SEE in the chart."
        )

    @staticmethod
    def _why_prompt(object_name: str, object_data: str, sky_context: str) -> str:
        return (
            f"A stargazer is looking at **{object_name}** and asks: 'Why is it there?'\n\n"
            f"Object data: {object_data}\n\n"
            f"Full sky context:\n{sky_context}\n\n"
            "Explain in an engaging way:\n"
            "1. **Why it's visible right now** — Earth's position, season, time of night\n"
            "2. **Why it's in that direction** — orbital mechanics or stellar position\n"
            "3. **How it will move** — what happens over the next few hours\n"
            "4. **When it's best** — peak viewing times this month\n"
            "5. **Connection to other objects** — what's nearby and why\n\n"
            "Make orbital mechanics intuitive. Use analogies."
        )

    @staticmethod
    def _compare_prompt(sky_now: str, sky_later: str, time_diff: str) -> str:
        return (
            f"Compare these two sky snapshots ({time_diff} apart) and tell the story "
            f"of how the sky transforms:\n\n"
            f"## Sky NOW\n{sky_now}\n\n"
            f"## Sky LATER\n{sky_later}\n\n"
            "Describe:\n"
            "1. **What appears** — objects that rise into view\n"
            "2. **What disappears** — objects that set below the horizon\n"
            "3. **What moves** — how constellations and planets shift\n"
            "4. **The drama** — any particularly beautiful moments "
            "(planet near Moon, constellation rising, etc.)\n"
            "5. **Best moment** — when should the observer be outside "
            "for the peak experience?\n\n"
            "Write as a narrative — tell the story of the night unfolding."
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def identify_sky(
        self, image_path: str, location: str = "", timestamp: str = ""
    ) -> dict:
        """Identify celestial objects in a night sky photograph.

        Uses Gemma 4's native multimodal vision to recognize stars, constellations,
        planets, the Moon, and deep-sky objects from a photo.
        """
        logger.info(
            "LLM identify_sky: model=%s, image=%s", _cfg.model_identify, image_path
        )
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(path, "rb") as f:
            image_bytes = f.read()

        location_hint = f"\nPhoto taken from: {location}" if location else ""
        time_hint = f"\nPhoto taken at: {timestamp}" if timestamp else ""

        prompt = (
            f"You are an expert astronomer analyzing a night sky photograph."
            f"{location_hint}{time_hint}\n\n"
            "Analyze this image and identify:\n"
            "1. **Constellations** visible (even partial ones)\n"
            "2. **Bright stars** with their names (e.g., Sirius, Vega, Betelgeuse)\n"
            "3. **Planets** visible (they don't twinkle and appear as steady dots)\n"
            "4. **The Moon** if present, and its phase\n"
            "5. **Deep-sky objects** if any (nebulae, galaxies, clusters)\n"
            "6. **Overall sky conditions** (light pollution, clarity, direction facing)\n\n"
            "Respond with a JSON object:\n"
            '{"constellations": [{"name": "...", "confidence": "high/medium/low"}], '
            '"stars": [{"name": "...", "brightness": "..."}], '
            '"planets": [{"name": "...", "position": "..."}], '
            '"moon": {"visible": true/false, "phase": "..."}, '
            '"deep_sky": [{"name": "...", "type": "..."}], '
            '"sky_conditions": "...", '
            '"direction_facing": "estimated cardinal direction"}'
        )

        suffix = path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime)
        raw = self._call(
            [image_part, prompt],
            temperature=_opts.temperature_identify,
            model=_cfg.model_identify,
        )
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
        return {"raw_analysis": raw}

    def explain_object(self, object_name: str, catalog_context: str = "") -> str:
        """Use Gemma 4 reasoning to explain an astronomical object.

        Covers mythology, science, observation tips, and interesting facts.
        The 256K context window allows feeding full catalog data.
        """
        cached = cache.get("explain", object_name)
        if cached is not None:
            logger.debug("Cache hit for explain: %s", object_name)
            return cached

        logger.info("LLM explain_object: %s, model=%s", object_name, self.default_model)
        result = self._call(
            self._explain_prompt(object_name, catalog_context),
            system=_SYS_ASTRONOMER,
            temperature=_opts.temperature_explain,
        )
        cache.put("explain", object_name, value=result, ttl=_ttl_llm)
        return result

    def explain_object_stream(self, object_name: str, catalog_context: str = ""):
        """Streaming version of explain_object."""
        yield from self._stream(
            self._explain_prompt(object_name, catalog_context),
            system=_SYS_ASTRONOMER,
            temperature=_opts.temperature_explain,
        )

    def plan_observation(
        self, visible_objects: str, location: str, preferences: str = ""
    ) -> str:
        """Use Gemma 4 reasoning to plan a stargazing session.

        Takes the full list of currently visible objects and creates
        a time-ordered observation plan.
        """
        logger.info(
            "LLM plan_observation: location=%s, model=%s", location, self.default_model
        )
        pref_hint = f"\nObserver preferences: {preferences}" if preferences else ""
        prompt = (
            f"You are a stargazing guide planning tonight's observation session.\n\n"
            f"**Location:** {location}{pref_hint}\n\n"
            f"**Currently visible objects:**\n{visible_objects}\n\n"
            "Create a detailed observation plan:\n"
            "1. **Best time windows** for each object (when they're highest)\n"
            "2. **Viewing order** — optimized so you're not jumping around the sky\n"
            "3. **What to look for** — specific features visible with "
            "naked eye vs binoculars vs telescope\n"
            "4. **Highlights** — what's the single best thing to see tonight and why\n"
            "5. **Photography tips** — camera settings for the best shots\n\n"
            "Be specific with directions (N/S/E/W), altitudes (degrees above horizon), and times."
        )
        return self._call(prompt, system=_SYS_GUIDE, temperature=_opts.temperature_plan)

    def narrate_sky(self, sky_context: str) -> str:
        """Generate an engaging narrative about tonight's sky."""
        cached = cache.get("narrate", sky_context)
        if cached is not None:
            logger.debug("Cache hit for narrate")
            return cached

        logger.info("LLM narrate_sky: model=%s", self.default_model)
        prompt = (
            f"Based on this astronomical data, write a 2-3 paragraph engaging narration "
            f"of tonight's sky — something you'd share with a friend to get them excited "
            f"about going outside and looking up. Mix science with wonder.\n\n{sky_context}"
        )
        result = self._call(prompt, temperature=_opts.temperature_narrate)
        cache.put("narrate", sky_context, value=result, ttl=_ttl_llm)
        return result

    def chat(
        self, user_message: str, sky_context: str, history: list[dict] | None = None
    ) -> str:
        """Interactive sky chat — Gemma answers questions grounded in real sky data.

        The sky context (computed ephemeris) is injected as system knowledge
        so Gemma reasons from real data, not hallucination.
        """
        system = f"{_SYS_CHAT}\n\n## Current Sky Data\n\n{sky_context}"

        contents: list = []
        for msg in history or []:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
            )
        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
        )

        logger.info(
            "LLM chat: %d messages, model=%s", len(contents), self.default_model
        )
        response = self.client.models.generate_content(
            model=self.default_model,
            contents=contents,
            config=self._config(_opts.temperature_chat, system),
        )
        return response.text or ""

    def guided_tour(self, sky_context: str, step: int = 0, total_steps: int = 5) -> str:
        """Generate one step of a guided sky tour."""
        return self._call(
            self._tour_prompt(sky_context, step, total_steps),
            system=_SYS_TOUR_GUIDE,
            temperature=_opts.temperature_tour,
        )

    def guided_tour_stream(self, sky_context: str, step: int = 0, total_steps: int = 5):
        """Streaming version of guided_tour."""
        yield from self._stream(
            self._tour_prompt(sky_context, step, total_steps),
            system=_SYS_TOUR_GUIDE,
            temperature=_opts.temperature_tour,
        )

    def analyze_chart(self, chart_image_b64: str, sky_context: str) -> str:
        """Multimodal round-trip: Gemma analyzes a sky chart using native vision."""
        image_part = types.Part.from_bytes(
            data=base64.b64decode(chart_image_b64), mime_type="image/png"
        )
        return self._call(
            [image_part, self._chart_prompt(sky_context)],
            temperature=_opts.temperature_chart,
            model=_cfg.model_identify,
        )

    def analyze_chart_stream(self, chart_image_b64: str, sky_context: str):
        """Streaming version of analyze_chart — yields chunks as they arrive."""
        image_part = types.Part.from_bytes(
            data=base64.b64decode(chart_image_b64), mime_type="image/png"
        )
        yield from self._stream(
            [image_part, self._chart_prompt(sky_context)],
            temperature=_opts.temperature_chart,
            model=_cfg.model_identify,
        )

    def explain_why(self, object_name: str, object_data: str, sky_context: str) -> str:
        """Explain WHY an object is where it is — orbital mechanics and geometry."""
        return self._call(
            self._why_prompt(object_name, object_data, sky_context),
            system=_SYS_ORBITAL,
            temperature=_opts.temperature_why,
        )

    def explain_why_stream(self, object_name: str, object_data: str, sky_context: str):
        """Streaming version — yields text chunks as Gemma generates them."""
        yield from self._stream(
            self._why_prompt(object_name, object_data, sky_context),
            system=_SYS_ORBITAL,
            temperature=_opts.temperature_why,
        )

    def compare_skies(self, sky_now: str, sky_later: str, time_diff: str) -> str:
        """Compare two sky states and narrate what changes."""
        return self._call(
            self._compare_prompt(sky_now, sky_later, time_diff),
            temperature=_opts.temperature_tour,
        )

    def compare_skies_stream(self, sky_now: str, sky_later: str, time_diff: str):
        """Streaming compare — yields chunks as Gemma narrates the transformation."""
        logger.info("LLM compare_skies_stream: model=%s", self.default_model)
        yield from self._stream(
            self._compare_prompt(sky_now, sky_later, time_diff),
            temperature=_opts.temperature_tour,
        )
