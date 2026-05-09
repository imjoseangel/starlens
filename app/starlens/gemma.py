"""Gemma 4 client — the brain of StarLens.

Gemma 4 does ALL the intelligent work:
- Multimodal: identifies celestial objects from night sky photos
- Reasoning: explains astronomical phenomena with chain-of-thought
- 128K context: processes full star catalogs for observation planning
"""

import base64
import json
import logging
from pathlib import Path

import ollama

from . import cache
from .settings import settings

logger = logging.getLogger(__name__)

_cfg = settings.ollama
_opts = settings.options
_ttl_llm = settings.redis.ttl_llm


class GemmaClient:
    """Gemma 4 client via Ollama for astronomical intelligence."""

    def __init__(self, host: str | None = None, model: str | None = None):
        self.host = host or _cfg.host
        self.client = ollama.Client(host=self.host, timeout=_cfg.timeout)
        self.default_model = model or _cfg.model_reason

    def identify_sky(
        self, image_path: str, location: str = "", timestamp: str = ""
    ) -> dict:
        """Identify celestial objects in a night sky photograph.

        Uses Gemma 4's multimodal vision to recognize stars, constellations,
        planets, the Moon, and deep-sky objects from a photo.
        """
        logger.info(
            "LLM identify_sky: model=%s, image=%s", self.default_model, image_path
        )
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

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

        response = self.client.chat(
            model=self.default_model,
            messages=[{"role": "user", "content": prompt, "images": [image_b64]}],
            options={
                "temperature": _opts.temperature_identify,
                "num_ctx": _opts.num_ctx_identify,
            },
        )

        raw = response.message.content or ""
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
        The 128K context window allows feeding full catalog data.
        """
        cached = cache.get("explain", object_name)
        if cached is not None:
            logger.debug("Cache hit for explain: %s", object_name)
            return cached

        logger.info("LLM explain_object: %s, model=%s", object_name, self.default_model)

        prompt = (
            f"You are a passionate astronomer and science communicator.\n\n"
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

        response = self.client.chat(
            model=self.default_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert astronomer who makes"
                        " the cosmos accessible and exciting."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": _opts.temperature_explain,
                "num_ctx": _opts.num_ctx_large,
            },
        )
        result = response.message.content or ""
        cache.put("explain", object_name, value=result, ttl=_ttl_llm)
        return result

    def explain_object_stream(self, object_name: str, catalog_context: str = ""):
        """Streaming version of explain_object."""
        prompt = (
            f"You are a passionate astronomer and science communicator.\n\n"
            f"Tell me everything fascinating about **{object_name}**:\n\n"
            "1. **What it is** \u2014 type, distance, physical properties\n"
            "2. **Mythology & history** \u2014 stories from different cultures\n"
            "3. **How to find it** \u2014 practical observation tips\n"
            "4. **Why it matters** \u2014 scientific significance\n"
            "5. **Fun fact** \u2014 something surprising most people don't know\n\n"
            "Be engaging, accurate, and inspiring. Write for a curious beginner."
        )
        if catalog_context:
            prompt = f"## Reference Data\n\n{catalog_context}\n\n{prompt}"

        stream = self.client.chat(
            model=self.default_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert astronomer who makes"
                        " the cosmos accessible and exciting."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": _opts.temperature_explain,
                "num_ctx": _opts.num_ctx_large,
            },
            stream=True,
        )
        for chunk in stream:
            yield chunk.message.content

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

        response = self.client.chat(
            model=self.default_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an experienced stargazing guide."
                        " Be precise with times, directions,"
                        " and practical advice."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": _opts.temperature_plan,
                "num_ctx": _opts.num_ctx_large,
            },
        )
        return response.message.content or ""

    def narrate_sky(self, sky_context: str) -> str:
        """Generate an engaging narrative about tonight's sky.

        A short, poetic yet scientific description perfect for sharing.
        """
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

        response = self.client.chat(
            model=self.default_model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": _opts.temperature_narrate,
                "num_ctx": _opts.num_ctx_default,
            },
        )
        result = response.message.content or ""
        cache.put("narrate", sky_context, value=result, ttl=_ttl_llm)
        return result

    def chat(
        self, user_message: str, sky_context: str, history: list[dict] | None = None
    ) -> str:
        """Interactive sky chat — Gemma answers questions using real sky data.

        The sky context (computed ephemeris) is injected as system knowledge
        so Gemma reasons from real data, not hallucination.
        """
        system = (
            "You are StarLens, an expert AI astronomer companion. You have access to "
            "real-time astronomical data computed from NASA/JPL ephemeris and the Hipparcos "
            "star catalog. Use this data to answer questions accurately.\n\n"
            "When the user asks about something in the sky, reference the real computed "
            "positions below. Be conversational, enthusiastic, and precise.\n\n"
            "If asked 'what's that bright thing in the east?', look at the data for objects "
            "in the east with high altitude or low magnitude (bright).\n\n"
            f"## Current Sky Data\n\n{sky_context}"
        )

        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        logger.info(
            "LLM chat: %d messages, model=%s", len(messages), self.default_model
        )
        response = self.client.chat(
            model=self.default_model,
            messages=messages,
            options={
                "temperature": _opts.temperature_chat,
                "num_ctx": _opts.num_ctx_chat,
            },
        )
        return response.message.content or ""

    def guided_tour(self, sky_context: str, step: int = 0, total_steps: int = 5) -> str:
        """Generate one step of a guided sky tour.

        Each step tells the observer exactly where to look and what they'll see.
        """
        prompt = (
            f"You are a stargazing guide leading a live sky tour. This is step {step + 1} "
            f"of {total_steps}.\n\n"
            f"## Tonight's Sky Data\n{sky_context}\n\n"
            "Generate ONLY this one tour stop. Include:\n"
            "- **Direction to face** (cardinal direction)\n"
            "- **Where to look** (altitude in degrees — 'halfway up' or 'near the horizon')\n"
            "- **What you'll see** and why it's interesting\n"
            "- **A surprising fact** about this object\n"
            "- **Transition** — a teaser for the next stop\n\n"
            f"{'Start with the most impressive object visible right now.' if step == 0 else ''}"
            f"{'This is the final stop — end with something inspiring.'
               if step == total_steps - 1 else ''}\n"
            f"Step {step + 1}: pick the "
            f"{['most impressive', 'second most interesting',
               'a hidden gem', 'something unexpected',
               'the grand finale'][min(step, 4)]}"
            " object."
        )

        response = self.client.chat(
            model=self.default_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an enthusiastic, knowledgeable"
                        " stargazing guide. Be specific with"
                        " directions and altitudes. Write as if"
                        " you're standing next to the person."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": _opts.temperature_tour,
                "num_ctx": _opts.num_ctx_default,
            },
        )
        return response.message.content or ""

    def guided_tour_stream(self, sky_context: str, step: int = 0, total_steps: int = 5):
        """Streaming version of guided_tour."""
        prompt = (
            f"You are a stargazing guide leading a live sky tour. This is step {step + 1} "
            f"of {total_steps}.\n\n"
            f"## Tonight's Sky Data\n{sky_context}\n\n"
            "Generate ONLY this one tour stop. Include:\n"
            "- **Direction to face** (cardinal direction)\n"
            "- **Where to look** (altitude in degrees \u2014 'halfway up' or 'near the horizon')\n"
            "- **What you'll see** and why it's interesting\n"
            "- **A surprising fact** about this object\n"
            "- **Transition** \u2014 a teaser for the next stop\n\n"
            f"{'Start with the most impressive object visible right now.' if step == 0 else ''}"
            f"{'This is the final stop \u2014 end with something inspiring.'
               if step == total_steps - 1 else ''}\n"
            f"Step {step + 1}: pick the "
            f"{['most impressive', 'second most interesting',
               'a hidden gem', 'something unexpected',
               'the grand finale'][min(step, 4)]}"
            " object."
        )

        stream = self.client.chat(
            model=self.default_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an enthusiastic, knowledgeable"
                        " stargazing guide. Be specific with"
                        " directions and altitudes. Write as if"
                        " you're standing next to the person."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": _opts.temperature_tour,
                "num_ctx": _opts.num_ctx_default,
            },
            stream=True,
        )
        for chunk in stream:
            yield chunk.message.content

    def analyze_chart(self, chart_image_b64: str, sky_context: str) -> str:
        """Multimodal round-trip: Gemma analyzes a sky chart IT could have generated.

        Proves Gemma can both consume and reason about astronomical visualizations.
        """
        prompt = (
            "You are looking at a sky chart rendered from real ephemeris data. "
            "Analyze this chart and provide:\n\n"
            "1. **What's most striking** — the standout objects or patterns\n"
            "2. **Constellation highlights** — which constellations are well-placed\n"
            "3. **Planet positions** — identify the orange dots (planets)\n"
            "4. **Best targets** — what should an observer focus on first\n"
            "5. **Hidden treasures** — deep-sky objects near visible constellations\n\n"
            "Cross-reference with this computed data:\n"
            f"{sky_context}\n\n"
            "Be specific and observational — describe what you SEE in the chart."
        )

        response = self.client.chat(
            model=self.default_model,
            messages=[{"role": "user", "content": prompt, "images": [chart_image_b64]}],
            options={
                "temperature": _opts.temperature_chart,
                "num_ctx": _opts.num_ctx_default,
            },
        )
        return response.message.content or ""

    def analyze_chart_stream(self, chart_image_b64: str, sky_context: str):
        """Streaming version of analyze_chart — yields chunks as they arrive."""
        prompt = (
            "You are looking at a sky chart rendered from real ephemeris data. "
            "Analyze this chart and provide:\n\n"
            "1. **What's most striking** — the standout objects or patterns\n"
            "2. **Constellation highlights** — which constellations are well-placed\n"
            "3. **Planet positions** — identify the orange dots (planets)\n"
            "4. **Best targets** — what should an observer focus on first\n"
            "5. **Hidden treasures** — deep-sky objects near visible constellations\n\n"
            "Cross-reference with this computed data:\n"
            f"{sky_context}\n\n"
            "Be specific and observational — describe what you SEE in the chart."
        )

        stream = self.client.chat(
            model=self.default_model,
            messages=[{"role": "user", "content": prompt, "images": [chart_image_b64]}],
            options={
                "temperature": _opts.temperature_chart,
                "num_ctx": _opts.num_ctx_default,
            },
            stream=True,
        )
        for chunk in stream:
            yield chunk.message.content

    def explain_why(self, object_name: str, object_data: str, sky_context: str) -> str:
        """Explain WHY an object is where it is — orbital mechanics, seasons, geometry."""
        prompt = (
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

        response = self.client.chat(
            model=self.default_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an astronomer who makes orbital"
                        " mechanics and celestial geometry"
                        " intuitive and exciting."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": _opts.temperature_why,
                "num_ctx": _opts.num_ctx_chat,
            },
        )
        return response.message.content or ""

    def explain_why_stream(self, object_name: str, object_data: str, sky_context: str):
        """Streaming version of explain_why — yields chunks as they arrive."""
        prompt = (
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

        stream = self.client.chat(
            model=self.default_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an astronomer who makes orbital"
                        " mechanics and celestial geometry"
                        " intuitive and exciting."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": _opts.temperature_why,
                "num_ctx": _opts.num_ctx_chat,
            },
            stream=True,
        )
        for chunk in stream:
            yield chunk.message.content

    def compare_skies(self, sky_now: str, sky_later: str, time_diff: str) -> str:
        """Compare two sky states and narrate what changes."""
        prompt = (
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

        response = self.client.chat(
            model=self.default_model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": _opts.temperature_tour,
                "num_ctx": _opts.num_ctx_chat,
            },
        )
        return response.message.content or ""

    def compare_skies_stream(self, sky_now: str, sky_later: str, time_diff: str):
        """Streaming version of compare_skies — yields chunks as they arrive."""
        prompt = (
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

        logger.info("LLM compare_skies_stream: model=%s", self.default_model)
        stream = self.client.chat(
            model=self.default_model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": _opts.temperature_tour,
                "num_ctx": _opts.num_ctx_chat,
            },
            stream=True,
        )
        for chunk in stream:
            yield chunk.message.content
