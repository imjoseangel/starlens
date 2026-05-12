"""StarLens engine — orchestrates catalog + Gemma 4 for all features.

This is the glue layer: it feeds real astronomical data to Gemma 4 and
combines Gemma's intelligence with precise ephemeris computations.
"""

import base64
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .catalog import SkyCatalog
from .chart import render_sky_chart
from .gemma import GemmaClient
from .settings import settings

logger = logging.getLogger(__name__)

_cfg = settings.gemini


class StarLensEngine:
    """Main engine combining sky computation with Gemma 4 intelligence."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        api_key: str | None = None,
        model: str | None = None,
        *,
        catalog: SkyCatalog | None = None,
    ):
        self.catalog = catalog if catalog is not None else SkyCatalog(data_dir=data_dir)
        self.gemma = GemmaClient(api_key=api_key or _cfg.api_key, model=model)

    def identify_photo(
        self, image_path: str, lat: float | None = None, lon: float | None = None
    ) -> dict:
        """Identify celestial objects in a sky photo using Gemma 4 multimodal.

        Gemma 4 analyzes the photo, then we cross-reference with the real
        ephemeris to validate and enrich the identification.
        """
        logger.info("Identifying photo: lat=%s, lon=%s", lat, lon)
        location = f"{lat:.2f}°, {lon:.2f}°" if lat is not None and lon is not None else ""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        gemma_result = self.gemma.identify_sky(
            image_path, location=location, timestamp=timestamp
        )

        validation = {}
        if lat is not None and lon is not None:
            sky = self.catalog.whats_up(lat, lon)
            validation = self._validate_identification(gemma_result, sky)

        return {
            "gemma_identification": gemma_result,
            "ephemeris_validation": validation,
            "timestamp": timestamp,
        }

    def whats_up_tonight(
        self, lat: float, lon: float, when: datetime | None = None
    ) -> dict:
        """Get everything visible tonight with Gemma 4 narration.

        Combines precise ephemeris data with Gemma 4's ability to tell
        the story of tonight's sky.
        """
        if when is None:
            when = datetime.now(timezone.utc)

        logger.info("Computing tonight's sky: (%.2f, %.2f) at %s", lat, lon, when)
        sky = self.catalog.whats_up(lat, lon, when=when)

        sky_summary = json.dumps(sky, indent=2, default=str)
        narration = self.gemma.narrate_sky(sky_summary)

        sky["narration"] = narration
        return sky

    def explain(self, object_name: str) -> str:
        """Get a Gemma 4 deep-dive explanation of any celestial object.

        Feeds the star catalog as context via the 256K window so Gemma 4
        can reference precise data while writing engaging explanations.
        """
        logger.info("Deep dive on: %s", object_name)
        catalog_context = self.catalog.get_star_catalog_summary(max_mag=4.0)
        return self.gemma.explain_object(object_name, catalog_context=catalog_context)

    def plan_session(self, lat: float, lon: float, preferences: str = "") -> str:
        """Create an observation plan for tonight using Gemma 4 reasoning.

        Gemma 4 receives the full list of visible objects and plans the
        optimal viewing order with times and directions.
        """
        logger.info(
            "Planning session: (%.2f, %.2f), prefs=%s",
            lat,
            lon,
            preferences[:50] if preferences else "",
        )
        sky = self.catalog.whats_up(lat, lon)
        visible_summary = self._format_visible_objects(sky)
        location = f"{lat:.2f}°N, {lon:.2f}°E"

        return self.gemma.plan_observation(visible_summary, location, preferences)

    def render_chart(
        self,
        lat: float,
        lon: float,
        when: datetime | None = None,
        highlight: list[str] | None = None,
    ) -> bytes:
        """Render a sky chart for the given location and time."""
        return render_sky_chart(
            self.catalog,
            lat=lat,
            lon=lon,
            when=when,
            highlight_objects=highlight,
        )

    def chat(
        self,
        user_message: str,
        lat: float,
        lon: float,
        history: list[dict] | None = None,
    ) -> str:
        """Interactive sky chat — Gemma answers questions grounded in real sky data."""
        sky = self.catalog.whats_up(lat, lon)
        sky_context = self._format_visible_objects(sky)
        logger.debug("Chat with %d history messages", len(history) if history else 0)
        return self.gemma.chat(user_message, sky_context, history=history)

    def guided_tour_step(
        self, lat: float, lon: float, step: int = 0, total_steps: int = 5
    ) -> str:
        """Generate one step of a Gemma-guided sky tour."""
        sky = self.catalog.whats_up(lat, lon)
        sky_context = self._format_visible_objects(sky)
        return self.gemma.guided_tour(sky_context, step=step, total_steps=total_steps)

    def guided_tour_step_stream(
        self, lat: float, lon: float, step: int = 0, total_steps: int = 5
    ):
        """Streaming version of guided_tour_step."""
        sky = self.catalog.whats_up(lat, lon)
        sky_context = self._format_visible_objects(sky)
        yield from self.gemma.guided_tour_stream(
            sky_context, step=step, total_steps=total_steps
        )

    def explain_stream(self, object_name: str):
        """Streaming version of explain."""
        catalog_context = self.catalog.get_star_catalog_summary(max_mag=4.0)
        yield from self.gemma.explain_object_stream(
            object_name, catalog_context=catalog_context
        )

    def analyze_sky_chart(self, lat: float, lon: float) -> dict:
        """Multimodal round-trip: render chart → Gemma analyzes it.

        Demonstrates Gemma 4's vision capability on astronomical data.
        """
        sky = self.catalog.whats_up(lat, lon)
        sky_context = self._format_visible_objects(sky)

        chart_bytes = self.render_chart(lat, lon)
        chart_b64 = base64.b64encode(chart_bytes).decode("utf-8")

        analysis = self.gemma.analyze_chart(chart_b64, sky_context)

        return {
            "chart": chart_bytes,
            "analysis": analysis,
        }

    def analyze_sky_chart_stream(self, lat: float, lon: float):
        """Streaming version — yields text chunks as Gemma analyzes the chart."""
        sky = self.catalog.whats_up(lat, lon)
        sky_context = self._format_visible_objects(sky)
        chart_bytes = self.render_chart(lat, lon)
        chart_b64 = base64.b64encode(chart_bytes).decode("utf-8")
        yield from self.gemma.analyze_chart_stream(chart_b64, sky_context)

    def explain_why(self, object_name: str, lat: float, lon: float) -> str:
        """Explain WHY an object is where it is — orbital mechanics and geometry."""
        sky = self.catalog.whats_up(lat, lon)
        sky_context = self._format_visible_objects(sky)
        object_data = self._find_object_data(object_name, sky)
        return self.gemma.explain_why(object_name, object_data, sky_context)

    def explain_why_stream(self, object_name: str, lat: float, lon: float):
        """Streaming version — yields text chunks as Gemma generates them."""
        sky = self.catalog.whats_up(lat, lon)
        sky_context = self._format_visible_objects(sky)
        object_data = self._find_object_data(object_name, sky)
        yield from self.gemma.explain_why_stream(object_name, object_data, sky_context)

    def compare_skies(self, lat: float, lon: float, hours_ahead: float = 3.0) -> dict:
        """Compare current sky with a future sky state."""
        logger.info("Comparing skies: (%.2f, %.2f) +%.1fh", lat, lon, hours_ahead)
        now = datetime.now(timezone.utc)
        later = now + timedelta(hours=hours_ahead)

        sky_now = self.catalog.whats_up(lat, lon, when=now)
        sky_later = self.catalog.whats_up(lat, lon, when=later)

        text_now = self._format_visible_objects(sky_now)
        text_later = self._format_visible_objects(sky_later)

        time_diff = f"{hours_ahead:.0f} hours"
        narration = self.gemma.compare_skies(text_now, text_later, time_diff)

        return {
            "sky_now": sky_now,
            "sky_later": sky_later,
            "narration": narration,
            "time_diff": time_diff,
        }

    def compare_skies_stream(self, lat: float, lon: float, hours_ahead: float = 3.0):
        """Streaming compare — yields (sky_now, sky_later, time_diff, chunk).

        First yield contains the sky data with chunk="". Subsequent yields
        stream the LLM narration text.
        """
        logger.info(
            "Comparing skies (stream): (%.2f, %.2f) +%.1fh", lat, lon, hours_ahead
        )
        now = datetime.now(timezone.utc)
        later = now + timedelta(hours=hours_ahead)

        sky_now = self.catalog.whats_up(lat, lon, when=now)
        sky_later = self.catalog.whats_up(lat, lon, when=later)

        text_now = self._format_visible_objects(sky_now)
        text_later = self._format_visible_objects(sky_later)

        time_diff = f"{hours_ahead:.0f} hours"

        yield sky_now, sky_later, time_diff, ""

        for chunk in self.gemma.compare_skies_stream(text_now, text_later, time_diff):
            yield sky_now, sky_later, time_diff, chunk

    def get_sky_context_text(self, lat: float, lon: float) -> str:
        """Get formatted sky context for external use (e.g. chat)."""
        sky = self.catalog.whats_up(lat, lon)
        return self._format_visible_objects(sky)

    def _find_object_data(self, object_name: str, sky: dict) -> str:
        """Find a named object in sky data and return its JSON representation."""
        name_lower = object_name.lower()
        for p in sky.get("planets", []):
            if p["name"].lower() == name_lower:
                return json.dumps(p)
        for s in sky.get("bright_stars", []):
            if s["name"].lower() == name_lower:
                return json.dumps(s)
        moon = sky.get("moon", {})
        if name_lower == "moon" and moon.get("visible"):
            return json.dumps(moon)
        return ""

    def _validate_identification(self, gemma_result: dict, ephemeris: dict) -> dict:
        """Cross-reference Gemma 4's identification with ephemeris truth."""
        validation: dict[str, list[str]] = {
            "confirmed": [],
            "unconfirmed": [],
            "missed": [],
        }

        real_planets = {p["name"].lower() for p in ephemeris.get("planets", [])}
        gemma_planets = {
            p.get("name", "").lower() for p in gemma_result.get("planets", [])
        }

        for p in gemma_planets & real_planets:
            validation["confirmed"].append(f"Planet {p.capitalize()}")
        for p in gemma_planets - real_planets:
            validation["unconfirmed"].append(
                f"Planet {p.capitalize()} (not in ephemeris)"
            )
        for p in real_planets - gemma_planets:
            validation["missed"].append(
                f"Planet {p.capitalize()} (visible but not identified)"
            )

        real_constellations = {
            c["name"].lower() for c in ephemeris.get("constellations", [])
        }
        gemma_constellations = {
            c.get("name", "").lower() for c in gemma_result.get("constellations", [])
        }

        for c in gemma_constellations & real_constellations:
            validation["confirmed"].append(f"Constellation {c.capitalize()}")

        return validation

    def _format_visible_objects(self, sky: dict) -> str:
        """Format sky data as text for Gemma 4 consumption."""
        lines = []

        sun = sky.get("sun", {})
        lines.append(
            f"Sky condition: {sun.get('condition', 'unknown')} (sun at {sun.get('altitude', '?')}°)"
        )

        moon = sky.get("moon", {})
        if moon.get("visible"):
            lines.append(
                f"Moon: {moon['phase_name']} ({moon['phase_percent']}% illuminated), "
                f"alt {moon['altitude']}° {moon['direction']}"
            )

        planets = sky.get("planets", [])
        if planets:
            lines.append(f"\nPlanets ({len(planets)}):")
            for p in planets:
                mag = f", mag {p['magnitude']}" if p.get("magnitude") else ""
                lines.append(
                    f"  - {p['name']}: alt {p['altitude']}° {p['direction']}{mag}"
                )

        stars = sky.get("bright_stars", [])
        if stars:
            lines.append(f"\nBright stars ({len(stars)}):")
            for s in stars[:20]:
                lines.append(
                    f"  - {s['name']}: mag {s['magnitude']}, alt {s['altitude']}° {s['direction']}"
                )

        constellations = sky.get("constellations", [])
        if constellations:
            lines.append(f"\nConstellations ({len(constellations)}):")
            for c in constellations[:15]:
                lines.append(f"  - {c['name']}: {c['completeness']}% visible")

        return "\n".join(lines)
