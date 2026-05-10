"""Sky chart renderer — generates beautiful sky maps using matplotlib.

Produces dark-themed star charts with constellation lines, planet markers,
and labels. Used both in the Gradio UI and for Gemma 4 multimodal
verification (render a chart → feed it back to Gemma to confirm identification).
"""

import io
import logging
from datetime import datetime, timezone

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from skyfield.api import Star  # type: ignore[import-untyped]
from skyfield.projections import build_stereographic_projection  # type: ignore[import-untyped]

from .catalog import NAMED_STARS, SkyCatalog

logger = logging.getLogger(__name__)

# Color palette
COLORS = {
    "background": "#0a0e1a",
    "grid": "#1a2040",
    "constellation_lines": "#2a4080",
    "star": "#ffffff",
    "bright_star": "#ffffcc",
    "planet": "#ff6644",
    "moon": "#ffffaa",
    "text": "#8899cc",
    "highlight": "#44aaff",
}


def render_sky_chart(
    catalog: SkyCatalog,
    lat: float,
    lon: float,
    when: datetime | None = None,
    fov: float = 180.0,
    max_mag: float = 5.5,
    highlight_objects: list[str] | None = None,
) -> bytes:
    """Render a sky chart as PNG bytes.

    Args:
        catalog: SkyCatalog instance with loaded data.
        lat: Observer latitude.
        lon: Observer longitude.
        when: Observation time (UTC). Defaults to now.
        fov: Field of view in degrees.
        max_mag: Maximum magnitude to plot.
        highlight_objects: Names of objects to highlight.

    Returns:
        PNG image as bytes.
    """
    if when is None:
        when = datetime.now(timezone.utc)

    logger.info("Rendering chart: (%.2f, %.2f) at %s, fov=%.0f", lat, lon, when, fov)
    t = catalog.ts.from_datetime(when)
    observer = catalog.get_observer(lat, lon)

    # Center projection on zenith
    zenith = observer.at(t).from_altaz(alt_degrees=90, az_degrees=0)
    projection = build_stereographic_projection(zenith)

    # Project all stars into a temporary copy (avoid mutating shared catalog)
    star_positions = observer.at(t).observe(Star.from_dataframe(catalog.stars))
    stars_df = catalog.stars.copy()
    stars_df["x"], stars_df["y"] = projection(star_positions)

    # Filter by magnitude
    bright_mask = stars_df.magnitude <= max_mag
    magnitude = stars_df["magnitude"][bright_mask]
    marker_size = (0.5 + max_mag - magnitude) ** 2.0

    # Constellation lines
    edges = [edge for _, edges in catalog.constellations for edge in edges]
    edges_s1 = [s1 for s1, _ in edges]

    xy1 = (
        stars_df[["x", "y"]]
        .loc[[s for s in edges_s1 if s in stars_df.index]]
        .values
    )
    xy2 = (
        stars_df[["x", "y"]]
        .loc[[s for s in [s2 for _, s2 in edges] if s in stars_df.index]]
        .values
    )

    min_len = min(len(xy1), len(xy2))
    xy1, xy2 = xy1[:min_len], xy2[:min_len]
    lines_xy = np.rollaxis(np.array([xy1, xy2]), 1)

    # Set up the figure
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 10), facecolor=COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    # Draw constellation lines
    ax.add_collection(
        LineCollection(
            list(lines_xy),
            colors=COLORS["constellation_lines"],
            linewidths=0.8,
            alpha=0.6,
        )
    )

    # Draw stars
    ax.scatter(
        stars_df["x"][bright_mask],
        stars_df["y"][bright_mask],
        s=marker_size,
        color=COLORS["star"],
        alpha=0.8,
        zorder=2,
    )

    # Label bright named stars
    highlight_set = set(highlight_objects or [])
    for hip_id, name in NAMED_STARS.items():
        if hip_id not in stars_df.index:
            continue
        row = stars_df.loc[hip_id]
        if float(row["magnitude"]) > 2.5:  # type: ignore[arg-type]
            continue

        color = COLORS["highlight"] if name in highlight_set else COLORS["text"]
        fontsize = 9 if name in highlight_set else 7
        ax.annotate(
            name,
            (float(row["x"]), float(row["y"])),  # type: ignore[arg-type]
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=fontsize,
            color=color,
            alpha=0.9,
        )

    # Draw planets
    for planet_key in [
        "mercury",
        "venus",
        "mars",
        "jupiter barycenter",
        "saturn barycenter",
    ]:
        try:
            planet = catalog.eph[planet_key]
            astrometric = observer.at(t).observe(planet)
            x, y = projection(astrometric)
            name = planet_key.replace(" barycenter", "").capitalize()

            ax.scatter(x, y, s=80, color=COLORS["planet"], marker="o", zorder=4)
            ax.annotate(
                name,
                (x, y),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=9,
                color=COLORS["planet"],
                fontweight="bold",
            )
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    # Draw Moon
    try:
        moon_astrometric = observer.at(t).observe(catalog.moon)
        mx, my = projection(moon_astrometric)
        ax.scatter(mx, my, s=200, color=COLORS["moon"], marker="o", zorder=4, alpha=0.9)
        ax.annotate(
            "Moon",
            (mx, my),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=10,
            color=COLORS["moon"],
            fontweight="bold",
        )
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    # Limits and styling
    angle = np.deg2rad(fov / 2.0)
    limit = np.sin(angle) / (1.0 - np.cos(angle))
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.axis("off")

    # Cardinal directions
    for label, x, y in [
        ("N", 0, limit * 0.95),
        ("S", 0, -limit * 0.95),
        ("E", -limit * 0.95, 0),
        ("W", limit * 0.95, 0),
    ]:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=14,
            color=COLORS["text"],
            fontweight="bold",
            alpha=0.7,
        )

    # Title
    time_str = when.strftime("%Y-%m-%d %H:%M UTC")
    ax.set_title(
        f"StarLens — {time_str} · {lat:.1f}°, {lon:.1f}°",
        color=COLORS["text"],
        fontsize=12,
        pad=15,
    )

    # Render to bytes
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=120, bbox_inches="tight", facecolor=COLORS["background"]
    )
    plt.close(fig)
    buf.seek(0)
    return buf.read()
