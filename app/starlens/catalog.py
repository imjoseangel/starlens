"""Sky catalog — star data, constellations, and ephemeris computations.

Uses Skyfield + Hipparcos + JPL ephemeris to compute real-time positions
of stars, planets, the Moon, and constellations for any location and time.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from skyfield.api import Loader, Star, wgs84  # type: ignore[import-untyped]
from skyfield.data import hipparcos, stellarium  # type: ignore[import-untyped]
from skyfield.magnitudelib import planetary_magnitude  # type: ignore[import-untyped]

from . import cache

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

# Constellation names for the 88 IAU constellations (common subset)
CONSTELLATION_NAMES = {
    "And": "Andromeda",
    "Ant": "Antlia",
    "Aps": "Apus",
    "Aqr": "Aquarius",
    "Aql": "Aquila",
    "Ara": "Ara",
    "Ari": "Aries",
    "Aur": "Auriga",
    "Boo": "Boötes",
    "Cae": "Caelum",
    "Cam": "Camelopardalis",
    "Cnc": "Cancer",
    "CVn": "Canes Venatici",
    "CMa": "Canis Major",
    "CMi": "Canis Minor",
    "Cap": "Capricornus",
    "Car": "Carina",
    "Cas": "Cassiopeia",
    "Cen": "Centaurus",
    "Cep": "Cepheus",
    "Cet": "Cetus",
    "Cha": "Chamaeleon",
    "Cir": "Circinus",
    "Col": "Columba",
    "Com": "Coma Berenices",
    "CrA": "Corona Australis",
    "CrB": "Corona Borealis",
    "Crv": "Corvus",
    "Crt": "Crater",
    "Cru": "Crux",
    "Cyg": "Cygnus",
    "Del": "Delphinus",
    "Dor": "Dorado",
    "Dra": "Draco",
    "Equ": "Equuleus",
    "Eri": "Eridanus",
    "For": "Fornax",
    "Gem": "Gemini",
    "Gru": "Grus",
    "Her": "Hercules",
    "Hor": "Horologium",
    "Hya": "Hydra",
    "Hyi": "Hydrus",
    "Ind": "Indus",
    "Lac": "Lacerta",
    "Leo": "Leo",
    "LMi": "Leo Minor",
    "Lep": "Lepus",
    "Lib": "Libra",
    "Lup": "Lupus",
    "Lyn": "Lynx",
    "Lyr": "Lyra",
    "Men": "Mensa",
    "Mic": "Microscopium",
    "Mon": "Monoceros",
    "Mus": "Musca",
    "Nor": "Norma",
    "Oct": "Octans",
    "Oph": "Ophiuchus",
    "Ori": "Orion",
    "Pav": "Pavo",
    "Peg": "Pegasus",
    "Per": "Perseus",
    "Phe": "Phoenix",
    "Pic": "Pictor",
    "Psc": "Pisces",
    "PsA": "Piscis Austrinus",
    "Pup": "Puppis",
    "Pyx": "Pyxis",
    "Ret": "Reticulum",
    "Sge": "Sagitta",
    "Sgr": "Sagittarius",
    "Sco": "Scorpius",
    "Scl": "Sculptor",
    "Sct": "Scutum",
    "Ser": "Serpens",
    "Sex": "Sextans",
    "Tau": "Taurus",
    "Tel": "Telescopium",
    "Tri": "Triangulum",
    "TrA": "Triangulum Australe",
    "Tuc": "Tucana",
    "UMa": "Ursa Major",
    "UMi": "Ursa Minor",
    "Vel": "Vela",
    "Vir": "Virgo",
    "Vol": "Volans",
    "Vul": "Vulpecula",
}

# Named bright stars (Hipparcos IDs → common names)
NAMED_STARS = {
    32349: "Sirius",
    30438: "Canopus",
    69673: "Arcturus",
    71683: "Alpha Centauri A",
    91262: "Vega",
    24436: "Rigel",
    37279: "Procyon",
    24608: "Betelgeuse",
    7588: "Achernar",
    68702: "Hadar",
    97649: "Altair",
    60718: "Acrux",
    21421: "Aldebaran",
    65474: "Spica",
    80763: "Antares",
    37826: "Pollux",
    113368: "Fomalhaut",
    62434: "Mimosa",
    102098: "Deneb",
    84012: "Shaula",
    27989: "Bellatrix",
    26311: "Elnath",
    25336: "Mira",
    25930: "Alnilam",
    25428: "Mintaka",
    26727: "Alnitak",
    11767: "Polaris",
    677: "Alpheratz",
    15863: "Mirfak",
    9884: "Mirach",
    5447: "Schedar",
    746: "Caph",
    3419: "Almaak",
    54061: "Regulus",
    49669: "Denebola",
    57632: "Mizar",
    62956: "Alkaid",
}

PLANETS = ["mercury", "venus", "mars", "jupiter barycenter", "saturn barycenter"]
PLANET_NAMES = {
    "mercury": "Mercury",
    "venus": "Venus",
    "mars": "Mars",
    "jupiter barycenter": "Jupiter",
    "saturn barycenter": "Saturn",
}


class SkyCatalog:
    """Real-time sky computation engine using Skyfield."""

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Loading sky catalog from %s", self.data_dir)

        self.load = Loader(str(self.data_dir))
        self.ts = self.load.timescale()

        # Load ephemeris
        self.eph = self.load("de421.bsp")
        self.earth = self.eph["earth"]
        self.sun = self.eph["sun"]
        self.moon = self.eph["moon"]

        # Load star catalog
        hip_path = self.data_dir / "hip_main.dat"
        if hip_path.exists():
            with open(hip_path, "rb") as f:
                self.stars = hipparcos.load_dataframe(f)
        else:
            with self.load.open(hipparcos.URL) as f:
                self.stars = hipparcos.load_dataframe(f)

        # Load constellations
        constellation_path = self.data_dir / "constellationship.fab"
        if constellation_path.exists():
            with open(constellation_path, "rb") as f:
                self.constellations = stellarium.parse_constellations(f)
        else:
            with self.load.open("constellationship.fab") as f:
                self.constellations = stellarium.parse_constellations(f)

    def get_observer(self, lat: float, lon: float, elevation: float = 0):
        """Create an observer at a given location."""
        return self.earth + wgs84.latlon(lat, lon, elevation_m=elevation)

    def whats_up(
        self,
        lat: float,
        lon: float,
        when: datetime | None = None,
        min_altitude: float = 10.0,
    ) -> dict:
        """Compute what's visible in the sky right now from a location.

        This is the core function — it computes positions for stars, planets,
        the Moon, and constellations, returning everything Gemma 4 needs
        for reasoning and planning.
        """
        if when is None:
            when = datetime.now(timezone.utc)

        # Round to nearest 5 min for cache hits
        rounded_min = (when.minute // 5) * 5
        cache_time = when.replace(minute=rounded_min, second=0, microsecond=0)
        cache_key_parts = (
            round(lat, 2),
            round(lon, 2),
            cache_time.isoformat(),
            min_altitude,
        )

        cached = cache.get("sky", *cache_key_parts)
        if cached is not None:
            logger.debug("Cache hit for sky data at (%.2f, %.2f)", lat, lon)
            return cached

        logger.info("Computing sky: (%.2f, %.2f) at %s", lat, lon, when)

        t = self.ts.from_datetime(when)
        observer = self.get_observer(lat, lon)

        result = {
            "time_utc": when.isoformat(),
            "location": {"latitude": lat, "longitude": lon},
            "planets": self._compute_planets(observer, t, min_altitude),
            "moon": self._compute_moon(observer, t),
            "bright_stars": self._compute_bright_stars(observer, t, min_altitude),
            "constellations": self._compute_visible_constellations(
                observer, t, min_altitude
            ),
            "sun": self._compute_sun(observer, t),
        }

        from .settings import settings  # pylint: disable=import-outside-toplevel

        cache.put("sky", *cache_key_parts, value=result, ttl=settings.redis.ttl_sky)

        logger.debug(
            "Sky result: %d planets, %d stars, %d constellations",
            len(result["planets"]),
            len(result["bright_stars"]),
            len(result["constellations"]),
        )
        return result

    def _compute_planets(self, observer, t, min_alt: float) -> list[dict]:
        """Compute visible planet positions."""
        planets = []
        for planet_key in PLANETS:
            try:
                planet = self.eph[planet_key]
                astrometric = observer.at(t).observe(planet)
                apparent = astrometric.apparent()
                alt, az, _ = apparent.altaz()

                if alt.degrees > min_alt:
                    # Try to get magnitude
                    try:
                        mag = planetary_magnitude(astrometric)
                    except Exception:  # pylint: disable=broad-exception-caught
                        mag = None

                    planets.append(
                        {
                            "name": PLANET_NAMES[planet_key],
                            "altitude": round(float(alt.degrees), 1),
                            "azimuth": round(float(az.degrees), 1),
                            "direction": self._az_to_direction(float(az.degrees)),
                            "magnitude": (
                                round(float(mag), 1) if mag is not None else None
                            ),
                        }
                    )
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.debug("Could not compute %s: %s", planet_key, e)

        return planets

    def _compute_moon(self, observer, t) -> dict:
        """Compute Moon position and phase."""
        astrometric = observer.at(t).observe(self.moon)
        apparent = astrometric.apparent()
        alt, az, _ = apparent.altaz()

        # Moon phase (elongation from sun)
        sun_astrometric = observer.at(t).observe(self.sun)
        moon_apparent = astrometric.apparent()
        sun_apparent = sun_astrometric.apparent()

        # Phase angle approximation
        e = moon_apparent.separation_from(sun_apparent)
        elongation = e.degrees
        phase_pct = round((1 - np.cos(np.radians(elongation))) / 2 * 100, 1)

        phase_name = self._phase_name(phase_pct)

        return {
            "altitude": round(float(alt.degrees), 1),
            "azimuth": round(float(az.degrees), 1),
            "direction": self._az_to_direction(float(az.degrees)),
            "visible": float(alt.degrees) > 0,
            "phase_percent": round(float(phase_pct), 1),
            "phase_name": phase_name,
            "elongation": round(float(elongation), 1),
        }

    def _compute_bright_stars(
        self, observer, t, min_alt: float, max_mag: float = 3.0
    ) -> list[dict]:
        """Compute positions of bright named stars."""
        bright = []

        for hip_id, name in NAMED_STARS.items():
            if hip_id not in self.stars.index:
                continue

            row = self.stars.loc[hip_id]
            mag_val: float = float(row["magnitude"])  # type: ignore[arg-type]
            if mag_val > max_mag:
                continue

            star = Star.from_dataframe(self.stars.loc[[hip_id]])
            astrometric = observer.at(t).observe(star)
            apparent = astrometric.apparent()
            alt, az, _ = apparent.altaz()

            alt_deg = float(np.atleast_1d(alt.degrees)[0])
            az_deg = float(np.atleast_1d(az.degrees)[0])

            if alt_deg > min_alt:
                bright.append(
                    {
                        "name": name,
                        "hip_id": hip_id,
                        "altitude": round(alt_deg, 1),
                        "azimuth": round(az_deg, 1),
                        "direction": self._az_to_direction(az_deg),
                        "magnitude": round(mag_val, 2),
                    }
                )

        bright.sort(key=lambda x: float(x["magnitude"]))  # type: ignore[arg-type]
        return bright

    def _compute_visible_constellations(
        self, observer, t, min_alt: float
    ) -> list[dict]:
        """Determine which constellations are visible."""
        visible = []

        for name, edges in self.constellations:
            star_ids = set()
            for s1, s2 in edges:
                star_ids.add(s1)
                star_ids.add(s2)

            # Check if at least half the constellation stars are above min_alt
            above = 0
            total = 0
            for sid in star_ids:
                if sid not in self.stars.index:
                    continue
                total += 1
                star = Star.from_dataframe(self.stars.loc[[sid]])
                astrometric = observer.at(t).observe(star)
                alt, _, _ = astrometric.apparent().altaz()
                if float(np.atleast_1d(alt.degrees)[0]) > min_alt:
                    above += 1

            if total > 0 and above / total >= 0.5:
                full_name = CONSTELLATION_NAMES.get(name, name)
                visible.append(
                    {
                        "abbreviation": name,
                        "name": full_name,
                        "stars_visible": above,
                        "stars_total": total,
                        "completeness": round(above / total * 100, 0),
                    }
                )

        visible.sort(key=lambda x: x["completeness"], reverse=True)
        return visible

    def _compute_sun(self, observer, t) -> dict:
        """Compute sun position for twilight info."""
        astrometric = observer.at(t).observe(self.sun)
        apparent = astrometric.apparent()
        alt, _az, _ = apparent.altaz()

        sun_alt = float(alt.degrees)
        if sun_alt > 0:
            condition = "daylight"
        elif sun_alt > -6:
            condition = "civil_twilight"
        elif sun_alt > -12:
            condition = "nautical_twilight"
        elif sun_alt > -18:
            condition = "astronomical_twilight"
        else:
            condition = "night"

        return {
            "altitude": round(sun_alt, 1),
            "condition": condition,
            "is_dark_enough": sun_alt < -12,
        }

    def get_star_catalog_summary(self, max_mag: float = 4.0) -> str:
        """Build a text summary of the star catalog for Gemma 4's 128K context."""
        bright = self.stars[self.stars.magnitude <= max_mag].sort_values("magnitude")

        lines = [
            f"# Star Catalog — {len(bright)} stars brighter than magnitude {max_mag}\n"
        ]
        for hip_id, row in bright.iterrows():
            name = NAMED_STARS.get(int(hip_id), f"HIP {hip_id}")  # type: ignore[arg-type]
            lines.append(
                f"- {name}: mag {row['magnitude']:.2f}, "
                f"RA {row['ra_hours']:.4f}h, Dec {row['dec_degrees']:.2f}°"
            )

        lines.append(f"\n# Constellations — {len(self.constellations)} loaded\n")
        for name, edges in self.constellations:
            full_name = CONSTELLATION_NAMES.get(name, name)
            lines.append(f"- {full_name} ({name}): {len(edges)} line segments")

        return "\n".join(lines)

    @staticmethod
    def _az_to_direction(az: float) -> str:
        """Convert azimuth to cardinal direction."""
        dirs = [
            "N",
            "NNE",
            "NE",
            "ENE",
            "E",
            "ESE",
            "SE",
            "SSE",
            "S",
            "SSW",
            "SW",
            "WSW",
            "W",
            "WNW",
            "NW",
            "NNW",
        ]
        idx = round(az / 22.5) % 16
        return dirs[idx]

    @staticmethod
    def _phase_name(pct: float) -> str:
        """Convert illumination percentage to phase name."""
        if pct < 3:
            return "New Moon"
        elif pct < 25:
            return "Waxing Crescent"
        elif pct < 50:
            return "First Quarter"
        elif pct < 75:
            return "Waxing Gibbous"
        elif pct < 97:
            return "Full Moon" if pct > 90 else "Waning Gibbous"
        else:
            return "Full Moon"
