"""
Default capability resolvers.

These ship with the framework so ``agent.requires(...)`` is demonstrable
out of the box, with zero extra dependencies and (for two of the three)
zero API keys. They're intentionally simple reference implementations,
not production-grade integrations -- the point is that application code
written against the capability name (``"weather"``, ``"internet_search"``,
``"filesystem"``) keeps working unchanged when a real plugin, an MCP
server, or a paid API registers a higher-priority provider for the same
capability later.

Register a better provider for the same capability at higher priority to
have it take over automatically:

>>> from requisite.capabilities import default_registry
>>> default_registry.register(
...     "weather", my_paid_weather_tool, provider_name="acme-weather", priority=10,
...     is_available=lambda: bool(os.environ.get("ACME_API_KEY")),
... )  # doctest: +SKIP
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from requisite.capabilities.registry import CapabilityRegistry

logger = logging.getLogger("requisite.capabilities.resolvers")

_MAX_FILE_BYTES = 200_000
_REQUEST_TIMEOUT = 10.0


def read_file(path: str) -> str:
    """Read and return the text contents of a local file.

    Parameters
    ----------
    path:
        Path to a text file on disk, relative or absolute.

    Notes
    -----
    Reference implementation of the ``"filesystem"`` capability. Reads are
    capped at 200KB to keep tool results reasonably sized for a model's
    context window; register your own provider for larger-file handling
    (e.g. chunked reads, summarization) at a higher priority.
    """
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        return f"Error: '{path}' is not a file or does not exist."
    data = file_path.read_bytes()[:_MAX_FILE_BYTES]
    return data.decode("utf-8", errors="replace")


def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Parameters
    ----------
    city:
        City name, e.g. ``"Paris"``, ``"Tokyo"``.

    Notes
    -----
    Reference implementation of the ``"weather"`` capability, backed by
    the free `Open-Meteo <https://open-meteo.com>`_ geocoding + forecast
    APIs. No API key required. Geocodes the city name, then fetches
    current conditions for the first match.
    """
    geocode_url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(
        {"name": city, "count": 1}
    )
    try:
        with urllib.request.urlopen(geocode_url, timeout=_REQUEST_TIMEOUT) as response:
            geocode_data = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return f"Error looking up location '{city}': {exc}"

    results = geocode_data.get("results") or []
    if not results:
        return f"Could not find a location matching '{city}'."

    location = results[0]
    latitude, longitude = location["latitude"], location["longitude"]
    resolved_name = location.get("name", city)

    forecast_url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(
        {"latitude": latitude, "longitude": longitude, "current_weather": "true"}
    )
    try:
        with urllib.request.urlopen(forecast_url, timeout=_REQUEST_TIMEOUT) as response:
            forecast_data = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return f"Error fetching weather for '{resolved_name}': {exc}"

    current = forecast_data.get("current_weather") or {}
    temperature = current.get("temperature")
    windspeed = current.get("windspeed")
    if temperature is None:
        return f"No current weather data available for '{resolved_name}'."
    return f"{resolved_name}: {temperature}C, wind {windspeed} km/h."


def search_web(query: str) -> str:
    """Search the web for a query and return a short summary.

    Parameters
    ----------
    query:
        The search query.

    Notes
    -----
    Reference implementation of the ``"internet_search"`` capability,
    backed by the free DuckDuckGo Instant Answer API. No API key
    required, but coverage is limited to topics DuckDuckGo has a direct
    answer/summary for -- it is not general web search. For production
    use, register a real search provider (e.g. Bing, SerpAPI, Tavily) at
    a higher priority; this default will only be used as a fallback.
    """
    search_url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
    )
    try:
        with urllib.request.urlopen(search_url, timeout=_REQUEST_TIMEOUT) as response:
            data = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return f"Error searching for '{query}': {exc}"

    abstract = data.get("AbstractText")
    if abstract:
        source = data.get("AbstractSource", "DuckDuckGo")
        return f"{abstract} (source: {source})"

    related = data.get("RelatedTopics") or []
    snippets: list[str] = [
        text
        for t in related
        if isinstance(t, dict) and isinstance(text := t.get("Text"), str) and text
    ]
    if snippets:
        return " | ".join(snippets[:3])

    return f"No quick summary found for '{query}'. Try a more specific query."


def register_default_capabilities(registry: CapabilityRegistry) -> None:
    """Register the framework's built-in capability providers on ``registry``.

    Registers, all at ``priority=0`` (the baseline -- register your own
    provider at a higher priority to take precedence automatically):

    - ``"filesystem"`` -> :func:`read_file`
    - ``"weather"`` -> :func:`get_weather`
    - ``"internet_search"`` -> :func:`search_web`
    """
    registry.register("filesystem", read_file, provider_name="local-filesystem")
    registry.register("weather", get_weather, provider_name="open-meteo")
    registry.register("internet_search", search_web, provider_name="duckduckgo-instant-answer")
