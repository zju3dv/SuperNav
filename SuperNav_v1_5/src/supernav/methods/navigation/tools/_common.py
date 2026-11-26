"""Shared visual payload, image collection and boolean parsing helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from supernav.methods.navigation.tools.base import ToolContext

def visual_payload(ctx: ToolContext) -> Dict[str, Any]:
    """Build the output_dir block for visual bridge calls."""
    return {
        "output_dir": ctx.output_dir,
        "agent_image_max_size": 256,
    }


def collect_images(result: Dict[str, Any], ctx: ToolContext) -> None:
    """Extract captured image paths from a bridge response into RoundState.

    The bridge returns captured visuals in two possible shapes:
      - `result["visuals"]`: dict keyed by sensor name, each entry with
        `path`. Used by `step_and_capture`, `get_visuals`.
      - `result["images"]`: list of dicts with `path`
        and optional `direction`. Used by `get_panorama`.
      - `result["panorama_images"]`: list of direction-labelled
        panorama dicts. Used by MCP movement tools that automatically
        append surround-camera observations.

    Appends every extracted path to `ctx.round_state.captured_images`
    and updates `ctx.round_state.last_visual_path` using the priority
    color_sensor > front panorama > any image. The last_visual_path
    falls back to its previous value if nothing new is collected so
    the mapless auto-injection in `update_nav_status` stays stable
    across turns where no image was taken.
    """
    color_path: Optional[str] = None
    color_metadata: Optional[Dict[str, Any]] = None
    any_path: Optional[str] = None
    any_metadata: Optional[Dict[str, Any]] = None
    front_pano_path: Optional[str] = None
    front_pano_metadata: Optional[Dict[str, Any]] = None

    visuals = result.get("visuals", {})
    if isinstance(visuals, dict):
        for sensor_name, sensor_data in visuals.items():
            if isinstance(sensor_data, dict):
                mp = sensor_data.get("path")
                if mp and isinstance(mp, str):
                    ctx.round_state.captured_images.append(mp)
                    any_path = mp
                    any_metadata = dict(sensor_data)
                    if sensor_name == "color_sensor" or (
                        color_path is None and "color" in sensor_name
                    ):
                        color_path = mp
                        color_metadata = dict(sensor_data)

    images = result.get("images", [])
    panorama_images = result.get("panorama_images", [])
    image_rows = []
    if isinstance(images, list):
        image_rows.extend(images)
    if isinstance(panorama_images, list):
        image_rows.extend(panorama_images)
    if image_rows:
        for img in image_rows:
            if isinstance(img, dict):
                mp = img.get("path")
                if mp and isinstance(mp, str):
                    ctx.round_state.captured_images.append(mp)
                    any_path = mp
                    any_metadata = dict(img)
                    if img.get("direction", "") == "front":
                        front_pano_path = mp
                        front_pano_metadata = dict(img)
                    if color_path is None:
                        color_path = mp  # panorama images are always color
                        color_metadata = dict(img)

    # Priority: color_sensor > front panorama > any captured image.
    # Fall back to previous value so mapless auto-injection stays stable.
    ctx.round_state.last_visual_path = (
        color_path or front_pano_path or any_path or ctx.round_state.last_visual_path
    )
    ctx.round_state.last_visual_metadata = (
        color_metadata
        or front_pano_metadata
        or any_metadata
        or ctx.round_state.last_visual_metadata
    )


_FALSE_STRINGS = frozenset({"false", "0", "no", "n", "off", ""})
_TRUE_STRINGS = frozenset({"true", "1", "yes", "y", "on"})


def parse_bool_flag(value: Any, default: bool = False) -> bool:
    """Tolerantly parse a boolean flag from heterogeneous caller input.

    Python's built-in `bool()` has a well-known pitfall: `bool("false")`
    returns True because any non-empty string is truthy. Some LLMs emit
    JSON string booleans ("false" / "true") even when the schema
    declares a boolean parameter, so a naive `bool(args.get(...))` on
    LLM-supplied tool arguments would misinterpret their intent.

    Parsing rules (case-insensitive for strings):
      - Native `bool` → passed through
      - `int`/`float` → `bool(value)` (0 → False, non-zero → True)
      - String `"true"`/`"yes"`/`"y"`/`"on"`/`"1"` → True
      - String `"false"`/`"no"`/`"n"`/`"off"`/`"0"`/`""` → False
      - `None` or any other type → `default`

    Used by tools and MCP environment settings to parse boolean values
    without treating the string "false" as true.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _TRUE_STRINGS:
            return True
        if v in _FALSE_STRINGS:
            return False
        return default
    return default


__all__ = ["visual_payload", "collect_images", "parse_bool_flag"]
