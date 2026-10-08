"""Agent-facing tools for native demand-driven navigation; no evaluator files exposed."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18861)
    parser.add_argument("--bridge-port", type=int, default=None)
    parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    parser.add_argument("--views", choices=("front", "four"), default="front")
    args = parser.parse_args()
    from supernav.backends.ai2thor.storage import configure_storage
    configure_storage(args.storage_root, "mcp")
    if args.views == "four":
        from supernav.methods.demand_driven.fourview_mcp import main as fourview_main
        return fourview_main(["--port", str(args.port), "--bridge-port", str(args.bridge_port or args.port),
                              "--transport", args.transport])
    from mcp.server.fastmcp import FastMCP, Image

    mcp = FastMCP("native-demand-navigation", host="127.0.0.1", port=args.port)

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, payload=None):
        body = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(f"http://127.0.0.1:{args.bridge_port or args.port}{path}", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with opener.open(req, timeout=300) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise ValueError(json.loads(exc.read()).get("error", "Native bridge rejected request")) from None
        if "images" in result:
            raise ValueError("Four-view bridge requires MCP --views four")
        encoded = result.pop("rgb_jpeg_base64", None)
        content = [json.dumps(result, ensure_ascii=False)]
        if encoded:
            content.append(Image(data=base64.b64decode(encoded), format="jpeg"))
        return content

    @mcp.tool()
    def ddn_observe():
        """Read the demand instruction and current front RGB image. No privileged metadata."""
        return request("/observation")

    @mcp.tool()
    def ddn_step(action: str):
        """Execute one forward/backward/left/right/look_up/look_down primitive, then observe."""
        result = request("/step", {"action": action})
        return result + request("/observation")

    @mcp.tool()
    def ddn_local_navigate(observation_id: str, x: float, y: float, max_replans: int = 4):
        """Use NoMaD to move toward a normalized pixel in the current RGB; a local stop is NOT task success."""
        return request("/navigate", dict(observation_id=observation_id, pixel=[x, y], max_replans=max_replans))

    @mcp.tool()
    def ddn_claim_resource(observation_id: str, description: str, x: float, y: float):
        """Record your own arrival claim and pixel. Scoring and correctness feedback are withheld."""
        return request("/claim", dict(observation_id=observation_id, description=description, pixel=[x, y]))

    @mcp.tool()
    def ddn_stop():
        """Explicitly terminate navigation. This records STOP but does not assign success."""
        return request("/stop", {})

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
