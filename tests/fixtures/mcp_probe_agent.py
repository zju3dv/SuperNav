"""Deterministic OpenCode-shaped transport probe. No LLM or navigation policy."""
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    spec = next(iter(json.loads(Path("opencode.json").read_text())["mcp"].values()))
    parameters = StdioServerParameters(command=spec["command"][0], args=spec["command"][1:], env=spec["environment"])
    async with stdio_client(parameters) as streams:
        async with ClientSession(*streams) as client:
            await client.initialize()
            schemas = await client.list_tools()
            Path("probe.schemas.json").write_text(schemas.model_dump_json(indent=2))

            async def call(name, args):
                result = await client.call_tool(name, args)
                print(json.dumps(dict(type="tool_use", timestamp=datetime.now(timezone.utc).isoformat(),
                    part=dict(type="tool", tool="mcp__navigation__"+name,
                              state=dict(input=args, status="error" if result.isError else "completed",
                                         output=result.model_dump_json())))), flush=True)
                assert not result.isError
                return result

            observed = await call("ddn_observe", {})
            count = sum(block.type == "image" for block in observed.content)
            assert count in (1, 4)
            result = await call("ddn_step", {"action": "left"})
            texts = [block.text for block in result.content if block.type == "text" and block.text.startswith("{")]
            obs = next(json.loads(text) for text in texts if "observation_id" in text and "instruction" in text)
            await call("ddn_claim_resource", dict(observation_id=obs["observation_id"], x=.5, y=.5,
                description="Engineering probe only; no resource success claim",
                **({"view": "right"} if count == 4 else {})))
            if "--no-stop" not in sys.argv and "--fail-run" not in sys.argv:
                await call("ddn_stop", {})
            print(json.dumps(dict(type="text", part=dict(text="Engineering transport probe completed; no LLM was invoked."))), flush=True)
    return 7 if "--fail-run" in sys.argv else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
