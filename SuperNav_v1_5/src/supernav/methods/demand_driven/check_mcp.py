"""Exercise the native MCP observation/action/claim/STOP lifecycle without a brain model."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from supernav.backends.ai2thor.storage import configure_storage
from supernav.evaluation.demand_driven.dataset import write_json


async def check(args):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    configure_storage(args.storage_root, "mcp_check")
    four_view = args.views == "four"
    view_names = ("front", "right", "back", "left")
    image_count = 4 if four_view else 1
    params = StdioServerParameters(command=str(args.python),
             args=["-B", "-m", "supernav.methods.demand_driven.mcp", "--storage-root", str(args.storage_root), "--port", str(args.port), "--views", args.views],
             env=dict(os.environ))
    results = []

    def record(name, arguments, result):
        with args.output.with_suffix(".tools.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(name=name, arguments=arguments,
                         result=result.model_dump(mode="json")), ensure_ascii=False) + "\n")

    async def call(client, name, arguments):
        result = await client.call_tool(name, arguments)
        record(name, arguments, result)
        return result

    def inspect(result):
        images = sum(c.type == "image" for c in result.content)
        labels = [c.text for c in result.content if c.type == "text" and c.text in [v + " view" for v in view_names]]
        texts = [json.loads(c.text) for c in result.content if c.type == "text" and c.text not in labels]
        if four_view and images:
            assert labels == [v + " view" for v in view_names]
        for value in texts:
            if any(k in value for k in ("objects", "stage_plan", "gt", "pose", "scene_id")):
                raise AssertionError("Evaluator metadata crossed the MCP boundary")
        return images, texts

    with (args.output.with_suffix(".stderr.log")).open("x") as errlog:
        async with stdio_client(params, errlog=errlog) as streams:
            async with ClientSession(*streams) as client:
                await client.initialize()
                listed = await client.list_tools()
                write_json(args.output.with_suffix(".schemas.json"), listed.model_dump(mode="json"))
                names = sorted(t.name for t in listed.tools)
                assert names == sorted(["ddn_observe", "ddn_step", "ddn_local_navigate", "ddn_claim_resource", "ddn_stop"])
                if four_view:
                    for tool in listed.tools:
                        if tool.name in ("ddn_claim_resource", "ddn_local_navigate"):
                            assert "view" in tool.inputSchema["required"]
                obs = await call(client, "ddn_observe", {})
                images, texts = inspect(obs)
                assert not obs.isError and images == image_count
                results.append(dict(operation="observe", image_count=images, valid=True))
                step = await call(client, "ddn_step", {"action": "left"})
                images, texts = inspect(step)
                assert not step.isError and images == image_count
                current = next(t for t in texts if "observation_id" in t and "instruction" in t)
                results.append(dict(operation="step", image_count=images, valid=True))
                claim = await call(client, "ddn_claim_resource", dict(observation_id=current["observation_id"],
                             description="Engineering-only pixel claim, not a resource success", x=.5, y=.5,
                             **({"view": "right"} if four_view else {})))
                _, texts = inspect(claim)
                assert not claim.isError and texts == [dict(recorded=True, success_scoring="withheld")]
                results.append(dict(operation="claim_unscored", valid=True))
                stop = await call(client, "ddn_stop", {})
                _, texts = inspect(stop)
                assert not stop.isError and texts[0]["stop_called"] and texts[0]["success_scoring"] == "withheld"
                action_count = texts[0]["action_count"]
                results.append(dict(operation="stop", valid=True))
                rejected = await call(client, "ddn_step", {"action": "forward"})
                if four_view:
                    _, stopped = inspect(rejected)
                    assert not rejected.isError and stopped[0]["terminal"]
                    assert stopped[0]["action_index"] == action_count
                else:
                    assert rejected.isError
                results.append(dict(operation="reject_after_stop", valid=True))
    write_json(args.output, dict(valid=True, tools=names, checks=results,
                                 observation_views=list(view_names) if four_view else ["front"],
                                 success_scoring="withheld", brain_model_called=False))
    print(json.dumps(dict(valid=True, checks=len(results), success_scoring="withheld")), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", type=int, default=18861)
    parser.add_argument("--views", choices=("front", "four"), default="front")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(check(args))


if __name__ == "__main__":
    main()
