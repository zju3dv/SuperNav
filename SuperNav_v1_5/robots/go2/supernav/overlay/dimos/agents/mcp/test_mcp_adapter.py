# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from unittest.mock import Mock

import pytest

from dimos.agents.mcp.mcp_adapter import McpAdapter
from dimos.core.run_registry import RunEntry


def test_mcp_adapter_from_run_entry_uses_registered_mcp_url() -> None:
    entry = RunEntry(
        run_id="run-1",
        pid=123,
        blueprint="unitree-go2-agentic",
        started_at="2026-06-08T00:00:00Z",
        log_dir="/tmp/dimos/run-1",
        config_overrides={"mcp_port": 9998, "listen_host": "0.0.0.0"},
    )

    adapter = McpAdapter.from_run_entry(entry)

    assert adapter.url == "http://127.0.0.1:9998/mcp"


def test_mcp_adapter_from_run_entry_discovers_latest_live_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = RunEntry(
        run_id="run-latest",
        pid=456,
        blueprint="unitree-go2-agentic",
        started_at="2026-06-08T00:00:01Z",
        log_dir="/tmp/dimos/run-latest",
        config_overrides={"mcp_port": 9997},
    )
    monkeypatch.setattr(
        "dimos.core.run_registry.get_most_recent",
        lambda alive_only=True: entry,
    )

    adapter = McpAdapter.from_run_entry()

    assert adapter.url == "http://127.0.0.1:9997/mcp"


def test_mcp_adapter_call_tool_can_send_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    posted_payloads: list[dict[str, object]] = []

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"result": {"content": [{"type": "text", "text": "ok"}]}}

    def fake_post(url: str, *, json: dict[str, object], timeout: int) -> FakeResponse:
        posted_payloads.append(json)
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)
    adapter = McpAdapter("http://test/mcp")

    result = adapter.call_tool(
        "rotate_in_place",
        {"degrees": 0.0},
        meta={"dimos/external_harness": True},
    )

    assert result["content"][0]["text"] == "ok"
    params = posted_payloads[0]["params"]
    assert isinstance(params, dict)
    assert params["_meta"] == {"dimos/external_harness": True}


def test_mcp_adapter_initialize_returns_unwrapped_server_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posted_payloads: list[dict[str, object]] = []

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "jsonrpc": "2.0",
                "id": "init-1",
                "result": {"serverInfo": {"name": "fake-dimos"}},
            }

    def fake_post(url: str, *, json: dict[str, object], timeout: int) -> FakeResponse:
        posted_payloads.append(json)
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)
    adapter = McpAdapter("http://test/mcp")

    result = adapter.initialize()

    assert result == {"serverInfo": {"name": "fake-dimos"}}
    assert posted_payloads[0]["method"] == "initialize"


@pytest.mark.parametrize(
    ("arguments", "expected_timeout"),
    [
        ({"timeout_s": 180}, 480.0),
        ({"timeout_s": 60}, 240.0),
        ({}, 320.0),
        ({"timeout_s": "180"}, 480.0),
        ({"timeout_s": None}, 320.0),
        ({"timeout_s": float("nan")}, 320.0),
        ({"timeout_s": float("inf")}, 320.0),
    ],
)
def test_memory_navigation_has_request_specific_http_timeout(
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, object],
    expected_timeout: float,
) -> None:
    timeouts: list[float] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, object]:
            return {"result": {"content": []}}

    def fake_post(url: str, *, json: dict[str, object], timeout: float) -> FakeResponse:
        timeouts.append(timeout)
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)
    adapter = McpAdapter("http://test/mcp")
    adapter.call_tool("go2_navigate_to_memory_node", arguments)
    adapter.call_tool("pixelgoal_status", {"include_images": False})
    adapter.list_tools()
    assert timeouts == [expected_timeout, 30.0, 30.0]
    assert adapter.timeout == 30


def test_memory_navigation_preserves_larger_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = Mock()
    post.return_value.json.return_value = {"result": {"content": []}}
    monkeypatch.setattr("requests.post", post)
    adapter = McpAdapter("http://test/mcp", timeout=600)
    adapter.call_tool("go2_navigate_to_memory_node", {"timeout_s": 180})
    assert post.call_args.kwargs["timeout"] == 600

