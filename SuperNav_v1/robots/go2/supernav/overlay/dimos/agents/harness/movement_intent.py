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

"""User-message movement authorization helpers.

The MCP server owns enforcement. This module only decides whether a DimOS agent
turn should ask the server to grant a bounded movement authority for the user's
current task.
"""

from __future__ import annotations

from dimos.agents.runtime_policy import MovementMode, is_task_authority_mode

_MOVEMENT_CONFIRM_MARKERS = (
    "confirm",
    "confirmed",
    "go ahead",
    "yes, move",
    "yes move",
    "proceed",
    "execute",
    "确认",
    "可以执行",
    "可以移动",
    "开始执行",
    "开始移动",
    "执行吧",
    "动吧",
)
_MOVEMENT_TASK_MARKERS = (
    "approach",
    "come here",
    "explore",
    "follow",
    "go to",
    "lie down",
    "move",
    "navigate",
    "patrol",
    "rotate",
    "sit",
    "stand up",
    "turn",
    "walk",
    "前进",
    "后退",
    "跟随",
    "靠近",
    "开始探索",
    "开始巡逻",
    "去",
    "向前",
    "向后",
    "坐下",
    "导航",
    "左转",
    "巡逻",
    "探索",
    "旋转",
    "移动",
    "站起来",
    "趴下",
    "转",
    "过来",
    "走",
)
_VISUAL_SEARCH_TASK_MARKERS = (
    "find ",
    "look for ",
    "search for ",
    "locate ",
    "找到",
    "找一下",
    "寻找",
    "找找",
    "找",
    "搜寻",
)
_INFO_QUERY_SEARCH_MARKERS = (
    "find out",
    "look up",
    "search the web",
    "status",
    "battery",
    "log",
    "docs",
    "documentation",
    "查找",
    "查询",
    "状态",
    "电量",
    "日志",
    "文档",
    "代码",
    "资料",
)
_NEGATED_MOVEMENT_MARKERS = (
    "do not move",
    "do not navigate",
    "do not walk",
    "don't move",
    "don't navigate",
    "don't walk",
    "dont move",
    "dont navigate",
    "dont walk",
    "no movement",
    "without moving",
    "不要动",
    "不要导航",
    "不要移动",
    "不要走",
    "别动",
    "别导航",
    "别移动",
    "别走",
)


def text_confirms_movement(text: str) -> bool:
    """Return whether *text* explicitly confirms a proposed movement."""

    normalized = _normalize_text(text)
    return any(marker in normalized for marker in _MOVEMENT_CONFIRM_MARKERS)


def text_requests_movement_task(text: str) -> bool:
    """Return whether *text* appears to request a robot movement task."""

    normalized = _normalize_text(text)
    if any(marker in normalized for marker in _NEGATED_MOVEMENT_MARKERS):
        return False
    return any(marker in normalized for marker in _MOVEMENT_TASK_MARKERS) or (
        _requests_visual_search_task(normalized)
    )


def movement_authorization_reason(text: str, movement_mode: MovementMode) -> str | None:
    """Return why *text* authorizes movement for this turn, if it does.

    `confirm` mode still requires explicit confirmation. Task-authority modes
    additionally treat explicit movement/exploration/navigation instructions as
    the user's bounded task authorization.
    """

    if text_confirms_movement(text):
        return "explicit_confirmation"
    if is_task_authority_mode(movement_mode) and text_requests_movement_task(text):
        return "task_request"
    return None


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _requests_visual_search_task(normalized: str) -> bool:
    if any(marker in normalized for marker in _INFO_QUERY_SEARCH_MARKERS):
        return False
    return any(marker in normalized for marker in _VISUAL_SEARCH_TASK_MARKERS)
