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

"""Model-compatible messages emitted by the native harness itself."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.messages.base import BaseMessage

_NATIVE_SOURCE_KEY = "dimos/native_source"
_NATIVE_SOURCE_PREFIX = "dimos_native:"


def native_internal_message(content: Any, *, source: str) -> HumanMessage:
    """Create a model-visible message that is not external human input.

    LangChain chat models are conservative about accepted message roles, so the
    fallback agent keeps the provider-compatible human role while carrying a
    structured DimOS source marker out-of-band. Turn preparation uses the marker
    to avoid treating harness events as new user commands.
    """

    return HumanMessage(
        content=content,
        additional_kwargs={_NATIVE_SOURCE_KEY: f"{_NATIVE_SOURCE_PREFIX}{source}"},
    )


def is_native_internal_message(message: BaseMessage) -> bool:
    """Return whether a message originated from the native harness runtime."""

    marker = message.additional_kwargs.get(_NATIVE_SOURCE_KEY)
    return isinstance(marker, str) and marker.startswith(_NATIVE_SOURCE_PREFIX)
