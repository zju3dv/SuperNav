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

from typing import Any


def install_message_factory_get_prototype_compat() -> bool:
    """Restore the protobuf 5-era MessageFactory.GetPrototype shim when needed.

    Some native extensions still look up ``MessageFactory().GetPrototype`` while
    protobuf 6 exposes the equivalent module-level ``GetMessageClass`` helper.
    """
    try:
        from google.protobuf import message_factory
    except ImportError:
        return False

    factory_cls = message_factory.MessageFactory
    if hasattr(factory_cls, "GetPrototype"):
        return False

    get_message_class = getattr(message_factory, "GetMessageClass", None)
    if get_message_class is None:
        return False

    def get_prototype(self: object, descriptor: Any) -> type[Any]:
        return get_message_class(descriptor)

    factory_cls.GetPrototype = get_prototype  # type: ignore[attr-defined]
    return True
