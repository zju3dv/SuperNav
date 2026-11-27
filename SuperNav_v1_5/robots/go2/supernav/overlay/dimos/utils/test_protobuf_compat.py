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

from google.protobuf import descriptor_pb2, message_factory

from dimos.utils.protobuf_compat import install_message_factory_get_prototype_compat


def test_message_factory_get_prototype_compat_installs_on_protobuf_6() -> None:
    original = getattr(message_factory.MessageFactory, "GetPrototype", None)
    if original is not None:
        delattr(message_factory.MessageFactory, "GetPrototype")

    try:
        installed = install_message_factory_get_prototype_compat()

        assert installed is True
        descriptor = descriptor_pb2.FileDescriptorProto.DESCRIPTOR
        assert message_factory.MessageFactory().GetPrototype(descriptor) is (  # type: ignore[attr-defined]
            message_factory.GetMessageClass(descriptor)
        )
    finally:
        if original is not None:
            message_factory.MessageFactory.GetPrototype = original  # type: ignore[attr-defined]


def test_message_factory_get_prototype_compat_is_idempotent() -> None:
    install_message_factory_get_prototype_compat()

    assert install_message_factory_get_prototype_compat() is False
