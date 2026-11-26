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

"""Artifact ledger for rich tool results.

The agent brain may inspect images during the tool turn that produced them, and
later turns may explicitly re-read selected images. Durable chat history still
keeps compact artifact references by default; this ledger owns the persisted raw
blocks used for replay, UI, and MCP readback.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from dimos.agents.harness.image_blocks import (
    image_block_dimensions,
    image_block_mime_type,
    image_block_payload_bytes,
    image_block_token_estimate,
    is_image_block,
)

_ARTIFACT_ID_RE = re.compile(r"^artifact_[0-9]{6,}$")


@dataclass(frozen=True)
class ArtifactRef:
    """Compact reference to a rich artifact removed from chat history."""

    artifact_id: str
    kind: str
    mime_type: str | None
    created_at: float
    metadata: dict[str, Any]
    path: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)

    def placeholder_text(self) -> str:
        return artifact_placeholder_text(
            self.artifact_id,
            kind=self.kind,
            mime_type=self.mime_type,
            metadata=self.metadata,
        )


def artifact_placeholder_text(
    artifact_id: str,
    *,
    kind: str,
    mime_type: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Return the compact transcript placeholder for a redacted artifact."""

    mime = f" mime={mime_type}" if mime_type else ""
    summary = _artifact_placeholder_summary(kind=kind, metadata=metadata or {})
    return f"[artifact:{artifact_id} kind={kind}{mime}{summary} redacted_from_history]"


@dataclass(frozen=True)
class ImageArtifactReadback:
    """MCP-ready image artifact readback payload."""

    ref: ArtifactRef
    block: dict[str, Any]

    def agent_encode(self) -> list[dict[str, Any]]:
        metadata = {
            "artifact_id": self.ref.artifact_id,
            "kind": self.ref.kind,
            "mime_type": self.ref.mime_type,
            "created_at": self.ref.created_at,
            "metadata": self.ref.metadata,
            "path": self.ref.path,
        }
        return [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "message": "Re-injected persisted image artifact.",
                        "artifact": metadata,
                    },
                    ensure_ascii=True,
                    default=str,
                ),
            },
            self.block,
        ]

    def __str__(self) -> str:
        dimensions = ""
        width = self.ref.metadata.get("width_px")
        height = self.ref.metadata.get("height_px")
        if width is not None and height is not None:
            dimensions = f" {width}x{height}"
        mime = f" {self.ref.mime_type}" if self.ref.mime_type else ""
        return f"ImageArtifactReadback({self.ref.artifact_id}{mime}{dimensions})"


class ArtifactLedger:
    """Store rich content blocks and return compact refs for transcript use."""

    def __init__(self, artifact_dir: str | Path | None = None) -> None:
        self._artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self._ids = itertools.count(self._next_artifact_index())
        self._artifacts: dict[str, dict[str, Any]] = {}

    def record_block(
        self,
        block: dict[str, Any],
        *,
        source: str,
        message_type: str,
    ) -> ArtifactRef:
        artifact_id = f"artifact_{next(self._ids):06d}"
        kind = _block_kind(block)
        mime_type = _block_mime_type(block)
        dimensions = image_block_dimensions(block) if kind == "image" else None
        metadata: dict[str, Any] = {
            "source": source,
            "message_type": message_type,
            "block_type": str(block.get("type") or "unknown"),
        }
        if dimensions is not None:
            metadata["width_px"], metadata["height_px"] = dimensions
        if kind == "image":
            metadata["estimated_tokens"] = image_block_token_estimate(block)
            content_sha256 = _image_content_sha256(block)
            if content_sha256 is not None:
                metadata["content_sha256"] = content_sha256
                metadata["hash_algorithm"] = "sha256"
        artifact_path = self._artifact_path(artifact_id)
        ref = ArtifactRef(
            artifact_id=artifact_id,
            kind=kind,
            mime_type=mime_type,
            created_at=time.time(),
            metadata=metadata,
            path=str(artifact_path) if artifact_path is not None else None,
        )
        self._artifacts[artifact_id] = {"ref": ref, "block": block}
        self._persist_block(ref, block)
        return ref

    def get_ref(self, artifact_id: str) -> ArtifactRef | None:
        entry = self._artifacts.get(artifact_id)
        if entry is None:
            payload = self._load_persisted_payload(artifact_id)
            ref_payload = payload.get("ref") if isinstance(payload, dict) else None
            if not isinstance(ref_payload, dict):
                return None
            ref = _artifact_ref_from_payload(ref_payload)
            if ref is None or ref.artifact_id != artifact_id:
                return None
            return ref
        ref = entry.get("ref")
        return ref if isinstance(ref, ArtifactRef) else None

    def get_block(self, artifact_id: str) -> dict[str, Any] | None:
        entry = self._artifacts.get(artifact_id)
        if entry is None:
            block = self._load_persisted_block(artifact_id)
            return block if isinstance(block, dict) else None
        block = entry.get("block")
        return block if isinstance(block, dict) else None

    def read_image(self, artifact_id: str) -> ImageArtifactReadback | None:
        """Return a model-visible image block for a persisted artifact id."""

        ref = self.get_ref(artifact_id)
        block = self.get_block(artifact_id)
        if ref is None or not isinstance(block, dict) or not is_image_block(block):
            return None
        return ImageArtifactReadback(ref=ref, block=block)

    def list_refs(self, *, kind: str | None = None) -> list[ArtifactRef]:
        """Return known artifact refs from memory and persisted storage."""

        refs: dict[str, ArtifactRef] = {}
        for artifact_id, entry in self._artifacts.items():
            ref = entry.get("ref")
            if isinstance(ref, ArtifactRef) and is_valid_artifact_id(artifact_id):
                refs[artifact_id] = ref
        for ref in self._iter_persisted_refs():
            refs[ref.artifact_id] = ref
        filtered = [ref for ref in refs.values() if kind is None or ref.kind == kind]
        return sorted(filtered, key=lambda ref: (ref.created_at, ref.artifact_id))

    def latest_ref(self, *, kind: str | None = None) -> ArtifactRef | None:
        """Return the newest known artifact ref, optionally filtered by kind."""

        refs = self.list_refs(kind=kind)
        return refs[-1] if refs else None

    def latest_image_ref(self) -> ArtifactRef | None:
        """Return the newest known image artifact ref."""

        return self.latest_ref(kind="image")

    def snapshot(self) -> dict[str, Any]:
        return {
            artifact_id: entry["ref"].model_dump()
            for artifact_id, entry in sorted(self._artifacts.items())
        }

    def _artifact_path(self, artifact_id: str) -> Path | None:
        if self._artifact_dir is None or not is_valid_artifact_id(artifact_id):
            return None
        return self._artifact_dir / f"{artifact_id}.json"

    def _next_artifact_index(self) -> int:
        if self._artifact_dir is None or not self._artifact_dir.exists():
            return 1
        highest = 0
        for path in self._artifact_dir.glob("artifact_*.json"):
            suffix = path.stem.removeprefix("artifact_")
            if suffix.isdigit():
                highest = max(highest, int(suffix))
        return highest + 1

    def _persist_block(self, ref: ArtifactRef, block: dict[str, Any]) -> None:
        artifact_path = self._artifact_path(ref.artifact_id)
        if artifact_path is None:
            return
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ref": ref.model_dump(), "block": block}
        artifact_path.write_text(
            json.dumps(payload, ensure_ascii=True, default=str), encoding="utf-8"
        )
        with (artifact_path.parent / "index.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(ref.model_dump(), ensure_ascii=True, default=str) + "\n")

    def _iter_persisted_refs(self) -> Iterable[ArtifactRef]:
        if self._artifact_dir is None or not self._artifact_dir.exists():
            return []
        refs: dict[str, ArtifactRef] = {}
        index_path = self._artifact_dir / "index.jsonl"
        if index_path.exists():
            for ref in self._load_refs_from_index(index_path):
                artifact_ref = self.get_ref(ref.artifact_id)
                if artifact_ref is not None:
                    refs[artifact_ref.artifact_id] = artifact_ref
        for path in self._artifact_dir.glob("artifact_*.json"):
            artifact_id = path.stem
            if not is_valid_artifact_id(artifact_id):
                continue
            ref = self.get_ref(artifact_id)
            if ref is not None:
                refs[ref.artifact_id] = ref
        return refs.values()

    def _load_refs_from_index(self, index_path: Path) -> Iterable[ArtifactRef]:
        try:
            lines = index_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        refs: list[ArtifactRef] = []
        for line in lines:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            ref = _artifact_ref_from_payload(payload)
            if ref is not None:
                refs.append(ref)
        return refs

    def _load_persisted_block(self, artifact_id: str) -> dict[str, Any] | None:
        payload = self._load_persisted_payload(artifact_id)
        block = payload.get("block") if isinstance(payload, dict) else None
        return block if isinstance(block, dict) else None

    def _load_persisted_payload(self, artifact_id: str) -> dict[str, Any]:
        artifact_path = self._artifact_path(artifact_id)
        if artifact_path is None or not artifact_path.exists():
            return {}
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}


def is_valid_artifact_id(artifact_id: str) -> bool:
    """Return whether *artifact_id* can safely map to an artifact file."""

    return _ARTIFACT_ID_RE.fullmatch(artifact_id) is not None


def _artifact_ref_from_payload(payload: Any) -> ArtifactRef | None:
    if not isinstance(payload, dict):
        return None
    artifact_id = str(payload.get("artifact_id") or "")
    if not is_valid_artifact_id(artifact_id):
        return None
    try:
        return ArtifactRef(**payload)
    except TypeError:
        return None


def _artifact_placeholder_summary(*, kind: str, metadata: dict[str, Any]) -> str:
    if kind != "image":
        return ""
    parts = ["summary=image_frame"]
    width = _metadata_int(metadata.get("width_px"))
    height = _metadata_int(metadata.get("height_px"))
    if width is not None and height is not None:
        parts.append(f"dimensions={width}x{height}")
    estimated_tokens = _metadata_int(metadata.get("estimated_tokens"))
    if estimated_tokens is not None:
        parts.append(f"estimated_image_tokens={estimated_tokens}")
    content_sha256 = _metadata_token(metadata.get("content_sha256"))
    if content_sha256:
        parts.append(f"sha256={content_sha256[:12]}")
    source = _metadata_token(metadata.get("source"))
    if source:
        parts.append(f"source={source}")
    return " " + " ".join(parts)


def _metadata_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _metadata_token(value: Any) -> str:
    if value is None:
        return ""
    token = str(value).strip()
    if not token:
        return ""
    return re.sub(r"[^a-zA-Z0-9_.:-]+", "_", token)[:80]


def default_artifact_dir() -> Path | None:
    """Return the current run's default artifact directory, if available."""

    from dimos.utils.logging_config import get_run_log_dir

    run_log_dir = get_run_log_dir()
    if run_log_dir is not None:
        return run_log_dir / "artifacts"
    env_log_dir = os.environ.get("DIMOS_RUN_LOG_DIR")
    if env_log_dir:
        return Path(env_log_dir) / "artifacts"
    return None


def resolve_artifact_dir(
    *,
    artifact_dir: str | Path | None = None,
    runtime_config: Any | None = None,
) -> Path | None:
    """Resolve explicit config, runtime config, or the current run artifact dir."""

    configured = artifact_dir
    if configured is None and runtime_config is not None:
        configured = getattr(runtime_config, "agent_artifact_dir", None)
    if configured:
        return Path(configured)
    return default_artifact_dir()


def _block_kind(block: dict[str, Any]) -> str:
    block_type = str(block.get("type") or "")
    if is_image_block(block):
        return "image"
    return block_type or "artifact"


def _block_mime_type(block: dict[str, Any]) -> str | None:
    mime_type = image_block_mime_type(block)
    if mime_type is not None:
        return mime_type
    mime = block.get("mime_type") or block.get("mime")
    return str(mime) if mime else None


def _image_content_sha256(block: dict[str, Any]) -> str | None:
    payload = image_block_payload_bytes(block)
    if payload is None:
        return None
    return hashlib.sha256(payload).hexdigest()
