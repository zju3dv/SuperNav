"""Habitat session lifecycle integration for memory-node SLAM."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from supernav.backends.habitat.memory_slam.area_captioning import (
    SlamAreaCaptionWorker,
    enrich_slam_area_nodes_with_captions,
)
from supernav.backends.habitat.memory_slam.area_segmentation import (
    analyze_slam_occupancy,
    render_slam_segmentation_overlay,
)
from supernav.backends.habitat.memory_slam.cartographer import (
    _write_reported_memory_node_prior,
    dual_stream_from_environment,
    enabled_from_environment,
    odometry_from_habitat_state,
    scan_from_depth,
    scan_from_projected_depth_floor_band,
    stream_from_environment,
    timestamp_for_step,
)
from supernav.backends.habitat.memory_slam.occupancy_navigation import (
    build_navigation_costmap,
    compile_slam_occupancy_route,
    load_slam_occupancy_map,
    plan_slam_occupancy_path,
    render_slam_occupancy_plan_overlay,
    snap_to_traversable,
)
from supernav.backends.habitat.simulator.types import HabitatAdapterError, _Session


class HabitatMemorySlamMixin:
    """Optional mapping lifecycle; disabled unless explicitly configured."""

    def _configure_memory_slam_session(
        self, session: _Session, payload: Mapping[str, Any]
    ) -> None:
        config = payload.get("memory_slam", {})
        if config is None:
            config = {}
        if not isinstance(config, Mapping):
            raise HabitatAdapterError('Field "payload.memory_slam" must be an object')
        aliases = {
            "enabled": "cartographer_enabled",
            "output_dir": "cartographer_output_dir",
            "worker_command": "cartographer_worker_command",
            "use_odometry": "cartographer_use_odometry",
            "odometry_source": "cartographer_odometry_source",
            "odometry_mode": "cartographer_odometry_mode",
            "memory_nodes_path": "cartographer_memory_nodes_path",
            "relocalizer_command": "cartographer_relocalizer_command",
            "relocalization_enabled": "cartographer_relocalization_enabled",
            "agent_guided_relocalization_enabled": "cartographer_agent_guided_relocalization_enabled",
            "return_occupancy_image": "cartographer_return_occupancy_image",
            "caption_enabled": "slam_area_caption_enabled",
            "dual_stream": "cartographer_dual_stream",
            "planning_scan_source": "cartographer_occ_scan_source",
        }
        for public_name, setting_name in aliases.items():
            if public_name in config:
                session.settings[setting_name] = config[public_name]
        localization = config.get("localization")
        if isinstance(localization, Mapping):
            session.settings["cartographer_localization"] = dict(localization)
        caption_sources = config.get("caption_sources")
        if isinstance(caption_sources, list):
            session.settings["slam_area_caption_sources"] = [
                dict(source)
                for source in caption_sources
                if isinstance(source, Mapping)
                and isinstance(source.get("artifact_dir"), str)
                and source["artifact_dir"]
            ]

    def _start_memory_slam(self, session: _Session) -> None:
        enabled = bool(session.settings.get("cartographer_enabled", enabled_from_environment()))
        if not enabled:
            return
        if "depth_sensor" not in (session.last_sensor_obs or {}):
            raise HabitatAdapterError("spatial memory requires the Habitat depth sensor")
        source = str(session.settings.get("cartographer_odometry_source", "simulator_evaluator"))
        if source != "simulator_evaluator":
            raise HabitatAdapterError(
                "public spatial memory supports only simulator_evaluator odometry"
            )
        session.settings.setdefault("cartographer_use_odometry", True)
        session.settings.setdefault("cartographer_odometry_mode", "authoritative")
        planning_enabled = bool(session.settings.get("cartographer_dual_stream", True))
        if planning_enabled:
            session.cartographer_stream = dual_stream_from_environment(
                session.session_id, session.settings
            )
            session.cartographer_occ_stream = session.cartographer_stream.planning_proxy
        else:
            session.cartographer_stream = stream_from_environment(
                session.session_id, session.settings
            )
        caption_enabled = bool(session.settings.get("slam_area_caption_enabled", False))
        if caption_enabled:
            if not os.environ.get("HAB_SLAM_AREA_CAPTION_URL", "").strip():
                raise HabitatAdapterError(
                    "caption-enabled spatial memory requires HAB_SLAM_AREA_CAPTION_URL"
                )
            artifact_dir = Path(session.cartographer_stream.artifact_dir)
            session.slam_area_caption_worker = SlamAreaCaptionWorker.from_environment(
                session_id=session.session_id,
                artifact_dir=artifact_dir,
                logger=self._logger,
            )

    def _submit_memory_slam_scan(
        self,
        session: _Session,
        observation: Mapping[str, Any],
        dt_s: float,
        *,
        publish_snapshot: bool = True,
    ) -> None:
        stream = session.cartographer_stream
        if stream is None or "depth_sensor" not in observation:
            return
        try:
            capture_surround = (
                os.environ.get("HAB_CARTOGRAPHER_SURROUND_SCAN", "0").strip()
                == "1"
            )
            depth_views: list[Any] = [observation["depth_sensor"]]
            if capture_surround:
                # Match the research implementation: capture right, back, and
                # left without advancing simulation time, then restore the
                # authoritative agent state before submitting the scan.
                import habitat_sim

                agent = session.simulator.get_agent(session.agent_id)
                saved_state = agent.get_state()
                try:
                    base_w, base_x, base_y, base_z = self._current_rotation(session)
                    for turn_right_deg in (90.0, 180.0, 270.0):
                        half_angle = -math.radians(turn_right_deg) / 2.0
                        delta_y = math.sin(half_angle)
                        delta_w = math.cos(half_angle)
                        view_state = habitat_sim.AgentState()
                        view_state.position = saved_state.position
                        view_state.rotation = [
                            base_x * delta_w - base_z * delta_y,
                            base_w * delta_y + base_y * delta_w,
                            base_x * delta_y + base_z * delta_w,
                            base_w * delta_w - base_y * delta_y,
                        ]
                        agent.set_state(view_state, infer_sensor_states=True)
                        capture = self._capture_sensor_observations(session)
                        depth_views.append(capture["depth_sensor"])
                finally:
                    agent.set_state(saved_state, infer_sensor_states=True)

            sequence = session.cartographer_sequence + 1
            timestamp_s = max(
                timestamp_for_step(session.step_count, dt_s),
                (session.cartographer_last_timestamp_s or 0.0) + 1e-6,
            )

            def _convert(source_stream: Any) -> Any:
                converter = (
                    scan_from_projected_depth_floor_band
                    if getattr(source_stream, "scan_source", "center_band") == "projected_depth"
                    else scan_from_depth
                )
                kwargs: dict[str, Any] = {}
                if converter is scan_from_projected_depth_floor_band:
                    kwargs = {
                        "sensor_height_m": source_stream.sensor_height_m,
                        "row_start_fraction": source_stream.project_row_start_fraction,
                        "pixel_stride": source_stream.project_pixel_stride,
                        "min_obstacle_height_m": source_stream.min_obstacle_height_m,
                        "max_obstacle_height_m": source_stream.max_obstacle_height_m,
                    }
                def _convert_view(depth: Any, beam_count: int) -> Any:
                    return converter(
                        depth,
                        sequence=sequence,
                        timestamp_s=timestamp_s,
                        beam_count=beam_count,
                        range_min_m=source_stream.range_min_m,
                        range_max_m=source_stream.range_max_m,
                        hfov_deg=source_stream.hfov_deg,
                        **kwargs,
                    )

                scan = _convert_view(depth_views[0], source_stream.beam_count)
                if len(depth_views) > 1:
                    scans = [
                        _convert_view(
                            depth,
                            max(2, source_stream.beam_count // 4),
                        )
                        for depth in depth_views
                    ]
                    # Cartographer X is forward and Y is left. Preserve the
                    # source angular order: right, front, left, back.
                    ordered_scans = (scans[1], scans[0], scans[3], scans[2])
                    scan = dataclasses.replace(
                        scan,
                        ranges_m=[
                            value
                            for view in ordered_scans
                            for value in view.ranges_m
                        ],
                        hfov_deg=360.0,
                        camera_hfov_deg=source_stream.hfov_deg,
                    )
                return scan

            scan = _convert(stream)
            planning_scan = _convert(session.cartographer_occ_stream) if session.cartographer_occ_stream is not None else None
            scan = dataclasses.replace(scan, publish_snapshot=publish_snapshot)
            if planning_scan is not None:
                planning_scan = dataclasses.replace(
                    planning_scan, publish_snapshot=publish_snapshot
                )
            if stream.use_odometry:
                state = session.simulator.get_agent(session.agent_id).get_state()
                scan = dataclasses.replace(
                    scan,
                    odometry=odometry_from_habitat_state(
                        state.position, self._current_rotation(session)
                    ),
                )
                if planning_scan is not None:
                    planning_scan = dataclasses.replace(
                        planning_scan, odometry=scan.odometry
                    )
            stream.submit(scan)
            if planning_scan is not None:
                session.cartographer_occ_stream.submit(planning_scan)
            session.cartographer_sequence = scan.sequence
            session.cartographer_last_timestamp_s = scan.timestamp_s
            self._maybe_enqueue_caption_keyframe(session, observation)
        except Exception as exc:  # Mapping cannot invalidate a completed action.
            self._logger.warning(
                "spatial-memory scan failed session=%s step=%s: %s",
                session.session_id, session.step_count, exc,
            )

    def _maybe_enqueue_caption_keyframe(
        self, session: _Session, observation: Mapping[str, Any]
    ) -> None:
        worker = session.slam_area_caption_worker
        if worker is None or "color_sensor" not in observation:
            return
        try:
            cadence = max(1, int(os.environ.get("HAB_SLAM_AREA_CAPTION_CADENCE_STEPS", "8")))
        except ValueError:
            cadence = 8
        if session.slam_area_caption_last_step >= 0 and (
            session.step_count - session.slam_area_caption_last_step
        ) < cadence:
            return
        image = np.asarray(observation["color_sensor"])
        if image.ndim != 3 or image.shape[2] < 3:
            return
        image_dir = Path(worker.artifact_dir) / "caption_keyframes"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"step{session.step_count:06d}_front.png"
        Image.fromarray(image[:, :, :3].astype(np.uint8), mode="RGB").save(image_path)
        worker.enqueue_keyframe({
            "keyframe_id": f"{session.session_id}:{session.step_count}",
            "session_id": session.session_id,
            "step_count": session.step_count,
            "cartographer_sequence": session.cartographer_sequence,
            "images": [{
                "direction": "front", "heading_deg": 0.0, "path": str(image_path)
            }],
        })
        session.slam_area_caption_last_step = session.step_count

    def _close_memory_slam(self, session: _Session, reason: str) -> None:
        worker = session.slam_area_caption_worker
        session.slam_area_caption_worker = None
        if worker is not None:
            try:
                worker.close(timeout_s=2.0)
            except Exception as exc:
                self._logger.warning("caption worker close failed (%s): %s", reason, exc)
        session.cartographer_occ_stream = None
        stream = session.cartographer_stream
        session.cartographer_stream = None
        if stream is not None:
            try:
                stream.close(timeout_s=self._env_float("HAB_CARTOGRAPHER_DRAIN_TIMEOUT_S", 30.0))
            except Exception as exc:
                self._logger.warning("Cartographer close failed (%s): %s", reason, exc)

    def _memory_slam_status(
        self, session: _Session, *, wait_for_pose: bool = False
    ) -> Dict[str, Any]:
        stream = session.cartographer_stream
        if stream is None:
            return {"enabled": False, "slam_pose": None}
        status = stream.status()
        artifact_dir_value = status.get("artifact_dir") or getattr(
            stream, "artifact_dir", None
        )
        artifact_dir = (
            Path(artifact_dir_value)
            if isinstance(artifact_dir_value, (str, os.PathLike))
            else None
        )
        wait_timeout_s = 0.0
        if wait_for_pose:
            try:
                wait_timeout_s = max(
                    0.0,
                    float(
                        os.environ.get(
                            "HAB_CARTOGRAPHER_POSE_WAIT_TIMEOUT_S", "5.0"
                        )
                    ),
                )
            except ValueError:
                wait_timeout_s = 5.0
        status["slam_pose"] = (
            self._latest_scan_matched_pose(artifact_dir, wait_timeout_s)
            if artifact_dir is not None
            else None
        )
        status["slam_areas"] = (
            self._read_slam_area_nodes(artifact_dir)
            if artifact_dir is not None
            else []
        )
        return status

    @staticmethod
    def _latest_scan_matched_pose(
        artifact_dir: Path, wait_timeout_s: float = 0.0
    ) -> dict[str, Any] | None:
        """Read the Cartographer map-frame pose used by the original system."""
        pose_path = artifact_dir / "occupancy.png.poses.jsonl"
        deadline = time.monotonic() + max(0.0, wait_timeout_s)
        while True:
            try:
                lines = pose_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                lines = []
            for line in reversed(lines):
                try:
                    pose = json.loads(line)
                    sequence = pose["sequence"]
                    if not isinstance(sequence, int) or isinstance(sequence, bool):
                        continue
                    return {
                        "sequence": sequence,
                        "x": float(pose["x"]),
                        "y": float(pose["y"]),
                        "yaw_rad": float(pose["yaw_rad"]),
                    }
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        try:
            metadata = json.loads(
                (artifact_dir / "metadata.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return None
        localization = metadata.get("localization") if isinstance(metadata, dict) else None
        if isinstance(localization, dict) and localization.get("mode") == "mapping":
            return {"sequence": 1, "x": 0.0, "y": 0.0, "yaw_rad": 0.0}
        return None

    @staticmethod
    def _read_slam_area_nodes(artifact_dir: Path) -> list[dict[str, Any]]:
        try:
            payload = json.loads(
                (artifact_dir / "slam_area_nodes.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return []
        nodes = payload.get("nodes") if isinstance(payload, Mapping) else None
        return [dict(node) for node in nodes or [] if isinstance(node, Mapping)]

    def _attach_memory_slam_map_context(
        self, session: _Session, response: Dict[str, Any], output_dir: str
    ) -> None:
        """Attach the latest immutable, agent-visible annotated SLAM map."""
        if not bool(session.settings.get("cartographer_return_occupancy_image")):
            return
        stream = session.cartographer_stream
        if stream is None:
            return
        requested = int(
            getattr(stream, "_latest_odometry_sequence", None)
            or session.cartographer_sequence
            or session.step_count
        )
        try:
            timeout_s = max(
                0.0, float(os.environ.get("HAB_CARTOGRAPHER_RENDER_TIMEOUT_S", "0.5"))
            )
        except ValueError:
            timeout_s = 0.5
        stream.wait_for_rendered_sequence(requested, timeout_s)
        artifact_dir = Path(stream.artifact_dir)
        map_artifact_dir = artifact_dir
        prior_static = False
        source = artifact_dir / f"occupancy_step{requested}.png"
        if not source.is_file():
            try:
                for line in reversed(
                    (artifact_dir / "occupancy.png.updates.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ):
                    sequence = int(json.loads(line).get("sequence", -1))
                    candidate = artifact_dir / f"occupancy_step{sequence}.png"
                    if sequence >= 0 and candidate.is_file() and candidate.stat().st_size:
                        source = candidate
                        break
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        if not source.is_file() or not source.stat().st_size:
            prior_dir_value = os.environ.get(
                "HAB_CARTOGRAPHER_PRIOR_VISUAL_ARTIFACT_DIR", ""
            ).strip()
            if prior_dir_value:
                prior_dir = Path(prior_dir_value)
                candidate = prior_dir / "occupancy_step0.png"
                if candidate.is_file() and candidate.stat().st_size:
                    map_artifact_dir = prior_dir
                    source = candidate
                    prior_static = True
        if not source.is_file() or not source.stat().st_size:
            response["2d_slam_map"] = {
                "available": False,
                "status": "pending",
                "step_count": requested,
                "description": "2D SLAM map is pending its first rendered geometry snapshot.",
            }
            return
        try:
            snapshot_sequence = int(source.stem.removeprefix("occupancy_step"))
        except ValueError:
            snapshot_sequence = requested
        try:
            status = self._memory_slam_status(session)
        except Exception:
            status = stream.status()
        memory_nodes = self._memory_nodes_for_map(map_artifact_dir, source)
        slam_area_nodes: list[dict[str, Any]] = []
        destination_root = Path(output_dir) if output_dir else artifact_dir
        destination_root.mkdir(parents=True, exist_ok=True)
        destination = destination_root / f"slam_2d_step{session.step_count}_occupancy.png"
        temporary = destination.with_suffix(".tmp.png")
        segmentation_overlay = destination_root / f".agent_segmentation_{snapshot_sequence}.png"
        segmentation: dict[str, Any] = {"enabled": False, "source": None}
        pose_projected = False
        try:
            segmentation_source = source.with_suffix(".amcl.pgm")
            if segmentation_source.is_file():
                segmentation_result, class_grid, area_grid = analyze_slam_occupancy(
                    segmentation_source
                )
                render_slam_segmentation_overlay(
                    segmentation_source, segmentation_overlay
                )
                rendered = self._segmentation_display_image(
                    segmentation_overlay, map_artifact_dir, source
                )
                if prior_static:
                    # The pair snapshot is immutable.  Its semantic areas were
                    # finalized during exploration, so read them without
                    # publishing sidecars back into the snapshot.
                    slam_area_nodes = self._read_slam_area_nodes(map_artifact_dir)
                else:
                    planning_artifact_dir = None
                    if session.cartographer_occ_stream is not None:
                        candidate = getattr(
                            session.cartographer_occ_stream, "artifact_dir", None
                        )
                        if isinstance(candidate, (str, os.PathLike)):
                            planning_artifact_dir = Path(candidate)
                    slam_area_nodes, slam_area_payload = self._publish_slam_area_nodes(
                        artifact_dir=map_artifact_dir,
                        source=source,
                        segmentation_source=segmentation_source,
                        segmentation_result=segmentation_result,
                        memory_nodes=memory_nodes,
                        planning_artifact_dir=planning_artifact_dir,
                    )
                    if session.settings.get("slam_area_caption_enabled") is not False:
                        slam_area_nodes = enrich_slam_area_nodes_with_captions(
                            artifact_dir=map_artifact_dir,
                            inherited_sources=session.settings.get(
                                "slam_area_caption_sources", []
                            ),
                            segmentation_source=segmentation_source,
                            segmentation_result=segmentation_result,
                            class_grid=class_grid,
                            area_grid=area_grid,
                            nodes=slam_area_nodes,
                        )
                    slam_area_payload["nodes"] = slam_area_nodes
                    self._write_slam_area_nodes_sidecar(
                        map_artifact_dir, slam_area_payload
                    )
                segmentation = {
                    "enabled": True,
                    "source": "raw_amcl_pgm",
                    "area_count": len(segmentation_result.get("areas", [])),
                    "slam_area_node_count": len(slam_area_nodes),
                }
            else:
                with Image.open(source) as image:
                    rendered = image.convert("RGB")
            memory_nodes = self._map_nodes_in_rendered_crop(
                map_artifact_dir, source, memory_nodes, "memory"
            )
            slam_area_nodes = self._map_nodes_in_rendered_crop(
                map_artifact_dir, source, slam_area_nodes, "area"
            )
            self._draw_map_labels(rendered, map_artifact_dir, source, memory_nodes, "memory")
            self._draw_map_labels(rendered, map_artifact_dir, source, slam_area_nodes, "area")
            if not prior_static:
                pose_projected = self._draw_current_pose_marker(
                    rendered, map_artifact_dir, source, status
                )
            rendered.resize((256, 256), Image.Resampling.LANCZOS).save(
                temporary, format="PNG", optimize=True
            )
            temporary.replace(destination)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            temporary.unlink(missing_ok=True)
            response["2d_slam_map"] = {
                "available": False,
                "status": "pending",
                "step_count": snapshot_sequence,
                "description": f"2D SLAM map annotation is pending: {type(exc).__name__}.",
            }
            return
        finally:
            segmentation_overlay.unlink(missing_ok=True)
        response["2d_slam_map"] = {
            "available": True,
            "status": "current" if pose_projected else "static",
            "path": str(destination),
            "image": str(destination),
            "step_count": snapshot_sequence,
            "memory_nodes": memory_nodes,
            "slam_area_nodes": self._public_slam_area_nodes(slam_area_nodes),
            "memory_overlay": bool(memory_nodes),
            "pose_source": "odometry_projected" if pose_projected else None,
            "segmentation": segmentation,
            "description": (
                "Latest colored SLAM area segmentation with memory-node labels "
                "and the current odometry-projected pose."
                if pose_projected
                else (
                    "Prior-session static colored SLAM area segmentation; relocalization "
                    "has not yet produced a verified current pose marker."
                    if prior_static
                    else "Latest colored SLAM area segmentation without a verified current pose marker."
                )
            ),
        }

    @staticmethod
    def _memory_nodes_for_map(artifact_dir: Path, source: Path) -> list[dict[str, Any]]:
        try:
            payload = json.loads(
                (artifact_dir / f"{source.name}.nodes.json").read_text(encoding="utf-8")
            )
            nodes = payload.get("nodes") if isinstance(payload, dict) else []
            return [dict(node) for node in nodes if isinstance(node, Mapping)]
        except (OSError, json.JSONDecodeError):
            return []

    @staticmethod
    def _public_slam_area_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        public: list[dict[str, Any]] = []
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            row = {
                key: node[key]
                for key in ("id", "kind", "source", "area_id", "name", "label")
                if key in node
            }
            captions = node.get("captions")
            if isinstance(captions, list):
                row["captions"] = [
                    {"text": caption["text"]}
                    for caption in captions
                    if isinstance(caption, Mapping) and isinstance(caption.get("text"), str)
                ]
            public.append(row)
        return public

    @classmethod
    def _map_nodes_in_rendered_crop(
        cls,
        artifact_dir: Path,
        source: Path,
        nodes: list[dict[str, Any]],
        kind: str,
    ) -> list[dict[str, Any]]:
        """Keep response metadata aligned with labels actually drawn on the map."""
        metadata = json.loads(
            (artifact_dir / f"{source.name}.render.json").read_text(encoding="utf-8")
        )
        visible: list[dict[str, Any]] = []
        for node in nodes:
            try:
                if kind == "memory":
                    x, y = float(node["x"]), float(node["y"])
                else:
                    pose = node["slam_pose"]
                    x, y = float(pose["x"]), float(pose["y"])
            except (KeyError, TypeError, ValueError):
                continue
            if cls._map_coordinate_is_rendered(metadata, x, y):
                visible.append(node)
        return visible

    @staticmethod
    def _segmentation_display_image(
        overlay: Path, artifact_dir: Path, source: Path
    ) -> Image.Image:
        metadata = json.loads(
            (artifact_dir / f"{source.name}.render.json").read_text(encoding="utf-8")
        )
        with Image.open(overlay) as image:
            raw = np.asarray(image.convert("RGB"), dtype=np.uint8)
        display = np.flipud(np.transpose(raw, (1, 0, 2)))
        output_size = int(metadata.get("output_size", 1000))
        rendered = Image.new("RGB", (output_size, output_size), (255, 255, 255))
        display_image = Image.fromarray(display, mode="RGB")
        scale = float(metadata["scale"])
        display_image = display_image.resize(
            (
                max(1, int(round(display_image.width * scale))),
                max(1, int(round(display_image.height * scale))),
            ),
            Image.Resampling.NEAREST,
        )
        rendered.paste(
            display_image,
            (
                int(round(float(metadata["offset_x"]) + float(metadata["left_padding"]) * scale)),
                int(round(float(metadata["offset_y"]) + float(metadata["top_padding"]) * scale)),
            ),
        )
        return rendered

    @staticmethod
    def _map_pixel(metadata: Mapping[str, Any], x: float, y: float) -> tuple[float, float]:
        resolution = float(metadata["resolution"])
        scale = float(metadata["scale"])
        display_x = float(metadata["mosaic_origin_y"]) - y / resolution
        display_y = float(metadata["source_width"]) - 1.0 - (
            x / resolution + float(metadata["mosaic_origin_x"])
        )
        return (
            float(metadata["offset_x"]) + (float(metadata["left_padding"]) + display_x) * scale,
            float(metadata["offset_y"]) + (float(metadata["top_padding"]) + display_y) * scale,
        )

    @staticmethod
    def _map_coordinate_is_rendered(
        metadata: Mapping[str, Any], x: float, y: float
    ) -> bool:
        """Return whether a SLAM coordinate belongs to the rendered map crop."""
        resolution = float(metadata["resolution"])
        display_x = float(metadata["mosaic_origin_y"]) - y / resolution
        display_y = float(metadata["source_width"]) - 1.0 - (
            x / resolution + float(metadata["mosaic_origin_x"])
        )
        return (
            0.0 <= display_x < float(metadata["display_width"])
            and 0.0 <= display_y < float(metadata["display_height"])
        )

    @classmethod
    def _draw_map_labels(
        cls,
        image: Image.Image,
        artifact_dir: Path,
        source: Path,
        nodes: list[dict[str, Any]],
        kind: str,
    ) -> None:
        if not nodes:
            return
        metadata = json.loads(
            (artifact_dir / f"{source.name}.render.json").read_text(encoding="utf-8")
        )
        font_size = max(14, int(round((1.5 if kind == "memory" else 1.0) * float(metadata["scale"]) / float(metadata["resolution"]))))
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
        draw = ImageDraw.Draw(image)
        for node in nodes:
            try:
                if kind == "memory":
                    label = str(int(node["order_id"]))
                    x, y = float(node["x"]), float(node["y"])
                else:
                    label = str(node.get("label") or node.get("id") or "")
                    pose = node["slam_pose"]
                    x, y = float(pose["x"]), float(pose["y"])
                if not cls._map_coordinate_is_rendered(metadata, x, y):
                    continue
                px, py = cls._map_pixel(metadata, x, y)
            except (KeyError, TypeError, ValueError):
                continue
            if not label or not (0 <= px < image.width and 0 <= py < image.height):
                continue
            box = draw.textbbox((0, 0), label, font=font)
            pos = (px - (box[2] - box[0]) / 2, py - (box[3] - box[1]) / 2)
            draw.text(
                pos,
                label,
                font=font,
                fill=(0, 0, 0) if kind == "memory" else (255, 255, 255),
                stroke_width=4 if kind == "memory" else 3,
                stroke_fill=(255, 255, 255) if kind == "memory" else (0, 0, 0),
            )

    @classmethod
    def _draw_current_pose_marker(
        cls,
        image: Image.Image,
        artifact_dir: Path,
        source: Path,
        status: Mapping[str, Any],
    ) -> bool:
        current = status.get("latest_odometry")
        if not isinstance(current, (list, tuple)) or len(current) != 3:
            return False
        try:
            sequence = int(source.stem.removeprefix("occupancy_step"))
            poses = (artifact_dir / "occupancy.png.poses.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            pose = next(
                row
                for row in (json.loads(line) for line in reversed(poses))
                if int(row.get("sequence", -1)) == sequence
            )
            matched_x, matched_y = float(pose["x"]), float(pose["y"])
            matched_yaw = float(pose["yaw_rad"])
            odom_x, odom_y = float(pose["odometry_x"]), float(pose["odometry_y"])
            odom_yaw = float(pose["odometry_yaw_rad"])
            dx, dy = float(current[0]) - odom_x, float(current[1]) - odom_y
            cosine, sine = math.cos(matched_yaw - odom_yaw), math.sin(matched_yaw - odom_yaw)
            map_x = matched_x + cosine * dx - sine * dy
            map_y = matched_y + sine * dx + cosine * dy
            yaw = matched_yaw + float(current[2]) - odom_yaw
            metadata = json.loads(
                (artifact_dir / f"{source.name}.render.json").read_text(encoding="utf-8")
            )
            if not cls._map_coordinate_is_rendered(metadata, map_x, map_y):
                return False
            x, y = cls._map_pixel(metadata, map_x, map_y)
            heading = -yaw - math.pi / 2.0
            draw = ImageDraw.Draw(image)
            tip = (x + 72 * math.cos(heading), y + 72 * math.sin(heading))
            left = (x + 42 * math.cos(heading + 2.5), y + 42 * math.sin(heading + 2.5))
            right = (x + 42 * math.cos(heading - 2.5), y + 42 * math.sin(heading - 2.5))
            draw.polygon((tip, left, right), fill=(220, 30, 30), outline=(242, 199, 5), width=8)
            return True
        except (OSError, StopIteration, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False

    @staticmethod
    def _publish_slam_area_nodes(
        *,
        artifact_dir: Path,
        source: Path,
        segmentation_source: Path,
        segmentation_result: Mapping[str, Any],
        memory_nodes: list[Any],
        planning_artifact_dir: Path | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Build visual-map area nodes using the original occupancy logic."""
        areas = segmentation_result.get("areas")
        if not isinstance(areas, list):
            areas = []
        if segmentation_source.suffix != ".pgm":
            return [], {
                "sequence": None,
                "source_map": source.name,
                "segmentation_source": segmentation_source.name,
                "nodes": [],
                "decisions": [],
                "disabled_reason": "display_png_segmentation_has_no_raw_grid_pose",
            }
        try:
            slam_map = load_slam_occupancy_map(segmentation_source)
            source_sequence = int(source.stem.removeprefix("occupancy_step"))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return [], {
                "sequence": None,
                "source_map": source.name,
                "segmentation_source": segmentation_source.name,
                "nodes": [],
                "decisions": [],
                "disabled_reason": "raw_grid_pose_unavailable",
            }

        planning_slam_map = None
        if planning_artifact_dir is not None:
            try:
                planning_slam_map = load_slam_occupancy_map(planning_artifact_dir)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                planning_slam_map = None
        clear_slam_map = planning_slam_map or slam_map

        try:
            marked_radius_m = max(
                0.0,
                float(os.environ.get("HAB_SLAM_AREA_NODE_MARKED_RADIUS_M", "0.75")),
            )
        except ValueError:
            marked_radius_m = 0.75
        try:
            resample_radius_m = max(
                0.0,
                float(os.environ.get("HAB_SLAM_AREA_NODE_RESAMPLE_RADIUS_M", "1.0")),
            )
        except ValueError:
            resample_radius_m = 1.0
        try:
            resample_inflation_m = max(
                0.0,
                float(
                    os.environ.get(
                        "HAB_SLAM_AREA_NODE_RESAMPLE_INFLATION_RADIUS_M",
                        os.environ.get("HAB_SLAM_OCC_INFLATION_RADIUS_M", "0.15"),
                    )
                ),
            )
        except ValueError:
            resample_inflation_m = 0.15
        try:
            clear_costmap, _ = build_navigation_costmap(
                clear_slam_map,
                inflation_radius_m=resample_inflation_m,
                allow_unknown=False,
            )
        except Exception:  # Sidecar generation is best effort.
            clear_costmap = None

        generated: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        memory_points: list[tuple[float, float]] = []
        for node in memory_nodes:
            if not isinstance(node, Mapping):
                continue
            try:
                memory_points.append((float(node["x"]), float(node["y"])))
            except (KeyError, TypeError, ValueError):
                continue

        for area in areas:
            if not isinstance(area, Mapping):
                continue
            try:
                area_id = int(area["id"])
                centroid = area["centroid"]
                row = int(round(float(centroid[0])))
                col = int(round(float(centroid[1])))
                bbox = area["bbox"]
                pixel_count = int(area.get("pixel_count", 0))
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue

            x, y = slam_map.grid_to_map(row, col)
            nav_x, nav_y = x, y
            resampled = False
            resample_status = "not_needed"
            snapped_cell: tuple[int, int] | None = None
            center_cell = clear_slam_map.map_to_grid(x, y)
            if clear_costmap is not None:
                max_radius_cells = max(
                    0,
                    int(math.ceil(resample_radius_m / clear_slam_map.resolution)),
                )
                snapped_cell = snap_to_traversable(
                    clear_costmap, center_cell, max_radius_cells
                )
                if snapped_cell is None:
                    resample_status = "no_clear_cell"
                elif snapped_cell != center_cell:
                    nav_x, nav_y = clear_slam_map.grid_to_map(*snapped_cell)
                    resampled = True
                    resample_status = "resampled"
            if resample_status == "no_clear_cell":
                decisions.append(
                    {
                        "area_id": area_id,
                        "skipped": True,
                        "reason": "no_clear_position_near_center",
                        "resample_status": resample_status,
                        "pose_resample_center_cell": [
                            int(center_cell[0]),
                            int(center_cell[1]),
                        ],
                    }
                )
                continue

            memory_node_inside_area = False
            for memory_x, memory_y in memory_points:
                memory_row, memory_col = slam_map.map_to_grid(memory_x, memory_y)
                if (
                    int(bbox[0]) <= memory_row <= int(bbox[2])
                    and int(bbox[1]) <= memory_col <= int(bbox[3])
                ):
                    memory_node_inside_area = True
                    break
            nearest_memory_distance_m = min(
                (math.hypot(x - mx, y - my) for mx, my in memory_points),
                default=float("inf"),
            )
            if (
                memory_node_inside_area
                or nearest_memory_distance_m <= marked_radius_m
            ):
                decisions.append(
                    {
                        "area_id": area_id,
                        "skipped": True,
                        "reason": (
                            "contains_memory_node"
                            if memory_node_inside_area
                            else "near_memory_node"
                        ),
                        "nearest_memory_distance_m": round(
                            nearest_memory_distance_m, 3
                        ),
                        "resample_status": resample_status,
                    }
                )
                continue

            node = {
                "id": f"slam_area_{area_id}",
                "kind": "slam_area",
                "source": "slam_area_segmentation",
                "area_id": area_id,
                "name": f"Unmarked SLAM area {area_id}",
                "label": f"A{area_id}",
                "slam_pose": {
                    "sequence": source_sequence,
                    "x": float(nav_x),
                    "y": float(nav_y),
                    "yaw_rad": 0.0,
                },
                "center_slam_pose": {
                    "sequence": source_sequence,
                    "x": float(x),
                    "y": float(y),
                    "yaw_rad": 0.0,
                },
                "pose_resampled": resampled,
                "pose_resample_status": resample_status,
                "pose_resample_source": (
                    "planning_occupancy"
                    if planning_slam_map is not None
                    else "visual_occupancy"
                ),
                "pose_resample_center_cell": [
                    int(center_cell[0]),
                    int(center_cell[1]),
                ],
                "pose_resample_snapped_cell": (
                    [int(snapped_cell[0]), int(snapped_cell[1])]
                    if snapped_cell is not None
                    else None
                ),
                "centroid_px": [row, col],
                "bbox_px": [int(value) for value in bbox],
                "pixel_count": pixel_count,
            }
            generated.append(node)
            decisions.append(
                {
                    "area_id": area_id,
                    "skipped": False,
                    "node_id": node["id"],
                    "nearest_memory_distance_m": (
                        None
                        if math.isinf(nearest_memory_distance_m)
                        else round(nearest_memory_distance_m, 3)
                    ),
                    "resample_status": resample_status,
                    "pose_resampled": resampled,
                }
            )

        payload = {
            "sequence": source_sequence,
            "source_map": source.name,
            "segmentation_source": segmentation_source.name,
            "marked_radius_m": marked_radius_m,
            "resample_radius_m": resample_radius_m,
            "resample_inflation_m": resample_inflation_m,
            "resample_source": (
                "planning_occupancy"
                if planning_slam_map is not None
                else "visual_occupancy"
            ),
            "planning_sequence": (
                planning_slam_map.sequence if planning_slam_map is not None else None
            ),
            "nodes": generated,
            "decisions": decisions,
        }
        return generated, payload

    @staticmethod
    def _write_slam_area_nodes_sidecar(
        artifact_dir: Path, payload: Mapping[str, Any]
    ) -> None:
        temporary = artifact_dir / ".slam_area_nodes.tmp"
        try:
            temporary.write_text(
                json.dumps(payload, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, artifact_dir / "slam_area_nodes.json")
        finally:
            temporary.unlink(missing_ok=True)

    def _get_memory_slam_status(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        del payload
        return self._memory_slam_status(
            self._require_session(session_id), wait_for_pose=True
        )

    def _check_relocalization_prior(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        session = self._require_session(session_id)
        node_id = payload.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            raise HabitatAdapterError('Field "payload.node_id" must be a non-empty string')
        tool_seq = payload.get("tool_seq")
        if not isinstance(tool_seq, int) or isinstance(tool_seq, bool) or tool_seq < 0:
            raise HabitatAdapterError(
                'Field "payload.tool_seq" must be a non-negative integer'
            )
        node = payload.get("node")
        if not isinstance(node, Mapping) or node.get("id") != node_id:
            raise HabitatAdapterError(
                'Field "payload.node" must match payload.node_id'
            )
        stream = session.cartographer_stream
        recorded = False
        if stream is not None:
            _write_reported_memory_node_prior(
                stream.artifact_dir,
                session_id=session.session_id,
                node=node,
                tool_seq=tool_seq,
            )
            recorded = True
        accepted = bool(
            stream is not None
            and stream.request_relocalization_prior_check(
                tool_seq=tool_seq, node_id=node_id
            )
        )
        return {
            "node_id": node_id,
            "tool_seq": tool_seq,
            "prior_recorded": recorded,
            "accepted": accepted,
        }

    def _refresh_memory_nodes(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        session = self._require_session(session_id)
        nodes = payload.get("nodes")
        if not isinstance(nodes, list) or any(not isinstance(node, Mapping) for node in nodes):
            raise HabitatAdapterError('Field "payload.nodes" must be an array of objects')
        stream = session.cartographer_stream
        if stream is None:
            return {"updated": False, "node_count": len(nodes)}
        stream.update_memory_nodes([dict(node) for node in nodes])
        return {"updated": True, "node_count": len(nodes)}

    def _navigate_slam_occupancy(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        session = self._require_session(session_id)
        goal_payload = payload.get("goal_slam_pose")
        if not isinstance(goal_payload, Mapping):
            raise HabitatAdapterError(
                'Field "payload.goal_slam_pose" must be an object with x, y, yaw_rad'
            )
        try:
            goal_pose = {
                "x": float(goal_payload["x"]),
                "y": float(goal_payload["y"]),
                "yaw_rad": float(goal_payload.get("yaw_rad", 0.0)),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise HabitatAdapterError(
                'Field "payload.goal_slam_pose" must contain numeric x and y'
            ) from exc
        stream = session.cartographer_occ_stream or session.cartographer_stream
        artifact_dir_value = getattr(stream, "artifact_dir", None)
        if artifact_dir_value is None:
            return {
                "session_id": session.session_id,
                "ok": False,
                "status": "slam_unavailable",
                "error": "navigate_slam_occupancy requires an active Cartographer stream.",
            }
        artifact_dir = Path(artifact_dir_value)
        def coerce_int(value: Any, field_name: str) -> int:
            if isinstance(value, bool):
                raise HabitatAdapterError(f'Field "{field_name}" must be int')
            try:
                parsed = int(value)
            except (TypeError, ValueError) as exc:
                raise HabitatAdapterError(f'Field "{field_name}" must be int') from exc
            if isinstance(value, float) and not value.is_integer():
                raise HabitatAdapterError(f'Field "{field_name}" must be int')
            return parsed

        preview_only = self._coerce_bool(
            payload.get("preview_only", False), field_name="payload.preview_only"
        )
        max_steps = coerce_int(payload.get("max_steps", 40), "payload.max_steps")
        if max_steps < 1:
            raise HabitatAdapterError('Field "payload.max_steps" must be >= 1')
        goal_radius_m = self._coerce_float(
            payload.get("goal_radius_m", 0.45), field_name="payload.goal_radius_m"
        )
        replan_attempts = coerce_int(
            payload.get("replan_attempts", 2), "payload.replan_attempts"
        )
        inflation_radius_m = self._coerce_float(
            payload.get(
                "inflation_radius_m",
                self._env_float("HAB_SLAM_OCC_INFLATION_RADIUS_M", 0.15),
            ),
            field_name="payload.inflation_radius_m",
        )
        unknown_cost = self._coerce_float(
            payload.get("unknown_cost", 80.0), field_name="payload.unknown_cost"
        )
        allow_unknown = self._coerce_bool(
            payload.get("allow_unknown", False), field_name="payload.allow_unknown"
        )
        heading_tolerance_deg = max(
            0.0, self._env_float("HAB_SLAM_OCC_HEADING_TOLERANCE_DEG", 2.0)
        )
        turn_quantum_deg = max(
            1e-6, self._env_float("HAB_SLAM_OCC_TURN_QUANTUM_DEG", 10.0)
        )
        forward_step_m = max(
            1e-6, self._env_float("HAB_SLAM_OCC_FORWARD_STEP_M", 0.25)
        )
        start_clear_radius_m = max(
            0.0, self._env_float("HAB_SLAM_OCC_START_CLEAR_RADIUS_M", 0.20)
        )
        include_map = self._coerce_bool(
            payload.get("include_cartographer_map", False),
            field_name="payload.include_cartographer_map",
        )
        output_dir = payload.get("output_dir", "data/nav_artifacts")
        if not isinstance(output_dir, str) or not output_dir:
            raise HabitatAdapterError('Field "payload.output_dir" must be a non-empty str')
        nav_session_dir = Path(self._session_output_dir(output_dir, session.session_id))
        nav_session_dir.mkdir(parents=True, exist_ok=True)
        pose_wait_timeout_s = max(
            0.0,
            self._env_float(
                "HAB_SLAM_OCC_PLAN_RENDER_TIMEOUT_S",
                self._env_float("HAB_CARTOGRAPHER_POSE_WAIT_TIMEOUT_S", 30.0),
            ),
        )
        lag_tolerance = max(
            0, int(self._env_float("HAB_SLAM_OCC_POSE_SEQUENCE_LAG_TOLERANCE", 1.0))
        )
        pose_syncs: list[dict[str, Any]] = []

        def latest_pose() -> dict[str, Any] | None:
            try:
                lines = (artifact_dir / "occupancy.png.poses.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            except OSError:
                return None
            for line in reversed(lines):
                try:
                    pose = json.loads(line)
                    return {
                        "sequence": int(pose["sequence"]),
                        "x": float(pose["x"]),
                        "y": float(pose["y"]),
                        "yaw_rad": float(pose["yaw_rad"]),
                    }
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
            return None

        def latest_submitted_sequence() -> int | None:
            sequence = getattr(stream, "_latest_odometry_sequence", None)
            if isinstance(sequence, int) and not isinstance(sequence, bool):
                return sequence
            try:
                stream_status = stream.status()
            except Exception:
                return None
            sequence = stream_status.get("latest_odometry_sequence")
            return sequence if isinstance(sequence, int) and not isinstance(sequence, bool) else None

        def pose_after_render(
            *, min_sequence: int | None, reason: str
        ) -> dict[str, Any] | None:
            # One configured timeout covers both the worker render wait and
            # pose-file catch-up.  Starting a second full deadline after the
            # render wait can exceed BridgeClient's 60 s request timeout and
            # turn a useful slam_map_pending result into a transport timeout.
            deadline = time.monotonic() + pose_wait_timeout_s
            target = (
                max(0, min_sequence - lag_tolerance)
                if isinstance(min_sequence, int) and not isinstance(min_sequence, bool)
                else None
            )
            rendered = None
            if target is not None and hasattr(stream, "wait_for_rendered_sequence"):
                try:
                    rendered = bool(
                        stream.wait_for_rendered_sequence(
                            target, max(0.0, deadline - time.monotonic())
                        )
                    )
                except Exception:
                    rendered = False
            pose = latest_pose()
            while (
                target is not None
                and (pose is None or int(pose.get("sequence", -1)) < target)
                and time.monotonic() < deadline
            ):
                time.sleep(0.02)
                pose = latest_pose()
            pose_syncs.append(
                {
                    "reason": reason,
                    "target_sequence": target,
                    "submitted_sequence": min_sequence,
                    "sequence_lag_tolerance": lag_tolerance,
                    "pose_sequence": int(pose["sequence"]) if pose is not None else None,
                    "rendered_target_sequence": rendered,
                    "timeout_s": pose_wait_timeout_s,
                }
            )
            return pose

        def make_plan(start_pose: Mapping[str, Any]):
            slam_map = load_slam_occupancy_map(artifact_dir)
            plan, costmap = plan_slam_occupancy_path(
                slam_map,
                start=start_pose,
                goal=goal_pose,
                inflation_radius_m=inflation_radius_m,
                unknown_cost=unknown_cost,
                allow_unknown=allow_unknown,
                start_clear_radius_m=start_clear_radius_m,
            )
            return slam_map, plan, costmap

        start_pose = pose_after_render(
            min_sequence=latest_submitted_sequence(), reason="initial_plan"
        )
        if start_pose is None:
            return {
                "session_id": session.session_id,
                "ok": False,
                "status": "missing_slam_pose",
                "error": "No current Cartographer pose is available.",
            }
        expected_sequence = latest_submitted_sequence()
        if (
            expected_sequence is not None
            and int(start_pose.get("sequence", -1)) + lag_tolerance < expected_sequence
        ):
            return {
                "session_id": session.session_id,
                "ok": False,
                "status": "slam_map_pending",
                "error": "Planning pose has not caught up to the latest Cartographer scan.",
                "executor": {"pose_syncs": pose_syncs},
            }
        try:
            slam_map, plan, costmap = make_plan(start_pose)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            return {
                "session_id": session.session_id,
                "ok": False,
                "status": "slam_map_unavailable",
                "error": f"{type(exc).__name__}: {exc}",
            }
        compiled_route = None
        if plan.reachable:
            compiled_route = compile_slam_occupancy_route(
                slam_map,
                plan,
                costmap,
                start_pose=start_pose,
                goal=goal_pose,
                goal_radius_m=goal_radius_m,
                turn_quantum_deg=turn_quantum_deg,
                forward_step_m=forward_step_m,
                heading_tolerance_deg=heading_tolerance_deg,
                max_actions=4096,
            )
        if preview_only or not plan.reachable:
            executable = bool(
                plan.reachable and compiled_route is not None and compiled_route.executable
            )
            response: Dict[str, Any] = {
                "session_id": session.session_id,
                "ok": executable,
                "status": (
                    "preview_ready"
                    if executable
                    else ("unexecutable" if plan.reachable else plan.status)
                ),
                "backend": "slam_occupancy",
                "allow_unknown": allow_unknown,
                "plan": plan.summary(),
                "executor": (
                    {**compiled_route.summary(), "pose_syncs": pose_syncs}
                    if compiled_route is not None
                    else {
                        "compile_status": "not_compiled",
                        "compile_failure_reason": plan.status,
                        "pose_syncs": pose_syncs,
                    }
                ),
            }
            if preview_only:
                preview_path = nav_session_dir / "slam_occupancy_preview.png"
                try:
                    render_slam_occupancy_plan_overlay(
                        slam_map,
                        preview_path,
                        plan=plan,
                        compiled_route=compiled_route,
                        display_frame=True,
                    )
                    response["2d_slam_map"] = {
                        "available": True,
                        "status": "preview_ready",
                        "path": str(preview_path),
                        "image": str(preview_path),
                        "step_count": plan.map_sequence,
                        "description": "Preview of the SLAM occupancy plan and compiled executor trace.",
                    }
                except (OSError, ValueError) as exc:
                    response["2d_slam_map"] = {
                        "available": False,
                        "status": "error",
                        "step_count": plan.map_sequence,
                        "description": "SLAM occupancy plan preview could not be rendered.",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            return response

        actions_taken: list[str] = []
        plans: list[dict[str, Any]] = [plan.summary()]
        attempts = 0
        collided = False
        status = "en_route"
        response: Dict[str, Any] = {}
        executed_action_index = -1
        executor: dict[str, Any] = {
            "heading_tolerance_deg": heading_tolerance_deg,
            "turn_quantum_deg": turn_quantum_deg,
            "forward_step_m": forward_step_m,
            "start_clear_radius_m": start_clear_radius_m,
            "pose_syncs": pose_syncs,
        }
        while len(actions_taken) < max_steps:
            if compiled_route is None:
                status = "blocked"
                break
            executor.update(compiled_route.summary())
            if compiled_route.status == "already_reached":
                status = "reached"
                break
            if not compiled_route.executable:
                status = "blocked"
                response["error"] = compiled_route.failure_reason
                break
            if not compiled_route.actions:
                status = "en_route"
                break
            collision_replan = False
            for compiled_action in compiled_route.actions:
                if len(actions_taken) >= max_steps:
                    status = "en_route"
                    break
                step_result = self._step_action(
                    session.session_id, compiled_action.payload()
                )
                actions_taken.append(str(compiled_action.action))
                executed_action_index += 1
                response = dict(step_result)
                collided = bool(step_result.get("collided"))
                executor.update(
                    {
                        "executed_action_index": executed_action_index,
                        "last_compiled_action": compiled_action.summary(),
                    }
                )
                if collided:
                    attempts += 1
                    collision_replan = True
                    break
            if not collision_replan:
                if len(actions_taken) >= max_steps:
                    status = "en_route"
                    break
                expected_sequence = latest_submitted_sequence()
                current_pose = pose_after_render(
                    min_sequence=expected_sequence, reason="final_status"
                )
                if current_pose is None:
                    status = "missing_slam_pose"
                    break
                if (
                    expected_sequence is not None
                    and int(current_pose.get("sequence", -1)) + lag_tolerance
                    < expected_sequence
                ):
                    status = "slam_map_pending"
                    response["error"] = "Planning pose has not caught up after route execution."
                    break
                distance = math.hypot(
                    float(current_pose["x"]) - goal_pose["x"],
                    float(current_pose["y"]) - goal_pose["y"],
                )
                status = "reached" if distance <= goal_radius_m else "en_route"
                break
            if attempts > max(0, replan_attempts):
                status = "blocked"
                break
            expected_sequence = latest_submitted_sequence()
            current_pose = pose_after_render(
                min_sequence=expected_sequence, reason="collision_replan"
            )
            if current_pose is None:
                status = "missing_slam_pose"
                break
            if (
                expected_sequence is not None
                and int(current_pose.get("sequence", -1)) + lag_tolerance
                < expected_sequence
            ):
                status = "slam_map_pending"
                response["error"] = (
                    "Planning pose has not caught up after collision; refusing to replan "
                    "from a stale pose."
                )
                break
            try:
                slam_map, plan, costmap = make_plan(current_pose)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                status = "slam_map_unavailable"
                response["error"] = f"{type(exc).__name__}: {exc}"
                break
            plans.append(plan.summary())
            if not plan.reachable:
                status = plan.status
                break
            compiled_route = compile_slam_occupancy_route(
                slam_map,
                plan,
                costmap,
                start_pose=current_pose,
                goal=goal_pose,
                goal_radius_m=goal_radius_m,
                turn_quantum_deg=turn_quantum_deg,
                forward_step_m=forward_step_m,
                heading_tolerance_deg=heading_tolerance_deg,
                max_actions=4096,
            )
            collided = False

        response.update(
            {
                "session_id": session.session_id,
                "ok": status in {"reached", "en_route"},
                "status": status,
                "nav_status": status,
                "backend": "slam_occupancy",
                "allow_unknown": allow_unknown,
                "actions_taken": actions_taken,
                "steps_executed": len(actions_taken),
                "replans_attempted": attempts,
                "plans": plans,
                "collided": collided,
                "step_count": session.step_count,
                "executor": executor,
            }
        )
        if include_map:
            self._attach_memory_slam_map_context(
                session, response, str(nav_session_dir)
            )
        return response


__all__ = ["HabitatMemorySlamMixin"]
