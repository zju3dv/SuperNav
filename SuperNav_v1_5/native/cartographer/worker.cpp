// Pose-free Cartographer 2D scan-matching worker.
// Inputs are NDJSON records from stdin: sequence, timestamp_s, and ranges_m.

#include <algorithm>
#include <cmath>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <deque>
#include <exception>
#include <fstream>
#include <filesystem>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <regex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cartographer/common/configuration_file_resolver.h"
#include "cartographer/common/lua_parameter_dictionary.h"
#include "cartographer/common/time.h"
#include "cartographer/io/submap_painter.h"
#include "cartographer/mapping/map_builder.h"
#include "cartographer/mapping/map_builder_interface.h"
#include "cartographer/mapping/pose_graph.h"
#include "cartographer/mapping/proto/submap_visualization.pb.h"
#include "cartographer/sensor/timed_point_cloud_data.h"
#include "cartographer/sensor/odometry_data.h"
#include "cartographer/transform/transform.h"
#include "delayed_relocalization.h"
#include "scan.h"

namespace {

using habitat_cartographer::BeamAngle;
using habitat_cartographer::BuildRecentTrajectoryOccupancySource;
using habitat_cartographer::DelayedRelocalization;
using habitat_cartographer::ParseScan;
using habitat_cartographer::RelocalizationSource;
using habitat_cartographer::ResetRelocalizationSource;
using habitat_cartographer::Scan;
using habitat_cartographer::AppendScanMatchedScanPoints;
using habitat_cartographer::StartRelocalizationAttempt;
using habitat_cartographer::ToPointCloud;
using habitat_cartographer::WriteLocalizationStatus;
using habitat_cartographer::WriteXYCloud;

struct MemoryNodeLabel {
  int order_id;
  std::string name;
  double x;
  double y;
};

struct ReportedMemoryNodePrior {
  std::string node_id;
  int tool_seq;
  int sequence;
  double x;
  double y;
  double yaw_rad;
  std::filesystem::path path;
};

struct AnchorError {
  double translation_m;
  double yaw_rad;
  cartographer::transform::Rigid3d current;
};

cartographer::transform::Rigid3d TrajectoryFromGlobal(
    cartographer::mapping::MapBuilder* map_builder, int trajectory_id);

RelocalizationSource SelectRelocalizationSource(
    const DelayedRelocalization& delayed, bool sparse_node_prior);

bool LooksLikeDoorNode(const std::string& id, const std::string& name) {
  std::string value = id + " " + name;
  for (char& ch : value) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return value.find("door") != std::string::npos ||
         value.find("doorway") != std::string::npos;
}

double EnvDouble(const char* name, double fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || std::string(raw).empty()) return fallback;
  try {
    return std::stod(raw);
  } catch (const std::exception&) {
    return fallback;
  }
}

bool EnvBool(const char* name, bool fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || std::string(raw).empty()) return fallback;
  std::string value(raw);
  for (char& ch : value) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  if (value == "1" || value == "true" || value == "yes" || value == "on") return true;
  if (value == "0" || value == "false" || value == "no" || value == "off") return false;
  return fallback;
}

int EnvInt(const char* name, int fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || std::string(raw).empty()) return fallback;
  try {
    return std::stoi(raw);
  } catch (const std::exception&) {
    return fallback;
  }
}

void AppendMemoryNodeLabel(std::vector<MemoryNodeLabel>* labels,
                          const std::smatch& match,
                          int order_index,
                          int name_index,
                          int x_index,
                          int y_index) {
  try {
    labels->push_back({std::stoi(match[order_index]), match[name_index],
                       std::stod(match[x_index]), std::stod(match[y_index])});
  } catch (const std::exception&) {
    return;
  }
}

AnchorError PoseGraphAnchorError(
    cartographer::mapping::MapBuilder* map_builder, int local_trajectory_id,
    int old_trajectory_id,
    const cartographer::transform::Rigid3d& local_pose,
    const cartographer::transform::Rigid3d& requested_old_from_local) {
  const auto old_from_global = TrajectoryFromGlobal(map_builder, old_trajectory_id);
  const auto current_old_from_local = old_from_global *
      map_builder->pose_graph()->GetLocalToGlobalTransform(local_trajectory_id) *
      local_pose;
  const auto error = requested_old_from_local.inverse() * current_old_from_local;
  const auto rotation = error.rotation().toRotationMatrix();
  return {error.translation().head<2>().norm(),
          std::abs(std::atan2(rotation(1, 0), rotation(0, 0))),
          current_old_from_local};
}

AnchorError TransformAnchorError(
    const cartographer::transform::Rigid3d& old_from_local,
    const cartographer::transform::Rigid3d& local_pose,
    const cartographer::transform::Rigid3d& requested_old_from_pose) {
  const auto current_old_from_pose = old_from_local * local_pose;
  const auto error = requested_old_from_pose.inverse() * current_old_from_pose;
  const auto rotation = error.rotation().toRotationMatrix();
  return {error.translation().head<2>().norm(),
          std::abs(std::atan2(rotation(1, 0), rotation(0, 0))),
          current_old_from_pose};
}

cartographer::transform::Rigid3d OdomPose(const Scan& scan) {
  return cartographer::transform::Rigid3d(
      {scan.odometry_x, scan.odometry_y, 0.},
      Eigen::AngleAxisd(scan.odometry_yaw_rad, Eigen::Vector3d::UnitZ()));
}

cartographer::transform::Rigid3d TrajectoryFromGlobal(
    cartographer::mapping::MapBuilder* map_builder, int trajectory_id) {
  return map_builder->pose_graph()->GetLocalToGlobalTransform(trajectory_id)
      .inverse();
}

double YawOf(const cartographer::transform::Rigid3d& pose) {
  const auto rotation = pose.rotation().toRotationMatrix();
  return std::atan2(rotation(1, 0), rotation(0, 0));
}

std::vector<MemoryNodeLabel> ReadMemoryNodeLabels(const std::string& output_path) {
  std::ifstream input(std::filesystem::path(output_path).parent_path() /
                      "memory_nodes.json");
  if (!input) return {};
  const std::string content((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
  std::vector<MemoryNodeLabel> labels;
  // New sidecars include kind, which is the authoritative door filter.
  const std::regex typed_pattern(
      R"graph(\{"id":"([^"]*)","order_id":([0-9]+),"kind":"([^"]*)","name":"([^"]*)","slam_pose":\{"sequence":[0-9]+,"x":([-0-9.eE]+),"y":([-0-9.eE]+),"yaw_rad":[-0-9.eE]+\},"has_pano_images":(?:true|false)\})graph");
  for (std::sregex_iterator it(content.begin(), content.end(), typed_pattern), end;
       it != end; ++it) {
    if ((*it)[3] == "door") continue;
    AppendMemoryNodeLabel(&labels, *it, 2, 4, 5, 6);
  }
  if (!labels.empty()) return labels;
  // Legacy sidecars produced before kind was published should not crash the
  // worker. Skip obvious door nodes by id/name until the bridge is restarted
  // and the authoritative kind field is available.
  const std::regex legacy_pattern(
      R"graph(\{"id":"([^"]*)","order_id":([0-9]+),"name":"([^"]*)","slam_pose":\{"sequence":[0-9]+,"x":([-0-9.eE]+),"y":([-0-9.eE]+),"yaw_rad":[-0-9.eE]+\},"has_pano_images":(?:true|false)\})graph");
  for (std::sregex_iterator it(content.begin(), content.end(), legacy_pattern), end;
       it != end; ++it) {
    if (LooksLikeDoorNode((*it)[1], (*it)[3])) continue;
    AppendMemoryNodeLabel(&labels, *it, 2, 3, 4, 5);
  }
  return labels;
}

std::optional<ReportedMemoryNodePrior> ReadReportedMemoryNodePrior(
    const std::filesystem::path& artifact_dir, int expected_tool_seq = -1,
    const std::string& expected_node_id = "") {
  const auto path = artifact_dir / "reported_memory_node.json";
  std::ifstream input(path);
  if (!input) return std::nullopt;
  const std::string content((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
  std::smatch node_match;
  if (!std::regex_search(content, node_match,
                         std::regex(R"json("node_id":"([^"]+)")json"))) {
    return std::nullopt;
  }
  std::smatch tool_seq_match;
  if (!std::regex_search(content, tool_seq_match,
                         std::regex(R"("tool_seq":([0-9]+))"))) {
    return std::nullopt;
  }
  std::smatch pose_match;
  const std::regex pose_pattern(
      R"("slam_pose":\{"sequence":([0-9]+),"x":([-0-9.eE]+),"y":([-0-9.eE]+),"yaw_rad":([-0-9.eE]+)\})");
  if (!std::regex_search(content, pose_match, pose_pattern)) {
    return std::nullopt;
  }
  try {
    const int tool_seq = std::stoi(tool_seq_match[1]);
    const std::string node_id = node_match[1];
    if ((expected_tool_seq >= 0 && tool_seq != expected_tool_seq) ||
        (!expected_node_id.empty() && node_id != expected_node_id)) {
      return std::nullopt;
    }
    const auto snapshot_path = artifact_dir /
        ("reported_memory_node_tool" + std::to_string(tool_seq) + ".json");
    std::ofstream snapshot(snapshot_path);
    snapshot << content;
    snapshot.close();
    return ReportedMemoryNodePrior{
        node_id,
        tool_seq,
        std::stoi(pose_match[1]),
        std::stod(pose_match[2]),
        std::stod(pose_match[3]),
        std::stod(pose_match[4]),
        snapshot_path};
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

bool IsRelocalizationPriorCheckControl(const std::string& line) {
  return line.find(R"("type":"check_relocalization_prior")") != std::string::npos;
}

std::optional<std::pair<int, std::string>> ParseRelocalizationPriorCheckControl(
    const std::string& line) {
  if (!IsRelocalizationPriorCheckControl(line)) return std::nullopt;
  std::smatch tool_seq_match;
  std::smatch node_match;
  if (!std::regex_search(line, tool_seq_match,
                         std::regex(R"("tool_seq":([0-9]+))")) ||
      !std::regex_search(line, node_match,
                         std::regex(R"json("node_id":"([^"]+)")json"))) {
    return std::nullopt;
  }
  return std::make_pair(std::stoi(tool_seq_match[1]),
                        std::string(node_match[1]));
}

void WritePriorCheckEvent(const std::string& output, int sequence,
                          const std::string& event,
                          const std::string& trigger_source,
                          const DelayedRelocalization& delayed,
                          int node_prior_min_points,
                          const std::string& reason = "") {
  std::ofstream events(output + ".localization.events.jsonl", std::ios::app);
  events << "{\"sequence\":" << sequence
         << ",\"event\":\"" << event << "\""
         << ",\"trigger\":\"agent_reported_node\""
         << ",\"trigger_source\":\"" << trigger_source << "\""
         << ",\"unique_points\":" << delayed.unique_points
         << ",\"node_prior_min_points\":" << node_prior_min_points;
  if (!reason.empty()) {
    events << ",\"reason\":\"" << reason << "\"";
  }
  events << "}\n";
}

bool TryStartNodePriorRelocalization(const std::string& output,
                                     DelayedRelocalization* delayed,
                                     int node_prior_min_points,
                                     int sequence,
                                     const std::string& trigger_source,
                                     bool emit_deferred_event) {
  if (!delayed->enabled || delayed->anchor_requested ||
      delayed->attempt_in_flight || !delayed->pending_node_prior.valid) {
    return false;
  }
  if (emit_deferred_event) {
    WritePriorCheckEvent(output, sequence, "prior_check_requested",
                         trigger_source, *delayed, node_prior_min_points);
  }
  auto& request = delayed->pending_node_prior;
  if (static_cast<int>(request.source.points.size()) < node_prior_min_points) {
    if (emit_deferred_event) {
      WritePriorCheckEvent(output, sequence, "prior_check_deferred",
                           trigger_source, *delayed, node_prior_min_points,
                           "insufficient_points");
    }
    return false;
  }
  const auto request_snapshot = request;
  request.valid = false;
  delayed->node_prior_attempted = true;
  delayed->localization_failed = false;
  delayed->active_tool_seq = request_snapshot.tool_seq;
  std::ofstream events(output + ".localization.events.jsonl", std::ios::app);
  events << "{\"sequence\":" << sequence
         << ",\"event\":\"early_relocalization_trigger\""
         << ",\"trigger\":\"agent_reported_node\""
         << ",\"trigger_source\":\"" << trigger_source << "\""
         << ",\"node_id\":\"" << request_snapshot.node_id << "\""
         << ",\"tool_seq\":" << request_snapshot.tool_seq
         << ",\"unique_points\":" << delayed->unique_points
         << ",\"node_prior_min_points\":" << node_prior_min_points
         << ",\"agent_local_pose_sequence\":"
         << request_snapshot.agent_local_pose_sequence
         << ",\"agent_local_x\":"
         << request_snapshot.agent_local_pose.translation().x()
         << ",\"agent_local_y\":"
         << request_snapshot.agent_local_pose.translation().y()
         << ",\"agent_local_yaw_rad\":"
         << YawOf(request_snapshot.agent_local_pose)
         << "}\n";
  Scan attempt_scan;
  attempt_scan.sequence = sequence;
  StartRelocalizationAttempt(output, attempt_scan, delayed,
                             "agent_reported_node", request_snapshot.node_id,
                             request_snapshot.prior_path,
                             request_snapshot.has_agent_local_pose,
                             request_snapshot.agent_local_pose,
                             request_snapshot.agent_local_pose_sequence,
                             &request_snapshot.source.points,
                             request_snapshot.source.representation,
                             &request_snapshot.source);
  return true;
}

bool QueueNodePriorRelocalization(const std::string& output,
                                  DelayedRelocalization* delayed,
                                  int node_prior_min_points, int sequence,
                                  int tool_seq, const std::string& node_id) {
  if (!delayed->enabled) return false;
  if (delayed->anchor_requested) {
    WritePriorCheckEvent(output, sequence, "prior_check_ignored",
                         "hab_report_memory_pos", *delayed,
                         node_prior_min_points, "anchoring_in_progress");
    return false;
  }
  const auto prior = ReadReportedMemoryNodePrior(
      delayed->artifact_dir, tool_seq, node_id);
  if (!prior.has_value()) {
    WritePriorCheckEvent(output, sequence, "prior_check_deferred",
                         "hab_report_memory_pos", *delayed,
                         node_prior_min_points, "prior_snapshot_mismatch");
    return false;
  }
  habitat_cartographer::NodePriorRequest request;
  request.valid = true;
  request.tool_seq = tool_seq;
  request.node_id = node_id;
  request.prior_path = prior->path;
  request.source = SelectRelocalizationSource(*delayed, true);
  request.has_agent_local_pose = delayed->has_latest_local_pose;
  request.agent_local_pose_sequence = delayed->latest_local_pose_sequence;
  request.agent_local_pose = delayed->latest_local_pose;
  delayed->pending_node_prior = std::move(request);
  WritePriorCheckEvent(output, sequence,
                       delayed->attempt_in_flight ? "prior_check_queued"
                                                  : "prior_check_requested",
                       "hab_report_memory_pos", *delayed,
                       node_prior_min_points);
  return TryStartNodePriorRelocalization(
      output, delayed, node_prior_min_points, sequence,
      "hab_report_memory_pos", true);
}

std::string RelocalizerCommand() {
  const char* command_env = std::getenv("HAB_CARTOGRAPHER_RELOCALIZER_COMMAND");
  if (command_env != nullptr && std::string(command_env).size() > 0) {
    return std::string(command_env);
  }
  return "python -m supernav.backends.habitat.memory_slam.relocalize";
}

bool RelocalizerHasOpen3d() {
  const std::string command = RelocalizerCommand() + " --check-open3d";
  return std::system(command.c_str()) == 0;
}

void WriteSnapshotNodeManifest(const std::filesystem::path& snapshot,
                               int sequence,
                               const std::vector<MemoryNodeLabel>& labels) {
  const auto temporary = snapshot.string() + ".nodes.json.tmp";
  std::ofstream output(temporary);
  output << "{\"sequence\":" << sequence << ",\"nodes\":[";
  for (size_t index = 0; index < labels.size(); ++index) {
    if (index > 0) output << ',';
    // Memory-node names are agent-authored strings; emit a conservative JSON escape.
    std::string name;
    for (const char ch : labels[index].name) {
      if (ch == '"' || ch == '\\') {
        name.push_back('\\');
        name.push_back(ch);
      } else if (ch == '\n') {
        name += "\\n";
      } else if (ch == '\r') {
        name += "\\r";
      } else if (ch == '\t') {
        name += "\\t";
      } else {
        name.push_back(ch);
      }
    }
    output << "{\"order_id\":" << labels[index].order_id << ",\"name\":\""
           << name << "\",\"x\":" << labels[index].x << ",\"y\":"
           << labels[index].y << "}";
  }
  output << "]}\n";
  output.close();
  std::error_code error;
  std::filesystem::rename(temporary, snapshot.string() + ".nodes.json", error);
}

struct AmclOccupancyGrid {
  static constexpr double kResolution = 0.05;
  int width = 0;
  int height = 0;
  Eigen::Vector2d painter_origin = Eigen::Vector2d::Zero();
  std::vector<uint8_t> cells;  // 0 = free, 100 = occupied, 255 = unknown.
  std::vector<float> obstacle_distance_m;
};

struct AmclLaserScan {
  struct Beam {
    double angle_rad;
    double range_m;
  };
  std::vector<Beam> beams;
  double range_min_m = 0.15;
  double range_max_m = 8.0;
};

struct AmclParticle {
  double x;
  double y;
  double yaw;
  double log_weight;
};

bool IsGridCell(const AmclOccupancyGrid& grid, int col, int row) {
  return col >= 0 && col < grid.width && row >= 0 && row < grid.height;
}

size_t GridIndex(const AmclOccupancyGrid& grid, int col, int row) {
  return static_cast<size_t>(row) * grid.width + col;
}

std::vector<Eigen::Vector2f> OccupiedGridPoints(const AmclOccupancyGrid& grid) {
  std::vector<Eigen::Vector2f> points;
  for (int row = 0; row < grid.height; ++row) {
    for (int col = 0; col < grid.width; ++col) {
      if (grid.cells[GridIndex(grid, col, row)] != 100) continue;
      points.emplace_back((col - grid.painter_origin.x()) * grid.kResolution,
                          (grid.painter_origin.y() - row) * grid.kResolution);
    }
  }
  return points;
}

std::vector<Eigen::Vector2f> FreeGridPoints(const AmclOccupancyGrid& grid) {
  std::vector<Eigen::Vector2f> points;
  for (int row = 0; row < grid.height; ++row) {
    for (int col = 0; col < grid.width; ++col) {
      if (grid.cells[GridIndex(grid, col, row)] != 0) continue;
      points.emplace_back((col - grid.painter_origin.x()) * grid.kResolution,
                          (grid.painter_origin.y() - row) * grid.kResolution);
    }
  }
  return points;
}

void ToGridCell(const AmclOccupancyGrid& grid, double x, double y, int* col, int* row) {
  *col = static_cast<int>(std::lround(grid.painter_origin.x() + x / grid.kResolution));
  *row = static_cast<int>(std::lround(grid.painter_origin.y() - y / grid.kResolution));
}

bool BuildAmclOccupancyGrid(cartographer::mapping::MapBuilder* map_builder,
                            AmclOccupancyGrid* grid,
                            std::optional<int> trajectory_filter = std::nullopt,
                            bool trajectory_local_frame = false) {
  std::map<cartographer::mapping::SubmapId, cartographer::io::SubmapSlice> submaps;
  const auto global_poses = map_builder->pose_graph()->GetAllSubmapPoses();
  const auto local_from_global =
      trajectory_filter.has_value() && trajectory_local_frame
          ? TrajectoryFromGlobal(map_builder, *trajectory_filter)
          : cartographer::transform::Rigid3d::Identity();
  for (const auto& entry : global_poses) {
    const auto& id = entry.id;
    if (trajectory_filter.has_value() && id.trajectory_id != *trajectory_filter) {
      continue;
    }
    cartographer::mapping::proto::SubmapQuery::Response response;
    if (!map_builder->SubmapToProto(id, &response).empty() || response.textures_size() == 0) continue;
    const auto& texture = response.textures(0);
    auto pixels = cartographer::io::UnpackTextureData(
        texture.cells(), texture.width(), texture.height());
    auto& slice = submaps[id];
    slice.width = texture.width();
    slice.height = texture.height();
    slice.version = response.submap_version();
    slice.resolution = texture.resolution();
    slice.slice_pose = cartographer::transform::ToRigid3(texture.slice_pose());
    slice.pose = trajectory_local_frame ? local_from_global * entry.data.pose
                                        : entry.data.pose;
    slice.surface = cartographer::io::DrawTexture(
        pixels.intensity, pixels.alpha, slice.width, slice.height, &slice.cairo_data);
  }
  if (submaps.empty()) return false;
  auto mosaic = cartographer::io::PaintSubmapSlices(submaps, AmclOccupancyGrid::kResolution);
  grid->width = cairo_image_surface_get_width(mosaic.surface.get());
  grid->height = cairo_image_surface_get_height(mosaic.surface.get());
  if (grid->width <= 0 || grid->height <= 0) return false;
  grid->painter_origin = mosaic.origin.cast<double>();
  cairo_surface_flush(mosaic.surface.get());
  const auto* pixels = reinterpret_cast<const uint32_t*>(
      cairo_image_surface_get_data(mosaic.surface.get()));
  grid->cells.assign(static_cast<size_t>(grid->width) * grid->height, 255);
  std::vector<Eigen::Vector2i> queue;
  queue.reserve(grid->cells.size());
  grid->obstacle_distance_m.assign(grid->cells.size(), std::numeric_limits<float>::infinity());
  for (int row = 0; row < grid->height; ++row) {
    for (int col = 0; col < grid->width; ++col) {
      const uint32_t pixel = pixels[GridIndex(*grid, col, row)];
      const uint8_t intensity = static_cast<uint8_t>((pixel >> 16) & 0xff);
      const uint8_t observed = static_cast<uint8_t>((pixel >> 8) & 0xff);
      if (observed == 0) continue;
      const bool occupied = intensity < 112;
      grid->cells[GridIndex(*grid, col, row)] = occupied ? 100 : 0;
      if (occupied) {
        grid->obstacle_distance_m[GridIndex(*grid, col, row)] = 0.f;
        queue.emplace_back(col, row);
      }
    }
  }
  // A multi-source distance propagation creates the AMCL likelihood field.
  // Distances beyond the sensor model's useful radius are intentionally capped.
  constexpr int kOffset[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
  for (size_t cursor = 0; cursor < queue.size(); ++cursor) {
    const auto cell = queue[cursor];
    const float distance = grid->obstacle_distance_m[GridIndex(*grid, cell.x(), cell.y())];
    if (distance >= 2.f) continue;
    for (const auto& offset : kOffset) {
      const int next_col = cell.x() + offset[0];
      const int next_row = cell.y() + offset[1];
      if (!IsGridCell(*grid, next_col, next_row)) continue;
      const size_t next = GridIndex(*grid, next_col, next_row);
      const float candidate = distance + AmclOccupancyGrid::kResolution;
      if (candidate >= grid->obstacle_distance_m[next]) continue;
      grid->obstacle_distance_m[next] = candidate;
      queue.emplace_back(next_col, next_row);
    }
  }
  return !queue.empty();
}

RelocalizationSource SelectRelocalizationSource(
    const DelayedRelocalization& delayed, bool sparse_node_prior) {
  const int default_min_points = sparse_node_prior ? 120 : 600;
  const int min_points = std::max(
      1, EnvInt(sparse_node_prior
                    ? "HAB_CARTOGRAPHER_RELOCALIZATION_SPARSE_OCCUPANCY_MIN_POINTS"
                    : "HAB_CARTOGRAPHER_RELOCALIZATION_OCCUPANCY_MIN_POINTS",
                default_min_points));
  return BuildRecentTrajectoryOccupancySource(delayed, min_points);
}

AmclLaserScan ToAmclLaserScan(const Scan& scan) {
  AmclLaserScan laser;
  laser.range_max_m = 8.;
  const size_t stride = std::max<size_t>(1, scan.ranges_m.size() / 120);
  for (size_t index = 0; index < scan.ranges_m.size(); index += stride) {
    const float range = scan.ranges_m[index];
    if (!std::isfinite(range) || range < laser.range_min_m || range > laser.range_max_m) continue;
    laser.beams.push_back({BeamAngle(scan, index), range});
  }
  return laser;
}

void WriteAmclMap(const AmclOccupancyGrid& grid, const std::string& output_path) {
  const std::string pgm_path = output_path + ".amcl.pgm";
  std::ofstream pgm(pgm_path, std::ios::binary);
  pgm << "P5\n" << grid.width << " " << grid.height << "\n255\n";
  for (const uint8_t cell : grid.cells) {
    const char pixel = cell == 255 ? static_cast<char>(205) :
                       cell == 100 ? static_cast<char>(0) : static_cast<char>(254);
    pgm.write(&pixel, 1);
  }
  std::ofstream yaml(output_path + ".amcl.yaml");
  yaml << "image: " << std::filesystem::path(pgm_path).filename().string() << "\n"
       << "resolution: " << AmclOccupancyGrid::kResolution << "\n"
       // ROS map YAML origins refer to the bottom-left image pixel.
       << "origin: [" << -grid.painter_origin.x() * AmclOccupancyGrid::kResolution
       << ", " << (grid.height - 1 - grid.painter_origin.y()) * AmclOccupancyGrid::kResolution
       << ", 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n";
}

bool WriteRenderedAmclMap(const std::vector<uint8_t>& cells, int width, int height,
                          const Eigen::Vector2d& painter_origin,
                          double resolution, const std::string& output_path) {
  if (width <= 0 || height <= 0 ||
      cells.size() != static_cast<size_t>(width * height)) {
    return false;
  }
  const std::string pgm_path = output_path + ".amcl.pgm";
  const std::string temporary_pgm_path = pgm_path + ".tmp";
  {
    std::ofstream pgm(temporary_pgm_path, std::ios::binary);
    if (!pgm) return false;
    pgm << "P5\n" << width << " " << height << "\n255\n";
    pgm.write(reinterpret_cast<const char*>(cells.data()),
              static_cast<std::streamsize>(cells.size()));
    if (!pgm) return false;
  }
  std::error_code error;
  std::filesystem::rename(temporary_pgm_path, pgm_path, error);
  if (error) {
    std::filesystem::remove(temporary_pgm_path, error);
    return false;
  }
  const std::string yaml_path = output_path + ".amcl.yaml";
  const std::string temporary_yaml_path = yaml_path + ".tmp";
  {
    std::ofstream yaml(temporary_yaml_path);
    if (!yaml) return false;
    yaml << "image: " << std::filesystem::path(pgm_path).filename().string() << "\n"
         << "resolution: " << resolution << "\n"
         // ROS map YAML origins refer to the bottom-left image pixel.
         << "origin: [" << -painter_origin.x() * resolution
         << ", " << (height - 1 - painter_origin.y()) * resolution
         << ", 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n";
    if (!yaml) return false;
  }
  std::filesystem::rename(temporary_yaml_path, yaml_path, error);
  if (error) {
    std::filesystem::remove(temporary_yaml_path, error);
    return false;
  }
  return true;
}

void CopyAmclMapSnapshot(const std::string& output_path,
                         const std::filesystem::path& snapshot) {
  const auto source = std::filesystem::path(output_path + ".amcl.pgm");
  if (!std::filesystem::exists(source)) return;
  auto target = snapshot;
  target.replace_extension(".amcl.pgm");
  std::error_code error;
  std::filesystem::copy_file(source, target,
                             std::filesystem::copy_options::overwrite_existing,
                             error);
}

bool GlobalAmclLocalize(const AmclOccupancyGrid& grid, const AmclLaserScan& laser,
                        cartographer::transform::Rigid3d* pose,
                        double* confidence_margin, double* best_score,
                        double* effective_particles) {
  if (laser.beams.size() < 12) return false;
  std::vector<Eigen::Vector2i> free_cells;
  for (int row = 0; row < grid.height; row += 2) {
    for (int col = 0; col < grid.width; col += 2) {
      if (grid.cells[GridIndex(grid, col, row)] == 0) free_cells.emplace_back(col, row);
    }
  }
  if (free_cells.empty()) return false;
  // Global localization deliberately has no initial-pose distribution. This is
  // AMCL's uniform free-space initialization, made deterministic for audits.
  constexpr size_t kParticleCount = 30000;
  std::mt19937 generator(0x414d434c);
  std::uniform_int_distribution<size_t> cell_distribution(0, free_cells.size() - 1);
  std::uniform_real_distribution<double> yaw_distribution(-M_PI, M_PI);
  std::vector<AmclParticle> particles;
  particles.reserve(kParticleCount);
  constexpr double kZHit = 0.95;
  constexpr double kZRand = 0.05;
  constexpr double kSigmaHit = 0.20;
  constexpr double kMaxDistance = 2.0;
  double max_log_weight = -std::numeric_limits<double>::infinity();
  for (size_t particle_index = 0; particle_index < kParticleCount; ++particle_index) {
    const auto cell = free_cells[cell_distribution(generator)];
    const double x = (cell.x() - grid.painter_origin.x()) * grid.kResolution;
    const double y = (grid.painter_origin.y() - cell.y()) * grid.kResolution;
    const double yaw = yaw_distribution(generator);
    double log_weight = 0.;
    for (const auto& beam : laser.beams) {
      int col = 0;
      int row = 0;
      ToGridCell(grid, x + beam.range_m * std::cos(yaw + beam.angle_rad),
                 y + beam.range_m * std::sin(yaw + beam.angle_rad), &col, &row);
      double distance = kMaxDistance;
      if (IsGridCell(grid, col, row)) {
        distance = std::min<double>(kMaxDistance, grid.obstacle_distance_m[GridIndex(grid, col, row)]);
      }
      const double likelihood = kZHit * std::exp(-0.5 * distance * distance /
                                                   (kSigmaHit * kSigmaHit)) +
                                kZRand / laser.range_max_m;
      log_weight += std::log(likelihood);
    }
    particles.push_back({x, y, yaw, log_weight});
    max_log_weight = std::max(max_log_weight, log_weight);
  }
  double total_weight = 0.;
  double squared_weight = 0.;
  AmclParticle best = particles.front();
  for (const auto& particle : particles) {
    const double weight = std::exp(particle.log_weight - max_log_weight);
    total_weight += weight;
    squared_weight += weight * weight;
    if (particle.log_weight > best.log_weight) best = particle;
  }
  *effective_particles = total_weight * total_weight / squared_weight;
  *best_score = best.log_weight;
  // Estimate ambiguity only between spatially distinct modes, not adjacent
  // particles from the same global-localization basin.
  double alternative_log_weight = -std::numeric_limits<double>::infinity();
  for (const auto& particle : particles) {
    const double position_delta = std::hypot(particle.x - best.x, particle.y - best.y);
    const double yaw_delta = std::abs(std::remainder(particle.yaw - best.yaw, 2. * M_PI));
    if (position_delta > 0.75 || yaw_delta > 20. * M_PI / 180.) {
      alternative_log_weight = std::max(alternative_log_weight, particle.log_weight);
    }
  }
  *confidence_margin = best.log_weight - alternative_log_weight;
  *pose = cartographer::transform::Rigid3d(
      {best.x, best.y, 0.}, Eigen::AngleAxisd(best.yaw, Eigen::Vector3d::UnitZ()));
  // A first scan is accepted only if a separate global mode is substantially
  // less likely. Otherwise Cartographer must not be anchored to a guess.
  return std::isfinite(alternative_log_weight) && *confidence_margin >= 3.0 &&
         *effective_particles <= 2500.;
}

bool GlobalAmclRelocalize(cartographer::mapping::MapBuilder* map_builder,
                          const Scan& scan, const std::string& output_path,
                          cartographer::transform::Rigid3d* pose,
                          double* confidence_margin, double* best_score,
                          double* effective_particles) {
  AmclOccupancyGrid grid;
  if (!BuildAmclOccupancyGrid(map_builder, &grid)) return false;
  WriteAmclMap(grid, output_path);
  const AmclLaserScan laser = ToAmclLaserScan(scan);
  return GlobalAmclLocalize(grid, laser, pose, confidence_margin, best_score,
                             effective_particles);
}

bool RenderCartographerOccupancy(cartographer::mapping::MapBuilder* map_builder,
                                     int local_trajectory_id,
                                    int reference_trajectory_id,
                                    const cartographer::transform::Rigid3d& local_pose,
                                   const std::vector<Eigen::Vector2f>& scan_matched_trace,
                                   const std::string& output_path,
                                    Eigen::Vector2f* agent_pixel,
                                    std::vector<MemoryNodeLabel>* rendered_labels,
                                    bool draw_agent = true,
                                    const cartographer::transform::Rigid3d* old_from_local_override = nullptr) {
  std::map<cartographer::mapping::SubmapId, cartographer::io::SubmapSlice> submaps;
  const auto global_poses = map_builder->pose_graph()->GetAllSubmapPoses();
  const bool use_local_pose_override =
      old_from_local_override != nullptr && local_trajectory_id != reference_trajectory_id;
  const auto current_global_from_local =
      map_builder->pose_graph()->GetLocalToGlobalTransform(local_trajectory_id);
  const auto global_from_local_override =
      use_local_pose_override
          ? map_builder->pose_graph()->GetLocalToGlobalTransform(reference_trajectory_id) *
                *old_from_local_override
          : current_global_from_local;
  int rendered_reference_submaps = 0;
  int rendered_active_submaps = 0;
  int skipped_other_submaps = 0;
  for (const auto& entry : global_poses) {
    const auto& submap_id = entry.id;
    if (submap_id.trajectory_id != local_trajectory_id &&
        submap_id.trajectory_id != reference_trajectory_id) {
      ++skipped_other_submaps;
      continue;
    }
    cartographer::mapping::proto::SubmapQuery::Response candidate;
    if (!map_builder->SubmapToProto(submap_id, &candidate).empty() || candidate.textures_size() == 0) continue;
    const auto& texture = candidate.textures(0);
    auto pixels = cartographer::io::UnpackTextureData(
        texture.cells(), texture.width(), texture.height());
    auto& slice = submaps[submap_id];
    slice.width = texture.width();
    slice.height = texture.height();
    slice.version = candidate.submap_version();
    slice.resolution = texture.resolution();
    slice.slice_pose = cartographer::transform::ToRigid3(texture.slice_pose());
    if (use_local_pose_override && submap_id.trajectory_id == local_trajectory_id) {
      // The accepted relocalization transform is authoritative for display and
      // planning. Re-express active post-handoff submaps in the frozen map frame
      // so retrieve scans become part of the visible updated occupancy map.
      const auto local_from_submap =
          current_global_from_local.inverse() * entry.data.pose;
      slice.pose = global_from_local_override * local_from_submap;
      ++rendered_active_submaps;
    } else {
      slice.pose = entry.data.pose;
      if (submap_id.trajectory_id == local_trajectory_id) {
        ++rendered_active_submaps;
      } else {
        ++rendered_reference_submaps;
      }
    }
    slice.surface = cartographer::io::DrawTexture(
        pixels.intensity, pixels.alpha, slice.width, slice.height, &slice.cairo_data);
  }
  if (submaps.empty()) return false;

  constexpr double kResolution = 0.05;
  auto mosaic = cartographer::io::PaintSubmapSlices(submaps, kResolution);
  const int width = cairo_image_surface_get_width(mosaic.surface.get());
  const int height = cairo_image_surface_get_height(mosaic.surface.get());
  if (width <= 0 || height <= 0) return false;

  // Cartographer's painter encodes intensity in red and observation alpha in
  // green. Convert that internal diagnostic palette to a normal occupancy map.
  cairo_surface_flush(mosaic.surface.get());
  const auto* source = reinterpret_cast<const uint32_t*>(
      cairo_image_surface_get_data(mosaic.surface.get()));
  constexpr uint32_t kUnknownGray = 0xffcdcdcdu;
  std::vector<uint32_t> grayscale_pixels(width * height, kUnknownGray);
  std::vector<uint8_t> raw_class_pixels(width * height, 205);
  for (size_t index = 0; index < grayscale_pixels.size(); ++index) {
    const uint32_t pixel = source[index];
    const uint8_t intensity = static_cast<uint8_t>((pixel >> 16) & 0xff);
    const uint8_t observed = static_cast<uint8_t>((pixel >> 8) & 0xff);
    if (observed == 0) continue;
    // Cartographer textures encode probability near the neutral value 128.
    // Keep that broad range visible: the previous 12x expansion incorrectly
    // turned weak/free-space evidence around the scan origin nearly black.
    const uint8_t shade = static_cast<uint8_t>(
        255 - std::min(210, std::max(0, (128 - static_cast<int>(intensity)) * 8)));
    grayscale_pixels[index] = 0xff000000u | (static_cast<uint32_t>(shade) << 16) |
                              (static_cast<uint32_t>(shade) << 8) | shade;
    raw_class_pixels[index] = intensity < 112 ? 0 : 254;
  }
  auto grayscale_surface = cairo_image_surface_create_for_data(
      reinterpret_cast<unsigned char*>(grayscale_pixels.data()), CAIRO_FORMAT_RGB24,
      width, height, width * sizeof(uint32_t));

  // Present Cartographer's X-forward/Y-left map in Habitat's X-right/Z-down
  // top-down convention so this panel matches the evaluator trajectory audit.
  // Habitat world-right maps to negative Cartographer Y, so preserve Cairo's
  // image-row direction on display X instead of mirroring that horizontal axis.
  const int display_width = height;
  const int display_height = width;
  std::vector<uint32_t> display_pixels(display_width * display_height, kUnknownGray);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int display_x = y;
      const int display_y = width - 1 - x;
      display_pixels[display_y * display_width + display_x] =
          grayscale_pixels[y * width + x];
    }
  }
  auto display_surface = cairo_image_surface_create_for_data(
      reinterpret_cast<unsigned char*>(display_pixels.data()), CAIRO_FORMAT_RGB24,
      display_width, display_height, display_width * sizeof(uint32_t));

  const auto global_pose = global_from_local_override * local_pose;
  // PaintSubmapSlices reports the top-left pixel in map-frame coordinates.
  const double agent_x = global_pose.translation().x() / kResolution + mosaic.origin.x();
  // Cairo image rows increase downward while Cartographer map Y increases up.
  const double agent_y = mosaic.origin.y() - global_pose.translation().y() / kResolution;
  const double display_agent_x = agent_y;
  const double display_agent_y = width - 1 - agent_x;
  if (agent_pixel != nullptr) {
    *agent_pixel = {static_cast<float>(display_agent_x),
                    static_cast<float>(display_agent_y)};
  }
  const auto rotation = global_pose.rotation().toRotationMatrix();
  const double agent_yaw = std::atan2(rotation(1, 0), rotation(0, 0));
  // Keep the current scan-matched pose visible even when scan-only SLAM has
  // drifted beyond the probability-grid submap bounds.
  constexpr double kMarkerMargin = 24.;
  const double left_padding = std::max(0., kMarkerMargin - display_agent_x);
  const double top_padding = std::max(0., kMarkerMargin - display_agent_y);
  const double right_padding = std::max(0., display_agent_x - display_width + kMarkerMargin);
  const double bottom_padding = std::max(0., display_agent_y - display_height + kMarkerMargin);
  const int canvas_width = static_cast<int>(std::ceil(left_padding + display_width + right_padding));
  const int canvas_height = static_cast<int>(std::ceil(top_padding + display_height + bottom_padding));
  constexpr int kOutputSize = 1000;
  auto output_surface = cairo_image_surface_create(CAIRO_FORMAT_RGB24, kOutputSize, kOutputSize);
  auto* cr = cairo_create(output_surface);
  cairo_set_source_rgb(cr, 205. / 255., 205. / 255., 205. / 255.);
  cairo_paint(cr);
  const double scale = std::min(
      static_cast<double>(kOutputSize) / canvas_width,
      static_cast<double>(kOutputSize) / canvas_height);
  const double offset_x = (kOutputSize - canvas_width * scale) / 2.;
  const double offset_y = (kOutputSize - canvas_height * scale) / 2.;
  cairo_save(cr);
  cairo_translate(cr, offset_x, offset_y);
  cairo_scale(cr, scale, scale);
  cairo_set_source_surface(cr, display_surface, left_padding, top_padding);
  cairo_paint(cr);
  cairo_restore(cr);

  // The trace contains only Cartographer scan-matched poses transformed into
  // the same global frame as the painted submaps; simulator odometry is never
  // drawn on this map.
  if (scan_matched_trace.size() > 1) {
    cairo_set_source_rgba(cr, 0.02, 0.58, 0.72, 0.82);
    cairo_set_line_width(cr, 4.);
    bool started = false;
    for (const auto& pose : scan_matched_trace) {
      const double trace_x = mosaic.origin.y() - pose.y() / kResolution;
      const double trace_y = width - 1 - (pose.x() / kResolution + mosaic.origin.x());
      const double x = offset_x + (left_padding + trace_x) * scale;
      const double y = offset_y + (top_padding + trace_y) * scale;
      if (!started) {
        cairo_move_to(cr, x, y);
        started = true;
      } else {
        cairo_line_to(cr, x, y);
      }
    }
    cairo_stroke(cr);
  }

  // The sidecar is atomically published by memory-node mutations. Drawing it
  // here makes canonical occupancy.png and every occupancy_step<N>.png match.
  // Size labels by their physical map footprint, not the fixed 1000 px canvas.
  const double label_size = std::max(16., 1.5 * scale / kResolution);
  for (const auto& node : ReadMemoryNodeLabels(output_path)) {
    const double display_x = mosaic.origin.y() - node.y / kResolution;
    const double display_y = width - 1 - (node.x / kResolution + mosaic.origin.x());
    const double anchor_x = offset_x + (left_padding + display_x) * scale;
    const double anchor_y = offset_y + (top_padding + display_y) * scale;
    if (anchor_x < 0. || anchor_x >= kOutputSize || anchor_y < 0. || anchor_y >= kOutputSize) {
      continue;
    }
    if (rendered_labels != nullptr) rendered_labels->push_back(node);
    const std::string label = std::to_string(node.order_id);
    cairo_select_font_face(cr, "Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, label_size);
    cairo_text_extents_t extents;
    cairo_text_extents(cr, label.c_str(), &extents);
    cairo_move_to(cr, anchor_x - (extents.width / 2. + extents.x_bearing),
                  anchor_y - (extents.height / 2. + extents.y_bearing));
    cairo_text_path(cr, label.c_str());
    cairo_set_source_rgb(cr, 1., 1., 1.);
    cairo_set_line_width(cr, 7.);
    cairo_stroke_preserve(cr);
    cairo_set_source_rgb(cr, 0., 0., 0.);
    cairo_fill(cr);
  }

  // The pose is Cartographer local-SLAM output transformed into the same
  // global map frame as the composed probability-grid submaps.
  if (draw_agent) cairo_translate(cr, offset_x + (left_padding + display_agent_x) * scale,
                                  offset_y + (top_padding + display_agent_y) * scale);
  // The position transform maps Cartographer X-forward/Y-left into the
  // evaluator top-down frame. Cairo's positive rotation is clockwise on its
  // Y-down canvas, so invert the Cartographer counterclockwise yaw as well.
  if (draw_agent) cairo_rotate(cr, -agent_yaw - M_PI / 2.);
  // A high-contrast outline keeps the scan-matched pose legible after the
  // agent-facing map is downsampled to 256 px.
  if (draw_agent) {
    cairo_set_source_rgb(cr, 0.95, 0.78, 0.02);
    cairo_move_to(cr, 30., 0.);
    cairo_line_to(cr, -18., 16.);
    cairo_line_to(cr, -18., -16.);
    cairo_close_path(cr);
    cairo_fill(cr);
    cairo_arc(cr, 0., 0., 11., 0., 2. * M_PI);
    cairo_fill(cr);
    cairo_set_source_rgb(cr, 0.85, 0.05, 0.05);
    cairo_move_to(cr, 24., 0.);
    cairo_line_to(cr, -14., 12.);
    cairo_line_to(cr, -14., -12.);
    cairo_close_path(cr);
    cairo_fill(cr);
    cairo_arc(cr, 0., 0., 7., 0., 2. * M_PI);
    cairo_fill(cr);
  }
  cairo_destroy(cr);
  cairo_surface_destroy(display_surface);
  cairo_surface_destroy(grayscale_surface);
  std::ofstream render_metadata(output_path + ".render.json");
  render_metadata << "{\"resolution\":" << kResolution
                  << ",\"mosaic_origin_x\":" << mosaic.origin.x()
                  << ",\"mosaic_origin_y\":" << mosaic.origin.y()
                  << ",\"source_width\":" << width
                  << ",\"source_height\":" << height
                  << ",\"display_width\":" << display_width
                  << ",\"display_height\":" << display_height
                  << ",\"left_padding\":" << left_padding
                  << ",\"top_padding\":" << top_padding
                  << ",\"scale\":" << scale
                  << ",\"offset_x\":" << offset_x
                  << ",\"offset_y\":" << offset_y
                  << ",\"output_size\":" << kOutputSize
                  << ",\"local_trajectory_id\":" << local_trajectory_id
                  << ",\"reference_trajectory_id\":" << reference_trajectory_id
                  << ",\"active_relocalized_submaps_painted\":"
                  << (use_local_pose_override ? "true" : "false")
                  << ",\"rendered_active_submaps\":" << rendered_active_submaps
                  << ",\"rendered_reference_submaps\":"
                  << rendered_reference_submaps
                  << ",\"skipped_other_submaps\":" << skipped_other_submaps
                  << "}\n";
  const std::string temporary_path = output_path + ".tmp";
  const auto status = cairo_surface_write_to_png(output_surface, temporary_path.c_str());
  cairo_surface_destroy(output_surface);
  if (status != CAIRO_STATUS_SUCCESS) return false;
  std::error_code error;
  std::filesystem::rename(temporary_path, output_path, error);
  if (error) return false;
  if (!WriteRenderedAmclMap(raw_class_pixels, width, height,
                            mosaic.origin.cast<double>(), kResolution,
                            output_path)) {
    std::filesystem::remove(output_path + ".amcl.pgm", error);
    std::filesystem::remove(output_path + ".amcl.yaml", error);
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--capabilities") {
    std::cout
        << "{\"protocol\":\"supernav-cartographer-worker/v1\","
        << "\"authoritative_odometry\":true,"
        << "\"delayed_relocalization\":true,\"dual_stream\":true}\n";
    return 0;
  }
  if (argc != 3) {
    std::cerr << "usage: habitat_cartographer_worker <configuration_dir> <output_png>\n";
    return 2;
  }
  const std::string config_dir = argv[1];
  const char* dual_stream_env = std::getenv("HAB_CARTOGRAPHER_DUAL_STREAM");
  const bool dual_stream =
      dual_stream_env != nullptr && std::string(dual_stream_env) == "1";
  const char* planning_output_env = std::getenv("HAB_CARTOGRAPHER_PLANNING_OUTPUT");
  const std::string planning_output =
      planning_output_env != nullptr && std::string(planning_output_env).size() > 0
          ? std::string(planning_output_env)
          : "";
  // A replay replaces this session's derived mapper artifacts rather than
  // appending duplicate pose/update records from a prior worker invocation.
  std::filesystem::remove(std::string(argv[2]) + ".poses.jsonl");
  std::filesystem::remove(std::string(argv[2]) + ".updates.jsonl");
  if (dual_stream && !planning_output.empty()) {
    std::filesystem::remove(planning_output + ".poses.jsonl");
    std::filesystem::remove(planning_output + ".updates.jsonl");
  }
  auto resolver = std::make_unique<cartographer::common::ConfigurationFileResolver>(
      std::vector<std::string>{config_dir});
  const bool use_odometry = std::getenv("HAB_CARTOGRAPHER_USE_ODOMETRY") != nullptr &&
                             std::string(std::getenv("HAB_CARTOGRAPHER_USE_ODOMETRY")) == "1";
  const char* odometry_mode_env = std::getenv("HAB_CARTOGRAPHER_ODOMETRY_MODE");
  const std::string odometry_mode =
      odometry_mode_env == nullptr || std::string(odometry_mode_env).empty()
          ? "cartographer_input"
          : std::string(odometry_mode_env);
  const bool authoritative_odometry = odometry_mode == "authoritative";
  if (odometry_mode != "cartographer_input" && odometry_mode != "authoritative") {
    std::cerr << "HAB_CARTOGRAPHER_ODOMETRY_MODE must be cartographer_input or authoritative\n";
    return 2;
  }
  if (authoritative_odometry && !use_odometry) {
    std::cerr << "authoritative odometry requires HAB_CARTOGRAPHER_USE_ODOMETRY=1\n";
    return 2;
  }
  if (authoritative_odometry) {
    std::cerr << "authoritative odometry requires a Cartographer library built "
              << "from third_party/cartographer with the local trajectory "
              << "builder patch applied\n";
  }
  const char* load_state = std::getenv("HAB_CARTOGRAPHER_LOAD_STATE");
  const char* visual_load_state = std::getenv("HAB_CARTOGRAPHER_VISUAL_LOAD_STATE");
  if (visual_load_state != nullptr && std::string(visual_load_state).size() > 0) {
    load_state = visual_load_state;
  }
  const char* planning_load_state = std::getenv("HAB_CARTOGRAPHER_PLANNING_LOAD_STATE");
  const bool frozen_localization = load_state != nullptr && std::string(load_state).size() > 0;
  const bool planning_frozen_localization = dual_stream && planning_load_state != nullptr &&
                                            std::string(planning_load_state).size() > 0;
  const char* localization_mode = std::getenv("HAB_CARTOGRAPHER_LOCALIZATION_MODE");
  const bool delayed_relocalization = frozen_localization && localization_mode != nullptr &&
      std::string(localization_mode) == "delayed_fpfh_relocalization";
  const bool agent_guided_relocalization = EnvBool(
      "HAB_CARTOGRAPHER_AGENT_GUIDED_RELOCALIZATION_ENABLED", true);
  const bool static_map_no_relocalization =
      frozen_localization && localization_mode != nullptr &&
      std::string(localization_mode) == "static_map_no_relocalization";
  const bool use_online_correlative_scan_matching = EnvBool(
      "HAB_CARTOGRAPHER_USE_ONLINE_CORRELATIVE_SCAN_MATCHING", true);
  const int optimize_every_n_nodes = EnvInt(
      "HAB_CARTOGRAPHER_POSE_GRAPH_OPTIMIZE_EVERY_N_NODES",
      90);
  const double default_scan_match_cost = 0.01;
  const std::string translation_delta_cost_weight =
      std::to_string(EnvDouble("HAB_CARTOGRAPHER_RTCS_TRANSLATION_WEIGHT",
                               default_scan_match_cost));
  const std::string rotation_delta_cost_weight =
      std::to_string(EnvDouble("HAB_CARTOGRAPHER_RTCS_ROTATION_WEIGHT",
                               default_scan_match_cost));
  const double default_ceres_translation_weight = 10.0;
  const double default_ceres_rotation_weight = 40.0;
  const std::string ceres_translation_weight =
      std::to_string(EnvDouble("HAB_CARTOGRAPHER_CERES_TRANSLATION_WEIGHT",
                               default_ceres_translation_weight));
  const std::string ceres_rotation_weight =
      std::to_string(EnvDouble("HAB_CARTOGRAPHER_CERES_ROTATION_WEIGHT",
                               default_ceres_rotation_weight));
  const std::string authoritative_odometry_config =
      "TRAJECTORY_BUILDER_2D.authoritative_odometry = " +
      std::string(authoritative_odometry ? "true\n" : "false\n");
  const std::string config = R"(
include "map_builder.lua"
include "trajectory_builder.lua"
MAP_BUILDER.use_trajectory_builder_2d = true
POSE_GRAPH.optimize_every_n_nodes = )" + std::to_string(optimize_every_n_nodes) + R"(
TRAJECTORY_BUILDER_2D.use_imu_data = false
TRAJECTORY_BUILDER_2D.use_odometry = )" + std::string(use_odometry ? "true" : "false") + R"(
)" + authoritative_odometry_config + R"(
TRAJECTORY_BUILDER_2D.min_range = 0.15
TRAJECTORY_BUILDER_2D.max_range = 8.0
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = )" + std::string(
    use_online_correlative_scan_matching ? "true" : "false") + R"(
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.2
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.angular_search_window = 0.7
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.translation_delta_cost_weight = )" + translation_delta_cost_weight + R"(
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.rotation_delta_cost_weight = )" + rotation_delta_cost_weight + R"(
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.translation_weight = )" + ceres_translation_weight + R"(
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight = )" + ceres_rotation_weight + R"(
 TRAJECTORY_BUILDER.pure_localization = )" + std::string(
     frozen_localization && !delayed_relocalization ? "true" : "false") + R"(
return {map_builder = MAP_BUILDER, trajectory_builder = TRAJECTORY_BUILDER}
)";
  auto dictionary = cartographer::common::LuaParameterDictionary::NonReferenceCounted(
      config, std::move(resolver));
  auto map_dictionary = dictionary->GetDictionary("map_builder");
  auto trajectory_dictionary = dictionary->GetDictionary("trajectory_builder");
  auto map_options = cartographer::mapping::CreateMapBuilderOptions(map_dictionary.get());
  auto trajectory_options =
      cartographer::mapping::CreateTrajectoryBuilderOptions(trajectory_dictionary.get());
  cartographer::mapping::MapBuilder map_builder(map_options);
  std::unique_ptr<cartographer::mapping::MapBuilder> planning_map_builder;
  if (dual_stream) {
    planning_map_builder =
        std::make_unique<cartographer::mapping::MapBuilder>(map_options);
  }
  int explore_trajectory_id = -1;
  int planning_explore_trajectory_id = -1;
  if (frozen_localization) {
    const auto id_remapping = map_builder.LoadStateFromFile(load_state, true);
    if (id_remapping.empty()) {
      std::cerr << "Cartographer state contains no exploration trajectory\n";
      return 1;
    }
    explore_trajectory_id = id_remapping.begin()->second;
  }
  if (planning_frozen_localization) {
    if (planning_output.empty()) {
      std::cerr << "dual Cartographer planning output is required\n";
      return 2;
    }
    const auto id_remapping =
        planning_map_builder->LoadStateFromFile(planning_load_state, true);
    if (id_remapping.empty()) {
      std::cerr << "Planning Cartographer state contains no exploration trajectory\n";
      return 1;
    }
    planning_explore_trajectory_id = id_remapping.begin()->second;
  }
  DelayedRelocalization delayed;
  delayed.enabled = delayed_relocalization;
  delayed.artifact_dir = std::filesystem::path(argv[2]).parent_path();
  delayed.source_window_length_m = std::max(
      0.1, EnvDouble("HAB_CARTOGRAPHER_RELOCALIZATION_SOURCE_WINDOW_M", 8.0));
  const int node_prior_min_points = EnvInt(
      "HAB_CARTOGRAPHER_NODE_PRIOR_MIN_POINTS", 120);
  if (delayed.enabled) {
    if (!RelocalizerHasOpen3d()) {
      delayed.localization_failed = true;
      WriteLocalizationStatus(argv[2], "error", delayed, "open3d_missing");
      std::cerr << "Cartographer delayed relocalization requires Open3D in "
                << "HAB_CARTOGRAPHER_RELOCALIZER_COMMAND\n";
      return 1;
    }
    AmclOccupancyGrid old_grid;
    if (!BuildAmclOccupancyGrid(&map_builder, &old_grid)) {
      std::cerr << "failed to build old-map relocalization cloud\n";
      return 1;
    }
    delayed.target_path = delayed.artifact_dir / "relocalization_old_global.xy";
    delayed.target_free_path =
        delayed.artifact_dir / "relocalization_old_global_free.xy";
    WriteXYCloud(delayed.target_path, OccupiedGridPoints(old_grid));
    WriteXYCloud(delayed.target_free_path, FreeGridPoints(old_grid));
    WriteLocalizationStatus(argv[2], "collecting", delayed);
  }
  std::set<cartographer::mapping::TrajectoryBuilderInterface::SensorId> sensors = {
      {cartographer::mapping::TrajectoryBuilderInterface::SensorId::SensorType::RANGE, "depth_scan"}};
  if (use_odometry) {
    sensors.insert({cartographer::mapping::TrajectoryBuilderInterface::SensorId::SensorType::ODOMETRY,
                    "odometry"});
  }
  std::vector<std::pair<Eigen::Vector3f, Scan>> rendered_scans;
  std::vector<Eigen::Vector2f> scan_matched_trace;
  std::mutex rendered_scans_mutex;
  std::vector<std::pair<Eigen::Vector3f, Scan>> planning_rendered_scans;
  std::vector<Eigen::Vector2f> planning_scan_matched_trace;
  std::mutex planning_rendered_scans_mutex;
  std::mutex planning_localization_mutex;
  bool planning_localized_from_visual = !planning_frozen_localization;
  cartographer::transform::Rigid3d planning_requested_old_from_local =
      cartographer::transform::Rigid3d::Identity();
  auto active_trajectory_id = std::make_shared<std::atomic<int>>(-1);
  auto planning_active_trajectory_id = std::make_shared<std::atomic<int>>(-1);
  auto make_local_slam_callback = [&](bool callback_is_planning) {
    return
      [&map_builder, &planning_map_builder, active_trajectory_id, planning_active_trajectory_id,
       &explore_trajectory_id, &planning_explore_trajectory_id, &rendered_scans,
       &planning_rendered_scans, &scan_matched_trace,
       &planning_scan_matched_trace, &rendered_scans_mutex,
       &planning_rendered_scans_mutex, &planning_localization_mutex,
       &planning_localized_from_visual, &planning_requested_old_from_local,
       &delayed, node_prior_min_points, dual_stream, planning_frozen_localization,
       callback_is_planning, output = std::string(argv[2]), planning_output](
          int callback_trajectory_id, cartographer::common::Time time,
          cartographer::transform::Rigid3d pose, cartographer::sensor::RangeData,
          std::unique_ptr<const cartographer::mapping::TrajectoryBuilderInterface::InsertionResult>) {
        const int trajectory_id = active_trajectory_id->load();
        const int planning_trajectory_id = planning_active_trajectory_id->load();
        const bool planning_callback = callback_is_planning;
        if (planning_callback) {
          if (!dual_stream || callback_trajectory_id != planning_trajectory_id) return;
        } else if (callback_trajectory_id != trajectory_id) {
          return;
        }
        auto* active_map_builder =
            planning_callback ? planning_map_builder.get() : &map_builder;
        auto& active_rendered_scans =
            planning_callback ? planning_rendered_scans : rendered_scans;
        auto& active_scan_matched_trace =
            planning_callback ? planning_scan_matched_trace : scan_matched_trace;
        auto& active_rendered_scans_mutex =
            planning_callback ? planning_rendered_scans_mutex : rendered_scans_mutex;
        const std::string active_output =
            planning_callback && !planning_output.empty() ? planning_output : output;
        const int reference_trajectory_id =
            planning_callback ? planning_explore_trajectory_id : explore_trajectory_id;
        const int active_callback_trajectory_id =
            planning_callback ? planning_trajectory_id : trajectory_id;
        std::lock_guard<std::mutex> lock(active_rendered_scans_mutex);
        if (active_rendered_scans.empty()) return;
        const double timestamp_s = cartographer::common::ToUniversal(time) / 10000000.0;
        const auto item = std::min_element(
            active_rendered_scans.begin(), active_rendered_scans.end(),
            [timestamp_s](const auto& left, const auto& right) {
              return std::abs(left.second.timestamp_s - timestamp_s) <
                     std::abs(right.second.timestamp_s - timestamp_s);
            });
        const auto overlay_pose = pose;
        std::lock_guard<std::mutex> delayed_lock(delayed.mutex);
        if (!planning_callback && delayed.enabled && !delayed.anchor_requested &&
            (!delayed.localization_failed || delayed.localized)) {
          AppendScanMatchedScanPoints(item->second, overlay_pose, &delayed);
          if (delayed.pending_node_prior.valid &&
              static_cast<int>(delayed.pending_node_prior.source.points.size()) <
                  node_prior_min_points) {
            delayed.pending_node_prior.source =
                SelectRelocalizationSource(delayed, true);
          }
          if (!delayed.localized && !delayed.attempt_in_flight) {
            WriteLocalizationStatus(active_output, "collecting", delayed);
          }
          TryStartNodePriorRelocalization(active_output, &delayed,
                                          node_prior_min_points,
                                          item->second.sequence,
                                          "scan_processing", false);
          if (!delayed.localized && !delayed.attempt_in_flight &&
              delayed.unique_points >= delayed.next_attempt_points) {
            const auto source = SelectRelocalizationSource(delayed, false);
            StartRelocalizationAttempt(active_output, item->second, &delayed,
                                       "accumulated_points", "", {}, false,
                                       cartographer::transform::Rigid3d::Identity(),
                                       -1, &source.points, source.representation,
                                       &source);
          }
        }
        if (!planning_callback && delayed.enabled && delayed.anchor_requested) {
          constexpr double kTranslationToleranceM = 0.30;
          constexpr double kYawToleranceRad = 5. * M_PI / 180.;
          constexpr int kMaxAnchorChecks = 20;
          const auto expected_old_from_pose =
              delayed.requested_old_from_local *
              delayed.active_local_from_stream_odometry * OdomPose(item->second);
          const AnchorError anchor_error = TransformAnchorError(
              delayed.requested_old_from_local, overlay_pose, expected_old_from_pose);
          if (anchor_error.translation_m > kTranslationToleranceM ||
              anchor_error.yaw_rad > kYawToleranceRad) {
            const AnchorError pose_graph_anchor_error = PoseGraphAnchorError(
                &map_builder, trajectory_id, explore_trajectory_id, overlay_pose,
                expected_old_from_pose);
            ++delayed.anchor_check_count;
            std::ofstream events(active_output + ".localization.events.jsonl", std::ios::app);
            events << "{\"event\":\"anchor_check\",\"translation_error_m\":"
                   << anchor_error.translation_m << ",\"yaw_error_deg\":"
                   << anchor_error.yaw_rad * 180. / M_PI
                   << ",\"check_count\":" << delayed.anchor_check_count
                   << ",\"current_x\":" << anchor_error.current.translation().x()
                   << ",\"current_y\":" << anchor_error.current.translation().y()
                   << ",\"current_yaw_rad\":" << YawOf(anchor_error.current)
                   << ",\"requested_x\":"
                   << expected_old_from_pose.translation().x()
                   << ",\"requested_y\":"
                   << expected_old_from_pose.translation().y()
                   << ",\"requested_yaw_rad\":"
                   << YawOf(expected_old_from_pose)
                   << ",\"pose_graph_current_x\":"
                   << pose_graph_anchor_error.current.translation().x()
                   << ",\"pose_graph_current_y\":"
                   << pose_graph_anchor_error.current.translation().y()
                   << ",\"pose_graph_current_yaw_rad\":"
                   << YawOf(pose_graph_anchor_error.current)
                   << ",\"local_x\":" << overlay_pose.translation().x()
                   << ",\"local_y\":" << overlay_pose.translation().y()
                   << ",\"local_yaw_rad\":" << YawOf(overlay_pose)
                   << "}\n";
            if (delayed.anchor_check_count >= kMaxAnchorChecks) {
              delayed.localization_failed = !delayed.correction_anchor;
              delayed.anchor_requested = false;
              events << "{\"event\":\"anchor_failed\",\"translation_error_m\":"
                     << anchor_error.translation_m << ",\"yaw_error_deg\":"
                     << anchor_error.yaw_rad * 180. / M_PI
                     << ",\"check_count\":" << delayed.anchor_check_count << "}\n";
              WriteLocalizationStatus(
                  active_output,
                  delayed.correction_anchor ? "localized" : "ambiguous",
                  delayed, "anchor_error");
              delayed.correction_anchor_failed = delayed.correction_anchor;
              delayed.correction_anchor = false;
            } else {
              WriteLocalizationStatus(active_output, "anchoring", delayed);
            }
            return;
          }
          delayed.localized = true;
          delayed.anchor_requested = false;
          delayed.correction_anchor_succeeded = delayed.correction_anchor;
          delayed.correction_anchor = false;
          delayed.anchor_check_count = 0;
          ResetRelocalizationSource(&delayed);
          WriteLocalizationStatus(active_output, "localized", delayed);
        }
        if (!planning_callback && delayed.enabled && !delayed.localized) return;
        cartographer::transform::Rigid3d old_from_local_override =
            cartographer::transform::Rigid3d::Identity();
        const cartographer::transform::Rigid3d* old_from_local_override_ptr = nullptr;
        bool use_delayed_transform = !planning_callback && delayed.enabled && delayed.localized;
        if (use_delayed_transform) {
          old_from_local_override = delayed.requested_old_from_local;
          old_from_local_override_ptr = &old_from_local_override;
        }
        if (planning_callback && planning_frozen_localization) {
          std::lock_guard<std::mutex> planning_lock(planning_localization_mutex);
          if (!planning_localized_from_visual) return;
          use_delayed_transform = true;
          old_from_local_override = planning_requested_old_from_local;
          old_from_local_override_ptr = &old_from_local_override;
        }
        const auto global_pose =
            use_delayed_transform
                ? active_map_builder->pose_graph()->GetLocalToGlobalTransform(reference_trajectory_id) *
                      old_from_local_override * overlay_pose
                : active_map_builder->pose_graph()->GetLocalToGlobalTransform(active_callback_trajectory_id) *
                      overlay_pose;
        const auto global_rotation = global_pose.rotation().toRotationMatrix();
        item->first = {static_cast<float>(global_pose.translation().x()),
                       static_cast<float>(global_pose.translation().y()),
                       static_cast<float>(std::atan2(global_rotation(1, 0),
                                                    global_rotation(0, 0)))};
        active_scan_matched_trace.push_back({
            static_cast<float>(global_pose.translation().x()),
            static_cast<float>(global_pose.translation().y())});
        if (!item->second.publish_snapshot) return;
        Eigen::Vector2f agent_pixel;
        std::vector<MemoryNodeLabel> rendered_labels;
        if (!RenderCartographerOccupancy(active_map_builder, active_callback_trajectory_id,
                                         reference_trajectory_id, overlay_pose,
                                         active_scan_matched_trace, active_output, &agent_pixel,
                                         &rendered_labels, false,
                                         old_from_local_override_ptr)) return;
        std::ofstream poses(active_output + ".poses.jsonl", std::ios::app);
        poses << "{\"sequence\":" << item->second.sequence
              << ",\"callback_timestamp_s\":" << timestamp_s
              << ",\"x\":" << item->first.x() << ",\"y\":" << item->first.y()
              << ",\"yaw_rad\":" << item->first.z()
              << ",\"odometry_x\":" << (item->second.has_odometry ? item->second.odometry_x : 0.)
              << ",\"odometry_y\":" << (item->second.has_odometry ? item->second.odometry_y : 0.)
              << ",\"odometry_yaw_rad\":"
              << (item->second.has_odometry ? item->second.odometry_yaw_rad : 0.)
              << ",\"map_pixel_x\":" << agent_pixel.x()
              << ",\"map_pixel_y\":" << agent_pixel.y() << "}\n";
        const int sequence = item->second.sequence;
        const auto snapshot = std::filesystem::path(active_output).parent_path() /
                              ("occupancy_step" + std::to_string(sequence) + ".png");
        std::filesystem::copy_file(active_output, snapshot,
                                   std::filesystem::copy_options::overwrite_existing);
        std::filesystem::copy_file(active_output + ".render.json", snapshot.string() + ".render.json",
                                   std::filesystem::copy_options::overwrite_existing);
        CopyAmclMapSnapshot(active_output, snapshot);
        WriteSnapshotNodeManifest(snapshot, sequence, rendered_labels);
        std::ofstream updates(active_output + ".updates.jsonl", std::ios::app);
        updates << "{\"sequence\":" << sequence
                << ",\"scan_count\":" << active_rendered_scans.size() << "}\n";
      };
  };
  auto add_trajectory = [&] {
    return map_builder.AddTrajectoryBuilder(
        sensors, trajectory_options, make_local_slam_callback(false));
  };
  auto add_planning_trajectory = [&] {
    return planning_map_builder->AddTrajectoryBuilder(
        sensors, trajectory_options, make_local_slam_callback(true));
  };
  int trajectory_id = add_trajectory();
  active_trajectory_id->store(trajectory_id);
  int planning_trajectory_id = -1;
  cartographer::mapping::TrajectoryBuilderInterface* planning_builder = nullptr;
  struct RetiredTrajectory {
    int id;
    cartographer::mapping::TrajectoryBuilderInterface* builder;
    cartographer::transform::Rigid3d local_from_stream_odometry;
  };
  std::vector<RetiredTrajectory> retired_trajectories;
  std::vector<RetiredTrajectory> retired_planning_trajectories;
  struct CorrectionCheckpoint {
    RetiredTrajectory visual;
    std::optional<RetiredTrajectory> planning;
    cartographer::transform::Rigid3d requested_old_from_local;
  };
  std::optional<CorrectionCheckpoint> correction_checkpoint;
  bool planning_rebase_active_odometry = false;
  cartographer::transform::Rigid3d planning_active_local_from_stream_odometry =
      cartographer::transform::Rigid3d::Identity();
  if (dual_stream) {
    planning_trajectory_id = add_planning_trajectory();
    planning_active_trajectory_id->store(planning_trajectory_id);
    planning_builder =
        planning_map_builder->GetTrajectoryBuilder(planning_trajectory_id);
  }
  bool localization_initialized = !frozen_localization;
  auto* builder = map_builder.GetTrajectoryBuilder(trajectory_id);
  bool rebase_active_odometry = false;
  cartographer::transform::Rigid3d active_local_from_stream_odometry =
      cartographer::transform::Rigid3d::Identity();
  auto finish_open_trajectories = [&] {
    for (const auto& retired : retired_trajectories) {
      map_builder.FinishTrajectory(retired.id);
    }
    retired_trajectories.clear();
    for (const auto& retired : retired_planning_trajectories) {
      planning_map_builder->FinishTrajectory(retired.id);
    }
    retired_planning_trajectories.clear();
    if (trajectory_id >= 0) {
      map_builder.FinishTrajectory(trajectory_id);
      trajectory_id = -1;
    }
    if (planning_trajectory_id >= 0) {
      planning_map_builder->FinishTrajectory(planning_trajectory_id);
      planning_trajectory_id = -1;
    }
  };
  auto submit_scan = [&](cartographer::mapping::TrajectoryBuilderInterface* target,
                         const Scan& submitted_scan,
                         bool record_for_render,
                         const cartographer::transform::Rigid3d& local_from_stream_odometry,
                         bool planning_record = false) {
    if (use_odometry) {
      const int64_t odometry_universal_time =
          static_cast<int64_t>(submitted_scan.timestamp_s * 10000000.0);
      cartographer::sensor::OdometryData odometry;
      odometry.pose = local_from_stream_odometry * OdomPose(submitted_scan);
      if (authoritative_odometry && odometry_universal_time > 0) {
        odometry.time =
            cartographer::common::FromUniversal(odometry_universal_time - 1);
        target->AddSensorData("odometry", odometry);
      }
      odometry.time = cartographer::common::FromUniversal(odometry_universal_time);
      target->AddSensorData("odometry", odometry);
    }
    if (record_for_render) {
      if (planning_record) {
        std::lock_guard<std::mutex> lock(planning_rendered_scans_mutex);
        planning_rendered_scans.push_back({Eigen::Vector3f::Zero(), submitted_scan});
      } else {
        std::lock_guard<std::mutex> lock(rendered_scans_mutex);
        rendered_scans.push_back({Eigen::Vector3f::Zero(), submitted_scan});
      }
    }
    const double hfov_rad = submitted_scan.hfov_deg * M_PI / 180.;
    if (std::abs(submitted_scan.hfov_deg - 360.) < 1e-6 &&
        submitted_scan.ranges_m.size() % 4 == 0) {
      cartographer::sensor::TimedPointCloudData point_cloud;
      point_cloud.time = cartographer::common::FromUniversal(
          static_cast<int64_t>(submitted_scan.timestamp_s * 10000000.0));
      point_cloud.origin = Eigen::Vector3f::Zero();
      for (size_t index = 0; index < submitted_scan.ranges_m.size(); ++index) {
        const float range = submitted_scan.ranges_m[index];
        if (!std::isfinite(range)) continue;
        const double angle = BeamAngle(submitted_scan, index);
        point_cloud.ranges.push_back({{
            static_cast<float>(range * std::cos(angle)),
            static_cast<float>(range * std::sin(angle)), 0.f}, 0.f});
      }
      target->AddSensorData("depth_scan", point_cloud);
    } else {
      const double angle_increment = hfov_rad / (submitted_scan.ranges_m.size() - 1);
      target->AddSensorData("depth_scan",
                            ToPointCloud(submitted_scan, -hfov_rad / 2., angle_increment));
    }
  };
  if (frozen_localization) {
    Eigen::Vector2f ignored_agent_pixel;
    std::vector<MemoryNodeLabel> rendered_labels;
    const std::string output = argv[2];
    if (RenderCartographerOccupancy(&map_builder, trajectory_id, explore_trajectory_id,
                                    cartographer::transform::Rigid3d::Identity(), {}, output,
                                    &ignored_agent_pixel, &rendered_labels, false)) {
      const auto snapshot = std::filesystem::path(output).parent_path() / "occupancy_step0.png";
      std::filesystem::copy_file(output, snapshot,
                                 std::filesystem::copy_options::overwrite_existing);
      std::filesystem::copy_file(output + ".render.json", snapshot.string() + ".render.json",
                                 std::filesystem::copy_options::overwrite_existing);
      CopyAmclMapSnapshot(output, snapshot);
      WriteSnapshotNodeManifest(snapshot, 0, rendered_labels);
    }
    if (static_map_no_relocalization) {
      std::ofstream diagnostic(std::string(argv[2]) + ".localization.json");
      diagnostic << "{\"mode\":\"static_map_no_relocalization\","
                 << "\"status\":\"disabled\","
                 << "\"reason\":\"benchmark_capability_profile\"}\n";
    }
  }
  if (planning_frozen_localization && planning_trajectory_id >= 0 &&
      !planning_output.empty()) {
    Eigen::Vector2f ignored_agent_pixel;
    std::vector<MemoryNodeLabel> rendered_labels;
    if (RenderCartographerOccupancy(
            planning_map_builder.get(), planning_trajectory_id,
            planning_explore_trajectory_id,
            cartographer::transform::Rigid3d::Identity(), {}, planning_output,
            &ignored_agent_pixel, &rendered_labels, false)) {
      const auto snapshot =
          std::filesystem::path(planning_output).parent_path() / "occupancy_step0.png";
      std::filesystem::copy_file(planning_output, snapshot,
                                 std::filesystem::copy_options::overwrite_existing);
      std::filesystem::copy_file(planning_output + ".render.json",
                                 snapshot.string() + ".render.json",
                                 std::filesystem::copy_options::overwrite_existing);
      CopyAmclMapSnapshot(planning_output, snapshot);
      WriteSnapshotNodeManifest(snapshot, 0, rendered_labels);
      std::ofstream diagnostic(planning_output + ".localization.json");
      diagnostic << "{\"mode\":\"localized_from_visual\",\"status\":\"waiting\","
                 << "\"localization_source\":\"visual_transform\"}\n";
    }
  }
  std::string line;
  int previous_sequence = -1;
  int previous_visual_sequence = -1;
  int previous_planning_sequence = -1;
  std::map<int, Scan> visual_scan_history;
  std::map<int, Scan> planning_scan_history;
  auto remember_scan = [](std::map<int, Scan>* history, const Scan& scan) {
    (*history)[scan.sequence] = scan;
    constexpr size_t kMaxScanHistory = 64;
    while (history->size() > kMaxScanHistory) {
      history->erase(history->begin());
    }
  };
  auto resolve_correction_anchor = [&] {
    std::lock_guard<std::mutex> delayed_lock(delayed.mutex);
    if (delayed.correction_anchor_succeeded) {
      delayed.correction_anchor_succeeded = false;
      correction_checkpoint.reset();
      return;
    }
    if (!delayed.correction_anchor_failed || !correction_checkpoint.has_value()) {
      return;
    }
    const auto checkpoint = *correction_checkpoint;
    retired_trajectories.erase(
        std::remove_if(retired_trajectories.begin(), retired_trajectories.end(),
                       [&](const RetiredTrajectory& retired) {
                         return retired.id == checkpoint.visual.id;
                       }),
        retired_trajectories.end());
    retired_trajectories.push_back(
        {trajectory_id, builder,
         rebase_active_odometry
             ? active_local_from_stream_odometry
             : cartographer::transform::Rigid3d::Identity()});
    trajectory_id = checkpoint.visual.id;
    builder = checkpoint.visual.builder;
    active_local_from_stream_odometry =
        checkpoint.visual.local_from_stream_odometry;
    rebase_active_odometry = true;
    active_trajectory_id->store(trajectory_id);
    if (checkpoint.planning.has_value()) {
      retired_planning_trajectories.erase(
          std::remove_if(
              retired_planning_trajectories.begin(),
              retired_planning_trajectories.end(),
              [&](const RetiredTrajectory& retired) {
                return retired.id == checkpoint.planning->id;
              }),
          retired_planning_trajectories.end());
      retired_planning_trajectories.push_back(
          {planning_trajectory_id, planning_builder,
           planning_rebase_active_odometry
               ? planning_active_local_from_stream_odometry
               : cartographer::transform::Rigid3d::Identity()});
      planning_trajectory_id = checkpoint.planning->id;
      planning_builder = checkpoint.planning->builder;
      planning_active_local_from_stream_odometry =
          checkpoint.planning->local_from_stream_odometry;
      planning_rebase_active_odometry = true;
      planning_active_trajectory_id->store(planning_trajectory_id);
      std::lock_guard<std::mutex> planning_lock(planning_localization_mutex);
      planning_requested_old_from_local =
          checkpoint.requested_old_from_local;
    }
    delayed.requested_old_from_local = checkpoint.requested_old_from_local;
    delayed.active_local_from_stream_odometry =
        checkpoint.visual.local_from_stream_odometry;
    delayed.localized = true;
    delayed.localization_failed = false;
    delayed.correction_anchor_failed = false;
    ResetRelocalizationSource(&delayed);
    WriteLocalizationStatus(argv[2], "localized", delayed,
                            "correction_anchor_rolled_back");
    std::ofstream events(std::string(argv[2]) +
                             ".localization.events.jsonl",
                         std::ios::app);
    events << "{\"event\":\"correction_anchor_rolled_back\","
           << "\"restored_trajectory_id\":" << trajectory_id << "}\n";
    correction_checkpoint.reset();
  };
  while (std::getline(std::cin, line)) {
    resolve_correction_anchor();
    if (IsRelocalizationPriorCheckControl(line)) {
      if (!agent_guided_relocalization) {
        std::ofstream events(std::string(argv[2]) +
                                 ".localization.events.jsonl",
                             std::ios::app);
        events << "{\"event\":\"prior_check_ignored\","
               << "\"reason\":\"agent_guided_relocalization_disabled\"}\n";
        continue;
      }
      if (static_map_no_relocalization) {
        std::ofstream events(std::string(argv[2]) +
                                 ".localization.events.jsonl",
                             std::ios::app);
        events << "{\"event\":\"prior_check_ignored\","
               << "\"reason\":\"relocalization_disabled\"}\n";
        continue;
      }
      std::lock_guard<std::mutex> delayed_lock(delayed.mutex);
      const int check_sequence =
          std::max(0, std::max(previous_sequence,
                               delayed.latest_local_pose_sequence));
      const auto control = ParseRelocalizationPriorCheckControl(line);
      if (!control.has_value()) {
        WritePriorCheckEvent(std::string(argv[2]), check_sequence,
                             "prior_check_ignored", "hab_report_memory_pos",
                             delayed, node_prior_min_points,
                             "invalid_control_record");
      } else {
        QueueNodePriorRelocalization(
            std::string(argv[2]), &delayed, node_prior_min_points,
            check_sequence, control->first, control->second);
      }
      continue;
    }
    Scan scan;
    if (!ParseScan(line, &scan)) {
      std::cerr << "invalid or out-of-order pose-free scan record\n";
      finish_open_trajectories();
      return 1;
    }
    int* previous_stream_sequence = &previous_sequence;
    if (dual_stream) {
      previous_stream_sequence =
          scan.stream == "planning" ? &previous_planning_sequence
                                    : &previous_visual_sequence;
    }
    if (scan.sequence <= *previous_stream_sequence) {
      std::cerr << "invalid or out-of-order pose-free scan record\n";
      finish_open_trajectories();
      return 1;
    }
    *previous_stream_sequence = scan.sequence;
    previous_sequence = std::max(previous_sequence, scan.sequence);
    if (scan.has_odometry != use_odometry) {
      std::cerr << "odometry record does not match HAB_CARTOGRAPHER_USE_ODOMETRY\n";
      finish_open_trajectories();
      return 1;
    }
    if (static_map_no_relocalization) {
      // Preserve the inherited map as immutable visual evidence. Do not submit
      // scans to Cartographer: scan matching, AMCL, matcher calls, and transform
      // handoffs are all forbidden in this benchmark profile.
      continue;
    }
    if (dual_stream && scan.stream == "planning") {
      remember_scan(&planning_scan_history, scan);
    } else {
      remember_scan(&visual_scan_history, scan);
    }
    if (dual_stream && scan.stream == "planning") {
      if (planning_builder == nullptr) {
        std::cerr << "planning Cartographer trajectory is not initialized\n";
        finish_open_trajectories();
        return 1;
      }
      for (const auto& retired : retired_planning_trajectories) {
        submit_scan(retired.builder, scan, false,
                    retired.local_from_stream_odometry);
      }
      if (!planning_frozen_localization || planning_localized_from_visual) {
        submit_scan(planning_builder, scan, true,
                    planning_rebase_active_odometry
                        ? planning_active_local_from_stream_odometry
                        : cartographer::transform::Rigid3d::Identity(),
                    true);
      } else {
        submit_scan(planning_builder, scan, false,
                    cartographer::transform::Rigid3d::Identity(),
                    true);
      }
      continue;
    }
    bool begin_handoff = false;
    bool correction_handoff = false;
    auto previous_requested_old_from_local =
        cartographer::transform::Rigid3d::Identity();
    auto old_from_local = cartographer::transform::Rigid3d::Identity();
    auto handoff_local_pose = cartographer::transform::Rigid3d::Identity();
    int handoff_sequence = -1;
    {
    std::lock_guard<std::mutex> delayed_lock(delayed.mutex);
    if (delayed.enabled && !delayed.localized && !delayed.anchor_requested &&
        !delayed.localization_failed) {
      if (!delayed.attempt_in_flight) {
        WriteLocalizationStatus(argv[2], "collecting", delayed);
      }
    }
    if (delayed.enabled && !delayed.anchor_requested &&
        (!delayed.localization_failed || delayed.localized) &&
        delayed.attempt_in_flight &&
        delayed.matcher.valid() &&
        delayed.matcher.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
      const int matcher_status = delayed.matcher.get();
      delayed.attempt_in_flight = false;
      std::ifstream result_input(delayed.result_path);
      const std::string result((std::istreambuf_iterator<char>(result_input)),
                               std::istreambuf_iterator<char>());
      const bool accepted = matcher_status == 0 &&
          std::regex_search(result, std::regex(R"("accepted":true)"));
      const std::string attempt_trigger = delayed.active_trigger;
      const std::string attempt_node_id = delayed.active_node_id;
      std::smatch reason_match;
      const std::string reject_reason =
          std::regex_search(result, reason_match, std::regex(R"json("reason":"([^"]+)")json"))
              ? std::string(reason_match[1])
              : "";
      std::ofstream events(std::string(argv[2]) + ".localization.events.jsonl", std::ios::app);
      events << "{\"sequence\":" << scan.sequence << ",\"attempt\":" << delayed.attempt
             << ",\"trigger\":\"" << attempt_trigger << "\"";
      if (!attempt_node_id.empty()) {
        events << ",\"node_id\":\"" << attempt_node_id << "\"";
      }
      events
             << ",\"accepted\":" << (accepted ? "true" : "false")
             << ",\"unique_points\":" << delayed.unique_points;
      if (!reject_reason.empty()) {
        events << ",\"reason\":\"" << reject_reason << "\"";
      }
      events << "}\n";
      if (delayed.pending_node_prior.valid) {
        events << "{\"sequence\":" << scan.sequence
               << ",\"event\":\"attempt_superseded\",\"attempt\":"
               << delayed.attempt << ",\"tool_seq\":"
               << delayed.active_tool_seq << ",\"superseded_by_tool_seq\":"
               << delayed.pending_node_prior.tool_seq << "}\n";
        delayed.active_trigger = "accumulated_points";
        delayed.active_node_id.clear();
        delayed.active_prior_path.clear();
        delayed.active_tool_seq = -1;
        WriteLocalizationStatus(argv[2],
                                delayed.localized ? "localized" : "collecting",
                                delayed, "superseded");
        TryStartNodePriorRelocalization(
            std::string(argv[2]), &delayed, node_prior_min_points,
            scan.sequence, "queued_report", true);
      } else if (accepted) {
        std::smatch match;
        const std::regex matrix_pattern(
            R"("transform_old_from_local":\[\[([-0-9.eE]+),([-0-9.eE]+),[-0-9.eE]+,([-0-9.eE]+)\],\[([-0-9.eE]+),([-0-9.eE]+),[-0-9.eE]+,([-0-9.eE]+)\])");
        if (std::regex_search(result, match, matrix_pattern)) {
          Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
          rotation(0, 0) = std::stod(match[1]);
          rotation(0, 1) = std::stod(match[2]);
          rotation(1, 0) = std::stod(match[4]);
          rotation(1, 1) = std::stod(match[5]);
          delayed.old_from_local = cartographer::transform::Rigid3d(
              {std::stod(match[3]), std::stod(match[6]), 0.},
              Eigen::Quaterniond(rotation));
          old_from_local = delayed.old_from_local;
          if (delayed.has_latest_local_pose) {
            handoff_local_pose = delayed.latest_local_pose;
            handoff_sequence = delayed.latest_local_pose_sequence;
          } else {
            handoff_local_pose = OdomPose(scan);
            handoff_sequence = scan.sequence;
          }
          begin_handoff = true;
          correction_handoff = delayed.localized;
          previous_requested_old_from_local = delayed.requested_old_from_local;
          WriteLocalizationStatus(argv[2], "anchoring", delayed);
        } else {
          if (attempt_trigger != "agent_reported_node") {
            delayed.next_attempt_points = delayed.unique_points + 300;
          }
          WriteLocalizationStatus(argv[2], "ambiguous", delayed, "invalid_transform");
          delayed.active_trigger = "accumulated_points";
          delayed.active_node_id.clear();
          delayed.active_prior_path.clear();
        }
      } else {
        if (attempt_trigger != "agent_reported_node") {
          delayed.next_attempt_points = delayed.unique_points + 300;
        }
        WriteLocalizationStatus(argv[2],
                                delayed.localized ? "localized" : "ambiguous",
                                delayed, "rejected");
        delayed.active_trigger = "accumulated_points";
        delayed.active_node_id.clear();
        delayed.active_prior_path.clear();
      }
    }
    }
    if (begin_handoff) {
      const auto visual_handoff_it = visual_scan_history.find(handoff_sequence);
      if (visual_handoff_it == visual_scan_history.end() ||
          !visual_handoff_it->second.has_odometry) {
        std::cerr << "accepted delayed relocalization has no matching visual odometry scan\n";
        finish_open_trajectories();
        return 1;
      }
      const Scan handoff_scan = visual_handoff_it->second;
      if (!scan.has_odometry) {
        std::cerr << "accepted delayed relocalization has no odometry scan\n";
        finish_open_trajectories();
        return 1;
      }
      // The registration source is accumulated in the provisional
      // scan-matched Cartographer local frame. The accepted transform is tied
      // to delayed.latest_local_pose_sequence, while the worker may receive it
      // on a later scan. Derive the stable old-from-stream-odometry transform
      // from the matched handoff sequence, but start the new active trajectory
      // at the current scan time so Cartographer never sees non-monotonic
      // sensor data on the newly opened trajectory.
      const auto old_from_stream_odometry =
          old_from_local * handoff_local_pose * OdomPose(handoff_scan).inverse();
      const auto requested_old_from_local =
          old_from_stream_odometry * OdomPose(scan);
      const auto retired_local_from_stream_odometry =
          rebase_active_odometry
              ? active_local_from_stream_odometry
              : cartographer::transform::Rigid3d::Identity();
      active_local_from_stream_odometry = OdomPose(scan).inverse();
      rebase_active_odometry = true;
      // Keep the provisional input advancing after handoff. This Cartographer
      // collator will not release a later trajectory while an earlier one is
      // open but no longer receiving sensor data. Finishing it midstream is
      // terminal in this build, so both trajectories finish at stdin shutdown.
      if (correction_handoff) {
        correction_checkpoint = CorrectionCheckpoint{
            {trajectory_id, builder, retired_local_from_stream_odometry},
            std::nullopt, previous_requested_old_from_local};
      }
      retired_trajectories.push_back(
          {trajectory_id, builder, retired_local_from_stream_odometry});
      trajectory_id = add_trajectory();
      active_trajectory_id->store(trajectory_id);
      auto* pose_graph = dynamic_cast<cartographer::mapping::PoseGraph*>(map_builder.pose_graph());
      if (pose_graph == nullptr) {
        finish_open_trajectories();
        return 1;
      }
      pose_graph->SetInitialTrajectoryPose(
          trajectory_id, explore_trajectory_id, requested_old_from_local,
          cartographer::common::FromUniversal(
              static_cast<int64_t>(scan.timestamp_s * 10000000.0)));
      {
        std::lock_guard<std::mutex> lock(rendered_scans_mutex);
        rendered_scans.clear();
        scan_matched_trace.clear();
      }
      if (planning_frozen_localization && planning_builder != nullptr) {
        const auto retired_planning_local_from_stream_odometry =
            planning_rebase_active_odometry
                ? planning_active_local_from_stream_odometry
                : cartographer::transform::Rigid3d::Identity();
        planning_active_local_from_stream_odometry = active_local_from_stream_odometry;
        planning_rebase_active_odometry = true;
        retired_planning_trajectories.push_back(
            {planning_trajectory_id, planning_builder,
             retired_planning_local_from_stream_odometry});
        if (correction_checkpoint.has_value()) {
          correction_checkpoint->planning = retired_planning_trajectories.back();
        }
        planning_trajectory_id = add_planning_trajectory();
        planning_active_trajectory_id->store(planning_trajectory_id);
        auto* planning_pose_graph =
            dynamic_cast<cartographer::mapping::PoseGraph*>(
                planning_map_builder->pose_graph());
        if (planning_pose_graph == nullptr) {
          finish_open_trajectories();
          return 1;
        }
        planning_pose_graph->SetInitialTrajectoryPose(
            planning_trajectory_id, planning_explore_trajectory_id,
            requested_old_from_local,
            cartographer::common::FromUniversal(
                static_cast<int64_t>(scan.timestamp_s * 10000000.0)));
        {
          std::lock_guard<std::mutex> lock(planning_rendered_scans_mutex);
          planning_rendered_scans.clear();
          planning_scan_matched_trace.clear();
        }
        {
          std::lock_guard<std::mutex> planning_lock(planning_localization_mutex);
          planning_requested_old_from_local = requested_old_from_local;
          planning_localized_from_visual = true;
        }
        std::ofstream diagnostic(planning_output + ".localization.json");
        diagnostic << "{\"mode\":\"localized_from_visual\",\"status\":"
                   << "\"localized_from_visual\","
                   << "\"localization_source\":\"visual_transform\","
                   << "\"sequence\":" << scan.sequence
                   << ",\"old_from_local_x\":"
                   << requested_old_from_local.translation().x()
                   << ",\"old_from_local_y\":"
                   << requested_old_from_local.translation().y()
                   << ",\"old_from_local_yaw_rad\":"
                   << YawOf(requested_old_from_local) << "}\n";
        planning_builder =
            planning_map_builder->GetTrajectoryBuilder(planning_trajectory_id);
      }
      {
        std::lock_guard<std::mutex> delayed_lock(delayed.mutex);
        ResetRelocalizationSource(&delayed);
        delayed.requested_old_from_local = requested_old_from_local;
        delayed.active_local_from_stream_odometry = active_local_from_stream_odometry;
        delayed.anchor_requested = true;
        delayed.correction_anchor = correction_handoff;
        delayed.anchor_check_count = 0;
      }
      std::ofstream events(std::string(argv[2]) + ".localization.events.jsonl", std::ios::app);
      events << "{\"sequence\":" << scan.sequence
             << ",\"event\":\"trajectory_handoff\",\"replay_scans\":0"
             << ",\"trigger\":\"" << delayed.active_trigger << "\"";
      if (!delayed.active_node_id.empty()) {
        events << ",\"node_id\":\"" << delayed.active_node_id << "\"";
      }
      events
             << ",\"old_from_local_x\":" << old_from_local.translation().x()
             << ",\"old_from_local_y\":" << old_from_local.translation().y()
             << ",\"old_from_local_yaw_rad\":" << YawOf(old_from_local)
             << ",\"handoff_x\":" << requested_old_from_local.translation().x()
             << ",\"handoff_y\":" << requested_old_from_local.translation().y()
             << ",\"handoff_yaw_rad\":" << YawOf(requested_old_from_local)
             << ",\"handoff_local_pose_sequence\":" << handoff_sequence
             << ",\"odom_rebase_sequence\":" << handoff_scan.sequence
             << ",\"active_start_sequence\":" << scan.sequence
             << ",\"old_from_stream_odometry_x\":"
             << old_from_stream_odometry.translation().x()
             << ",\"old_from_stream_odometry_y\":"
             << old_from_stream_odometry.translation().y()
             << ",\"old_from_stream_odometry_yaw_rad\":"
             << YawOf(old_from_stream_odometry)
             << ",\"current_scan_sequence\":" << scan.sequence
             << ",\"handoff_lag\":" << (scan.sequence - handoff_sequence)
             << "}\n";
      builder = map_builder.GetTrajectoryBuilder(trajectory_id);
      for (const auto& retired : retired_trajectories) {
        submit_scan(retired.builder, scan, false,
                    retired.local_from_stream_odometry);
      }
      submit_scan(builder, scan, true, active_local_from_stream_odometry);
      continue;
    }
    if (!localization_initialized && !delayed.enabled) {
      cartographer::transform::Rigid3d initial_pose;
      double margin = 0.;
      double score = 0.;
      double effective_particles = 0.;
      const bool localized = GlobalAmclRelocalize(
          &map_builder, scan, std::string(argv[2]), &initial_pose, &margin, &score,
          &effective_particles);
      std::ofstream events(std::string(argv[2]) + ".localization.events.jsonl", std::ios::app);
      events << std::setprecision(12)
             << "{\"sequence\":" << scan.sequence << ",\"accepted\":"
             << (localized ? "true" : "false")
             << ",\"confidence_margin_log\":" << margin
             << ",\"effective_particles\":" << effective_particles << "}\n";
      std::ofstream diagnostic(std::string(argv[2]) + ".localization.json");
      diagnostic << "{\"mode\":\"amcl_global\",\"status\":\""
                 << (localized ? "localized" : "ambiguous") << "\"}\n";
      if (!localized) {
        // Do not anchor the frozen map to an ambiguous global pose, but keep
        // consuming scans: a later viewpoint can disambiguate localization.
        std::cerr << "global AMCL relocalization is still ambiguous; retrying\n";
        continue;
      }
      auto* pose_graph = dynamic_cast<cartographer::mapping::PoseGraph*>(map_builder.pose_graph());
      if (pose_graph == nullptr) {
        finish_open_trajectories();
        return 1;
      }
      pose_graph->SetInitialTrajectoryPose(
          trajectory_id, explore_trajectory_id, initial_pose,
          cartographer::common::FromUniversal(
              static_cast<int64_t>(scan.timestamp_s * 10000000.0)));
      localization_initialized = true;
    }
    for (const auto& retired : retired_trajectories) {
      submit_scan(retired.builder, scan, false,
                  retired.local_from_stream_odometry);
    }
    submit_scan(builder, scan, true,
                rebase_active_odometry
                    ? active_local_from_stream_odometry
                    : cartographer::transform::Rigid3d::Identity());
  }
  resolve_correction_anchor();
  for (const auto& retired : retired_trajectories) {
    map_builder.FinishTrajectory(retired.id);
  }
  for (const auto& retired : retired_planning_trajectories) {
    planning_map_builder->FinishTrajectory(retired.id);
  }
  map_builder.FinishTrajectory(trajectory_id);
  if (planning_trajectory_id >= 0) {
    planning_map_builder->FinishTrajectory(planning_trajectory_id);
  }
  if (previous_sequence >= 0 && (!delayed.enabled || delayed.localized)) {
    Eigen::Vector2f ignored_agent_pixel;
    std::vector<MemoryNodeLabel> rendered_labels;
    cartographer::transform::Rigid3d delayed_old_from_local =
        cartographer::transform::Rigid3d::Identity();
    const cartographer::transform::Rigid3d* delayed_old_from_local_ptr = nullptr;
    {
      std::lock_guard<std::mutex> delayed_lock(delayed.mutex);
      if (delayed.enabled && delayed.localized) {
        delayed_old_from_local = delayed.requested_old_from_local;
        delayed_old_from_local_ptr = &delayed_old_from_local;
      }
    }
    if (RenderCartographerOccupancy(
            &map_builder, trajectory_id, explore_trajectory_id,
            cartographer::transform::Rigid3d::Identity(), scan_matched_trace,
            std::string(argv[2]), &ignored_agent_pixel, &rendered_labels, false,
            delayed_old_from_local_ptr)) {
      const auto snapshot = std::filesystem::path(argv[2]).parent_path() /
                            ("occupancy_step" + std::to_string(previous_sequence) + ".png");
      std::filesystem::copy_file(std::string(argv[2]), snapshot,
                                 std::filesystem::copy_options::overwrite_existing);
      std::filesystem::copy_file(std::string(argv[2]) + ".render.json",
                                 snapshot.string() + ".render.json",
                                 std::filesystem::copy_options::overwrite_existing);
      CopyAmclMapSnapshot(std::string(argv[2]), snapshot);
      WriteSnapshotNodeManifest(snapshot, previous_sequence, rendered_labels);
      std::ofstream updates(std::string(argv[2]) + ".updates.jsonl", std::ios::app);
      updates << "{\"sequence\":" << previous_sequence
              << ",\"scan_count\":" << rendered_scans.size()
              << ",\"final_graph_overlay\":true}\n";
    }
  }
  map_builder.SerializeStateToFile(true, std::string(argv[2]) + ".pbstream");
  if (dual_stream && planning_trajectory_id >= 0 && !planning_output.empty() &&
      (!planning_frozen_localization || planning_localized_from_visual)) {
    Eigen::Vector2f ignored_agent_pixel;
    std::vector<MemoryNodeLabel> rendered_labels;
    cartographer::transform::Rigid3d planning_old_from_local =
        cartographer::transform::Rigid3d::Identity();
    const cartographer::transform::Rigid3d* planning_old_from_local_ptr = nullptr;
    {
      std::lock_guard<std::mutex> planning_lock(planning_localization_mutex);
      if (planning_frozen_localization && planning_localized_from_visual) {
        planning_old_from_local = planning_requested_old_from_local;
        planning_old_from_local_ptr = &planning_old_from_local;
      }
    }
    if (RenderCartographerOccupancy(
            planning_map_builder.get(), planning_trajectory_id,
            planning_explore_trajectory_id,
            cartographer::transform::Rigid3d::Identity(),
            planning_scan_matched_trace, planning_output, &ignored_agent_pixel,
            &rendered_labels, false, planning_old_from_local_ptr)) {
      const auto snapshot = std::filesystem::path(planning_output).parent_path() /
                            ("occupancy_step" +
                             std::to_string(std::max(0, previous_planning_sequence)) +
                             ".png");
      std::filesystem::copy_file(planning_output, snapshot,
                                 std::filesystem::copy_options::overwrite_existing);
      std::filesystem::copy_file(planning_output + ".render.json",
                                 snapshot.string() + ".render.json",
                                 std::filesystem::copy_options::overwrite_existing);
      CopyAmclMapSnapshot(planning_output, snapshot);
      WriteSnapshotNodeManifest(snapshot, std::max(0, previous_planning_sequence),
                                rendered_labels);
      std::ofstream updates(planning_output + ".updates.jsonl", std::ios::app);
      updates << "{\"sequence\":" << std::max(0, previous_planning_sequence)
              << ",\"scan_count\":" << planning_rendered_scans.size()
              << ",\"final_graph_overlay\":true}\n";
    }
    planning_map_builder->SerializeStateToFile(true, planning_output + ".pbstream");
  }
  if (delayed.enabled && !delayed.localized) {
    std::lock_guard<std::mutex> delayed_lock(delayed.mutex);
    if (delayed.attempt_in_flight && delayed.matcher.valid()) {
      const int matcher_status = delayed.matcher.get();
      delayed.attempt_in_flight = false;
      std::ifstream result_input(delayed.result_path);
      const std::string result((std::istreambuf_iterator<char>(result_input)),
                               std::istreambuf_iterator<char>());
      const bool accepted = matcher_status == 0 &&
          std::regex_search(result, std::regex(R"("accepted":true)"));
      std::ofstream events(std::string(argv[2]) + ".localization.events.jsonl", std::ios::app);
      events << "{\"event\":\"matcher_finished_at_shutdown\",\"attempt\":"
             << delayed.attempt
             << ",\"trigger\":\"" << delayed.active_trigger << "\"";
      if (!delayed.active_node_id.empty()) {
        events << ",\"node_id\":\"" << delayed.active_node_id << "\"";
      }
      events << ",\"accepted\":"
             << (accepted ? "true" : "false")
             << ",\"unique_points\":" << delayed.unique_points << "}\n";
      if (accepted) {
        WriteLocalizationStatus(argv[2], "ambiguous", delayed,
                                "input_ended_before_anchor");
        delayed.localization_failed = true;
      }
    }
    if (!delayed.localization_failed) {
      delayed.localization_failed = true;
      WriteLocalizationStatus(argv[2], "ambiguous", delayed,
                              delayed.anchor_requested
                                  ? "input_ended_before_anchor"
                                  : "input_ended_before_localization");
    }
  }
  return 0;
}
