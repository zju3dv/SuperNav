#include "delayed_relocalization.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_set>

#include "cartographer/mapping/2d/map_limits.h"
#include "cartographer/mapping/2d/probability_grid.h"
#include "cartographer/mapping/2d/probability_grid_range_data_inserter_2d.h"
#include "cartographer/mapping/value_conversion_tables.h"
#include "cartographer/sensor/range_data.h"

namespace habitat_cartographer {
namespace {

int64_t VoxelKey(const Eigen::Vector2f& point) {
  constexpr float kVoxel = 0.1f;
  const int32_t x = static_cast<int32_t>(std::floor(point.x() / kVoxel));
  const int32_t y = static_cast<int32_t>(std::floor(point.y() / kVoxel));
  return (static_cast<int64_t>(x) << 32) ^ static_cast<uint32_t>(y);
}

constexpr float kOccupancyResolution = 0.05f;
constexpr float kMinRange = 0.15f;
constexpr float kMaxRange = 8.f;
constexpr float kMissingDataRayLength = 5.f;
constexpr float kHitProbability = 0.55f;
constexpr float kMissProbability = 0.49f;
// The frozen-map point cloud uses a deliberately conservative rendered-map
// cutoff. Requiring approximately three net hit updates gives the temporary
// grid a comparable confidence floor and prevents one-off depth endpoints
// from becoming matcher geometry.
constexpr float kOccupiedProbabilityThreshold = 0.65f;
constexpr float kFreeProbabilityThreshold = 0.46f;
constexpr int kTrajectoryBinCount = 4;

int64_t OccupancyCellKey(const Eigen::Vector2f& point) {
  const int32_t x =
      static_cast<int32_t>(std::floor(point.x() / kOccupancyResolution));
  const int32_t y =
      static_cast<int32_t>(std::floor(point.y() / kOccupancyResolution));
  const uint64_t packed =
      (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 32) |
      static_cast<uint32_t>(y);
  return static_cast<int64_t>(packed);
}

double ScanBeamAngle(const Scan& scan, size_t index) {
  const bool surround = std::abs(scan.hfov_deg - 360.) < 1e-6 &&
                        scan.ranges_m.size() % 4 == 0;
  if (surround) return BeamAngle(scan, index);
  if (scan.ranges_m.size() <= 1) return 0.;
  const double hfov_rad = scan.hfov_deg * M_PI / 180.;
  return -hfov_rad / 2. +
         index * hfov_rad / (scan.ranges_m.size() - 1);
}

cartographer::sensor::RangeData BuildScanRangeData(
    const ScanMatchedScan& entry,
    std::unordered_set<int64_t>* raw_endpoint_cells,
    RelocalizationSource* source) {
  const auto rotation = entry.local_pose.rotation().toRotationMatrix();
  const double yaw = std::atan2(rotation(1, 0), rotation(0, 0));
  const double cosine = std::cos(yaw);
  const double sine = std::sin(yaw);
  const double pose_x = entry.local_pose.translation().x();
  const double pose_y = entry.local_pose.translation().y();
  cartographer::sensor::RangeData range_data;
  range_data.origin = Eigen::Vector3f(static_cast<float>(pose_x),
                                      static_cast<float>(pose_y), 0.f);
  std::vector<cartographer::sensor::RangefinderPoint> returns;
  std::vector<cartographer::sensor::RangefinderPoint> misses;
  for (size_t index = 0; index < entry.scan.ranges_m.size(); ++index) {
    const float range = entry.scan.ranges_m[index];
    if (!std::isfinite(range) || range < kMinRange) continue;
    const double angle = ScanBeamAngle(entry.scan, index);
    const double ray_length = range <= kMaxRange ? range : kMissingDataRayLength;
    const Eigen::Vector2f sensor(
        static_cast<float>(ray_length * std::cos(angle)),
        static_cast<float>(ray_length * std::sin(angle)));
    const Eigen::Vector3f local(
        static_cast<float>(pose_x + cosine * sensor.x() - sine * sensor.y()),
        static_cast<float>(pose_y + sine * sensor.x() + cosine * sensor.y()),
        0.f);
    cartographer::sensor::RangefinderPoint point;
    point.position = local;
    if (range <= kMaxRange) {
      returns.push_back(point);
      ++source->return_point_count;
      const Eigen::Vector2f endpoint = local.head<2>();
      if (raw_endpoint_cells->insert(OccupancyCellKey(endpoint)).second) {
        source->raw_points.push_back(endpoint);
      }
    } else {
      misses.push_back(point);
      ++source->miss_point_count;
    }
  }
  range_data.returns = cartographer::sensor::PointCloud(std::move(returns));
  range_data.misses = cartographer::sensor::PointCloud(std::move(misses));
  return range_data;
}

int NearestTrajectoryBin(
    const Eigen::Vector2f& point,
    const std::vector<cartographer::sensor::RangeData>& scans) {
  size_t nearest_index = 0;
  float nearest_squared_distance = std::numeric_limits<float>::infinity();
  for (size_t index = 0; index < scans.size(); ++index) {
    const float squared_distance =
        (point - scans[index].origin.head<2>()).squaredNorm();
    if (squared_distance < nearest_squared_distance) {
      nearest_squared_distance = squared_distance;
      nearest_index = index;
    }
  }
  return std::min(
      kTrajectoryBinCount - 1,
      static_cast<int>(nearest_index * kTrajectoryBinCount / scans.size()));
}

std::vector<Eigen::Vector2f> BuildProbabilityGridOccupancy(
    const std::vector<cartographer::sensor::RangeData>& scans,
    RelocalizationSource* source) {
  if (scans.empty()) return {};

  Eigen::AlignedBox2f bounds;
  for (const auto& scan : scans) {
    bounds.extend(scan.origin.head<2>());
    for (const auto& point : scan.returns) {
      bounds.extend(point.position.head<2>());
    }
    for (const auto& point : scan.misses) {
      bounds.extend(point.position.head<2>());
    }
  }
  if (bounds.isEmpty()) return {};

  constexpr float kPadding = 0.5f;
  // Cartographer's Array2i grid indexing is rotated relative to world X/Y.
  // A square initial grid avoids axis-size ambiguity and prevents GrowLimits()
  // from shifting cell centers while this temporary map is being inserted.
  const float side_length =
      std::max(bounds.sizes().x(), bounds.sizes().y()) + 2.f * kPadding;
  const int cells = std::max(
      1, static_cast<int>(std::ceil(side_length / kOccupancyResolution)) + 2);
  const Eigen::Vector2f maximum =
      bounds.center() +
      Eigen::Vector2f::Constant(0.5f * cells * kOccupancyResolution);

  cartographer::mapping::ValueConversionTables conversion_tables;
  cartographer::mapping::ProbabilityGrid probability_grid(
      cartographer::mapping::MapLimits(
          kOccupancyResolution, maximum.cast<double>(),
          cartographer::mapping::CellLimits(cells, cells)),
      &conversion_tables);
  cartographer::mapping::proto::ProbabilityGridRangeDataInserterOptions2D
      options;
  options.set_insert_free_space(true);
  options.set_hit_probability(kHitProbability);
  options.set_miss_probability(kMissProbability);
  cartographer::mapping::ProbabilityGridRangeDataInserter2D inserter(options);
  for (const auto& scan : scans) inserter.Insert(scan, &probability_grid);

  std::vector<Eigen::Vector2f> occupied;
  const auto& limits = probability_grid.limits();
  const auto& cell_limits = limits.cell_limits();
  for (int row = 0; row < cell_limits.num_x_cells; ++row) {
    for (int column = 0; column < cell_limits.num_y_cells; ++column) {
      const Eigen::Array2i index(row, column);
      if (!probability_grid.IsKnown(index)) continue;
      ++source->observed_cell_count;
      const float probability = probability_grid.GetProbability(index);
      const Eigen::Vector2f center = limits.GetCellCenter(index);
      if (probability >= kOccupiedProbabilityThreshold) {
        occupied.push_back(center);
        source->grid_cells.push_back(
            {center, RelocalizationCellState::kOccupied, probability,
             NearestTrajectoryBin(center, scans)});
      } else if (probability <= kFreeProbabilityThreshold) {
        source->grid_cells.push_back(
            {center, RelocalizationCellState::kFree, probability,
             NearestTrajectoryBin(center, scans)});
        ++source->free_cell_count;
      }
    }
  }
  return occupied;
}

}  // namespace

void WriteXYCloud(const std::filesystem::path& path,
                  const std::vector<Eigen::Vector2f>& points) {
  const auto temporary = path.string() + ".tmp";
  std::ofstream output(temporary);
  output << std::setprecision(9);
  for (const auto& point : points) {
    output << point.x() << ' ' << point.y() << '\n';
  }
  output.close();
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
}

void WriteRelocalizationGrid(
    const std::filesystem::path& path,
    const std::vector<RelocalizationGridCell>& cells) {
  const auto temporary = path.string() + ".tmp";
  std::ofstream output(temporary);
  output << std::setprecision(9);
  for (const auto& cell : cells) {
    output << cell.center.x() << '\t' << cell.center.y() << '\t'
           << (cell.state == RelocalizationCellState::kOccupied ? 1 : 0)
           << '\t' << cell.probability << '\t' << cell.trajectory_bin << '\n';
  }
  output.close();
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
}

void WriteLocalizationStatus(const std::string& output, const std::string& status,
                             const DelayedRelocalization& state,
                             const std::string& reason) {
  std::ofstream diagnostic(output + ".localization.json");
  diagnostic << "{\"mode\":\"delayed_fpfh_relocalization\",\"status\":\"" << status
             << "\",\"unique_points\":" << state.unique_points
             << ",\"threshold_points\":1500,\"next_attempt_points\":"
             << state.next_attempt_points << ",\"attempt\":" << state.attempt
             << ",\"trigger\":\"" << state.active_trigger << "\"";
  if (state.active_tool_seq >= 0) {
    diagnostic << ",\"tool_seq\":" << state.active_tool_seq;
  }
  if (state.pending_node_prior.valid) {
    diagnostic << ",\"pending_tool_seq\":"
               << state.pending_node_prior.tool_seq;
  }
  if (!state.active_node_id.empty()) {
    diagnostic << ",\"node_id\":\"" << state.active_node_id << "\"";
  }
  if (!reason.empty()) diagnostic << ",\"reason\":\"" << reason << "\"";
  diagnostic << "}\n";
}

void AppendScanMatchedScanPoints(const Scan& scan,
                                 const cartographer::transform::Rigid3d& local_pose,
                                 DelayedRelocalization* state) {
  const auto rotation = local_pose.rotation().toRotationMatrix();
  const double yaw = std::atan2(rotation(1, 0), rotation(0, 0));
  const double cosine = std::cos(yaw);
  const double sine = std::sin(yaw);
  const double pose_x = local_pose.translation().x();
  const double pose_y = local_pose.translation().y();
  const bool surround = std::abs(scan.hfov_deg - 360.) < 1e-6 &&
                        scan.ranges_m.size() % 4 == 0;
  const double hfov_rad = scan.hfov_deg * M_PI / 180.;
  for (size_t index = 0; index < scan.ranges_m.size(); ++index) {
    const float range = scan.ranges_m[index];
    if (!std::isfinite(range)) continue;
    const double angle = surround ? BeamAngle(scan, index)
                                  : -hfov_rad / 2. + index * hfov_rad /
                                                          (scan.ranges_m.size() - 1);
    const Eigen::Vector2f sensor(static_cast<float>(range * std::cos(angle)),
                                 static_cast<float>(range * std::sin(angle)));
    const Eigen::Vector2f local(
        static_cast<float>(pose_x + cosine * sensor.x() - sine * sensor.y()),
        static_cast<float>(pose_y + sine * sensor.x() + cosine * sensor.y()));
    if (state->occupied_voxels.insert(VoxelKey(local)).second) {
      state->local_points.push_back(local);
    }
  }
  state->unique_points = static_cast<int>(state->local_points.size());
  state->latest_local_pose = local_pose;
  state->latest_local_pose_sequence = scan.sequence;
  state->has_latest_local_pose = true;

  ScanMatchedScan history_entry;
  history_entry.scan = scan;
  history_entry.local_pose = local_pose;
  if (!state->recent_scans.empty()) {
    const auto delta = local_pose.translation() -
                       state->recent_scans.back().local_pose.translation();
    history_entry.distance_from_previous_m = std::hypot(delta.x(), delta.y());
    state->recent_scan_path_length_m += history_entry.distance_from_previous_m;
  }
  state->recent_scans.push_back(std::move(history_entry));
  while (state->recent_scans.size() > 1) {
    const double oldest_edge =
        state->recent_scans[1].distance_from_previous_m;
    if (state->recent_scan_path_length_m - oldest_edge <
        state->source_window_length_m) {
      break;
    }
    state->recent_scan_path_length_m -= oldest_edge;
    state->recent_scans.pop_front();
    state->recent_scans.front().distance_from_previous_m = 0.;
  }
}

void ResetRelocalizationSource(DelayedRelocalization* state) {
  state->occupied_voxels.clear();
  state->local_points.clear();
  state->recent_scans.clear();
  state->recent_scan_path_length_m = 0.;
  state->unique_points = 0;
  state->has_latest_local_pose = false;
  state->latest_local_pose_sequence = -1;
  state->latest_local_pose = cartographer::transform::Rigid3d::Identity();
}

RelocalizationSource BuildRecentTrajectoryOccupancySource(
    const DelayedRelocalization& state, int min_occupied_points) {
  const auto build_started = std::chrono::steady_clock::now();
  RelocalizationSource source;
  source.occupancy_probability_threshold = kOccupiedProbabilityThreshold;
  source.free_probability_threshold = kFreeProbabilityThreshold;
  source.requested_window_length_m = state.source_window_length_m;
  source.actual_window_length_m = state.recent_scan_path_length_m;
  source.scan_count = static_cast<int>(state.recent_scans.size());
  if (!state.recent_scans.empty()) {
    source.first_sequence = state.recent_scans.front().scan.sequence;
    source.last_sequence = state.recent_scans.back().scan.sequence;
  }

  std::unordered_set<int64_t> raw_endpoint_cells;
  std::vector<cartographer::sensor::RangeData> scans;
  scans.reserve(state.recent_scans.size());
  for (const auto& entry : state.recent_scans) {
    scans.push_back(BuildScanRangeData(entry, &raw_endpoint_cells, &source));
  }
  source.points = BuildProbabilityGridOccupancy(scans, &source);
  source.occupied_cell_count = static_cast<int>(source.points.size());
  if (source.occupied_cell_count >= std::max(1, min_occupied_points)) {
    source.representation = "recent_trajectory_occupancy";
  } else {
    source.points = source.raw_points;
  }
  source.occupancy_build_duration_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - build_started)
          .count();
  return source;
}

void StartRelocalizationAttempt(const std::string& output, const Scan& scan,
                                DelayedRelocalization* state,
                                const std::string& trigger,
                                const std::string& node_id,
                                const std::filesystem::path& prior_path,
                                bool has_agent_local_pose,
                                const cartographer::transform::Rigid3d& agent_local_pose,
                                int agent_local_pose_sequence,
                                const std::vector<Eigen::Vector2f>* source_points,
                                const std::string& source_representation,
                                const RelocalizationSource* source_metadata) {
  state->attempt_in_flight = true;
  ++state->attempt;
  state->active_trigger = trigger.empty() ? "accumulated_points" : trigger;
  if (state->active_trigger != "agent_reported_node") {
    state->active_tool_seq = -1;
  }
  state->active_node_id = node_id;
  state->active_prior_path = prior_path;
  const auto source_path = state->artifact_dir /
      ("relocalization_local_attempt" + std::to_string(state->attempt) + ".xy");
  state->result_path = state->artifact_dir /
      ("relocalization_attempt" + std::to_string(state->attempt) + ".json");
  const auto& matcher_points =
      source_points != nullptr ? *source_points : state->local_points;
  WriteXYCloud(source_path, matcher_points);
  std::filesystem::path source_grid_path;
  if (source_metadata != nullptr && !source_metadata->grid_cells.empty()) {
    source_grid_path = state->artifact_dir /
        ("relocalization_source_grid_attempt" +
         std::to_string(state->attempt) + ".tsv");
    WriteRelocalizationGrid(source_grid_path, source_metadata->grid_cells);
  }
  if (source_representation == "recent_trajectory_occupancy") {
    const auto raw_path = state->artifact_dir /
        ("relocalization_raw_attempt" + std::to_string(state->attempt) + ".xy");
    WriteXYCloud(raw_path, source_metadata != nullptr
                                   ? source_metadata->raw_points
                                   : state->local_points);
  }
  WriteLocalizationStatus(output, "matching", *state);
  const char* command_env = std::getenv("HAB_CARTOGRAPHER_RELOCALIZER_COMMAND");
  const std::string command =
      command_env != nullptr && std::string(command_env).size() > 0
          ? std::string(command_env)
          : "conda run --no-capture-output -n habitat-gs python tools/cartographer/relocalize.py";
  std::ostringstream agent_pose_args;
  if (state->active_trigger == "agent_reported_node" && has_agent_local_pose) {
    const auto rotation = agent_local_pose.rotation().toRotationMatrix();
    const double yaw = std::atan2(rotation(1, 0), rotation(0, 0));
    agent_pose_args << std::setprecision(17)
                    << " --agent-local-pose "
                    << agent_local_pose.translation().x() << ' '
                    << agent_local_pose.translation().y() << ' ' << yaw
                    << " --agent-local-sequence " << agent_local_pose_sequence;
  }
  const std::string invocation = command + " --source '" + source_path.string() +
      "' --target '" + state->target_path.string() + "' --output '" +
      state->result_path.string() + "'" +
      (state->active_trigger == "agent_reported_node"
           ? " --profile sparse_node_prior --prior '" +
                 state->active_prior_path.string() + "'"
           : " --profile accumulated_points") +
      " --source-representation '" + source_representation + "'" +
      (!source_grid_path.empty()
           ? " --source-grid '" + source_grid_path.string() + "'"
           : "") +
      (!state->target_free_path.empty()
           ? " --target-free '" + state->target_free_path.string() + "'"
           : "") +
      (source_metadata != nullptr
           ? " --source-window-length-m " +
                 std::to_string(source_metadata->requested_window_length_m) +
                 " --source-window-actual-length-m " +
                 std::to_string(source_metadata->actual_window_length_m) +
                 " --source-window-first-sequence " +
                 std::to_string(source_metadata->first_sequence) +
                 " --source-window-last-sequence " +
                 std::to_string(source_metadata->last_sequence) +
                 " --source-window-scan-count " +
                 std::to_string(source_metadata->scan_count) +
                 " --source-window-return-point-count " +
                 std::to_string(source_metadata->return_point_count) +
                 " --source-window-miss-point-count " +
                 std::to_string(source_metadata->miss_point_count) +
                 " --source-window-observed-cell-count " +
                 std::to_string(source_metadata->observed_cell_count) +
                 " --source-window-occupied-cell-count " +
                 std::to_string(source_metadata->occupied_cell_count) +
                 " --source-window-free-cell-count " +
                 std::to_string(source_metadata->free_cell_count) +
                 " --source-window-occupancy-probability-threshold " +
                 std::to_string(
                     source_metadata->occupancy_probability_threshold) +
                 " --source-window-free-probability-threshold " +
                 std::to_string(source_metadata->free_probability_threshold) +
                 " --source-window-occupancy-build-duration-ms " +
                 std::to_string(source_metadata->occupancy_build_duration_ms)
           : "") +
      agent_pose_args.str();
  state->matcher = std::async(std::launch::async, [invocation] {
    return std::system(invocation.c_str());
  });
}

}  // namespace habitat_cartographer
