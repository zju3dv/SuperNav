#pragma once

#include <deque>
#include <filesystem>
#include <future>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "cartographer/transform/rigid_transform.h"
#include "scan.h"

namespace habitat_cartographer {

struct ScanMatchedScan {
  Scan scan;
  cartographer::transform::Rigid3d local_pose =
      cartographer::transform::Rigid3d::Identity();
  double distance_from_previous_m = 0.;
};

enum class RelocalizationCellState { kFree = 0, kOccupied = 1 };

struct RelocalizationGridCell {
  Eigen::Vector2f center = Eigen::Vector2f::Zero();
  RelocalizationCellState state = RelocalizationCellState::kFree;
  float probability = 0.5f;
  int trajectory_bin = 0;
};

struct RelocalizationSource {
  std::vector<Eigen::Vector2f> points;
  std::vector<Eigen::Vector2f> raw_points;
  std::vector<RelocalizationGridCell> grid_cells;
  std::string representation = "recent_trajectory_raw_endpoints";
  double requested_window_length_m = 8.;
  double actual_window_length_m = 0.;
  int first_sequence = -1;
  int last_sequence = -1;
  int scan_count = 0;
  int return_point_count = 0;
  int miss_point_count = 0;
  int observed_cell_count = 0;
  int occupied_cell_count = 0;
  int free_cell_count = 0;
  double occupancy_probability_threshold = 0.65;
  double free_probability_threshold = 0.46;
  double occupancy_build_duration_ms = 0.;
};

struct NodePriorRequest {
  bool valid = false;
  int tool_seq = -1;
  std::string node_id;
  std::filesystem::path prior_path;
  RelocalizationSource source;
  bool has_agent_local_pose = false;
  int agent_local_pose_sequence = -1;
  cartographer::transform::Rigid3d agent_local_pose =
      cartographer::transform::Rigid3d::Identity();
};

struct DelayedRelocalization {
  bool enabled = false;
  bool localized = false;
  bool attempt_in_flight = false;
  bool localization_failed = false;
  int anchor_check_count = 0;
  int unique_points = 0;
  int next_attempt_points = 1500;
  int attempt = 0;
  std::string active_trigger = "accumulated_points";
  std::string active_node_id;
  std::filesystem::path active_prior_path;
  int active_tool_seq = -1;
  NodePriorRequest pending_node_prior;
  bool node_prior_attempted = false;
  // The relocalizer source cloud is expressed in the provisional Cartographer
  // local trajectory frame, using scan-matched poses rather than raw odometry.
  cartographer::transform::Rigid3d old_from_local =
      cartographer::transform::Rigid3d::Identity();
  bool has_latest_local_pose = false;
  int latest_local_pose_sequence = -1;
  cartographer::transform::Rigid3d latest_local_pose =
      cartographer::transform::Rigid3d::Identity();
  cartographer::transform::Rigid3d active_local_from_stream_odometry =
      cartographer::transform::Rigid3d::Identity();
  bool anchor_requested = false;
  bool correction_anchor = false;
  bool correction_anchor_failed = false;
  bool correction_anchor_succeeded = false;
  cartographer::transform::Rigid3d requested_old_from_local =
      cartographer::transform::Rigid3d::Identity();
  std::unordered_set<int64_t> occupied_voxels;
  std::vector<Eigen::Vector2f> local_points;
  std::deque<ScanMatchedScan> recent_scans;
  double recent_scan_path_length_m = 0.;
  double source_window_length_m = 8.;
  std::filesystem::path artifact_dir;
  std::filesystem::path target_path;
  std::filesystem::path target_free_path;
  std::filesystem::path result_path;
  std::future<int> matcher;
  std::mutex mutex;
};

void ResetRelocalizationSource(DelayedRelocalization* state);

void WriteXYCloud(const std::filesystem::path& path,
                  const std::vector<Eigen::Vector2f>& points);
void WriteRelocalizationGrid(
    const std::filesystem::path& path,
    const std::vector<RelocalizationGridCell>& cells);
void WriteLocalizationStatus(const std::string& output, const std::string& status,
                             const DelayedRelocalization& state,
                             const std::string& reason = "");
void AppendScanMatchedScanPoints(const Scan& scan,
                                 const cartographer::transform::Rigid3d& local_pose,
                                 DelayedRelocalization* state);
RelocalizationSource BuildRecentTrajectoryOccupancySource(
    const DelayedRelocalization& state, int min_occupied_points);
void StartRelocalizationAttempt(const std::string& output, const Scan& scan,
                                DelayedRelocalization* state,
                                const std::string& trigger = "accumulated_points",
                                const std::string& node_id = "",
                                const std::filesystem::path& prior_path = {},
                                bool has_agent_local_pose = false,
                                const cartographer::transform::Rigid3d& agent_local_pose =
                                    cartographer::transform::Rigid3d::Identity(),
                                int agent_local_pose_sequence = -1,
                                const std::vector<Eigen::Vector2f>* source_points = nullptr,
                                const std::string& source_representation =
                                    "recent_trajectory_raw_endpoints",
                                const RelocalizationSource* source_metadata = nullptr);

}  // namespace habitat_cartographer
