#include "delayed_relocalization.h"

#include <cmath>
#include <iostream>
#include <limits>

namespace {

bool HasPointNear(const std::vector<Eigen::Vector2f>& points, float x, float y,
                  float tolerance) {
  for (const auto& point : points) {
    if (std::hypot(point.x() - x, point.y() - y) <= tolerance) return true;
  }
  return false;
}

}  // namespace

int main() {
  habitat_cartographer::DelayedRelocalization state;
  for (int index = 0; index < 5; ++index) {
    habitat_cartographer::Scan scan;
    scan.sequence = index;
    scan.hfov_deg = 360.;
    scan.camera_hfov_deg = 0.;
    scan.ranges_m = {
        std::numeric_limits<float>::quiet_NaN(),
        index == 0 ? 2.f : 3.f,
        std::numeric_limits<float>::quiet_NaN(),
        9.f,
    };
    habitat_cartographer::AppendScanMatchedScanPoints(
        scan, cartographer::transform::Rigid3d::Identity(), &state);
  }

  const auto source =
      habitat_cartographer::BuildRecentTrajectoryOccupancySource(state, 1);
  if (source.representation != "recent_trajectory_occupancy") {
    std::cerr << "probability-grid source unexpectedly fell back to endpoints\n";
    return 1;
  }
  if (!HasPointNear(source.points, 3.f, 0.f, 0.08f)) {
    std::cerr << "repeated occupied return was not retained; points:";
    for (const auto& point : source.points) {
      std::cerr << ' ' << point.x() << ',' << point.y();
    }
    std::cerr << "; raw:";
    for (const auto& point : source.raw_points) {
      std::cerr << ' ' << point.x() << ',' << point.y();
    }
    std::cerr << '\n';
    return 1;
  }
  if (HasPointNear(source.points, 2.f, 0.f, 0.08f)) {
    std::cerr << "one-off return contradicted by free-space rays was retained\n";
    return 1;
  }
  if (HasPointNear(source.points, -5.f, 0.f, 0.08f)) {
    std::cerr << "beyond-max-range miss became occupied\n";
    return 1;
  }
  if (source.return_point_count != 5 || source.miss_point_count != 5 ||
      source.observed_cell_count <= source.occupied_cell_count ||
      source.free_cell_count <= 0 || source.grid_cells.empty() ||
      std::abs(source.occupancy_probability_threshold - 0.65) > 1e-6 ||
      std::abs(source.free_probability_threshold - 0.46) > 1e-6) {
    std::cerr << "probability-grid diagnostics are inconsistent: observed="
              << source.observed_cell_count
              << " occupied=" << source.occupied_cell_count
              << " free=" << source.free_cell_count
              << " grid=" << source.grid_cells.size() << '\n';
    return 1;
  }
  for (const auto& cell : source.grid_cells) {
    if (cell.trajectory_bin < 0 || cell.trajectory_bin > 3) {
      std::cerr << "probability-grid trajectory bin is invalid\n";
      return 1;
    }
  }
  habitat_cartographer::ResetRelocalizationSource(&state);
  if (state.unique_points != 0 || state.has_latest_local_pose ||
      !state.local_points.empty() || !state.recent_scans.empty() ||
      state.recent_scan_path_length_m != 0.) {
    std::cerr << "trajectory handoff did not reset relocalization source state\n";
    return 1;
  }
  return 0;
}
