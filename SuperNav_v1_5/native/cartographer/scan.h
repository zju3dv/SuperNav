#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "cartographer/sensor/timed_point_cloud_data.h"

namespace habitat_cartographer {

struct Scan {
  std::string stream;
  int sequence;
  double timestamp_s;
  std::vector<float> ranges_m;
  double hfov_deg = 90.;
  double camera_hfov_deg = 90.;
  bool has_odometry = false;
  double odometry_x = 0.;
  double odometry_y = 0.;
  double odometry_yaw_rad = 0.;
  bool publish_snapshot = true;
};

double BeamAngle(const Scan& scan, size_t index);
bool ParseScan(const std::string& line, Scan* scan);
cartographer::sensor::TimedPointCloudData ToPointCloud(
    const Scan& scan, double angle_min, double angle_increment);

}  // namespace habitat_cartographer
