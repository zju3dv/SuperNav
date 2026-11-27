#include "scan.h"

#include <cmath>
#include <cstdlib>
#include <regex>

#include "cartographer/common/time.h"

namespace habitat_cartographer {

double BeamAngle(const Scan& scan, size_t index) {
  const size_t beams_per_view = scan.ranges_m.size() / 4;
  const size_t view = index / beams_per_view;
  const size_t beam = index % beams_per_view;
  const double view_center = -M_PI / 2. + view * M_PI / 2.;
  const double view_hfov = scan.camera_hfov_deg * M_PI / 180.;
  if (beams_per_view <= 1) return view_center;
  return view_center - view_hfov / 2. +
         beam * view_hfov / (beams_per_view - 1);
}

bool ParseScan(const std::string& line, Scan* scan) {
  if (line.find("pose") != std::string::npos || line.find("imu") != std::string::npos) {
    return false;
  }
  const char* dual_stream = std::getenv("HAB_CARTOGRAPHER_DUAL_STREAM");
  const bool require_stream = dual_stream != nullptr && std::string(dual_stream) == "1";
  std::smatch match;
  if (std::regex_search(line, match, std::regex(R"json("stream":"([^"]+)")json"))) {
    scan->stream = match[1];
    if (scan->stream != "visual" && scan->stream != "planning") return false;
  } else if (require_stream) {
    return false;
  } else {
    scan->stream = "visual";
  }
  if (!std::regex_search(line, match, std::regex(R"("sequence":([0-9]+))"))) return false;
  scan->sequence = std::stoi(match[1]);
  if (!std::regex_search(line, match, std::regex(R"("timestamp_s":([-0-9.eE]+))"))) return false;
  scan->timestamp_s = std::stod(match[1]);
  if (!std::regex_search(line, match, std::regex(R"("ranges_m":\[([^\]]*)\])"))) return false;
  std::regex value(R"((null)|([-0-9.eE]+))");
  for (std::sregex_iterator it(match[1].first, match[1].second, value), end;
       it != end; ++it) {
    scan->ranges_m.push_back((*it)[1].matched ? NAN : std::stof((*it)[2]));
  }
  if (std::regex_search(line, match, std::regex(R"("hfov_deg":([-0-9.eE]+))"))) {
    scan->hfov_deg = std::stod(match[1]);
  }
  if (std::regex_search(line, match,
                        std::regex(R"("camera_hfov_deg":([-0-9.eE]+))"))) {
    scan->camera_hfov_deg = std::stod(match[1]);
  }
  if (std::regex_search(
          line, match,
          std::regex(R"("odometry":\{"x":([-0-9.eE]+),"y":([-0-9.eE]+),"yaw_rad":([-0-9.eE]+)\})"))) {
    scan->has_odometry = true;
    scan->odometry_x = std::stod(match[1]);
    scan->odometry_y = std::stod(match[2]);
    scan->odometry_yaw_rad = std::stod(match[3]);
  }
  if (std::regex_search(line, match, std::regex(R"("publish_snapshot":(true|false))"))) {
    scan->publish_snapshot = match[1] == "true";
  }
  return !scan->ranges_m.empty();
}

cartographer::sensor::TimedPointCloudData ToPointCloud(
    const Scan& scan, double angle_min, double angle_increment) {
  cartographer::sensor::TimedPointCloudData data;
  data.time = cartographer::common::FromUniversal(
      static_cast<int64_t>(scan.timestamp_s * 10000000.0));
  data.origin = Eigen::Vector3f::Zero();
  for (size_t index = 0; index < scan.ranges_m.size(); ++index) {
    const float range = scan.ranges_m[index];
    if (!std::isfinite(range)) continue;
    const double angle = angle_min + index * angle_increment;
    data.ranges.push_back({{static_cast<float>(range * std::cos(angle)),
                            static_cast<float>(range * std::sin(angle)), 0.f},
                           0.f});
  }
  return data;
}

}  // namespace habitat_cartographer
