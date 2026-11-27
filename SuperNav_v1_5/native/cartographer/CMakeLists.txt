cmake_minimum_required(VERSION 3.20)
project(habitat_cartographer_worker LANGUAGES C CXX)

find_package(Eigen3 REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(CAIRO REQUIRED IMPORTED_TARGET cairo)
set(HAB_CARTOGRAPHER_PREFIX "$ENV{HAB_CARTOGRAPHER_PREFIX}")
if("${HAB_CARTOGRAPHER_PREFIX}" STREQUAL "")
  set(HAB_CARTOGRAPHER_PREFIX "$ENV{CONDA_PREFIX}")
endif()
find_library(GLOG_LIBRARY NAMES glog REQUIRED PATHS "${HAB_CARTOGRAPHER_PREFIX}/lib")
find_library(CARTOGRAPHER_LIBRARY NAMES cartographer REQUIRED
  PATHS "${HAB_CARTOGRAPHER_PREFIX}/lib"
  NO_DEFAULT_PATH)
find_library(CERES_LIBRARY NAMES ceres REQUIRED
  PATHS "$ENV{CONDA_PREFIX}/lib" "${HAB_CARTOGRAPHER_PREFIX}/lib")
find_library(LUA_LIBRARY NAMES lua lua5.4 REQUIRED
  PATHS "$ENV{CONDA_PREFIX}/lib" "${HAB_CARTOGRAPHER_PREFIX}/lib")
find_library(PROTOBUF_LIBRARY NAMES protobuf REQUIRED
  PATHS "$ENV{CONDA_PREFIX}/lib" "${HAB_CARTOGRAPHER_PREFIX}/lib")
find_library(BOOST_IOSTREAMS_LIBRARY NAMES boost_iostreams REQUIRED
  PATHS "$ENV{CONDA_PREFIX}/lib" "${HAB_CARTOGRAPHER_PREFIX}/lib")
find_library(GFLAGS_LIBRARY NAMES gflags REQUIRED
  PATHS "$ENV{CONDA_PREFIX}/lib" "${HAB_CARTOGRAPHER_PREFIX}/lib")

function(configure_habitat_cartographer_target target)
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_compile_definitions(${target} PRIVATE
    GLOG_USE_GLOG_EXPORT
    GUARDED_BY=ABSL_GUARDED_BY
    PT_GUARDED_BY=ABSL_PT_GUARDED_BY
    LOCKS_EXCLUDED=ABSL_LOCKS_EXCLUDED
    EXCLUSIVE_LOCKS_REQUIRED=ABSL_EXCLUSIVE_LOCKS_REQUIRED
    SHARED_LOCKS_REQUIRED=ABSL_SHARED_LOCKS_REQUIRED
    LOCK_RETURNED=ABSL_LOCK_RETURNED
    LOCKABLE=ABSL_LOCKABLE
    SCOPED_LOCKABLE=ABSL_SCOPED_LOCKABLE
    EXCLUSIVE_LOCK_FUNCTION=ABSL_EXCLUSIVE_LOCK_FUNCTION
    SHARED_LOCK_FUNCTION=ABSL_SHARED_LOCK_FUNCTION
    UNLOCK_FUNCTION=ABSL_UNLOCK_FUNCTION
    NO_THREAD_SAFETY_ANALYSIS=ABSL_NO_THREAD_SAFETY_ANALYSIS
  )
  target_include_directories(${target} PRIVATE "${HAB_CARTOGRAPHER_PREFIX}/include")
  target_link_directories(${target} PRIVATE "${HAB_CARTOGRAPHER_PREFIX}/lib")
  target_link_libraries(${target} PRIVATE
    "${CARTOGRAPHER_LIBRARY}"
    "${CERES_LIBRARY}"
    "${LUA_LIBRARY}"
    "${PROTOBUF_LIBRARY}"
    "${BOOST_IOSTREAMS_LIBRARY}"
    "${GFLAGS_LIBRARY}"
    "${GLOG_LIBRARY}"
    absl::algorithm
    absl::base
    absl::debugging
    absl::flat_hash_map
    absl::check
    absl::log
    absl::log_internal_check_op
    absl::memory
    absl::meta
    absl::numeric
    absl::str_format
    absl::strings
    absl::synchronization
    absl::time
    absl::utility
    Eigen3::Eigen
    PkgConfig::CAIRO)
  set_target_properties(${target} PROPERTIES
    BUILD_RPATH "${HAB_CARTOGRAPHER_PREFIX}/lib")
endfunction()

add_executable(habitat_cartographer_worker worker.cpp scan.cpp delayed_relocalization.cpp)
configure_habitat_cartographer_target(habitat_cartographer_worker)

include(CTest)
if(BUILD_TESTING)
  add_executable(habitat_cartographer_relocalization_source_test
    delayed_relocalization_test.cpp scan.cpp delayed_relocalization.cpp)
  configure_habitat_cartographer_target(
    habitat_cartographer_relocalization_source_test)
  add_test(NAME habitat_cartographer_relocalization_source_test
    COMMAND habitat_cartographer_relocalization_source_test)
endif()
