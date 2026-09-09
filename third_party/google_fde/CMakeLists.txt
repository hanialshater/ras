cmake_minimum_required(VERSION 3.24)
project(ras_google_fde LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
include(FetchContent)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(abseil
  URL https://codeload.github.com/abseil/abseil-cpp/tar.gz/refs/tags/20250127.1
  URL_HASH SHA256=b396401fd29e2e679cace77867481d388c807671dc2acc602a0259eeb79b7811
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_WITH_ZLIB OFF CACHE BOOL "" FORCE)
FetchContent_Declare(protobuf
  URL https://codeload.github.com/protocolbuffers/protobuf/tar.gz/refs/tags/v21.12
  URL_HASH SHA256=22fdaf641b31655d4b2297f9981fa5203b2866f8332d3c6333f6b0107bb320de
  SOURCE_SUBDIR cmake DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(eigen
  URL https://codeload.github.com/eigenteam/eigen-git-mirror/tar.gz/refs/tags/3.3.7
  URL_HASH SHA256=a8d87c8df67b0404e97bcef37faf3b140ba467bc060e2b883192165b319cea8d
  SOURCE_SUBDIR no-cmake-needed DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(abseil protobuf eigen)
set(PC sketching/point_cloud)
set(PROTO ${PC}/fixed_dimensional_encoding_config)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${PC})
add_custom_command(OUTPUT ${PROTO}.pb.cc ${PROTO}.pb.h
  COMMAND $<TARGET_FILE:protoc> --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
    --proto_path=${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/${PROTO}.proto
  DEPENDS protoc ${PROTO}.proto VERBATIM)
add_library(google_fde SHARED bridge.cc ${PC}/fixed_dimensional_encoding.cc ${PROTO}.pb.cc)
target_include_directories(google_fde PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR} ${eigen_SOURCE_DIR} ${protobuf_SOURCE_DIR}/src)
find_package(OpenMP REQUIRED)
target_link_libraries(google_fde PRIVATE libprotobuf absl::statusor absl::strings
  absl::log absl::check OpenMP::OpenMP_CXX)
