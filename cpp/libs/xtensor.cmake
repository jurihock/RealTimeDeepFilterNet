# https://github.com/xtensor-stack/xtensor

CPMAddPackage(
  NAME xtensor
  VERSION 0.25.0
  GIT_TAG 0.25.0
  GITHUB_REPOSITORY xtensor-stack/xtensor
  DOWNLOAD_ONLY YES)

if(xtensor_ADDED)

  add_library(xtensor INTERFACE)

  target_include_directories(xtensor
    INTERFACE "${xtensor_SOURCE_DIR}/include")

  target_link_libraries(xtensor
    INTERFACE xtl)

endif()
