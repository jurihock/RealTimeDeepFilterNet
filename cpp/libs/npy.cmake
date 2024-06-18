# https://github.com/llohse/libnpy

CPMAddPackage(
  NAME npy
  VERSION 1.0.1
  GIT_REPOSITORY https://github.com/llohse/libnpy
  DOWNLOAD_ONLY YES)

if(npy_ADDED)

  add_library(npy INTERFACE)

  target_include_directories(npy
    INTERFACE "${npy_SOURCE_DIR}/include")

endif()
