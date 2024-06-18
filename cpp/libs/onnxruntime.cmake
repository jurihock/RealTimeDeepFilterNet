# https://github.com/microsoft/onnxruntime

set(VERSION 1.18.0)
set(URL https://github.com/microsoft/onnxruntime/releases/download/v${VERSION})

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(URL ${URL}/onnxruntime-osx-arm64-${VERSION}.tgz)
endif()

CPMAddPackage(
  NAME onnxruntime
  VERSION ${VERSION}
  URL ${URL})

if(onnxruntime_ADDED)

  add_library(onnxruntime SHARED IMPORTED)

  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set_target_properties(onnxruntime PROPERTIES
      IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.${VERSION}.dylib"
      IMPORTED_IMPLIB "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.dylib"
      INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include")
  endif()

endif()
