# https://github.com/microsoft/onnxruntime

set(VERSION 1.18.0)
set(GITHUB https://github.com/microsoft/onnxruntime/releases/download/v${VERSION})
set(MAVEN https://repo.maven.apache.org/maven2/com/microsoft/onnxruntime/onnxruntime-mobile/${VERSION})
set(URL "")

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  message(STATUS "ONNX Runtime ${CMAKE_SYSTEM_NAME} ${ANDROID_ABI}")
  set(URL ${MAVEN}/onnxruntime-mobile-${VERSION}.aar)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message(STATUS "ONNX Runtime ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
  set(URL ${GITHUB}/onnxruntime-osx-${CMAKE_SYSTEM_PROCESSOR}-${VERSION}.tgz)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(STATUS "ONNX Runtime ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
  set(URL ${GITHUB}/onnxruntime-linux-x64-${VERSION}.tgz)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  message(STATUS "ONNX Runtime ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
  set(URL ${GITHUB}/onnxruntime-win-x64-${VERSION}.zip)
endif()

string(COMPARE EQUAL "${URL}" "" NOK)
if(NOK)
  message(FATAL_ERROR "Unable to determine the ONNX Runtime prebuilt binary url!")
endif()

CPMAddPackage(
  NAME onnxruntime
  VERSION ${VERSION}
  URL ${URL})

if(onnxruntime_ADDED)

  add_library(onnxruntime SHARED IMPORTED)

  if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    set_target_properties(onnxruntime PROPERTIES
      IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/jni/${ANDROID_ABI}/libonnxruntime.so"
      INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/headers")
  endif()

  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set_target_properties(onnxruntime PROPERTIES
      IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.${VERSION}.dylib"
      IMPORTED_IMPLIB "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.dylib"
      INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include")
  endif()

  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_target_properties(onnxruntime PROPERTIES
      IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so.${VERSION}"
      IMPORTED_IMPLIB "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so"
      INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include")
  endif()

  if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set_target_properties(onnxruntime PROPERTIES
      IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll"
      IMPORTED_IMPLIB "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib"
      INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include")
  endif()

endif()
