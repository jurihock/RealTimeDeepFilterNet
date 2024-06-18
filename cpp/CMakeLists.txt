cmake_minimum_required(VERSION 3.21)

project(debug_cpp)

include("${CMAKE_CURRENT_LIST_DIR}/libs/cpm.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/libs/deepfilternet.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/libs/npy.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/libs/onnxruntime.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/libs/pocketfft.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/libs/stftpitchshift.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/libs/xtensor.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/libs/xtl.cmake")

file(GLOB_RECURSE HDR "${CMAKE_CURRENT_LIST_DIR}/src/*.h")
file(GLOB_RECURSE SRC "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE "${HDR}" "${SRC}")
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/src")
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries(${PROJECT_NAME} PRIVATE deepfilternet npy onnxruntime pocketfft stftpitchshift xtensor)

if(MSVC)
  target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE /fp:fast)
  target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE /W3 /WX)
else()
  target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE -ffast-math)
  target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE -Wall -Werror)
endif()

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:${PROJECT_NAME}> $<TARGET_RUNTIME_DLLS:${PROJECT_NAME}>
  COMMAND_EXPAND_LISTS)
