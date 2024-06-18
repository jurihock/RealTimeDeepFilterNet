# https://github.com/Rikorose/DeepFilterNet

CPMAddPackage(
  NAME deepfilternet
  VERSION 0.5.6
  GIT_REPOSITORY https://github.com/Rikorose/DeepFilterNet
  DOWNLOAD_ONLY YES)

if(deepfilternet_ADDED)

  add_library(deepfilternet INTERFACE)

  file(ARCHIVE_EXTRACT
    INPUT "${deepfilternet_SOURCE_DIR}/models/DeepFilterNet3_onnx.tar.gz"
    DESTINATION "${CMAKE_BINARY_DIR}/DeepFilterNet3")

  target_compile_definitions(deepfilternet
    INTERFACE -DDeepFilterNetOnnx="${CMAKE_BINARY_DIR}/DeepFilterNet3/tmp/export")

endif()
