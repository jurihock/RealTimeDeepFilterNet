#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

#include <onnxruntime_cxx_api.h>

class DeepFilterInference
{

public:

  DeepFilterInference();
  virtual ~DeepFilterInference() = default;

  float samplerate() const { return 48000; }
  size_t framesize() const { return 1024; }
  size_t erbsize() const { return 32; }
  size_t cpxsize() const { return 96; }

  std::string probe() const;

protected:

  struct Tensor
  {
    std::vector<int64_t> shape;
    std::vector<float> value;
  };

  const std::map<std::string, std::shared_ptr<DeepFilterInference::Tensor>> tensors;
  const std::map<std::string, std::shared_ptr<Ort::Session>> sessions;

  void inference() const;

private:

  static std::map<std::string, std::shared_ptr<DeepFilterInference::Tensor>> get_tensors();
  static std::map<std::string, std::shared_ptr<Ort::Session>> get_sessions();

};
