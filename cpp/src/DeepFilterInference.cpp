#include <DeepFilterInference.h>

DeepFilterInference::DeepFilterInference() :
  tensors(get_tensors()),
  sessions(get_sessions())
{
}

void DeepFilterInference::inference() const
{
  Ort::MemoryInfo cpu(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));
  Ort::RunOptions opt;

  auto tensor = [&](const std::string& name)
  {
    auto tensor = tensors.at(name);

    return Ort::Value::CreateTensor<float>(
      cpu,
      tensor->value.data(),
      tensor->value.size(),
      tensor->shape.data(),
      tensor->shape.size());
  };

  auto session = [&](const std::string& name,
                     const std::initializer_list<std::string>& inputs,
                     const std::initializer_list<std::string>& outputs)
  {
    std::vector<const char*> input_names;
    for (const auto& input : inputs)
    {
      input_names.emplace_back(input.c_str());
    }

    std::vector<const char*> output_names;
    for (const auto& output : outputs)
    {
      output_names.emplace_back(output.c_str());
    }

    std::vector<Ort::Value> input_values;
    for (const auto& input : inputs)
    {
      input_values.emplace_back(tensor(input));
    }

    std::vector<Ort::Value> output_values;
    for (const auto& output : outputs)
    {
      output_values.emplace_back(tensor(output));
    }

    sessions.at(name)->Run(
      opt,
      input_names.data(),
      input_values.data(),
      inputs.size(),
      output_names.data(),
      output_values.data(),
      outputs.size());
  };

  session("enc", { "feat_erb", "feat_spec" }, { "e0", "e1", "e2", "e3", "c0", "emb" });
  session("erb_dec", { "e0", "e1", "e2", "e3", "emb" }, { "m" });
  session("df_dec", { "c0", "emb" }, { "coefs" });
}

std::string DeepFilterInference::warmup() const
{
  auto session2str = [](const std::map<std::string, std::shared_ptr<Ort::Session>>& sessions, const std::string name)
  {
    auto type2str = [](ONNXTensorElementDataType type)
    {
      switch (type)
      {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
          return std::string("float");
        default:
          return std::string("not float");
      }
    };

    auto shape2str = [](std::vector<int64_t> shape)
    {
      auto value2str = [](int64_t value)
      {
        switch (value)
        {
          case -1:
            return std::string("*");
          default:
            return std::to_string(value);
        }
      };

      std::ostringstream result;

      result << "(";
      if (!shape.empty())
        result << value2str(shape.front());
      for (size_t i = 1; i < shape.size(); ++i)
        result << "," << value2str(shape.at(i));
      result << ")";

      return result.str();
    };

    auto shape2infer = [](std::vector<int64_t> shape, int64_t infervalue)
    {
      std::vector<int64_t> values;

      for (int64_t value : shape)
      {
        values.push_back((value < 0) ? infervalue : value);
      }

      return values;
    };

    std::ostringstream result;

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::RunOptions opt;

    std::shared_ptr<Ort::Session> session(sessions.at(name));
    std::vector<std::string> inputs(session->GetInputCount());
    std::vector<std::string> outputs(session->GetOutputCount());
    std::vector<const char*> input_names(session->GetInputCount());
    std::vector<const char*> output_names(session->GetOutputCount());
    std::vector<Ort::Value> input_values, output_values;

    for (size_t i = 0; i < inputs.size(); ++i)
    {
      inputs.at(i) = session->GetInputNameAllocated(i, allocator).get();
      input_names.at(i) = inputs.at(i).c_str();
    }

    for (size_t i = 0; i < outputs.size(); ++i)
    {
      outputs.at(i) = session->GetOutputNameAllocated(i, allocator).get();
      output_names.at(i) = outputs.at(i).c_str();
    }

    result << name << " inputs" << std::endl;

    for (size_t i = 0; i < inputs.size(); ++i)
    {
      auto info = session->GetInputTypeInfo(i);
      auto name = std::string(session->GetInputNameAllocated(i, allocator).get());
      auto type = info.GetTensorTypeAndShapeInfo().GetElementType();
      auto shape = info.GetTensorTypeAndShapeInfo().GetShape();
      auto infer = shape2infer(shape, 1);

      result
        << (i+1) << ") " << name << " "
        << type2str(type) << " "
        << shape2str(shape) << " -> "
        << shape2str(infer)
        << std::endl;

      input_values.emplace_back(
        Ort::Value::CreateTensor(
          allocator,
          infer.data(),
          infer.size(),
          type));
    }

    output_values = sessions.at(name)->Run(
      opt,
      input_names.data(),
      input_values.data(),
      inputs.size(),
      output_names.data(),
      outputs.size());

    result << name << " outputs" << std::endl;

    for (size_t i = 0; i < outputs.size(); ++i)
    {
      auto info = session->GetOutputTypeInfo(i);
      auto name = std::string(session->GetOutputNameAllocated(i, allocator).get());
      auto type = info.GetTensorTypeAndShapeInfo().GetElementType();
      auto shape = info.GetTensorTypeAndShapeInfo().GetShape();
      auto infer = output_values.at(i).GetTensorTypeAndShapeInfo().GetShape();

      result
        << (i+1) << ") "
        << name << " "
        << type2str(type) << " "
        << shape2str(shape) << " -> "
        << shape2str(infer)
        << std::endl;
    }

    return result.str();
  };

  return
    session2str(sessions, "enc") +
    session2str(sessions, "erb_dec") +
    session2str(sessions, "df_dec");
}

std::map<std::string, std::shared_ptr<DeepFilterInference::Tensor>> DeepFilterInference::get_tensors()
{
  const std::map<std::string, std::vector<int64_t>> shapes
  {
    { "feat_erb", {1,1,1,32} },
    { "feat_spec", {1,2,1,96} },
    { "e0", {1,64,1,32} },
    { "e1", {1,64,1,16} },
    { "e2", {1,64,1,8} },
    { "e3", {1,64,1,8} },
    { "c0", {1,64,1,96} },
    { "emb", {1,1,512} },
    { "m", {1,1,1,32} },
    { "coefs", {1,1,96,10} }
  };

  auto size = [](const std::vector<int64_t>& shape)
  {
    size_t product = 1;

    for (auto value : shape)
    {
      if (value < 0)
      {
        throw std::runtime_error(
          "Negative tensor shape element!");
      }

      product *= static_cast<size_t>(value);
    }

    return product;
  };

  std::map<std::string, std::shared_ptr<DeepFilterInference::Tensor>> tensors;

  for (const auto& [name, shape] : shapes)
  {
    auto tensor = new DeepFilterInference::Tensor();

    tensor->shape = shape;
    tensor->value.resize(size(shape));

    tensors[name] = std::shared_ptr<DeepFilterInference::Tensor>(tensor);
  }

  return tensors;
}

std::map<std::string, std::shared_ptr<Ort::Session>> DeepFilterInference::get_sessions()
{
  const std::filesystem::path& onnxpath = DeepFilterNetOnnx;

  const std::map<std::string, std::filesystem::path> onnxpaths
  {
    { "enc", onnxpath / "enc.onnx" },
    { "erb_dec", onnxpath / "erb_dec.onnx" },
    { "df_dec", onnxpath / "df_dec.onnx" }
  };

  Ort::Env env;
  Ort::SessionOptions opt;

  auto session = [&](const std::string& name)
  {
    return std::make_shared<Ort::Session>(
      env, onnxpaths.at(name).c_str(), opt);
  };

  return
  {
    { "enc", session("enc") },
    { "erb_dec", session("erb_dec") },
    { "df_dec", session("df_dec") }
  };
}
