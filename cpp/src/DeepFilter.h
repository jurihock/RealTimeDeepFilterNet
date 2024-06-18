#pragma once

#include <complex>
#include <span>

#include <DeepFilterInference.h>

class DeepFilter final : public DeepFilterInference
{

public:

  DeepFilter();

  void operator()(
    const std::span<const std::complex<float>> input,
    const std::span<std::complex<float>> output);

private:

};
