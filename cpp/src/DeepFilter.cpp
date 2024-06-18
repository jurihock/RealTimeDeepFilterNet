#include <DeepFilter.h>

DeepFilter::DeepFilter()
{
}

void DeepFilter::operator()(
  const std::span<const std::complex<float>> input,
  const std::span<std::complex<float>> output)
{
  inference();
}
