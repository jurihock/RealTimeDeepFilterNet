#include <iostream>

#include <DeepFilter.h>
#include <FFT.h>
#include <STFT.h>

int main()
{
  DeepFilter filter;

  // std::cout << filter.probe() << std::endl;

  auto samplerate = filter.samplerate();
  auto framesize = filter.framesize();
  auto hopsize = filter.hopsize();
  auto chronometry = true;
  auto samples = static_cast<size_t>(10 * samplerate);

  auto fft = std::make_shared<FFT>();
  auto stft = std::make_shared<STFT>(fft, framesize, hopsize, chronometry);

  std::vector<float> x(samples);
  std::vector<float> y(samples);

  (*stft)(x, y, [&](std::span<std::complex<float>> dft)
  {
    filter(dft, dft);
  });
}
