#include <array>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <xt.h>

#include <npy.hpp>

#include <DeepFilter.h>

int main()
{
  DeepFilter df;

  std::cout << df.probe() << std::endl;

  if (false) // test npy
  {
    const std::vector<double> data{1, 2, 3, 4, 5, 6};

    npy::npy_data<double> d;
    d.data = data;
    d.shape = {2, 3};
    d.fortran_order = false; // optional

    const std::string path{"out.npy"};
    npy::write_npy(path, d);
  }
}
