#pragma once

#include <initializer_list>
#include <numeric>
#include <span>
#include <vector>

#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmath.hpp>
// #include <xtensor/xsort.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

namespace xt
{
  template<typename T>
  inline xt::xarray<T> adapt_vector(const std::span<T> span)
  {
    return xt::adapt(
      span.data(),
      span.size(),
      xt::no_ownership(),
      std::vector<size_t>{span.size()});
  }

  template<typename T>
  inline xt::xarray<T> adapt_matrix(const std::span<T> span, const std::initializer_list<size_t>& shape)
  {
    const auto size = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>{});

    if (size != span.size())
    {
      throw std::runtime_error("Invalid matrix shape!");
    }

    return xt::adapt(
      span.data(),
      span.size(),
      xt::no_ownership(),
      std::vector<size_t>{shape});
  }
}
