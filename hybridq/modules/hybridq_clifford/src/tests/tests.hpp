/*
Author: Salvatore Mandra (salvatore.mandra@nasa.gov)

Copyright © 2021, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The HybridQ: A Hybrid Simulator for Quantum Circuits platform is licensed under
the Apache License, Version 2.0 (the "License"); you may not use this file
except in compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
*/

#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>
#include <random>

#include "../archive.hpp"
#include "../branch.hpp"
#include "../extras/otoc_info.hpp"
#include "../info.hpp"
#include "../state.hpp"

namespace hybridq_clifford::tests {

namespace hqc = hybridq_clifford;
using namespace hybridq_clifford::archive;

template <typename T, typename RNG>
auto RandomVector1D(RNG &&rng, std::size_t n) {
  hqc::vector_type<T> v_;
  if constexpr (std::is_floating_point_v<T>) {
    std::uniform_real_distribution<T> uni_;
    std::generate_n(std::back_inserter(v_), n,
                    [&rng, &uni_]() { return uni_(rng); });
  } else {
    std::uniform_int_distribution<T> uni_;
    std::generate_n(std::back_inserter(v_), n,
                    [&rng, &uni_]() { return uni_(rng); });
  }
  return v_;
}

template <typename RNG>
auto RandomState(RNG &&rng, std::size_t n) {
  State<hqc::vector_type> state_(n);
  for (std::size_t i_ = 0, end_ = std::size(state_._data); i_ < end_; ++i_)
    state_._data[i_] = rng();
  return state_;
}

template <typename RNG>
auto RandomBranch(RNG &&rng, std::size_t n) {
  std::uniform_real_distribution<float> runi_(-1.0, 1.0);
  std::uniform_int_distribution<int> iuni_(0, 100);
  return Branch<State<hqc::vector_type>, float, int>{
      RandomState(rng, n), runi_(rng), runi_(rng), iuni_(rng)};
}

template <typename RNG>
auto RandomInfo(RNG &&rng) {
  std::uniform_int_distribution<std::size_t> iuni_(0, 100);
  return hqc::Info{iuni_(rng), iuni_(rng), iuni_(rng), iuni_(rng),
                   iuni_(rng), iuni_(rng), iuni_(rng), iuni_(rng)};
}

template <typename RNG>
auto RandomOTOCInfo(RNG &&rng) {
  std::uniform_int_distribution<std::size_t> iuni_(0, 100);
  return hqc::otoc::Info{RandomInfo(rng),
                         iuni_(rng),
                         RandomVector1D<hqc::float_type>(rng, 121),
                         RandomVector1D<hqc::float_type>(rng, 122),
                         RandomVector1D<hqc::float_type>(rng, 123),
                         RandomVector1D<hqc::float_type>(rng, 124),
                         RandomVector1D<hqc::float_type>(rng, 125),
                         RandomVector1D<hqc::index_type>(rng, 126),
                         RandomVector1D<hqc::index_type>(rng, 127)};
}

void TestLoadDump() {
  // Get random number generator
  std::mt19937_64 rng_(std::random_device{}());

  std::cerr << "# Checking 'State' ... ";
  {
    hqc::vector_type<decltype(RandomState(rng_, 0))> states_;
    std::generate_n(std::back_inserter(states_), 100,
                    [&rng_]() { return RandomState(rng_, 100); });
    if (load<decltype(states_)>(dump(states_).data()).second != states_)
      throw std::logic_error("Failed!");
  }
  std::cerr << "Done!" << std::endl;

  std::cerr << "# Checking 'Branch' ... ";
  {
    hqc::list_type<decltype(RandomBranch(rng_, 0))> branches_;
    std::generate_n(std::back_inserter(branches_), 100,
                    [&rng_]() { return RandomBranch(rng_, 100); });
    if (load<decltype(branches_)>(dump(branches_).data()).second != branches_)
      throw std::logic_error("Failed!");
  }
  std::cerr << "Done!" << std::endl;

  std::cerr << "# Checking 'Info' ... ";
  {
    hqc::vector_type<decltype(RandomInfo(rng_))> infos_;
    std::generate_n(std::back_inserter(infos_), 100,
                    [&rng_]() { return RandomInfo(rng_); });
    if (load<decltype(infos_)>(dump(infos_).data()).second != infos_)
      throw std::logic_error("Failed!");
  }
  std::cerr << "Done!" << std::endl;

  std::cerr << "# Checking 'OTOC Info' ... ";
  {
    hqc::vector_type<decltype(RandomOTOCInfo(rng_))> infos_;
    std::generate_n(std::back_inserter(infos_), 100,
                    [&rng_]() { return RandomOTOCInfo(rng_); });
    if (load<decltype(infos_)>(dump(infos_).data()).second != infos_)
      throw std::logic_error("Failed!");
  }
  std::cerr << "Done!" << std::endl;
}

}  // namespace hybridq_clifford::tests
