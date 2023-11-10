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

#include <pybind11/pybind11.h>

#include <limits>

#include "archive.hpp"

namespace hybridq_clifford {

namespace py = pybind11;

template <typename T, bool>
struct Inf_ {};

template <typename T>
struct Inf_<T, false> {
  static constexpr auto value = std::numeric_limits<T>::max();
};

template <typename T>
struct Inf_<T, true> {
  static constexpr auto value = std::numeric_limits<T>::infinity();
};

template <typename T, typename T_ = std::decay_t<T>>
struct Inf {
  static constexpr auto value =
      Inf_<T_, std::numeric_limits<T_>::has_infinity>::value;
};

template <typename T>
static constexpr auto inf = Inf<T>::value;

template <typename T>
auto isinf(T &&t) {
  return t == inf<T>;
}

struct Info {
  std::size_t n_explored_branches{0};
  std::size_t n_completed_branches{0};
  std::size_t n_remaining_branches{inf<std::size_t>};
  std::size_t n_total_branches{inf<std::size_t>};
  std::size_t n_threads{inf<std::size_t>};
  std::size_t runtime_ms{inf<std::size_t>};
  std::size_t branching_time_ms{inf<std::size_t>};
  std::size_t expanding_time_ms{inf<std::size_t>};

  bool operator==(const Info &other) const {
    return n_explored_branches == other.n_explored_branches &&
           n_completed_branches == other.n_completed_branches &&
           n_remaining_branches == other.n_remaining_branches &&
           n_total_branches == other.n_total_branches &&
           n_threads == other.n_threads && runtime_ms == other.runtime_ms &&
           branching_time_ms == other.branching_time_ms &&
           expanding_time_ms == other.expanding_time_ms;
  }

  auto dict() const {
    py::dict out_;
    if (!isinf(n_explored_branches))
      out_["n_explored_branches"] = n_explored_branches;
    if (!isinf(n_completed_branches))
      out_["n_completed_branches"] = n_completed_branches;
    if (!isinf(n_remaining_branches))
      out_["n_remaining_branches"] = n_remaining_branches;
    if (!isinf(n_total_branches)) out_["n_total_branches"] = n_total_branches;
    if (!isinf(n_threads)) out_["n_threads"] = n_threads;
    if (!isinf(runtime_ms)) out_["runtime_ms"] = runtime_ms;
    if (!isinf(branching_time_ms))
      out_["branching_time_ms"] = branching_time_ms;
    if (!isinf(expanding_time_ms))
      out_["expanding_time_ms"] = expanding_time_ms;
    return out_;
  }
};

}  // namespace hybridq_clifford
