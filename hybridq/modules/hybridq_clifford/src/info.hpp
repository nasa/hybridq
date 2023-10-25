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

namespace hybridq_clifford {

namespace py = pybind11;

struct Info {
  std::size_t n_explored_branches{0};
  std::size_t n_remaining_branches{0};
  std::size_t n_completed_branches{0};
  std::size_t n_threads{std::numeric_limits<std::size_t>::max()};
  std::size_t runtime_ms{std::numeric_limits<std::size_t>::max()};
  std::size_t branching_time_ms{std::numeric_limits<std::size_t>::max()};
  std::size_t expanding_time_ms{std::numeric_limits<std::size_t>::max()};

  auto dict() const {
    py::dict out_;
    out_["n_explored_branches"] = n_explored_branches;
    out_["n_remaining_branches"] = n_remaining_branches;
    out_["n_completed_branches"] = n_completed_branches;
    if (n_threads != std::numeric_limits<decltype(n_threads)>::max())
      out_["n_threads"] = n_threads;
    if (runtime_ms != std::numeric_limits<decltype(runtime_ms)>::max())
      out_["runtime_ms"] = runtime_ms;
    if (branching_time_ms !=
        std::numeric_limits<decltype(branching_time_ms)>::max())
      out_["branching_time_ms"] = branching_time_ms;
    if (expanding_time_ms !=
        std::numeric_limits<decltype(expanding_time_ms)>::max())
      out_["expanding_time_ms"] = expanding_time_ms;
    return out_;
  }
};

}  // namespace hybridq_clifford
