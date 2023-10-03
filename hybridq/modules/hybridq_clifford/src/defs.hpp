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

#include <ostream>

namespace hybridq_clifford {

struct Info {
  std::size_t n_explored_branches{0};
  std::size_t n_remaining_branches{0};
  std::size_t n_completed_branches{0};
  std::size_t n_threads{std::numeric_limits<std::size_t>::max()};
  float runtime_s{std::numeric_limits<float>::infinity()};
  float branching_time_us{std::numeric_limits<float>::infinity()};
  float merging_time_ms{std::numeric_limits<float>::infinity()};
  float expanding_time_ms{std::numeric_limits<float>::infinity()};

  // Print to ostream
  std::ostream &operator<<(std::ostream &out) const {
    out << *this;
    return out;
  }
  //
  friend std::ostream &operator<<(std::ostream &out, const Info &info) {
    out << "Number Explored Branches: " << info.n_explored_branches
        << std::endl;
    out << "Number Completed Branches: " << info.n_completed_branches
        << std::endl;
    out << "Number Remaining Branches: " << info.n_remaining_branches
        << std::endl;
    out << "Number of threads: " << info.n_threads << std::endl;
    out << "Expanding Time (ms): " << info.expanding_time_ms << std::endl;
    out << "Merging Time (ms): " << info.merging_time_ms << std::endl;
    out << "Runtime (s): " << info.runtime_s << std::endl;
    out << "Branching Time (μs): " << info.branching_time_us << std::endl;
    return out;
  }
};
using info_type = Info;

using float_type = float;
using index_type = std::size_t;

using IVector1D = std::vector<index_type>;
using IVector2D = std::vector<std::vector<index_type>>;
using IVector3D = std::vector<std::vector<std::vector<index_type>>>;
using FVector1D = std::vector<float_type>;
using FVector2D = std::vector<std::vector<float_type>>;
using FVector3D = std::vector<std::vector<std::vector<float_type>>>;

using phase_type = float_type;
using state_type = std::vector<bool>;
using phases_type = FVector2D;
using positions_type = IVector2D;
using qubits_type = IVector1D;

using SVector1D = std::vector<state_type>;
using SFVector1D = std::vector<std::tuple<state_type, phase_type>>;

struct Branch {
  state_type state;
  phase_type phase;
  phase_type norm_phase;
  index_type gate_idx;

  std::ostream &operator<<(std::ostream &out) const {
    out << *this;
    return out;
  }
  friend std::ostream &operator<<(std::ostream &out, const Branch &branch) {
    out << "(state=";
    for (std::size_t i_ = 0, end_ = std::size(branch.state) / 2; i_ < end_;
         ++i_)
      switch (branch.state[2 * i_ + 0] + 2 * branch.state[2 * i_ + 1]) {
        case 0:
          out << "I";
          break;
        case 1:
          out << "X";
          break;
        case 2:
          out << "Y";
          break;
        case 3:
          out << "Z";
          break;
      }
    out << ", phase=" << branch.phase;
    out << ", norm_phase=" << branch.norm_phase;
    out << ", gate_idx=" << branch.gate_idx << ")";
    return out;
  }
};
using branch_type = Branch;
using branches_type = std::unordered_map<state_type, float_type>;

}  // namespace hybridq_clifford
