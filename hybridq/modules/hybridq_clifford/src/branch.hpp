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

namespace hybridq_clifford {

template <typename StateType, typename PhaseType, typename IndexType>
struct Branch {
  using state_type = StateType;
  using phase_type = PhaseType;
  using index_type = IndexType;

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

}  // namespace hybridq_clifford
