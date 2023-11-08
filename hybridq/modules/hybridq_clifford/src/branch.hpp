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

#include <iostream>

#include "archive.hpp"

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

  bool operator==(const Branch &other) const {
    return state == other.state && phase == other.phase &&
           norm_phase == other.norm_phase && gate_idx == other.gate_idx;
  }

  std::ostream &operator<<(std::ostream &out) const {
    out << *this;
    return out;
  }
  friend std::ostream &operator<<(std::ostream &out, const Branch &branch) {
    out << "(state=";
    for (std::size_t i_ = 0, end_ = std::size(branch.state) / 2; i_ < end_;
         ++i_)
      switch (branch.state.get(2 * i_ + 0) + 2 * branch.state.get(2 * i_ + 1)) {
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

namespace hybridq_clifford::archive {

template <typename StateType, typename PhaseType, typename IndexType>
struct Dump<Branch<StateType, PhaseType, IndexType>> {
  using Branch_ = Branch<StateType, PhaseType, IndexType>;
  auto operator()(const Branch_ &branch) {
    return dump(branch.state) + dump(branch.phase) + dump(branch.norm_phase) +
           dump(branch.gate_idx);
  }
};

template <typename StateType, typename PhaseType, typename IndexType>
struct Load<Branch<StateType, PhaseType, IndexType>> {
  using Branch_ = Branch<StateType, PhaseType, IndexType>;
  auto operator()(const char *buffer) {
    auto [h1_, state_] = load<typename Branch_::state_type>(buffer);
    auto [h2_, phase_] = load<typename Branch_::phase_type>(h1_);
    auto [h3_, norm_phase_] = load<typename Branch_::phase_type>(h2_);
    auto [h4_, gate_idx_] = load<typename Branch_::index_type>(h3_);
    return std::pair{
        h4_, Branch_{std::move(state_), phase_, norm_phase_, gate_idx_}};
  }
};

}  // namespace hybridq_clifford::archive
