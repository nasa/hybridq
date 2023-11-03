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

#include <cstring>

namespace hybridq_clifford {

template <typename State>
inline auto GetPauli(State &&state, std::size_t pos) {
  return state.get(2 * pos + 0) + 2 * state.get(2 * pos + 1);
}

template <typename State>
inline auto SetPauli(State &&state, std::size_t pos, std::size_t op) {
  state.set(2 * pos + 0, op & 0b01);
  state.set(2 * pos + 1, op & 0b10);
}

template <typename State>
inline auto SetPauliFromChar(State &&state, std::size_t pos, char op) {
  switch (op) {
    case 'I':
      return SetPauli(state, pos, 0);
    case 'X':
      return SetPauli(state, pos, 1);
    case 'Y':
      return SetPauli(state, pos, 2);
    case 'Z':
      return SetPauli(state, pos, 3);
  }
}

template <typename State>
auto StateFromPauli(const std::string &paulis) {
  State state_(2 * std::size(paulis));
  for (std::size_t i_ = 0; i_ < std::size(paulis); ++i_)
    SetPauliFromChar(state_, i_, std::toupper(paulis[i_]));
  return state_;
}

template <typename State>
auto PauliFromState(State &&state) {
  std::string paulis_;
  for (std::size_t i_ = 0, end_ = std::size(state) / 2; i_ < end_; ++i_)
    switch (GetPauli(state, i_)) {
      case 0:
        paulis_ += 'I';
        break;
      case 1:
        paulis_ += 'X';
        break;
      case 2:
        paulis_ += 'Y';
        break;
      case 3:
        paulis_ += 'Z';
        break;
    }
  return paulis_;
}

template <typename Branch>
auto DumpBranch(const Branch &branch) {
  // Compute size
  const std::size_t size_ = sizeof(decltype(std::size(branch.state))) +
                            sizeof(typename Branch::state_type::base_type) *
                                std::size(branch.state.data()) +
                            sizeof(typename Branch::phase_type) +
                            sizeof(typename Branch::phase_type) +
                            sizeof(typename Branch::index_type);

  // Initialize output
  std::string out_(size_, '\00');

  // Initialize head of the string
  char *head_ = &out_[0];

  // Dump state
  {
    const auto n_ = std::size(branch.state);
    const auto s_ = sizeof(typename Branch::state_type::base_type) *
                    std::size(branch.state.data());
    //
    std::memcpy(head_, &n_, sizeof(n_));
    head_ += sizeof(n_);
    //
    std::memcpy(head_, &branch.state.data()[0], s_);
    head_ += s_;
  }

  // Dump phase
  {
    std::memcpy(head_, &branch.phase, sizeof(branch.phase));
    head_ += sizeof(branch.phase);
  }

  // Dump norm_phase
  {
    std::memcpy(head_, &branch.norm_phase, sizeof(branch.norm_phase));
    head_ += sizeof(branch.norm_phase);
  }

  // Dump gate_idx
  {
    std::memcpy(head_, &branch.gate_idx, sizeof(branch.gate_idx));
    head_ += sizeof(branch.gate_idx);
  }

  // Return output
  return out_;
}

template <typename Branches>
auto DumpBranches(const Branches &branches) {
  // Initialize output
  std::string out_;

  // For each branch, get dump
  for (const auto &br_ : branches) out_ += DumpBranch(br_);

  // Return dump
  return out_;
}

template <typename Branch>
auto LoadBranch(const char *buffer) {
  // Initialize branch
  Branch branch_;

  // Load state
  {
    auto n_ = std::size(branch_.state);
    //
    std::memcpy(&n_, buffer, sizeof(n_));
    buffer += sizeof(n_);
    //
    decltype(branch_.state) state_(n_);
    const auto s_ = sizeof(typename Branch::state_type::base_type) *
                    std::size(state_.data());
    std::memcpy(&state_.data()[0], buffer, s_);
    branch_.state = std::move(state_);
    buffer += s_;
  }

  // Load phase
  {
    std::memcpy(&branch_.phase, buffer, sizeof(branch_.phase));
    buffer += sizeof(branch_.phase);
  }

  // Load norm_phase
  {
    std::memcpy(&branch_.norm_phase, buffer, sizeof(branch_.norm_phase));
    buffer += sizeof(branch_.norm_phase);
  }

  // Load gate_idx
  {
    std::memcpy(&branch_.gate_idx, buffer, sizeof(branch_.gate_idx));
    buffer += sizeof(branch_.gate_idx);
  }

  // Return output
  return std::tuple{buffer, branch_};
}

template <template <typename...> typename Vector, typename Branch>
auto LoadBranches(const char *buffer, const char *end) {
  // Initialize vector of branches
  Vector<Branch> branches_;

  // Load till the whole buffer is empty
  while (buffer < end) {
    auto [new_buffer_, branch_] = LoadBranch<Branch>(buffer);
    branches_.push_back(std::move(branch_));
    buffer = new_buffer_;
  }

  // Return vector of branches
  return branches_;
}

template <template <typename...> typename Vector, typename Branch>
auto LoadBranches(const std::string &buffer) {
  return LoadBranches<Vector, Branch>(buffer.data(), &buffer.back());
}

}  // namespace hybridq_clifford
