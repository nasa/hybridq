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

#include <fstream>

namespace hybridq_clifford {

template <typename State>
inline auto GetPauli(State &&state, std::size_t pos) {
  return state[2 * pos + 0] + 2 * state[2 * pos + 1];
}

template <typename State>
inline auto SetPauli(State &&state, std::size_t pos, std::size_t op) {
  state[2 * pos + 0] = op & 0b01;
  state[2 * pos + 1] = op & 0b10;
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

auto GetMemory() {
  // Get element from meminfo line
  static constexpr auto get_ = [](std::string str) {
    str = str.substr(str.find_first_of(' '));
    str = str.substr(str.find_first_not_of(' '));
    str = str.substr(0, str.find_first_of(' '));
    return std::stof(str) / (1uL << 20);
  };

  float total_mem_{0}, free_mem_{0};
  std::string line_;

  if (auto meminfo = std::ifstream("/proc/meminfo"); meminfo.good()) {
    while (std::getline(meminfo, line_))
      if (line_.find("MemTotal:") != std::string::npos)
        total_mem_ = get_(line_);
      else if (line_.find("MemFree:") != std::string::npos)
        free_mem_ = get_(line_);
  }
  return std::tuple{total_mem_, free_mem_};
}

}  // namespace hybridq_clifford
