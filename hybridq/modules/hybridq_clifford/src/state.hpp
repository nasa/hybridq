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

#include <cstdint>

#include "archive.hpp"

namespace hybridq_clifford {

template <template <typename...> typename vector_type,
          typename BaseType = std::uint64_t>
struct State {
  using base_type = BaseType;
  using data_type = vector_type<base_type>;
  static constexpr std::size_t block_size = 8 * sizeof(base_type);

  // Array to store data
  data_type _data;

  // Initialize bitset
  State(std::size_t n = 0)
      : _data(n / block_size + ((n % block_size) != 0), 0), _n{n} {}

  template <typename Vector, typename Vector_ = typename std::decay_t<Vector>,
            std::enable_if_t<!std::is_integral_v<Vector> &&
                                 !std::is_same_v<Vector_, State>,
                             bool> = true>
  State(Vector &&v) : State(std::size(v)) {
    std::size_t i_ = 0;
    for (const auto &x : v) this->set(i_++, x);
  }

  // Set defaults copy/move constructors
  State(State &&) = default;
  State(const State &) = default;
  State &operator=(State &&) = default;
  State &operator=(const State &) = default;

  // Get size of the bitset
  auto size() const { return _n; }

  // Check if two bitset are the same
  auto operator==(const State &s) const {
    return _n == s._n && _data == s._data;
  }

  // Get value of a bit
  auto get(std::size_t p) const {
    return (_data[p / block_size] >> (p % block_size)) & base_type{1};
  }

  // Set value of a bit
  auto set(std::size_t p, bool v) {
    if (v)
      _data[p / block_size] |= base_type{1} << (p % block_size);
    else
      _data[p / block_size] &= ~(base_type{1} << (p % block_size));
  }

  // Return hash
  auto hash() const {
    std::size_t seed = std::size(_data);
    for (const auto &x : _data) {
      seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }

 private:
  std::size_t _n{0};
};

}  // namespace hybridq_clifford

template <template <typename...> typename vector_type, typename base_type>
struct std::hash<hybridq_clifford::State<vector_type, base_type>> {
  std::size_t operator()(
      const hybridq_clifford::State<vector_type, base_type> &s) const noexcept {
    return s.hash();
  }
};

namespace hybridq_clifford::archive {

template <template <typename...> typename vector_type, typename BaseType>
struct Dump<State<vector_type, BaseType>> {
  auto operator()(const State<vector_type, BaseType> &state) const {
    using Array = typename State<vector_type, BaseType>::data_type;
    return dump<std::size_t>(std::size(state)) + dump<Array>(state._data);
  }
};

template <template <typename...> typename vector_type, typename BaseType>
struct Load<State<vector_type, BaseType>> {
  auto operator()(const char *buffer) const {
    using Array = typename State<vector_type, BaseType>::data_type;
    auto [h1_, size_] = load<std::size_t>(buffer);
    auto [h2_, array_] = load<Array>(h1_);
    State<vector_type, BaseType> state_(size_);
    state_._data = std::move(array_);
    return std::pair{h2_, state_};
  }
};

}  // namespace hybridq_clifford::archive
