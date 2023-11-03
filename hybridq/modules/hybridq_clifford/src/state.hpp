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

template <template <typename...> typename vector_type,
          typename BaseType = std::uint64_t>
struct State {
  using base_type = BaseType;
  static constexpr std::size_t block_size = 8 * sizeof(base_type);

  // Initialize bitset
  State(std::size_t n = 0)
      : _n{n}, _data(n / block_size + ((n % block_size) != 0), 0) {}

  template <typename Vector, typename Vector_ = typename std::decay_t<Vector>,
            typename = std::enable_if_t<!std::is_same_v<Vector_, State>>>
  State(Vector &&v) : State(std::size(v)) {
    std::size_t i_ = 0;
    for (const auto &x : v) this->set(i_++, x);
  }

  // Set defaults copy/move constructors
  State(State &&) = default;
  State(const State &) = default;
  State &operator=(State &&) = default;
  State &operator=(const State &) = default;

  // Get underlying data
  auto &data() { return _data; }
  const auto &data() const { return _data; }

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
  vector_type<base_type> _data;
};

}  // namespace hybridq_clifford

template <template <typename...> typename vector_type, typename base_type>
struct std::hash<hybridq_clifford::State<vector_type, base_type>> {
  std::size_t operator()(
      const hybridq_clifford::State<vector_type, base_type> &s) const noexcept {
    return s.hash();
  }
};
