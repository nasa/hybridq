/*
 * Author: Salvatore Mandra (salvatore.mandra@nasa.gov)
 *
 * Copyright © 2021, United States Government, as represented by the
 * Administrator of the National Aeronautics and Space Administration. All
 * rights reserved.
 *
 * The HybridQ: A Hybrid Simulator for Quantum Circuits platform is licensed
 * under the Apache License, Version 2.0 (the "License"); you may not use this
 * file except in compliance with the License. You may obtain a copy of the
 * License at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#include <array>
#include <cstdint>

// Get number of bytes for array_type
static constexpr std::size_t n_bytes = HYBRIDQ_ARRAY_SWAP_N_BYTES;

// Get number of positions
static constexpr std::size_t n_pos = HYBRIDQ_ARRAY_SWAP_N_POS;

// Get array_type
template <std::size_t n_bytes>
struct ArrayType {
  typedef uint64_t type
      __attribute__((vector_size(sizeof(uint64_t) * n_bytes / 8)));
};
template <>
struct ArrayType<1> {
  using type = uint8_t;
};
template <>
struct ArrayType<2> {
  using type = uint16_t;
};
template <>
struct ArrayType<4> {
  using type = uint32_t;
};
template <>
struct ArrayType<8> {
  using type = uint64_t;
};

// Get array_type
using array_type = typename ArrayType<n_bytes>::type;

uint32_t swap_bits(uint32_t x, const uint32_t *pos) {
  /*
   * Swap bits accordingly to 'pos'.
   */
  uint32_t _x = x & ~uint32_t(0) << n_pos;
  for (std::size_t i = 0; i < n_pos; ++i)
    _x |= ((x >> i) & uint32_t(1)) << pos[i];
  return _x;
}

extern "C" {

int32_t swap(void *array, const uint32_t *pos, const uint32_t n_qubits) {
  /*
   * Swap array accordingly to pos.
   */

  // Check that array is not empty
  if (array == nullptr) return 1;

  // Check that pos is not empty
  if (pos == nullptr) return 2;

  // Reinterpret to the right pointer
  array_type *_array = reinterpret_cast<array_type *>(array);

  // Get swap size
  static constexpr std::size_t size = 1uLL << n_pos;

  // Get array size
  const std::size_t array_size = 1uLL << n_qubits;

  // Expand positions
  std::array<std::size_t, size> _expanded;
  for (std::size_t i = 0; i < size; ++i) {
    _expanded[i] = swap_bits(i, pos);
  }

#pragma omp parallel for
  for (std::size_t i = 0; i < array_size; i += size) {
    // Create temporary array
    std::array<array_type, size> _buffer;

    // Swap to buffer
    for (std::size_t j = 0; j < size; ++j)
      _buffer[j] = _array[i + _expanded[j]];

    // Copy
    for (std::size_t j = 0; j < size; ++j) _array[i + j] = _buffer[j];
  }

  // Everything is ok
  return 0;
}
}
