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

#ifndef HYBRIDQ__SWAP_H
#define HYBRIDQ__SWAP_H

#include "pack.h"
#include "utils.h"

namespace hybridq::swap {

template <typename Positions, std::size_t size = array_size_v<Positions>>
constexpr inline auto swap(std::size_t x, Positions&& pos) {
  std::size_t y{0};
  for (std::size_t i = 0; i < size; ++i) y ^= ((x >> i) & 1uL) << pos[i];
  return y;
}

template <typename pack_type, typename Positions, std::size_t... I>
constexpr inline void _swap(pack_type&& pack, Positions&& pos,
                            std::index_sequence<I...>) {
  pack = remove_pcvr_t<pack_type>{pack[swap(I, pos)]...};
}

template <typename pack_type, typename Positions,
          std::size_t size = pack_size_v<pack_type>>
constexpr inline void swap(pack_type&& pack, Positions&& pos) {
  _swap(pack, pos, std::make_index_sequence<size>{});
}

template <typename float_type, typename Positions,
          std::size_t swap_size = array_size_v<Positions>>
int swap_array(float_type* array, Positions&& pos, const std::size_t size) {
  // Reinterpret
  auto* _array = reinterpret_cast<
      typename hybridq::__pack__<float_type, 1uL << swap_size>::value_type*>(
      array);

#pragma omp parallel for
  for (std::size_t i = 0; i < (size >> swap_size); ++i) swap(_array[i], pos);

  return 0;
}

template <typename float_type, typename index_type>
int swap_array(float_type* array, index_type* pos, const std::size_t size,
               const std::size_t n_pos) {
  // Get U_Size
  const std::size_t swap_size = 1uL << n_pos;

  // Compute offset
  std::size_t offset[n_pos];
  for (std::size_t i = 0; i < n_pos; ++i) {
    offset[i] = 0;
    for (std::size_t j = i + 1; j < n_pos; ++j) offset[i] += (pos[j] < pos[i]);
  }

  // Allocate buffers
  float_type _array[swap_size];
  std::size_t _swap_pos[swap_size];
  for (std::size_t x = 0; x < swap_size; ++x) {
    std::size_t y{0};
    for (std::size_t i = 0; i < n_pos; ++i) y ^= ((x >> i) & 1uL) << pos[i];
    _swap_pos[x] = y;
  }

#pragma omp parallel for private(_array)
  for (std::size_t i = 0; i < size >> n_pos; ++i) {
    // Load buffer
    for (std::size_t j = 0; j < swap_size; ++j)
      _array[j] = array[_swap_pos[j] ^ (i << n_pos)];

    // Dump buffer
    for (std::size_t j = 0; j < swap_size; ++j)
      array[j ^ (i << n_pos)] = _array[j];
  }

  return 0;
}

}  // namespace hybridq::swap

#endif
