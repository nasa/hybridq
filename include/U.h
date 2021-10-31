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

#ifndef HYBRIDQ__U_HPP
#define HYBRIDQ__U_HPP

#include "pack.h"
#include "utils.h"

namespace hybridq::U {

template <std::size_t log2_pack_size, typename float_type,
          typename... Positions>
int apply(float_type *psi_re_ptr, float_type *psi_im_ptr,
          const float_type *U_ptr, const std::size_t state_size_ptr,
          Positions &&...pos) {
  // Check if pointer are correctly aligned
  if (reinterpret_cast<std::size_t>(psi_re_ptr) % 32 or
      reinterpret_cast<std::size_t>(psi_im_ptr) % 32)
    return 1;

  // Get size of pack
  static const std::size_t pack_size = 1uL << log2_pack_size;

  // Get pack_type
  using pack_type = typename __pack__<float_type, pack_size>::value_type;

  // Get number of positions
  static const std::size_t n_pos = sizeof...(pos);

  // Check that all positions are positive numbers
  if (not [](auto &&...x) { return ((x >= 0) & ...); }(pos...)) return 1;

  // Check that all positions are above log2_pack_size
  if ([](auto &&...x) {
        return ((static_cast<std::size_t>(x) < log2_pack_size) + ...);
      }(pos...) != 0)
    return 1;

  // Recast to the right size
  auto *psi_re = reinterpret_cast<pack_type *>(psi_re_ptr);
  auto *psi_im = reinterpret_cast<pack_type *>(psi_im_ptr);
  const std::size_t state_size = state_size_ptr >> log2_pack_size;

  // Split in real and imaginary parts
  static const std::size_t U_size = 1uL << n_pos;
  const auto U_re = subset<0, 2 * U_size * U_size, 2>(U_ptr);
  const auto U_im = subset<1, 2 * U_size * U_size, 2>(U_ptr);

  // Shift positions
  const auto shift_pos = [](auto &&pos) {
    std::array<std::size_t, n_pos> shift_pos;
    for (std::size_t i = 0; i < n_pos; ++i)
      shift_pos[i] = pos[i] - log2_pack_size;
    return shift_pos;
  }(std::array{pos...});

  // Get zero
  static const auto _zero = __pack__<float_type, pack_size>::get(0);

#pragma omp parallel for
  for (std::size_t i = 0; i < (state_size >> n_pos); ++i) {
    // Get indexes to expand
    const auto _pos = expand(i, shift_pos);

    // Buffer real and imaginary parts from state
    const auto _psi_re = get(psi_re, _pos);
    const auto _psi_im = get(psi_im, _pos);

    // Compute matrix multiplication
    for (std::size_t i = 0; i < U_size; ++i) {
      auto _re{_zero};
      auto _im{_zero};
      for (std::size_t j = 0; j < U_size; ++j) {
        const auto _U_re = U_re[i * U_size + j];
        const auto _U_im = U_im[i * U_size + j];
        _re += _U_re * _psi_re[j] - _U_im * _psi_im[j];
        _im += _U_re * _psi_im[j] + _U_im * _psi_re[j];
      }
      psi_re[_pos[i]] = _re;
      psi_im[_pos[i]] = _im;
    }
  }

  return 0;
}

template <std::size_t log2_pack_size, typename float_type, typename Positions,
          std::size_t n_pos = array_size_v<Positions>, std::size_t... I>
int apply(float_type *psi_re_ptr, float_type *psi_im_ptr,
          const float_type *U_ptr, Positions &&pos,
          const std::size_t state_size_ptr, std::index_sequence<I...>) {
  return apply<log2_pack_size>(psi_re_ptr, psi_im_ptr, U_ptr, state_size_ptr,
                               pos[I]...);
}

template <std::size_t log2_pack_size, typename float_type, typename Positions,
          std::size_t n_pos = array_size_v<Positions>>
int apply(float_type *psi_re_ptr, float_type *psi_im_ptr,
          const float_type *U_ptr, Positions &&pos,
          const std::size_t state_size_ptr) {
  return apply<log2_pack_size>(psi_re_ptr, psi_im_ptr, U_ptr, pos,
                               state_size_ptr,
                               std::make_index_sequence<n_pos>{});
}

template <std::size_t log2_pack_size, typename float_type, typename index_type>
int apply(float_type *psi_re_ptr, float_type *psi_im_ptr,
          const float_type *U_ptr, const index_type *pos,
          const std::size_t state_size_ptr, const std::size_t n_pos) {
  // Check if pointer are correctly aligned
  if (reinterpret_cast<std::size_t>(psi_re_ptr) % 32 or
      reinterpret_cast<std::size_t>(psi_im_ptr) % 32)
    return 1;

  // Get size of pack
  const std::size_t pack_size = 1uL << log2_pack_size;

  // Get U_Size
  const std::size_t U_size = 1uL << n_pos;

  // Get pack_type
  using pack_type = typename __pack__<float_type, pack_size>::value_type;

  // Check that all positions are positive numbers
  for (std::size_t i = 0; i < n_pos; ++i)
    if (pos[i] < 0 or static_cast<std::size_t>(pos[i]) < log2_pack_size)
      return 1;

  // Recast to the right size
  auto *psi_re = reinterpret_cast<pack_type *>(psi_re_ptr);
  auto *psi_im = reinterpret_cast<pack_type *>(psi_im_ptr);
  const std::size_t state_size = state_size_ptr >> log2_pack_size;

  // Compute offset
  std::size_t offset[n_pos];
  for (std::size_t i = 0; i < n_pos; ++i) {
    offset[i] = log2_pack_size;
    for (std::size_t j = i + 1; j < n_pos; ++j) offset[i] += (pos[j] < pos[i]);
  }

  // Allocate buffers
  pack_type _psi_re[U_size];
  pack_type _psi_im[U_size];
  std::size_t _pos[U_size];

  // Generator of positions
  auto _get_position = [&pos, &offset, n_pos](std::size_t i, std::size_t j) {
    std::size_t y{i};
    for (std::size_t i = 0; i < n_pos; ++i) {
      const std::size_t p = pos[i] - offset[i];
      const std::size_t y_mask = (1uL << p) - 1;
      y = ((y & ~y_mask) << 1) ^ (y & y_mask) ^ (((j >> i) & 1uL) << p);
    }
    return y;
  };

#pragma omp parallel for private(_psi_re, _psi_im, _pos)
  for (std::size_t i = 0; i < (state_size >> n_pos); ++i) {
    // Get positions
    for (std::size_t j = 0; j < U_size; ++j) _pos[j] = _get_position(i, j);

    // Load buffer
    for (std::size_t j = 0; j < U_size; ++j) {
      _psi_re[j] = psi_re[_pos[j]];
      _psi_im[j] = psi_im[_pos[j]];
    }

    // Compute matrix multiplication
    for (std::size_t i = 0; i < U_size; ++i) {
      auto _re = pack_type{0};
      auto _im = pack_type{0};
      for (std::size_t j = 0; j < U_size; ++j) {
        const auto _U_re = U_ptr[2 * i * U_size + 2 * j];
        const auto _U_im = U_ptr[2 * i * U_size + 2 * j + 1];
        _re += _U_re * _psi_re[j] - _U_im * _psi_im[j];
        _im += _U_re * _psi_im[j] + _U_im * _psi_re[j];
      }
      // Update state
      psi_re[_pos[i]] = _re;
      psi_im[_pos[i]] = _im;
    }
  }

  return 0;
}

}  // namespace hybridq::U

#endif
