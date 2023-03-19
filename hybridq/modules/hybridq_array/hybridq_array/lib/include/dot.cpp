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

#include "dot.h"

namespace hybridq_new {

template <std::size_t... I>
inline static constexpr auto get_pack(const array_type *array,
                                      std::index_sequence<I...>) {
  return pack_type{array[I]...};
}

inline static constexpr auto get_pack(const array_type *array) {
  return get_pack(array, std::make_index_sequence<pack_size>{});
}

template <std::size_t... I>
inline static constexpr auto split_pack(const array_type *array,
                                        std::index_sequence<I...>) {
  return std::array<pack_type, 2>{pack_type{array[2 * I + 0]...},
                                  pack_type{array[2 * I + 1]...}};
}

inline static constexpr auto split_pack(const array_type *array) {
  return split_pack(array, std::make_index_sequence<pack_size>{});
}

template <typename Positions>
constexpr inline auto expand(std::size_t x, Positions &&pos) {
  // For each position p_i, count the number of positions
  // p_j, with j > i, such that p_i > p_j.
  std::array<std::size_t, n_pos> shift;
  for (std::size_t i = 0; i < n_pos; ++i) {
    shift[i] = 0;
    for (std::size_t j = i + 1; j < n_pos; ++j) shift[i] += (pos[j] < pos[i]);
  }

  auto _expand = [x, &pos, &shift](std::size_t mask) {
    std::size_t y{x};
    for (std::size_t i = 0; i < n_pos; ++i) {
      const std::size_t p = pos[i] - shift[i];
      const std::size_t y_mask = (1uL << p) - 1;
      y = ((y & ~y_mask) << 1) ^ (y & y_mask) ^ (((mask >> i) & 1uL) << p);
    }
    return y;
  };
  std::array<std::size_t, 1uL << n_pos> out;
  for (std::size_t i = 0; i < 1uL << n_pos; ++i) out[i] = _expand(i);

  // Return expanded positions
  return out;
}

extern "C" {

int32_t apply_cc(array_type *psi, const array_type *U, const uint32_t *pos,
                 const uint32_t n_qubits) {
  // Check that psi is not empty
  if (psi == nullptr) return 1;

  // Check that U is not empty
  if (U == nullptr) return 2;

  // Check that pos is not empty
  if (pos == nullptr) return 3;

  // Get real and imaginary parts of
  auto *U_re =
      static_cast<array_type *>(malloc(sizeof(array_type) * U_size * U_size));
  auto *U_im =
      static_cast<array_type *>(malloc(sizeof(array_type) * U_size * U_size));
  for (std::size_t i = 0; i < U_size * U_size; ++i) {
    U_re[i] = U[2 * i + 0];
    U_im[i] = U[2 * i + 1];
  }

  // Shift positions
  const auto shift_pos = [pos]() {
    std::array<std::size_t, n_pos> shift_pos;
    for (std::size_t i = 0; i < n_pos; ++i)
      shift_pos[i] = pos[i] - log2_pack_size;
    return shift_pos;
  }();

  // Get number of elements to compute
  const std::size_t psi_size = 1uLL << (n_qubits - log2_pack_size - n_pos);

#pragma omp parallel for
  for (std::size_t x = 0; x < psi_size; ++x) {
    // Get indexes
    const auto pos_ = expand(x, shift_pos);

    // Buffer psi
    std::array<std::array<pack_type, 2>, U_size> buffer_;
    for (std::size_t i = 0; i < U_size; ++i)
      buffer_[i] = split_pack(psi + 2 * (pos_[i] << log2_pack_size));

    // For each row ...
    for (std::size_t i = 0; i < U_size; ++i) {
      // Initialize real and imaginary part
      auto _re{zero};
      auto _im{zero};

      // Multiply row and psi ...
      for (std::size_t j = 0; j < U_size; ++j) {
        const auto U_re_ = U_re[i * U_size + j];
        const auto U_im_ = U_im[i * U_size + j];
        _re += U_re_ * buffer_[j][0] - U_im_ * buffer_[j][1];
        _im += U_re_ * buffer_[j][1] + U_im_ * buffer_[j][0];
      }

      // Dump real and imaginary part
      auto psi_red_ = psi + 2 * (pos_[i] << log2_pack_size);
      for (std::size_t j = 0; j < pack_size; ++j) {
        psi_red_[2 * j + 0] = _re[j];
        psi_red_[2 * j + 1] = _im[j];
      }
    }
  }

  // Free memory
  free(U_re);
  free(U_im);

  // Everything OK
  return 0;
}

int32_t apply_cr(array_type *psi, const array_type *U_re, const uint32_t *pos,
                 const uint32_t n_qubits) {
  // Check that psi is not empty
  if (psi == nullptr) return 1;

  // Check that U is not empty
  if (U_re == nullptr) return 2;

  // Check that pos is not empty
  if (pos == nullptr) return 3;

  // Shift positions
  const auto shift_pos = [pos]() {
    std::array<std::size_t, n_pos> shift_pos;
    for (std::size_t i = 0; i < n_pos; ++i)
      shift_pos[i] = pos[i] - log2_pack_size;
    return shift_pos;
  }();

  // Get number of elements to compute
  const std::size_t psi_size = 1uLL << (n_qubits - log2_pack_size - n_pos);

#pragma omp parallel for
  for (std::size_t x = 0; x < psi_size; ++x) {
    // Get indexes
    const auto pos_ = expand(x, shift_pos);

    // Buffer psi
    std::array<std::array<pack_type, 2>, U_size> buffer_;
    for (std::size_t i = 0; i < U_size; ++i)
      buffer_[i] = split_pack(psi + 2 * (pos_[i] << log2_pack_size));

    // For each row ...
    for (std::size_t i = 0; i < U_size; ++i) {
      // Initialize real and imaginary part
      auto _re{zero};
      auto _im{zero};

      // Multiply row and psi ...
      for (std::size_t j = 0; j < U_size; ++j) {
        const auto U_re_ = U_re[i * U_size + j];
        _re += U_re_ * buffer_[j][0];
        _im += U_re_ * buffer_[j][1];
      }

      // Dump real and imaginary part
      auto psi_red_ = psi + 2 * (pos_[i] << log2_pack_size);
      for (std::size_t j = 0; j < pack_size; ++j) {
        psi_red_[2 * j + 0] = _re[j];
        psi_red_[2 * j + 1] = _im[j];
      }
    }
  }

  return 0;
}

int32_t apply_rr(array_type *psi_re, const array_type *U_re,
                 const uint32_t *pos, const uint32_t n_qubits) {
  // Check that psi is not empty
  if (psi_re == nullptr) return 1;

  // Check that U is not empty
  if (U_re == nullptr) return 2;

  // Shift positions
  const auto shift_pos = [pos]() {
    std::array<std::size_t, n_pos> shift_pos;
    for (std::size_t i = 0; i < n_pos; ++i)
      shift_pos[i] = pos[i] - log2_pack_size;
    return shift_pos;
  }();

  // Get number of elements to compute
  const std::size_t psi_size = 1uLL << (n_qubits - log2_pack_size - n_pos);

#pragma omp parallel for
  for (std::size_t x = 0; x < psi_size; ++x) {
    // Get indexes
    const auto pos_ = expand(x, shift_pos);

    // Buffer psi
    std::array<pack_type, U_size> buffer_;
    for (std::size_t i = 0; i < U_size; ++i)
      buffer_[i] = get_pack(psi_re + (pos_[i] << log2_pack_size));

    // For each row ...
    for (std::size_t i = 0; i < U_size; ++i) {
      // Initialize real and imaginary part
      auto _re{zero};

      // Multiply row and psi ...
      for (std::size_t j = 0; j < U_size; ++j) {
        const auto U_re_ = U_re[i * U_size + j];
        _re += U_re_ * buffer_[j];
      }

      // Dump real and imaginary part
      auto psi_red_ = psi_re + (pos_[i] << log2_pack_size);
      for (std::size_t j = 0; j < pack_size; ++j) {
        psi_red_[j] = _re[j];
      }
    }
  }

  return 0;
}
}

}  // namespace hybridq_new
