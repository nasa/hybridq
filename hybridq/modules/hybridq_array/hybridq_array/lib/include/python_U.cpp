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

#ifndef HYBRIDQ_PYTHON_H
#define HYBRIDQ_PYTHON_H

#include <unordered_map>

#include "U.h"
#include "utils.h"

inline constexpr std::size_t log2_pack_size = LOG2_PACK_SIZE;
inline constexpr std::size_t max_log2_pack_size = 5;

namespace hybridq::python {

template <std::size_t log2_pack_size, typename float_type>
int apply_U(float_type *psi_re_ptr, float_type *psi_im_ptr,
            const float_type *U_ptr, const unsigned int *pos,
            const unsigned int n_qubits, const unsigned int n_pos) {
  switch (n_pos) {
    case 0:
      return 0;
    case 1:
      return U::apply<log2_pack_size>(psi_re_ptr, psi_im_ptr, U_ptr,
                                      hybridq::to_array<1>(pos),
                                      1uL << n_qubits);
    case 2:
      return U::apply<log2_pack_size>(psi_re_ptr, psi_im_ptr, U_ptr,
                                      hybridq::to_array<2>(pos),
                                      1uL << n_qubits);
    case 3:
      return U::apply<log2_pack_size>(psi_re_ptr, psi_im_ptr, U_ptr,
                                      hybridq::to_array<3>(pos),
                                      1uL << n_qubits);
    case 4:
      return U::apply<log2_pack_size>(psi_re_ptr, psi_im_ptr, U_ptr,
                                      hybridq::to_array<4>(pos),
                                      1uL << n_qubits);
    default:
      return hybridq::U::apply<log2_pack_size>(psi_re_ptr, psi_im_ptr, U_ptr,
                                               pos, 1uL << n_qubits, n_pos);
  }
}

template <typename X, typename... Y>
auto min(X &&x, Y &&...y) {
  if constexpr (sizeof...(y)) {
    const auto _y = min(y...);
    return x < _y ? x : _y;
  } else
    return x;
}

template <typename X, typename... Y>
auto max(X &&x, Y &&...y) {
  if constexpr (sizeof...(y)) {
    const auto _y = min(y...);
    return x > _y ? x : _y;
  } else
    return x;
}

template <typename float_type>
int apply_U(float_type *psi_re_ptr, float_type *psi_im_ptr,
            const float_type *U_ptr, const unsigned int *pos,
            const unsigned int n_qubits, const unsigned int n_pos) {
  // Get minimum position
  const std::size_t min_pos = [pos, n_pos]() {
    std::size_t min_pos = ~0;
    for (std::size_t i = 0; i < n_pos; ++i)
      if (pos[i] < min_pos) min_pos = pos[i];
    return min_pos;
  }();

  switch (min(max(min_pos, log2_pack_size), max_log2_pack_size)) {
    case 1:
      return hybridq::python::apply_U<1>(psi_re_ptr, psi_im_ptr, U_ptr, pos,
                                         n_qubits, n_pos);
    case 2:
      return hybridq::python::apply_U<2>(psi_re_ptr, psi_im_ptr, U_ptr, pos,
                                         n_qubits, n_pos);
    case 3:
      return hybridq::python::apply_U<3>(psi_re_ptr, psi_im_ptr, U_ptr, pos,
                                         n_qubits, n_pos);
    case 4:
      return hybridq::python::apply_U<4>(psi_re_ptr, psi_im_ptr, U_ptr, pos,
                                         n_qubits, n_pos);
    case 5:
      return hybridq::python::apply_U<5>(psi_re_ptr, psi_im_ptr, U_ptr, pos,
                                         n_qubits, n_pos);
    default:
      return hybridq::python::apply_U<5>(psi_re_ptr, psi_im_ptr, U_ptr, pos,
                                         n_qubits, n_pos);
  }
}

template <typename float_type>
int to_complex(float_type *psi_re_ptr, float_type *psi_im_ptr,
               float_type *psi_ptr, const unsigned int size) {
#pragma omp parallel for
  for (std::size_t i = 0; i < size; ++i) {
    psi_ptr[2 * i + 0] = psi_re_ptr[i];
    psi_ptr[2 * i + 1] = psi_im_ptr[i];
  }
  return 0;
}

}  // namespace hybridq::python

extern "C" {

// Check sizes
static_assert(sizeof(float) * 8 == 32);
static_assert(sizeof(double) * 8 == 64);
static_assert(sizeof(long double) * 8 == 128);

unsigned int get_log2_pack_size() { return log2_pack_size; }

int apply_U_float32(float *psi_re_ptr, float *psi_im_ptr, const float *U_ptr,
                    const unsigned int *pos, const unsigned int n_qubits,
                    const unsigned int n_pos) {
  return hybridq::python::apply_U(psi_re_ptr, psi_im_ptr, U_ptr, pos, n_qubits,
                                  n_pos);
}

int apply_U_float64(double *psi_re_ptr, double *psi_im_ptr, const double *U_ptr,
                    const unsigned int *pos, const unsigned int n_qubits,
                    const unsigned int n_pos) {
  return hybridq::python::apply_U(psi_re_ptr, psi_im_ptr, U_ptr, pos, n_qubits,
                                  n_pos);
}

int apply_U_float128(long double *psi_re_ptr, long double *psi_im_ptr,
                     const long double *U_ptr, const unsigned int *pos,
                     const unsigned int n_qubits, const unsigned int n_pos) {
  return hybridq::python::apply_U(psi_re_ptr, psi_im_ptr, U_ptr, pos, n_qubits,
                                  n_pos);
}

int to_complex64(float *psi_re_ptr, float *psi_im_ptr, float *psi_ptr,
                 const unsigned int size) {
  return hybridq::python::to_complex(psi_re_ptr, psi_im_ptr, psi_ptr, size);
}

int to_complex128(double *psi_re_ptr, double *psi_im_ptr, double *psi_ptr,
                  const unsigned int size) {
  return hybridq::python::to_complex(psi_re_ptr, psi_im_ptr, psi_ptr, size);
}
}

#endif
