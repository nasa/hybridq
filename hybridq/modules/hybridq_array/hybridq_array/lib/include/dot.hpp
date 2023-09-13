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

#ifndef HYBRIDQ_ARRAY_DOT__HPP
#define HYBRIDQ_ARRAY_DOT__HPP

#include <omp.h>

#include <array>
#include <cstdint>
#include <cstdlib>

#include "type.hpp"

namespace hybridq {

// Set  array type
using array_type =
    get_type<HYBRIDQ_ARRAY_DOT_ARRAY_TYPE, HYBRIDQ_ARRAY_DOT_ARRAY_NBITS>::type;

// Get number of positions
static constexpr std::size_t n_pos = HYBRIDQ_ARRAY_DOT_NPOS;

// Get size of U
static constexpr std::size_t U_size = 1uLL << n_pos;

// Set log2 size of the pack
static constexpr std::size_t log2_pack_size = HYBRIDQ_ARRAY_DOT_LOG2_PACK_SIZE;

// Define size of the pack
static const std::size_t pack_size = 1uL << log2_pack_size;

// Get types
typedef array_type pack_type
    __attribute__((vector_size(sizeof(array_type) * pack_size)));
typedef array_type cpack_type
    __attribute__((vector_size(sizeof(array_type) * 2 * pack_size)));

// Define union two manipulate real and imaginary part
typedef union {
  pack_type pp[2];
  cpack_type c;
} ctype;

extern "C" int64_t apply_qc(array_type *psi_re, array_type *psi_im,
                            const array_type *U, const uint64_t *pos,
                            const uint64_t n_qubits,
                            const uint64_t num_threads);

extern "C" int64_t apply_qr(array_type *psi_re, array_type *psi_im,
                            const array_type *U_re, const uint64_t *pos,
                            const uint64_t n_qubits,
                            const uint64_t num_threads);

extern "C" int64_t apply_rr(array_type *psi_re, const array_type *U_re,
                            const uint64_t *pos, const uint64_t n_qubits,
                            const uint64_t num_threads);

extern "C" int64_t apply_cc(array_type *psi, const array_type *U,
                            const uint64_t *pos, const uint64_t n_qubits,
                            const uint64_t num_threads);

extern "C" int64_t apply_cr(array_type *psi, const array_type *U_re,
                            const uint64_t *pos, const uint64_t n_qubits,
                            const uint64_t num_threads);

}  // namespace hybridq

#endif