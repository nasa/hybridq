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

#include <iostream>

#include "pack.h"
#include "swap.h"
#include "utils.h"

namespace hybridq::python {

template <typename float_type>
int swap(float_type *array, const unsigned int *pos,
         const unsigned int n_qubits, const unsigned int n_pos) {
  switch (n_pos) {
    case 0:
      return 0;
    case 1:
      return hybridq::swap::swap_array(array, hybridq::to_array<1>(pos),
                                       1uL << n_qubits);
    case 2:
      return hybridq::swap::swap_array(array, hybridq::to_array<2>(pos),
                                       1uL << n_qubits);
    case 3:
      return hybridq::swap::swap_array(array, hybridq::to_array<3>(pos),
                                       1uL << n_qubits);
    case 4:
      return hybridq::swap::swap_array(array, hybridq::to_array<4>(pos),
                                       1uL << n_qubits);
    case 5:
      return hybridq::swap::swap_array(array, hybridq::to_array<5>(pos),
                                       1uL << n_qubits);
    case 6:
      return hybridq::swap::swap_array(array, hybridq::to_array<6>(pos),
                                       1uL << n_qubits);
    case 7:
      return hybridq::swap::swap_array(array, hybridq::to_array<7>(pos),
                                       1uL << n_qubits);
    case 8:
      return hybridq::swap::swap_array(array, hybridq::to_array<8>(pos),
                                       1uL << n_qubits);
    default:
      return hybridq::swap::swap_array(array, pos, 1uL << n_qubits, n_pos);
  }
}

}  // namespace hybridq::python

extern "C" {

int swap_float32(float *array, const unsigned int *pos,
                 const unsigned int n_qubits, const unsigned int n_pos) {
  return hybridq::python::swap(array, pos, n_qubits, n_pos);
}

int swap_float64(double *array, const unsigned int *pos,
                 const unsigned int n_qubits, const unsigned int n_pos) {
  return hybridq::python::swap(array, pos, n_qubits, n_pos);
}

int swap_int32(int *array, const unsigned int *pos, const unsigned int n_qubits,
               const unsigned int n_pos) {
  return hybridq::python::swap(array, pos, n_qubits, n_pos);
}

int swap_int64(long *array, const unsigned int *pos,
               const unsigned int n_qubits, const unsigned int n_pos) {
  return hybridq::python::swap(array, pos, n_qubits, n_pos);
}

int swap_uint32(unsigned int *array, const unsigned int *pos,
                const unsigned int n_qubits, const unsigned int n_pos) {
  return hybridq::python::swap(array, pos, n_qubits, n_pos);
}

int swap_uint64(unsigned long *array, const unsigned int *pos,
                const unsigned int n_qubits, const unsigned int n_pos) {
  return hybridq::python::swap(array, pos, n_qubits, n_pos);
}
}

#endif
