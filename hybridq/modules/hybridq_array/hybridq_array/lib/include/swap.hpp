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

#ifndef HYBRIDQ_ARRAY_SWAP__HPP
#define HYBRIDQ_ARRAY_SWAP__HPP

#include <omp.h>

#include <array>
#include <cstdint>
#include <cstdlib>

extern "C" int32_t swap(void *array, const uint32_t *pos,
                        const uint32_t n_qubits, const uint32_t num_threads);

#endif
