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

#ifndef HYBRIDQ_GLOBAL_H
#define HYBRIDQ_GLOBAL_H

#include <immintrin.h>

#include <stdexcept>
#include <utility>

namespace hybridq {

// Define pack structure
template <typename _base_type, std::size_t _size>
struct __pack__ {
  using base_type = _base_type;
  static constexpr std::size_t size = _size;

  typedef base_type value_type
      __attribute__((vector_size(sizeof(base_type) * size)));

  static constexpr value_type get(base_type value) {
    return _get(value, std::make_index_sequence<size>{});
  }

 private:
  template <std::size_t... I>
  static constexpr value_type _get(base_type value, std::index_sequence<I...>) {
    return value_type{(static_cast<void>(I), value)...};
  }
};

}  // namespace hybridq

#endif
