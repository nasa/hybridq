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

#ifndef HYBRIDQ_ARRAY_UTILS__HPP
#define HYBRIDQ_ARRAY_UTILS__HPP

#include <cstdlib>

namespace hybridq_new {

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

// Get type given index and nbits
template <int type, std::size_t nbits>
struct get_type {};

template <>
struct get_type<0, 8 * 2> {
  using type = _Float16;
  static_assert(sizeof(type) == 2, "Wrong number of bytes");
};

template <>
struct get_type<0, 8 * 4> {
  using type = float;
  static_assert(sizeof(type) == 4, "Wrong number of bytes");
};

template <>
struct get_type<0, 8 * 8> {
  using type = double;
  static_assert(sizeof(type) == 8, "Wrong number of bytes");
};

template <>
struct get_type<0, 8 * 16> {
  using type = _Float128;
  static_assert(sizeof(type) == 16, "Wrong number of bytes");
};

template <>
struct get_type<1, 8 * 1> {
  using type = uint8_t;
  static_assert(sizeof(type) == 1, "Wrong number of bytes");
};

template <>
struct get_type<1, 8 * 2> {
  using type = uint16_t;
  static_assert(sizeof(type) == 2, "Wrong number of bytes");
};

template <>
struct get_type<1, 8 * 4> {
  using type = uint32_t;
  static_assert(sizeof(type) == 4, "Wrong number of bytes");
};

template <>
struct get_type<1, 8 * 8> {
  using type = uint64_t;
  static_assert(sizeof(type) == 8, "Wrong number of bytes");
};

template <>
struct get_type<1, 8 * 16> {
  using type = unsigned __int128;
  static_assert(sizeof(type) == 16, "Wrong number of bytes");
};

template <>
struct get_type<2, 8 * 1> {
  using type = int8_t;
  static_assert(sizeof(type) == 1, "Wrong number of bytes");
};

template <>
struct get_type<2, 8 * 2> {
  using type = int16_t;
  static_assert(sizeof(type) == 2, "Wrong number of bytes");
};

template <>
struct get_type<2, 8 * 4> {
  using type = int32_t;
  static_assert(sizeof(type) == 4, "Wrong number of bytes");
};

template <>
struct get_type<2, 8 * 8> {
  using type = int64_t;
  static_assert(sizeof(type) == 8, "Wrong number of bytes");
};

template <>
struct get_type<2, 8 * 16> {
  using type = __int128;
  static_assert(sizeof(type) == 16, "Wrong number of bytes");
};

}  // namespace hybridq_new

#endif
