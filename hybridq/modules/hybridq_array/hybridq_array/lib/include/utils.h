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

#ifndef HYBRIDQ__UTILS_H
#define HYBRIDQ__UTILS_H

#include <array>
#include <cstdlib>

namespace hybridq {

template <typename T>
T* aligned_alloc(std::size_t alignment, std::size_t size) {
  void* p_ptr{nullptr};
  if (posix_memalign(&p_ptr, alignment, sizeof(T) * size)) p_ptr = nullptr;
  return static_cast<T*>(p_ptr);
}

// -----------------------

template <typename T>
struct remove_pcvr {
  using type = typename std::remove_cv<
      typename std::remove_pointer<typename std::remove_reference<
          typename std::remove_cv<T>::type>::type>::type>::type;
};

template <typename T>
using remove_pcvr_t = typename remove_pcvr<T>::type;

// -----------------------

template <typename Array>
struct array_size {
  static const std::size_t value = std::size(remove_pcvr_t<Array>{});
};
template <typename Array>
constexpr inline std::size_t array_size_v = array_size<Array>::value;

// -----------------------

template <typename Pack>
struct pack_size {
  using value_type = decltype(remove_pcvr_t<Pack>{}[0]);
  static const std::size_t value =
      sizeof(remove_pcvr_t<Pack>) / sizeof(value_type);
};
template <typename Pack>
constexpr inline std::size_t pack_size_v = pack_size<Pack>::value;

// -----------------------

template <typename Array>
struct array_value {
  using type = typename remove_pcvr_t<Array>::value_type;
};
template <typename Array>
using array_value_t = typename array_value<Array>::type;

// -----------------------

template <typename Positions, std::size_t N = array_size_v<Positions>>
constexpr inline auto expand(std::size_t x, Positions&& pos) {
  // Count number smaller
  const auto shift = [&pos]() {
    std::array<std::size_t, N> shift{};
    for (std::size_t i = 0; i < N; ++i) {
      shift[i] = 0;
      for (std::size_t j = i + 1; j < N; ++j) shift[i] += (pos[j] < pos[i]);
    }
    return shift;
  }();

  if constexpr (N > 0) {
    auto _expand = [x, &pos, &shift](std::size_t mask) {
      std::size_t y{x};
      for (std::size_t i = 0; i < N; ++i) {
        const std::size_t p = pos[i] - shift[i];
        const std::size_t y_mask = (1uL << p) - 1;
        y = ((y & ~y_mask) << 1) ^ (y & y_mask) ^ (((mask >> i) & 1uL) << p);
      }
      return y;
    };
    std::array<std::size_t, 1uL << N> out{};
    for (std::size_t i = 0; i < 1uL << N; ++i) out[i] = _expand(i);
    return out;
  } else
    return std::array{x};
}

// -----------------------

template <typename Array, typename Positions,
          std::size_t N = array_size_v<Positions>>
constexpr inline auto get(Array&& array, Positions&& pos) {
  using value_type = remove_pcvr_t<decltype(array[0])>;
  std::array<value_type, N> out{};
  for (std::size_t i = 0; i < N; ++i) out[i] = array[pos[i]];
  return out;
}

template <typename Array, typename Values, typename Positions,
          std::size_t N = array_size_v<Positions>>
constexpr inline auto put(Array&& array, Values&& values, Positions&& pos) {
  for (std::size_t i = 0; i < N; ++i) array[pos[i]] = values[i];
}

// -----------------------

template <std::size_t mask_size>
struct swap_bits {
  template <std::size_t type, std::size_t mask, std::size_t x>
  static constexpr auto _get_swap_pos() {
    std::size_t y{0}, nl{0}, nr{0};
    const std::size_t q{mask_size - __builtin_popcount(mask)};
    for (std::size_t i = 0; i < mask_size; ++i)
      if constexpr (type == 0)
        y ^= ((mask >> i) & 1uL) ? ((x >> i) & 1uL) << (q + nr++)
                                 : ((x >> i) & 1uL) << nl++;
      else if constexpr (type == 1)
        y ^= ((mask >> i) & 1uL) ? ((x >> (q + nr++)) & 1uL) << i
                                 : ((x >> nl++) & 1uL) << i;
      else
        throw "Error";
    return y;
  }

  template <std::size_t type, std::size_t I, std::size_t... x>
  static constexpr auto _get_swap_poss(std::index_sequence<x...>) {
    return std::array{_get_swap_pos<type, I, x>()...};
  }

  template <std::size_t type, std::size_t... I>
  static constexpr auto _get_values(std::index_sequence<I...>) {
    return std::array{_get_swap_poss<type, I>(
        std::make_index_sequence<1uL << mask_size>{})...};
  }

  static constexpr auto values_fw{
      _get_values<0>(std::make_index_sequence<1uL << mask_size>{})};
  static constexpr auto values_bw{
      _get_values<1>(std::make_index_sequence<1uL << mask_size>{})};
};

template <std::size_t type, typename pack_type,
          std::size_t pack_size = pack_size_v<pack_type>,
          std::size_t mask_size = __builtin_ctz(pack_size), std::size_t... I>
constexpr inline void _swap_bits_pack(pack_type&& pack, std::size_t mask,
                                      std::index_sequence<I...>) {
  const auto& ps = [&mask]() {
    if constexpr (type == 0)
      return swap_bits<mask_size>::values_fw[mask];
    else if constexpr (type == 1)
      return swap_bits<mask_size>::values_bw[mask];
    else
      ;
  }();
  pack = remove_pcvr_t<pack_type>{pack[ps[I]]...};
}

template <std::size_t type, typename pack_type,
          std::size_t pack_size = pack_size_v<pack_type>>
constexpr inline void swap_bits_pack(pack_type&& pack, std::size_t mask) {
  _swap_bits_pack<type>(pack, mask, std::make_index_sequence<pack_size>{});
}

// -----------------------

template <std::size_t start, std::size_t step, typename Array, std::size_t... I>
constexpr inline auto _subset(Array&& array, std::index_sequence<I...>) {
  return std::array{array[start + step * I]...};
}

template <std::size_t start, std::size_t stop, std::size_t step, typename Array>
constexpr inline auto subset(Array&& array) {
  return _subset<start, step>(
      array,
      std::make_index_sequence<(stop - start) / step + ((stop - start) % 2)>{});
}

// -----------------------

template <std::size_t N, typename Array>
constexpr inline auto to_array(Array&& array) {
  using value_type = remove_pcvr_t<decltype(array[0])>;
  std::array<value_type, N> out{};
  for (std::size_t i = 0; i < N; ++i) out[i] = array[i];
  return out;
}

}  // namespace hybridq

#endif
