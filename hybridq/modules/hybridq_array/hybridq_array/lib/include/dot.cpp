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

#include "dot.hpp"

namespace hybridq {

// Get zero
template <std::size_t size, std::size_t... I>
static constexpr auto zero(std::index_sequence<I...>) {
  typedef array_type _type
      __attribute__((vector_size(sizeof(array_type) * size)));
  return _type{(static_cast<void>(I), 0)...};
}
template <std::size_t size>
static constexpr auto zero() {
  return zero<size>(std::make_index_sequence<size>{});
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

template <std::size_t... I>
inline static constexpr auto get_pack(const array_type *array,
                                      std::index_sequence<I...>) {
  return pack_type{array[I]...};
}

inline static constexpr auto get_pack(const array_type *array) {
  return get_pack(array, std::make_index_sequence<pack_size>{});
}

template <std::size_t... I>
auto shuffle_fw(const ctype &x, std::index_sequence<I...>) {
  return ctype{x.c[2 * I + 0]..., x.c[2 * I + 1]...};
}

auto shuffle_fw(const ctype &x) {
  return shuffle_fw(x, std::make_index_sequence<pack_size>{});
}

template <std::size_t... I>
auto shuffle_fw(const array_type *x, std::index_sequence<I...>) {
  return ctype{x[2 * I + 0]..., x[2 * I + 1]...};
}

auto shuffle_fw(const array_type *x) {
  return shuffle_fw(x, std::make_index_sequence<pack_size>{});
}

template <std::size_t... I>
auto shuffle_bw(const ctype &x, std::index_sequence<I...>) {
  return ctype{
      x.c[I == 2 * pack_size - 1 ? 2 * pack_size - 1
                                 : (I * pack_size) % (2 * pack_size - 1)]...};
}

auto shuffle_bw(const ctype &x) {
  return shuffle_bw(x, std::make_index_sequence<2 * pack_size>{});
}

template <bool aligned>
int64_t apply_qc(array_type *psi_re, array_type *psi_im, const array_type *U,
                 const uint64_t *pos, const uint64_t n_qubits,
                 const uint64_t num_threads = 0) {
  // Check that psi is not empty
  if (psi_re == nullptr or psi_im == nullptr) return 1;

  // Check that U is not empty
  if (U == nullptr) return 2;

  // Check that pos is not empty
  if (pos == nullptr) return 3;

  // Check if pointer are correctly aligned
  if constexpr (aligned)
    if (reinterpret_cast<std::size_t>(psi_re) % 32 or
        reinterpret_cast<std::size_t>(psi_im) % 32)
      return 4;

  // Get number of threads
  std::size_t num_threads_ = omp_get_max_threads();

  // If 'num_threads' is provided, use it
  if (num_threads > 0) num_threads_ = num_threads;

  // Otherwise, check if env variable is provided
  else if (const auto _nt = std::getenv("HYBRIDQ_ARRAY_NUM_THREADS"))
    num_threads_ = atoi(_nt);

  // Recast to the right size
  auto *psi_re_ = reinterpret_cast<pack_type *>(psi_re);
  auto *psi_im_ = reinterpret_cast<pack_type *>(psi_im);

  // Split in real and imaginary parts
  static const std::size_t U_size = 1uL << n_pos;

  // Get real and imaginary parts of
  std::array<array_type, U_size * U_size> U_re, U_im;
  for (std::size_t i = 0; i < U_size * U_size; ++i) {
    U_re[i] = U[2 * i + 0];
    U_im[i] = U[2 * i + 1];
  }

  // Shift positions
  std::array<std::size_t, n_pos> shift_pos;
  for (std::size_t i = 0; i < n_pos; ++i)
    shift_pos[i] = pos[i] - log2_pack_size;

#pragma omp parallel for num_threads(num_threads_)
  for (std::size_t i = 0; i < (1uLL << (n_qubits - log2_pack_size - n_pos));
       ++i) {
    // Get indexes to expand
    const auto pos_ = expand(i, shift_pos);

    // Buffer real and imaginary parts from state
    std::array<pack_type, U_size> buffer_re_;
    std::array<pack_type, U_size> buffer_im_;
    for (std::size_t i = 0; i < U_size; ++i) {
      if constexpr (aligned) {
        buffer_re_[i] = psi_re_[pos_[i]];
        buffer_im_[i] = psi_im_[pos_[i]];
      } else {
        buffer_re_[i] = get_pack(psi_re + (pos_[i] << log2_pack_size));
        buffer_im_[i] = get_pack(psi_im + (pos_[i] << log2_pack_size));
      }
    }

    // Compute matrix multiplication
    for (std::size_t i = 0; i < U_size; ++i) {
      auto re_{zero<pack_size>()};
      auto im_{zero<pack_size>()};
      for (std::size_t j = 0; j < U_size; ++j) {
        const auto U_re_ = U_re[i * U_size + j];
        const auto U_im_ = U_im[i * U_size + j];
        re_ += U_re_ * buffer_re_[j] - U_im_ * buffer_im_[j];
        im_ += U_re_ * buffer_im_[j] + U_im_ * buffer_re_[j];
      }

      // Dump real and imaginary part
      if constexpr (aligned) {
        psi_re_[pos_[i]] = re_;
        psi_im_[pos_[i]] = im_;
      } else {
        auto psi_re_red_ = psi_re + (pos_[i] << log2_pack_size);
        auto psi_im_red_ = psi_im + (pos_[i] << log2_pack_size);
        for (std::size_t j = 0; j < pack_size; ++j) {
          psi_re_red_[j] = re_[j];
          psi_im_red_[j] = im_[j];
        }
      }
    }
  }

  // Everything is OK
  return 0;
}

template <bool aligned>
int64_t apply_qr(array_type *psi_re, array_type *psi_im, const array_type *U_re,
                 const uint64_t *pos, const uint64_t n_qubits,
                 const uint64_t num_threads = 0) {
  // Check that psi is not empty
  if (psi_re == nullptr or psi_im == nullptr) return 1;

  // Check that U is not empty
  if (U_re == nullptr) return 2;

  // Check that pos is not empty
  if (pos == nullptr) return 3;

  // Check if pointer are correctly aligned
  if constexpr (aligned)
    if (reinterpret_cast<std::size_t>(psi_re) % 32 or
        reinterpret_cast<std::size_t>(psi_im) % 32)
      return 4;

  // Get number of threads
  std::size_t num_threads_ = omp_get_max_threads();

  // If 'num_threads' is provided, use it
  if (num_threads > 0) num_threads_ = num_threads;

  // Otherwise, check if env variable is provided
  else if (const auto _nt = std::getenv("HYBRIDQ_ARRAY_NUM_THREADS"))
    num_threads_ = atoi(_nt);

  // Recast to the right size
  auto *psi_re_ = reinterpret_cast<pack_type *>(psi_re);
  auto *psi_im_ = reinterpret_cast<pack_type *>(psi_im);

  // Split in real and imaginary parts
  static const std::size_t U_size = 1uL << n_pos;

  // Shift positions
  std::array<std::size_t, n_pos> shift_pos;
  for (std::size_t i = 0; i < n_pos; ++i)
    shift_pos[i] = pos[i] - log2_pack_size;

#pragma omp parallel for num_threads(num_threads_)
  for (std::size_t i = 0; i < (1uLL << (n_qubits - log2_pack_size - n_pos));
       ++i) {
    // Get indexes to expand
    const auto pos_ = expand(i, shift_pos);

    // Buffer real and imaginary parts from state
    std::array<pack_type, U_size> buffer_re_;
    std::array<pack_type, U_size> buffer_im_;
    for (std::size_t i = 0; i < U_size; ++i) {
      if constexpr (aligned) {
        buffer_re_[i] = psi_re_[pos_[i]];
        buffer_im_[i] = psi_im_[pos_[i]];
      } else {
        buffer_re_[i] = get_pack(psi_re + (pos_[i] << log2_pack_size));
        buffer_im_[i] = get_pack(psi_im + (pos_[i] << log2_pack_size));
      }
    }

    // Compute matrix multiplication
    for (std::size_t i = 0; i < U_size; ++i) {
      auto re_{zero<pack_size>()};
      auto im_{zero<pack_size>()};
      for (std::size_t j = 0; j < U_size; ++j) {
        const auto U_re_ = U_re[i * U_size + j];
        re_ += U_re_ * buffer_re_[j];
        im_ += U_re_ * buffer_im_[j];
      }

      // Dump real and imaginary part
      if constexpr (aligned) {
        psi_re_[pos_[i]] = re_;
        psi_im_[pos_[i]] = im_;
      } else {
        auto psi_re_red_ = psi_re + (pos_[i] << log2_pack_size);
        auto psi_im_red_ = psi_im + (pos_[i] << log2_pack_size);
        for (std::size_t j = 0; j < pack_size; ++j) {
          psi_re_red_[j] = re_[j];
          psi_im_red_[j] = im_[j];
        }
      }
    }
  }

  // Everything is OK
  return 0;
}

template <bool aligned>
int64_t apply_rr(array_type *psi_re, const array_type *U_re,
                 const uint64_t *pos, const uint64_t n_qubits,
                 const uint64_t num_threads = 0) {
  // Check that psi is not empty
  if (psi_re == nullptr) return 1;

  // Check that U is not empty
  if (U_re == nullptr) return 2;

  // Check that pos is not empty
  if (pos == nullptr) return 3;

  // Check if pointer are correctly aligned
  if constexpr (aligned)
    if (reinterpret_cast<std::size_t>(psi_re) % 32) return 4;

  // Get number of threads
  std::size_t num_threads_ = omp_get_max_threads();

  // If 'num_threads' is provided, use it
  if (num_threads > 0) num_threads_ = num_threads;

  // Otherwise, check if env variable is provided
  else if (const auto _nt = std::getenv("HYBRIDQ_ARRAY_NUM_THREADS"))
    num_threads_ = atoi(_nt);

  // Recast to the right size
  auto *psi_re_ = reinterpret_cast<pack_type *>(psi_re);

  // Split in real and imaginary parts
  static const std::size_t U_size = 1uL << n_pos;

  // Shift positions
  std::array<std::size_t, n_pos> shift_pos;
  for (std::size_t i = 0; i < n_pos; ++i)
    shift_pos[i] = pos[i] - log2_pack_size;

#pragma omp parallel for num_threads(num_threads_)
  for (std::size_t i = 0; i < (1uLL << (n_qubits - log2_pack_size - n_pos));
       ++i) {
    // Get indexes to expand
    const auto pos_ = expand(i, shift_pos);

    // Buffer real and imaginary parts from state
    std::array<pack_type, U_size> buffer_re_;
    for (std::size_t i = 0; i < U_size; ++i) {
      if constexpr (aligned)
        buffer_re_[i] = psi_re_[pos_[i]];
      else
        buffer_re_[i] = get_pack(psi_re + (pos_[i] << log2_pack_size));
    }

    // Compute matrix multiplication
    for (std::size_t i = 0; i < U_size; ++i) {
      auto re_{zero<pack_size>()};
      for (std::size_t j = 0; j < U_size; ++j) {
        const auto U_re_ = U_re[i * U_size + j];
        re_ += U_re_ * buffer_re_[j];
      }

      // Dump real and imaginary part
      if constexpr (aligned) {
        psi_re_[pos_[i]] = re_;
      } else {
        auto psi_re_red_ = psi_re + (pos_[i] << log2_pack_size);
        for (std::size_t j = 0; j < pack_size; ++j) psi_re_red_[j] = re_[j];
      }
    }
  }

  // Everything is OK
  return 0;
}

template <bool aligned>
int64_t apply_cc(array_type *psi, const array_type *U, const uint64_t *pos,
                 const uint64_t n_qubits, const uint64_t num_threads = 0) {
  // Check that psi is not empty
  if (psi == nullptr) return 1;

  // Check that U is not empty
  if (U == nullptr) return 2;

  // Check that pos is not empty
  if (pos == nullptr) return 3;

  // Check if pointer are correctly aligned
  if constexpr (aligned)
    if (reinterpret_cast<std::size_t>(psi) % 32) return 4;

  // Get number of threads
  std::size_t num_threads_ = omp_get_max_threads();

  // If 'num_threads' is provided, use it
  if (num_threads > 0) num_threads_ = num_threads;

  // Otherwise, check if env variable is provided
  else if (const auto _nt = std::getenv("HYBRIDQ_ARRAY_NUM_THREADS"))
    num_threads_ = atoi(_nt);

  // Recast to the right size
  auto *psi_ = reinterpret_cast<ctype *>(psi);

  // Split in real and imaginary parts
  static const std::size_t U_size = 1uL << n_pos;

  // Get real and imaginary parts of
  std::array<array_type, U_size * U_size> U_re, U_im;
  for (std::size_t i = 0; i < U_size * U_size; ++i) {
    U_re[i] = U[2 * i + 0];
    U_im[i] = U[2 * i + 1];
  }

  // Shift positions
  std::array<std::size_t, n_pos> shift_pos;
  for (std::size_t i = 0; i < n_pos; ++i)
    shift_pos[i] = pos[i] - log2_pack_size;

#pragma omp parallel for num_threads(num_threads_)
  for (std::size_t i = 0; i < (1uLL << (n_qubits - log2_pack_size - n_pos));
       ++i) {
    // Get indexes to expand
    const auto pos_ = expand(i, shift_pos);

    // Buffer real and imaginary parts from state
    std::array<ctype, U_size> buffer_;
    for (std::size_t i = 0; i < U_size; ++i) {
      if constexpr (aligned) {
        buffer_[i].c = shuffle_fw(psi_[pos_[i]]).c;
      } else {
        buffer_[i].c = shuffle_fw(psi + 2 * (pos_[i] << log2_pack_size)).c;
      }
    }

    // Compute matrix multiplication
    for (std::size_t i = 0; i < U_size; ++i) {
      ctype q_{zero<pack_size>(), zero<pack_size>()};
      for (std::size_t j = 0; j < U_size; ++j) {
        const auto U_re_ = U_re[i * U_size + j];
        const auto U_im_ = U_im[i * U_size + j];
        q_.pp[0] += U_re_ * buffer_[j].pp[0] - U_im_ * buffer_[j].pp[1];
        q_.pp[1] += U_re_ * buffer_[j].pp[1] + U_im_ * buffer_[j].pp[0];
      }

      // Dump real and imaginary part
      q_ = shuffle_bw(q_);
      if constexpr (aligned) {
        psi_[pos_[i]].c = q_.c;
      } else {
        auto psi_red_ = psi + 2 * (pos_[i] << log2_pack_size);
        for (std::size_t j = 0; j < 2 * pack_size; ++j) psi_red_[j] = q_.c[j];
      }
    }
  }

  // Everything is OK
  return 0;
}

template <bool aligned>
int64_t apply_cr(array_type *psi, const array_type *U_re, const uint64_t *pos,
                 const uint64_t n_qubits, const uint64_t num_threads = 0) {
  // Check that psi is not empty
  if (psi == nullptr) return 1;

  // Check that U is not empty
  if (U_re == nullptr) return 2;

  // Check that pos is not empty
  if (pos == nullptr) return 3;

  // Check if pointer are correctly aligned
  if constexpr (aligned)
    if (reinterpret_cast<std::size_t>(psi) % 32) return 4;

  // Get number of threads
  std::size_t num_threads_ = omp_get_max_threads();

  // If 'num_threads' is provided, use it
  if (num_threads > 0) num_threads_ = num_threads;

  // Otherwise, check if env variable is provided
  else if (const auto _nt = std::getenv("HYBRIDQ_ARRAY_NUM_THREADS"))
    num_threads_ = atoi(_nt);

  // Recast to the right size
  auto *psi_ = reinterpret_cast<ctype *>(psi);

  // Split in real and imaginary parts
  static const std::size_t U_size = 1uL << n_pos;

  // Shift positions
  std::array<std::size_t, n_pos> shift_pos;
  for (std::size_t i = 0; i < n_pos; ++i)
    shift_pos[i] = pos[i] - log2_pack_size;

#pragma omp parallel for num_threads(num_threads_)
  for (std::size_t i = 0; i < (1uLL << (n_qubits - log2_pack_size - n_pos));
       ++i) {
    // Get indexes to expand
    const auto pos_ = expand(i, shift_pos);

    // Buffer real and imaginary parts from state
    std::array<ctype, U_size> buffer_;
    for (std::size_t i = 0; i < U_size; ++i) {
      if constexpr (aligned) {
        buffer_[i].c = shuffle_fw(psi_[pos_[i]]).c;
      } else {
        buffer_[i].c = shuffle_fw(psi + 2 * (pos_[i] << log2_pack_size)).c;
      }
    }

    // Compute matrix multiplication
    for (std::size_t i = 0; i < U_size; ++i) {
      ctype q_{zero<pack_size>(), zero<pack_size>()};
      for (std::size_t j = 0; j < U_size; ++j) {
        const auto U_re_ = U_re[i * U_size + j];
        q_.pp[0] += U_re_ * buffer_[j].pp[0];
        q_.pp[1] += U_re_ * buffer_[j].pp[1];
      }

      // Dump real and imaginary part
      q_ = shuffle_bw(q_);
      if constexpr (aligned) {
        psi_[pos_[i]].c = q_.c;
      } else {
        auto psi_red_ = psi + 2 * (pos_[i] << log2_pack_size);
        for (std::size_t j = 0; j < 2 * pack_size; ++j) psi_red_[j] = q_.c[j];
      }
    }
  }

  // Everything is OK
  return 0;
}

extern "C" {

int64_t apply_qc(array_type *psi_re, array_type *psi_im, const array_type *U,
                 const uint64_t *pos, const uint64_t n_qubits,
                 const uint64_t num_threads = 0) {
  // Check if arrays are aligned
  const bool aligned = not(reinterpret_cast<std::size_t>(psi_re) % 32 or
                           reinterpret_cast<std::size_t>(psi_im) % 32);

  // Call the right function
  switch (aligned) {
    case true:
      return apply_qc<true>(psi_re, psi_im, U, pos, n_qubits, num_threads);
    case false:
      return apply_qc<false>(psi_re, psi_im, U, pos, n_qubits, num_threads);
  }
}

int64_t apply_qr(array_type *psi_re, array_type *psi_im, const array_type *U_re,
                 const uint64_t *pos, const uint64_t n_qubits,
                 const uint64_t num_threads = 0) {
  // Check if arrays are aligned
  const bool aligned = not(reinterpret_cast<std::size_t>(psi_re) % 32 or
                           reinterpret_cast<std::size_t>(psi_im) % 32);

  // Call the right function
  switch (aligned) {
    case true:
      return apply_qr<true>(psi_re, psi_im, U_re, pos, n_qubits, num_threads);
    case false:
      return apply_qr<false>(psi_re, psi_im, U_re, pos, n_qubits, num_threads);
  }
}

int64_t apply_rr(array_type *psi_re, const array_type *U_re,
                 const uint64_t *pos, const uint64_t n_qubits,
                 const uint64_t num_threads = 0) {
  // Check if arrays are aligned
  const bool aligned = not(reinterpret_cast<std::size_t>(psi_re) % 32);

  // Call the right function
  switch (aligned) {
    case true:
      return apply_rr<true>(psi_re, U_re, pos, n_qubits, num_threads);
    case false:
      return apply_rr<false>(psi_re, U_re, pos, n_qubits, num_threads);
  }
}

int64_t apply_cc(array_type *psi, const array_type *U, const uint64_t *pos,
                 const uint64_t n_qubits, const uint64_t num_threads = 0) {
  // Check if array is aligned
  const bool aligned = not(reinterpret_cast<std::size_t>(psi) % 32);

  // Call the right function
  switch (aligned) {
    case true:
      return apply_cc<true>(psi, U, pos, n_qubits, num_threads);
    case false:
      return apply_cc<false>(psi, U, pos, n_qubits, num_threads);
  }
}

int64_t apply_cr(array_type *psi, const array_type *U_re, const uint64_t *pos,
                 const uint64_t n_qubits, const uint64_t num_threads = 0) {
  // Check if array is aligned
  const bool aligned = not(reinterpret_cast<std::size_t>(psi) % 32);

  // Call the right function
  switch (aligned) {
    case true:
      return apply_cr<true>(psi, U_re, pos, n_qubits, num_threads);
    case false:
      return apply_cr<false>(psi, U_re, pos, n_qubits, num_threads);
  }
}
}

}  // namespace hybridq
