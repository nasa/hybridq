/*
Author: Salvatore Mandra (salvatore.mandra@nasa.gov)

Copyright © 2021, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The HybridQ: A Hybrid Simulator for Quantum Circuits platform is licensed under
the Apache License, Version 2.0 (the "License"); you may not use this file
except in compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
*/

#pragma once

#include <complex>
#include <cstring>
#include <list>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define ENABLE_IF(...) std::enable_if_t<__VA_ARGS__, bool> = true

namespace hybridq_clifford::archive {

template <typename Test, template <typename...> typename Ref>
struct is_specialization : std::false_type {};

template <template <typename...> typename Ref, typename... T>
struct is_specialization<Ref<T...>, Ref> : std::true_type {};

template <typename Test, template <typename...> typename Ref>
static constexpr auto is_specialization_v = is_specialization<Test, Ref>::value;

template <typename Test>
static constexpr auto is_complex_v =
    is_specialization<Test, std::complex>::value;

template <typename Test>
static constexpr auto is_vector_v = is_specialization<Test, std::vector>::value;

template <typename Test>
static constexpr auto is_list_v = is_specialization<Test, std::list>::value;

template <typename Test>
static constexpr auto is_map_v =
    is_specialization<Test, std::map>::value ||
    is_specialization<Test, std::unordered_map>::value;

template <typename Test>
static constexpr auto is_set_v =
    is_specialization<Test, std::set>::value ||
    is_specialization<Test, std::unordered_set>::value;

/* PRE-DECLARATION */

template <typename T>
std::string dump(const T &);

template <typename T>
auto load(const char *buffer);

/* FUNDAMENTAL */

template <typename T, ENABLE_IF(std::is_trivially_copyable_v<T>)>
auto dump_(const T &a) {
  std::string out_(sizeof(T), '\00');
  std::memcpy(out_.data(), &a, sizeof(T));
  return out_;
}

template <typename T, ENABLE_IF(std::is_trivially_copyable_v<T>)>
auto load_(const char *buffer) {
  T out_;
  std::memcpy(&out_, buffer, sizeof(T));
  return std::pair{buffer + sizeof(T), out_};
}

/* STRINGS */

auto dump_(const std::string &a) { return dump<std::size_t>(std::size(a)) + a; }

template <typename T, ENABLE_IF(std::is_same_v<T, std::string>)>
auto load_(const char *buffer) {
  const auto [buffer_, size_] = load<std::size_t>(buffer);
  std::string out_(size_, '\00');
  std::memcpy(out_.data(), buffer_, size_);
  return std::pair{buffer_ + size_, out_};
}

/* VECTOR OF TRIVIALLY COPYABLE */

template <typename... T, typename V = typename std::vector<T...>::value_type,
          ENABLE_IF(std::is_trivially_copyable_v<V>)>
auto dump_(const std::vector<T...> &a) {
  std::string out_(sizeof(V) * std::size(a), '\00');
  std::memcpy(out_.data(), a.data(), sizeof(V) * std::size(a));
  return dump<std::size_t>(std::size(a)) + out_;
}

template <typename Vector, ENABLE_IF(is_vector_v<Vector>),
          typename V = typename Vector::value_type,
          ENABLE_IF(std::is_trivially_copyable_v<V>)>
auto load_(const char *buffer) {
  const auto [buffer_, size_] = load<std::size_t>(buffer);
  Vector out_(size_);
  std::memcpy(out_.data(), buffer_, sizeof(V) * size_);
  return std::pair{buffer_ + sizeof(V) * size_, out_};
}

/* ARBITRARY VECTOR/LIST/SET */

template <typename Vector, ENABLE_IF(is_set_v<Vector> || is_list_v<Vector> ||
                                     is_vector_v<Vector>)>
auto dump_(const Vector &a) {
  auto out_ = dump<std::size_t>(std::size(a));
  for (const auto &x_ : a) out_ += dump(x_);
  return out_;
}

template <typename Vector, ENABLE_IF(is_list_v<Vector> || is_vector_v<Vector>),
          typename V = typename Vector::value_type,
          ENABLE_IF(is_list_v<Vector> ||
                    (!std::is_trivially_copyable_v<V> && !is_complex_v<V>))>
auto load_(const char *buffer) {
  auto [buffer_, size_] = load<std::size_t>(buffer);
  Vector out_;
  for (std::size_t i_ = 0; i_ < size_; ++i_) {
    auto [new_head_, el_] = load<V>(buffer_);
    buffer_ = new_head_;
    out_.push_back(std::move(el_));
  }
  return std::pair{buffer_, out_};
}

template <typename Set, ENABLE_IF(is_set_v<Set>),
          typename V = typename Set::value_type>
auto load_(const char *buffer) {
  auto [buffer_, size_] = load<std::size_t>(buffer);
  Set out_;
  for (std::size_t i_ = 0; i_ < size_; ++i_) {
    auto [new_head_, el_] = load<V>(buffer_);
    buffer_ = new_head_;
    out_.insert(std::move(el_));
  }
  return std::pair{buffer_, out_};
}

/* ARBITRARY MAP/UNORDERED_MAP */

template <typename Map, ENABLE_IF(is_map_v<Map>)>
auto dump_(const Map &a) {
  auto out_ = dump<std::size_t>(std::size(a));
  for (const auto &[k_, v_] : a) out_ += dump(k_) + dump(v_);
  return out_;
}

template <typename Map, ENABLE_IF(is_map_v<Map>),
          typename K = typename Map::key_type,
          typename V = typename Map::mapped_type>
auto load_(const char *buffer) {
  auto [buffer_, size_] = load<std::size_t>(buffer);
  Map map_;
  for (std::size_t i_ = 0; i_ < size_; ++i_) {
    auto [h1_, k_] = load<K>(buffer_);
    auto [h2_, v_] = load<V>(h1_);
    buffer_ = h2_;
    map_.insert({std::move(k_), std::move(v_)});
  }
  return std::pair{buffer_, map_};
}

/* API */

template <typename T>
struct Dump {
  auto operator()(const T &t) const { return dump_(t); }
};

template <typename T>
struct Load {
  auto operator()(const char *buffer) const { return load_<T>(buffer); }
};

template <typename T>
std::string dump(const T &t) {
  return Dump<T>()(t);
}

template <typename T>
auto load(const char *buffer) {
  return Load<T>()(buffer);
}

#undef ENABLE_IF

}  // namespace hybridq_clifford::archive
