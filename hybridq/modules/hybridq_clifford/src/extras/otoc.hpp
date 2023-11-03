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

#include <pybind11/pybind11.h>

#include "../defs.hpp"

namespace hybridq_clifford_otoc {

namespace py = pybind11;
namespace hqc = hybridq_clifford;

struct Info : hqc::Info {
  std::size_t cleaning_time_ms{std::numeric_limits<std::size_t>::max()};
  hqc::FVector1D log10_norm_atols;
  hqc::FVector1D otoc_values;
  hqc::FVector1D otoc_values_no_int;
  hqc::FVector1D otoc_norms0;
  hqc::FVector1D otoc_norms1;
  hqc::IVector1D otoc_n_completed_branches;
  hqc::IVector1D otoc_n_explored_branches;

  auto dict() const {
    auto out_ = static_cast<const hqc::Info *>(this)->dict();
    if (std::size(log10_norm_atols))
      out_["log10_norm_atols"] = log10_norm_atols;
    if (std::size(otoc_values)) out_["otoc_values"] = otoc_values;
    if (std::size(otoc_values_no_int))
      out_["otoc_values_no_int"] = otoc_values_no_int;
    if (std::size(otoc_norms0)) out_["otoc_norms0"] = otoc_norms0;
    if (std::size(otoc_norms1)) out_["otoc_norms1"] = otoc_norms1;
    if (std::size(otoc_n_completed_branches))
      out_["otoc_n_completed_branches"] = otoc_n_completed_branches;
    if (std::size(otoc_n_explored_branches))
      out_["otoc_n_explored_branches"] = otoc_n_explored_branches;
    return out_;
  }
};

// Set specific types
using info_type = Info;
using state_type = hqc::State<hqc::vector_type>;
using phase_type = hqc::float_type;
using branch_type = hqc::Branch<state_type, phase_type, hqc::index_type>;
using list_branches_type = hqc::list_type<branch_type>;
using phases_type = hqc::FVector2D;
using positions_type = hqc::IVector2D;
using qubits_type = hqc::IVector1D;
//
using complex_type = std::complex<hqc::float_type>;
using completed_branches_type =
    hqc::map_type<state_type, std::array<complex_type, 2>>;

template <typename Branches>
auto UpdateBranches_(
    Branches &&branches, const hqc::vector_type<phases_type> &phases,
    const hqc::vector_type<positions_type> &positions,
    const hqc::vector_type<qubits_type> &qubits, std::size_t target_position,
    const hqc::IVector1D &initial_state, const hqc::float_type atol,
    const hqc::float_type norm_atol, const hqc::float_type merge_atol,
    const std::size_t n_threads, const std::size_t log2_n_buckets,
    bool expand_branches_only, int &stop_,
    hqc::vector_type<info_type> &infos_) {
  // Set delta_norm_atol
  static constexpr float log10_dnorm_atol = 0.1;

  // Initialize branches
  list_branches_type branches_(std::begin(branches), std::end(branches));

  // Get bucket corresponding to a given normalization phase
  const auto get_na_bucket_ = [norm_atol](auto &&x) {
    return -std::floor(std::log10(std::max(norm_atol, std::abs(x))) /
                       log10_dnorm_atol);
  };

  // Get total number of na buckets
  const std::size_t n_na_buckets = get_na_bucket_(norm_atol) + 1;

  // Initialize number of buckets
  const std::size_t n_buckets = 1uL << log2_n_buckets;

  // Inizialize global info
  info_type ginfo_;

  // Initialize infos
  infos_.clear();
  infos_.resize(n_threads);

  // Initialize completed branches
  hqc::vector_type<completed_branches_type> completed_branches_(n_buckets);

  // Initialize otoc values and norms
  hqc::FVector2D v_otoc_value_(n_buckets, hqc::FVector1D(n_na_buckets, 0));
  hqc::FVector2D v_otoc_value_no_int_(n_buckets,
                                      hqc::FVector1D(n_na_buckets, 0));
  hqc::FVector2D v_norm0_(n_buckets, hqc::FVector1D(n_na_buckets, 0));
  hqc::FVector2D v_norm1_(n_buckets, hqc::FVector1D(n_na_buckets, 0));
  hqc::IVector2D v_n_completed_branches_(n_threads,
                                         hqc::IVector1D(n_na_buckets, 0));
  hqc::IVector2D v_n_explored_branches_(n_threads,
                                        hqc::IVector1D(n_na_buckets, 0));

  // Initialize mutex
  std::vector<std::mutex> g_i_mutex_(n_buckets);

  // Get hash of a given state
  auto hash_ = [mask =
                    log2_n_buckets == state_type::block_size
                        ? ~state_type::base_type{0}
                        : ((state_type::base_type{1} << log2_n_buckets) - 1)](
                   auto &&state) { return state.data()[0] & mask; };

  // Update dataset of completed branches using explored branches
  const auto update_ = [&hash_, &completed_branches_, merge_atol,
                        target_position, &v_otoc_value_, &v_otoc_value_no_int_,
                        &v_norm0_, &v_norm1_, &v_n_completed_branches_,
                        &g_i_mutex_, &initial_state,
                        &get_na_bucket_](std::size_t idx, auto &&new_branch) {
    // Transformation for initial position
    static constexpr std::size_t v_phase_[][4] = {
        {0, 0, 1, 0},  // |0>
        {0, 0, 3, 2},  // |1>
        {0, 0, 3, 0},  // |+>
        {0, 2, 1, 0},  // |->
    };
    static constexpr std::size_t v_pauli_[][4] = {
        {0, 1, 1, 0},  // |0>
        {1, 0, 0, 1},  // |1>
        {0, 0, 1, 1},  // |+>
        {1, 1, 0, 0},  // |->
    };

    // Initialize phases for a given number of Y's
    static constexpr complex_type ph_[] = {
        complex_type{1, 0},
        complex_type{0, 1},
        complex_type{-1, 0},
        complex_type{0, -1},
    };

    // Get hash of the new state
    const auto n_ = hash_(new_branch.state);

    // Get target Pauli
    const auto target_pauli_ = hqc::GetPauli(new_branch.state, target_position);

    // Initialize projection and phase
    std::size_t proj_ph_ = 0;
    state_type proj_(std::size(new_branch.state) / 2);

    for (std::size_t i_ = 0, end_ = std::size(proj_); i_ < end_; ++i_) {
      const auto p_ = hqc::GetPauli(new_branch.state, i_);
      proj_.set(i_, v_pauli_[initial_state[i_]][p_]);
      proj_ph_ += v_phase_[initial_state[i_]][p_];
    }

    // Get phases
    const auto phase0_ = new_branch.phase * ph_[proj_ph_ % 4];
    const auto phase1_ =
        target_pauli_ == 1 || target_pauli_ == 2 ? -phase0_ : phase0_;

    {
      // Get norm_atol bucket
      const std::size_t n_na_ = get_na_bucket_(new_branch.norm_phase);

      // Lock database and update
      const std::lock_guard<std::mutex> lock(g_i_mutex_[n_]);

      // Update number of completed branches
      v_n_completed_branches_[idx][n_na_] += 1;

      // Get old phases
      auto &phs_ = completed_branches_[n_][proj_];

      // Update otoc value and norm
      v_otoc_value_[n_][n_na_] -= std::real(std::conj(phs_[1]) * phs_[0]);
      v_otoc_value_no_int_[n_][n_na_] +=
          std::real(std::conj(phs_[1]) * phs_[0]);
      v_norm0_[n_][n_na_] -= std::norm(phs_[0]);
      v_norm1_[n_][n_na_] -= std::norm(phs_[1]);

      if (const auto x_ = (phs_[0] += phase0_), y_ = (phs_[1] += phase1_);
          std::abs(x_) < merge_atol && std::abs(y_) < merge_atol)
        completed_branches_[n_].erase(proj_);
      else {
        v_otoc_value_[n_][n_na_] += std::real(std::conj(phs_[1]) * phs_[0]);
        v_norm0_[n_][n_na_] += std::norm(phs_[0]);
        v_norm1_[n_][n_na_] += std::norm(phs_[1]);
      }
    }
  };

  // Update dataset of completed branches using explored branches
  const auto update_completed_branches_ =
      [&hash_, &update_, n_gates = std::size(phases), &get_na_bucket_,
       &v_n_explored_branches_,
       &infos_](std::size_t idx, auto &&branch, auto &&branches_stack,
                auto &&new_branches, auto &&mutex) {
        // Update explored branches
        infos_[idx].n_explored_branches += 1;
        v_n_explored_branches_[idx][get_na_bucket_(branch.norm_phase)] += 1;

        // If not completed branches, return
        if (new_branches.back().gate_idx != n_gates) {
          // lock database
          const std::lock_guard<std::mutex> lock_(mutex);

          // Update
          branches_stack.splice(std::end(branches_stack), new_branches);

        } else {
          // Update completed branches
          infos_[idx].n_completed_branches += std::size(new_branches);

          // For each of the new branches ...
          for (const auto &new_branch_ : new_branches)
            update_(idx, new_branch_);
        }

        // Update number of remaining branches
        infos_[idx].n_remaining_branches = std::size(branches_stack);
      };

  // Run the actual simulation
  hqc::UpdateAllBranches<hqc::vector_type>(
      branches_, phases, positions, qubits, atol, norm_atol, n_threads,
      expand_branches_only, stop_, ginfo_, update_completed_branches_);

  // Update last info
  ginfo_.n_threads = n_threads;
  ginfo_.n_explored_branches = 0;
  ginfo_.n_completed_branches = 0;
  ginfo_.n_remaining_branches = 0;
  ginfo_.n_total_branches = 0;
  for (auto &i_ : infos_) {
    ginfo_.n_explored_branches += i_.n_explored_branches;
    ginfo_.n_completed_branches += i_.n_completed_branches;
    ginfo_.n_remaining_branches += i_.n_remaining_branches;
  }
  for (const auto &br_ : completed_branches_)
    ginfo_.n_total_branches += std::size(br_);

  // Accumulate otoc
  {
    auto &log10_norm_atols_ = infos_[0].log10_norm_atols;
    auto &otoc_values_ = infos_[0].otoc_values;
    auto &otoc_values_no_int_ = infos_[0].otoc_values_no_int;
    auto &otoc_norms0_ = infos_[0].otoc_norms0;
    auto &otoc_norms1_ = infos_[0].otoc_norms1;
    auto &otoc_n_completed_branches_ = infos_[0].otoc_n_completed_branches;
    auto &otoc_n_explored_branches_ = infos_[0].otoc_n_explored_branches;

    // Resize
    log10_norm_atols_.resize(n_na_buckets);
    otoc_values_.resize(n_na_buckets);
    otoc_values_no_int_.resize(n_na_buckets);
    otoc_norms0_.resize(n_na_buckets);
    otoc_norms1_.resize(n_na_buckets);
    otoc_n_completed_branches_.resize(n_na_buckets);
    otoc_n_explored_branches_.resize(n_na_buckets);

    // Initialize
    for (std::size_t j_ = 0; j_ < n_na_buckets; ++j_)
      log10_norm_atols_[j_] = -log10_dnorm_atol * j_;
    std::fill(std::begin(otoc_values_), std::end(otoc_values_), 0);
    std::fill(std::begin(otoc_values_no_int_), std::end(otoc_values_no_int_),
              0);
    std::fill(std::begin(otoc_norms0_), std::end(otoc_norms0_), 0);
    std::fill(std::begin(otoc_norms1_), std::end(otoc_norms1_), 0);
    std::fill(std::begin(otoc_n_completed_branches_),
              std::end(otoc_n_completed_branches_), 0);
    std::fill(std::begin(otoc_n_explored_branches_),
              std::end(otoc_n_explored_branches_), 0);

    // Merge otocs
    for (std::size_t j_ = 0; j_ < n_na_buckets; ++j_) {
      for (std::size_t i_ = 0; i_ < n_buckets; ++i_) {
        otoc_values_[j_] += v_otoc_value_[i_][j_];
        otoc_values_no_int_[j_] += v_otoc_value_no_int_[i_][j_];
        otoc_norms0_[j_] += v_norm0_[i_][j_];
        otoc_norms1_[j_] += v_norm1_[i_][j_];
      }
      for (std::size_t i_ = 0; i_ < n_threads; ++i_) {
        otoc_n_completed_branches_[j_] += v_n_completed_branches_[i_][j_];
        otoc_n_explored_branches_[j_] += v_n_explored_branches_[i_][j_];
      }
    }

    // Accumulate
    for (std::size_t j_ = 1; j_ < n_na_buckets; ++j_) {
      otoc_values_[j_] += otoc_values_[j_ - 1];
      otoc_values_no_int_[j_] += otoc_values_no_int_[j_ - 1];
      otoc_norms0_[j_] += otoc_norms0_[j_ - 1];
      otoc_norms1_[j_] += otoc_norms1_[j_ - 1];
      otoc_n_completed_branches_[j_] += otoc_n_completed_branches_[j_ - 1];
      otoc_n_explored_branches_[j_] += otoc_n_explored_branches_[j_ - 1];
    }

    // Get sqrt of norms
    std::transform(std::begin(otoc_norms0_), std::end(otoc_norms0_),
                   std::begin(otoc_norms0_),
                   [](auto &&x) { return std::sqrt(x); });
    std::transform(std::begin(otoc_norms1_), std::end(otoc_norms1_),
                   std::begin(otoc_norms1_),
                   [](auto &&x) { return std::sqrt(x); });
  }

  // Clean-up
  {
    // Start time
    auto tic_ = std::chrono::system_clock::now();

    // Initialize cleaning routine
    std::atomic_uint32_t index_{0};
    auto clean_ = [&index_, &completed_branches_]() {
      std::size_t i_;
      while ((i_ = index_++) < std::size(completed_branches_))
        completed_branches_type temp_ = std::move(completed_branches_[i_]);
    };

    // Initialize threads
    std::vector<decltype(std::async(clean_))> threads_;
    for (std::size_t i_ = 0; i_ < n_threads; ++i_)
      threads_.push_back(std::async(clean_));

    // Release
    for (auto &t_ : threads_) t_.get();

    // Stop time
    auto toc_ = std::chrono::system_clock::now();

    // Get delta time
    const auto dt_ =
        std::chrono::duration_cast<std::chrono::milliseconds>(toc_ - tic_)
            .count();

    // Update info
    ginfo_.cleaning_time_ms = dt_;
  }

  // Return global info
  return std::tuple{ginfo_, std::move(branches_)};
}

struct Simulator {
  using core_output = std::tuple<info_type, list_branches_type>;
  const hqc::vector_type<phases_type> phases;
  const hqc::vector_type<positions_type> positions;
  const hqc::vector_type<qubits_type> qubits;
  const std::size_t target_position;
  const hqc::IVector1D initial_state;
  const hqc::float_type atol{1e-8};
  const hqc::float_type norm_atol{1e-8};
  const hqc::float_type merge_atol{1e-8};
  const std::size_t n_threads{0};
  const std::size_t log2_n_buckets{12};

  Simulator(const hqc::vector_type<phases_type> &phases,
            const hqc::vector_type<positions_type> &positions,
            const hqc::vector_type<qubits_type> &qubits,
            std::size_t target_position, const hqc::IVector1D &initial_state,
            hqc::float_type atol = 1e-8, hqc::float_type norm_atol = 1e-8,
            hqc::float_type merge_atol = 1e-8, std::size_t n_threads = 0,
            std::size_t log2_n_buckets = 12)
      : phases{phases},
        positions{positions},
        qubits{qubits},
        target_position{target_position},
        initial_state{initial_state},
        atol{atol},
        norm_atol{norm_atol},
        merge_atol{merge_atol},
        n_threads{n_threads > 0 ? n_threads
                                : std::thread::hardware_concurrency()},
        log2_n_buckets{log2_n_buckets} {
    // Check norm atol
    if (norm_atol > 1 || norm_atol < 0)
      throw std::logic_error("'norm_atol' must be within 0 and 1 (included).");
  }

  template <typename Branches>
  auto start(Branches &&branches, bool expand_branches_only = false) {
    // Initialize stop signal
    _stop = false;

    // Initialize core
    _core = [branches, expand_branches_only, this]() -> core_output {
      return UpdateBranches_(branches, phases, positions, qubits,
                             target_position, initial_state, atol, norm_atol,
                             merge_atol, n_threads, log2_n_buckets,
                             expand_branches_only, _stop, _infos);
    };

    // Run simulation (in background)
    _thread = std::async(_core);
  }

  // Join simulation
  auto join() {
    // Wait simulation to finish and get global info
    auto &&[ginfo_, branches_] = _thread.get();

    // Convert list to vector
    std::vector<branch_type> v_branches_;
    std::move(std::begin(branches_), std::end(branches_),
              std::back_inserter(v_branches_));

    // Return results
    return std::tuple{std::move(ginfo_), std::move(v_branches_),
                      std::move(_infos[0])};
  }

  // Stop simulation
  auto stop() {
    // Set stop signal to stop
    _stop = true;

    // Join
    return join();
  }

  // Return true if simulation is ready
  auto ready(std::size_t ms = 0) {
    return _thread.wait_for(std::chrono::milliseconds(ms)) ==
           std::future_status::ready;
  }

  // Return constant reference to infos
  const auto &infos() const { return _infos; }

 private:
  int _stop;
  hqc::vector_type<info_type> _infos;
  std::function<core_output()> _core;
  std::future<core_output> _thread;
};

}  // namespace hybridq_clifford_otoc

#if 0
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <src/branch.hpp>
#include <src/defs.hpp>
#include <src/info.hpp>
#include <src/simulation.hpp>
#include <src/state.hpp>
#include <src/utils.hpp>

#define ADD_OPAQUE_VECTOR(MOD, TYPE, NAME)       \
  py::bind_vector<TYPE>(m, NAME, NAME);          \
  py::implicitly_convertible<py::list, TYPE>();  \
  py::implicitly_convertible<py::tuple, TYPE>(); \
  py::implicitly_convertible<py::array, TYPE>();
#define ADD_OPAQUE_MAP(MOD, TYPE, NAME) \
  py::bind_map<TYPE>(m, NAME, NAME);    \
  py::implicitly_convertible<py::dict, TYPE>();

namespace py = pybind11;
namespace hqc = hybridq_clifford;

struct Info : hqc::Info {
  std::size_t cleaning_time_ms{std::numeric_limits<std::size_t>::max()};
  hqc::FVector1D log10_norm_atols;
  hqc::FVector1D otoc_values;
  hqc::FVector1D otoc_values_no_int;
  hqc::FVector1D otoc_norms0;
  hqc::FVector1D otoc_norms1;
  hqc::IVector1D otoc_n_completed_branches;
  hqc::IVector1D otoc_n_explored_branches;

  auto dict() const {
    auto out_ = static_cast<const hqc::Info *>(this)->dict();
    if (std::size(log10_norm_atols))
      out_["log10_norm_atols"] = log10_norm_atols;
    if (std::size(otoc_values)) out_["otoc_values"] = otoc_values;
    if (std::size(otoc_values_no_int))
      out_["otoc_values_no_int"] = otoc_values_no_int;
    if (std::size(otoc_norms0)) out_["otoc_norms0"] = otoc_norms0;
    if (std::size(otoc_norms1)) out_["otoc_norms1"] = otoc_norms1;
    if (std::size(otoc_n_completed_branches))
      out_["otoc_n_completed_branches"] = otoc_n_completed_branches;
    if (std::size(otoc_n_explored_branches))
      out_["otoc_n_explored_branches"] = otoc_n_explored_branches;
    return out_;
  }
};

// Set specific types
using info_type = Info;
using state_type = hqc::State<hqc::vector_type>;
using phase_type = hqc::float_type;
using branch_type = hqc::Branch<state_type, phase_type, hqc::index_type>;
using list_branches_type = hqc::list_type<branch_type>;
using phases_type = hqc::FVector2D;
using positions_type = hqc::IVector2D;
using qubits_type = hqc::IVector1D;

using complex_type = std::complex<hqc::float_type>;
using completed_branches_type =
    hqc::map_type<state_type, std::array<complex_type, 2>>;

template <typename Branches>
auto UpdateBranches_(
    Branches &&branches, const hqc::vector_type<phases_type> &phases,
    const hqc::vector_type<positions_type> &positions,
    const hqc::vector_type<qubits_type> &qubits, std::size_t target_position,
    const hqc::IVector1D &initial_state, const hqc::float_type atol,
    const hqc::float_type norm_atol, const hqc::float_type merge_atol,
    const std::size_t n_threads, const std::size_t log2_n_buckets,
    bool expand_branches_only, int &stop_,
    hqc::vector_type<info_type> &infos_) {
  // Set delta_norm_atol
  static constexpr float log10_dnorm_atol = 0.1;

  // Initialize branches
  list_branches_type branches_(std::begin(branches), std::end(branches));

  // Get bucket corresponding to a given normalization phase
  const auto get_na_bucket_ = [norm_atol](auto &&x) {
    return -std::floor(std::log10(std::max(norm_atol, std::abs(x))) /
                       log10_dnorm_atol);
  };

  // Get total number of na buckets
  const std::size_t n_na_buckets = get_na_bucket_(norm_atol) + 1;

  // Initialize number of buckets
  const std::size_t n_buckets = 1uL << log2_n_buckets;

  // Inizialize global info
  info_type ginfo_;

  // Initialize infos
  infos_.clear();
  infos_.resize(n_threads);

  // Initialize completed branches
  hqc::vector_type<completed_branches_type> completed_branches_(n_buckets);

  // Initialize otoc values and norms
  hqc::FVector2D v_otoc_value_(n_buckets, hqc::FVector1D(n_na_buckets, 0));
  hqc::FVector2D v_otoc_value_no_int_(n_buckets,
                                      hqc::FVector1D(n_na_buckets, 0));
  hqc::FVector2D v_norm0_(n_buckets, hqc::FVector1D(n_na_buckets, 0));
  hqc::FVector2D v_norm1_(n_buckets, hqc::FVector1D(n_na_buckets, 0));
  hqc::IVector2D v_n_completed_branches_(n_threads,
                                         hqc::IVector1D(n_na_buckets, 0));
  hqc::IVector2D v_n_explored_branches_(n_threads,
                                        hqc::IVector1D(n_na_buckets, 0));

  // Initialize mutex
  std::vector<std::mutex> g_i_mutex_(n_buckets);

  // Get hash of a given state
  auto hash_ = [mask =
                    log2_n_buckets == state_type::block_size
                        ? ~state_type::base_type{0}
                        : ((state_type::base_type{1} << log2_n_buckets) - 1)](
                   auto &&state) { return state.data()[0] & mask; };

  // Update dataset of completed branches using explored branches
  const auto update_ = [&hash_, &completed_branches_, merge_atol,
                        target_position, &v_otoc_value_, &v_otoc_value_no_int_,
                        &v_norm0_, &v_norm1_, &v_n_completed_branches_,
                        &g_i_mutex_, &initial_state,
                        &get_na_bucket_](std::size_t idx, auto &&new_branch) {
    // Transformation for initial position
    static constexpr std::size_t v_phase_[][4] = {
        {0, 0, 1, 0},  // |0>
        {0, 0, 3, 2},  // |1>
        {0, 0, 3, 0},  // |+>
        {0, 2, 1, 0},  // |->
    };
    static constexpr std::size_t v_pauli_[][4] = {
        {0, 1, 1, 0},  // |0>
        {1, 0, 0, 1},  // |1>
        {0, 0, 1, 1},  // |+>
        {1, 1, 0, 0},  // |->
    };

    // Initialize phases for a given number of Y's
    static constexpr complex_type ph_[] = {
        complex_type{1, 0},
        complex_type{0, 1},
        complex_type{-1, 0},
        complex_type{0, -1},
    };

    // Get hash of the new state
    const auto n_ = hash_(new_branch.state);

    // Get target Pauli
    const auto target_pauli_ = hqc::GetPauli(new_branch.state, target_position);

    // Initialize projection and phase
    std::size_t proj_ph_ = 0;
    state_type proj_(std::size(new_branch.state) / 2);

    for (std::size_t i_ = 0, end_ = std::size(proj_); i_ < end_; ++i_) {
      const auto p_ = hqc::GetPauli(new_branch.state, i_);
      proj_.set(i_, v_pauli_[initial_state[i_]][p_]);
      proj_ph_ += v_phase_[initial_state[i_]][p_];
    }

    // Get phases
    const auto phase0_ = new_branch.phase * ph_[proj_ph_ % 4];
    const auto phase1_ =
        target_pauli_ == 1 || target_pauli_ == 2 ? -phase0_ : phase0_;

    {
      // Get norm_atol bucket
      const std::size_t n_na_ = get_na_bucket_(new_branch.norm_phase);

      // Lock database and update
      const std::lock_guard<std::mutex> lock(g_i_mutex_[n_]);

      // Update number of completed branches
      v_n_completed_branches_[idx][n_na_] += 1;

      // Get old phases
      auto &phs_ = completed_branches_[n_][proj_];

      // Update otoc value and norm
      v_otoc_value_[n_][n_na_] -= std::real(std::conj(phs_[1]) * phs_[0]);
      v_otoc_value_no_int_[n_][n_na_] +=
          std::real(std::conj(phs_[1]) * phs_[0]);
      v_norm0_[n_][n_na_] -= std::norm(phs_[0]);
      v_norm1_[n_][n_na_] -= std::norm(phs_[1]);

      if (const auto x_ = (phs_[0] += phase0_), y_ = (phs_[1] += phase1_);
          std::abs(x_) < merge_atol && std::abs(y_) < merge_atol)
        completed_branches_[n_].erase(proj_);
      else {
        v_otoc_value_[n_][n_na_] += std::real(std::conj(phs_[1]) * phs_[0]);
        v_norm0_[n_][n_na_] += std::norm(phs_[0]);
        v_norm1_[n_][n_na_] += std::norm(phs_[1]);
      }
    }
  };

  // Update dataset of completed branches using explored branches
  const auto update_completed_branches_ =
      [&hash_, &update_, n_gates = std::size(phases), &get_na_bucket_,
       &v_n_explored_branches_,
       &infos_](std::size_t idx, auto &&branch, auto &&branches_stack,
                auto &&new_branches, auto &&mutex) {
        // Update explored branches
        infos_[idx].n_explored_branches += 1;
        v_n_explored_branches_[idx][get_na_bucket_(branch.norm_phase)] += 1;

        // If not completed branches, return
        if (new_branches.back().gate_idx != n_gates) {
          // lock database
          const std::lock_guard<std::mutex> lock_(mutex);

          // Update
          branches_stack.splice(std::end(branches_stack), new_branches);

        } else {
          // Update completed branches
          infos_[idx].n_completed_branches += std::size(new_branches);

          // For each of the new branches ...
          for (const auto &new_branch_ : new_branches)
            update_(idx, new_branch_);
        }

        // Update number of remaining branches
        infos_[idx].n_remaining_branches = std::size(branches_stack);
      };

  // Run the actual simulation
  hqc::UpdateAllBranches<hqc::vector_type>(
      branches_, phases, positions, qubits, atol, norm_atol, n_threads,
      expand_branches_only, stop_, ginfo_, update_completed_branches_);

  // Update last info
  ginfo_.n_threads = n_threads;
  ginfo_.n_explored_branches = 0;
  ginfo_.n_completed_branches = 0;
  ginfo_.n_remaining_branches = 0;
  ginfo_.n_total_branches = 0;
  for (auto &i_ : infos_) {
    ginfo_.n_explored_branches += i_.n_explored_branches;
    ginfo_.n_completed_branches += i_.n_completed_branches;
    ginfo_.n_remaining_branches += i_.n_remaining_branches;
  }
  for (const auto &br_ : completed_branches_)
    ginfo_.n_total_branches += std::size(br_);

  // Accumulate otoc
  {
    auto &log10_norm_atols_ = infos_[0].log10_norm_atols;
    auto &otoc_values_ = infos_[0].otoc_values;
    auto &otoc_values_no_int_ = infos_[0].otoc_values_no_int;
    auto &otoc_norms0_ = infos_[0].otoc_norms0;
    auto &otoc_norms1_ = infos_[0].otoc_norms1;
    auto &otoc_n_completed_branches_ = infos_[0].otoc_n_completed_branches;
    auto &otoc_n_explored_branches_ = infos_[0].otoc_n_explored_branches;

    // Resize
    log10_norm_atols_.resize(n_na_buckets);
    otoc_values_.resize(n_na_buckets);
    otoc_values_no_int_.resize(n_na_buckets);
    otoc_norms0_.resize(n_na_buckets);
    otoc_norms1_.resize(n_na_buckets);
    otoc_n_completed_branches_.resize(n_na_buckets);
    otoc_n_explored_branches_.resize(n_na_buckets);

    // Initialize
    for (std::size_t j_ = 0; j_ < n_na_buckets; ++j_)
      log10_norm_atols_[j_] = -log10_dnorm_atol * j_;
    std::fill(std::begin(otoc_values_), std::end(otoc_values_), 0);
    std::fill(std::begin(otoc_values_no_int_), std::end(otoc_values_no_int_),
              0);
    std::fill(std::begin(otoc_norms0_), std::end(otoc_norms0_), 0);
    std::fill(std::begin(otoc_norms1_), std::end(otoc_norms1_), 0);
    std::fill(std::begin(otoc_n_completed_branches_),
              std::end(otoc_n_completed_branches_), 0);
    std::fill(std::begin(otoc_n_explored_branches_),
              std::end(otoc_n_explored_branches_), 0);

    // Merge otocs
    for (std::size_t j_ = 0; j_ < n_na_buckets; ++j_) {
      for (std::size_t i_ = 0; i_ < n_buckets; ++i_) {
        otoc_values_[j_] += v_otoc_value_[i_][j_];
        otoc_values_no_int_[j_] += v_otoc_value_no_int_[i_][j_];
        otoc_norms0_[j_] += v_norm0_[i_][j_];
        otoc_norms1_[j_] += v_norm1_[i_][j_];
      }
      for (std::size_t i_ = 0; i_ < n_threads; ++i_) {
        otoc_n_completed_branches_[j_] += v_n_completed_branches_[i_][j_];
        otoc_n_explored_branches_[j_] += v_n_explored_branches_[i_][j_];
      }
    }

    // Accumulate
    for (std::size_t j_ = 1; j_ < n_na_buckets; ++j_) {
      otoc_values_[j_] += otoc_values_[j_ - 1];
      otoc_values_no_int_[j_] += otoc_values_no_int_[j_ - 1];
      otoc_norms0_[j_] += otoc_norms0_[j_ - 1];
      otoc_norms1_[j_] += otoc_norms1_[j_ - 1];
      otoc_n_completed_branches_[j_] += otoc_n_completed_branches_[j_ - 1];
      otoc_n_explored_branches_[j_] += otoc_n_explored_branches_[j_ - 1];
    }

    // Get sqrt of norms
    std::transform(std::begin(otoc_norms0_), std::end(otoc_norms0_),
                   std::begin(otoc_norms0_),
                   [](auto &&x) { return std::sqrt(x); });
    std::transform(std::begin(otoc_norms1_), std::end(otoc_norms1_),
                   std::begin(otoc_norms1_),
                   [](auto &&x) { return std::sqrt(x); });
  }

  // Clean-up
  {
    // Start time
    auto tic_ = std::chrono::system_clock::now();

    // Initialize cleaning routine
    std::atomic_uint32_t index_{0};
    auto clean_ = [&index_, &completed_branches_]() {
      std::size_t i_;
      while ((i_ = index_++) < std::size(completed_branches_))
        completed_branches_type temp_ = std::move(completed_branches_[i_]);
    };

    // Initialize threads
    std::vector<decltype(std::async(clean_))> threads_;
    for (std::size_t i_ = 0; i_ < n_threads; ++i_)
      threads_.push_back(std::async(clean_));

    // Release
    for (auto &t_ : threads_) t_.get();

    // Stop time
    auto toc_ = std::chrono::system_clock::now();

    // Get delta time
    const auto dt_ =
        std::chrono::duration_cast<std::chrono::milliseconds>(toc_ - tic_)
            .count();

    // Update info
    ginfo_.cleaning_time_ms = dt_;
  }

  // Return global info
  return std::tuple{ginfo_, std::move(branches_)};
}

struct Simulator {
  using core_output = std::tuple<info_type, list_branches_type>;
  const hqc::vector_type<phases_type> phases;
  const hqc::vector_type<positions_type> positions;
  const hqc::vector_type<qubits_type> qubits;
  const std::size_t target_position;
  const hqc::IVector1D initial_state;
  const hqc::float_type atol{1e-8};
  const hqc::float_type norm_atol{1e-8};
  const hqc::float_type merge_atol{1e-8};
  const std::size_t n_threads{0};
  const std::size_t log2_n_buckets{12};
  const bool verbose{false};

  Simulator(const hqc::vector_type<phases_type> &phases,
            const hqc::vector_type<positions_type> &positions,
            const hqc::vector_type<qubits_type> &qubits,
            std::size_t target_position, const hqc::IVector1D &initial_state,
            hqc::float_type atol = 1e-8, hqc::float_type norm_atol = 1e-8,
            hqc::float_type merge_atol = 1e-8, std::size_t n_threads = 0,
            std::size_t log2_n_buckets = 12, bool verbose = false)
      : phases{phases},
        positions{positions},
        qubits{qubits},
        target_position{target_position},
        initial_state{initial_state},
        atol{atol},
        norm_atol{norm_atol},
        merge_atol{merge_atol},
        n_threads{n_threads > 0 ? n_threads
                                : std::thread::hardware_concurrency()},
        log2_n_buckets{log2_n_buckets},
        verbose{verbose} {
    // Check norm atol
    if (norm_atol > 1 || norm_atol < 0)
      throw std::logic_error("'norm_atol' must be within 0 and 1 (included).");
  }

  template <typename Branches>
  auto start(Branches &&branches, bool expand_branches_only = false) {
    // Initialize stop signal
    _stop = false;

    // Initialize core
    _core = [branches, expand_branches_only, this]() -> core_output {
      return UpdateBranches_(branches, phases, positions, qubits,
                             target_position, initial_state, atol, norm_atol,
                             merge_atol, n_threads, log2_n_buckets,
                             expand_branches_only, _stop, _infos);
    };

    // Run simulation (in background)
    _thread = std::async(_core);
  }

  // Join simulation
  auto join() {
    // Wait simulation to finish and get global info
    auto &&[ginfo_, branches_] = _thread.get();

    // Convert list to vector
    std::vector<branch_type> v_branches_;
    std::move(std::begin(branches_), std::end(branches_),
              std::back_inserter(v_branches_));

    // Return results
    return std::tuple{std::move(ginfo_), std::move(v_branches_),
                      std::move(_infos[0])};
  }

  // Stop simulation
  auto stop() {
    // Set stop signal to stop
    _stop = true;

    // Join
    return join();
  }

  // Return true if simulation is ready
  auto ready(std::size_t ms = 0) {
    return _thread.wait_for(std::chrono::milliseconds(ms)) ==
           std::future_status::ready;
  }

  // Return constant reference to infos
  const auto &infos() const { return _infos; }

 private:
  int _stop;
  hqc::vector_type<info_type> _infos;
  std::function<core_output()> _core;
  std::future<core_output> _thread;
};

PYBIND11_MAKE_OPAQUE(hqc::BVector1D);
PYBIND11_MAKE_OPAQUE(hqc::FVector1D);
PYBIND11_MAKE_OPAQUE(hqc::FVector2D);
PYBIND11_MAKE_OPAQUE(hqc::FVector3D);
PYBIND11_MAKE_OPAQUE(hqc::IVector1D);
PYBIND11_MAKE_OPAQUE(hqc::IVector2D);
PYBIND11_MAKE_OPAQUE(hqc::IVector3D);
PYBIND11_MAKE_OPAQUE(completed_branches_type);
PYBIND11_MAKE_OPAQUE(hqc::vector_type<completed_branches_type>);
PYBIND11_MAKE_OPAQUE(hqc::vector_type<info_type>);
PYBIND11_MAKE_OPAQUE(hqc::vector_type<branch_type>);

PYBIND11_MODULE(otoc, m) {
  m.doc() =
      "Simulation of quantum circuits using the "
      "Clifford-expansion technique.";
  //
  auto m_utils = m.def_submodule("utils", "Utilities.");
  //
  ADD_OPAQUE_VECTOR(m, hqc::BVector1D, "BVector1D");
  ADD_OPAQUE_VECTOR(m, hqc::FVector1D, "FVector1D");
  ADD_OPAQUE_VECTOR(m, hqc::FVector2D, "FVector2D");
  ADD_OPAQUE_VECTOR(m, hqc::FVector3D, "FVector3D");
  ADD_OPAQUE_VECTOR(m, hqc::IVector1D, "IVector1D");
  ADD_OPAQUE_VECTOR(m, hqc::IVector2D, "IVector2D");
  ADD_OPAQUE_VECTOR(m, hqc::IVector3D, "IVector3D");
  ADD_OPAQUE_MAP(m, completed_branches_type, "CompletedBranches");
  ADD_OPAQUE_VECTOR(m, hqc::vector_type<completed_branches_type>,
                    "VectorCompletedBranches");
  ADD_OPAQUE_VECTOR(m, hqc::vector_type<info_type>, "VectorInfo");
  ADD_OPAQUE_VECTOR(m, hqc::vector_type<branch_type>, "VectorBranch");
  //
  m_utils.def("StateFromPauli", &hqc::StateFromPauli<state_type>,
              py::arg("paulis"), py::pos_only(),
              "Return `State` from a Pauli string.");
  m_utils.def("PauliFromState", &hqc::PauliFromState<const state_type &>,
              py::arg("state"), py::pos_only(),
              "Return Pauli string from `State`.");
  m_utils.def(
      "DumpBranches",
      [](const hqc::vector_type<branch_type> &branches) {
        return py::bytes(hqc::DumpBranches(branches));
      },
      py::arg("branches"), py::pos_only(), "Dump branches.");
  m_utils.def(
      "LoadBranches",
      [](const std::string &buffer) {
        return hqc::LoadBranches<hqc::vector_type, branch_type>(buffer);
      },
      py::arg("buffer"), py::pos_only(), "Load branches.");
  //
  py::class_<state_type>(m, "State")
      .def(py::init<std::size_t>(), py::arg("n"), py::pos_only(),
           "Initialize state.")
      .def(py::init<const hqc::BVector1D &>(), py::arg("a"), py::pos_only(),
           "Initialize state.")
      .def("get", &state_type::get, py::arg("pos"), py::pos_only(),
           "Get bit in given position.")
      .def("set", &state_type::set, py::arg("pos"), py::arg("value"),
           py::pos_only(), "Set bit in given position.")
      .def(
          "data", [](const state_type &self) { return self.data(); },
          "Get underlying data.")
      .def("__eq__", [](const state_type &self,
                        const state_type &other) { return self == other; })
      .def("__len__", [](const state_type &self) { return std::size(self); })
      .def("__repr__", [](const state_type &self) {
        std::string repr_;
        repr_ = "State(";
        for (std::size_t i_ = 0, end_ = std::size(self); i_ < end_; ++i_)
          repr_ += self.get(i_) ? '1' : '0';
        repr_ += ")";
        return repr_;
      });
  //
  py::class_<branch_type>(m, "Branch")
      .def(py::init<>())
      .def(py::init<state_type, phase_type, phase_type, hqc::index_type>())
      .def_readonly("state", &branch_type::state)
      .def_readonly("phase", &branch_type::phase)
      .def_readonly("norm_phase", &branch_type::norm_phase)
      .def_readonly("gate_idx", &branch_type::gate_idx)
      .def("__repr__", [](const branch_type &b) {
        std::stringstream ss;
        b << ss;
        return "Branch" + ss.str();
      });

  //
  py::class_<info_type>(m, "Info")
      .def(py::init<>())
      .def_readonly("n_explored_branches", &info_type::n_explored_branches)
      .def_readonly("n_remaining_branches", &info_type::n_remaining_branches)
      .def_readonly("n_completed_branches", &info_type::n_completed_branches)
      .def_readonly("n_threads", &info_type::n_threads)
      .def_readonly("runtime_ms", &info_type::runtime_ms)
      .def_readonly("branching_time_ms", &info_type::branching_time_ms)
      .def_readonly("expanding_time_ms", &info_type::expanding_time_ms)
      .def_readonly("log10_norm_atols", &info_type::log10_norm_atols)
      .def_readonly("otoc_values", &info_type::otoc_values)
      .def_readonly("otoc_values_no_int", &info_type::otoc_values_no_int)
      .def_readonly("otoc_norms0", &info_type::otoc_norms0)
      .def_readonly("otoc_norms1", &info_type::otoc_norms1)
      .def_readonly("otoc_n_completed_branches",
                    &info_type::otoc_n_completed_branches)
      .def_readonly("otoc_n_explored_branches",
                    &info_type::otoc_n_explored_branches)
      .def("dict", &info_type::dict);
  //
  py::class_<Simulator>(m, "Simulator")
      .def(py::init<hqc::vector_type<phases_type>,
                    hqc::vector_type<positions_type>,
                    hqc::vector_type<qubits_type>, std::size_t, hqc::IVector1D,
                    hqc::float_type, hqc::float_type, hqc::float_type,
                    std::size_t, std::size_t, bool>(),
           py::arg("phases"), py::arg("positions"), py::arg("qubits"),
           py::kw_only(), py::arg("target_position"), py::arg("initial_state"),
           py::arg("atol") = hqc::float_type{1e-8},
           py::arg("norm_atol") = hqc::float_type{1e-8},
           py::arg("merge_atol") = hqc::float_type{1e-8},
           py::arg("n_threads") = 0, py::arg("log2_n_buckets") = 12,
           py::arg("verbose") = false)
      .def("start", &Simulator::start<const hqc::vector_type<branch_type> &>,
           py::arg("branches"), py::pos_only(), py::kw_only(),
           py::arg("expand_branches_only") = false)
      .def("stop", &Simulator::stop)
      .def("join", &Simulator::join)
      .def("ready", &Simulator::ready, py::arg("ms") = 0, py::pos_only())
      .def_readonly("atol", &Simulator::atol)
      .def_readonly("norm_atol", &Simulator::norm_atol)
      .def_readonly("merge_atol", &Simulator::merge_atol)
      .def_readonly("n_threads", &Simulator::n_threads)
      .def_readonly("log2_n_buckets", &Simulator::log2_n_buckets)
      .def_readonly("verbose", &Simulator::verbose)
      .def_property_readonly("infos", &Simulator::infos)
      .def("__repr__", [](const Simulator &self) {
        std::string out_;
        out_ += "Simulator(";
        out_ += "atol=" + std::to_string(self.atol) + ", ";
        out_ += "norm_atol=" + std::to_string(self.norm_atol) + ", ";
        out_ += "merge_atol=" + std::to_string(self.merge_atol) + ", ";
        out_ += "n_threads=" + std::to_string(self.n_threads) + ", ";
        out_ += "log2_n_buckets=" + std::to_string(self.log2_n_buckets) + ", ";
        out_ += "verbose=" + std::to_string(self.verbose) + ")";
        return out_;
      });
}
#endif