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

#include <cassert>
#include <chrono>
#include <future>
#include <mutex>
#include <thread>

#include "utils.hpp"

namespace hybridq_clifford {

template <typename ListBranches, typename Branch, typename VectorPhases,
          typename VectorPositions, typename VectorQubits, typename Float>
auto UpdateBranch(Branch &&branch, VectorPhases &&phases,
                  VectorPositions &&positions, VectorQubits &&qubits,
                  const Float atol = 1e-8, const Float norm_atol = 1e-8) {
  /*
   * @assumption: phases[0] is always the largest phase.
   */

  // Initialize new branches
  ListBranches branches_;

  // Get qubits
  const auto &qubits_ = qubits[branch.gate_idx];

  // Get substate
  std::size_t ss_ = 0;
  for (std::size_t i_ = 0, end_ = std::size(qubits_); i_ < end_; ++i_) {
    auto &&q_ = qubits_[i_];
    ss_ += (branch.state[2 * q_] + 2 * branch.state[2 * q_ + 1]) *
           (std::size_t{1} << (2 * i_));
  }

  // Get phases and positions
  const auto &phases_ = phases[branch.gate_idx][ss_];
  const auto &positions_ = positions[branch.gate_idx][ss_];

  // Get largest phase
  const auto max_ph_ = std::abs(phases_[0]);

  // Get new branches
  for (std::size_t i_ = 0, end_ = std::size(phases_); i_ < end_; ++i_) {
    const auto ph_ = branch.phase * phases_[i_];
    const auto norm_ph_ = branch.norm_phase * phases_[i_] / max_ph_;
    const auto ps_ = positions_[i_];

    // If new phase is large enough ...
    if (i_ == 0 || (std::abs(ph_) > atol && std::abs(norm_ph_) > norm_atol)) {
      // Get a new branch
      branches_.push_back(std::decay_t<Branch>{branch.state, ph_, norm_ph_,
                                               branch.gate_idx + 1});
      auto &new_branch_ = branches_.back();

      // Update branch
      for (std::size_t i_ = 0, end_ = std::size(qubits_); i_ < end_; ++i_) {
        const auto q_ = qubits_[i_];
        const auto new_ss_ = (ps_ / (std::size_t{1} << (2 * i_))) % 4;
        new_branch_.state[2 * q_ + 0] = new_ss_ & 0b01;
        new_branch_.state[2 * q_ + 1] = new_ss_ & 0b10;
      }
    }
  }

  // Sort branches
  branches_.sort(
      [](auto &&x, auto &&y) { return std::abs(x.phase) < std::abs(y.phase); });

  // Return the new branches
  return branches_;
}

template <bool depth_first, bool exhaustive = true, typename VectorListBranches,
          typename VectorPhases, typename VectorPositions,
          typename VectorQubits, typename Float,
          typename UpdateCompletedBranches,
          typename ListBranches =
              typename std::decay_t<VectorListBranches>::value_type,
          typename Branch = typename ListBranches::value_type>
void UpdateBranches(std::size_t idx, VectorListBranches &&branches,
                    VectorPhases &&phases, VectorPositions &&positions,
                    VectorQubits &&qubits, const Float atol,
                    const Float norm_atol, int &stop_,
                    UpdateCompletedBranches &&update_completed_branches_,
                    std::vector<std::mutex> &g_i_mutex_) {
  // Get number of remaning branches
  auto get_n_branches_ = [&branches, &g_i_mutex_]() {
    std::size_t c_ = 0;
    for (std::size_t i_ = 0, end_ = std::size(branches); i_ < end_; ++i_) {
      const std::lock_guard<std::mutex> lock_(g_i_mutex_[i_]);
      c_ += std::size(branches[i_]);
    }
    return c_;
  };

  // Get the idx with the largest number of branches
  auto idx_largest_n_branches_ = [&branches, &g_i_mutex_]() {
    std::size_t c_ = 0;
    std::optional<std::size_t> idx_;
    for (std::size_t i_ = 0, end_ = std::size(branches); i_ < end_; ++i_) {
      const std::lock_guard<std::mutex> lock_(g_i_mutex_[i_]);
      if (const auto s_ = std::size(branches[i_]); s_ > 1 && c_ < s_) {
        c_ = s_;
        idx_ = i_;
      }
    }
    return idx_;
  };

  // Select local branches
  auto &this_branches_ = branches[idx];
  auto &this_mutex_ = g_i_mutex_[idx];

  // Initialize temporary branch
  Branch branch_;

  // While there are global branches
  do {
    // if local branches is empty, rebalance
    if (!std::size(this_branches_) && !stop_) {
      // Sleep for a bit ...
      std::this_thread::sleep_for(std::chrono::milliseconds(100));

      // Get bucket with largest number of branches, lock and update
      if (const auto from_idx_ = idx_largest_n_branches_(); from_idx_) {
        const std::lock_guard<std::mutex> lock_1_(this_mutex_);
        const std::lock_guard<std::mutex> lock_2_(
            g_i_mutex_[from_idx_.value()]);

        // Get branch
        auto &from_branches_ = branches[from_idx_.value()];

        // Splice half of the branches
        this_branches_.splice(std::begin(this_branches_), from_branches_,
                              std::begin(from_branches_),
                              std::next(std::begin(from_branches_),
                                        std::size(from_branches_) / 2));
      }
    }

    // While there are local branches
    while (std::size(this_branches_) && !stop_) {
      // Lock and update
      {
        const std::lock_guard<std::mutex> lock_(this_mutex_);
        if constexpr (depth_first) {
          branch_ = std::move(this_branches_.back());
          this_branches_.pop_back();
        } else {
          branch_ = std::move(this_branches_.front());
          this_branches_.pop_front();
        }
      }

      // Pop last branch and update it
      auto new_branches_ = UpdateBranch<ListBranches>(
          branch_, phases, positions, qubits, atol, norm_atol);

      // Update completed branches
      update_completed_branches_(idx, branch_, this_branches_, new_branches_,
                                 this_mutex_);
    }
  } while (exhaustive && get_n_branches_());
}

template <template <typename> typename Vector, typename ListBranches,
          typename VectorPhases, typename VectorPositions,
          typename VectorQubits, typename Float,
          typename UpdateCompletedBranches>
auto ExpandBranches(ListBranches &&branches, VectorPhases &&phases,
                    VectorPositions &&positions, VectorQubits &&qubits,
                    const std::size_t max_time_ms,
                    const std::size_t min_n_branches, const Float atol,
                    const Float norm_atol,
                    UpdateCompletedBranches &&update_completed_branches_) {
  // Initialize stop signal and info
  int stop_ = false;

  // Move branches to a temporary vector
  Vector<std::decay_t<ListBranches>> branches_(1);
  branches_[0].splice(std::end(branches_[0]), branches);

  // Initialize temporary mutex
  std::vector<std::mutex> g_i_mutex_(1);

  // Initialize core to call
  auto update_brs_ = [&]() {
    // Breadth-First
    return UpdateBranches<false, false>(0, branches_, phases, positions, qubits,
                                        atol, norm_atol, stop_,
                                        update_completed_branches_, g_i_mutex_);
  };

  // Initialize job
  auto th_ = std::async(update_brs_);

  // Expand
  while (th_.wait_for(std::chrono::milliseconds(max_time_ms)) !=
             std::future_status::ready &&
         std::size(branches_[0]) < min_n_branches)
    ;

  // Flag to stop
  stop_ = true;

  // Release
  th_.get();

  // Move back to the original branches
  branches.splice(std::end(branches), branches_[0]);
}

template <template <typename...> typename Vector, typename ListBranches>
auto SplitBranches(ListBranches &&branches, std::size_t n_buckets) {
  // Initialize buckets
  Vector<std::decay_t<ListBranches>> v_branches_(n_buckets);

  // Split
  if (n_buckets == 1)
    v_branches_[0].splice(std::end(v_branches_[0]), branches);
  else {
    std::size_t i_{0};
    while (std::size(branches)) {
      v_branches_[i_++ % n_buckets].push_back(std::move(branches.front()));
      branches.pop_front();
    }
  }

  // Sort branches
  for (auto &br_ : v_branches_)
    br_.sort([](auto &&x, auto &&y) {
      return std::abs(x.phase) < std::abs(y.phase);
    });

  // Return split branches
  return v_branches_;
}

template <template <typename...> typename Vector, typename ListBranches,
          typename VectorPhases, typename VectorPositions,
          typename VectorQubits, typename Float, typename Info,
          typename UpdateCompletedBranches>
auto UpdateAllBranches(ListBranches &&branches, VectorPhases &&phases,
                       VectorPositions &&positions, VectorQubits &&qubits,
                       const Float atol, const Float norm_atol,
                       const std::size_t n_threads, bool expand_branches_only,
                       int &stop_, Info &&info_,
                       UpdateCompletedBranches &&update_completed_branches_) {
  // Check number of threads
  assert(n_threads > 0);

  // Get initial time
  const auto tic1_ = std::chrono::high_resolution_clock::now();

  // Update number of threads
  info_.n_threads = n_threads;

  if (expand_branches_only || n_threads > 1)
    ExpandBranches<Vector>(branches, phases, positions, qubits, 1000,
                           n_threads * 100, atol, norm_atol,
                           update_completed_branches_);

  // Just return if only the expansion of branches is needed
  if (expand_branches_only) return;

  // Update expansion time
  const auto tic2_ = std::chrono::high_resolution_clock::now();
  info_.expanding_time_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(tic2_ - tic1_)
          .count();

  // Split branches
  auto v_branches_ = SplitBranches<Vector>(branches, n_threads);

  // Branches should be empty
  assert(!std::size(branches));

  // Initialize mutex
  Vector<std::mutex> g_i_mutex_(n_threads);

  // Initialize core to call
  auto update_brs_ = [&](std::size_t idx) {
    // Depth-First
    return UpdateBranches<true>(idx, v_branches_, phases, positions, qubits,
                                atol, norm_atol, stop_,
                                update_completed_branches_, g_i_mutex_);
  };

  // Initialize threads
  Vector<decltype(std::async(update_brs_, 0))> threads_;
  for (std::size_t i_ = 0; i_ < n_threads; ++i_)
    threads_.push_back(std::async(update_brs_, i_));

  // Release
  for (auto &th_ : threads_) th_.get();

  // Update branching time
  const auto tic3_ = std::chrono::high_resolution_clock::now();
  info_.branching_time_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(tic3_ - tic2_)
          .count();

  // Merge branches
  for (auto &x_ : v_branches_) branches.splice(std::end(branches), x_);

  // Update runtime time
  const auto tic4_ = std::chrono::high_resolution_clock::now();
  info_.runtime_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(tic4_ - tic1_)
          .count();
}

}  // namespace hybridq_clifford
