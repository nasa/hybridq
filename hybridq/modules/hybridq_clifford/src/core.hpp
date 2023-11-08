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

#include "branch.hpp"
#include "defs.hpp"
#include "info.hpp"
#include "simulation.hpp"
#include "state.hpp"
#include "utils.hpp"

namespace hybridq_clifford {

// Set specific types
using info_type = Info;
using state_type = State<vector_type>;
using phase_type = float_type;
using branch_type = Branch<state_type, phase_type, index_type>;
using list_branches_type = list_type<branch_type>;
using phases_type = FVector2D;
using positions_type = IVector2D;
using qubits_type = IVector1D;
using completed_branches_type = map_type<state_type, phase_type>;

template <typename Branches>
auto UpdateBranches_(Branches &&branches,
                     const vector_type<phases_type> &phases,
                     const vector_type<positions_type> &positions,
                     const vector_type<qubits_type> &qubits,
                     const float_type atol, const float_type norm_atol,
                     const float_type merge_atol, const std::size_t n_threads,
                     const std::size_t log2_n_buckets,
                     bool expand_branches_only, int &stop_,
                     vector_type<info_type> &infos_) {
  // Initialize number of buckets
  const std::size_t n_buckets = 1uL << log2_n_buckets;

  // Initialize branches
  list_branches_type branches_(std::begin(branches), std::end(branches));

  // Initialize global infos
  info_type ginfo_;

  // Initialize infos
  infos_.clear();
  infos_.resize(n_threads);

  // Initialize completed branches
  vector_type<completed_branches_type> completed_branches_(n_buckets);

  // Get hash of a given state
  auto hash_ = [mask =
                    log2_n_buckets == state_type::block_size
                        ? ~state_type::base_type{0}
                        : ((state_type::base_type{1} << log2_n_buckets) - 1)](
                   auto &&state) { return state._data[0] & mask; };

  // Update dataset of completed branches using explored branches
  auto update_completed_branches_ =
      [&completed_branches_, &hash_, n_buckets, merge_atol,
       n_gates = std::size(phases),
       &infos_](std::size_t idx, auto &&branch, auto &&branches_stack,
                auto &&new_branches, auto &&mutex) {
        // Update explored branches
        infos_[idx].n_explored_branches += 1;

        // If not completed branches, return
        if (new_branches.back().gate_idx != n_gates) {
          // lock database
          const std::lock_guard<std::mutex> lock_(mutex);

          // Update
          branches_stack.splice(std::end(branches_stack), new_branches);
        } else {
          // Update completed branches
          infos_[idx].n_completed_branches += std::size(new_branches);

          // Initialize mutex
          static std::vector<std::mutex> g_i_mutex_(n_buckets);

          // For each of the new branches ...
          for (const auto &new_branch_ : new_branches) {
            // Get hash of the new state
            const auto n_ = hash_(new_branch_.state);

            // Lock database and update
            const std::lock_guard<std::mutex> lock(g_i_mutex_[n_]);
            if (const auto x_ = (completed_branches_[n_][new_branch_.state] +=
                                 new_branch_.phase);
                std::abs(x_) < merge_atol)
              completed_branches_[n_].erase(new_branch_.state);
          }
        }

        // Update number of remaining branches
        infos_[idx].n_remaining_branches = std::size(branches_stack);
      };

  // Run the actual simulation
  UpdateAllBranches<vector_type>(branches_, phases, positions, qubits, atol,
                                 norm_atol, n_threads, expand_branches_only,
                                 stop_, ginfo_, update_completed_branches_);

  // Update last info
  ginfo_.n_threads = n_threads;
  ginfo_.n_explored_branches = 0;
  ginfo_.n_completed_branches = 0;
  ginfo_.n_remaining_branches = 0;
  ginfo_.n_total_branches = 0;
  for (const auto &i_ : infos_) {
    ginfo_.n_explored_branches += i_.n_explored_branches;
    ginfo_.n_completed_branches += i_.n_completed_branches;
    ginfo_.n_remaining_branches += i_.n_remaining_branches;
  }
  for (const auto &br_ : completed_branches_)
    ginfo_.n_total_branches += std::size(br_);

  // Return info
  return std::tuple{ginfo_, std::move(completed_branches_),
                    std::move(branches_)};
}

struct Simulator {
  using core_output =
      std::tuple<info_type, vector_type<completed_branches_type>,
                 list_branches_type>;
  const vector_type<phases_type> phases;
  const vector_type<positions_type> positions;
  const vector_type<qubits_type> qubits;
  const float_type atol{1e-8};
  const float_type norm_atol{1e-8};
  const float_type merge_atol{1e-8};
  const std::size_t n_threads{0};
  const std::size_t log2_n_buckets{12};

  Simulator(const vector_type<phases_type> &phases,
            const vector_type<positions_type> &positions,
            const vector_type<qubits_type> &qubits, float_type atol = 1e-8,
            float_type norm_atol = 1e-8, float_type merge_atol = 1e-8,
            std::size_t n_threads = 0, std::size_t log2_n_buckets = 12)
      : phases{phases},
        positions{positions},
        qubits{qubits},
        atol{atol},
        norm_atol{norm_atol},
        merge_atol{merge_atol},
        n_threads{n_threads > 0 ? n_threads
                                : std::thread::hardware_concurrency()},
        log2_n_buckets{log2_n_buckets} {}

  template <typename Branches>
  auto start(Branches &&branches, bool expand_branches_only = false) {
    // Initialize stop signal
    _stop = false;

    // Initialize core
    _core = [branches, expand_branches_only, this]() -> core_output {
      return UpdateBranches_(branches, phases, positions, qubits, atol,
                             norm_atol, merge_atol, n_threads, log2_n_buckets,
                             expand_branches_only, _stop, _infos);
    };

    // Run simulation (in background)
    _thread = std::async(_core);
  }

  // Join simulation
  auto join() {
    // Wait simulation to complete
    auto &&[ginfo_, completed_branches_, branches_] = _thread.get();

    // Convert list to vector
    std::vector<branch_type> v_branches_;
    std::move(std::begin(branches_), std::end(branches_),
              std::back_inserter(v_branches_));

    // Return results
    return std::tuple{std::move(ginfo_), std::move(v_branches_),
                      std::move(completed_branches_)};
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
  vector_type<info_type> _infos;
  std::function<core_output()> _core;
  std::future<core_output> _thread;
};

}  // namespace hybridq_clifford
