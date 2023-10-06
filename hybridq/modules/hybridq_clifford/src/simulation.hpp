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

#include <pybind11/pybind11.h>

#include <cassert>
#include <chrono>
#include <future>
#include <iomanip>
#include <limits>
#include <sstream>

#include "defs.hpp"
#include "utils.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

namespace hybridq_clifford {

auto UpdateBranch(const branch_type &branch,
                  const std::vector<phases_type> &phases,
                  const std::vector<positions_type> &positions,
                  const std::vector<qubits_type> &qubits,
                  const float_type atol = 1e-8,
                  const float_type norm_atol = 1e-8) {
  /*
   * @assumption: phases[0] is always the largest phase.
   */

  // Initialize new branches
  std::list<branch_type> branches_;

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
  const auto max_ph_ = [&phases_]() {
    phase_type max_ph_{0};
    for (const auto &ph_ : phases_)
      if (auto aph_ = std::abs(ph_); max_ph_ < aph_) max_ph_ = aph_;
    return max_ph_;
  }();

  // Get new branches
  for (std::size_t i_ = 0, end_ = std::size(phases_); i_ < end_; ++i_) {
    const auto ph_ = branch.phase * phases_[i_];
    const auto norm_ph_ = branch.norm_phase * phases_[i_] / max_ph_;
    const auto ps_ = positions_[i_];

    // If new phase is large enough ...
    if (i_ == 0 || (std::abs(ph_) > atol && std::abs(norm_ph_) > norm_atol)) {
      // Get a new branch
      branches_.push_back(
          branch_type{branch.state, ph_, norm_ph_, branch.gate_idx + 1});
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

  // Return the new branches
  return branches_;
}

template <bool depth_first, typename InitializeCompletedBranches,
          typename UpdateCompletedBranches>
auto UpdateBranches_(
    std::list<branch_type> &branches, const std::vector<phases_type> &phases,
    const std::vector<positions_type> &positions,
    const std::vector<qubits_type> &qubits, const float_type atol,
    const float_type norm_atol, std::shared_ptr<info_type> info_,
    std::shared_ptr<bool> stop_,
    InitializeCompletedBranches &&initialize_completed_branches_,
    UpdateCompletedBranches &&update_completed_branches_) {
  // Get number of gates
  const std::size_t n_gates_ = std::size(phases);

  // Initialize completed branches
  auto completed_branches_ = initialize_completed_branches_();

  // While there are branches
  while (std::size(branches) && !*stop_) {
    // Get branch
    auto branch_ = [&branches]() {
      if constexpr (depth_first)
        return branches.back();
      else
        return branches.front();
    }();
    if constexpr (depth_first)
      branches.pop_back();
    else
      branches.pop_front();

    // Pop last branch and update it
    auto new_branches_ =
        UpdateBranch(branch_, phases, positions, qubits, atol, norm_atol);

    // Append to completed if all gates have been explored
    if (new_branches_.back().gate_idx == n_gates_) {
      info_->n_completed_branches += std::size(new_branches_);
      update_completed_branches_(completed_branches_, new_branches_);
    }

    // Otherwise, append them to branches
    else
      branches.splice(std::end(branches), new_branches_);

    // Update infos
    info_->n_explored_branches += 1;
    info_->n_remaining_branches = std::size(branches);
  }

  // Return both partial and completed branches
  return completed_branches_;
}

template <typename InitializeCompletedBranches,
          typename UpdateCompletedBranches>
auto ExpandBranches_(
    std::list<branch_type> &branches, const std::vector<phases_type> &phases,
    const std::vector<positions_type> &positions,
    const std::vector<qubits_type> &qubits, const std::size_t max_time_ms,
    const std::size_t min_n_branches, const float_type atol,
    const float_type norm_atol,
    InitializeCompletedBranches &&initialize_completed_branches_,
    UpdateCompletedBranches &&update_completed_branches_) {
  // Initialize stop signal and info
  auto stop_ = std::shared_ptr<bool>(new bool{false});
  auto info_ = std::shared_ptr<info_type>(new info_type{});

  // Initialize core to call
  auto update_brs_ = [&]() {
    // Breadth-First
    return UpdateBranches_<false>(
        branches, phases, positions, qubits, atol, norm_atol, info_, stop_,
        initialize_completed_branches_, update_completed_branches_);
  };

  // Initialize job
  auto th_ = std::async(update_brs_);

  // Expand
  while (th_.wait_for(std::chrono::milliseconds(max_time_ms)) !=
             std::future_status::ready &&
         std::size(branches) < min_n_branches)
    ;

  // Flag to stop
  *stop_ = true;

  // Return completed branches
  return std::tuple{std::move(th_.get()), info_};
}

auto SplitBranches_(std::list<branch_type> &branches, std::size_t n_buckets) {
  std::vector<std::list<branch_type>> v_branches_(n_buckets);
  if (n_buckets == 1)
    v_branches_[0].splice(std::end(v_branches_[0]), branches);
  else {
    std::size_t i_{0};
    while (std::size(branches)) {
      v_branches_[i_++ % n_buckets].push_back(std::move(branches.front()));
      branches.pop_front();
    }
  }
  return v_branches_;
}

auto MergeBranches_(std::vector<std::list<branch_type>> &v_branches) {
  std::list<branch_type> branches_;
  for (auto &x_ : v_branches) branches_.splice(std::end(branches_), x_);
  return branches_;
}

template <typename Time>
auto PrintInfo_(const std::vector<std::shared_ptr<info_type>> &infos,
                Time &&initial_time, std::size_t n_running_threads) {
  // Get stderr
  auto stderr_ = py::module_::import("sys").attr("stderr");

  const auto dt_ = std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::system_clock::now() - initial_time)
                       .count();
  const auto hrs_ = static_cast<std::size_t>(1e-6 * dt_ / 3600);
  const auto min_ = static_cast<std::size_t>(1e-6 * dt_ / 60) % 60;
  const auto sec_ = static_cast<std::size_t>(1e-6 * dt_) % 60;

  std::size_t n_explored_branches_ = 0;
  std::size_t n_remaining_branches_ = 0;
  std::size_t n_completed_branches_ = 0;
  for (const auto &info_ : infos) {
    n_explored_branches_ += info_->n_explored_branches;
    n_remaining_branches_ += info_->n_remaining_branches;
    n_completed_branches_ += info_->n_completed_branches;
  }

  // Build message
  std::stringstream ss_;
  ss_ << std::setprecision(2);
  ss_ << "NT=" << n_running_threads << "/" << std::size(infos);
  ss_ << ", EB=" << n_explored_branches_;
  ss_ << ", RB=" << n_remaining_branches_;
  ss_ << ", CB=" << n_completed_branches_;
  ss_ << " (ET=" << std::setfill('0') << std::setw(2) << hrs_ << ":";
  ss_ << std::setw(2) << min_ << ":" << std::setw(2) << sec_;
  ss_ << ", BT=" << (static_cast<float>(dt_) / n_explored_branches_) << "μs)";

  // Cut message if needed
  auto msg_ = ss_.str();
  if (std::size(msg_) < 100)
    msg_.insert(std::end(msg_), 100 - std::size(msg_), ' ');
  else {
    msg_ = msg_.substr(0, 97);
    msg_ += "...";
  }

  // Print message
  py::print(msg_, "end"_a = "\r", "flush"_a = true, "file"_a = stderr_);
}

template <typename InitializeCompletedBranches, typename MergeCompletedBranches,
          typename UpdateCompletedBranches>
auto UpdateBranches(
    std::list<branch_type> &branches, const std::vector<phases_type> &phases,
    const std::vector<positions_type> &positions,
    const std::vector<qubits_type> &qubits, const float_type atol,
    const float_type norm_atol, unsigned int n_threads, const bool verbose,
    InitializeCompletedBranches &&initialize_completed_branches_,
    MergeCompletedBranches &&merge_completed_branches_,
    UpdateCompletedBranches &&update_completed_branches_) {
  // Get stderr
  auto stderr_ = py::module_::import("sys").attr("stderr");

  // Get max number of threads
  n_threads = n_threads ? n_threads : std::thread::hardware_concurrency();

  // Initialize stop signal
  auto stop_ = std::shared_ptr<bool>(new bool{false});

  // Initialize infos
  std::vector<std::shared_ptr<info_type>> infos_(n_threads);
  std::generate(std::begin(infos_), std::end(infos_),
                []() { return std::shared_ptr<info_type>(new info_type{}); });

  // Get expanding time
  auto tic_1_ = std::chrono::system_clock::now();

  // Expand branches
  if (verbose)
    py::print("Expading branches ... ", "end"_a = "", "flush"_a = true,
              "file"_a = stderr_);
  auto [completed_brs_, init_info_] =
      n_threads == 1 ? std::tuple{initialize_completed_branches_(),
                                  std::shared_ptr<info_type>(new info_type{})}
                     : ExpandBranches_(branches, phases, positions, qubits, 100,
                                       n_threads * 10, atol, norm_atol,
                                       initialize_completed_branches_,
                                       update_completed_branches_);
  if (verbose) py::print("Done!", "flush"_a = true, "file"_a = stderr_);

  // Split branches
  auto v_branches_ = SplitBranches_(branches, n_threads);

  // Branches should be empty
  assert(!std::size(branches));

  // Get initial time
  auto tic_2_ = std::chrono::system_clock::now();

  // Initialize core to call
  auto update_brs_ = [&](std::size_t idx) {
    // Depth-First
    return UpdateBranches_<true>(v_branches_[idx], phases, positions, qubits,
                                 atol, norm_atol, infos_[idx], stop_,
                                 initialize_completed_branches_,
                                 update_completed_branches_);
  };

  // Initialize threads
  std::vector<decltype(std::async(update_brs_, 0))> threads_;
  for (std::size_t i_ = 0; i_ < n_threads; ++i_)
    threads_.push_back(std::async(update_brs_, i_));

  // Print info
  if (verbose) {
    // Define checks
    auto any_thread_running_ = [&threads_]() {
      for (const auto &th_ : threads_)
        if (th_.wait_for(std::chrono::seconds(1)) != std::future_status::ready)
          return true;
      return false;
    };
    auto n_running_threads_ = [&threads_]() {
      std::size_t n_{0};
      for (const auto &th_ : threads_)
        n_ +=
            th_.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
      return n_;
    };
    // auto any_thread_stopped_ = [&threads_]() {
    //   for (const auto &th_ : threads_)
    //     if (th_.wait_for(std::chrono::nanoseconds(0)) ==
    //         std::future_status::ready)
    //       return true;
    //   return false;
    // };
    while (any_thread_running_())
      PrintInfo_(infos_, tic_2_, n_running_threads_());

    // Otherwise, wait until ready
  } else
    for (const auto &th_ : threads_) th_.wait();

  // Get end time
  auto tic_3_ = std::chrono::system_clock::now();

  // Collect results
  for (std::size_t i_ = 0, end_ = std::size(threads_); i_ < end_; ++i_) {
    if (verbose)
      py::print(i_ ? "" : "\n", "Collecting results (", i_, "/", end_, ") ...",
                "sep"_a = "", "end"_a = "\r", "flush"_a = true,
                "file"_a = stderr_);

    // Get results
    auto &&partial_completed_ = threads_[i_].get();

    // Merge
    merge_completed_branches_(completed_brs_, partial_completed_);

    // Clean partial (no longer needed)
    partial_completed_.clear();
  }
  if (verbose)
    py::print("Collecting results ... Done!", "flush"_a = true,
              "file"_a = stderr_);

  // Get end time
  auto tic_4_ = std::chrono::system_clock::now();

  if (verbose)
    py::print("Merging partial branches ... ", "end"_a = "", "flush"_a = true,
              "file"_a = stderr_);

  // Merge branches
  branches = MergeBranches_(v_branches_);

  if (verbose) py::print("Done!", "flush"_a = true, "file"_a = stderr_);

  // Merge infos
  {
    auto exp_time_ =
        std::chrono::duration_cast<std::chrono::microseconds>(tic_2_ - tic_1_)
            .count();
    auto br_time_ =
        std::chrono::duration_cast<std::chrono::microseconds>(tic_3_ - tic_2_)
            .count();
    auto mrg_time_ =
        std::chrono::duration_cast<std::chrono::milliseconds>(tic_4_ - tic_3_)
            .count();
    auto rn_time_ =
        std::chrono::duration_cast<std::chrono::milliseconds>(tic_4_ - tic_1_)
            .count();
    init_info_->n_total_branches = std::size(completed_brs_);
    init_info_->n_threads = n_threads;
    init_info_->n_remaining_branches = std::size(branches);
    for (std::size_t i_ = 0; i_ < n_threads; ++i_) {
      init_info_->n_explored_branches += infos_[i_]->n_explored_branches;
      init_info_->n_completed_branches += infos_[i_]->n_completed_branches;
    }
    init_info_->expanding_time_ms = exp_time_;
    init_info_->merging_time_ms = mrg_time_;
    init_info_->branching_time_us =
        static_cast<float>(br_time_) / init_info_->n_explored_branches;
    init_info_->runtime_s = 1e-3 * rn_time_;
  }

  // Print stats
  if (verbose)
    py::print("\n", *init_info_, "sep"_a = "", "flush"_a = true,
              "file"_a = stderr_);

  // Return results
  return std::tuple{std::move(branches), std::move(completed_brs_),
                    std::move(init_info_)};
}

}  // namespace hybridq_clifford
