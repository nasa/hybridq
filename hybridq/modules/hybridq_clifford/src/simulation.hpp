#pragma once

#include <pybind11/pybind11.h>

#include <cassert>
#include <chrono>
#include <future>
#include <iomanip>
#include <limits>
#include <sstream>

#include "defs.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

namespace hybridq_clifford {

// Get stderr
auto stderr_ = py::module_::import("sys").attr("stderr");

auto StateFromPauli(const std::string &paulis) {
  state_type state_(2 * std::size(paulis));
  for (std::size_t i_ = 0; i_ < std::size(paulis); ++i_)
    switch (std::toupper(paulis[i_])) {
      case 'I':
        state_[2 * i_ + 0] = 0;
        state_[2 * i_ + 1] = 0;
        break;
      case 'X':
        state_[2 * i_ + 0] = 1;
        state_[2 * i_ + 1] = 0;
        break;
      case 'Y':
        state_[2 * i_ + 0] = 0;
        state_[2 * i_ + 1] = 1;
        break;
      case 'Z':
        state_[2 * i_ + 0] = 1;
        state_[2 * i_ + 1] = 1;
        break;
    }
  return state_;
}

auto PauliFromState(const state_type &state) {
  std::string paulis_;
  for (std::size_t i_ = 0, end_ = std::size(state) / 2; i_ < end_; ++i_)
    switch (state[2 * i_ + 0] + 2 * state[2 * i_ + 1]) {
      case 0:
        paulis_ += 'I';
        break;
      case 1:
        paulis_ += 'X';
        break;
      case 2:
        paulis_ += 'Y';
        break;
      case 3:
        paulis_ += 'Z';
        break;
    }
  return paulis_;
}

auto UpdateBranch(const branch_type &branch,
                  const std::vector<phases_type> &phases,
                  const std::vector<positions_type> &positions,
                  const std::vector<qubits_type> &qubits,
                  const float atol = 1e-8, const float norm_atol = 1e-8) {
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

template <bool depth_first>
auto UpdateBranches_(std::list<branch_type> &branches,
                     const std::vector<phases_type> &phases,
                     const std::vector<positions_type> &positions,
                     const std::vector<qubits_type> &qubits, const float atol,
                     const float norm_atol, std::shared_ptr<info_type> info_,
                     std::shared_ptr<bool> stop_) {
  // Get number of gates
  const std::size_t n_gates_ = std::size(phases);

  // Initialize completed branches
  branches_type completed_branches_;

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
    if (new_branches_.back().gate_idx == n_gates_)
      for (auto &&b_ : new_branches_)
        completed_branches_[std::move(b_.state)] += b_.phase;

    // Otherwise, append them to branches
    else
      branches.splice(std::end(branches), new_branches_);

    // Update infos
    info_->n_explored_branches += 1;
    info_->n_remaining_branches = std::size(branches);
    info_->n_completed_branches = std::size(completed_branches_);
  }

  // Return both partial and completed branches
  return completed_branches_;
}

const auto &UpdateBranchesBreadthFirst = UpdateBranches_<false>;
const auto &UpdateBranchesDepthFirst = UpdateBranches_<true>;

auto ExpandBranches_(std::list<branch_type> &branches,
                     const std::vector<phases_type> &phases,
                     const std::vector<positions_type> &positions,
                     const std::vector<qubits_type> &qubits,
                     const std::size_t max_time_ms,
                     const std::size_t min_n_branches, const float atol,
                     const float norm_atol) {
  // Initialize stop signal and info
  auto stop_ = std::shared_ptr<bool>(new bool{false});
  auto info_ = std::shared_ptr<info_type>(new info_type{});

  // Initialize core to call
  auto update_brs_ = [&]() {
    return UpdateBranchesBreadthFirst(branches, phases, positions, qubits, atol,
                                      norm_atol, info_, stop_);
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
  return th_.get();
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
                Time &&initial_time) {
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
  ss_ << "NT=" << std::size(infos);
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

auto UpdateBranches(std::list<branch_type> &branches,
                    const std::vector<phases_type> &phases,
                    const std::vector<positions_type> &positions,
                    const std::vector<qubits_type> &qubits,
                    const float atol = 1e-8, const float norm_atol = 1e-8,
                    unsigned int n_threads = 0, const bool verbose = false) {
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
  auto completed_brs_ =
      n_threads == 1 ? branches_type{}
                     : ExpandBranches_(branches, phases, positions, qubits, 100,
                                       n_threads * 10, atol, norm_atol);
  if (verbose) py::print("Done!", "flush"_a = true, "file"_a = stderr_);

  // Split branches
  auto v_branches_ = SplitBranches_(branches, n_threads);

  // Branches should be empty
  assert(!std::size(branches));

  // Get initial time
  auto tic_2_ = std::chrono::system_clock::now();

  // Initialize core to call
  auto update_brs_ = [&](std::size_t idx) {
    return UpdateBranchesDepthFirst(v_branches_[idx], phases, positions, qubits,
                                    atol, norm_atol, infos_[idx], stop_);
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
    // auto any_thread_stopped_ = [&threads_]() {
    //   for (const auto &th_ : threads_)
    //     if (th_.wait_for(std::chrono::nanoseconds(0)) ==
    //         std::future_status::ready)
    //       return true;
    //   return false;
    // };
    while (any_thread_running_()) PrintInfo_(infos_, tic_2_);

    // Otherwise, wait until ready
  } else
    for (const auto &th_ : threads_) th_.wait();

  // Get end time
  auto tic_3_ = std::chrono::system_clock::now();

  // Get results
  if (n_threads == 1)
    completed_brs_ = threads_[0].get();
  else
    for (auto &th_ : threads_) {
      auto res_ = th_.get();
      for (auto w_ = std::begin(res_), end_ = std::end(res_); w_ != end_; ++w_)
        completed_brs_[w_->first] += w_->second;
    }

  // Get end time
  auto tic_4_ = std::chrono::system_clock::now();

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
    infos_[0]->n_threads = n_threads;
    for (std::size_t i_ = 1; i_ < n_threads; ++i_) {
      infos_[0]->n_explored_branches += infos_[i_]->n_explored_branches;
      infos_[0]->n_completed_branches += infos_[i_]->n_completed_branches;
      infos_[0]->n_remaining_branches += infos_[i_]->n_remaining_branches;
    }
    infos_[0]->expanding_time_ms = exp_time_;
    infos_[0]->merging_time_ms = mrg_time_;
    infos_[0]->branching_time_us =
        static_cast<float>(br_time_) / infos_[0]->n_explored_branches;
    infos_[0]->runtime_s = 1e-6 * rn_time_;
  }

  // Merge branches
  branches = MergeBranches_(v_branches_);

  // Print stats
  if (verbose)
    py::print("\n", *infos_[0], "sep"_a = "", "flush"_a = true,
              "file"_a = stderr_);

  // Return results
  return std::tuple{branches, completed_brs_, infos_[0]};
}

}  // namespace hybridq_clifford
