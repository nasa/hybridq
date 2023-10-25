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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "branch.hpp"
#include "defs.hpp"
#include "info.hpp"
#include "simulation.hpp"
#include "utils.hpp"

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

// Set specific types
using info_type = hqc::Info;
using state_type = hqc::BVector1D;
using phase_type = hqc::float_type;
using branch_type = hqc::Branch<state_type, phase_type, hqc::index_type>;
using list_branches_type = hqc::list_type<branch_type>;
using phases_type = hqc::FVector2D;
using positions_type = hqc::IVector2D;
using qubits_type = hqc::IVector1D;
using completed_branches_type = hqc::map_type<state_type, phase_type>;

auto UpdateBranches_(
    list_branches_type &branches, const hqc::vector_type<phases_type> &phases,
    const hqc::vector_type<positions_type> &positions,
    const hqc::vector_type<qubits_type> &qubits, const hqc::float_type atol,
    const hqc::float_type norm_atol, const hqc::float_type merge_atol,
    const std::size_t n_threads, const std::size_t log2_n_buckets,
    bool expand_branches_only, int &stop_,
    hqc::vector_type<completed_branches_type> &completed_branches_,
    hqc::vector_type<info_type> &infos_) {
  // Initialize number of buckets
  const std::size_t n_buckets = 1uL << log2_n_buckets;

  // Initialize global infos
  info_type ginfo_;

  // Initialize infos
  infos_.clear();
  infos_.resize(n_threads);

  // Initialize completed branches
  completed_branches_.clear();
  completed_branches_.resize(n_buckets);

  // Get hash of a given state
  auto hash_ = [log2_n_buckets](auto &&state) {
    std::size_t n_ = 0;
    for (std::size_t i_ = 0, end_ = std::min(log2_n_buckets, std::size(state));
         i_ < end_; ++i_)
      n_ |= (state[i_] << i_);
    return n_;
  };

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
  hqc::UpdateAllBranches<hqc::vector_type>(
      branches, phases, positions, qubits, atol, norm_atol, n_threads,
      expand_branches_only, stop_, ginfo_, update_completed_branches_);

  // Update last info
  ginfo_.n_threads = n_threads;
  ginfo_.n_explored_branches = 0;
  ginfo_.n_completed_branches = 0;
  ginfo_.n_remaining_branches = 0;
  for (auto &i_ : infos_) {
    ginfo_.n_explored_branches += i_.n_explored_branches;
    ginfo_.n_completed_branches += i_.n_completed_branches;
    ginfo_.n_remaining_branches += i_.n_remaining_branches;
  }

  // Return info
  return ginfo_;
}

struct Simulator {
  const hqc::vector_type<phases_type> phases;
  const hqc::vector_type<positions_type> positions;
  const hqc::vector_type<qubits_type> qubits;
  const hqc::float_type atol{1e-8};
  const hqc::float_type norm_atol{1e-8};
  const hqc::float_type merge_atol{1e-8};
  const std::size_t n_threads{0};
  const std::size_t log2_n_buckets{12};
  const bool verbose{false};

  Simulator(const hqc::vector_type<phases_type> &phases,
            const hqc::vector_type<positions_type> &positions,
            const hqc::vector_type<qubits_type> &qubits,
            hqc::float_type atol = 1e-8, hqc::float_type norm_atol = 1e-8,
            hqc::float_type merge_atol = 1e-8, std::size_t n_threads = 0,
            std::size_t log2_n_buckets = 12, bool verbose = false)
      : phases{phases},
        positions{positions},
        qubits{qubits},
        atol{atol},
        norm_atol{norm_atol},
        merge_atol{merge_atol},
        n_threads{n_threads > 0 ? n_threads
                                : std::thread::hardware_concurrency()},
        log2_n_buckets{log2_n_buckets},
        verbose{verbose} {}

  auto start(list_branches_type &branches, bool expand_branches_only = false) {
    // Initialize stop signal
    _stop = false;

    // Initialize branches
    _branches = std::move(branches);

    // Initialize core
    _core = [expand_branches_only, this]() -> info_type {
      return UpdateBranches_(_branches, phases, positions, qubits, atol,
                             norm_atol, merge_atol, n_threads, log2_n_buckets,
                             expand_branches_only, _stop, _completed_branches,
                             _infos);
    };

    // Run simulation (in background)
    _thread = std::async(_core);
  }

  // Join simulation
  auto join() {
    // Wait simulation to complete
    auto ginfo_ = _thread.get();

    // Return results
    return std::tuple{std::move(ginfo_), std::move(_branches),
                      std::move(_completed_branches)};
  }

  // Stop simulation
  auto stop() {
    // Set stop signal to stop
    _stop = true;

    // Join
    return _thread.get();
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
  list_branches_type _branches;
  hqc::vector_type<completed_branches_type> _completed_branches;
  hqc::vector_type<info_type> _infos;
  std::function<info_type()> _core;
  std::future<info_type> _thread;
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

PYBIND11_MODULE(hybridq_clifford_core, m) {
  m.doc() =
      "Simulation of quantum circuits using the "
      "Clifford-expansion technique.";
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
  //
  m.def("StateFromPauli", &hqc::StateFromPauli<state_type>, py::arg("paulis"),
        py::pos_only(), "Return `State` from a Pauli string.");
  m.def("PauliFromState", &hqc::PauliFromState<const state_type &>,
        py::arg("state"), py::pos_only(), "Return Pauli string from `State`.");
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
      .def("dict", &info_type::dict);
  //
  py::class_<Simulator>(m, "Simulator")
      .def(py::init<
               hqc::vector_type<phases_type>, hqc::vector_type<positions_type>,
               hqc::vector_type<qubits_type>, hqc::float_type, hqc::float_type,
               hqc::float_type, std::size_t, std::size_t, bool>(),
           py::arg("phases"), py::arg("positions"), py::arg("qubits"),
           py::kw_only(), py::arg("atol") = hqc::float_type{1e-8},
           py::arg("norm_atol") = hqc::float_type{1e-8},
           py::arg("merge_atol") = hqc::float_type{1e-8},
           py::arg("n_threads") = 0, py::arg("log2_n_buckets") = 12,
           py::arg("verbose") = false)
      .def("start", &Simulator::start, py::arg("branches"), py::pos_only(),
           py::kw_only(), py::arg("expand_branches_only") = false)
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
