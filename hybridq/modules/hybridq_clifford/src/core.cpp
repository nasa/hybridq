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
#include "state.hpp"
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
using state_type = hqc::State<hqc::vector_type>;
using phase_type = hqc::float_type;
using branch_type = hqc::Branch<state_type, phase_type, hqc::index_type>;
using list_branches_type = hqc::list_type<branch_type>;
using phases_type = hqc::FVector2D;
using positions_type = hqc::IVector2D;
using qubits_type = hqc::IVector1D;
using completed_branches_type = hqc::map_type<state_type, phase_type>;

template <typename Branches>
auto UpdateBranches_(
    Branches &&branches, const hqc::vector_type<phases_type> &phases,
    const hqc::vector_type<positions_type> &positions,
    const hqc::vector_type<qubits_type> &qubits, const hqc::float_type atol,
    const hqc::float_type norm_atol, const hqc::float_type merge_atol,
    const std::size_t n_threads, const std::size_t log2_n_buckets,
    bool expand_branches_only, int &stop_,
    hqc::vector_type<info_type> &infos_) {
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
  hqc::vector_type<completed_branches_type> completed_branches_(n_buckets);

  // Get hash of a given state
  auto hash_ = [mask =
                    log2_n_buckets == state_type::block_size
                        ? ~state_type::base_type{0}
                        : ((state_type::base_type{1} << log2_n_buckets) - 1)](
                   auto &&state) { return state.data()[0] & mask; };

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
      branches_, phases, positions, qubits, atol, norm_atol, n_threads,
      expand_branches_only, stop_, ginfo_, update_completed_branches_);

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
      std::tuple<info_type, hqc::vector_type<completed_branches_type>,
                 list_branches_type>;
  const hqc::vector_type<phases_type> phases;
  const hqc::vector_type<positions_type> positions;
  const hqc::vector_type<qubits_type> qubits;
  const hqc::float_type atol{1e-8};
  const hqc::float_type norm_atol{1e-8};
  const hqc::float_type merge_atol{1e-8};
  const std::size_t n_threads{0};
  const std::size_t log2_n_buckets{12};

  Simulator(const hqc::vector_type<phases_type> &phases,
            const hqc::vector_type<positions_type> &positions,
            const hqc::vector_type<qubits_type> &qubits,
            hqc::float_type atol = 1e-8, hqc::float_type norm_atol = 1e-8,
            hqc::float_type merge_atol = 1e-8, std::size_t n_threads = 0,
            std::size_t log2_n_buckets = 12)
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

PYBIND11_MODULE(hybridq_clifford_core, m) {
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
  m_utils.def("GetPauli", &hqc::GetPauli<const state_type &>, py::arg("state"),
              py::arg("pos"), py::pos_only(),
              "Return Pauli in position `pos` from `state`.");
  m_utils.def("SetPauli", &hqc::SetPauli<state_type &>, py::arg("state"),
              py::arg("pos"), py::arg("op"), py::pos_only(),
              "Set Pauli `op` in position `pos` of `state`.");
  m_utils.def("SetPauliFromChar", &hqc::SetPauliFromChar<state_type &>,
              py::arg("state"), py::arg("pos"), py::arg("op"), py::pos_only(),
              "Set Pauli `op` in position `pos` of `state`.");
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
      .def("__eq__", [](const branch_type &self,
                        const branch_type &other) { return self == other; })
      .def("__repr__", [](const branch_type &self) {
        std::stringstream ss;
        self << ss;
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
               hqc::float_type, std::size_t, std::size_t>(),
           py::arg("phases"), py::arg("positions"), py::arg("qubits"),
           py::kw_only(), py::arg("atol") = hqc::float_type{1e-8},
           py::arg("norm_atol") = hqc::float_type{1e-8},
           py::arg("merge_atol") = hqc::float_type{1e-8},
           py::arg("n_threads") = 0, py::arg("log2_n_buckets") = 12)
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
      .def_property_readonly("infos", &Simulator::infos)
      .def("__repr__", [](const Simulator &self) {
        std::string out_;
        out_ += "Simulator(";
        out_ += "atol=" + std::to_string(self.atol) + ", ";
        out_ += "norm_atol=" + std::to_string(self.norm_atol) + ", ";
        out_ += "merge_atol=" + std::to_string(self.merge_atol) + ", ";
        out_ += "n_threads=" + std::to_string(self.n_threads) + ", ";
        out_ += "log2_n_buckets=" + std::to_string(self.log2_n_buckets) + ", ";
        return out_;
      });
}
