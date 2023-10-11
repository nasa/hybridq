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

#include <pybind11/complex.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <limits>
#include <sstream>

#include "simulation.hpp"
#include "utils.hpp"

namespace py = pybind11;
namespace hqc = hybridq_clifford;

using completed_branches_type =
    std::unordered_map<hqc::state_type, hqc::float_type>;
auto UpdateBranches_(std::list<hqc::branch_type> &branches,
                     const std::vector<hqc::phases_type> &phases,
                     const std::vector<hqc::positions_type> &positions,
                     const std::vector<hqc::qubits_type> &qubits,
                     const hqc::float_type atol = 1e-8,
                     const hqc::float_type norm_atol = 1e-8,
                     const hqc::float_type merge_atol = 1e-8,
                     unsigned int n_threads = 0, const bool verbose = false) {
  // Return dataset for completed branches
  auto initialize_completed_branches_ = []() {
    return completed_branches_type{};
  };

  // Merge branches from threads to the dataset of completed branches
  auto merge_completed_branches_ = [merge_atol](auto &&completed_brs,
                                                auto &&partial_completed) {
    if (std::size(completed_brs)) {
      for (auto w_ = std::begin(partial_completed),
                end_ = std::end(partial_completed);
           w_ != end_; ++w_)
        if (const auto x_ = (completed_brs[w_->first] += w_->second);
            std::abs(x_) < merge_atol)
          completed_brs.erase(w_->first);
    } else
      completed_brs = std::move(partial_completed);
  };

  // Update dataset of completed branches using explored branches
  auto update_completed_branches_ = [merge_atol](auto &&completed_brs,
                                                 auto &&new_brs) {
    for (auto &&b_ : new_brs)
      if (const auto x_ = (completed_brs[b_.state] += b_.phase);
          std::abs(x_) < merge_atol)
        completed_brs.erase(b_.state);
  };

  // Call hqc::UpdateBranches for the actual simulation
  return hqc::UpdateBranches(
      branches, phases, positions, qubits, atol, norm_atol, n_threads, verbose,
      initialize_completed_branches_, merge_completed_branches_,
      update_completed_branches_);
};

PYBIND11_MAKE_OPAQUE(hqc::state_type);
PYBIND11_MAKE_OPAQUE(hqc::IVector1D);
PYBIND11_MAKE_OPAQUE(hqc::IVector2D);
PYBIND11_MAKE_OPAQUE(hqc::IVector3D);
PYBIND11_MAKE_OPAQUE(hqc::FVector1D);
PYBIND11_MAKE_OPAQUE(hqc::FVector2D);
PYBIND11_MAKE_OPAQUE(hqc::FVector3D);
PYBIND11_MAKE_OPAQUE(hqc::SVector1D);
PYBIND11_MAKE_OPAQUE(hqc::SFVector1D);
PYBIND11_MAKE_OPAQUE(completed_branches_type);

PYBIND11_MODULE(hybridq_clifford_core, m) {
  m.doc() =
      "Simulation of quantum circuits using the "
      "Clifford-expansion technique.";
  //
  py::bind_vector<hqc::state_type>(m, "State", "State");
  py::implicitly_convertible<py::list, hqc::state_type>();
  py::implicitly_convertible<py::tuple, hqc::state_type>();
  py::implicitly_convertible<py::array, hqc::state_type>();
  //
  py::bind_vector<hqc::IVector1D>(m, "IVector1D", "IVector1D");
  py::bind_vector<hqc::IVector2D>(m, "IVector2D", "IVector2D");
  py::bind_vector<hqc::IVector3D>(m, "IVector3D", "IVector3D");
  py::implicitly_convertible<py::list, hqc::IVector1D>();
  py::implicitly_convertible<py::list, hqc::IVector2D>();
  py::implicitly_convertible<py::list, hqc::IVector3D>();
  py::implicitly_convertible<py::tuple, hqc::IVector1D>();
  py::implicitly_convertible<py::tuple, hqc::IVector2D>();
  py::implicitly_convertible<py::tuple, hqc::IVector3D>();
  py::implicitly_convertible<py::array, hqc::IVector1D>();
  py::implicitly_convertible<py::array, hqc::IVector2D>();
  py::implicitly_convertible<py::array, hqc::IVector3D>();
  //
  py::bind_vector<hqc::FVector1D>(m, "FVector1D", "FVector1D");
  py::bind_vector<hqc::FVector2D>(m, "FVector2D", "FVector2D");
  py::bind_vector<hqc::FVector3D>(m, "FVector3D", "FVector3D");
  py::implicitly_convertible<py::list, hqc::FVector1D>();
  py::implicitly_convertible<py::list, hqc::FVector2D>();
  py::implicitly_convertible<py::list, hqc::FVector3D>();
  py::implicitly_convertible<py::tuple, hqc::FVector1D>();
  py::implicitly_convertible<py::tuple, hqc::FVector2D>();
  py::implicitly_convertible<py::tuple, hqc::FVector3D>();
  py::implicitly_convertible<py::array, hqc::FVector1D>();
  py::implicitly_convertible<py::array, hqc::FVector2D>();
  py::implicitly_convertible<py::array, hqc::FVector3D>();
  //
  py::bind_vector<hqc::SVector1D>(m, "SVector1D", "SVector1D");
  py::bind_vector<hqc::SFVector1D>(m, "SFVector1D", "SFVector1D");
  py::implicitly_convertible<py::list, hqc::SVector1D>();
  py::implicitly_convertible<py::list, hqc::SFVector1D>();
  py::implicitly_convertible<py::tuple, hqc::SVector1D>();
  py::implicitly_convertible<py::tuple, hqc::SFVector1D>();
  py::implicitly_convertible<py::array, hqc::SVector1D>();
  // py::implicitly_convertible<py::array, hqc::SFVector1D>();
  //
  py::bind_map<completed_branches_type>(m, "BranchesType", "BranchesType");
  //
  m.def("GetPauli", &hqc::GetPauli, py::arg("state"), py::pos_only(),
        py::arg("pos"), "Get Pauli in position `pos` from `state`.");
  m.def("SetPauli", &hqc::SetPauli, py::arg("state"), py::pos_only(),
        py::arg("pos"), py::arg("op"),
        "Set Pauli `op` in position `pos` for `state`.");
  m.def("SetPauliFromChar", &hqc::SetPauliFromChar, py::arg("state"),
        py::pos_only(), py::arg("pos"), py::arg("op"),
        "Set Pauli `op` in position `pos` for `state`.");
  m.def("CountPaulis", &hqc::CountPaulis, py::arg("state"), py::pos_only(),
        "Count number of Pauli's in `state`.");
  m.def("StateFromPauli", &hqc::StateFromPauli, py::arg("paulis"),
        py::pos_only(), "Return `State` from a Pauli string.");
  m.def("PauliFromState", &hqc::PauliFromState, py::arg("state"),
        py::pos_only(), "Return Pauli string from `State`.");
  m.def("VectorFromState", &hqc::VectorFromState, py::arg("state"),
        py::pos_only(), "Return `IVector1D` from `State`.");
  m.def("UpdateBranches", &UpdateBranches_, py::arg("branches"),
        py::arg("phases"), py::arg("positions"), py::arg("qubits"),
        py::kw_only(), py::arg("atol") = hqc::float_type{1e-8},
        py::arg("norm_atol") = hqc::float_type{1e-8},
        py::arg("merge_atol") = hqc::float_type{1e-8}, py::arg("n_threads") = 0,
        py::arg("verbose") = false, "Update branches.");
  //
  py::class_<hqc::branch_type>(m, "Branch")
      .def(py::init([](const hqc::state_type &state, hqc::phase_type phase,
                       hqc::phase_type norm_phase, hqc::index_type gate_idx) {
        return new hqc::branch_type{state, phase, norm_phase, gate_idx};
      }))
      .def_readonly("state", &hqc::branch_type::state)
      .def_readonly("phase", &hqc::branch_type::phase)
      .def_readonly("norm_phase", &hqc::branch_type::norm_phase)
      .def_readonly("gate_idx", &hqc::branch_type::gate_idx)
      .def("__repr__", [](const hqc::branch_type &b) {
        std::stringstream ss;
        b << ss;
        return "Branch" + ss.str();
      });
  //
  py::class_<hqc::info_type>(m, "Info")
      .def(py::init())
      .def_readonly("n_explored_branches", &hqc::info_type::n_explored_branches)
      .def_readonly("n_remaining_branches",
                    &hqc::info_type::n_remaining_branches)
      .def_readonly("n_completed_branches",
                    &hqc::info_type::n_completed_branches)
      .def_readonly("n_total_branches", &hqc::info_type::n_total_branches)
      .def_readonly("n_threads", &hqc::info_type::n_threads)
      .def_readonly("runtime_s", &hqc::info_type::runtime_s)
      .def_readonly("branching_time_us", &hqc::info_type::branching_time_us)
      .def_readonly("merging_time_ms", &hqc::info_type::merging_time_ms)
      .def_readonly("expanding_time_ms", &hqc::info_type::expanding_time_ms)
      .def("__repr__", [](const hqc::info_type &info) {
        std::stringstream ss_;
        ss_ << info;
        return ss_.str();
      });
}
