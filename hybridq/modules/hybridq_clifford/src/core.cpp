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

#include "core.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "extras/otoc.hpp"

#define ADD_OPAQUE_VECTOR(MOD, TYPE, NAME)       \
  py::bind_vector<TYPE>(MOD, NAME, NAME);        \
  py::implicitly_convertible<py::list, TYPE>();  \
  py::implicitly_convertible<py::tuple, TYPE>(); \
  py::implicitly_convertible<py::array, TYPE>();
#define ADD_OPAQUE_MAP(MOD, TYPE, NAME) \
  py::bind_map<TYPE>(MOD, NAME, NAME);  \
  py::implicitly_convertible<py::dict, TYPE>();

namespace py = pybind11;
namespace hqc = hybridq_clifford;
namespace hqc_otoc = hybridq_clifford::otoc;

PYBIND11_MAKE_OPAQUE(hqc::BVector1D);
PYBIND11_MAKE_OPAQUE(hqc::FVector1D);
PYBIND11_MAKE_OPAQUE(hqc::FVector2D);
PYBIND11_MAKE_OPAQUE(hqc::FVector3D);
PYBIND11_MAKE_OPAQUE(hqc::IVector1D);
PYBIND11_MAKE_OPAQUE(hqc::IVector2D);
PYBIND11_MAKE_OPAQUE(hqc::IVector3D);
PYBIND11_MAKE_OPAQUE(hqc::completed_branches_type);
PYBIND11_MAKE_OPAQUE(hqc::vector_type<hqc::completed_branches_type>);
PYBIND11_MAKE_OPAQUE(hqc::vector_type<hqc::info_type>);
PYBIND11_MAKE_OPAQUE(hqc::vector_type<hqc::branch_type>);
//
PYBIND11_MAKE_OPAQUE(hqc_otoc::completed_branches_type);
PYBIND11_MAKE_OPAQUE(hqc::vector_type<hqc_otoc::completed_branches_type>);
PYBIND11_MAKE_OPAQUE(hqc::vector_type<hqc_otoc::info_type>);
// PYBIND11_MAKE_OPAQUE(hqc::vector_type<hqc_otoc::branch_type>);

PYBIND11_MODULE(hybridq_clifford_core, m) {
  m.doc() =
      "Simulation of quantum circuits using the "
      "Clifford-expansion technique.";
  //
  auto m_utils = m.def_submodule("utils", "Utilities.");
  auto m_extras = m.def_submodule("extras", "Extras.");
  auto m_otoc = m_extras.def_submodule("otoc", "OTOC");
  //
  ADD_OPAQUE_VECTOR(m, hqc::BVector1D, "BVector1D");
  ADD_OPAQUE_VECTOR(m, hqc::FVector1D, "FVector1D");
  ADD_OPAQUE_VECTOR(m, hqc::FVector2D, "FVector2D");
  ADD_OPAQUE_VECTOR(m, hqc::FVector3D, "FVector3D");
  ADD_OPAQUE_VECTOR(m, hqc::IVector1D, "IVector1D");
  ADD_OPAQUE_VECTOR(m, hqc::IVector2D, "IVector2D");
  ADD_OPAQUE_VECTOR(m, hqc::IVector3D, "IVector3D");
  ADD_OPAQUE_VECTOR(m, hqc::vector_type<hqc::branch_type>, "VectorBranch");
  ADD_OPAQUE_MAP(m, hqc::completed_branches_type, "CompletedBranches");
  ADD_OPAQUE_VECTOR(m, hqc::vector_type<hqc::completed_branches_type>,
                    "VectorCompletedBranches");
  ADD_OPAQUE_VECTOR(m, hqc::vector_type<hqc::info_type>, "VectorInfo");
  //
  // ADD_OPAQUE_VECTOR(m_otoc, hqc::vector_type<hqc_otoc::branch_type>,
  // "VectorBranch");
  ADD_OPAQUE_MAP(m_otoc, hqc_otoc::completed_branches_type,
                 "CompletedBranches");
  ADD_OPAQUE_VECTOR(m_otoc, hqc::vector_type<hqc_otoc::completed_branches_type>,
                    "VectorCompletedBranches");
  ADD_OPAQUE_VECTOR(m_otoc, hqc::vector_type<hqc_otoc::info_type>,
                    "VectorInfo");
  //
  m_utils.def("GetPauli", &hqc::GetPauli<const hqc::state_type &>,
              py::arg("state"), py::arg("pos"), py::pos_only(),
              "Return Pauli in position `pos` from `state`.");
  m_utils.def("SetPauli", &hqc::SetPauli<hqc::state_type &>, py::arg("state"),
              py::arg("pos"), py::arg("op"), py::pos_only(),
              "Set Pauli `op` in position `pos` of `state`.");
  m_utils.def("SetPauliFromChar", &hqc::SetPauliFromChar<hqc::state_type &>,
              py::arg("state"), py::arg("pos"), py::arg("op"), py::pos_only(),
              "Set Pauli `op` in position `pos` of `state`.");
  m_utils.def("StateFromPauli", &hqc::StateFromPauli<hqc::state_type>,
              py::arg("paulis"), py::pos_only(),
              "Return `State` from a Pauli string.");
  m_utils.def("PauliFromState", &hqc::PauliFromState<const hqc::state_type &>,
              py::arg("state"), py::pos_only(),
              "Return Pauli string from `State`.");
  m_utils.def(
      "DumpBranches",
      [](const hqc::vector_type<hqc::branch_type> &branches) {
        return py::bytes(hqc::DumpBranches(branches));
      },
      py::arg("branches"), py::pos_only(), "Dump branches.");
  m_utils.def(
      "LoadBranches",
      [](const std::string &buffer) {
        return hqc::LoadBranches<hqc::vector_type, hqc::branch_type>(buffer);
      },
      py::arg("buffer"), py::pos_only(), "Load branches.");
  //
  py::class_<hqc::state_type>(m, "State")
      .def(py::init<std::size_t>(), py::arg("n"), py::pos_only(),
           "Initialize state.")
      .def(py::init<const hqc::BVector1D &>(), py::arg("a"), py::pos_only(),
           "Initialize state.")
      .def("get", &hqc::state_type::get, py::arg("pos"), py::pos_only(),
           "Get bit in given position.")
      .def("set", &hqc::state_type::set, py::arg("pos"), py::arg("value"),
           py::pos_only(), "Set bit in given position.")
      .def(
          "data", [](const hqc::state_type &self) { return self.data(); },
          "Get underlying data.")
      .def("__eq__", [](const hqc::state_type &self,
                        const hqc::state_type &other) { return self == other; })
      .def("__len__",
           [](const hqc::state_type &self) { return std::size(self); })
      .def("__repr__", [](const hqc::state_type &self) {
        std::string repr_;
        repr_ = "State(";
        for (std::size_t i_ = 0, end_ = std::size(self); i_ < end_; ++i_)
          repr_ += self.get(i_) ? '1' : '0';
        repr_ += ")";
        return repr_;
      });
  //
  py::class_<hqc::branch_type>(m, "Branch")
      .def(py::init<>())
      .def(py::init<hqc::state_type, hqc::phase_type, hqc::phase_type,
                    hqc::index_type>())
      .def_readonly("state", &hqc::branch_type::state)
      .def_readonly("phase", &hqc::branch_type::phase)
      .def_readonly("norm_phase", &hqc::branch_type::norm_phase)
      .def_readonly("gate_idx", &hqc::branch_type::gate_idx)
      .def("__eq__",
           [](const hqc::branch_type &self, const hqc::branch_type &other) {
             return self == other;
           })
      .def("__repr__", [](const hqc::branch_type &self) {
        std::stringstream ss;
        self << ss;
        return "Branch" + ss.str();
      });
  //
  py::class_<hqc::info_type>(m, "Info")
      .def(py::init<>())
      .def_readonly("n_explored_branches", &hqc::info_type::n_explored_branches)
      .def_readonly("n_remaining_branches",
                    &hqc::info_type::n_remaining_branches)
      .def_readonly("n_completed_branches",
                    &hqc::info_type::n_completed_branches)
      .def_readonly("n_threads", &hqc::info_type::n_threads)
      .def_readonly("runtime_ms", &hqc::info_type::runtime_ms)
      .def_readonly("branching_time_ms", &hqc::info_type::branching_time_ms)
      .def_readonly("expanding_time_ms", &hqc::info_type::expanding_time_ms)
      .def("dict", &hqc::info_type::dict);
  //
  py::class_<hqc::Simulator>(m, "Simulator")
      .def(py::init<hqc::vector_type<hqc::phases_type>,
                    hqc::vector_type<hqc::positions_type>,
                    hqc::vector_type<hqc::qubits_type>, hqc::float_type,
                    hqc::float_type, hqc::float_type, std::size_t,
                    std::size_t>(),
           py::arg("phases"), py::arg("positions"), py::arg("qubits"),
           py::kw_only(), py::arg("atol") = hqc::float_type{1e-8},
           py::arg("norm_atol") = hqc::float_type{1e-8},
           py::arg("merge_atol") = hqc::float_type{1e-8},
           py::arg("n_threads") = 0, py::arg("log2_n_buckets") = 12)
      .def("start",
           &hqc::Simulator::start<const hqc::vector_type<hqc::branch_type> &>,
           py::arg("branches"), py::pos_only(), py::kw_only(),
           py::arg("expand_branches_only") = false)
      .def("stop", &hqc::Simulator::stop)
      .def("join", &hqc::Simulator::join)
      .def("ready", &hqc::Simulator::ready, py::arg("ms") = 0, py::pos_only())
      .def_readonly("atol", &hqc::Simulator::atol)
      .def_readonly("norm_atol", &hqc::Simulator::norm_atol)
      .def_readonly("merge_atol", &hqc::Simulator::merge_atol)
      .def_readonly("n_threads", &hqc::Simulator::n_threads)
      .def_readonly("log2_n_buckets", &hqc::Simulator::log2_n_buckets)
      .def_property_readonly("infos", &hqc::Simulator::infos)
      .def("__repr__", [](const hqc::Simulator &self) {
        std::string out_;
        out_ += "Simulator(";
        out_ += "atol=" + std::to_string(self.atol) + ", ";
        out_ += "norm_atol=" + std::to_string(self.norm_atol) + ", ";
        out_ += "merge_atol=" + std::to_string(self.merge_atol) + ", ";
        out_ += "n_threads=" + std::to_string(self.n_threads) + ", ";
        out_ += "log2_n_buckets=" + std::to_string(self.log2_n_buckets) + ", ";
        return out_;
      });
  //
  py::class_<hqc_otoc::info_type>(m_otoc, "Info")
      .def(py::init<>())
      .def_readonly("n_explored_branches",
                    &hqc_otoc::info_type::n_explored_branches)
      .def_readonly("n_remaining_branches",
                    &hqc_otoc::info_type::n_remaining_branches)
      .def_readonly("n_completed_branches",
                    &hqc_otoc::info_type::n_completed_branches)
      .def_readonly("n_threads", &hqc_otoc::info_type::n_threads)
      .def_readonly("runtime_ms", &hqc_otoc::info_type::runtime_ms)
      .def_readonly("branching_time_ms",
                    &hqc_otoc::info_type::branching_time_ms)
      .def_readonly("expanding_time_ms",
                    &hqc_otoc::info_type::expanding_time_ms)
      .def_readonly("log10_norm_atols", &hqc_otoc::info_type::log10_norm_atols)
      .def_readonly("otoc_values", &hqc_otoc::info_type::otoc_values)
      .def_readonly("otoc_values_no_int",
                    &hqc_otoc::info_type::otoc_values_no_int)
      .def_readonly("otoc_norms0", &hqc_otoc::info_type::otoc_norms0)
      .def_readonly("otoc_norms1", &hqc_otoc::info_type::otoc_norms1)
      .def_readonly("otoc_n_completed_branches",
                    &hqc_otoc::info_type::otoc_n_completed_branches)
      .def_readonly("otoc_n_explored_branches",
                    &hqc_otoc::info_type::otoc_n_explored_branches)
      .def("dict", &hqc_otoc::info_type::dict);
  //
  py::class_<hqc_otoc::Simulator>(m_otoc, "Simulator")
      .def(py::init<hqc::vector_type<hqc_otoc::phases_type>,
                    hqc::vector_type<hqc_otoc::positions_type>,
                    hqc::vector_type<hqc_otoc::qubits_type>, std::size_t,
                    hqc::IVector1D, hqc::float_type, hqc::float_type,
                    hqc::float_type, std::size_t, std::size_t>(),
           py::arg("phases"), py::arg("positions"), py::arg("qubits"),
           py::kw_only(), py::arg("target_position"), py::arg("initial_state"),
           py::arg("atol") = hqc::float_type{1e-8},
           py::arg("norm_atol") = hqc::float_type{1e-8},
           py::arg("merge_atol") = hqc::float_type{1e-8},
           py::arg("n_threads") = 0, py::arg("log2_n_buckets") = 12)
      .def("start",
           &hqc_otoc::Simulator::start<
               const hqc::vector_type<hqc_otoc::branch_type> &>,
           py::arg("branches"), py::pos_only(), py::kw_only(),
           py::arg("expand_branches_only") = false)
      .def("stop", &hqc_otoc::Simulator::stop)
      .def("join", &hqc_otoc::Simulator::join)
      .def("ready", &hqc_otoc::Simulator::ready, py::arg("ms") = 0,
           py::pos_only())
      .def_readonly("atol", &hqc_otoc::Simulator::atol)
      .def_readonly("norm_atol", &hqc_otoc::Simulator::norm_atol)
      .def_readonly("merge_atol", &hqc_otoc::Simulator::merge_atol)
      .def_readonly("n_threads", &hqc_otoc::Simulator::n_threads)
      .def_readonly("log2_n_buckets", &hqc_otoc::Simulator::log2_n_buckets)
      .def_property_readonly("infos", &hqc_otoc::Simulator::infos)
      .def("__repr__", [](const hqc_otoc::Simulator &self) {
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
