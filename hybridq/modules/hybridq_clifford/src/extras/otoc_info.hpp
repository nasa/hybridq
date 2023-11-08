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

#include "../defs.hpp"
#include "../info.hpp"

namespace hybridq_clifford::otoc {

struct Info : hybridq_clifford::Info {
  std::size_t cleaning_time_ms{std::numeric_limits<std::size_t>::max()};
  FVector1D log10_norm_atols;
  FVector1D otoc_values;
  FVector1D otoc_values_no_int;
  FVector1D otoc_norms0;
  FVector1D otoc_norms1;
  IVector1D otoc_n_completed_branches;
  IVector1D otoc_n_explored_branches;

  auto dict() const {
    auto out_ = static_cast<const hybridq_clifford::Info *>(this)->dict();
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

}  // namespace hybridq_clifford::otoc

namespace hybridq_clifford::archive {

namespace hqc = hybridq_clifford;

template <>
struct Dump<hqc::otoc::Info> {
  auto operator()(const hqc::otoc::Info &info) {
    return dump(*static_cast<const hqc::Info *>(&info)) +
           dump(info.cleaning_time_ms) + dump(info.log10_norm_atols) +
           dump(info.otoc_values) + dump(info.otoc_values_no_int) +
           dump(info.otoc_norms0) + dump(info.otoc_norms1) +
           dump(info.otoc_n_completed_branches) +
           dump(info.otoc_n_explored_branches);
  }
};

template <>
struct Load<hqc::otoc::Info> {
  auto operator()(const char *buffer) {
    auto [h1_, info_] = load<hqc::Info>(buffer);
    auto [h2_, cleaning_time_] = load<std::size_t>(h1_);
    auto [h3_, log10_norm_atols_] = load<FVector1D>(h2_);
    auto [h4_, otoc_values_] = load<FVector1D>(h3_);
    auto [h5_, otoc_values_no_int_] = load<FVector1D>(h4_);
    auto [h6_, otoc_norms0_] = load<FVector1D>(h5_);
    auto [h7_, otoc_norms1_] = load<FVector1D>(h6_);
    auto [h8_, otoc_n_completed_branches_] = load<IVector1D>(h7_);
    auto [h9_, otoc_n_explored_branches_] = load<IVector1D>(h8_);
    return std::pair{h9_, hqc::otoc::Info{
                              info_,
                              cleaning_time_,
                              std::move(log10_norm_atols_),
                              std::move(otoc_values_),
                              std::move(otoc_values_no_int_),
                              std::move(otoc_norms0_),
                              std::move(otoc_norms1_),
                              std::move(otoc_n_completed_branches_),
                              std::move(otoc_n_explored_branches_),
                          }};
  }
};

}  // namespace hybridq_clifford::archive
