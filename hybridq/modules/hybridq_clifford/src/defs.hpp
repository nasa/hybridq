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

namespace hybridq_clifford {

using float_type = float;
using index_type = std::size_t;
template <typename... T>
using vector_type = std::vector<T...>;
template <typename... T>
using list_type = std::list<T...>;
template <typename... T>
using map_type = std::unordered_map<T...>;
using BVector1D = vector_type<bool>;
using FVector1D = vector_type<float_type>;
using FVector2D = vector_type<FVector1D>;
using FVector3D = vector_type<FVector2D>;
using IVector1D = vector_type<index_type>;
using IVector2D = vector_type<IVector1D>;
using IVector3D = vector_type<IVector2D>;

}  // namespace hybridq_clifford
