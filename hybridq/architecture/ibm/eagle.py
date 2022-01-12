"""
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

Types
-----
**`Qubit`**: `tuple[int, int]`

**`QpuLayout`**: `list[Qubit]`

**`Coupling`**: `tuple[Qubit, Qubit]`

Attributes
----------
drawing: str
    Drawing of the IBM Eagle QPU.

layout: QpuLayout
    Qubits available in IBM Eagle QPU.

couplings: list[Coupling]
    All couplings available in IBM Eagle QPU.
"""

from __future__ import annotations
from typing import List, Tuple, Callable
from hybridq.architecture.utils import get_layout_from_drawing as get_layout
from hybridq.utils import sort, argsort

__all__ = ['drawing', 'layout', 'couplings']

# Define Qubit type
Qubit = Tuple[int, int]

# Define Coupling
Coupling = Tuple[Qubit, Qubit]

# Define QpuLayout
QpuLayout = List[Qubit]

# Drawing of the QPU
drawing = r"""
X-X-X-X-X-X-X-X-X-X-X-X-X-X
|       |       |       |
X       X       X       X
|       |       |       |
X-X-X-X-X-X-X-X-X-X-X-X-X-X-X
    |       |       |       |
    X       X       X       X
    |       |       |       |
X-X-X-X-X-X-X-X-X-X-X-X-X-X-X
|       |       |       |
X       X       X       X
|       |       |       |
X-X-X-X-X-X-X-X-X-X-X-X-X-X-X
    |       |       |       |
    X       X       X       X
    |       |       |       |
X-X-X-X-X-X-X-X-X-X-X-X-X-X-X
|       |       |       |
X       X       X       X
|       |       |       |
X-X-X-X-X-X-X-X-X-X-X-X-X-X-X
    |       |       |       |
    X       X       X       X
    |       |       |       |
  X-X-X-X-X-X-X-X-X-X-X-X-X-X
"""

# Get couplings and layout
layout, couplings = get_layout(drawing)
