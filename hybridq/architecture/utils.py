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
"""

from __future__ import annotations
from typing import List, Tuple, Callable

__all__ = ['xy_to_index', 'index_to_xy', 'get_all_couplings']

# Define Qubit type
Qubit = Tuple[int, int]

# Define Coupling
Coupling = Tuple[Qubit, Qubit]

# Define QpuLayout
QpuLayout = List[Qubit]


# Map from (x,y) -> idx (qpu_layout order is preserved)
def xy_to_index(qpu_layout: QpuLayout) -> dict[Qubit, int]:
    """
    Given `qpu_layout` of `Qubit`s, return a one-to-one between `Qubit`s and
    indexes.

    Parameters
    ----------
    qpu_layout: QpuLayout
        List of `Qubit`s to use as `QpuLayout`.

    Returns
    -------
    dict[Qubit, int]
        One-to-one map between `Qubit`s and indexes.

    Note
    ----
    ```xy_to_index``` and ```index_to_xy``` are by construction one the inverse
    of the other.
    """

    return {q: index for index, q in enumerate(qpu_layout)}


# Map from idx -> (x,y) (qpu_layout order is preserved)
def index_to_xy(qpu_layout: QpuLayout) -> dict[int, Qubit]:
    """
    Given `qpu_layout` of `Qubit`s, return a one-to-one between indexes and
    `Qubit`s.

    Parameters
    ----------
    qpu_layout: QpuLayout
        List of `Qubit`s to use as `QpuLayout`.

    Returns
    -------
    dict[int, Qubit]
        One-to-one map between indexes and `Qubit`s.

    Note
    ----
    ```xy_to_index``` and ```index_to_xy``` are by construction one the inverse
    of the other.
    """
    return {index: q for q, index in xy_to_index(qpu_layout).items()}


def get_all_couplings(qpu_layout: QpuLayout) -> list[Coupling]:
    """
    Given `qpu_layout` of `Qubit`s, return all couplings between nearest
    neighbors.

    Parameters
    ----------
    qpu_layout: QpuLayout
        List of `Qubit`s to use as `QpuLayout`.

    Returns
    -------
    list[Coupling]
        List of all possible couplings between nearest neighbor `Qubit`s.

    Example
    -------
    >>> get_all_couplings(qpu_layout=((0, 0), (0, 1), (1, 0), (1, 1)))
    [((0, 0), (0, 1)), ((0, 0), (1, 0)), ((0, 1), (1, 1)), ((1, 0), (1, 1))]
    """
    from hybridq.utils import sort

    return sort({
        tuple(sort(((x1, y1), (x2, y2))))
        for x1, y1 in qpu_layout
        for x2, y2 in qpu_layout
        if x1 == x2 and abs(y1 - y2) == 1 or y1 == y2 and abs(x1 - x2) == 1
    })
