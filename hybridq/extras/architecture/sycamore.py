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
gmon54: QpuLayout
    Qubits available in Google Sycamore QPU.
"""

from __future__ import annotations
from typing import List, Tuple, Callable
from hybridq.utils import sort, argsort

# Define Qubit type
Qubit = Tuple[int, int]

# Define Coupling
Coupling = Tuple[Qubit, Qubit]

# Define QpuLayout
QpuLayout = List[Qubit]

# (x,y) layout of Google Sycamore QPU
gmon54 = [
    (0, 5), (0, 6), (1, 4), (1, 5), (1, 6), (1, 7), (2, 4), (2, 5), (2, 6),
    (2, 7), (2, 8), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
    (3, 9), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8),
    (4, 9), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7),
    (5, 8), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (7, 2),
    (7, 3), (7, 4), (7, 5), (7, 6), (8, 3), (8, 4), (8, 5), (9, 4)
]


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

    Example
    -------
    >>> sum(q != sycamore.index_to_xy(qpu_layout=sycamore.gmon54)[x]
    >>>     for q, x in sycamore.xy_to_index(qpu_layout=sycamore.gmon54).items())
    0
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

    Example
    -------
    >>> sum(q != sycamore.index_to_xy(qpu_layout=sycamore.gmon54)[x]
    >>>     for q, x in sycamore.xy_to_index(qpu_layout=sycamore.gmon54).items())
    0
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

    return sort({
        tuple(sort(((x1, y1), (x2, y2))))
        for x1, y1 in qpu_layout
        for x2, y2 in qpu_layout
        if x1 == x2 and abs(y1 - y2) == 1 or y1 == y2 and abs(x1 - x2) == 1
    })


def _in_simplifiable_layout(layout_idx: int) -> Callable[[Coupling], bool]:
    """
    Return if (q1,q2) qubit is in simplifiable layout.
    """

    if layout_idx == 0:
        return lambda q: (not q[0][1] % 2) and q[0][0] == q[1][0]
    if layout_idx == 1:
        return lambda q: (not q[0][0] % 2) and q[0][1] == q[1][1]
    if layout_idx == 2:
        return lambda q: (q[0][1] % 2) and q[0][0] == q[1][0]
    if layout_idx == 3:
        return lambda q: (q[0][0] % 2) and q[0][1] == q[1][1]


def _in_supremacy_layout(layout_idx: int) -> Callable[[Coupling], bool]:
    """
    Return if (q1,q2) qubit is in supremacy (non-simplifiable) layout.
    """

    if layout_idx == 0:
        return lambda q: (not (q[0][0] + q[0][1]) % 2) and q[0][0] == q[1][0]
    if layout_idx == 1:
        return lambda q: (not (q[0][0] + q[0][1]) % 2) and q[0][1] == q[1][1]
    if layout_idx == 2:
        return lambda q: ((q[0][0] + q[0][1]) % 2) and q[0][1] == q[1][1]
    if layout_idx == 3:
        return lambda q: ((q[0][0] + q[0][1]) % 2) and q[0][0] == q[1][0]


def get_layers(qpu_layout: list[Qubit]) -> dict[str, list[Coupling]]:
    """
    Return layers used in Google Quantum Supremacy Paper [Nature 574 (7779), 505-510].

    Parameters
    ----------
    qpu_layout: QpuLayout
        List of `Qubit`s to use as `QpuLayout`.

    Returns
    -------
    dict[str, list[Coupling]]
        Map between layer name and the list of corresponding `Coupling`s.

    Example
    -------
    >>> from hybridq.architecture.plot import plot_qubits
    >>> qpu_layout = [(x, y) for x, y in gmon54 if x + y < 10]
    >>> layers = get_layers(qpu_layout=qpu_layout)
    >>> layers.keys()
    dict_keys(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
    >>> layers['A']
    [((0, 6), (1, 6)),
     ((1, 5), (2, 5)),
     ((1, 7), (2, 7)),
     ((2, 4), (3, 4)),
     ((2, 6), (3, 6)),
     ((3, 3), (4, 3)),
     ((3, 5), (4, 5)),
     ((4, 2), (5, 2)),
     ((4, 4), (5, 4)),
     ((5, 1), (6, 1)),
     ((5, 3), (6, 3)),
     ((6, 2), (7, 2))]
    >>> plot_qubits(qpu_layout=qpu_layout, layout=layers['A'])

    .. image:: ../../images/qpu_layout_plot_small.png
    """

    layer_A = list(
        filter(_in_supremacy_layout(1), get_all_couplings(qpu_layout)))
    layer_B = list(
        filter(_in_supremacy_layout(2), get_all_couplings(qpu_layout)))
    layer_C = list(
        filter(_in_supremacy_layout(3), get_all_couplings(qpu_layout)))
    layer_D = list(
        filter(_in_supremacy_layout(0), get_all_couplings(qpu_layout)))
    layer_E = list(
        filter(_in_simplifiable_layout(0), get_all_couplings(qpu_layout)))
    layer_F = list(
        filter(_in_simplifiable_layout(2), get_all_couplings(qpu_layout)))
    layer_G = list(
        filter(_in_simplifiable_layout(1), get_all_couplings(qpu_layout)))
    layer_H = list(
        filter(_in_simplifiable_layout(3), get_all_couplings(qpu_layout)))

    return {
        'A': layer_A,
        'B': layer_B,
        'C': layer_C,
        'D': layer_D,
        'E': layer_E,
        'F': layer_F,
        'G': layer_G,
        'H': layer_H,
    }
