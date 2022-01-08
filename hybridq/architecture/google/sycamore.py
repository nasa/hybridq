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
layout: QpuLayout
    Qubits available in Google Sycamore QPU.

get_layers: callable[QpuLayout] -> dict[str, list[Coupling]]
    Given a `QpuLayout` returns layers of couplings as defined in
    "Quantum supremacy using a programmable superconducting processor"
    [Nature 574 (7779), 505-510].
"""

from __future__ import annotations
from typing import List, Tuple, Callable
from hybridq.utils import sort, argsort

__all__ = ['layout', 'gmon54', 'get_layers']

# Define Qubit type
Qubit = Tuple[int, int]

# Define Coupling
Coupling = Tuple[Qubit, Qubit]

# Define QpuLayout
QpuLayout = List[Qubit]

# (x,y) layout of Google Sycamore QPU
layout = [
    (0, 5), (0, 6), (1, 4), (1, 5), (1, 6), (1, 7), (2, 4), (2, 5), (2, 6),
    (2, 7), (2, 8), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
    (3, 9), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8),
    (4, 9), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7),
    (5, 8), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (7, 2),
    (7, 3), (7, 4), (7, 5), (7, 6), (8, 3), (8, 4), (8, 5), (9, 4)
]

# For consistency with previous HybridQ versions
gmon54 = layout


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
    >>> from hybridq.architecture.google.sycamore import layout as gmon54, get_layers
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
    from hybridq.architecture.utils import get_all_couplings

    # Get all couplings
    all_couplings = get_all_couplings(qpu_layout)

    return {
        l: list(filter(f(x), all_couplings))
        for l, f, x in zip('ABCDEFGH', [_in_supremacy_layout] * 4 +
                           [_in_simplifiable_layout] *
                           4, [1, 2, 3, 0, 0, 2, 1, 3])
    }
