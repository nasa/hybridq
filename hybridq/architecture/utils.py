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

__all__ = ['get_layout_from_drawing']

# Define Qubit type
Qubit = Tuple[int, int]

# Define Coupling
Coupling = Tuple[Qubit, Qubit]

# Define QpuLayout
QpuLayout = List[Qubit]


def get_layout_from_drawing(drawing: str) -> tuple[list[Qubit], list[Coupling]]:
    """
    Given a valid `drawing`, return the corresponding qubits and couplings. A
    valid `drawing` is a string with `X` representing a qubit and one of the
    following token `/\|-` to represent a coupling.

    Parameters
    ----------
    drawing: str
        A valid layout.

    Returns
    -------
    tuple[list[Qubit], list[Coupling]]
        Qubits and couplings representing the layout.

    Example
    -------
    # Define layout
    layout = r\"\"\".
      X-X
     /  |
    X   X
    |   |
    X-X-X
    \"\"\"

    # Get qubits and couplings
    get_layout_from_drawing(layout)
    > ([(0, 0), (0, 1), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)],
    >  [((0, 0), (1, 0)),
    >   ((0, 1), (0, 0)),
    >   ((1, 0), (2, 0)),
    >   ((1, 2), (0, 1)),
    >   ((1, 2), (2, 2)),
    >   ((2, 1), (2, 0)),
    >   ((2, 2), (2, 1))])
    """
    # Layout must be a valid string
    if not isinstance(drawing, str):
        raise ValueError("'drawing' must be a valid string")

    # Split drawing and remove empty rows
    drawing = [x for x in drawing.upper().split('\n') if x]

    # Trim left
    drawing = [
        l[min(next(x
                   for x, c in enumerate(l)
                   if c != ' ')
              for l in drawing):]
        for l in drawing
    ]

    # Layout must contain only X to indicate a qubit and either /, \ or | to indicate a coupling.
    if any(set(l).difference(r'X-|/\ ') for l in drawing):
        raise ValueError("'drawing' must be a valid layout")

    # Get qubits locations
    qubits = sorted((x, y)
                    for y, l in enumerate(drawing)
                    for x, q in enumerate(l)
                    if q == 'X')

    # Given coupling, get qubits in coupling
    def _get_qubits(c, x, y):
        if c == '-':
            return ((x - 1, y), (x + 1, y))
        elif c == '|':
            return ((x, y - 1), (x, y + 1))
        elif c == '\\':
            return ((x - 1, y - 1), (x + 1, y + 1))
        elif c == '/':
            return ((x + 1, y - 1), (x - 1, y + 1))
        else:
            raise ValueError(f"'{c}' is not supported")

    # Check all couplings are valid
    if not all(
            all(q in qubits
                for q in _get_qubits(c, x, y))
            for y, l in enumerate(drawing)
            for x, c in enumerate(l)
            if c in r'/\|-'):
        raise ValueError("'drawing' has not valid couplings")

    # Get all couplings
    couplings = sorted(
        _get_qubits(c, x, y)
        for y, l in enumerate(drawing)
        for x, c in enumerate(l)
        if c in r'/\|-')

    # Find common denominator of qubits indexes
    from numpy import gcd
    _gcd = gcd.reduce([x for q in qubits for x in q])

    # If gcd is different from 1, rescale everything
    if _gcd > 1:
        qubits = [(x // _gcd, y // _gcd) for x, y in qubits]
        couplings = [((x1 // _gcd, y1 // _gcd), (x2 // _gcd, y2 // _gcd))
                     for (x1, y1), (x2, y2) in couplings]

    # Reverse y
    _sy = max(y for _, y in qubits)
    qubits = sorted((x, _sy - y) for x, y in qubits)
    couplings = sorted(
        tuple(sorted(((x1, _sy - y1), (x2, _sy - y2))))
        for (x1, y1), (x2, y2) in couplings)

    # Return layout
    return qubits, couplings
