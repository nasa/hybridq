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
"""

__all__ = ['layout']


def layout(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    """
    Generate a Sycamore layout for a given number of rows and columns.

    Parameters
    ----------
    n_rows: int
        Number of rows.
    n_cols: int
        Number of columns.

    Returns
    -------
    list[tuple[int, int]]
        A list of qubits.
    """

    # Get limits
    x_min = (n_cols + (n_cols % 2)) // 2 - 1
    x_max = (n_rows + (n_rows % 2)) // 2
    y_min = 0
    y_max = (n_rows + n_cols) // 2

    # Generate sycamore layout
    layout = [
        (x, y) for x in range(-x_min, x_max) for y in range(-y_min, y_max)
    ]
    layout = [(x, y) for x, y in layout if 0 <= y - x < n_cols]
    layout = [(x, y) for x, y in layout if 0 <= y + x < n_rows]

    # Shift
    layout = [(y, x + x_min) for x, y in layout]

    # Check
    assert (len(layout) == n_rows * (n_cols // 2) + (n_cols % 2) *
            ((n_rows // 2) + (n_rows % 2)))

    # Return layout
    return sorted(layout)
