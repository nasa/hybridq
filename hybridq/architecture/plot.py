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
"""

from __future__ import annotations
from matplotlib import pyplot as plt
from typing import List, Tuple

__all__ = ['plot_qubits']

# Define Qubit type
Qubit = Tuple[int, int]

# Define Coupling
Coupling = Tuple[Qubit, Qubit]

# Define QpuLayout
QpuLayout = List[Qubit]


def plot_qubits(qpu_layout: QpuLayout,
                layout: list[Coupling] = None,
                subset: list[Qubit] = None,
                selection: list[Qubit] = None,
                scale: float = 6,
                figsize: tuple[int, int] = None,
                draw_border: bool = False,
                title: str = None) -> matplotlib.figure.Figure:
    """
    Plot qubits for 2D architectures.

    Parameters
    ----------
    qpu_layout: QpuLayout
        List of qubits, with each qubits represented by the 2D coordinate, (x, y).
    layout: list[Coupling], optional
        List of couplings between qubits.
    subset: list[Qubit], optional
        Qubits in `subset` are highlated.
    selection: list[Qubit], optional
        Box are added to qubits in `selection`.
    figsize: tuple[int, int], optional
        Size of figure.
    draw_border: bool, optional
        Draw border around qubits in `qpu_layout` (default: False).
    title: str, optional
        Add title to plot (default: "").

    Example
    -------
    >>> from hybridq.architecture import sycamore
    >>> qpu_layout = sycamore.gmon54
    >>> layer = sycamore.get_layers(qpu_layout=qpu_layout)['A']
    >>> plot_qubits(qpu_layout, layout=layer)

    .. image:: ../../images/qpu_layout_plot.png
    """

    # Initialize
    if layout is None:
        layout = []
    if subset is None:
        subset = []
    if selection is None:
        selection = []

    # Get ratio
    _ratio = (max(y for _, y in qpu_layout) - min(y for _, y in qpu_layout)) / (
        max(x for x, _ in qpu_layout) - min(x for x, _ in qpu_layout))

    # Set figsize
    fig = plt.figure(figsize=(scale,
                              scale * _ratio) if figsize is None else figsize)

    # Plot couplings in layout
    for (x1, y1), (x2, y2) in layout:
        c = 'tab:red' if ((x1, y1) in subset) ^ (
            (x2, y2) in subset) else 'tab:green'
        plt.plot([x1, x2], [y1, y2], lw=5, c=c)

    # Plot qubits
    for x, y in qpu_layout:
        c = 'tab:orange' if (x, y) in subset else 'tab:blue'
        plt.plot([x], [y], 'o', mfc='white', ms=15, mew=3, c=c)

    # Plot qubits in selection
    for x, y in selection:
        plt.plot([x], [y], 's', mfc='none', ms=25, mew=3, c='tab:purple')

    # Set limits of the plot
    plt.xlim(plt.xlim()[0] - 0.5, plt.xlim()[1] + 0.5)
    plt.ylim(plt.ylim()[0] - 0.5, plt.ylim()[1] + 0.5)

    # Plot grid
    plt.grid(linestyle=':')

    # Plot ticks
    plt.xticks(
        range(min([x for x, _ in qpu_layout]),
              max([x + 1 for x, _ in qpu_layout])))
    plt.yticks(
        range(min([y for _, y in qpu_layout]),
              max([y + 1 for _, y in qpu_layout])))

    # Print title
    if title:
        plt.title(title)

    # Draw border of chip
    if draw_border:
        for i in range(10):
            for j in range(10):
                if (i, j) in qpu_layout:
                    if (i - 1, j) not in qpu_layout:
                        plt.plot([i - 0.5, i - 0.5], [j - 0.5, j + 0.5],
                                 color='#434343',
                                 ls='-',
                                 lw=1.5)
                    if (i + 1, j) not in qpu_layout:
                        plt.plot([i + 0.5, i + 0.5], [j - 0.5, j + 0.5],
                                 color='#434343',
                                 ls='-',
                                 lw=1.5)
                    if (i, j - 1) not in qpu_layout:
                        plt.plot([i - 0.5, i + 0.5], [j - 0.5, j - 0.5],
                                 color='#434343',
                                 ls='-',
                                 lw=1.5)
                    if (i, j + 1) not in qpu_layout:
                        plt.plot([i - 0.5, i + 0.5], [j + 0.5, j + 0.5],
                                 color='#434343',
                                 ls='-',
                                 lw=1.5)

    # Close plot
    plt.close()

    # Return figure
    return fig
