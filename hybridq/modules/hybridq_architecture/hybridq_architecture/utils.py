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
import more_itertools as mit
import itertools as its

__all__ = ['Layer', 'all_couplings', 'plot_qubits']


class Layer:
    """
    Filter qubits to build layers.
    """

    def __init__(self, col_offset: 0 | 1, vertical: bool, stagger: bool):
        self._col_offset = col_offset
        self._vertical = vertical
        self._stagger = stagger

    def __str__(self) -> str:
        return (f'{type(self).__qualname__}'
                f'(col_offset={self._col_offset}, '
                f'vertical={self._vertical}, '
                f'stagger={self._stagger})')

    def __repr__(self) -> str:
        return str(self)

    def __contains__(self, q):
        # Filter vertical/horizontal
        cv_ = lambda q: q[0][0] == q[1][0] if not self._vertical else q[0][
            1] == q[1][1]

        # Filter stagger
        cs_ = lambda q: ((q[0][0] + q[0][1]) % 2 if self._stagger else q[0][
            not self._vertical] % 2) == self._col_offset

        # Filter
        return cv_(q) and cs_(q)

    def __call__(self, q):
        return q in self


def all_couplings(
    layout: list[tuple[int,
                       int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Given `layout`, return all available couplings. Two qubits have a coupling
    if they are first neighbors.
    """

    # Define neighbors
    def _nn(q, w):
        return q < w and ((q[0] == w[0] and abs(q[1] - w[1]) == 1) or
                          (q[1] == w[1] and abs(q[0] - w[0]) == 1))

    # Two qubits share a coupling if either x or y are the same
    return sorted(filter(lambda x: _nn(*x), its.product(layout, repeat=2)))


def plot_qubits(layout,
                couplings=(),
                missing_qubits=(),
                subset=(),
                ms=12,
                lw=5,
                **plt_args):
    """
    Plot `layout`.
    """
    if len(layout):
        layout = list(map(lambda x, y: (y, x), *mit.transpose(layout)))
    if len(subset):
        subset = list(map(lambda x, y: (y, x), *mit.transpose(subset)))
    if len(missing_qubits):
        missing_qubits = list(
            map(lambda x, y: (y, x), *mit.transpose(missing_qubits)))
    if len(couplings):
        couplings = list(
            map(lambda qs: ((qs[0][1], qs[0][0]), (qs[1][1], qs[1][0])),
                couplings))

    # Initialize figure
    fig = plt.figure(**plt_args).set_facecolor('#00000000')
    plt.gca().set_facecolor('#00000000')

    # Plot couplings
    for c in couplings:
        _color = 'tab:red' if (c[0] in subset) ^ (c[1]
                                                  in subset) else 'tab:green'
        plt.plot(*zip(*c), '-', color=_color, lw=lw, alpha=0.8)

    # Plot all qubits not in subset
    plt.plot(*zip(*filter(lambda q: q not in subset, layout)),
             'o',
             c='tab:blue',
             mfc='white',
             ms=ms,
             mew=3)

    # Plot qubits in subset
    if len(subset):
        plt.plot(*zip(*subset), 'o', c='tab:orange', mfc='white', ms=ms, mew=3)

    # Plot missing qubits
    if len(missing_qubits):
        plt.plot(*zip(*missing_qubits),
                 'x',
                 c='tab:red',
                 mfc='white',
                 ms=ms,
                 mew=3)

    # Add grid
    plt.grid(ls=':')

    # Get min/max
    x_min = min(*(x for x, _ in layout + missing_qubits))
    x_max = max(*(x for x, _ in layout + missing_qubits))
    y_min = min(*(y for _, y in layout + missing_qubits))
    y_max = max(*(y for _, y in layout + missing_qubits))

    # Fix ticks
    plt.xticks(range(x_min, x_max + 1))
    plt.yticks(range(y_min, y_max + 1))

    # Invert y-axis
    plt.gca().invert_yaxis()

    # Return figure
    return plt
