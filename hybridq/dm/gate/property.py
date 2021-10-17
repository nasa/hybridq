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

from __future__ import annotations
from hybridq.base import generate, staticvars, compare, requires, __Base__
from hybridq.base.property import Tuple
import numpy as np


class BaseTupleSuperGate(Tuple):
    """
    Gate defined as a tuple of gates.
    """

    @property
    def qubits(self) -> tuple[tuple[any, ...], tuple[any, ...]]:
        from hybridq.gate import BaseGate

        # Define flatten
        def _unique_flatten(l):
            from hybridq.utils import sort
            return tuple(sort(set(y for x in l for y in x)))

        # Get qubits
        def _get_qubits(gate):
            if isinstance(gate, BaseGate):
                # Get qubits
                qubits = gate.qubits
                return (qubits, qubits)
            else:
                return gate.qubits

        # Get all qubits
        _qubits = tuple(
            _get_qubits(g) if g.provides('qubits') else (None, None)
            for g in self)

        # Split in left and right qubits
        try:
            _lq, _rq = map(tuple, zip(*_qubits))
        except ValueError:
            return tuple(), tuple()

        # If any none is present, set to None
        _lq = None if any(q is None for q in _lq) else _unique_flatten(_lq)
        _rq = None if any(q is None for q in _rq) else _unique_flatten(_rq)

        # Return qubits
        return (_lq, _rq)

    @property
    def n_qubits(self) -> int:
        # Get left and right qubits
        _lq, _rq = self.qubits
        return None if _lq is None else len(_lq), None if _rq is None else len(
            _rq)


@requires('qubits,Matrix')
class Map(__Base__):

    def map(self, order: iter[any] = None):
        """
        Return map.

        Parameters
        ----------
        order: tuple[any, ...], optional
            If provided, Kraus' map is ordered accordingly to `order`.
        """
        # Get left and right qubits
        l_qubits, r_qubits = self.qubits

        # Get order
        if order is not None:
            from hybridq.utils import sort

            # Get order
            order = tuple(order)

            try:
                # Split
                l_order, r_order = order

                # Convert to tuples
                l_order = tuple(l_order)
                r_order = tuple(r_order)

                # Check that qubits are consistent
                if sort(l_order) != sort(l_qubits) or sort(r_order) != sort(
                        r_qubits):
                    raise RuntimeError("Something went wrong.")

                # Get order
                order = (l_order, r_order)
            except:
                if l_qubits != r_qubits or sort(order) != sort(l_qubits):
                    raise ValueError(
                        "'order' is not a valid permutation of qubits.")

                # Get order
                order = (order, order)

        # Get Matrix representing the map
        _U = self.Matrix

        # Transpose if order is provided
        if order is not None and (order[0] != self.l_qubits or
                                  order[1] != self.r_qubits):
            from hybridq.gate import MatrixGate

            # Get gate
            _g = MatrixGate(_U,
                            qubits=tuple((0, q) for q in l_qubits) + tuple(
                                (1, q) for q in r_qubits),
                            copy=False)

            # Transpose
            _U = _g.matrix(order=tuple((0, q) for q in order[0]) + tuple(
                (1, q) for q in order[1]))

        # Return map
        return _U

    def isclose(self, gate: Map, atol: float = 1e-8):

        # If gate is not a SuperGate or they qubits differ, they are different
        if not isinstance(gate, Map) or self.qubits != gate.qubits:
            return False

        # Get matrices
        _U1 = self.map(order=self.qubits)
        _U2 = gate.map(order=self.qubits)

        # Check
        return np.allclose(_U1, _U2, atol=atol)

    def commutes_with(self, gate: Map, atol: float = 1e-7):
        from hybridq.gate import MatrixGate

        # Gate must be a map
        if not isinstance(gate, Map):
            raise ValueError(
                f"Cannot compute commutation between 'Map' and '{type(gate).__name__}'"
            )

        # Get gates
        g1 = MatrixGate(self.Matrix,
                        qubits=[(0, q) for q in self.l_qubits] +
                        [(1, q) for q in self.r_qubits],
                        copy=False)
        g2 = MatrixGate(gate.Matrix,
                        qubits=[(0, q) for q in gate.l_qubits] +
                        [(1, q) for q in gate.r_qubits],
                        copy=False)

        # Return if they commute
        return g1.commutes_with(g2, atol=atol)
