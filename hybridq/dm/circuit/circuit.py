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
from hybridq.gate import BaseGate
from hybridq.dm.gate import BaseSuperGate
from hybridq.circuit import BaseCircuit


class Circuit(BaseCircuit):

    @staticmethod
    def __check_gate__(gate: Gate):
        from hybridq.dm.gate import TupleSuperGate
        from hybridq.base.property import Tuple
        if isinstance(gate, tuple) or isinstance(gate, Tuple):
            return TupleSuperGate(map(Circuit.__check_gate__, gate))
        elif any(isinstance(gate, t) for t in (BaseGate, BaseSuperGate)):
            return gate
        else:
            raise ValueError(f"'type(gate).__name__' not supported.")

    def __init__(self, gates: iter[Gate] = tuple(), *args, **kwargs) -> None:
        super().__init__(gates=map(self.__check_gate__, gates), **kwargs)

    def all_qubits(
            self,
            *,
            ignore_missing_qubits: bool = False) -> tuple[list[any], list[any]]:
        """
        Get all qubits in `Circuit`. It raises a `ValueError` if qubits in
        `Gate` are missing, unless `ignore_missing_qubits` is `True`. The
        returned qubits are always sorted using `hybridq.utils.sort` for
        consistency.

        Parameters
        ----------
        ignore_missing_qubits: bool, optional
            If `True`, ignore gates without specified qubits. Otherwise, raise
            `ValueError`.

        Returns
        -------
        tuple[list[any], list[any]]
            Sorted list of all qubits in `Circuit`.
        """
        # If Circuit has not qubits, return empty list
        if not len(self):
            return ([], [])

        # Define flatten
        def _unique_flatten(l):
            from hybridq.utils import sort
            return sort(set(y for x in l for y in x))

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

        # Check if there are virtual gates with no qubits
        if not ignore_missing_qubits and any(
                q1 is None or q2 is None for q1, q2 in zip(_lq, _rq)):
            raise ValueError("Circuit contains virtual gates with no qubits.")

        # Flatten qubits and remove None's
        _lq = _unique_flatten(q for q in _lq if q is not None)
        _rq = _unique_flatten(q for q in _rq if q is not None)

        # Return qubits
        return (_lq, _rq)
