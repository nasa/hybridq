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
        if any(isinstance(gate, t) for t in (BaseGate, BaseSuperGate)):
            return gate
        else:
            raise ValueError(f"'type(gate).__name__' not supported.")

    def __init__(self, gates: iter[Gate] = tuple(), *args, **kwargs) -> None:
        super().__init__(gates=map(self.__check_gate__, gates), **kwargs)

    def all_qubits(
            self,
            *,
            ignore_missing_qubits: bool = False) -> tuple[list[any], list[any]]:
        from hybridq.utils import sort

        # Check if there are virtual gates with no qubits
        if not ignore_missing_qubits and any(
                not gate.provides('qubits') or gate.qubits is None
                for gate in self
                if gate.n_qubits):
            raise ValueError("Circuit contains virtual gates with no qubits.")

        # Initialize qubits
        l_qubits = []
        r_qubits = []

        # For all gates ...
        for gate in self:
            if gate.provides('qubits') and gate.qubits:
                qubits = gate.qubits
                if isinstance(gate, BaseGate):
                    l_qubits += qubits
                    r_qubits += qubits
                elif isinstance(gate, BaseSuperGate):
                    l_qubits += qubits[0]
                    r_qubits += qubits[1]
                else:
                    raise ValueError(f"{type(gate).__name__} is not supported")

        # Sort qubits
        l_qubits = sort(set(l_qubits))
        r_qubits = sort(set(r_qubits))

        # Return qubits
        return (l_qubits, r_qubits)
