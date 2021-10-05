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
from hybridq.dm.gate import MatrixSuperGate


def to_matrix_supergate(gate: SuperGate, copy: bool = True):
    """
    Convert `gate` to `MatrixSuperGate`.

    Parameters
    ----------
    gate: SuperGate
        SuperGate to convert.m
    copy: bool, optional
        A copy of `Map` is used instead of a reference if `copy` is True
        (default: True).

    Returns
    -------
    MatrixSuperGate
    """

    # Get left/right qubits
    l_qubits, r_qubits = gate.qubits

    return MatrixSuperGate(Map=gate.Matrix,
                           l_qubits=l_qubits,
                           r_qubits=r_qubits,
                           tags=gate.tags,
                           copy=copy)
