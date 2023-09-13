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

import numpy as np
import numba

__all__ = []

# I=0, X=1, Y=2, Z=3) Pauli(P1 * P2) = mat_p_[P1][P2]
mat_p_ = np.array([[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]])

# I=0, X=1, Y=2, Z=3) Phase(P1 * P2) = mat_s_[P1][P2]
mat_s_ = np.array([[1, 1, 1, 1], [1, 1, 1j, -1j], [1, -1j, 1, 1j],
                   [1, 1j, -1j, 1]])


@numba.njit
def mul_(x, y, s_x=1, s_y=1):
    """
    Multiply y *= x.
    """

    # Initialize sign
    s_ = s_x * s_y

    # For all paulis ...
    for i_, (x_, y_) in enumerate(zip(x, y)):
        # Update sign
        s_ *= mat_s_[x_, y_]

        # Update pauli
        y[i_] = mat_p_[x_, y_]

    # Return updated paulis and new sign
    return y, s_


@numba.njit
def remove_xy_(stab, phases, i):
    """
    Remove all X and Y paulis from stabilizer for a given qubit in position `i`.
    """

    # Find locations of Xs and Ys
    pos_ = np.where(stab[:, i] % 3)[0]

    # If at least one X or Y is present, multiply it for all the remaining
    for p_ in pos_[1:]:
        _, phases[p_] = mul_(stab[pos_[0]], stab[p_], phases[pos_[0]],
                             phases[p_])

    # Move X (Y) to last position and remove it
    if pos_.size:
        stab[pos_[0]] = stab[-1]
        stab = stab[:-1]
        phases[pos_[0]] = phases[-1]
        phases = phases[:-1]

    # Return stab
    return stab, phases


@numba.njit
def remove_xyz_(stab, phases, i):
    """
    Remove all X, Y and Z paulis from stabilizer for a given qubit in position `i`.
    """

    # Remove Xs and Ys
    stab, phases = remove_xy_(stab, phases, i)

    # Find locations of Zs
    pos_ = np.where(stab[:, i] == 3)[0]

    # If at least one Z, multiply it for all the remaining
    for p_ in pos_[1:]:
        _, phases[p_] = mul_(stab[pos_[0]], stab[p_], phases[pos_[0]],
                             phases[p_])

    # Move Z to last position and remove it
    if pos_.size:
        stab[pos_[0]] = stab[-1]
        stab = stab[:-1]
        phases[pos_[0]] = phases[-1]
        phases = phases[:-1]

    # Return stab
    return stab, phases


@numba.njit
def diag_z_(stab, phases):
    """
    Reduce the stabilizer by removing all X and Y paulis.
    """

    for i_ in range(stab.shape[1]):
        stab, phases = remove_xy_(stab, phases, i_)
    return stab, phases


@numba.njit
def trace_(stab, phases, qubits):
    """
    Reduce the stabilizer by removing all X, Y and Z paulis.
    """

    for i_ in qubits:
        stab, phases = remove_xyz_(stab, phases, i_)
    return stab, phases
