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
from hybridq.gate.property import staticvars, generate
from hybridq.gate import property as pr
from hybridq.gate import BaseGate
import numpy as np


def _Projection(a: array_like,
                axes: tuple[int, ...],
                state: tuple[any, ...],
                *,
                renormalize: bool = True,
                new_a: array_like = None,
                atol: float = 1e-6):
    from hybridq.utils import aligned

    # Check number of axes is correct
    if len(axes) > a.ndim:
        raise ValueError("'axes' are not valid.")

    # Check that state and axes have the same number of dimensions
    if len(state) != len(axes):
        raise ValueError("'state' is not consistent with 'axes'.")

    # Only strings are accepted as states
    if any(x not in list('01') for x in state):
        raise ValueError(
            "Only projections to the z-basis are supported at the moment.")

    # Convert state to tuples
    state = tuple(map(int, state))

    # Get the right slices
    axes = tuple(state[axes.index(x)] if x in axes else slice(b)
                 for x, b in enumerate(a.shape))

    # Get new array if not provided
    new_a = aligned.zeros_like(a) if new_a is None else new_a

    # Compute norm
    norm = np.linalg.norm(a[axes].ravel())

    # Update new_a (if norm > atol)
    if norm > atol:
        new_a[axes] = a[axes]

    # Normalize if required
    if renormalize and norm > atol:
        new_a /= norm

    # Return projection
    return new_a


def _ProjectionGateApply(self, psi, order, renormalize: bool = True):
    from hybridq.utils import aligned

    # Check that psi is numpy array
    if not isinstance(psi, np.ndarray):
        raise ValueError("Only 'numpy.ndarray' are supported.")

    # Check dimensions
    if not 0 <= (psi.ndim - len(order)) <= 1:
        raise ValueError("'psi' is not consistent with order")

    # Check if psi is split in real and imaginary part
    complex_array = psi.ndim > len(order)

    # If complex_array, first dimension must be equal to 2
    if complex_array and (not psi.shape[0] == 2 or np.iscomplexobj(psi)):
        raise ValueError("'psi' is not valid.")

    # Get axes
    axes = tuple(order.index(q) for q in self.qubits)

    if complex_array:
        # Create a new empty array
        new_psi = aligned.zeros_like(psi)

        # Project
        _Projection(psi[0],
                    axes,
                    self.state,
                    renormalize=False,
                    new_a=new_psi[0])
        _Projection(psi[1],
                    axes,
                    self.state,
                    renormalize=False,
                    new_a=new_psi[1])

        # Renormalize if needed
        if renormalize:
            norm = np.linalg.norm(new_psi.ravel())
            if norm != 0:
                new_psi /= norm

    else:
        new_psi = _Projection(psi, axes, self.state, renormalize=renormalize)

    # Return projected
    return new_psi, order


@pr.staticvars('state')
class ProjectionGate(BaseGate):
    pass


def Projection(state: iter[any],
               qubits: iter[any] = None,
               tags: dict[any, any] = None) -> pr.BaseGate:
    """
    Generator of projectors.

    Parameters
    ----------
    state: str,
        State to project to.
    qubits: iter[any], optional
        List of qubits `Projection` is acting on.
    tags: dict[any, any], optional
        Dictionary of tags.

    See Also
    --------
    ProjectionGate, FunctionalGate
    """

    # Convert state to tuple
    state = tuple(str(state))

    # Check that state is a valid state
    if any(x not in '01' for x in state):
        raise ValueError("'state' must be a valid string of 0s and 1s.")

    # Get number of qubits
    n_qubits = len(state)

    # Check that state is consistent with n_qubits
    if len(state) != n_qubits:
        raise ValueError("'state' must be consistent with 'n_qubits'.")

    # Convert qubits to tuple
    if qubits is not None:
        qubits = tuple(qubits)

        # Check number of qubits
        if len(qubits) != n_qubits:
            raise ValueError("'qubits' has the wrong number of qubits "
                             f"(expected {n_qubits}, got {len(qubits)})")

    # Return gate
    return generate(
        'ProjectionGate', (ProjectionGate, pr.FunctionalGate, pr.QubitGate,
                           pr.TagGate, pr.NameGate),
        apply=_ProjectionGateApply,
        n_qubits=n_qubits,
        name='PROJECTION',
        state=state,
        methods=dict(__print__=lambda self:
                     {'state': (200, f"state='{''.join(self.state)}'", 0)}))(
                         qubits=qubits, tags=tags)
