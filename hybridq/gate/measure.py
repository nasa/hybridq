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
from hybridq.gate import Projection, BaseGate
from hybridq.gate import property as pr
import numpy as np


def _Measure(a: array_like,
             axes: tuple[int, ...],
             *,
             renormalize: bool = True,
             new_a: array_like = None,
             get_probs_only: bool = False,
             get_state_only: bool = False):
    from hybridq.utils import transpose, aligned

    # Check number of axes is correct
    if len(axes) > a.ndim:
        raise ValueError("'axes' are not valid.")

    # Get size and shape
    shape = a.shape
    size = np.prod([shape[x] for x in axes])

    # Complete axes
    axes += tuple(x for x in range(len(shape)) if x not in axes)

    # Transpose and reshape
    a = np.reshape(transpose(a, axes), (size, np.prod(shape) // size))

    # Get probabilities
    probs = np.sum(np.real(a * a.conj()), axis=1)

    # Return only probs if required
    if get_probs_only:
        return probs

    # Select state
    state = np.random.choice(size, p=probs)

    # Return only state if required
    if get_state_only:
        return state

    # Create an empty state if not provided
    new_a = aligned.zeros_like(a) if new_a is None else new_a

    # Assign to new_a (renormalize if required)
    new_a[state] = a[state] / np.linalg.norm(
        a[state]) if renormalize else a[state]

    # Transpose back
    new_a = transpose(np.reshape(new_a, shape),
                      [axes.index(x) for x in range(len(axes))])

    # Return new_a
    return new_a


def _MeasureGateApply(self, psi, order, renormalize: bool = True, **kwargs):
    from hybridq.utils.dot import to_complex, to_complex_array
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

    # Set default
    kwargs.setdefault('get_probs_only', False)
    kwargs.setdefault('get_state_only', False)

    # Get axes
    axes = tuple(order.index(q) for q in self.qubits)

    # If complex_array, just merge
    if complex_array:
        psi = to_complex(psi[0], psi[1])

    # If only probabilities are required, just return them
    if kwargs['get_probs_only']:
        return _Measure(psi, axes, get_probs_only=True)

    # If only the sampled state is required, just return it
    elif kwargs['get_state_only']:
        return _Measure(psi, axes, get_state_only=True)

    # Otherwise, return state with the right probability
    else:
        # Get state
        psi = _Measure(psi, axes, renormalize=renormalize)

        # If complex_array, split in real and imaginary part
        if complex_array:
            psi = to_complex_array(psi)

        # Return state and order
        return psi, order


def Measure(qubits: iter[any] = None,
            n_qubits: int = None,
            tags: dict[any, any] = None) -> pr.BaseGate:
    """
    Generator of measurement gates.

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
    MeasureGate, FunctionalGate
    """

    # Check
    if qubits is None and n_qubits is None:
        raise ValueError("Either 'qubits' or 'n_qubits' must be specified.")

    # Convert qubits to tuple
    if qubits is not None:
        qubits = tuple(qubits)

    # Convert n_qubits to int
    if n_qubits is not None:
        n_qubits = int(n_qubits)
    else:
        n_qubits = len(qubits)

    # Check
    if n_qubits and qubits and n_qubits != len(qubits):
        raise ValueError("'qubits' has the wrong number of qubits "
                         f"(expected {n_qubits}, got {len(qubits)})")

    # Return gate
    return generate(
        'MeasureGate',
        (BaseGate, pr.FunctionalGate, pr.QubitGate, pr.TagGate, pr.NameGate),
        apply=_MeasureGateApply,
        n_qubits=n_qubits,
        name='MEASURE',
    )(qubits=qubits, tags=tags)
