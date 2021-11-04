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
from hybridq.gate import Gate
import numpy as np


def get_available_gates() -> tuple[str, ...]:
    """
    Return available gates.
    """
    from hybridq.gate.gate import _available_gates
    return tuple(_available_gates)


def get_clifford_gates() -> tuple[str, ...]:
    """
    Return available Clifford gates.
    """
    from hybridq.gate.gate import _available_gates
    from hybridq.gate.property import CliffordGate
    return tuple(
        k for k, v in _available_gates.items() if CliffordGate in v['mro'])


def merge(a: Gate, *bs) -> Gate:
    """
    Merge two gates `a` and `b`. The merged `Gate` will be equivalent to apply
    ```
    new_psi = bs.matrix() @ ... @ b.matrix() @ a.matrix() @ psi
    ```
    with `psi` a quantum state.

    Parameters
    ----------
    a, ...: Gate
        `Gate`s to merge.
    qubits_order: iter[any], optional
        If provided, qubits in new `Gate` will be sorted using `qubits_order`.

    Returns
    -------
    Gate('MATRIX')
        The merged `Gate`
    """
    # If no other gates are provided, return
    if len(bs) == 0:
        return a

    # Pop first gate
    b, bs = bs[0], bs[1:]

    # Check
    if any(not x.provides(['matrix', 'qubits']) or x.qubits is None
           for x in [a, b]):
        raise ValueError(
            "Both 'a' and 'b' must provides 'qubits' and 'matrix'.")

    # Get unitaries
    Ua, Ub = a.matrix(), b.matrix()

    # Get shared qubits
    shared_qubits = set(a.qubits).intersection(b.qubits)
    all_qubits = b.qubits + tuple(q for q in a.qubits if q not in b.qubits)

    # Get sizes
    n_a = len(a.qubits)
    n_b = len(b.qubits)
    n_ab = len(shared_qubits)
    n_c = len(all_qubits)

    if shared_qubits:
        from opt_einsum import get_symbol, contract
        # Build map
        _map_b_l = ''.join(get_symbol(x) for x in range(n_b))
        _map_b_r = ''.join(get_symbol(x + n_b) for x in range(n_b))
        _map_a_l = ''.join(_map_b_r[b.qubits.index(q)] if q in
                           shared_qubits else get_symbol(x + 2 * n_b)
                           for x, q in enumerate(a.qubits))
        _map_a_r = ''.join(get_symbol(x + 2 * n_b + n_a) for x in range(n_a))
        _map_c_l = ''.join(_map_b_l[b.qubits.index(q)] if q in
                           b.qubits else _map_a_l[a.qubits.index(q)]
                           for q in all_qubits)
        _map_c_r = ''.join(
            _map_b_r[b.qubits.index(q)] if q in b.qubits and
            q not in shared_qubits else _map_a_r[a.qubits.index(q)]
            for q in all_qubits)
        _map = _map_b_l + _map_b_r + ',' + _map_a_l + _map_a_r + '->' + _map_c_l + _map_c_r

        # Get matrix
        U = np.reshape(
            contract(_map, np.reshape(Ub, (2,) * 2 * n_b),
                     np.reshape(Ua, (2,) * 2 * n_a)), (2**n_c, 2**n_c))
    else:
        # Get matrix
        U = np.kron(Ub, Ua)

    # Get merged gate
    gate = Gate('MATRIX', qubits=all_qubits, U=U)

    # Iteratively call merge
    if len(bs) == 0:
        return gate
    else:
        return merge(gate, *bs)


def pad(gate: Gate,
        qubits: iter[any],
        order: iter[any] = None,
        return_matrix_only: bool = False) -> {MatrixGate, np.ndarray}:
    """
    Pad `gate` to act on `qubits`. More precisely, if `gate` is acting on a
    subset of `qubits`, extend `gate` with identities to act on all `qubits`.

    Parameters
    ----------
    gate: Gate
        The gate to pad.
    qubits: iter[any]
        Qubits used to pad `gate`. If `gate.qubits` is not a subset of
        `qubits`, raise an error.
    order: iter[any], optional
        If provided, reorder qubits in the final gate accordingly to `qubits`.
    return_matrix_only: bool, optional
        If `True`, the matrix representing the state is returned instead of
        `MatrixGate` (default: `False`).

    Returns
    -------
    MatrixGate
        The padded gate acting on `qubits`.
    """
    from hybridq.gate import MatrixGate
    from hybridq.utils import sort

    # Convert qubits to tuple
    qubits = tuple(qubits)

    # Convert order to tuple if provided
    order = None if order is None else tuple(order)

    # Check that order is a permutation of qubits
    if order and sort(qubits) != sort(order):
        raise ValueError("'order' must be a permutation of 'qubits'")

    # 'gate' must have qubits and it must be a subset of 'qubits'
    if not gate.provides('qubits') or set(gate.qubits).difference(qubits):
        raise ValueError("'gate' must provide qubits and those "
                         "qubits must be a subset of 'qubits'.")

    # Get matrix
    M = gate.matrix()

    # Pad matrix with identity
    if gate.n_qubits != len(qubits):
        M = np.kron(M, np.eye(2**(len(qubits) - gate.n_qubits)))

    # Get new qubits
    qubits = gate.qubits + tuple(set(qubits).difference(gate.qubits))

    # Reorder if required
    if order and order != qubits:
        # Get new matrix
        M = MatrixGate(M, qubits=qubits).matrix(order=order)

        # Set new qubits
        qubits = order

    # Return gate
    return M if return_matrix_only else MatrixGate(
        M, qubits=qubits, tags=gate.tags if gate.provides('tags') else {})


def decompose(gate: Gate,
              qubits: iter[any],
              return_matrices: bool = False,
              atol: float = 1e-8) -> SchmidtGate:
    """
    Decompose `gate` using the Schmidt decomposition.

    Parameters
    ----------
    gate: Gate
        `Gate` to decompose.
    qubits: iter[any]
        Subset of qubits used to decompose `gate`.
    return_matrices: bool, optional
        If `True`, return matrices instead of gates (default: `False`)
    atol: float
        Tollerance.

    Returns
    -------
    d: tuple(list[float], tuple[Gate, ...], tuple[Gate, ...])
        Decomposition of `gate`.

    See Also
    --------
    `hybridq.utils.svd`
    """
    from hybridq.gate import SchmidtGate
    from hybridq.utils import svd

    # Check qubits
    try:
        qubits = tuple(qubits)
    except:
        raise ValueError("'qubits' must be convertible to tuple.")

    # Get number of qubits in subset
    ns = len(qubits)

    # Get qubits not in subset
    alt_qubits = tuple(q for q in gate.qubits if q not in qubits)

    # Check is valid subset
    if set(qubits).difference(gate.qubits):
        raise ValueError("'qubits' must be a valid subset of `gate.qubits`.")

    # Get order
    axes = [gate.qubits.index(x) for x in qubits]
    axes += [x + gate.n_qubits for x in axes]

    # Get matrix and decompose it
    s, uh, vh = svd(np.reshape(gate.matrix(), (2,) * 2 * gate.n_qubits),
                    axes,
                    atol=atol)

    # Reshape
    uh = np.reshape(uh, (len(s), 2**ns, 2**ns))
    vh = np.reshape(vh,
                    (len(s), 2**(gate.n_qubits - ns), 2**(gate.n_qubits - ns)))

    # Return gates
    return (s, uh, vh) if return_matrices else SchmidtGate(
        gates=((Gate('MATRIX', qubits=qubits, U=x) for x in uh),
               (Gate('MATRIX', qubits=alt_qubits, U=x) for x in vh)),
        s=s)
