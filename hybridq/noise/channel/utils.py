"""
Authors: Salvatore Mandra (salvatore.mandra@nasa.gov),
         Jeffrey Marshall (jeffrey.s.marshall@nasa.gov)

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
import numpy as np


def is_dm(rho: np.ndarray, atol=1e-6) -> bool:
    """
    check if the given input a valid density matrix.
    """
    rho = np.asarray(rho)
    d = int(np.sqrt(np.prod(rho.shape)))
    rho_full = np.reshape(rho, (d, d))

    hc = np.allclose(rho_full, rho_full.T.conj(), atol=atol)
    tp = np.isclose(np.trace(rho_full), 1, atol=atol)

    apprx_gtr = lambda y, x: np.real(y) >= x or np.isclose(y, x, atol=atol)
    ev = np.linalg.eigvals(rho_full)
    psd = np.all([apprx_gtr(e, 0) for e in ev])

    return (hc and tp and psd)


def ptrace(state: np.ndarray,
           keep: {int, list[int]},
           dims: {int, list[int]} = None) -> np.ndarray:
    """
    compute the partial trace of a pure state (vector) or density matrix.
    state: np.array
        One dimensional for pure state e.g. np.array([1,0,0,0])
        or two dimensional for density matrix e.g. np.array([[1,0],[0,0]])
    keep: list of int
        the qubits we want to keep (all others traced out).
        Can also specify a single int if only keeping one qubit.
    dims: list of int, optional
        List of qudit dimensions respecting the ordering of `state`.
        Number of qubits is `len(dims)`, and full Hilbert space
        dimension is `product(dims)`.
        If unspecified, assumes 2 for all.
    Returns the density matrix of the remaining qubits.
    """
    state = np.asarray(state)
    if len(state.shape) not in (1, 2):
        raise ValueError('should be pure state (one dimensional) '
                         'or density matrix (two dimensional). '
                         f'Received dimension {len(state.shape)}')

    # pure state or not
    pure = len(state.shape) == 1
    if not pure and state.shape[0] != state.shape[1]:
        raise ValueError('invalid state input.')

    full_dim = np.prod(state.shape[0])
    if dims is not None and full_dim != np.prod(dims):
        raise ValueError('specified dimensions inconsistent with state')

    n_qubits = np.log2(full_dim) if dims is None else len(dims)
    if np.isclose(n_qubits, round(n_qubits)):
        n_qubits = int(round(n_qubits))
    else:
        raise ValueError('invalid state size')

    keep = [keep] if isinstance(keep, int) else list(keep)
    if not np.all([q in range(n_qubits)
                   for q in keep]) or len(keep) >= n_qubits:
        raise ValueError('invalid axes')
    if dims is None:
        dims = [2] * n_qubits

    # dimensions of qubits we keep
    final_dims = [dims[i] for i in keep]
    final_dim = np.prod(final_dims)
    # dimensions to trace out
    drop_dim = int(round(full_dim / final_dim))

    if pure:
        state = state.reshape(dims)
        perm = keep + [q for q in range(n_qubits) if q not in keep]
        state = np.transpose(state, perm).reshape(final_dim, drop_dim)
        return np.einsum('ij,kj->ik', state, state.conj())
    else:
        # now we have to redefine things in case of a density matrix
        # basically we double the sizes
        density_dims = dims + dims
        keep += [q + n_qubits for q in keep]
        perm = keep + [q for q in range(2 * n_qubits) if q not in keep]

        state = state.reshape(density_dims)
        state = np.transpose(state, perm)
        state = state.reshape((final_dim, final_dim, drop_dim, drop_dim))
        return np.einsum('ijkk->ij', state)


def is_channel(channel: MatrixChannel,
               atol=1e-8,
               order: tuple[any, ...] = None,
               **kwargs) -> bool:
    """
    Checks using the Choi matrix whether or not `channel` defines
    a valid quantum channel.
    That is, we check it is a valid CPTP map.

    Parameters
    ----------
    atol: float, optional
        absolute tolerance to use for determining channel is CPTP.
    order: tuple[any, ...], optional
        If provided, Kraus' map is ordered accordingly to `order`.
        See `MatrixChannel.map()`
    kwargs: kwargs for `MatrixChannel.map()`
    """
    C = choi_matrix(channel, order, **kwargs)
    dim = _channel_dim(channel)

    # trace preserving
    tp = np.isclose(C.trace(), dim, atol=atol)

    # hermiticity preserving
    hp = np.allclose(C, C.conj().T, atol=atol)

    # completely positive
    apprx_gtr = lambda e, x: np.real(e) >= x or np.isclose(e, x, atol=atol)
    cp = np.all([
        apprx_gtr(e, 0) and np.isclose(np.imag(e), 0, atol=atol)
        for e in np.linalg.eigvals(C)
    ])

    return tp and hp and cp


def choi_matrix(channel: MatrixChannel,
                order: tuple[any, ...] = None,
                **kwargs) -> np.ndarray:
    """
    return the Choi matrix for channel, of shape (d**2, d**2)
    for a d-dimensional Hilbert space.

    The channel can be applied as:
    Lambda(rho) = Tr_0[ (I \otimes rho^T) C]
    where C is the Choi matrix.

    Parameters
    ----------
    order: tuple[any, ...], optional
        If provided, Kraus' map is ordered accordingly to `order`.
        See `MatrixChannel.map()`
    kwargs: kwargs for `MatrixChannel.map()`
    """

    op = channel.map(order, **kwargs)
    d = _channel_dim(channel)

    C = np.zeros((d**2, d**2), dtype=complex)
    for ij in range(d**2):
        Eij = np.zeros(d**2)
        Eij[ij] = 1
        map = op @ Eij  # using vectorization
        C += np.kron(Eij.reshape((d, d)), map.reshape((d, d)))

    return C


def _channel_dim(channel):
    # dimension (assume all Kraus' have same shape)
    return channel.Kraus.gates[0][0].matrix().shape[0]
