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
from warnings import warn
import numpy as np
import scipy.linalg


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

    Parameters
    -----------
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

    Notes
    -----
    To convert shape to ket, one can use np.reshape(state, (d,)),
    where `d` is the dimension.
    To convert shape to density matrix, one can use np.reshape(state, (d, d)).
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


def is_channel(channel: SuperGate,
               atol=1e-8,
               order: tuple[any, ...] = None,
               **kwargs) -> bool:
    """
    Checks using the Choi matrix whether or not `channel` defines
    a valid quantum channel.
    That is, we check it is a valid CPTP map.

    Parameters
    ----------
    channel: MatrixSuperGate or KrausSuperGate
        Must have the method 'map()'.
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


def choi_matrix(channel: SuperGate,
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
    channel: MatrixSuperGate or KrausSuperGate
        Must have the method 'map()'.
    order: tuple[any, ...], optional
        If provided, Kraus' map is ordered accordingly to `order`.
        See `MatrixChannel.map()`
    kwargs: kwargs for `MatrixChannel.map()`
    """

    if not hasattr(channel, 'map'):
        raise ValueError("'channel' must have method 'map()'")

    op = channel.map(order, **kwargs)
    d = _channel_dim(channel)

    C = np.zeros((d**2, d**2), dtype=complex)
    for ij in range(d**2):
        Eij = np.zeros(d**2)
        Eij[ij] = 1
        map = op @ Eij  # using vectorization
        C += np.kron(Eij.reshape((d, d)), map.reshape((d, d)))

    return C


def fidelity(state1: np.ndarray,
             state2: np.ndarray,
             *,
             use_sqrt_def: bool = False,
             atol: float = 1e-8) -> float:
    """
    Compute the fidelity of two quantum states as:
    F(state1, state2) = ( Tr[ sqrt{sqrt(state1) * state2 * sqrt(state1)} ] )^2

    Parameters
    ----------
    state1: np.ndarray
        Either a ket or density matrix.
        If a ket, it should have shape (d,), where d is the dimension.
        If a density matrix, it should have shape (d, d).
    state2: np.ndarray
        Either a ket or density matrix.
        If a ket, it should have shape (d,), where d is the dimension.
        If a density matrix, it should have shape (d, d).
    use_sqrt_def: bool, optional
        If True, return the definition of fidelity without the square.
    atol: float, optional
        absolute tolerance used in rounding (imaginary parts
        smaller than this will be rounded to 0).


    Notes
    -----
    `state1` and `state2` must have consistent dimensions (but do not need
    to be both ket or both density matrix; one can be a ket and the other
    a density matrix).

    To convert shape to ket, one can use np.reshape(state, (d,)).
    To convert shape to density matrix, one can use np.reshape(state, (d, d)).

    If both states are pure, the definition is equivalent to
    |<psi1| psi2>|^2
    """

    state1 = np.asarray(state1)
    state2 = np.asarray(state2)

    def _validate_shape(rho_or_psi):
        valid = True
        dims = rho_or_psi.shape
        if len(dims) not in (1, 2):
            valid = False
        if len(dims) == 2 and dims[0] != dims[1]:
            valid = False
        if not valid:
            raise ValueError("Invalid state dimensions. "
                             "Ket type should be 1-dimensional (state.ndim==1)."
                             " Density matrix should be square d x d")

    _validate_shape(state1)
    _validate_shape(state2)

    dim1 = state1.shape[0]
    dim2 = state2.shape[0]

    if dim1 != dim2:
        raise ValueError(f"state dimensions inconsistent, got {dim1} != {dim2}")

    # ket or density matrix
    ket1 = state1.ndim == 1
    ket2 = state2.ndim == 1

    def _convert_to_real(F):
        if np.isclose(np.imag(F), 0, atol=atol):
            F = np.real(F)
        else:
            warn("Fidelity has non-trivial imaginary component")
        return F

    power = 1 if use_sqrt_def else 2
    if ket1 and ket2:
        # both states are kets
        return np.abs(np.inner(state1.conj(), state2))**power
    elif np.sum([ket1, ket2]) == 1:
        # one of the states is a ket, the other a density matrix
        # compute |<psi | rho | psi>|^2
        rho = state2 if ket1 else state1
        psi = state1 if ket1 else state2

        psi_right = rho @ psi
        F = np.sqrt(np.inner(psi.conj(), psi_right))
        return _convert_to_real(F)**power
    else:
        # both density matrices
        sqrt_rho = scipy.linalg.sqrtm(state1)

        _tmp = sqrt_rho @ state2 @ sqrt_rho

        # since we take the trace, we can just sum up the sqrt of the
        # eigenvalues, instead of computing the full matrix sqrt.
        eigs = np.linalg.eigvals(_tmp)

        F = np.sum([np.sqrt(e) for e in eigs])
        return _convert_to_real(F)**power


def reconstruct_dm(pure_states: list[np.ndarray],
                   probs: list[float] = None) -> np.ndarray:
    """
    Compute sum of pure states 1/N sum_i |psi_i><psi_i|.

    Parameters
    ----------
    pure_states: list[np.ndarray]
        A list of the pure states we wish to sum up to the density matrix.
    probs: list[float], optional
        If specified, it must be of the same length as `pure_states`.
        In this case, the computation will return
        sum_i P[i] |psi_i><psi_i|
        where P[i] is the i'th probability.
        Default will set each prob to 1/len(pure_states).


    Notes
    -----
    All states will be converted to be one-dimensional psi.shape = (d,),
    and the returned density matrix will be square (d,d).
    If there are inconsistencies in dims, a ValueError will be raised.
    """

    if probs is None:
        probs = [1 / len(pure_states)] * len(pure_states)

    if len(probs) != len(pure_states):
        raise ValueError("Invalid `probs`: length not consistent.")

    # here we convert to numpy arrays, then reshape to be one dimensional
    pure_states = [
        np.sqrt(probs[i]) * np.asarray(psi) for i, psi in enumerate(pure_states)
    ]
    pure_states = [
        np.reshape(psi, (np.prod(psi.shape),)) for psi in pure_states
    ]
    pure_states = np.asarray(pure_states)

    all_dims = set([np.prod(psi.shape) for psi in pure_states])
    if len(all_dims) != 1:
        raise ValueError(f"Recieved states with inconsistent dimensions. "
                         f"Received {all_dims}.")

    return np.einsum('ij,ik', pure_states, pure_states.conj())


def _channel_dim(channel):
    # map() gives the dimension squared of the channel
    full_dims = channel.map().shape
    assert len(full_dims) == 2
    assert full_dims[0] == full_dims[1]
    d = np.sqrt(full_dims[0])
    if not np.isclose(d, int(d)):
        raise ValueError('invalid shape for channel')
    return int(d)
