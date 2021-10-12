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
from hybridq.base import __Base__, generate, staticvars, compare, requires
from hybridq.gate import BaseGate, property as pr
from hybridq.dm.gate import BaseSuperGate, property as dm_pr
import numpy as np


class BaseChannel(__Base__):
    """
    Base class for `Channel`s.
    """
    pass


@compare('LMatrices,RMatrices,s')
@staticvars('LMatrices,RMatrices,s',
            transform=dict(LMatrices=lambda M: tuple(map(np.asarray, M)),
                           RMatrices=lambda M: tuple(map(np.asarray, M)),
                           s=lambda s: np.asarray(s)))
class _MatrixChannel(BaseChannel,
                     BaseGate,
                     BaseSuperGate,
                     pr.QubitGate,
                     pr.TagGate,
                     pr.NameGate,
                     n_qubits=any,
                     name=''):
    """
    Base class for `MatrixChannel`s.
    """

    def __init_subclass__(cls, **kwargs):
        from hybridq.dm.gate import KrausSuperGate
        from hybridq.gate import MatrixGate

        # Call super
        super().__init_subclass__(**kwargs)

        # Get static variables
        LMatrices = cls.__get_staticvar__('LMatrices')
        RMatrices = cls.__get_staticvar__('RMatrices')
        n_qubits = cls.__get_staticvar__('n_qubits')
        s = cls.__get_staticvar__('s')

        # Check dimensions
        if any(M.shape != (2**n_qubits, 2**n_qubits) for M in LMatrices):
            raise ValueError("Matrices in 'LMatrices' have shapes "
                             "not consistent with number of qubits.")
        if any(M.shape != (2**n_qubits, 2**n_qubits) for M in RMatrices):
            raise ValueError("Matrices in 'RMatrices' have shapes "
                             "not consistent with number of qubits.")

        # Check consistency
        if s.ndim == 0:
            if len(LMatrices) != len(RMatrices):
                raise ValueError(
                    "The number of 'LMatrices' must be the same of 'RMatrices'")
        elif s.ndim == 1:
            if len(s) != len(LMatrices) or len(s) != len(RMatrices):
                raise ValueError(
                    "'s.shape' is not consistent with number of matrices. "
                    f"(expected {(len(LMatrices), )}, "
                    f"got {s.shape})")
        elif s.ndim == 2:
            if s.shape != (len(LMatrices), len(RMatrices)):
                raise ValueError(
                    "'s.shape' is not consistent with number of matrices. "
                    f"(expected {(len(LMatrices), len(RMatrices))}, "
                    f"got {s.shape})")

        # Get gates
        LGates = tuple(map(MatrixGate, LMatrices))
        RGates = tuple(map(MatrixGate, RMatrices))

        # Initialize Kraus operator
        cls.__Kraus = KrausSuperGate(gates=(LGates, RGates), s=s)

    def __print__(self):
        return dict(s=(100, f's.shape={self.s.shape}', 0))

    @property
    def Kraus(self) -> KrausSuperGate:
        """
        Return `KrausSuperGate` representing `_MatrixChannel`.
        """

        # Check if qubits are current ...
        if self.__Kraus.gates[0][0].qubits != self.qubits:
            # ... otherwise, update qubits in all gates
            for lg, rg in zip(*self.__Kraus.gates):
                lg._on(self.qubits)
                rg._on(self.qubits)

        # Return Kraus operator
        return self.__Kraus

    def map(self,
            order: tuple[any, ...] = None,
            *,
            cache_map: bool = True) -> KrausMap:
        """
        Return `_MatrixChannel` Kraus' map.

        Parameters
        ----------
        order: tuple[any, ...], optional
            If provided, Kraus' map is ordered accordingly to `order`.
        cache_map: bool, optional
            If `cache_map == True`, then `KrausMap` is cached for the next
            call. (default: `True`)
        """

        # Check if KrausMap is already cached
        if cache_map and getattr(
                self, '_cache',
                None) and self._cache['qubits'] == self.qubits and self._cache[
                    'KrausMap'] is not None:
            # Get cached KrausMap
            KrausMap = self._cache['KrausMap']

        # Otherwise, compute fresh
        else:
            # Compute map
            KrausMap = self.Kraus.map(order=order)

            # Update cache
            if cache_map:
                self._cache = dict(qubits=self.qubits, KrausMap=KrausMap)

        # Return KrausMap
        return KrausMap

    def _clear_cache(self) -> None:
        """
        Clear `_MatrixChannel`'s cache.
        """
        try:
            delattr(self, '__cache')
        except AttributeError:
            pass


def MatrixChannel(LMatrices: tuple[array, ...],
                  RMatrices: tuple[array, ...] = None,
                  s: {float, array} = 1,
                  qubits: tuple[any, ...] = None,
                  tags: dict[any, any] = None,
                  name: str = 'MATRIX_CHANNEL',
                  copy: bool = True,
                  atol: float = 1e-8):
    """
    Return a channel described by `LMatrices` and `RMatrices`. More precisely,
    `MatrixChannel` will represent a channel of the form:

        rho -> E(rho) = \sum_ij s_ij L_i rho R_j^*

    with `rho` being a density matrix.

    Parameters
    ----------
    LMatrices: tuple[array, ...]
        Left matrices associated to `MatrixChannel`.
    RMatrices: tuple[array, ...], optional
        Right matrices associated to `MatrixChannel`. If not provided,
        `LMatrices` will be used as `RMatrices`.
    s: {float, array}, optional
        Weights for the left and right matrices associated to `MatrixChannel`.
        If `s` is a constant, then `s_ij = s` for `i == j` and zero otherwise.
        Similarly, if `s` is a one dimensional array, then `s_ij = s_i` for
        `i == j` and zero otherwise.
    tags: dict[any, any], optional
        Tags to add to `MatrixChannel`s.
    qubits: tuple[any, ...], optional
        Qubits the `MatrixChannel` will act on.
    name: str, optional
        Name of `MatrixChannel`.
    copy: bool, optional,
        If `copy == True`, then `LMatrices`, `RMatrices` and `s` are copied
        instead of passed by reference (default: `True`).
    atol: float, optional
        Use `atol` as absolute tollerance while checking.

    Returns
    -------
    MatrixChannel
    """

    from hybridq.utils import isintegral, isnumber

    # Define sample
    def __sample__(self, size: int = None, replace: bool = True):
        from hybridq.utils import isintegral

        # It is assumed here that s.ndim == 1
        assert (s.ndim == 1)

        # Get number of elements
        idxs = np.random.choice(range(len(s)), size=size, replace=replace, p=s)

        # Get gates (it is assumed here that left and right gates are the same)
        return self.Kraus.gates[0][idxs] if isintegral(idxs) else tuple(
            self.Kraus.gates[0][x] for x in idxs)

    # Define apply
    def __apply__(self, psi, order):
        raise NotImplementedError

    # Get matrices, and copy if needed
    LMatrices = tuple(map(np.array if copy else np.asarray, LMatrices))
    RMatrices = None if RMatrices is None else tuple(
        map(np.array if copy else np.asarray, RMatrices))

    # Get s
    if isnumber(s):
        # Convert to array
        s = float(s) * np.ones(len(LMatrices))

        # Check that LMatrices and RMatrices are consistent
        if RMatrices is not None and len(LMatrices) != len(RMatrices):
            raise ValueError("'s' cannot be a float if 'LMatrices' "
                             "and 'RMatrices' have different size")

    else:
        # Convert to array
        s = (np.array if copy else np.asarray)(s)

        # Try to reduce to vector
        if s.ndim == 2 and s.shape[0] == s.shape[1]:
            if np.allclose(s, np.diag(np.diag(s)), atol=atol):
                s = np.diag(s)

        # Check number of dimensions
        elif s.ndim > 2:
            raise ValueError("'s' not supported.")

        # If diagonal, convert to vector

    # At least on matrix should be provided
    if not len(LMatrices) or (RMatrices is not None and not (RMatrices)):
        raise ValueError("At least one matrix must be provided")

    # Get n_qubits from matrices
    n_qubits = np.log2(LMatrices[0].shape[0])
    if isintegral(n_qubits):
        n_qubits = int(n_qubits)
    else:
        raise ValueError("Only matrices acting on qubits are supported")

    # Check that all matrices have the same shape
    if any(m.shape != (2**n_qubits, 2**n_qubits) for m in LMatrices) or (
            RMatrices is not None and
            any(m.shape != (2**n_qubits, 2**n_qubits) for m in RMatrices)):
        raise ValueError("All matrices must have the same shape")

    # Get qubits
    qubits = None if qubits is None else tuple(qubits)

    # Check if qubits is consistent with number of qubits
    if qubits and len(qubits) != n_qubits:
        raise ValueError("'qubits' is not consistent with the size of matrices")

    # Get name
    name = str(name)

    # Set mro
    mro = (_MatrixChannel,)

    # Set static dictionary
    sdict = dict(name=name.upper(),
                 s=s,
                 LMatrices=LMatrices,
                 RMatrices=LMatrices if RMatrices is None else RMatrices,
                 n_qubits=n_qubits)

    # Initialize methods
    methods = {}

    # Check if all matrices are unitaries
    if s.ndim == 1 and RMatrices is None and np.isclose(np.sum(s), 1,
                                                        atol=atol):
        # Check if matrix is unitaries
        def _is_unitary(m):
            # Get multiplications
            p1 = m.conj().T @ m
            p2 = m @ m.conj().T
            # Check
            return np.allclose(p1, p2, atol=atol) and np.allclose(
                p1, np.eye(p1.shape[0]))

        # Check if stochasticity can be applied
        _stochastic = all(map(_is_unitary, LMatrices))
    else:
        _stochastic = False

    # If all unitaries, add sample
    if _stochastic:
        # Update mro
        mro = mro + (pr.StochasticGate,)

        # Add gates
        sdict.update(gates=None)

        # Add sample
        methods.update(sample=__sample__)

    # Otherwise, add apply
    else:
        # Update mro
        mro = mro + (pr.FunctionalGate,)

        # Add update
        sdict.update(apply=__apply__)

    # Generate gate
    return generate(''.join(name.title().split('_')),
                    mro,
                    methods=methods,
                    **sdict)(qubits=qubits, tags=tags)


def GlobalPauliChannel(qubits: tuple[any, ...],
                       s: {float, array, dict},
                       tags: dict[any, any] = None,
                       name: str = 'GLOBAL_PAULI_CHANNEL',
                       copy: bool = True,
                       atol: float = 1e-8) -> GlobalPauliChannel:
    """
    Return a `GlobalPauliChannel`s acting on `qubits`.
    More precisely, each `LocalPauliChannel` has the form:

        rho -> E(rho) = \sum_{i1,i2,...}{j1,j2,...}
                s_{i1,i2...}{j1,j2,...}
                    sigma_i1 sigma_i2 ... rho sigma_j1 sigma_j2 ...

    with `rho` being a density matrix and `sigma_i` being Pauli matrices.

    Parameters
    ----------
    qubits: tuple[any, ...]
        Qubits the `LocalPauliChannel`s will act on.
    s: {float, array, dict}
        Weight for Pauli matrices.
        If `s` is a float, the diagonal of the matrix s_ij is set to `s`.
        Similarly, if `s` is a one dimensional array, then the diagonal of
        matrix s_ij is set to that array.
        If `s` is a `dict`, weights can be specified by
        using the tokens `I`, `X`, `Y` and `Z`. For instance, `dict(XYYZ=0.2)`
        will set the weight for `sigma_i1 == X`, `sigma_i2 == Y`, `sigma_j1 == Y`
        and `sigma_j2 == Z` to `0.2`.
    tags: dict[any, any]
        Tags to add to `LocalPauliChannel`s.
    name: str, optional
        Alternative name for `GlobalPauliChannel`.
    copy: bool, optional,
        If `copy == True`, then `s` is copied instead of passed by reference
        (default: `True`).
    atol: float, optional
        Use `atol` as absolute tollerance while checking.
    """

    from hybridq.utils import isintegral, kron
    from itertools import product
    from hybridq.gate import Gate

    # Get qubits
    qubits = tuple(qubits)

    # Define n_qubits
    n_qubits = len(qubits)

    # If 's' is a 'dict'
    if isinstance(s, dict):
        # Convert to upper
        s = {str(k).upper(): v for k, v in s.items()}

        # Check if tokens are valid
        if any(len(k) != 2 * n_qubits for k in s):
            raise ValueError("Keys in 's' must have twice a number of "
                             "tokens which is twice the number of qubits")

        if any(set(k).difference('IXYZ') for k in s):
            raise ValueError("'s' contains non-valid tokens")

        # Get position
        def _get_position(k):
            return sum(
                (4**i * dict(I=0, X=1, Y=2, Z=3)[k]) for i, k in enumerate(k))

        # Build matrix
        _s = np.zeros((4**n_qubits, 4**n_qubits))
        for k, v in s.items():
            # Get positions
            x, y = _get_position(k[:n_qubits]), _get_position(k[n_qubits:])

            # Fill matrix
            _s[x, y] = v

        # assign
        s = _s

    # Otherwise, convert to array
    else:
        s = (np.array if copy else np.asarray)(s)

        # If a single float, return vector
        if s.ndim == 0:
            s = np.ones(4**n_qubits) * s

        # Otherwise, dimensions must be consistent
        elif s.ndim > 2 or set(s.shape) != {4**n_qubits}:
            raise ValueError("'s' must be either a vector of exactly "
                             f"{4**n_qubits} elements, or a "
                             f"{(4**n_qubits, 4**n_qubits)} matrix")

    # Get matrices
    Matrices = [
        kron(*m)
        for m in product(*([[Gate(g).Matrix for g in 'IXYZ']] * n_qubits))
    ]

    # Return gate
    return MatrixChannel(LMatrices=Matrices,
                         qubits=qubits,
                         s=s,
                         tags=tags,
                         name=name,
                         copy=False,
                         atol=atol)


def LocalPauliChannel(qubits: tuple[any, ...],
                      s: {float, array, dict},
                      tags: dict[any, any] = None,
                      name: str = 'LOCAL_PAULI_CHANNEL',
                      copy: bool = True,
                      atol: float = 1e-8) -> tuple[LocalPauliChannel, ...]:
    """
    Return a `tuple` of `LocalPauliChannel`s acting independently on `qubits`.
    More precisely, each `LocalPauliChannel` has the form:

        rho -> E_i(rho) = \sum_ij s_ij sigma_i rho sigma_j

    with `rho` being a density matrix and `sigma_i` being Pauli matrices.

    Parameters
    ----------
    qubits: tuple[any, ...]
        Qubits the `LocalPauliChannel`s will act on.
    s: {float, array, dict}
        Weight for Pauli matrices. If `s` is a constant, then `s_ij = s` for
        `i == j` and zero otherwise.  Similarly, if `s` is a one dimensional
        array, then `s_ij = s_i` for `i == j` and zero otherwise. If `s` is a
        `dict`, weights can be specified by using the tokens `I`, `X`, `Y` and
        `Z`. For instance, `dict(XY=0.2)` will set `s[1, 2] = 0.2`.
    tags: dict[any, any]
        Tags to add to `LocalPauliChannel`s.
    name: str, optional
        Alternative name for `LocalPauliChannel`s.
    copy: bool, optional,
        If `copy == True`, then `s` is copied instead of passed by reference
        (default: `True`).
    atol: float, optional
        Use `atol` as absolute tollerance while checking.
    """

    return tuple(
        GlobalPauliChannel(
            qubits=(q,), name=name, s=s, tags=tags, copy=copy, atol=atol)
        for q in qubits)
