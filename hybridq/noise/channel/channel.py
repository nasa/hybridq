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
from hybridq.base import __Base__, generate, staticvars, compare, requires
from hybridq.gate import BaseGate, property as pr
from hybridq.dm.gate import BaseSuperGate, property as dm_pr
import numpy as np


class BaseChannel(__Base__):
    """
    Base class for 'Channel's.
    """
    pass


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

    def __init__(self, _use_cache: bool = True, **kwargs):
        """
        Initialize _MatrixChannel.

        Parameters
        ----------
        _use_cache: bool, optional
            Cache Kraus' map when possible.
        """
        # Update cache
        self.__use_cache = _use_cache

        # Initialize cache
        self.__cache = dict(KrausMap=None, qubits=None)

        # Call super
        super().__init__(**kwargs)

    def on(self, qubits: tuple[any, ...] = None, *, inplace=False):
        # Update qubits
        g = pr.QubitGate.on(self, qubits=qubits, inplace=inplace)

        # Update Kraus operator
        lg, rg = g.Kraus.gates
        for _gate in lg:
            _gate._on(g.qubits)
        for _gate in rg:
            _gate._on(g.qubits)

        # Return gate
        return g

    def __print__(self):
        return dict(s=(100, f's.shape={self.s.shape}', 0))

    @property
    def Kraus(self):
        # Check if qubits are current
        if self.__Kraus.gates[0][0].qubits != self.qubits:
            # Update Kraus operator
            lg, rg = self.__Kraus.gates
            for _gate in lg:
                _gate._on(self.qubits)
            for _gate in rg:
                _gate._on(self.qubits)

        # Return Kraus operator
        return self.__Kraus

    def map(self, order: iter[any] = None):
        # Check if KrausMap is already cached
        if self.__use_cache and self.__cache[
                'qubits'] == self.qubits and self.__cache[
                    'KrausMap'] is not None:
            # Get cached KrausMap
            KrausMap = self.__cache['KrausMap']

        # Otherwise, compute fresh
        else:
            # Compute map
            KrausMap = self.Kraus.map()

            # Update cache
            if self.__use_cache:
                self.__cache.update(qubits=self.qubits, KrausMap=KrausMap)

        # Return KrausMap
        return KrausMap


def MatrixChannel(LMatrices: tuple[array, ...],
                  RMatrices: tuple[array, ...] = None,
                  s: {float, array} = 1,
                  qubits: tuple[any, ...] = None,
                  tags: dict[any, any] = None,
                  name: str = 'MATRIX_CHANNEL',
                  copy: bool = True,
                  atol: float = 1e-8):
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


def LocalPauliChannel(qubits: tuple[any, ...],
                      s: {float, array, dict},
                      tags: dict[any, any] = None,
                      name: str = 'LOCAL_PAULI_CHANNEL',
                      copy: bool = True):
    from hybridq.utils import isintegral
    from hybridq.gate import Gate

    # Get qubits
    qubits = tuple(qubits)

    # Try to convert s to dict
    try:
        # All upper cases
        s = {(2 * k if len(k) == 1 else k): v
             for k, v in ((str(k).upper(), v) for k, v in dict(s).items())}

        # Check if tokens are valid
        if any(len(k) != 2 or set(k).difference('IXYZ') for k in s):
            raise ValueError("'s' contains non-valid tokens")

        # Build matrix
        _s = np.zeros((4, 4))
        for (k1, k2), v in s.items():
            k1 = dict(I=0, X=1, Y=2, Z=3)[k1]
            k2 = dict(I=0, X=1, Y=2, Z=3)[k2]
            _s[k1, k2] = v

    # Otherwise, convert to array
    except:
        s = (np.array if copy else np.asarray)(s)

        # If a single float, return vector
        if s.ndim == 0:
            s = np.ones(4) * s

        # Otherwise, dimensions must be consistent
        if s.ndim > 2 or set(s.shape) != {4}:
            raise ValueError("'s' must be either a vector of exactly "
                             "4 elements, or a (4, 4) matrix")

    # Get matrices
    Matrices = [Gate(g).Matrix for g in 'IXYZ']

    # Return gates
    return tuple(
        MatrixChannel(
            LMatrices=Matrices, qubits=(q,), s=s, tags=tags, name=name)
        for q in qubits)


def GlobalPauliChannel(qubits: tuple[any, ...],
                       s: {float, array, dict},
                       tags: dict[any, any] = None,
                       name: str = 'GLOBAL_PAULI_CHANNEL',
                       copy: bool = True):
    from hybridq.utils import isintegral, kron
    from itertools import product
    from hybridq.gate import Gate

    # Get qubits
    qubits = tuple(qubits)

    # Define n_qubits
    n_qubits = len(qubits)

    # Try to convert s to dict
    try:
        # Convert to dict
        s = {str(k).upper(): v for k, v in dict(s).items()}

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

    except:
        s = (np.array if copy else np.asarray)(s)

        # If a single float, return vector
        if s.ndim == 0:
            s = np.ones(4**n_qubits) * s

        # Otherwise, dimensions must be consistent
        if s.ndim > 2 or set(s.shape) != {4**n_qubits}:
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
                         name=name)
