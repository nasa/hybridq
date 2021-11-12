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
@staticvars('LMatrices,RMatrices,s,_use_cache',
            _use_cache=True,
            transform=dict(LMatrices=lambda M: tuple(map(np.asarray, M)),
                           RMatrices=lambda M: tuple(map(np.asarray, M)),
                           s=lambda s: np.asarray(s),
                           _use_cache=lambda x: bool(x)))
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
        use_cache = cls.__get_staticvar__('_use_cache')

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
        cls.__Kraus = KrausSuperGate(gates=(LGates, RGates),
                                     s=s,
                                     use_cache=use_cache)

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

    def map(self, order: tuple[any, ...] = None) -> KrausMap:
        """
        Return `_MatrixChannel` Kraus' map.

        Parameters
        ----------
        order: tuple[any, ...], optional
            If provided, Kraus' map is ordered accordingly to `order`.
        """

        # Return map
        return self.Kraus.map(order=order)


def MatrixChannel(LMatrices: tuple[array, ...],
                  RMatrices: tuple[array, ...] = None,
                  s: {float, array} = 1,
                  qubits: tuple[any, ...] = None,
                  tags: dict[any, any] = None,
                  name: str = 'MATRIX_CHANNEL',
                  copy: bool = True,
                  atol: float = 1e-8,
                  methods: dict[any, any] = None,
                  use_cache: bool = True,
                  norm_atol: float = 1e-6):
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
    methods: dict[any, any]
        Add extra methods to the object.
    use_cache: bool, optional
        If `True`, extra memory is used to store a cached `Matrix`.
    norm_atol: float, optional
        State vectors with norm smaller than `norm_atol` are considered zero
        vectors.

    Returns
    -------
    MatrixChannel
    """
    from hybridq.utils import isintegral, isnumber

    # Convert norm_atol to float
    norm_atol = float(norm_atol)

    # Define sample
    def __sample__(self, size: int = None, replace: bool = True):
        from hybridq.utils import isintegral

        # It is assumed here that s.ndim == 1
        assert (self.s.ndim == 1)

        # Get number of elements
        idxs = np.random.choice(range(len(self.s)),
                                size=size,
                                replace=replace,
                                p=self.s)

        # Get gates (it is assumed here that left and right gates are the same)
        return self.Kraus.gates[0][idxs] if isintegral(idxs) else tuple(
            self.Kraus.gates[0][x] for x in idxs)

    # Define apply
    def __apply__(self, psi, order):
        from hybridq.utils import dot

        # It is assumed here that s.ndim == 1
        assert (self.s.ndim == 1)

        # Define how to get projection
        def _get_projection(idx):
            # Get s
            s = self.s[idx]

            # Get Kraus operator
            gate = self.Kraus.gates[0][idx]

            # Get projection
            proj = dot(a=gate.matrix(),
                       b=psi,
                       axes=tuple(map(order.index, gate.qubits)),
                       b_as_complex_array=not np.iscomplexobj(psi))

            # Get normalization
            norm = np.linalg.norm(proj.ravel())

            # If norm is smaller than 'norm_atol', state vector is assumed to
            # be zero
            if norm < norm_atol:
                norm = 0

            # Get probability
            prob = s * norm**2

            # Return projection, norm and probability
            return proj, norm, prob

        # Convert order to tuple
        order = tuple(order)

        # Get random number
        r = np.random.random()

        # Initialize cumulative
        c = 0

        # Last non-zero state
        proj_idx = None

        # For each Kraus operator ...
        for idx in self.__LMatrices_order:
            # Get projection, norm and probability
            proj, norm, prob = _get_projection(idx)

            # If norm is different from zero ...
            if norm > 0:
                # Store idx
                proj_idx = idx

                # Update cumulative
                c += prob

                # If the random number if smaller than the cumulative, break
                if c >= r:
                    break

        # This is triggered only if a rounding error happened
        else:
            # If c is not close to one within 'norm_atol', warn the user
            if not np.isclose(c, 1, atol=norm_atol):
                from hybridq.utils import Warning
                from warnings import warn
                warn(
                    f"The final cumulative, 'c={c}', is not close to '1'"
                    f"within the absolute tollerance, "
                    f"'norm_atol={norm_atol}'.", Warning)

            # Check that proj_idx has been succesfully assigned
            if proj_idx is None:
                raise RuntimeError("All state vectors have a norm smaller "
                                   "than the absolute tollerance, "
                                   f"'norm_atol={norm_atol}'")

            # Get projection, norm and probability
            proj, norm, _ = _get_projection(proj_idx)

        # Normalize state
        proj /= norm

        # Return projection and order
        return proj, order

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
                 n_qubits=n_qubits,
                 _use_cache=use_cache)

    # Initialize methods
    _methods = {}

    # FunctionalGate/StochasticGate can be used if s.ndim == 1 and left/right
    # gates are the same
    if s.ndim == 1 and (RMatrices is None or LMatrices == RMatrices):
        from hybridq.utils import isunitary

        # Check if StochastiGate's can be used
        if np.isclose(np.sum(s), 1, atol=atol) and all(map(
                isunitary, LMatrices)):
            # Update mro
            mro = mro + (pr.StochasticGate,)

            # Add sample
            _methods.update(sample=__sample__)

        # Otherwise, use FunctionalGate's
        elif np.allclose(sum(
                s * (m.conj().T @ m) for s, m in zip(s, LMatrices)),
                         np.eye(LMatrices[0].shape[0]),
                         atol=atol):
            from numpy.linalg import eigvals

            # Update mro
            mro = mro + (pr.FunctionalGate,)

            # Get order of matrices (largest norm first)
            _methods['__LMatrices_order'] = np.argsort(
                [np.linalg.norm(eigvals(m)) for m in LMatrices])[::-1]

            # Add update
            sdict.update(apply=__apply__)

    # Merge extra methods
    if methods is not None:
        _methods.update(methods)

    # Generate gate
    return generate(''.join(name.title().split('_')),
                    mro,
                    methods=_methods,
                    **sdict)(qubits=qubits, tags=tags)


def GlobalPauliChannel(qubits: tuple[any, ...],
                       s: {float, array, dict},
                       tags: dict[any, any] = None,
                       name: str = 'GLOBAL_PAULI_CHANNEL',
                       copy: bool = True,
                       atol: float = 1e-8,
                       methods: dict[any, any] = None,
                       use_cache: bool = True) -> GlobalPauliChannel:
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
    methods: dict[any, any]
        Add extra methods to the object.
    use_cache: bool, optional
        If `True`, extra memory is used to store a cached `Matrix`.
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
        for m in product(*([[Gate(g, n_qubits=1).Matrix for g in 'IXYZ']] *
                           n_qubits))
    ]

    # Return gate
    return MatrixChannel(LMatrices=Matrices,
                         qubits=qubits,
                         s=s,
                         tags=tags,
                         name=name,
                         copy=False,
                         atol=atol,
                         methods=methods,
                         use_cache=use_cache)


def LocalPauliChannel(qubits: tuple[any, ...],
                      s: {float, array, dict},
                      tags: dict[any, any] = None,
                      name: str = 'LOCAL_PAULI_CHANNEL',
                      copy: bool = True,
                      atol: float = 1e-8,
                      methods: dict[any, any] = None,
                      use_cache: bool = True) -> tuple[LocalPauliChannel, ...]:
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
        Use `atol` as absolute tolerance while checking.
    methods: dict[any, any]
        Add extra methods to the object.
    use_cache: bool, optional
        If `True`, extra memory is used to store a cached `Matrix`.
    """

    return tuple(
        GlobalPauliChannel(qubits=(q,),
                           name=name,
                           s=s,
                           tags=tags,
                           copy=copy,
                           atol=atol,
                           methods=methods,
                           use_cache=use_cache) for q in qubits)


def LocalDepolarizingChannel(qubits: tuple[any, ...],
                             p: {float, array, dict},
                             name: str = 'LOCAL_DEPOLARIZING_CHANNEL',
                             **kwargs) -> tuple[LocalDepolarizingChannel, ...]:
    """
    Return a `tuple` of `LocalDepolarizingChannel`s acting independently on
    `qubits`.  More precisely, each channel has the form:

        rho -> E_i(rho) = (1-p_i) rho + p_i * I/2

    with `rho` being a density matrix, p_i the user specified depolarizing
    probability for qubit i, and I the identity matrix.

    Parameters
    ----------
    qubits: tuple[any, ...]
        Qubits the `LocalDepolarizingChannel`s will act on.
    p: {float, array, dict}
        Depolarizing probability for qubits.
        If a single value is passed, the same is used for all qubits.
        If a one dimensional array is passed, it must be the same length
        of `qubits`, which corresponds to each depolarizing probability.
        Otherwise a dictionary mapping from `qubit`s to probability
        can be given.
    name: str, optional
        Alternative name for channel.
    kwargs: kwargs for GlobalPauliChannel
    """
    # Get probs
    p = __get_params(keys=qubits, args=p, key_name='qubit', value_type=float)

    # Return gates
    return tuple(
        GlobalDepolarizingChannel(qubits=(q,), name=name, p=p[q], **kwargs)
        for q in qubits)


def GlobalDepolarizingChannel(qubits: tuple[any, ...],
                              p: float,
                              name: str = 'GLOBAL_DEPOLARIZING_CHANNEL',
                              **kwargs) -> GlobalDepolarizingChannel:
    """
    Return a depolarizing channel that acts on all qubits

        rho -> E(rho) = (1-p) rho + p * I/d

    with `rho` being a density matrix of `len(qubits)` qubits,
    p the user specified depolarizing probability,
    I the identity matrix, and d=2**len(qubits) the full dimension.

    Parameters
    ----------
    qubits: tuple[any, ...]
        Qubits the `GlobalDepolarizingChannel` will act on.
    p: float
        Depolarizing probability.
    name: str, optional
        Alternative name for channel.
    kwargs: kwargs for GlobalPauliChannel
    """
    # Convert p to float
    p = float(p)

    # Get size of s
    ns = 4**len(qubits)

    # Get s
    s = [1 - p + p / ns] + [p / ns] * (ns - 1)

    # Return gate
    return GlobalPauliChannel(
        qubits=qubits,
        name=name,
        s=s,
        methods=dict(__print__=lambda self: {'p': (400, f'p={p}', 0)}),
        **kwargs)


def LocalDephasingChannel(qubits: tuple[any, ...],
                          p: {float, array, dict},
                          pauli_index: int = 3,
                          name: str = 'LOCAL_DEPHASING_CHANNEL',
                          **kwargs) -> tuple[LocalDephasingChannel, ...]:
    """
    Return a `tuple` of `LocalDephasingChannel`s acting independently
    on `qubits`.
    More precisely, each channel has the form:

        rho -> E_i(rho) = (1-p_i) rho + p_i * σ rho σ

    with `rho` being a density matrix, p_i the user specified depolarizing
    probability for qubit i, and σ a user specified Pauli matrix.

    Parameters
    ----------
    qubits: tuple[any, ...]
        Qubits the `LocalDephasingChannel`s will act on.
    p: {float, array, dict}
        Dephasing probability for qubits.
        If a single value is passed, the same is used for all qubits.
        If a one dimensional array is passed, it must be the same length
        of `qubits`, which corresponds to each dephasing probability.
        Otherwise a dictionary mapping from `qubit`s to probability
        can be given.
    pauli_index: int
        Integer in {0,1,2,3} representing the dephasing axis (Pauli matrix).
        If a single value is passed, the same is used for all qubits.
        If a one dimensional array is passed, it must be the same length of
        `qubits`, which corresponds to each dephasing axis.  Otherwise a
        dictionary mapping from `qubit`s to axis can be given.
    name: str, optional
        Alternative name for channel.
    kwargs: kwargs for GlobalPauliChannel
    """
    # Get probs and pauli_index
    p = __get_params(keys=qubits, args=p, key_name='qubit', value_type=float)
    pauli_index = __get_params(keys=qubits,
                               args=pauli_index,
                               key_name='qubit',
                               value_type=int)

    # Check pauli_index
    if any(v not in range(4) for v in pauli_index.values()):
        raise ValueError("`pauli_index` must be in {0,1,2,3}")

    # Get gate
    def _get_gate(q):
        # Get probability and axis
        _p = p[q]
        _x = pauli_index[q]
        _xl = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}[_x]

        # Get s
        s = [1 - _p, 0, 0, 0]
        s[_x] += _p

        # Return gate
        return GlobalPauliChannel(
            qubits=(q,),
            name=name,
            s=s,
            methods=dict(__print__=lambda self: dict(
                p=(400, f'p={_p}', 0), axis=(401, f'axis={_xl}', 0))),
            **kwargs)

    # Return gate
    return tuple(map(_get_gate, qubits))


def AmplitudeDampingChannel(qubits: tuple[any, ...],
                            gamma: {float, array, dict},
                            p: {float, array, dict} = 1,
                            name: str = 'AMPLITUDE_DAMPING_CHANNEL',
                            atol: float = 1e-8,
                            **kwargs) -> tuple[AmplitudeDampingChannel, ...]:
    """
    Return a `tuple` of `AmplitudeDampingChannel`s acting independently
    on `qubits`. There are 4 Kraus operators (for each qubit):
    sqrt(p) * [ [1, 0], [0, sqrt(1-gamma)] ]
    sqrt(p) * [ [0, sqrt(gamma)], [0, 0] ]
    sqrt(1-p) * [ [sqrt(1-gamma), 0], [0, 1] ]
    sqrt(1-p) * [ [0, 0], [sqrt(gamma), 0] ]

    Parameters
    ----------
    qubits: tuple[any, ...]
        Qubits the `AmplitudeDampingChannel`s will act on.
    gamma: {float, array, dict}
        Transition rate (0->1 and 1->0).
        If a single value is passed, the same is used for all qubits.
        If a one dimensional array is passed, it must be the same length
        of `qubits`, which corresponds to the value for each qubit.
        Otherwise a dictionary mapping from `qubit`s to gamma
        can be given.
    p: {float, array, dict}
        p is the probability for the 'damping' channel (governing 1->0), and
        1-p is probability for the 'excitation' channel (0->1).
        If a single value is passed, the same is used for all qubits.
        If a one dimensional array is passed, it must be the same length
        of `qubits`, which corresponds to the value for each qubit.
        Otherwise a dictionary mapping from `qubit`s to gamma
        can be given.
    name: str, optional
        Alternative name for channel.
    atol: float, optional
        Use `atol` as absolute tolerance while checking.
    kwargs: kwargs for MatrixChannel
    """
    # Get gammas and probs
    gamma = __get_params(keys=qubits,
                         args=gamma,
                         key_name='qubit',
                         value_type=float)
    p = __get_params(keys=qubits, args=p, key_name='qubit', value_type=float)

    def _get_gate(q):
        # Get gamma and p
        _gamma = gamma[q]
        _p = p[q]

        E0 = np.sqrt(_p) * np.diag([1, np.sqrt(1 - _gamma)])
        E1 = np.sqrt(_p) * np.array([[0, np.sqrt(_gamma)], [0, 0]])
        E2 = np.sqrt(1 - _p) * np.diag([np.sqrt(1 - _gamma), 1])
        E3 = np.sqrt(1 - _p) * np.array([[0, 0], [np.sqrt(_gamma), 0]])

        mats = []
        # drop zero operators
        for m in [E0, E1, E2, E3]:
            if not np.allclose(m, 0, atol=atol):
                mats += [m]

        # Return gate
        return MatrixChannel(
            LMatrices=tuple(mats),
            qubits=(q,),
            s=1,
            name=name,
            atol=atol,
            methods=dict(__print__=lambda self: dict(
                gamma=(400, f'gamma={_gamma}', 0), p=(401, f'p={_p}', 0))),
            **kwargs)

    # Return gate
    return tuple(map(_get_gate, qubits))


def __get_params(keys,
                 args,
                 key_type: callable = lambda x: x,
                 value_type: callable = lambda x: x,
                 key_name: str = 'key'):
    from hybridq.utils import to_dict, to_list
    from collections import defaultdict

    # Initialize output
    _args = None

    # Try to convert to float
    if _args is None:
        try:
            _args = {any: value_type(args)}
        except:
            pass

    # Try to convert to dict
    if _args is None:
        try:
            _args = to_dict(args, key_type=key_type, value_type=value_type)
        except:
            pass

    # Try to convert to list
    if _args is None:
        try:
            # Convert to list
            _args = to_list(args, value_type=value_type)

        except:
            pass

        else:
            # Check number of keys
            if len(_args) != len(keys):
                raise ValueError(f"Must have exactly one value per {key_name}")

            # Get dict
            _args = {key_type(k): x for k, x in zip(keys, _args)}

    # If _args is still None, raise
    if _args is None:
        raise TypeError(f"'{args}' not supported")

    # Check keys
    if any not in _args and set(keys) != set(_args):
        raise ValueError(f"All {key_name}s must be specified")

    # Return
    return defaultdict(lambda: _args[any], _args) if any in _args else _args
