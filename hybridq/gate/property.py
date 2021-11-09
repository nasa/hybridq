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
from hybridq.base import __Base__, generate
from hybridq.base import staticvars, compare, requires
from hybridq.base.property import Tags as TagGate, Name as NameGate, Params, Tuple, DocString
from scipy.linalg import expm, fractional_matrix_power as powm
from hybridq.utils import isintegral, isnumber
from copy import deepcopy
from hybridq.utils import sort
import numpy as np


def _truncate_print(x):
    if x is None:
        return ''
    else:
        if len(x) < 5:
            return f'{x}'
        else:
            _str = '(' if isinstance(x, tuple) else '['
            _str += ', '.join(str(x) for x in x[:2])
            _str += ', ..., '
            _str += ', '.join(str(x) for x in x[-2:])
            _str += ')' if isinstance(x, tuple) else ']'
            return _str


@compare('n_qubits,qubits')
@staticvars(
    'n_qubits',
    check=dict(n_qbits=(lambda n: n is any or (isintegral(n) and n >= 0),
                        "'n_qubits' must be a non-negative integer")))
class QubitGate(__Base__):
    """
    Class representing a gate with qubits.

    Attributes
    ----------
    qubits: iter[any], optional
    """

    def __init__(self, qubits: iter[any] = None, **kwargs) -> None:
        # Call super
        super().__init__(**kwargs)

        # Set qubits
        self._on(qubits)

    @property
    def qubits(self) -> tuple[any, ...]:
        return self.__qubits

    def __print__(self) -> dict[str, tuple[int, str, int]]:
        return {
            'n_qubits':
                (10, f"n_qubits={self.n_qubits}" if self.qubits is None else '',
                 0),
            'qubits':
                (100, f"qubits={_truncate_print(self.qubits)}"
                 if self.qubits is not None and self.n_qubits > 0 else '', 0),
        }

    def _on(self, qubits: iter[any]) -> None:
        """
        Set `qubits` to `QubitGate`.
        """

        self.on(qubits, inplace=True)

    def on(self,
           qubits: iter[any] = None,
           *,
           inplace: bool = False) -> QubitGate:
        """
        Return `QubitGate` applied to `qubits`. If `inplace` is `True`,
        `QubitGate` is modified in place.

        Parameters
        ----------
        qubits: iter[any]
            Qubits the new Gate will act on.
        inplace: bool, optional
            If `True`, `QubitGate` is modified in place. Otherwise, a new
            `QubitGate` is returned.

        Returns
        -------
        QubitGate
            New `QubitGate` acting on `qubits`. If `inplace` is `True`,
            `QubitGate` is modified in place.

        Example
        -------
        >>> QubitGate([1, 2]).qubits
        [1, 2]
        >>> QubitGate().on([42]).qubits
        [42]
        """

        # Convert to tuple
        qubits = None if qubits is None else tuple(qubits)

        # Check that all qubits are hashable
        if qubits and any(not getattr(q, '__hash__', None) for q in qubits):
            raise ValueError("Only hashable 'qubits' are allowed.")

        # Check that no qubits are repeated
        if qubits is not None and len(qubits) != len(set(qubits)):
            raise ValueError("Repeated qubits are not allowed.")

        # Check lenght
        if qubits is not None and len(qubits) != self.n_qubits:
            raise ValueError(f"Wrong number of 'qubits' "
                             f"(expected {self.n_qubits}, got {len(qubits)})")

        # Make a copy if needed
        if inplace:
            _g = self
        else:
            _g = deepcopy(self)

        # Set
        _g.__qubits = qubits

        # Return
        return _g


@compare('power')
class PowerGate(__Base__):
    """
    Class representing a gate that can be raised to a given power.

    Attributes
    ----------
    power: any, optional
    """

    def __init__(self, power: any = 1, **kwargs) -> None:
        # Call super
        super().__init__(**kwargs)

        # Assign power to object
        self._set_power(power)

    @property
    def power(self) -> any:
        return self.__power

    def __print__(self) -> dict[str, tuple[int, str, int]]:
        _power = ''
        if self.power != 1:
            if any(
                    isinstance(self.power, t)
                    for t in (int, float, np.integer, np.floating)):
                _power = f"**{np.round(self.power, 5)}"
            else:
                _power = f"**{self.power}"
        return {'power': (0, f"{_power}", 1)}

    def _set_power(self, power: any) -> None:
        """
        Set `power` to `PowerGate`.
        """

        self.set_power(power, inplace=True)

    def set_power(self, power: any, *, inplace: bool = False) -> PowerGate:
        """
        Return `PowerGate` to the given `power`. If `inplace` is `True`,
        `PowerGate` is modified in place.

        Parameters
        ----------
        power: any
            Power to elevate `PowerGate`.
        inplace: bool, optional
            If `True`, `PowerGate` is modified in place. Otherwise, a new
            `PowerGate` is returned.

        Returns
        -------
        PowerGate
            New `PowerGate` to the given `power`. If `inplace` is `True`,
            `PowerGate` is modified in place.

        Example
        -------
        >>> PowerGate(U=[[1, 0], [0, -1]]).matrix()
        array([[ 1,  0],
               [ 0, -1]])
        >>> PowerGate(U=[[1, 0], [0, -1]]).set_power(1.2345).matrix()
        array([[ 1.        +0.j        ,  0.        +0.j        ],
               [ 0.        +0.j        , -0.74068735-0.67184987j]])
        """

        # Make a copy if needed
        if inplace:
            _g = self
        else:
            _g = deepcopy(self)

        # Assign qubits
        _g.__power = 1 if power is None else power

        return _g

    def __pow__(self, p: any) -> PowerGate:
        """
        Return `PowerGate`**p.
        """

        return self.set_power(self.power * p, inplace=False)

    def _inv(self) -> None:
        """
        Modify `PowerGate` to its inverse.
        """

        self.inv(inplace=True)

    def inv(self, *, inplace: bool = False) -> PowerGate:
        """
        Return inverse of `PowerGate`. If `inplace` is `True`, `PowerGate` is modified in place.

        Parameters
        ----------
        inplace: bool, optional
            If `True`, PowerGate is modified in place. Otherwise, a new
            `PowerGate` is returned.

        Returns
        -------
        PowerGate
            Inverse of `PowerGate`. If `True`, `PowerGate` is modified in place.

        Example
        -------
        >>> g = PowerGate(U=[[1, 0], [0, np.exp(-0.23j)]])
        >>> g.matrix()
        array([[1.       +0.j        , 0.       +0.j        ],
               [0.       +0.j        , 0.9736664-0.22797752j]])
        >>> g.inv().matrix()
        array([[1.       -0.j        , 0.       -0.j        ],
               [0.       -0.j        , 0.9736664+0.22797752j]])
        >>> g.inv().matrix() @ g.matrix()
        array([[1.+0.j, 0.+0.j],
               [0.+0.j, 1.+0.j]])
        """

        return self.set_power(self.power * -1, inplace=inplace)


@compare('Matrix')
@staticvars('Matrix', transform=dict(Matrix=lambda Matrix: np.asarray(Matrix)))
class MatrixGate(__Base__):
    """
    Class for gates that can be represented as a matrix.
    """

    def __print__(self):
        return {
            'M': (
                400,
                f'M={type(self.Matrix).__module__}.{type(self.Matrix).__name__}(shape={self.Matrix.shape}, dtype={self.Matrix.dtype})',
                0)
        }


@requires('Matrix,qubits,n_qubits')
class PowerMatrixGate(PowerGate, __Base__):
    """
    Class representing a single matrix gate.
    """

    def __init__(self, *args, **kwargs):
        # Initialize local variables
        self.__conj = False
        self.__T = False

        # Continue
        super().__init__(*args, **kwargs)

    def __eq__(self, other: __Base__) -> bool:
        # Check if gates are conjugated
        _c1 = self.is_conjugated()
        _c2 = other.is_conjugated() if other.provides(
            'is_conjugated') else False

        # Check if gates are transposed
        _t1 = self.is_transposed()
        _t2 = other.is_transposed() if other.provides(
            'is_transposed') else False

        # If either the conjugation or the transposition is different, the
        # gates differ
        if _c1 != _c2 or _t1 != _t2:
            return False

        # Otherwise, keep going
        else:
            return super().__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()

    def __print__(self):
        return {
            'ct': (0, '^+' if self.__conj and self.__T else
                   '^T' if self.__T else '^*' if self.__conj else '', 2)
        }

    def _conj(self) -> PowerMatrixGate:
        self.conj(inplace=True)

    def conj(self, *, inplace: bool = False) -> PowerMatrixGate:
        # Create a copy if needed
        g = self if inplace else deepcopy(self)

        # Change conjugation
        g.__conj ^= True

        # Return gate
        return g

    def _T(self) -> PowerMatrixGate:
        self.T(inplace=True)

    def T(self, *, inplace: bool = False) -> PowerMatrixGate:
        # Create a copy if needed
        g = self if inplace else deepcopy(self)

        # Change transposition
        g.__T ^= True

        # Return gate
        return g

    def is_conjugated(self) -> bool:
        return self.__conj

    def is_transposed(self) -> bool:
        return self.__T

    def _adj(self) -> PowerMatrixGate:
        self.adj(inplace=True)

    def adj(self, *, inplace: bool = False) -> PowerMatrixGate:
        # Create a copy if needed
        g = self if inplace else deepcopy(self)

        # Change transposition
        g.__T ^= True

        # Change conjugation
        g.__conj ^= True

        # Return gate
        return g

    def matrix(self, order: iter[any] = None) -> np.ndarray:
        """
        Return matrix representing `MatrixPowerGate`. If `order` is provided,
        the given order of qubits is used to output its matrix.

        Parameters
        ----------
        order: iter[any]
            Order of qubits used to output the matrix.

        Returns
        -------
        array_like
            Matrix representing `MatrixPowerGate`.

        Example
        -------
        >>> g = PowerMatrixGate(qubits=[0, 1], U=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 1],
               [0, 0, 1, 0]])

        The order of qubits is `[1, 0]`. On the contrary:
        >>> g.on().matrix(order=[1, 0])
        array([[1, 0, 0, 0],
               [0, 0, 0, 1],
               [0, 0, 1, 0],
               [0, 1, 0, 0]])

        outputs a matrix with the qubits order being if `[0, 1]`.
        """

        # Get Unitary
        _U = np.asarray(self.Matrix)

        if order is not None:
            # Covert order to list
            order = list(order)

            # Order is allowed only if gate.qubits are specified
            if self.qubits is None or sort(order) != sort(self.qubits):
                raise ValueError(
                    "'order' is not a permutation of 'gate.qubits'.")

        # Reorder matrix in case qubits are out-of-order
        if order and order != self.qubits:
            # Transpose
            _U = np.reshape(
                np.transpose(
                    np.reshape(_U, (2,) * (2 * self.n_qubits)),
                    [self.qubits.index(q) for q in order] +
                    [self.n_qubits + self.qubits.index(q) for q in order]),
                (2**self.n_qubits, 2**self.n_qubits))

        # Apply power
        if self.power != 1:
            _U = powm(_U, float(self.power))

        # Apply conjugation
        if self.__conj:
            _U = _U.conj()

        # Apply transposition
        if self.__T:
            _U = _U.T

        # Return matrix
        return _U

    def isclose(self, gate: Gate, atol: float = 1e-8) -> bool:
        """
        Determine if the matrix of `gate` is close within an absolute
        tollerance. If the gates are acting on a different set of qubits,
        `isclose` will return `False`.

        Parameters
        ----------
        gate: PowerMatrixGate
            Gate to compare with.
        atol: float, optional
            Absolute tollerance.

        Returns
        -------
        bool
            `True` if the two gates are close withing the given absolute
            tollerance, otherwise `False`.

        Example
        -------
        >>> g1 = PowerMatrixGate(U = [[1, 2], [3, 4]])
        >>> g2 = PowerMatrixGate(U = [[4, 5], [3, 4]])
        >>> g1.isclose(g1)
        True
        >>> g1.isclose(g2)
        False
        >>> g1.on([3]).isclose(g1)
        False
        >>> g1.on([3]).isclose(g1.on([3])
        True
        """

        if not (self.qubits is None) ^ (gate.qubits is None):

            # The gates differ if they act on a different set of qubits
            if self.qubits is not None and sort(self.qubits) != sort(
                    gate.qubits):
                return False

            # Get unitaries
            _U1 = self.matrix(order=self.qubits if self.qubits else None)
            _U2 = gate.matrix(order=self.qubits if self.qubits else None)

            # If either matrix does not exist, the two gates differ
            return np.allclose(_U1, _U2, atol=atol)

        else:

            return False

    def commutes_with(self, gate: PowerMatrixGate, atol: float = 1e-7) -> bool:
        """
        Return `True` if the calling gate commutes with `gate`.

        Parameters
        ----------
        gate: PowerMatrixGate
            Gate to check commutation with.
        atol: float
            Absolute tollerance.

        Returns
        -------
        bool
            `True` if the calling gate commutes with `gate`, otherwise `False`.
        """
        from string import ascii_lowercase as alc, ascii_uppercase as auc

        # Check both gates have qubits
        if self.qubits is None or gate.qubits is None:
            raise ValueError("Cannot check commutation between virtual gates.")

        # Get shared qubits
        shared_qubits = sort(set(self.qubits).intersection(gate.qubits))

        # If no qubits are shared, the gates definitely commute
        if not shared_qubits:
            return True

        # Rename
        g1, g2 = self, gate

        # Get all qubits
        q12 = tuple(sort(set(g1.qubits + g2.qubits)))

        # Get number of qubits
        n12 = len(q12)

        # Get unitaries
        U1 = np.reshape(g1.matrix(), (2,) * 2 * g1.n_qubits)
        U2 = np.reshape(g2.matrix(), (2,) * 2 * g2.n_qubits)

        # Define how to multiply matrices
        def _mul(w1, w2):
            # Get qubits and unitaries
            q1, U1 = w1
            q2, U2 = w2

            # Get number of qubits
            n1 = len(q1)
            n2 = len(q2)

            # Construct map
            _map = ''
            _map += ''.join(alc[q12.index(q)] for q in q1)
            _map += ''.join(auc[-shared_qubits.index(q) -
                                1 if q in shared_qubits else q12.index(q)]
                            for q in q1)
            _map += ','
            _map += ''.join(auc[-shared_qubits.index(q) -
                                1] if q in shared_qubits else alc[q12.index(q)]
                            for q in q2)
            _map += ''.join(auc[q12.index(q)] for q in q2)
            _map += '->'
            _map += ''.join(alc[x] for x in range(n12))
            _map += ''.join(auc[x] for x in range(n12))

            # Multiply map
            return np.einsum(_map, U1, U2)

        # Compute products
        P1 = _mul((g1.qubits, U1), (g2.qubits, U2))
        P2 = _mul((g2.qubits, U2), (g1.qubits, U1))

        # Check if the same
        return np.allclose(P1, P2, atol=1e-5)


class UnitaryGate(PowerMatrixGate, skip_requirements=True):

    def set_power(self, power: any, *, inplace: bool = False) -> UnitaryGate:
        # If power is negative, just apply the absolute value and apply conj
        # and T as well.
        if isnumber(power) and power < 0:
            if power != -1:
                self = PowerMatrixGate.set_power(self, -power, inplace=inplace)
            self = self.adj(inplace=inplace)

        # Apply power
        else:
            self = PowerMatrixGate.set_power(self, power, inplace=inplace)

        return self

    def unitary(self, *args, **kwargs) -> np.ndarray:
        """
        Alias for `self.matrix`.
        """
        from hybridq.utils import DeprecationWarning
        from warnings import warn

        # Warn that `self.matrix` should be used instead of `self.unitary`
        warn("Since '0.7.0', 'self.matrix' should be used instead "
             "of the less general 'self.unitary'")

        # Call self.matrix
        return self.matrix(*args, **kwargs)


class SelfAdjointUnitaryGate(UnitaryGate, skip_requirements=True):

    def conj(self, *, inplace: bool = False) -> SelfAdjointUnitaryGate:
        """
        Apply conjugation to self.matrix().
        """
        if self.power == 1 and not self.is_conjugated() and self.is_transposed(
        ):
            return PowerMatrixGate.T(self, inplace=inplace)
        else:
            return PowerMatrixGate.conj(self, inplace=inplace)

    def T(self, *, inplace: bool = False) -> SelfAdjointUnitaryGate:
        """
        Apply transposition to self.matrix().
        """
        if self.power == 1 and self.is_conjugated(
        ) and not self.is_transposed():
            return PowerMatrixGate.conj(self, inplace=inplace)
        else:
            return PowerMatrixGate.T(self, inplace=inplace)

    def adj(self, *, inplace: bool = False) -> SelfAdjointUnitaryGate:
        """
        Apply adjunction to self.matrix().
        """
        if self.power == 1:
            return self if inplace else deepcopy(self)
        else:
            return PowerMatrixGate.adj(self, inplace=inplace)


@compare('Matrix_gen',
         cmp=dict(Matrix_gen=lambda x, y: x.__code__ == y.__code__))
@staticvars('Matrix_gen',
            check=dict(Matrix_gen=(lambda Matrix_gen: callable(Matrix_gen),
                                   "'Matrix_gen' must be callable")))
class ParamGate(Params, n_params=any):
    """
    Class representing a gate with qubits.
    """

    def __init_subclass__(cls, **kwargs):
        # Bind Matrix_gen to cls
        cls.Matrix_gen = cls.__get_staticvar__('Matrix_gen')

        # Continue
        super().__init_subclass__(**kwargs)

    def __setattr__(self, name, value):
        if name == 'Matrix_gen':
            raise AttributeError("Cannot set 'Matrix_gen'")
        else:
            super().__setattr__(name, value)

    @property
    def Matrix(self) -> np.ndarray:
        if self.params is None:
            raise ValueError("'params' must be provided.")
        return np.asarray(self.Matrix_gen(*self.params))


@staticvars('RMatrix',
            transform=dict(RMatrix=lambda RMatrix: np.asarray(RMatrix)))
class RotationGate(
        ParamGate,
        PowerGate,
        __Base__,
        n_params=1,
        Matrix_gen=lambda self, r: expm(-1j * float(r) * self.RMatrix / 2)):
    """
    Gate with form U = exp(-1j * r / 2 * O), with O an
    arbitrary matrix.
    """

    def __init_subclass__(cls, n_params=None, Matrix_gen=None, **kwargs):
        # Initialize everything
        super().__init_subclass__(n_params=cls._RotationGate__n_params,
                                  Matrix_gen=cls._RotationGate__Matrix_gen,
                                  **kwargs)

    def __print__(self) -> dict[str, tuple[int, str, int]]:
        _params = ''
        if self.params:
            from hybridq.utils import isnumber
            if isnumber(self.params[0]):
                _params = f'φ={np.round(self.params[0]/np.pi, 5)}π'
            else:
                _params = f'φ={self.params[0]}'

        return {'params': (101, _params, 0)}

    def set_params(self,
                   params: iter[any],
                   *,
                   inplace: bool = False) -> RotationGate:
        try:
            # Update parameters multiplying them by self.power
            _g = ParamGate.set_params(
                self, [(p * self.power) % (4 * np.pi) for p in params],
                inplace=inplace)

            # Set power to 1
            _g = PowerGate.set_power(_g, 1, inplace=inplace)

            # Return gate
            return _g
        except:
            return ParamGate.set_params(self, params, inplace=inplace)

    def set_power(self, power: any, *, inplace: bool = False) -> RotationGate:
        try:
            return self.set_params([p * power for p in self.params],
                                   inplace=inplace)
        except:
            return PowerGate.set_power(self, power, inplace=inplace)


class CliffordGate(__Base__):
    pass


@compare('apply', cmp=dict(apply=lambda x, y: x.__code__ == y.__code__))
@staticvars('apply',
            check=dict(apply=(lambda f: callable(f), "'f' must be callable")))
class FunctionalGate(QubitGate, __Base__, n_qubits=any):
    """
    `FunctionalGate` to manipulate state.
    """

    def __init_subclass__(cls, **kwargs):
        # Bind apply to cls
        cls.apply = cls.__get_staticvar__('apply')

        # Continue
        super().__init_subclass__(**kwargs)

    def __setattr__(self, name, value):
        if name == 'apply':
            raise AttributeError("Cannot set 'apply'")
        else:
            super().__setattr__(name, value)

    def __call__(self, psi: np.ndarray, order: iter[any], **kwargs):
        # Check qubits are specified
        if self.qubits is None:
            raise ValueError("'qubits' must be specified.")

        # Convert order to tuple
        order = tuple(order)

        # Check qubits
        if any(q not in order for q in self.qubits):
            raise ValueError(
                "'FunctionalGate' is expecting qubits not in 'order'.")

        # Apply transformation (it should return new_psi, new_order)
        return self.apply(psi, order, **kwargs)


class BaseTupleGate(Tuple):
    """
    Gate defined as a tuple of gates.
    """

    @property
    def qubits(self) -> tuple[any, ...]:
        from hybridq.utils import sort

        # If empty, return empty tuple
        if not len(self):
            return tuple()

        # Get all qubits
        _qubits = tuple(
            g.qubits if g.provides('qubits') else None for g in self)

        # If any None is present, return None
        if any(q is None for q in _qubits):
            return None

        # Flatten list and remove duplicates
        else:
            return tuple(sort(set(y for x in _qubits for y in x)))

    @property
    def n_qubits(self) -> int:
        # Get qubits
        qubits = self.qubits
        return None if qubits is None else len(self.qubits)


def _gate_transform(gates):
    # Get gates
    try:
        l_gates, r_gates = gates
        l_gates = BaseTupleGate(l_gates)
        r_gates = BaseTupleGate(r_gates)
    except:
        raise ValueError("'gates' must be a pair of lists of 'Gate's")

    # Return gates
    return (l_gates, r_gates)


def _s_transform(s):
    # Get np.array. If s == None, set to 1
    s = np.array(1) if s is None else np.asarray(s)

    # If scalar or vector, just return
    if s.ndim <= 1:
        return s

    # If diagonal, just return the diagonal
    elif s.ndim == 2:
        # Return
        return _s_transform(
            np.diag(s)) if s.shape[0] == s.shape[1] and np.allclose(
                s, np.diag(np.diag(s))) else s

    # Raise an implementation error
    else:
        raise NotImplementedError


@compare('gates,s,_conj_rgates')
@staticvars('gates,s,_conj_rgates,_use_cache',
            _conj_rgates=False,
            _use_cache=True,
            transform=dict(gates=_gate_transform,
                           s=_s_transform,
                           _use_cache=lambda x: bool(x)),
            check=dict(s=(lambda s: s is None or 0 <= s.ndim <= 2,
                          "'s' cannot have more than two dimensions.")))
class SchmidtGate(__Base__):

    def __init_subclass__(cls, **kwargs):
        # Get gates and s
        l_gates, r_gates = cls.__get_staticvar__('gates')
        s = cls.__get_staticvar__('s')

        # Get number of gates
        nl = len(l_gates)
        nr = len(r_gates)

        # Initialize error
        _err = False

        # If s is scalar, skip control
        if s.ndim == 0:
            if nr != nl:
                _err = True

        # s is a vector
        elif s.ndim == 1:
            if s.shape[0] != nl or (nr and nr != nl):
                _err = True

        # s is a matrix
        elif s.ndim == 2:
            if s.shape[0] != nl or s.shape[1] != (nr if nr else nl):
                _err = True

        # No other ndim are supported
        else:
            _err = True

        # Raise if there is any error
        if _err:
            raise ValueError("'s' must be consistent with number of 'gates'")

        # Continue
        super().__init_subclass__(**kwargs)

    def __print__(self):
        return {
            'gates':
                (200,
                 f'gates={self.gates if len(self.gates[1]) else self.gates[0]}',
                 0),
            's': (
                400, f's={self.s}' if self.s.ndim == 0 else
                f's={type(self.s).__module__}.{type(self.s).__name__}(shape={self.s.shape}, dtype={self.s.dtype})',
                0)
        }

    def __reduce__(self,
                   *,
                   ignore_sdict: tuple[str, ...] = tuple(),
                   ignore_methods: tuple[str, ...] = tuple(),
                   ignore_keys: tuple[str, ...] = tuple()):
        return super().__reduce__(ignore_sdict=ignore_sdict,
                                  ignore_methods=ignore_methods,
                                  ignore_keys=ignore_keys +
                                  ('_cached_hash', '_cached_Matrix'))

    @property
    def Matrix(self):
        """
        Construct Matrix representing the Map. Order of qubits for `Matrix`
        will be `SchmidtGate.gates[0].qubits + SchmidtGate.gates[1].qubits`,
        even if left and right gates have overlapping qubits.
        """
        from hybridq.gate import TupleGate, MatrixGate, NamedGate
        from scipy.sparse import dok_matrix, diags
        from hybridq.utils import sort

        # Check if a cached value is already present. If yes, return it
        if self._use_cache:
            # Get cached hash
            cached_hash = getattr(self, '_cached_hash', None)

            # Get cached Matrix
            cached_Matrix = getattr(self, '_cached_Matrix', None)

            # Compute new hash
            new_hash = hash(self)

            # Return cached matrix
            if new_hash == cached_hash and cached_Matrix is not None:
                return cached_Matrix

        # Get left and right gates
        l_gates, r_gates = self.gates

        # Cannot build map if qubits are not specified
        if not (l_gates.qubits and r_gates.qubits):
            raise ValueError(
                "Cannot build 'Matrix' if 'qubits' are not specified.")

        # Get order
        order = tuple((0, q) for q in l_gates.qubits) + tuple(
            (1, q) for q in r_gates.qubits)

        # Convert to MatrixGate (to speedup calculation)
        l_gates = TupleGate(
            MatrixGate(U=g.matrix(), qubits=((0, q)
                                             for q in g.qubits))
            for g in l_gates)
        r_gates = TupleGate(
            MatrixGate(U=(g.conj() if self._conj_rgates else g).matrix(),
                       qubits=((1, q) for q in g.qubits)) for g in r_gates)

        # Define how to merge gates
        def _merge(l_g, r_g):
            from hybridq.gate.utils import merge, pad

            # Merge the two gates and pad to the right number of qubits
            return pad(merge(l_g, r_g),
                       qubits=order,
                       order=order,
                       return_matrix_only=True)

        # Get s
        s = diags(
            [self.s] * len(l_gates)).todok() if self.s.ndim == 0 else diags(
                self.s).todok() if self.s.ndim == 1 else dok_matrix(self.s)

        # Merge all gates
        Matrix = np.sum(
            [s * _merge(l_gates[x], r_gates[y]) for (x, y), s in s.items()],
            axis=0)

        # Save cache
        if self._use_cache:
            # Cache hash
            self._cached_hash = new_hash

            # Cache Matrix
            self._cached_Matrix = Matrix

        # Return Matrix
        return Matrix


@requires('sample')
class StochasticGate(__Base__):
    pass


@compare('gate,c_qubits')
@staticvars(
    'gate,c_qubits',
    transform=dict(c_qubits=lambda c_qubits: tuple(c_qubits)),
    check=dict(
        gate=(lambda gate: isinstance(gate, __Base__) and gate.n_qubits > 0,
              f"Not a valid 'gate'.")))
class ControlledGate(__Base__):

    def __init_subclass__(cls, **kwargs):
        # Get gate and controlling qubits
        gate = cls.__get_staticvar__('gate')
        c_qubits = cls.__get_staticvar__('c_qubits')

        # Check gate provides qubits and n_qubits
        if any(not hasattr(gate, w) for w in ('qubits', 'n_qubits')):
            raise ValueError(f"'{type(gate).__name__}' must "
                             "provide 'qubits' and 'n_qubits'.")

        # Few checks
        if gate.n_qubits == 0 or gate.qubits is None:
            raise ValueError("'gate' must provide qubits.")
        if len(c_qubits) == 0:
            raise ValueError("'c_qubits' must non empty.")
        if len(set(c_qubits)) != len(c_qubits):
            raise ValueError("'c_qubits' cannot have repeated qubits.")
        if set(c_qubits).intersection(gate.qubits):
            raise ValueError(
                "Controlled and controlling qubits must be different.")

        # Continue
        super().__init_subclass__(**kwargs)

    @property
    def n_qubits(self):
        return len(self.qubits)

    @property
    def qubits(self):
        return tuple(self.c_qubits) + tuple(self.gate.qubits)

    def __print__(self):
        return {
            'n_qubits': (10, f"n_qubits={self.n_qubits}", 0),
            'c_qubits': (99, f"c_qubits={_truncate_print(self.c_qubits)}", 0),
            'qubits': (100, f"qubits={_truncate_print(self.qubits)}", 0),
            'gate': (101, f'gate={self.gate}', 0),
        }
