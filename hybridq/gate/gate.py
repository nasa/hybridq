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
from scipy.linalg import fractional_matrix_power as powm
from scipy.linalg import expm, sqrtm
from collections import defaultdict
import hybridq.gate.property as pr
from inspect import signature
from copy import deepcopy
from warnings import warn
import numpy as np


class BaseGate(pr.__Base__):
    """
    Common type for all gates.
    """
    pass


# Define docstrings
_gate_docstrings = defaultdict(
    lambda x: "", {
        'I':
            "Identity operator (n_qubits=any).",
        'H':
            "Hadamard operator (n_qubits=1). ",
        'X':
            "X Pauli matrix (n_qubits=1).",
        'Y':
            "Y Pauli matrix (n_qubits=1).",
        'Z':
            "Z Pauli matrix (n_qubits=1).",
        'U3':
            """
Arbitrary single qubit unitary gate (n_qubits=1, n_params=3). It accepts three
angles in radians and it is equivalent to the following operator:

U3(t, p, l) = exp(i (p + l) / 2) * RZ(p) RY(t) RZ(l),

with RY and RZ being rotations along the Y and Z axes.
""",
        'R_PI_2':
            """
Rotation in the X-Y plane (n_qubits=1, n_params=2). It accepts one angle in
radians and it is equivalent to the following operator:

R_PI_2(phi) = RZ(phi) RX(pi / 2) RZ(-phi),

with RX and RY being rotations along the X and Z axes.
""",
        'ZZ':
            """
The gate apply the Z Pauli operator to both the first and the second qubit
(n_qubits=2).
""",
        'CZ':
            "Controlled-Z gate (n_qubits=1).",
        'CX':
            "Controlled-X gate (n_qubits=1).",
        'SWAP':
            "Swap two qubits (n_qubits=2).",
        'ISWAP':
            """
Swap two qubits and apply a phase 'exp(i pi/2)' if either qubits (but not both)
is 1 (n_qubits=1).
""",
        'CPHASE':
            """
A phase 'exp(i phi)' is applied to the second qubit if the first qubit is 1
(n_qubits=1, n_params=1). It accepts an angle in radians.
""",
        'FSIM':
            """
Unitary gate implemented in the Sycamore@Google QPU [Nature 574.7779 (2019)]
(n_qubits=2, n_params=2). It accepts two angles, 't' and 'p', the first one in
unit of 'pi / 2' while the second in radians, and it is equivalent to the
following operator:

FSIM(t, p) = CPHASE(-p) ISWAP ** (-2 t / pi).
""",
        'RX':
            """
Rotation gate along the X-axis defined as 'exp(-i phi/2 X)', with X being the X
Pauli operator (n_qubits=1, n_params=1). It accepts an angle in radians.
""",
        'RY':
            """
Rotation gate along the Y-axis defined as 'exp(-i phi/2 Y)', with Y being the Y
Pauli operator (n_qubits=1, n_params=1). It accepts an angle in radians.
""",
        'RZ':
            """
Rotation gate along the Z-axis defined as 'exp(-i phi/2 Z)', with Z being the Z
Pauli operator (n_qubits=1, n_params=1). It accepts an angle in radians.
""",
        'SQRT_X':
            "Square root of X gate (n_qubits=1).",
        'SQRT_Y':
            "Square root of Y gate (n_qubits=1).",
        'P':
            "Phase gate defined as RZ(pi) (n_qubits=1).",
        'T':
            "Phase gate defined as RZ(pi / 2) (n_qubits=1).",
        'SQRT_SWAP':
            "Square root of SWAP gate (n_qubits=1).",
        'SQRT_ISWAP':
            "Square root of ISWAP gate (n_qubits=1).",
    })

# Define available gates
_available_gates = {
    'I':
        dict(mro=(pr.CliffordGate, pr.FunctionalGate, pr.ParamGate,
                  pr.SelfAdjointUnitaryGate, pr.QubitGate, pr.TagGate,
                  pr.NameGate),
             methods=dict(power=property(lambda self: 1),
                          _set_power=lambda self, power: None,
                          set_power=lambda self, power, inplace=False: self
                          if inplace else deepcopy(self)),
             static_dict=dict(n_qubits=any,
                              n_params=0,
                              docstring=_gate_docstrings['I'],
                              Matrix_gen=lambda self: np.eye(2**self.n_qubits),
                              apply=lambda self, psi, order: (psi, order))),
    'H':
        dict(mro=(pr.CliffordGate, pr.MatrixGate, pr.SelfAdjointUnitaryGate,
                  pr.QubitGate, pr.TagGate, pr.NameGate),
             methods={},
             static_dict=dict(n_qubits=1,
                              docstring=_gate_docstrings['H'],
                              Matrix=np.array([[1, 1], [1, -1]]) / np.sqrt(2))),
    'X':
        dict(mro=(pr.CliffordGate, pr.MatrixGate, pr.SelfAdjointUnitaryGate,
                  pr.QubitGate, pr.TagGate, pr.NameGate),
             methods={},
             static_dict=dict(n_qubits=1,
                              docstring=_gate_docstrings['X'],
                              Matrix=np.array([[0, 1], [1, 0]]))),
    'Y':
        dict(mro=(pr.CliffordGate, pr.MatrixGate, pr.SelfAdjointUnitaryGate,
                  pr.QubitGate, pr.TagGate, pr.NameGate),
             methods={},
             static_dict=dict(n_qubits=1,
                              docstring=_gate_docstrings['Y'],
                              Matrix=np.array([[0, -1j], [1j, 0]]))),
    'Z':
        dict(mro=(pr.CliffordGate, pr.MatrixGate, pr.SelfAdjointUnitaryGate,
                  pr.QubitGate, pr.TagGate, pr.NameGate),
             methods={},
             static_dict=dict(n_qubits=1,
                              docstring=_gate_docstrings['Z'],
                              Matrix=np.array([[1, 0], [0, -1]]))),
    'U3':
        dict(
            mro=(pr.ParamGate, pr.UnitaryGate, pr.QubitGate, pr.TagGate,
                 pr.NameGate),
            methods={},
            static_dict=dict(
                n_qubits=1,
                n_params=3,
                docstring=_gate_docstrings['U3'],
                Matrix_gen=lambda self, t, p, l: np.array([[
                    np.cos(float(t) / 2), -np.exp(1j * float(l)) * np.sin(
                        float(t) / 2)
                ],
                                                           [
                                                               np.exp(1j *
                                                                      float(p))
                                                               * np.sin(
                                                                   float(t) / 2
                                                               ),
                                                               np.exp(1j * (
                                                                   float(l) +
                                                                   float(p))) *
                                                               np.cos(
                                                                   float(t) / 2)
                                                           ]]))),
    'R_PI_2':
        dict(mro=(pr.ParamGate, pr.UnitaryGate, pr.QubitGate, pr.TagGate,
                  pr.NameGate),
             methods={},
             static_dict=dict(
                 n_qubits=1,
                 n_params=1,
                 docstring=_gate_docstrings['R_PI_2'],
                 Matrix_gen=lambda self, phi: np.array([[
                     1, -1j * np.exp(-1j * float(phi))
                 ], [-1j * np.exp(1j * float(phi)), 1]]) / np.sqrt(2))),
    'ZZ':
        dict(mro=(pr.CliffordGate, pr.MatrixGate, pr.SelfAdjointUnitaryGate,
                  pr.QubitGate, pr.TagGate, pr.NameGate),
             methods={},
             static_dict=dict(n_qubits=2,
                              docstring=_gate_docstrings['ZZ'],
                              Matrix=np.array([[1, 0, 0, 0], [0, -1, 0, 0],
                                               [0, 0, -1, 0], [0, 0, 0, 1]]))),
    'CZ':
        dict(mro=(pr.CliffordGate, pr.MatrixGate, pr.SelfAdjointUnitaryGate,
                  pr.QubitGate, pr.TagGate, pr.NameGate),
             methods={},
             static_dict=dict(n_qubits=2,
                              docstring=_gate_docstrings['CZ'],
                              Matrix=np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                                               [0, 0, 1, 0], [0, 0, 0, -1]]))),
    'CX':
        dict(mro=(pr.CliffordGate, pr.MatrixGate, pr.SelfAdjointUnitaryGate,
                  pr.QubitGate, pr.TagGate, pr.NameGate),
             methods={},
             static_dict=dict(n_qubits=2,
                              docstring=_gate_docstrings['CX'],
                              Matrix=np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                                               [0, 0, 0, 1], [0, 0, 1, 0]]))),
    'SWAP':
        dict(mro=(pr.CliffordGate, pr.MatrixGate, pr.SelfAdjointUnitaryGate,
                  pr.QubitGate, pr.TagGate, pr.NameGate),
             methods={},
             static_dict=dict(n_qubits=2,
                              docstring=_gate_docstrings['SWAP'],
                              Matrix=np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                                               [0, 1, 0, 0], [0, 0, 0, 1]]))),
    'ISWAP':
        dict(mro=(pr.CliffordGate, pr.MatrixGate, pr.UnitaryGate, pr.QubitGate,
                  pr.TagGate, pr.NameGate),
             methods={},
             static_dict=dict(n_qubits=2,
                              docstring=_gate_docstrings['ISWAP'],
                              Matrix=np.array([[1, 0, 0, 0], [0, 0, 1j, 0],
                                               [0, 1j, 0, 0], [0, 0, 0, 1]]))),
    'CPHASE':
        dict(mro=(pr.ParamGate, pr.UnitaryGate, pr.QubitGate, pr.TagGate,
                  pr.NameGate),
             methods={},
             static_dict=dict(n_qubits=2,
                              n_params=1,
                              docstring=_gate_docstrings['CPHASE'],
                              Matrix_gen=lambda self, p: np.array(
                                  [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                                   [0, 0, 0, np.exp(1j * float(p))]]))),
    'FSIM':
        dict(mro=(pr.ParamGate, pr.UnitaryGate, pr.QubitGate, pr.TagGate,
                  pr.NameGate),
             methods={},
             static_dict=dict(
                 n_qubits=2,
                 n_params=2,
                 docstring=_gate_docstrings['FSIM'],
                 Matrix_gen=lambda self, t, p: np.array(
                     [[1, 0, 0, 0],
                      [0, np.cos(float(t)), -1j * np.sin(float(t)), 0],
                      [0, -1j * np.sin(float(t)),
                       np.cos(float(t)), 0], [0, 0, 0,
                                              np.exp(-1j * float(p))]]))),
}

# Update available gates
_available_gates.update({
    'RX':
        dict(mro=(pr.RotationGate, pr.UnitaryGate, pr.QubitGate, pr.TagGate,
                  pr.NameGate),
             methods={},
             static_dict=dict(
                 n_qubits=1,
                 docstring=_gate_docstrings['RX'],
                 RMatrix=_available_gates['X']['static_dict']['Matrix'])),
    'RY':
        dict(mro=(pr.RotationGate, pr.UnitaryGate, pr.QubitGate, pr.TagGate,
                  pr.NameGate),
             methods={},
             static_dict=dict(
                 n_qubits=1,
                 docstring=_gate_docstrings['RY'],
                 RMatrix=_available_gates['Y']['static_dict']['Matrix'])),
    'RZ':
        dict(mro=(pr.RotationGate, pr.UnitaryGate, pr.QubitGate, pr.TagGate,
                  pr.NameGate),
             methods={},
             static_dict=dict(
                 n_qubits=1,
                 docstring=_gate_docstrings['RZ'],
                 RMatrix=_available_gates['Z']['static_dict']['Matrix'])),
    'SQRT_X':
        dict(mro=(pr.CliffordGate, pr.MatrixGate, pr.UnitaryGate, pr.QubitGate,
                  pr.TagGate, pr.NameGate),
             methods={},
             static_dict=dict(
                 n_qubits=1,
                 docstring=_gate_docstrings['SQRT_X'],
                 Matrix=sqrtm(_available_gates['X']['static_dict']['Matrix']))),
    'SQRT_Y':
        dict(mro=(pr.CliffordGate, pr.MatrixGate, pr.UnitaryGate, pr.QubitGate,
                  pr.TagGate, pr.NameGate),
             methods={},
             static_dict=dict(
                 n_qubits=1,
                 docstring=_gate_docstrings['SQRT_Y'],
                 Matrix=sqrtm(_available_gates['Y']['static_dict']['Matrix']))),
    'P':
        dict(mro=(pr.CliffordGate, pr.MatrixGate, pr.UnitaryGate, pr.QubitGate,
                  pr.TagGate, pr.NameGate),
             methods={},
             static_dict=dict(
                 n_qubits=1,
                 docstring=_gate_docstrings['P'],
                 Matrix=sqrtm(_available_gates['Z']['static_dict']['Matrix']))),
    'T':
        dict(mro=(pr.MatrixGate, pr.UnitaryGate, pr.QubitGate, pr.TagGate,
                  pr.NameGate),
             methods={},
             static_dict=dict(
                 n_qubits=1,
                 docstring=_gate_docstrings['T'],
                 Matrix=powm(_available_gates['Z']['static_dict']['Matrix'],
                             0.25))),
    'SQRT_SWAP':
        dict(mro=(pr.MatrixGate, pr.UnitaryGate, pr.QubitGate, pr.TagGate,
                  pr.NameGate),
             methods={},
             static_dict=dict(
                 n_qubits=2,
                 docstring=_gate_docstrings['SQRT_SWAP'],
                 Matrix=sqrtm(
                     _available_gates['SWAP']['static_dict']['Matrix']))),
    'SQRT_ISWAP':
        dict(mro=(pr.MatrixGate, pr.UnitaryGate, pr.QubitGate, pr.TagGate,
                  pr.NameGate),
             methods={},
             static_dict=dict(
                 n_qubits=2,
                 docstring=_gate_docstrings['SQRT_ISWAP'],
                 Matrix=sqrtm(
                     _available_gates['ISWAP']['static_dict']['Matrix']))),
})

# Define gate aliases
_gate_aliases = {
    'ID': 'I',
    'S': 'P',
    'Z_1_2': 'P',
    'SQRT_Z': 'P',
    'CNOT': 'CX',
    'X_1_2': 'SQRT_X',
    'Y_1_2': 'SQRT_Y',
    'FS': 'FSIM',
    'STOC': 'STOCHASTIC',
    'FUN': 'FUNCTIONAL',
    'FN': 'FUNCTIONAL',
    'PROJ': 'PROJECTION',
    'MEAS': 'MEASURE'
}


def Gate(name: str,
         qubits: iter[any] = None,
         params: iter[any] = None,
         n_qubits: int = None,
         power: any = 1,
         tags: dict[any, any] = None,
         **kwargs) -> Gate:
    """
    Generator of gates.

    Parameters
    ----------
    name: str
        Name of `Gate`.
    qubits: iter[any], optional
        List of qubits `Gate` is acting on.
    params: iter[any], optional
        List of parameters to define `Gate`.
    n_qubits: int, optional
        Specify the number of qubits if `qubits` is unspecified.
    power: float, optional
        The power the matrix of `Gate` is elevated to.
    tags: dict[any, any], optional
        Dictionary of tags.

    See Also
    --------
    NamedGate, MatrixGate, TupleGate, StochasticGate, FunctionalGate,
    Projection, Measure

    Returns
    -------
    Gate
    """
    # Convert name to upper
    name = name.upper()

    # Check if it is an alias
    if name in _gate_aliases:
        warn(f"'{name}' is an alias for '{_gate_aliases[name]}'. "
             "Using Gate(name='{_gate_aliases[name]}').")
        name = _gate_aliases[name]

    # Helper for not supported options
    def _not_supported(**kwargs):
        for k, v in kwargs.items():
            # Initialize error
            _err = False

            # Check for any error
            if k == 'power':
                _err |= v != 1
            else:
                _err |= v is not None

            # If error, raise
            if _err:
                raise NotImplementedError(f"'{k}' not supported for '{name}'.")

    # Generate a NamedGate
    if name in _available_gates:
        return NamedGate(name=name,
                         qubits=qubits,
                         params=params,
                         n_qubits=n_qubits,
                         power=power,
                         tags=tags)

    # Generate a MatrixGate
    elif name == 'MATRIX':
        _not_supported(params=params)
        return MatrixGate(qubits=qubits,
                          n_qubits=n_qubits,
                          power=power,
                          tags=tags,
                          **kwargs)

    # Generate a StochasticGate
    elif name == 'STOCHASTIC':
        _not_supported(qubits=qubits,
                       n_qubits=n_qubits,
                       power=power,
                       params=params)
        return StochasticGate(tags=tags, **kwargs)

    # Generate a FunctionalGate
    elif name == 'FUNCTIONAL':
        _not_supported(power=power, params=params)
        return FunctionalGate(qubits=qubits,
                              n_qubits=n_qubits,
                              tags=tags,
                              **kwargs)

    # Generate a TupleGate
    elif name == 'TUPLE':
        _not_supported(qubits=qubits,
                       n_qubits=n_qubits,
                       power=power,
                       params=params)
        return TupleGate(tags=tags, **kwargs)

    # Generate a SchmidtGate
    elif name == 'SCHMIDT':
        _not_supported(qubits=qubits,
                       n_qubits=n_qubits,
                       power=power,
                       params=params)
        return SchmidtGate(tags=tags, **kwargs)

    # Generate a ProjectionGate
    elif name == 'PROJECTION':
        from hybridq.gate.projection import Projection
        _not_supported(n_qubits=n_qubits, power=power, params=params)
        return Projection(qubits=qubits, tags=tags, **kwargs)

    # Generate a MeasureGate
    elif name == 'MEASURE':
        from hybridq.gate.measure import Measure
        _not_supported(power=power, params=params)
        return Measure(qubits=qubits, n_qubits=n_qubits, tags=tags, **kwargs)

    # Generate a ControlledGate
    elif name == 'CGATE':
        _not_supported(params=params, qubits=qubits, n_qubits=n_qubits)
        return Control(power=power, tags=tags, **kwargs)

    # Return implementation error
    else:
        raise NotImplementedError(f"'{name}' gate is not implemented.")


def NamedGate(name: str,
              qubits: iter[any] = None,
              params: iter[any] = None,
              n_qubits: int = None,
              power: any = 1,
              tags: dict[any, any] = None) -> NamedGate:
    """
    Generate named gates.

    Parameters
    ----------
    name: str
        Name of `Gate`.
    qubits: iter[any], optional
        List of qubits `Gate` is acting on.
    params: iter[any], optional
        List of parameters to define `Gate`.
    n_qubits: int, optional
        Specify the number of qubits if `qubits` is unspecified.
    power: float, optional
        The power the matrix of `Gate` is elevated to.
    tags: dict[any, any], optional
        Dictionary of tags.

    Returns
    -------
    NamedGate
    """

    # Convert name to upper
    name = name.upper()

    # Check if it is an alias
    if name in _gate_aliases:
        warn(f"'{name}' is an alias for '{_gate_aliases[name]}'. "
             "Using Gate(name='{_gate_aliases[name]}').")
        name = _gate_aliases[name]

    # Get mro, static dict and methods
    _sd = _available_gates[name].get('static_dict', {}).copy()
    _mro = _available_gates[name].get('mro', tuple())
    _methods = _available_gates[name].get('methods', None)

    # Convert qubits to tuple
    qubits = None if qubits is None else tuple(qubits)

    # Convert n_qubits to int
    n_qubits = None if n_qubits is None else int(n_qubits)

    # Get n_qubits from qubits if possible
    if n_qubits is None and qubits is not None:
        n_qubits = len(qubits)

    # If n_qubits is still None, let's use the static value
    if n_qubits is None and _sd.get('n_qubits', None) is not None:
        # Set default if any
        if _sd['n_qubits'] is any:
            # Set n_qubits to 1
            n_qubits = 1

            # Warn
            warn("Using default number of qubits 'n_qubits=1'.")

        # Otherwise, use the right number of qubits
        else:
            n_qubits = _sd['n_qubits']

    # At this point, if n_qubits is still None, something went wrong ..
    assert (n_qubits is not None)

    # Set static var
    if _sd.get('n_qubits', None) is any:
        _sd['n_qubits'] = n_qubits

    # Check that everything is consistent
    if qubits is not None and len(qubits) != n_qubits:
        raise ValueError(
            "'n_qubits' is not consistent with the number of 'qubits'.")

    # Check that everything is consistent
    if 'n_qubits' in _sd and _sd['n_qubits'] != n_qubits:
        raise ValueError("Wrong number of qubits "
                         f"(expected {_sd['n_qubits']}, got {n_qubits}).")

    # Get locals
    _locals = locals().copy()
    _locals = {
        k: v for k, v in ((k, _locals.get(k, None))
                          for k in ('qubits', 'params', 'power',
                                    'tags')) if v is not None
    }

    # Return gate
    return pr.generate(class_name='Gate_' + str(name),
                       mro=(BaseGate, pr.DocString) + _mro,
                       methods=_methods,
                       name=name,
                       **_sd)(**_locals)


def MatrixGate(U: np.ndarray,
               qubits: iter[any] = None,
               n_qubits: int = None,
               power: any = 1,
               tags: dict[any, any] = None,
               copy: bool = True,
               check_if_unitary: bool = True,
               atol: float = 1e-8) -> MatrixGate:
    """
    Generate matrix gates.

    Parameters
    ----------
    U: list[list[any]], optional
        The matrix representing the matrix gate.
    qubits: iter[any], optional
        List of qubits `Gate` is acting on.
    n_qubits: int, optional
        Specify the number of qubits if `qubits` is unspecified.
    power: float, optional
        The power the matrix of `Gate` is elevated to.
    tags: dict[any, any], optional
        Dictionary of tags.
    copy: bool, optional
        A copy of `U` is used instead of a reference if `copy` is True
        (default: True).
    check_if_unitary: bool, optional
        Check if `U` is unitary and use `UnitaryGate` instead of
        `PowerMatrixGate` accordingly.
    atol: float, optional
        Use `atol` as absolute precision for checks.

    Returns
    -------
    MatrixGate
    """

    # Get U
    U = (np.array if copy else np.asarray)(U)

    # Check U is well-formed
    if U.ndim != 2 or U.shape[0] != U.shape[1] or np.log2(U.shape[0]) % 1:
        raise ValueError("'U' must be a quadratic matrix "
                         "with the linear size being a power of 2.")

    # Get number of qubits from U if not provided
    n_qubits = int(np.log2(U.shape[0])) if n_qubits is None else int(n_qubits)

    # Get qubits and n_qubits
    qubits = None if qubits is None else tuple(qubits)

    # Check consistency
    if qubits and len(qubits) != n_qubits:
        raise ValueError(
            "Number of 'qubits' is not consistent with 'n_qubits'.")

    # Check U is well-formed
    if U.ndim != 2 or U.shape[0] != U.shape[1] or U.shape[0] != 2**n_qubits:
        raise ValueError("'U' is not consistent with the number of qubits.")

    # Check if unitary
    _unitary = False
    if check_if_unitary:
        _A = U @ U.conj().T
        _B = U.conj().T @ U
        _unitary = np.allclose(_A, _B, atol=atol) and np.allclose(
            _A, np.eye(_A.shape[0]), atol=atol)

    # Return Gate
    return pr.generate('MatrixGate',
                       (BaseGate, pr.MatrixGate,
                        pr.UnitaryGate if _unitary else pr.PowerMatrixGate,
                        pr.QubitGate, pr.TagGate, pr.NameGate),
                       Matrix=U,
                       name='MATRIX',
                       n_qubits=n_qubits)(qubits=qubits, power=power, tags=tags)


def TupleGate(gates: iter[BaseGate] = tuple(),
              tags: dict[any, any] = None) -> TupleGate:
    """
    Generate a tuple gate.

    Parameters
    ----------
    gates: iter[BaseGate]
        `gates` used to initialize the `TupleGate`.
    tags: dict[any, any], optional
        Dictionary of tags.

    Returns
    -------
    TupleGate
    """

    # Return gate
    return pr.generate('TupleGate', (BaseGate, pr.BaseTupleGate, pr.NameGate),
                       name='TUPLE',
                       _base_check={all: [BaseGate]})(gates, tags=tags)


def FunctionalGate(f: callable,
                   qubits: iter[any] = None,
                   n_qubits: int = None,
                   tags: dict[any, any] = None) -> FunctionalGate:
    """
    Generator of gates.

    Parameters
    ----------
    f: callable[self, psi={np.ndarray, tuple[np.ndarray, np.ndarray]}, order=list[any]], optional
        Function used to manipulate the quantum state.  `f` must be a `callable`
        function which accepts three parameters: `self`, the gate being called,
        `psi`, the quantum state, and `order`, the order of qubits in the
        quantum state. `psi` can either be a single array of complex numbers,
        or a `tuple` of two real-valued array representing the real and
        imaginary part of `psi` respectively. `order` is the ordered list of
        qubits in `psi`. Finally, `f` must change `psi` in place and return the
        new `order`.
    qubits: iter[any], optional
        List of qubits `Gate` is acting on.
    n_qubits: int, optional
        Specify the number of qubits if `qubits` is unspecified.
    tags: dict[any, any], optional
        Dictionary of tags.

    Returns
    -------
    FunctionalGate
    """

    # Either n_qubits or qubits must be provided
    if qubits is None and n_qubits is None:
        raise ValueError("Either 'qubits' or 'n_qubits' must be provided.")

    # Get qubits and n_qubits
    qubits = None if qubits is None else tuple(qubits)
    n_qubits = len(qubits) if n_qubits is None else int(n_qubits)

    # Check consistency
    if qubits and len(qubits) != n_qubits:
        raise ValueError(
            "Number of 'qubits' is not consistent with 'n_qubits'.")

    # Return gate
    return pr.generate(
        'FunctionalGate',
        (BaseGate, pr.FunctionalGate, pr.TagGate, pr.NameGate),
        name='FUNCTIONAL',
        n_qubits=n_qubits,
        apply=f,
    )(qubits=qubits, tags=tags)


@pr.staticvars(
    'p,gates',
    transform=dict(gates=lambda gates: tuple(gates)),
    check=dict(gates=lambda gates: all(isinstance(g, BaseGate) for g in gates)))
class _StochasticGate(BaseGate, pr.StochasticGate):

    def sample(self):
        raise NotImplementedError()


def StochasticGate(gates: iter[BaseGate],
                   p: iter[float],
                   tags: dict[any, any] = None) -> StochasticGate:
    """
    Generator of gates.

    Parameters
    ----------
    name: str
        Name of `Gate`.
    gates: iter[BaseGate]
        If `name` is `STOCHASTIC`, `gates` are used to initialize
        `Gate('STOCHASTIC')`.
    p: iter[float]
        If `name` is `STOCHASTIC`, `p` will be used as probabilities to sample
        from `Gate('STOCHASTIC')`.

    See Also
    --------
    TupleGate, TagGate, NameGate
    """

    # Get gates and probabilities
    gates = tuple(gates)
    p = tuple(p)

    # Check that sizes are compatible
    if len(p) != len(gates):
        raise ValueError(f"Too {'few' if len(p) < len(gates) else 'many'} "
                         f"probabilities (got {len(p)}, expected {len(gates)})")

    # Check probabilities are normalized
    if not np.isclose(np.sum(p), 1, atol=1e-5):
        raise ValueError(f"Probabilities must be normalized")

    # Define sample method
    def __sample__(self, size: int = None, replace: bool = True):
        """
        Sample accordingly to the provided `sample_fun`.
        """
        # Get indexes
        idxs = np.random.choice(len(self.gates),
                                size=size,
                                replace=replace,
                                p=self.p)

        # Return gates
        return self.gates[idxs] if size is None else [
            self.gates[x] for x in idxs
        ]

    # Define print method
    def __print__(self) -> dict[str, tuple[int, str, int]]:
        return {
            'p': (
                401,
                f"p={pr._truncate_print(tuple(np.round(p, 5) for p in self.p))}"
                if self.p is not None else "", 0),
        }

    # Define qubits and n_qubits
    def __qubits__(self):
        return self.gates.qubits

    def __n_qubits__(self):
        return self.gates.n_qubits

    # Return Gate
    return pr.generate('StochasticGate',
                       (_StochasticGate, pr.TagGate, pr.NameGate),
                       name='STOCHASTIC',
                       gates=gates,
                       p=p,
                       methods=dict(sample=__sample__,
                                    __print__=__print__,
                                    qubits=property(__qubits__),
                                    n_qubits=property(__n_qubits__)))(tags=tags)


def SchmidtGate(gates: {iter[Gate], tuple[iter[Gate], iter[Gate]]},
                s=None,
                tags: dict[any, any] = None,
                copy: bool = True,
                use_cache: bool = True) -> SchmidtGate:
    """
    Return a SchmidtGate.

    Parameters
    ----------
    gates: tuple[iter[Gate], iter[Gate]]
        Pair of lists of `Gate`s. `Gate`s must provide `qubits` and `matrix`
        and cannot have common qubits.
    s: np.ndarray
        Correlation matrix between `Gates`. More precisely,
        `SchmidtGate.Matrix` is built as follows:
        ```
        U = \sum_{ij} s_ij L_i R_j.
        ```
        `s` can be a single scalar, a vector or a matrix consistent with the
        number of gates.
    tags: dict[any, any], optional
        Dictionary of tags.
    copy: bool, optional
        If `True`, a copy of `gates` and `s` is used instead of a reference
        (default: `True`).
    use_cache: bool, optional
        If `True`, extra memory is used to store a cached `Matrix`.

    Returns
    -------
    SchmidtGate
    """

    # Copy if needed
    def _copy(x: iter[any, ...]):
        from copy import deepcopy
        return (deepcopy(y) for y in x) if copy else x

    # Get s
    s = None if s is None else (np.array if copy else np.asarray)(s)

    # Get left/right gates
    try:
        l_gates, r_gates = gates
        l_gates = TupleGate(_copy(l_gates))
        r_gates = TupleGate(_copy(r_gates))

        # Check all gates have qubits and matrix
        if any(
                any(not g.provides('qubits,matrix') or g.qubits is None
                    for g in gs)
                for gs in [l_gates, r_gates]):
            raise
    except:
        raise ValueError("'gates' must be a pair of lists of valid 'Gate's, "
                         "all providing 'qubits' and 'matrix'.")

    # No qubits should be shared
    if set(l_gates.qubits).intersection(r_gates.qubits):
        raise ValueError("Left and right gates should not share any qubits.")

    # Define qubits and n_qubits method
    def __qubits__(self):
        return self.gates[0].qubits + self.gates[1].qubits

    def __n_qubits__(self):
        return self.gates[0].n_qubits + self.gates[1].n_qubits

    # Return SchmidtGate
    return pr.generate(
        'SchmidtGate',
        (BaseGate, pr.SchmidtGate, pr.PowerMatrixGate, pr.TagGate, pr.NameGate),
        gates=(l_gates, r_gates),
        s=s,
        _use_cache=use_cache,
        methods=dict(qubits=property(__qubits__),
                     n_qubits=property(__n_qubits__)),
        name='SCHMIDT')(tags=tags)


def Control(c_qubits: iter[any],
            gate: BaseGate,
            power: any = 1,
            tags: dict[any, any] = None,
            copy: bool = True):
    """
    Generate a controlled `Gate`.

    Parameters
    ----------
    c_qubits: iter[any]
        List of controlling qubits.
    gate: BaseGate
        Gate to control.
    power: float, optional
        The power the matrix of `Gate` is elevated to.
    tags: dict[any, any], optional
        Dictionary of tags.
    copy: bool, optional
        A copy of `gate` is used instead of a reference if `copy` is True
        (default: True).

    Returns
    -------
    MatrixGate
    """

    # Only BaseGate can be used here
    if not isinstance(gate, BaseGate):
        raise ValueError(
            f"Controlled '{type(gate).__name__}' is not supported.")

    # Get name for Gate
    name = type(gate).__name__.replace('Gate_', 'Gate_C') if type(
        gate).__name__[:5] == 'Gate_' else 'Controlled_' + type(gate).__name__
    gate_name = ('C' * len(c_qubits)
                 if len(c_qubits) <= 3 else f'C^{len(c_qubits)}-') + gate.name

    # Return the correct gate
    if isinstance(gate, pr.FunctionalGate):
        # Define apply
        def apply(self, psi, order):
            from hybridq.gate.projection import Projection

            # Given the state, project state
            _proj, _order = Projection('1' * len(self.c_qubits),
                                       self.c_qubits).apply(psi,
                                                            order=order,
                                                            renormalize=False)

            # Check order hasn't changed
            assert (_order == order)

            # Remove projection from psi
            psi -= _proj

            # Apply gate to projection
            _proj, _order = self.gate.apply(psi=_proj, order=order)

            # Check order hasn't changed (this can be relaxed if needed)
            if _order != order:
                raise NotImplementedError("Not yet implemented.")

            # Add projection to psi
            psi += _proj

            # Return projection and order
            return psi, order

        # Return Gate
        return pr.generate(name, (BaseGate, pr.ControlledGate,
                                  pr.FunctionalGate, pr.NameGate, pr.TagGate),
                           gate=gate,
                           name=gate_name,
                           apply=apply,
                           n_qubits=any,
                           c_qubits=c_qubits)(tags=tags)

    elif isinstance(gate, pr.StochasticGate):
        # Define apply
        def apply(self, psi, order):
            # Given the state, project state
            _proj, _order = Projection('1' * len(self.c_qubits),
                                       c_qubits).apply(psi,
                                                       order=order,
                                                       renormalize=False)

            # Check order hasn't changed
            assert (_order == order)

            # Sample gate
            gate = self.sample()

            # Apply matrix to projection and update state
            psi += dot(gate.matrix() - np.eye(2**gate.n_qubits),
                       _proj,
                       axes_b=[order.index(q) for q in gate.qubits],
                       inplace=True)

            # Return projection and order
            return psi, order

        # Return Gate
        return pr.generate(name, (BaseGate, pr.ControlledGate,
                                  pr.FunctionalGate, pr.NameGate, pr.TagGate),
                           gate=gate,
                           name=gate_name,
                           apply=apply,
                           n_qubits=any,
                           c_qubits=c_qubits)(tags=tags)

    elif isinstance(gate, pr.UnitaryGate) or isinstance(gate,
                                                        pr.PowerMatrixGate):
        # Define Matrix
        @property
        def Matrix(self):
            # Get matrix of gate
            Ug = self.gate.matrix()

            # Generate full matrix
            U = np.eye(2**self.n_qubits, dtype=Ug.dtype)

            # Update matrix
            U[-Ug.shape[0]:, -Ug.shape[1]:] = Ug

            # Return matrix
            return U

        # Return Gate
        return pr.generate(name,
                           (BaseGate, pr.ControlledGate,
                            pr.UnitaryGate if isinstance(gate, pr.UnitaryGate)
                            else pr.PowerMatrixGate, pr.NameGate, pr.TagGate),
                           gate=gate,
                           name=gate_name,
                           methods=dict(Matrix=Matrix),
                           c_qubits=c_qubits)(power=power, tags=tags)

    else:
        raise NotImplementedError(
            f"Controlled '{type(gate).__name__}' not implemented.")
