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
from hybridq.gate import property as pr
from hybridq.dm.gate import property as dm_pr
import numpy as np


class BaseSuperGate(dm_pr.__Base__):
    """
    Common type for all gates.
    """
    pass


def TupleSuperGate(gates: iter[{BaseGate, BaseSuperGate}] = tuple(),
                   tags: dict[any, any] = None) -> TupleSuperGate:
    """
    Generate a tuple gate.

    Parameters
    ----------
    gates: iter[{BaseGate, BaseSuperGate}]
        `gates` used to initialize the `TupleGate`.
    tags: dict[any, any], optional
        Dictionary of tags.

    Returns
    -------
    TupleSuperGate
    """
    from hybridq.gate.gate import BaseGate

    # Return gate
    return pr.generate('TupleSuperGate',
                       (BaseSuperGate, dm_pr.BaseTupleSuperGate, pr.NameGate),
                       name='STUPLE',
                       _base_check={any: [BaseGate, BaseSuperGate]})(gates,
                                                                     tags=tags)


@dm_pr.staticvars('l_qubits,r_qubits')
class _MatrixSuperGate(BaseSuperGate):
    pass


def MatrixSuperGate(Map: np.ndarray,
                    l_qubits: iter[any],
                    r_qubits: iter[any] = None,
                    tags: dict[any, any] = None,
                    copy: bool = True) -> MatrixSuperGate:
    """
    Return a MatrixSuperGate.

    Parameters
    ----------
    Map: np.ndarray
        Map representing the `SuperGate`.
    l_qubits, r_qubits: iter[any], iter[any]
        Left (right respectively) qubits for the `Map`. If r_qubits is not
        provided, `r_qubits` is assumed to be equal to `l_qubits`.
    tags: dict[any, any], optional
        Dictionary of tags.
    copy: bool, optional
        A copy of `Map` is used instead of a reference if `copy` is True
        (default: True).

    Returns
    -------
    MatrixSuperGate
    """

    def __print_qubits__(self):
        return {
            'l_qubits':
                (100, f'l_qubits={pr._truncate_print(self.qubits[0])}', 0),
            'r_qubits':
                (101, f'r_qubits={pr._truncate_print(self.qubits[1])}', 0),
        }

    # Get Map
    Map = (np.array if copy else np.asarray)(Map)

    # Get qubits
    l_qubits = tuple(l_qubits)
    r_qubits = l_qubits if r_qubits is None else tuple(r_qubits)
    n_qubits = len(l_qubits) + len(r_qubits)

    # Check consistency
    if Map.shape != (2**n_qubits, 2**n_qubits):
        raise ValueError(
            "'Map' must be consistent with the total number of qubits.")

    # Get MatrixSuperGate
    return dm_pr.generate(
        "MatrixSuperGate",
        (_MatrixSuperGate, dm_pr.Map, pr.MatrixGate, pr.TagGate, pr.NameGate),
        methods=dict(
            qubits=property(lambda self: (self.l_qubits, self.r_qubits)),
            n_qubits=property(lambda self: tuple(len(q) for q in self.qubits)),
            __print__=__print_qubits__),
        name='SMATRIX',
        l_qubits=l_qubits,
        r_qubits=r_qubits,
        Matrix=Map)(tags=tags)


def KrausSuperGate(gates: {iter[Gate], tuple[iter[Gate], iter[Gate]]},
                   s: any = 1,
                   tags: dict[any, any] = None,
                   copy: bool = True,
                   use_cache: bool = True) -> KrausSuperGate:
    """
    Return a KrausSuperGate.

    Parameters
    ----------
    gates: {iter[Gate], tuple[iter[Gate], iter[Gate]]}
        List of valid `Gate`s representing the `KrausSuperGate`. If `gates` is
        a pair of list of `Gate`s, the first list is used for the left-hand
        side `Gate`s of the `KrausSuperGate`, while the second list is used for
        the right-hand side `Gate`s of `KrausSuperGate`. If `gates` is a single
        list of `Gate`'s, left/right-hand side `Gates` of the `KrausSuperGate`
        are assumed to be the same.
    s: np.ndarray
        Correlation matrix between left/right-hand side `Gate`s of the
        `KrausSuperOperator`. More precisely, `KrausSuperOperator` will act on
        a dentity matrix ρ as:
        ```
        K(ρ) = \sum_{ij} s_ij L_i ρ R_j^+
        ```
        `s` can be a single scalar, a vector or a matrix consistent with the
        number of gates.
    tags: dict[any, any], optional
        Dictionary of tags.
    copy: bool, optional
        A copy of `gates` and `s` is used instead of a reference if `copy` is
        `True` (default: True).
    use_cache: bool, optional
        If `True`, extra memory is used to store a cached `Matrix`.

    Returns
    -------
    KrausSuperGate
    """
    from hybridq.gate import TupleGate

    # Copy if needed
    def _copy(x: iter[any, ...]):
        from copy import deepcopy
        return (deepcopy(y) for y in x) if copy else x

    def __print_qubits__(self):
        return {
            'l_qubits': (100, '' if self.qubits[0] is None else
                         f'l_qubits={pr._truncate_print(self.qubits[0])}', 0),
            'r_qubits': (101, '' if self.qubits[1] is None else
                         f'r_qubits={pr._truncate_print(self.qubits[1])}', 0),
        }

    # Get gates
    gates = tuple(gates)

    # Get left and right gates
    try:
        # Try by first assuming that 'gates' is a tuple of gates ...
        l_gates, r_gates = gates
        l_gates = TupleGate(_copy(l_gates))
        r_gates = TupleGate(_copy(r_gates))

    # ... if an error occurs ...
    except:
        try:
            # ... try as a single tuple of gates.
            l_gates = TupleGate(_copy(gates))
            r_gates = _copy(l_gates)

        except:
            # Finally, raise an error.
            raise ValueError("'gates' must be either a single list "
                             "of 'Gate's or a pair of lists of 'Gate's")

    # If r_gates is not empty, l_gates must be not empty
    if r_gates and not l_gates:
        raise ValueError("'l_gates' cannot be empty if 'r_gates' is provided")

    return dm_pr.generate(
        'KrausSuperGate',
        (BaseSuperGate, pr.SchmidtGate, dm_pr.Map, pr.TagGate, pr.NameGate),
        methods=dict(qubits=property(
            lambda self: (self.gates[0].qubits, self.gates[1].qubits)),
                     n_qubits=property(lambda self: tuple(
                         None if q is None else len(q) for q in self.qubits)),
                     __print__=__print_qubits__),
        gates=(l_gates, r_gates),
        s=(np.array if copy else np.asarray)(s),
        _conj_rgates=True,
        name='KRAUS',
        _use_cache=use_cache)(tags=tags)


# Define gate aliases
_gate_aliases = {
    'KSG': 'KRAUS',
    'MSG': 'SMATRIX',
}


def Gate(name: str, **kwargs):
    # To upper
    name = name.upper()

    # Check if an alias is used
    if name in _gate_aliases:
        warn(f"'{name}' is an alias for '{_gate_aliases[name]}'. "
             "Using Gate(name='{_gate_aliases[name]}').")
        name = _gate_aliases[name]

    # Return gate
    if name == 'KRAUS':
        return KrausSuperGate(**kwargs)
    elif name == 'SMATRIX':
        return MatrixSuperGate(**kwargs)
    elif name == 'STUPLE':
        return TupleSuperGate(**kwargs)
    else:
        raise NotImplementedError(f"'{name}' not implemented.")
