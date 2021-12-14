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
from hybridq.gate import BaseGate


class BaseCircuit(list):
    """
    Class representing a circuit.

    Attributes
    ----------
    gates: iter[Gate], optional
        Gates to be added to `Circuit`.
    copy: bool, optional
        If `True`, every gate is copied using `deepcopy`.

    Example
    -------
    >>> c = Circuit(Gate('H', qubits=[q]) for q in range(10))
    >>> c
    Circuit([
            Gate(name=H, qubits=[0])
            Gate(name=H, qubits=[1])
            Gate(name=H, qubits=[2])
            Gate(name=H, qubits=[3])
            Gate(name=H, qubits=[4])
            Gate(name=H, qubits=[5])
            Gate(name=H, qubits=[6])
            Gate(name=H, qubits=[7])
            Gate(name=H, qubits=[8])
            Gate(name=H, qubits=[9])
    ])
    >>> c + [Gate('X')]
    Circuit([
            Gate(name=H, qubits=[0])
            Gate(name=H, qubits=[1])
            Gate(name=H, qubits=[2])
            Gate(name=H, qubits=[3])
            Gate(name=H, qubits=[4])
            Gate(name=H, qubits=[5])
            Gate(name=H, qubits=[6])
            Gate(name=H, qubits=[7])
            Gate(name=H, qubits=[8])
            Gate(name=H, qubits=[9])
    ])
    >>> c[5:2:-1]
    Circuit([
            Gate(name=H, qubits=[5])
            Gate(name=H, qubits=[4])
            Gate(name=H, qubits=[3])
    ])
    """

    def __init__(self, gates: iter[Gate] = None, copy: bool = False) -> None:
        from copy import deepcopy

        # Initialize gates
        gates = [] if gates is None else list(gates)

        # Assign gates
        super().__init__(map(deepcopy, gates) if copy else gates)

    def __str__(self) -> str:
        """
        Return string representation of `Circuit`.
        """

        if 0 < len(self) <= 10:
            s = 'Circuit([\n'
            for gate in self[:-1]:
                s += '\t' + str(gate) + ',\n'
            s += '\t' + str(self[-1]) + '\n'
            s += '])'
        elif len(self) > 10:
            s = 'Circuit([\n'
            for gate in self[:4]:
                s += '\t' + str(gate) + '\n'
            s += '\t...\n'
            for gate in self[-4:-1]:
                s += '\t' + str(gate) + ',\n'
            s += '\t' + str(self[-1]) + '\n'
            s += '])'
        else:
            s = 'Circuit([])'

        return s

    def __repr__(self) -> str:
        """
        Return string representation of `Circuit`.
        """

        return self.__str__()

    def __add__(self, circuit: Circuit) -> Circuit:
        """
        Add `circuit` to an existing `Circuit`, and return a new `Circuit`.
        """

        # Return new circuit
        return type(self)(list(self) + list(type(self)(circuit)))

    def __iadd__(self, circuit: Circuit) -> None:
        """
        Append `circuit` to an existing `Circuit`.
        """
        # Extend circuit
        self.extend(circuit)

        # Return
        return self

    def __getitem__(self, key: any) -> {Gate, Circuit}:
        """
        Get elements from `Circuit`.
        """
        # Convert key to int
        try:
            key = int(key)
        except:
            pass

        # Get item
        _out = super().__getitem__(key)

        # Convert to circuit if needed
        return _out if isinstance(key, int) else type(self)(_out)

    def __setitem__(self, key: any, values: {Gate, iter(Gate)}) -> None:
        """
        Set elements of `Circuit`
        """
        # Convert key to int
        try:
            key = int(key)
        except:
            pass

        # Check if key is integral
        if isinstance(key, int):
            key = slice(key, key + 1)
            values = [values]

        # Update
        super().__setitem__(key, values)

    def append(self, gate: Gate) -> None:
        """
        Append `Gate` to existing `Circuit`.

        Parameters
        ----------
        gate: Gate
            `Gate` to append.

        Example
        -------
        >>> c = Circuit()
        >>> c.append(Gate('H'))
        >>> c
        Circuit([
                Gate(name=H)
        ])
        """

        # Append
        super().extend(type(self)([gate]))

    def extend(self, circuit: iter[Gate]) -> None:
        """
        Extend existing `Circuit`.

        Parameters
        ----------
        circuit: iter[Gate]
            Extend `Circuit` using `circuit`.

        Example
        -------
        >>> c = Circuit()
        >>> c.extend(Gate('X', qubits=[q]) for q in range(10))
        Circuit([
                Gate(name=X, qubits=[0])
                Gate(name=X, qubits=[1])
                Gate(name=X, qubits=[2])
                Gate(name=X, qubits=[3])
                Gate(name=X, qubits=[4])
                Gate(name=X, qubits=[5])
                Gate(name=X, qubits=[6])
                Gate(name=X, qubits=[7])
                Gate(name=X, qubits=[8])
                Gate(name=X, qubits=[9])
        ])
        """

        # Extend
        super().extend(type(self)(circuit))

    def __neq__(self, circuit: iter[Gate]) -> bool:
        return not self.__eq__(circuit)

    def __eq__(self, circuit: iter[Gate]) -> bool:
        """
        Two circuits are equivalent if they match gate by gate
        """
        return isinstance(circuit, type(self)) and super().__eq__(circuit)

    def all_tags(self) -> list[dict]:
        """
        Return a list of all tags in each `Gate`.

        Returns
        -------
        list[dict]
            List of all tags in each `Gate`.

        Example
        -------
        >>> Circuit(Gate('X', tags={q:q}) for q in range(10)).all_tags()
        [{0: 0},
         {1: 1},
         {2: 2},
         {3: 3},
         {4: 4},
         {5: 5},
         {6: 6},
         {7: 7},
         {8: 8},
         {9: 9}]
        """

        return [
            gate.tags for gate in self if gate.provides('tags') and gate.tags
        ]

    def _update_all_tags(self, *args, **kwargs) -> None:
        """
        Update all `Gate`s' tags in `Circuit`.
        """

        self.update_all_tags(*args, inplace=True, **kwargs)

    def update_all_tags(self,
                        *args,
                        inplace: bool = False,
                        **kwargs) -> Circuit:
        """
        Update all `Gate`s' tags in `Circuit`. If `inplace` is `True`, `Circuit`
        is modified in place.

        Parameters
        ----------
        inplace: bool, optional
            If `True`, `Circuit` is modified in place. Otherwise, a new
            `Circuit` is returned.

        Returns
        -------
        Circuit
            `Circuit` with update tags in all `Gate`s. If `inplace` is `True`,
            `Circuit` is modified in place.

        Example
        -------
        >>> c = Circuit(Gate('H', tags={q%4:q}) for q in range(10))
        >>> c
        Circuit([
                Gate(name=H, tags={0: 0})
                Gate(name=H, tags={1: 1})
                Gate(name=H, tags={2: 2})
                Gate(name=H, tags={3: 3})
                Gate(name=H, tags={0: 4})
                Gate(name=H, tags={1: 5})
                Gate(name=H, tags={2: 6})
                Gate(name=H, tags={3: 7})
                Gate(name=H, tags={0: 8})
                Gate(name=H, tags={1: 9})
        ])
        >>> c.update_all_tags({-1:'x', '42': 1.23})
        Circuit([
                Gate(name=H, tags={0: 0, -1: 'x', '42': 1.23})
                Gate(name=H, tags={1: 1, -1: 'x', '42': 1.23})
                Gate(name=H, tags={2: 2, -1: 'x', '42': 1.23})
                Gate(name=H, tags={3: 3, -1: 'x', '42': 1.23})
                Gate(name=H, tags={0: 4, -1: 'x', '42': 1.23})
                Gate(name=H, tags={1: 5, -1: 'x', '42': 1.23})
                Gate(name=H, tags={2: 6, -1: 'x', '42': 1.23})
                Gate(name=H, tags={3: 7, -1: 'x', '42': 1.23})
                Gate(name=H, tags={0: 8, -1: 'x', '42': 1.23})
                Gate(name=H, tags={1: 9, -1: 'x', '42': 1.23})
        ])
        """

        if inplace:
            for gate in self:
                gate._update_tags(*args, **kwargs)
            return self
        else:
            return type(self)(gate.update_tags(*args, inplace=False, **kwargs)
                              for gate in self)

    def _remove_all_tags(self, keys: iter[any]) -> None:
        """
        Remove all tags matching `keys` from all `Gate`s.
        """

        self.remove_all_tags(keys, inplace=True)

    def remove_all_tags(self,
                        keys: iter[any],
                        *,
                        inplace: bool = False) -> Circuit:
        """
        Remove all tags matching `keys` from all `Gate`s. If `inplace` is
        `True`, `Circuit` is modified in place.

        Parameters
        ----------
        keys: iter[any]
            Keys to remove from tags.
        inplace: bool, optional
            If `True`, `Circuit` is modified in place. Otherwise, a new
            `Circuit` is returned.

        Returns
        -------
        Circuit
            `Circuit` with tags matching `keys` from all `Gate`s removed. If
            `inplace` is `True`, `Circuit` is modified in place.

        Example
        -------
        >>> c = Circuit(Gate('H', tags={q%4:q}) for q in range(10))
        >>> c
        Circuit([
                Gate(name=H, tags={0: 0})
                Gate(name=H, tags={1: 1})
                Gate(name=H, tags={2: 2})
                Gate(name=H, tags={3: 3})
                Gate(name=H, tags={0: 4})
                Gate(name=H, tags={1: 5})
                Gate(name=H, tags={2: 6})
                Gate(name=H, tags={3: 7})
                Gate(name=H, tags={0: 8})
                Gate(name=H, tags={1: 9})
        ])
        >>> c.remove_all_tags([0, 3])
        Circuit([
                Gate(name=H)
                Gate(name=H, tags={1: 1})
                Gate(name=H, tags={2: 2})
                Gate(name=H)
                Gate(name=H)
                Gate(name=H, tags={1: 5})
                Gate(name=H, tags={2: 6})
                Gate(name=H)
                Gate(name=H)
                Gate(name=H, tags={1: 9})
        ])
        """

        # Convert keys to set
        keys = set(keys)

        if inplace:
            for gate in self:
                gate._remove_tags(keys)
            return self
        else:
            return type(self)(
                gate.remove_tags(keys, inplace=False) for gate in self)


class Circuit(BaseCircuit):

    @staticmethod
    def __check_gate__(gate: Gate):
        from hybridq.gate import TupleGate
        from hybridq.base.property import Tuple
        if isinstance(gate, tuple) or isinstance(gate, Tuple):
            return TupleGate(map(Circuit.__check_gate__, gate))
        elif isinstance(gate, BaseGate):
            return gate
        else:
            raise ValueError(f"'{type(gate).__name__}' not supported.")

    def __init__(self, gates: iter[Gate] = tuple(), *args, **kwargs) -> None:
        super().__init__(gates=map(self.__check_gate__, gates), **kwargs)

    def all_qubits(self, *, ignore_missing_qubits: bool = False) -> list[any]:
        """
        Get all qubits in `Circuit`. It raises a `ValueError` if qubits in
        `Gate` are missing, unless `ignore_missing_qubits` is `True`. The
        returned qubits are always sorted using `hybridq.utils.sort` for
        consistency.

        Parameters
        ----------
        ignore_missing_qubits: bool, optional
            If `True`, ignore gates without specified qubits. Otherwise, raise
            `ValueError`.

        Returns
        -------
        list[any]
            Sorted list of all qubits in `Circuit`.

        Example
        -------
        >>> Circuit([Gate('H', qubits=[2]), Gate('X', qubits=[1])]).all_qubits()
        [1, 2]
        >>> Circuit([Gate('H', qubits=[2]), Gate('X', qubits=[1]), Gate('H')]).all_qubits()
        ValueError: Circuit contains virtual gates with no qubits.
        >>> Circuit([Gate('H', qubits=[2]), Gate('X', qubits=[1]), Gate('H')]).all_qubits(ignore_missing_qubits=True)
        [1, 2]
        """
        # If Circuit has not qubits, return empty list
        if not len(self):
            return []

        # Define flatten
        def _unique_flatten(l):
            from hybridq.utils import sort
            return sort(set(y for x in l for y in x))

        # Get all qubits
        _qubits = [
            gate.qubits if gate.provides('qubits') else None for gate in self
        ]

        # Check if there are virtual gates with no qubits
        if not ignore_missing_qubits and any(q is None for q in _qubits):
            raise ValueError("Circuit contains virtual gates with no qubits.")

        # Flatten qubits and remove None's
        return _unique_flatten(q for q in _qubits if q is not None)

    def inv(self) -> Circuit:
        """
        Return the inverse circuit of `Circuit`.

        Returns
        -------
        Circuit
            Inverse of `Circuit`.

        Example
        -------
        >>> from numpy.random import random
        >>> from hybridq.circuit.utils import simplify
        >>> c = Circuit(Gate('RX', qubits=[q], params=[random()]) for q in range(10))
        >>> simplify(c + c.inv())
        Circuit([
        ])
        """

        return type(self)(gate.inv() for gate in reversed(self))

    def conj(self) -> Circuit:
        """
        Return the conjugate circuit of `Circuit`.

        Returns
        -------
        Circuit
            Conjugation of `Circuit`.
        """

        return type(self)(gate.conj() for gate in self)

    def T(self) -> Circuit:
        """
        Return the transposed circuit of `Circuit`.

        Returns
        -------
        Circuit
            Transposition of `Circuit`.
        """

        return type(self)(gate.T() for gate in reversed(self))

    def adj(self) -> Circuit:
        """
        Return the adjoint circuit of `Circuit`.

        Returns
        -------
        Circuit
            Adjoint of `Circuit`.
        """

        return type(self)(gate.adj() for gate in reversed(self))
