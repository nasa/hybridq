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
from hybridq.gate import Gate, BaseGate
from hybridq.utils import sort, argsort
from hybridq.circuit import Circuit
from tqdm.auto import tqdm
import numpy as np


def flatten(a: Circuit) -> Circuit:
    """
    Return a flattened circuit. More precisely, `flatten` iteratively looks for
    gates that provide `flatten` in order to return a flattened circuit.

    Parameters
    ----------
    a: Circuit
        Circuit to flatten.

    Returns
    -------
    Circuit
        Flattened circuit.
    """
    return Circuit(
        g for gs in a for g in (gs if gs.provides('flatten') else (gs,)))


def isidentity(a: Circuit, atol: float = 1e-8) -> bool:
    """
    Check if `a` is close to identity within an absolute tollerance of `atol`.
    The check is done by getting the matrix representation of the circuit `a`.

    Parameters
    ----------
    a: Circuit
        Circuit to check.
    atol: float, optional
        Absolute tollerance.

    Returns
    -------
    bool
        `True` if `a` is close to the identity, otherwise `False`.
    """
    # Get matrix
    M = matrix(a)

    # Check if close to identity
    return np.allclose(M, np.eye(M.shape[0]), atol=atol)


def isclose(a: Circuit,
            b: Circuit,
            use_matrix_commutation: bool = True,
            max_n_qubits_matrix: int = 10,
            atol: float = 1e-8,
            verbose: bool = False) -> bool:
    """
    Check if `a` is close to `b` within the absolute tollerance
    `atol`.

    Parameters
    ----------
    circuit: Circuit[BaseGate]
        `Circuit` to compare with.
    use_matrix_commutation: bool, optional
        Use commutation rules. See `hybridq.circuit.utils.simplify`.
    max_n_qubits_matrix: int, optional
        Matrices are computes for gates with up to `max_n_qubits_matrix` qubits
        (default: 10).
    atol: float, optional
        Absolute tollerance.

    Returns
    -------
    bool
        `True` if the two circuits are close within the absolute tollerance
        `atol`, and `False` otherwise.

    See Also
    --------
    hybridq.circuit.utils.simplify

    Example
    -------
    >>> c = Circuit(Gate('H', [q]) for q in range(10))
    >>> c.isclose(Circuit(g**1.1 for g in c))
    False
    >>> c.isclose(Circuit(g**1.1 for g in c), atol=1e-1)
    True
    """

    # Get simplified circuit
    s = simplify(a + b.inv(),
                 use_matrix_commutation=use_matrix_commutation,
                 max_n_qubits_matrix=max_n_qubits_matrix,
                 atol=atol,
                 verbose=verbose)

    return not s or all(
        isidentity([g], atol=atol)
        for g in tqdm(s, disable=not verbose, desc='Check'))


def insert_from_left(circuit: iter[BaseGate],
                     gate: BaseGate,
                     atol: float = 1e-8,
                     *,
                     use_matrix_commutation: bool = True,
                     max_n_qubits_matrix: int = 10,
                     simplify: bool = True,
                     pop: bool = False,
                     pinned_qubits: list[any] = None,
                     inplace: bool = False) -> Circuit:
    """
    Add a gate to circuit starting from the left, commuting with existing gates
    if necessary.

    Parameters
    ----------
    circuit: Circuit
        `gate` will be added to `circuit`.
    gate: Gate
        Gate to add to `circuit`.
    atol: float, optional
        Absolute tollerance while simplifying.
    use_matrix_commutation: bool, optional
        Use matrix commutation while simplifying `circuit`.
    max_n_qubits_matrix: int, optional
        Matrices are computes for gates with up to `max_n_qubits_matrix` qubits
        (default: 10).
    simplify: bool, optional
        Simplify `circuit` while adding `gate` (default: `True`).
    pop: bool, optional
        Remove `gate` if it commutes with all gates in `circuit` (default: `False`).
    pinned_qubits: list[any], optional
        If `pop` is `True`, remove gates unless `gate` share qubits with
        `pinned_qubits` (default: `None`).
    inplace: bool, optional
        If `True`, add `gate` to `circuit` in-place (default: `False`)

    Returns
    -------
    Circuit
        Circuit with `gate` added to it.
    """
    from copy import deepcopy

    # Copy circuit if required
    if not inplace:
        circuit = Circuit(circuit, copy=True)

    # Get qubits
    if gate.provides('qubits') and gate.qubits is not None:
        _qubits = set(gate.qubits)
    else:
        # If gate has not qubits, just append to the left
        circuit.insert(0, deepcopy(gate))
        return circuit

    # Iterate over all the gates
    for _p, _g in enumerate(circuit):
        # Remove if gate simplifies with _g
        try:
            if simplify and gate.inv().isclose(_g, atol=atol):
                del (circuit[_p])
                return circuit
        except:
            pass

        # Otherwise, check if gate can commute with _g. If not, insert gate
        # and exit loop.
        _commute = False
        try:
            if _g.n_qubits <= max_n_qubits_matrix:
                _commute |= not _qubits.intersection(_g.qubits)
                _commute |= use_matrix_commutation and gate.commutes_with(
                    _g, atol=atol)
        except:
            pass
        finally:
            if not _commute:
                circuit.insert(_p, deepcopy(gate))
                return circuit

    # If commutes with everything, just append at the end
    if not pop or _qubits.intersection(pinned_qubits):
        circuit.append(deepcopy(gate))

    # Return circuit
    return circuit


def to_nx(circuit: iter[BaseGate],
          add_final_nodes: bool = True,
          node_tags: dict = None,
          edge_tags: dict = None,
          return_qubits_map: bool = False,
          leaves_prefix: str = 'q') -> networkx.Graph:
    """
    Return graph representation of circuit. `to_nx` is deterministic, so it can
    be reused elsewhere.

    Parameters
    ----------
    circuit: iter[BaseGate]
        Circuit to get graph representation from.
    add_final_nodes: bool, optional
        Add final nodes for each qubit to the graph representation of `circuit`.
    node_tags: dict, optional
        Add specific tags to nodes.
    edge_tags: dict, optional
        Add specific tags to edges.
    return_qubits_map: bool, optional
        Return map associated to the Circuit qubits.
    leaves_prefix: str, optional
        Specify prefix to use for leaves.

    Returns
    -------
    networkx.Graph
        Graph representing `circuit`.

    Example
    -------
    >>> import networkx as nx
    >>>
    >>> # Define circuit
    >>> circuit = Circuit(
    >>>     [Gate('X', qubits=[0])**1.2,
    >>>      Gate('ISWAP', qubits=[0, 1])**2.3], Gate('H', [1]))
    >>>
    >>> # Draw graph
    >>> nx.draw_planar(utils.to_nx(circuit))

    .. image:: ../../images/circuit_nx.png
    """
    import networkx as nx

    # Initialize
    if node_tags is None:
        node_tags = {}
    if edge_tags is None:
        edge_tags = {}

    # Check if node is a leaf
    def _is_leaf(node):
        return type(node) == str and node[:len(leaves_prefix)] == leaves_prefix

    # Convert iterable to Circuit
    circuit = Circuit(circuit)

    # Get graph
    graph = nx.DiGraph()

    # Get qubits
    qubits = circuit.all_qubits()

    # Get qubits_map
    qubits_map = {q: i for i, q in enumerate(qubits)}

    # Check that no qubits is 'confused' as leaf
    if any(_is_leaf(q) for q in qubits):
        raise ValueError(
            f"No qubits must start with 'leaves_prefix'={leaves_prefix}.")

    # Add first layer
    for q in qubits:
        graph.add_node(f'{leaves_prefix}_{qubits_map[q]}_i',
                       qubits=[q],
                       **node_tags)

    # Last leg
    last_leg = {q: f'{leaves_prefix}_{qubits_map[q]}_i' for q in qubits}

    # Build network
    for x, gate in enumerate(circuit):

        # Add node
        graph.add_node(x,
                       circuit=Circuit([gate]),
                       qubits=sort(gate.qubits),
                       **node_tags)

        # Add edges (time directed)
        graph.add_edges_from([(last_leg[q], x) for q in gate.qubits],
                             **edge_tags)

        # Update last_leg
        last_leg.update({q: x for q in gate.qubits})

    # Add last indexes if required
    if add_final_nodes:
        for q in qubits:
            graph.add_node(f'{leaves_prefix}_{qubits_map[q]}_f',
                           qubits=[q],
                           **node_tags)
        graph.add_edges_from([(x, f'{leaves_prefix}_{qubits_map[q]}_f')
                              for q, x in last_leg.items()], **edge_tags)

    if return_qubits_map:
        return graph, qubits_map
    else:
        return graph


def to_tn(circuit: iter[BaseGate],
          complex_type: any = 'complex64',
          return_qubits_map: bool = False,
          leaves_prefix: str = 'q_') -> quimb.tensor.TensorNetwork:
    """
    Return `quimb.tensor.TensorNetwork` representing `circuit`. `to_tn` is
    deterministic, so it can be reused elsewhere.

    Parameters
    ----------
    circuit: iter[BaseGate]
        Circuit to get `quimb.tensor.TensorNetwork` representation from.
    complex_type: any, optional
        Complex type to use while getting the `quimb.tensor.TensorNetwork`
        representation.
    return_qubits_map: bool, optional
        Return map associated to the Circuit qubits.
    leaves_prefix: str, optional
        Specify prefix to use for leaves.

    Returns
    -------
    quimb.tensor.TensorNetwork
        Tensor representing `circuit`.

    Example
    -------
    >>> # Define circuit
    >>> circuit = Circuit(
    >>>     [Gate('X', qubits=[0])**1.2,
    >>>      Gate('ISWAP', qubits=[0, 1])**2.3], Gate('H', [1]))
    >>>
    >>> # Draw graph
    >>> utils.to_tn(circuit).graph()

    .. image:: ../../images/circuit_tn.png
    """
    import quimb.tensor as tn

    # Convert iterable to Circuit
    circuit = Circuit(circuit)

    # Get all qubits
    all_qubits = circuit.all_qubits()

    # Get qubits map
    qubits_map = {q: i for i, q in enumerate(all_qubits)}

    # Get last_tag
    last_tag = {q: 'i' for q in all_qubits}

    # Node generator
    def _get_node(t, gate):

        # Get matrix
        U = np.reshape(gate.matrix().astype(complex_type),
                       [2] * (2 * len(gate.qubits)))

        # Get indexes
        inds = [f'{leaves_prefix}_{qubits_map[q]}_{t}' for q in gate.qubits] + [
            f'{leaves_prefix}_{qubits_map[q]}_{last_tag[q]}'
            for q in gate.qubits
        ]

        # Update last_tag
        for q in gate.qubits:
            last_tag[q] = t

        # Return node
        return tn.Tensor(
            U.astype(complex_type),
            inds=inds,
            tags=[f'{leaves_prefix}_{qubits_map[q]}' for q in gate.qubits] +
            [f'gate-idx_{t}'])

    # Get list of tensors
    tensor = [_get_node(t, gate) for t, gate in enumerate(circuit)]

    # Generate new output map
    output_map = {
        f'{leaves_prefix}_{qubits_map[q]}_{t}':
        f'{leaves_prefix}_{qubits_map[q]}_f' for q, t in last_tag.items()
    }

    # Rename output legs
    for node in tensor:
        node.reindex(output_map, inplace=True)

    # Return tensor network
    if return_qubits_map:
        return tn.TensorNetwork(tensor), qubits_map
    else:
        return tn.TensorNetwork(tensor)


def to_matrix_gate(circuit: iter[BaseGate],
                   complex_type: any = 'complex64',
                   **kwargs) -> BaseGate:
    """
    Convert `circuit` to a matrix `BaseGate`.

    Parameters
    ----------
    circuit: iter[BaseGate]
        Circuit to convert to `BaseGate`.
    complex_type: any, optional
        Float type to use while converting to `BaseGate`.

    Returns
    -------
    Gate
        `BaseGate` representing `circuit`.

    Example
    -------
    >>> # Define circuit
    >>> circuit = Circuit(
    >>>     [Gate('X', qubits=[0])**1.2,
    >>>      Gate('ISWAP', qubits=[0, 1])**2.3])
    >>>
    >>> gate = utils.to_matrix_gate(circuit)
    >>> gate
    Gate(name=MATRIX, qubits=[0, 1], U=np.array(shape=(4, 4), dtype=complex64))
    >>> gate.U
    array([[ 0.09549151-0.29389262j,  0.        +0.j        ,
             0.9045085 +0.29389262j,  0.        +0.j        ],
           [ 0.13342446-0.41063824j, -0.08508356+0.26186025j,
            -0.13342446-0.04335224j, -0.8059229 -0.26186025j],
           [-0.8059229 -0.26186025j, -0.13342446-0.04335224j,
            -0.08508356+0.26186025j,  0.13342446-0.41063824j],
           [ 0.        +0.j        ,  0.9045085 +0.29389262j,
             0.        +0.j        ,  0.09549151-0.29389262j]],
          dtype=complex64)
    """

    # Convert iterable to Circuit
    circuit = Circuit(circuit)

    return Gate('MATRIX',
                qubits=circuit.all_qubits(),
                U=matrix(circuit, complex_type=complex_type, **kwargs))


def compress(circuit: iter[BaseGate],
             max_n_qubits: int = 2,
             *,
             exclude_qubits: iter[any] = None,
             use_matrix_commutation: bool = True,
             max_n_qubits_matrix: int = 10,
             skip_compression: iter[{type, str}] = None,
             skip_commutation: iter[{type, str}] = None,
             atol: float = 1e-8,
             verbose: bool = False) -> list[Circuit]:
    """
    Compress gates together up to the specified number of qubits. `compress`
    is deterministic, so it can be reused elsewhere.

    Parameters
    ----------
    circuit: iter[BaseGate]
        Circuit to compress.
    max_n_qubits: int, optional
        Maximum number of qubits that a compressed gate may have.
    exclude_qubits: list[any], optional
        Exclude gates which act on `exclude_qubits` to be compressed.
    use_matrix_commutation: bool, optional
        If `True`, use commutation to maximize compression.
    max_n_qubits_matrix: int, optional
        Limit the size of matrices when checking for commutation.
    skip_compression: iter[{type, str}], optional
        If `BaseGate` is either an instance of any types in `skip_compression`,
        it provides any methods in `skip_compression`, or `BaseGate` name will
        match any names in `skip_compression`, `BaseGate` will not be
        compressed. However, if `use_matrix_commutation` is `True`, commutation will
        be checked against `BaseGate`.
    skip_commutation: iter[{type, str}], optional
        If `BaseGate` is either an instance of any types in `skip_commutation`,
        it provides any methods in `skip_commutation`, or `BaseGate` name will
        match any names in `skip_commutation`, `BaseGate` will not be checked
        against commutation.
    atol: float
        Absolute tollerance for commutation.
    verbose: bool, optional
        Verbose output.

    Returns
    -------
    list[Circuit]
        A list of `Circuit`s, with each `Circuit` representing a compressed
        `BaseGate`.

    See Also
    --------
    hybridq.gate.commutes_with

    Example
    -------
    >>> # Define circuit
    >>> circuit = Circuit(
    >>>     [Gate('X', qubits=[0])**1.2,
    >>>      Gate('ISWAP', qubits=[0, 1])**2.3,
    >>>      Gate('ISWAP', qubits=[0, 2])**2.3])
    >>>
    >>> # Compress circuit up to 1-qubit gates
    >>> utils.compress(circuit, 1)
    [Circuit([
        Gate(name=X, qubits=[0])**1.2
     ]),
     Circuit([
        Gate(name=ISWAP, qubits=[0, 1])**2.3
     ]),
     Circuit([
        Gate(name=ISWAP, qubits=[0, 2])**2.3
     ])]
    >>> # Compress circuit up to 2-qubit gates
    >>> utils.compress(circuit, 2)
    [Circuit([
        Gate(name=X, qubits=[0])**1.2
        Gate(name=ISWAP, qubits=[0, 1])**2.3
     ]),
     Circuit([
        Gate(name=ISWAP, qubits=[0, 2])**2.3
     ])]
    >>> # Compress circuit up to 3-qubit gates
    >>> utils.compress(circuit, 3)
    [Circuit([
        Gate(name=X, qubits=[0])**1.2
        Gate(name=ISWAP, qubits=[0, 1])**2.3
        Gate(name=ISWAP, qubits=[0, 2])**2.3
     ])]
    """
    # If max_n_qubits <= 0, split every gate
    if max_n_qubits <= 0:
        return [Circuit([g]) for g in circuit]

    # Initialize skip_compression and skip_commutation
    skip_compression = tuple() if skip_compression is None else tuple(
        skip_compression)
    skip_commutation = tuple() if skip_commutation is None else tuple(
        skip_commutation)

    def _check_skip(gate, x):
        if isinstance(x, type):
            return isinstance(gate, x)
        elif isinstance(x, str):
            return gate.name == x.upper() or gate.provides(x)
        else:
            raise ValueError(f"'{x}' not supported.")

    # Initialize exclude_qubits
    exclude_qubits = set([] if exclude_qubits is None else exclude_qubits)

    # Convert to Circuit
    circuit = Circuit(circuit)

    # Initialize compressed circuit
    new_circuit = []

    # For every gate in circuit ..
    for gate in tqdm(circuit,
                     disable=not verbose,
                     desc=f'Compress ({max_n_qubits})'):
        # Initialize matrix gate
        _gate = None

        # Initialize _compress
        gate_properties = dict(compress=True, commute=True)

        # Initialize index
        _merge_to = len(new_circuit)

        # If gate does not provide qubits or qubits is None,
        # then gate is not compressible
        if not gate.provides('qubits') or gate.qubits is None:
            gate_properties['compress'] = False
            gate_properties['commute'] = False

        # Otherwise ...
        else:
            # Get qubits
            _q = set(gate.qubits)

            # Get Matrix gate if possible
            try:
                _gate = to_matrix_gate(
                    [gate], max_compress=0) if use_matrix_commutation and len(
                        _q) <= max_n_qubits_matrix else None
            except:
                _gate = None

            # Check if gate must skip compression
            if any(_check_skip(gate, t) for t in skip_compression
                  ) or _q.intersection(exclude_qubits):
                gate_properties['compress'] = False

            # Check if gate must skip commutation
            if any(_check_skip(gate, t) for t in skip_commutation):
                gate_properties['commute'] = False

            # Check for each existing layer
            for i, (_circ, _circ_gate,
                    _circ_properties) in reversed(list(enumerate(new_circuit))):

                # If _circ does not provide any qubits, just break
                try:
                    # Get circuit qubits
                    _cq = set(_circ.all_qubits())
                except:
                    break

                # Check if both gate and _circ can be compressed
                if gate_properties['compress'] and _circ_properties['compress']:
                    # Check if it can be merged
                    if len(_q.union(_cq)) <= max(max_n_qubits, len(_cq),
                                                 len(_q)):
                        _merge_to = i

                # Check commutation
                if use_matrix_commutation and gate_properties[
                        'commute'] and _circ_properties['commute']:
                    # Check if gate and _circ share any qubit
                    if not _q.intersection(_cq):
                        continue

                    # Check if gate and _circ commute using matrix
                    try:
                        if _gate.commutes_with(_circ_gate):
                            continue
                    except:
                        pass

                # Otherwise, just break
                break

        # If it possible to merge the gate to an existing layer
        if _merge_to < len(new_circuit):
            # Get layer
            _nc = new_circuit[_merge_to]

            # Update circuit
            _nc[0].append(gate)

            # Update matrix
            try:
                _nc[1] = to_matrix_gate(
                    [_nc[1], _gate],
                    max_compress=0) if use_matrix_commutation and len(
                        set(_gate.qubits).union(
                            _nc[1].qubits)) <= max_n_qubits_matrix else None
            except:
                _nc[1] = None

            # Update properties
            for k in ['compress', 'commute']:
                _nc[2][k] &= gate_properties[k]

        # Otherwise, create a new layer
        else:
            new_circuit.append([Circuit([gate]), _gate, gate_properties])

    # Return only circuits
    return [c for c, _, _ in new_circuit]


def matrix(circuit: iter[BaseGate],
           order: iter[any] = None,
           complex_type: any = 'complex64',
           max_compress: int = 4,
           verbose: bool = False) -> numpy.ndarray:
    """
    Return matrix representing `circuit`.

    Parameters
    ----------
    circuit: iter[BaseGate]
        Circuit to get the matrix from.
    order: iter[any], optional
        If specified, a matrix is returned following the order given by
        `order`. Otherwise, `circuit.all_qubits()` is used.
    max_compress: int, optional
        To reduce the computational cost, `circuit` is compressed prior to
        compute the matrix.
    complex_type: any, optional
        Complex type to use to compute the matrix.
    verbose: bool, optional
        Verbose output.

    Returns
    -------
    numpy.ndarray
        Unitary matrix of `circuit`.

    Example
    -------
    >>> # Define circuit
    >>> circuit = Circuit([Gate('CX', [1, 0])])
    >>> # Show qubits
    [0, 1]
    >>> circuit.all_qubits()
    >>> # Get matrix without specifying any qubits order
    >>> # (therefore using circuit.all_qubits() == [0, 1])
    >>> utils.matrix()
    array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
           [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
           [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j]], dtype=complex64)
    >>> # Get matrix with a specific order of qubits
    >>> utils.matrix(Circuit([Gate('CX', [1, 0])]), order=[1, 0])
    array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
           [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]], dtype=complex64)
    """

    # Convert iterable to Circuit
    circuit = Circuit(circuit)

    # Check order
    if order is not None:
        # Conver to list
        order = list(order)
        if set(order).difference(circuit.all_qubits()):
            raise ValueError(
                "'order' must be a valid permutation of indexes in 'Circuit'.")

    # Compress circuit
    if max_compress > 0:
        return matrix(Circuit(
            to_matrix_gate(c, complex_type=complex_type, max_compress=0)
            for c in compress(circuit, max_n_qubits=max_compress)),
                      order=order,
                      complex_type=complex_type,
                      max_compress=0,
                      verbose=verbose)

    # Get qubits
    qubits = circuit.all_qubits()
    n_qubits = len(qubits)

    # Initialize matrix
    U = np.reshape(np.eye(2**n_qubits, order='C', dtype=complex_type),
                   [2] * (2 * n_qubits))

    for g in tqdm(circuit, disable=not verbose):

        # Get gate's qubits
        _qubits = g.qubits
        _n_qubits = len(_qubits)

        # Get map
        _map = [qubits.index(q) for q in _qubits]
        _map += [x for x in range(n_qubits) if x not in _map]

        # Reorder qubits
        qubits = [qubits[x] for x in _map]

        # Update U
        U = np.reshape(
            g.matrix().astype(complex_type) @ np.reshape(
                np.transpose(U, _map + list(range(n_qubits, 2 * n_qubits))),
                (2**_n_qubits, 2**(2 * n_qubits - _n_qubits))),
            (2,) * (2 * n_qubits))

    # Get U
    U = np.reshape(
        np.transpose(U,
                     argsort(qubits) + list(range(n_qubits, 2 * n_qubits))),
        (2**n_qubits, 2**n_qubits))

    # Reorder if required
    if order and order != circuit.all_qubits():
        qubits = circuit.all_qubits()
        U = np.reshape(
            np.transpose(np.reshape(U, (2,) * (2 * n_qubits)),
                         [qubits.index(q) for q in order] +
                         [n_qubits + qubits.index(q) for q in order]),
            (2**n_qubits, 2**n_qubits))

    # Check U has the right type and order
    assert (U.dtype == np.dtype(complex_type))
    assert (U.data.c_contiguous)

    # Return matrix
    return U


def unitary(*args, **kwargs):
    """
    Alias for `utils.matrix`.
    """
    from hybridq.utils import DeprecationWarning
    from warnings import warn

    # Warn that `self.matrix` should be used instead of `self.unitary`
    warn("Since '0.7.0', 'hybridq.circuit.utils.matrix' should be used instead "
         "of the less general 'hybridq.circuit.utils.unitary'")

    # Call matrix
    return matrix(*args, **kwargs)


def simplify(circuit: list[BaseGate],
             atol: float = 1e-8,
             use_matrix_commutation: bool = True,
             max_n_qubits_matrix: int = 10,
             remove_id_gates: bool = True,
             verbose: bool = False) -> Circuit:
    """
    Compress together gates up to the specified number of qubits.
    """

    # Initialize new circuit
    new_circuit = Circuit()

    # Remove gates if required
    if remove_id_gates:
        rev_circuit = (g for g in reversed(circuit) if g.name != 'I' and (
            not g.provides('matrix') or g.n_qubits > max_n_qubits_matrix or
            not isidentity([g], atol=atol)))
    else:
        rev_circuit = reversed(circuit)

    # Insert gates, one by one
    for gate in tqdm(rev_circuit,
                     disable=not verbose,
                     total=len(circuit),
                     desc='Simplify'):
        insert_from_left(new_circuit,
                         gate,
                         atol=atol,
                         use_matrix_commutation=use_matrix_commutation,
                         max_n_qubits_matrix=max_n_qubits_matrix,
                         simplify=True,
                         pop=False,
                         pinned_qubits=None,
                         inplace=True)

    # Return simplified circuit
    return new_circuit


def popright(circuit: list[BaseGate],
             pinned_qubits: list[any],
             atol: float = 1e-8,
             use_matrix_commutation: bool = True,
             max_n_qubits_matrix: int = 10,
             simplify: bool = True,
             verbose: bool = False) -> Circuit:
    """
  Remove gates outside the lightcone created by pinned_qubits.
  """

    # Initialize new circuit
    new_circuit = Circuit()

    # Insert gates, one by one
    for gate in tqdm(reversed(circuit),
                     disable=not verbose,
                     total=len(circuit),
                     desc='Pop'):
        insert_from_left(new_circuit,
                         gate,
                         atol=atol,
                         use_matrix_commutation=use_matrix_commutation,
                         max_n_qubits_matrix=max_n_qubits_matrix,
                         simplify=simplify,
                         pop=True,
                         pinned_qubits=pinned_qubits,
                         inplace=True)

    # Return simplified circuit
    return new_circuit


def popleft(circuit: list[BaseGate],
            pinned_qubits: list[any],
            atol: float = 1e-8,
            use_matrix_commutation: bool = True,
            simplify: bool = True,
            verbose: bool = False) -> Circuit:
    """
  Remove gates outside the lightcone created by pinned_qubits (starting from the right).
  """

    return Circuit(
        reversed(
            popright(list(reversed(circuit)),
                     pinned_qubits=pinned_qubits,
                     atol=atol,
                     use_matrix_commutation=use_matrix_commutation,
                     simplify=simplify,
                     verbose=verbose)))


def pop(circuit: list[BaseGate],
        direction: str,
        pinned_qubits: list[any],
        atol: float = 1e-8,
        use_matrix_commutation: bool = True,
        simplify: bool = True,
        verbose: bool = False) -> Circuit:
    """
    Remove gates outside the lightcone created by pinned_qubits.
    """
    from functools import partial as partial_func

    _popleft = partial_func(popleft,
                            pinned_qubits=pinned_qubits,
                            atol=atol,
                            use_matrix_commutation=use_matrix_commutation,
                            simplify=simplify,
                            verbose=verbose)
    _popright = partial_func(popright,
                             pinned_qubits=pinned_qubits,
                             atol=atol,
                             use_matrix_commutation=use_matrix_commutation,
                             simplify=simplify,
                             verbose=verbose)

    if direction == 'left':
        return _popleft(circuit)
    elif direction == 'right':
        return _popright(circuit)
    elif direction == 'both':
        return _popleft(_popright(circuit))
    else:
        raise ValueError(f"direction='{direction}' not supported.")


def moments(
        circuit: iter[{BaseGate, Circuit}]) -> list[list[{BaseGate, Circuit}]]:
    """
    Split circuit in moments.
    """
    from hybridq.gate import TupleGate

    # Convert iterable to list
    circuit = list(circuit)

    # If circuit is empty, return a single empty TupleGate
    if not circuit:
        return [TupleGate()]

    # Get qubits
    def _get_qubits(x):
        if isinstance(x, BaseGate):
            return x.qubits if x.n_qubits else tuple()
        elif isinstance(x, Circuit):
            return x.all_qubits()
        else:
            raise ValueError(f"'{x}' is not valid.")

    # Get all used qubits
    qubits = sort({q for x in circuit for q in _get_qubits(x)})

    # Get map of leves
    level_map = {q: 0 for q in qubits}
    level = [0] * len(circuit)

    # Get the right level for each object ..
    for i, x in enumerate(circuit):
        # Get qubits
        _qubits = _get_qubits(x)

        # If gate is acting on qubits, add gate to the right level
        if _qubits:
            # Get max level
            level[i] = np.max([level_map[q] for q in _get_qubits(x)]) + 1

            # Update level_map
            level_map.update({q: level[i] for q in _get_qubits(x)})

        # .. otherwise, simply update all qubits to create a new moment
        else:
            level[i] = np.max(level) + 1
            level_map = {q: level[i] for q in qubits}

    # Initialize moments
    moments = [[] for _ in range(np.max(level))]

    # Update moments
    for i, x in enumerate(circuit):
        moments[level[i] - 1].append(x)

    # Return moments
    return list(map(TupleGate, moments))


def remove_swap(circuit: Circuit) -> tuple[Circuit, dict[any, any]]:
    """
    Iteratively remove SWAP's from circuit by actually swapping qubits.
    The output map will have the form new_qubit -> old_qubit.
    """

    # Initialize map
    _qubits_map = {q: q for q in circuit.all_qubits()}

    # Initialize circuit
    _circ = Circuit()

    # Get ideal SWAP
    _SWAP = Gate('SWAP').matrix()

    # For each gate in circuit ..
    for gate in circuit:

        # Check if gate is close to SWAP
        if gate.n_qubits == 2 and gate.qubits and np.allclose(
                gate.matrix(), _SWAP):

            # If true, swap qubits
            _q0 = next(k for k, v in _qubits_map.items() if v == gate.qubits[0])
            _q1 = next(k for k, v in _qubits_map.items() if v == gate.qubits[1])
            _qubits_map[_q0], _qubits_map[_q1] = _qubits_map[_q1], _qubits_map[
                _q0]

        # Otherwise, remap qubits and append
        else:

            # Get the right qubits
            _qubits = [
                next(k
                     for k, v in _qubits_map.items()
                     if v == q)
                for q in gate.qubits
            ]

            # Append to the new circuit
            _circ.append(gate.on(_qubits))

    # Return circuit and map
    return _circ, _qubits_map


def expand_iswap(circuit: Circuit) -> Circuit:
    """
    Expand ISWAP's by iteratively replacing with SWAP's, CZ's and Phases.
    """
    from copy import deepcopy

    # Get ideal iSWAP
    _iSWAP = Gate('ISWAP').matrix()

    # Initialize circuit
    _circ = Circuit()

    # For each gate in circuit ..
    for gate in circuit:

        # Check if gate is close to SWAP
        if gate.n_qubits == 2 and gate.qubits and np.allclose(
                gate.matrix(), _iSWAP):

            # Get tags
            _tags = gate.tags if gate.provides('tags') else {}

            # Expand iSWAP
            _ext = [
                Gate('SWAP', qubits=gate.qubits, tags=_tags),
                Gate('CZ', qubits=gate.qubits, tags=_tags),
                Gate('P', qubits=[gate.qubits[0]], tags=_tags),
                Gate('P', qubits=[gate.qubits[1]], tags=_tags),
            ]

            # Append to circuit
            _circ.extend(_ext if gate.power == 1 else (
                g**-1 for g in reversed(_ext)))

        # Otherwise, just append
        else:
            _circ.append(deepcopy(gate))

    # Return circuit
    return _circ


def filter(circuit: iter,
           names: list[str] = any,
           qubits: list[any] = any,
           params: list[any] = any,
           n_qubits: int = any,
           n_params: int = any,
           virtual: bool = any,
           exact_match: bool = False,
           atol: float = 1e-8,
           **kwargs) -> iter:

    # Initialize
    f_circuit = iter(circuit)

    # Filter by name
    if names is not any:
        names = {str(name).upper() for name in names}
        f_circuit = (gate for gate in f_circuit if gate.name in names)

    # Filter by qubits
    if qubits is not any:
        if exact_match:
            qubits = tuple(qubits)
            f_circuit = (gate for gate in f_circuit
                         if gate.provides('qubits') and gate.qubits == qubits)
        else:
            qubits = set(qubits)
            f_circuit = (gate for gate in f_circuit
                         if gate.provides('qubits') and gate.qubits and
                         qubits.intersection(gate.qubits))

    # Filter by parameters
    if params is not any:

        def _isclose(x, y):
            try:
                _x = float(x)
                _y = float(y)
            except:
                return x == y
            else:
                return np.isclose(_x, _y, atol=atol)

        f_circuit = (gate for gate in f_circuit
                     if gate.provides('params') and gate.params and all(
                         _isclose(x, y) for x, y in zip(gate.params, params)))

    # Filter by number of qubits
    if n_qubits is not any:
        f_circuit = (gate for gate in f_circuit
                     if gate.provides('qubits') and gate.n_qubits == n_qubits)

    # Filter by number of parameters
    if n_params is not any:
        f_circuit = (gate for gate in f_circuit
                     if gate.provides('params') and gate.n_params == n_params)

    # Filter virtual gates
    if virtual is not any:
        f_circuit = (gate for gate in f_circuit if gate.isvirtual() == virtual)

    # Filter by tags
    for k, v in kwargs.items():

        # Define filter
        if exact_match:

            def _filter(gate):
                if gate.provides('tags'):
                    for k, v in kwargs.items():
                        if k not in gate.tags or (v is not any and
                                                  gate.tags[k] != v):
                            return False
                    return True
                else:
                    return False
        else:

            def _filter(gate):
                if gate.provides('tags'):
                    for k, v in kwargs.items():
                        if k in gate.tags and (v is any or gate.tags[k] == v):
                            return True
                    return False
                else:
                    return False

        f_circuit = (gate for gate in f_circuit if _filter(gate))

    return f_circuit
