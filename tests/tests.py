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

from hybridq.gate import Gate
from hybridq.utils import kron
from hybridq.gate.utils import get_available_gates
from hybridq.extras.random import get_random_gate, get_rqc, get_random_indexes
from hybridq.dm.circuit import Circuit as SuperCircuit
from hybridq.dm.circuit import simulation as dm_simulation
from hybridq.circuit import Circuit, simulation, utils
from hybridq.circuit.simulation import clifford
from hybridq.extras.io.cirq import to_cirq
from hybridq.extras.io.qasm import to_qasm, from_qasm
from hybridq.utils import sort, argsort, transpose, dot
from hybridq.utils.utils import _type
from functools import partial as partial_func
from opt_einsum import get_symbol, contract
from more_itertools import flatten
from itertools import chain
from tqdm.auto import tqdm
from warnings import warn
import numpy as np
import pytest
import cirq
import sys
import os

# Force to use random indexes
_get_rqc_non_unitary = partial_func(get_rqc,
                                    use_random_indexes=True,
                                    use_unitary_only=False)
_get_rqc_unitary = partial_func(get_rqc,
                                use_random_indexes=True,
                                use_unitary_only=True)


@pytest.fixture(autouse=True)
def set_seed():
    # Get random seed
    seed = np.random.randint(2**32 - 1)

    # Get state
    state = np.random.get_state()

    # Set seed
    np.random.seed(seed)

    # Print seed
    print(f"# Used seed [{os.environ['PYTEST_CURRENT_TEST']}]: {seed}",
          file=sys.stderr)

    # Wait for PyTest
    yield

    # Set state
    np.random.set_state(state)


################################ TEST UTILS ################################


@pytest.mark.parametrize(
    't', [t for t in ['float32', 'float64', 'float128'] for _ in range(100)])
def test_utils__to_complex(t):
    from hybridq.utils.dot import to_complex, to_complex_array

    # Get random shape
    shape = np.random.randint(2, 6, size=8)

    # Get random arrays
    a = np.random.random(shape).astype(t)
    b = np.random.random(shape).astype(t)
    c = to_complex(a, b)
    _a, _b = to_complex_array(c)

    # Check types
    assert (c.dtype == (a[:1] + 1j * b[:1]).dtype)
    assert (_a.dtype == a.dtype)
    assert (_b.dtype == b.dtype)

    # Check shape
    assert (c.shape == a.shape)
    assert (_a.shape == a.shape)
    assert (_b.shape == a.shape)

    # Check
    assert (np.allclose(c, a + 1j * b))
    assert (np.allclose(_a, a))
    assert (np.allclose(_b, b))


@pytest.mark.parametrize('order,alignment', [(o, a) for o in 'CF'
                                             for a in [16, 32, 64, 128]
                                             for _ in range(100)])
def test_utils__aligned_array(order, alignment):
    from hybridq.utils.aligned import array, asarray, empty, zeros, ones, isaligned

    # Get np.ndarray order
    def _get_order(a):
        order = 'C' if a.flags.c_contiguous else ''
        order += 'F' if a.flags.f_contiguous else ''
        return order

    # Get random shape
    shape = tuple(np.random.randint(2**4, size=1 + np.random.randint(5)) + 1)

    # Define available dtypes
    dtypes = [
        'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32',
        'uint64', 'float32', 'float64', 'float128', 'complex64', 'complex128'
    ]

    # Get random type
    dtype = np.dtype(np.random.choice(dtypes))

    # Print
    print('#  type:', dtype, file=sys.stderr)
    print('# shape:', shape, file=sys.stderr)
    print('# order:', order, file=sys.stderr)

    # Check all possible ways to generate an aligned array
    for __gen__ in [empty, ones, zeros, array]:

        # Generate an empty aligned array
        if __gen__ is array:
            r = np.asarray(np.random.random(shape), dtype=dtype, order=order)
            _a = array(r, alignment=alignment)

            # Checks
            assert (np.allclose(r, _a))
            assert (r.shape == _a.shape)
            assert (_get_order(r) == _get_order(_a))
        else:
            _a = __gen__(shape=shape,
                         dtype=dtype,
                         order=order,
                         alignment=alignment)

        # Get a new one
        a = array(_a, alignment=alignment)

        # Checks
        assert (not np.may_share_memory(a, _a))
        #
        assert (_a.shape == shape)
        assert (_a.dtype == dtype)
        assert (order in _get_order(_a))
        assert (isaligned(_a, alignment))
        if __gen__ == zeros:
            assert (np.allclose(_a, 0))
        elif __gen__ == ones:
            assert (np.allclose(_a, 1))
        #
        assert (a.shape == _a.shape)
        assert (a.dtype == _a.dtype)
        assert (order in _get_order(a))
        assert (isaligned(a, alignment))
        if __gen__ == zeros:
            assert (np.allclose(_a, 0))
        elif __gen__ == ones:
            assert (np.allclose(_a, 1))

        # These should be the same as a
        b1 = asarray(a, dtype=dtype, order=order, alignment=alignment)
        b2 = asarray(a, alignment=alignment)

        # Checks
        assert (b1 is a)
        assert (b2 is a)
        assert (np.may_share_memory(b1, a))
        assert (np.may_share_memory(b2, a))

        # These should be different from a
        _c1_dtype = next(t for t in dtypes if np.dtype(t) != a.dtype)
        c1 = asarray(a, dtype=_c1_dtype, alignment=alignment)
        c2 = asarray(a, order='C' if order is 'F' else 'F', alignment=alignment)

        # Checks
        assert (c1.shape == a.shape)
        assert (c1.dtype == _c1_dtype)
        assert ((c1.ctypes.data % alignment) == 0)
        assert (order in _get_order(c1))
        assert (not np.may_share_memory(c1, a))
        #
        if _get_order(a) == 'CF':
            assert (c2 is a)
        else:
            assert (c2.shape == a.shape)
            assert (c2.dtype == a.dtype)
            assert ((c2.ctypes.data % alignment) == 0)
            assert (('C' if order is 'F' else 'F') in _get_order(c2))
            assert (not np.may_share_memory(c2, a))


@pytest.mark.parametrize('t,n,backend', [(t + str(b), 14, backend)
                                         for t in ['float', 'int', 'uint']
                                         for b in [32, 64]
                                         for backend in ['numpy', 'jax']
                                         for _ in range(50)])
def test_utils__transpose(t, n, backend):
    # Get random vector
    v = np.reshape(np.random.randint(2**32 - 1, size=2**n).astype(t), (2,) * n)
    v0 = np.array(v)
    v1 = np.array(v)
    v2 = np.array(v)

    # Get random orders
    o_1 = np.random.permutation(range(n))
    o_2 = np.concatenate(
        (np.arange(n - 6), n - 6 + np.random.permutation(range(6))))

    # Get transposition
    to_1 = transpose(v0, o_1, raise_if_hcore_fails=True,
                     backend=backend).flatten()
    to_2 = transpose(v0, o_2, raise_if_hcore_fails=True,
                     backend=backend).flatten()
    v1 = transpose(v1,
                   o_1,
                   raise_if_hcore_fails=True,
                   inplace=True,
                   backend=backend)
    v2 = transpose(v2,
                   o_2,
                   raise_if_hcore_fails=True,
                   inplace=True,
                   backend=backend)

    # Check transposition
    assert (np.alltrue(v == v0))
    assert (np.alltrue(np.transpose(v, o_1).flatten() == to_1))
    assert (np.alltrue(np.transpose(v, o_2).flatten() == to_2))
    assert (np.alltrue(to_1 == v1.flatten()))
    assert (np.alltrue(to_2 == v2.flatten()))
    assert (np.alltrue(transpose(v, o_1, force_numpy=True).flatten() == to_1))
    assert (np.alltrue(transpose(v, o_2, force_numpy=True).flatten() == to_2))


@pytest.mark.parametrize('t,n,k,backend', [(t + str(b), 14, k, backend)
                                           for t in ['float'] for b in [32, 64]
                                           for k in [2, 3, 4, 5, 6]
                                           for backend in ['numpy', 'jax']
                                           for _ in range(20)])
def test_utils__dot(t, n, k, backend):
    from hybridq.utils.aligned import array

    # Generate random state
    psi = np.random.random((2, 2**n)).astype(t)
    psi = (psi.T / np.linalg.norm(psi, axis=1)).T
    psi1 = np.array(psi)
    psi2 = array(psi, alignment=32)

    # Generate random matrix
    U = (np.random.random((2**k, 2**k)) + 1j * np.random.random(
        (2**k, 2**k))).astype((1j * psi[0][:1]).dtype)

    # Generate random positions
    axes_b = np.random.choice(n, size=k, replace=False)

    b1 = dot(U,
             np.reshape(psi1, (2,) * (n + 1)),
             axes_b=axes_b,
             backend=backend,
             b_as_complex_array=True,
             force_numpy=True)

    b1h = dot(U,
              np.reshape(psi1, (2,) * (n + 1)),
              axes_b=axes_b,
              backend=backend,
              b_as_complex_array=True,
              raise_if_hcore_fails=True)

    psi2 = dot(U,
               np.reshape(psi2, (2,) * (n + 1)),
               axes_b=axes_b,
               backend=backend,
               b_as_complex_array=True,
               inplace=True,
               raise_if_hcore_fails=True)

    # Check
    assert (np.allclose(psi, psi1, atol=1e-3))
    assert (np.allclose(b1, b1h, atol=1e-3))
    assert (np.allclose(psi2, b1h, atol=1e-3))

    b1h_no_tr, tr1 = dot(U,
                         np.reshape(psi, (2,) * (n + 1)),
                         axes_b=axes_b,
                         backend=backend,
                         b_as_complex_array=True,
                         swap_back=False,
                         raise_if_hcore_fails=True)

    # Transpose back if needed
    if tr1 is not None:
        _br = transpose(b1h_no_tr[0], tr1, inplace=True)
        _bi = transpose(b1h_no_tr[1], tr1, inplace=True)

    # Check
    assert (np.allclose(b1, (b1h_no_tr if tr1 is None else (_br, _bi)),
                        atol=1e-3))

    b2 = dot(U,
             np.reshape(psi[0] + 1j * psi[1], (2,) * n),
             axes_b=axes_b,
             backend=backend,
             force_numpy=True)

    b2h = dot(U,
              np.reshape(psi[0] + 1j * psi[1], (2,) * n),
              axes_b=axes_b,
              backend=backend,
              raise_if_hcore_fails=True)

    # Check
    assert (np.allclose(b2, b2h, atol=1e-3))

    b2h_no_tr, tr2 = dot(U,
                         np.reshape(psi[0] + 1j * psi[1], (2,) * n),
                         axes_b=axes_b,
                         backend=backend,
                         swap_back=False,
                         raise_if_hcore_fails=True)

    # Transpose back if needed
    if tr2 is not None:
        b2h_no_tr = transpose(np.real(b2h_no_tr),
                              tr2) + 1j * transpose(np.imag(b2h_no_tr), tr2)

    # Check
    assert (np.allclose(b2, b2h_no_tr, atol=1e-3))


@pytest.mark.parametrize('n_qubits,n_gates', [(200, 5000) for _ in range(5)])
def test_utils__pickle(n_qubits, n_gates):
    import pickle

    # Get random circuit
    circuit = _get_rqc_non_unitary(n_qubits, n_gates)
    circuit[::10] = (utils.to_matrix_gate(Circuit([g])) for g in circuit[::10])
    circuit.append(
        Gate('STOC', gates=_get_rqc_non_unitary(10, 100), p=[1 / 100] * 100))

    # Dumps/loads with pickle
    circuit_copy = pickle.loads(pickle.dumps(circuit))

    # Check the two circuits are the same
    assert (circuit == circuit_copy)


@pytest.mark.parametrize('dummy', [_ for _ in range(50)])
def test_utils__sort_argsort(dummy):

    _n_floats = np.random.randint(1000)
    _n_ints = np.random.randint(1000)
    _n_tuples = np.random.randint(1000)
    _n_strings = np.random.randint(1000)

    # Add floats
    _array = (10000 * np.random.random(size=_n_floats)).tolist()

    # Add integers
    _array += np.random.randint(10000, size=_n_ints).tolist()

    # Add tuples
    _array += [tuple(x) for x in np.random.random(size=(_n_tuples, 2))]

    # Add strings
    _array += [
        ''.join(map(get_symbol, np.random.randint(1000, size=30)))
        for _ in range(_n_strings)
    ]

    # Random permutation
    _array = np.random.permutation(np.array(_array, dtype='object')).tolist()

    # Sort
    _sorted_array = sort(_array)

    # Check
    for _t in [float, tuple, str]:

        # Check array is sorted properly by type
        assert (sorted(filter(lambda x: _type(x) == _t, _array)) == list(
            filter(lambda x: _type(x) == _t, _sorted_array)))

        # Check that object of the same type are consecutive
        assert (np.alltrue(
            np.unique(
                np.diff(
                    list(
                        map(
                            lambda x: x[0],
                            filter(lambda x: _type(x[1]) == _t,
                                   enumerate(_sorted_array)))))) == [1]))

    # Check that argsort works properly
    assert (sort(_array) == [_array[i] for i in argsort(_array)])
    assert (sort(_array, reverse=True) == [
        _array[i] for i in argsort(_array, reverse=True)
    ])


################################ TEST GATES ################################


@pytest.mark.parametrize('dummy', [_ for _ in range(10)])
def test_gates__gates(dummy):
    for gate_name in get_available_gates():
        # Get Gate
        gate = Gate(gate_name)

        # Add qubits
        gate._on(np.random.choice(1024, size=gate.n_qubits, replace=False))

        # Add parameters
        if gate.provides('params'):
            gate._set_params(np.random.random(size=gate.n_params))

        # Add power
        gate._set_power(4 * np.random.random() - 2)

        # Get Matrix gate
        m_gate = Gate('MATRIX', qubits=gate.qubits, U=gate.matrix())

        # Get unitaries
        _U1 = gate.matrix(order=sort(gate.qubits))
        _U2 = m_gate.matrix(order=sort(gate.qubits))
        _U3 = cirq.unitary((to_cirq(Circuit([gate]))))

        # Check
        assert (gate.inv().isclose(gate**-1))
        assert (gate.isclose(m_gate))
        assert (gate.inv().isclose(m_gate.inv()))
        assert (np.allclose(_U1, _U2))
        assert (np.allclose(_U1, _U3))


@pytest.mark.parametrize('dummy', [_ for _ in range(50)])
def test_gates__gate_power(dummy):

    for gate_name in get_available_gates():
        # Get Gate
        gate = Gate(gate_name)

        # Add qubits
        gate._on(np.random.choice(1024, size=gate.n_qubits, replace=False))

        # Add parameters
        if gate.provides('params'):
            gate._set_params(np.random.random(size=gate.n_params))

        # Add power
        gate._set_power(4 * np.random.random() - 2)

        # Get matrix
        U = gate.matrix()

        # Check
        assert (np.allclose(U.dot(U.conj().T), np.eye(len(U)), atol=1e-3))
        assert (np.allclose(U.conj().T.dot(U), np.eye(len(U)), atol=1e-3))


@pytest.mark.parametrize('n_qubits', [4 for _ in range(100)])
def test_gates__matrix_gate(n_qubits):
    # Get random matrix
    U = np.random.random((2**n_qubits, 2**n_qubits))

    # Get gates
    g1 = Gate('MATRIX', U=U)
    g2 = Gate('MATRIX', U=U, copy=False)

    assert (g1.Matrix is not U)
    assert (g2.Matrix is U)


@pytest.mark.parametrize('n_qubits,depth', [(12, 200) for _ in range(5)])
def test_gates__cgates_1(n_qubits, depth):
    from hybridq.circuit.simulation.utils import prepare_state
    from hybridq.utils import dot
    from hybridq.gate import Projection
    from hybridq.gate import Control

    # Get random initial_state
    initial_state = ''.join(np.random.choice(list('01+-'), size=n_qubits))

    # Get random quantum circuit
    circuit = _get_rqc_non_unitary(n_qubits, depth)

    # Get qubits
    qubits = circuit.all_qubits()

    # Define how to get random qubits
    def _get_random_qubits(g):
        _q = list(set(qubits).difference(g.qubits))
        return [
            _q[x] for x in np.random.choice(len(_q),
                                            size=1 +
                                            np.random.choice(2, replace=False),
                                            replace=False)
        ]

    # Convert circuit to all cgates
    circuit = [Control(c_qubits=_get_random_qubits(g), gate=g) for g in circuit]

    # Simulate circuit
    psi1 = simulation.simulate(circuit,
                               initial_state=initial_state,
                               simplify=False,
                               verbose=True)

    # Simulate circuit by hand. Initialize state
    psi2 = prepare_state(initial_state)

    # Apply each gate
    for g in tqdm(circuit):
        # Get controlling qubits
        c_qubits = g.c_qubits

        # Project state
        _proj, _order = Projection('1' * len(c_qubits),
                                   c_qubits).apply(psi2,
                                                   order=qubits,
                                                   renormalize=False)

        # Check order hasn't changed
        assert (_order == qubits)

        # Apply matrix to projection and update state
        psi2 += dot(g.gate.matrix() - np.eye(2**g.gate.n_qubits),
                    _proj,
                    axes_b=[qubits.index(q) for q in g.gate.qubits],
                    inplace=True)

    # Check
    assert (np.allclose(psi1, psi2, atol=1e-3))


@pytest.mark.parametrize('n_qubits,depth', [(12, 200) for _ in range(5)])
def test_gates__cgates_2(n_qubits, depth):
    from hybridq.circuit.simulation.utils import prepare_state
    from hybridq.utils import dot
    from hybridq.gate import Projection
    from hybridq.gate import Control
    import pickle

    # Get random initial_state
    initial_state = ''.join(np.random.choice(list('01+-'), size=n_qubits))

    # Get random quantum circuit
    circuit = _get_rqc_non_unitary(n_qubits, depth)

    # Get qubits
    qubits = circuit.all_qubits()

    # Generate FunctionalGate from Gate
    def _get_fn(gate):
        # Get qubits
        qubits = gate.qubits

        # Get matrix
        U = gate.matrix()

        # Build function
        def f(self, psi, order):
            if not isinstance(psi, np.ndarray):
                raise ValueError("Only 'numpy.ndarray' are supported.")

            # Check dimension
            if not 0 <= (psi.ndim - len(order)) <= 1:
                raise ValueError("'psi' is not consistent with order")

            # Check if psi is split in real and imaginary part
            complex_array = psi.ndim > len(order)

            # If complex_array, first dimension must be equal to 2
            if complex_array and not psi.shape[0] == 2:
                raise ValueError("'psi' is not valid.")

            # Get axes
            axes = [
                next(i for i, y in enumerate(order) if y == x) for x in qubits
            ]

            # Apply matrix
            new_psi = dot(a=U,
                          b=psi,
                          axes_b=axes,
                          b_as_complex_array=complex_array,
                          inplace=True)

            return new_psi, order

        # Return FunctionalGate
        return Gate('fn', qubits=qubits, f=f)

    # Define how to get random qubits
    def _get_random_qubits(g):
        _q = list(set(qubits).difference(g.qubits))
        return [
            _q[x] for x in np.random.choice(len(_q),
                                            size=1 +
                                            np.random.choice(2, replace=False),
                                            replace=False)
        ]

    # Get controlling qubits
    c_qubits = [_get_random_qubits(g) for g in circuit]

    # Convert circuit to all cgates
    circuit_1 = Circuit(
        Control(c_qubits=cq, gate=g) for cq, g in zip(c_qubits, circuit))
    circuit_2 = Circuit(
        Control(c_qubits=cq, gate=_get_fn(g))
        for cq, g in zip(c_qubits, circuit))

    # Check pickle
    assert (circuit_1 == pickle.loads(pickle.dumps(circuit_1)))
    assert (circuit_2 == pickle.loads(pickle.dumps(circuit_2)))

    # Simulate circuit
    psi1 = simulation.simulate(circuit_1,
                               optimize='evolution',
                               initial_state=initial_state,
                               simplify=False,
                               verbose=True)
    psi2 = simulation.simulate(circuit_2,
                               optimize='evolution',
                               initial_state=initial_state,
                               simplify=False,
                               verbose=True)

    # Check
    assert (np.allclose(psi2, psi1, atol=1e-3))


@pytest.mark.parametrize('nq', [8 for _ in range(20)])
def test_gates__schmidt_gate(nq):
    from hybridq.gate import MatrixGate, SchmidtGate
    from hybridq.gate.utils import decompose

    # Get random gate
    g = utils.to_matrix_gate(_get_rqc_non_unitary(nq, 200))

    # Get random left/right qubits
    ln = np.random.randint(1, nq)
    rn = nq - ln

    # Decompose (with random left qubits)
    sg = decompose(g, [
        g.qubits[x]
        for x in np.random.choice(g.n_qubits, size=ln, replace=False)
    ])

    # Get matrix
    M1 = g.matrix(sg.gates[0].qubits + sg.gates[1].qubits)

    # Get matrix
    M2 = sg.Matrix

    # Check
    assert (np.allclose(M1, M2, atol=1e-3))


@pytest.mark.parametrize('n_qubits,k',
                         [(16, k) for k in range(1, 10) for _ in range(10)])
def test_gates__measure(n_qubits, k):
    from hybridq.gate.projection import _Projection
    from hybridq.utils.dot import to_complex, to_complex_array
    from hybridq.gate import Measure
    import pickle

    # Get a random state
    r = np.random.random((2,) * n_qubits) + 1j * np.random.random(
        (2,) * n_qubits)

    # Normalize state
    r /= np.linalg.norm(r.ravel())

    # Split to real and imaginary part
    r_split = to_complex_array(r)

    # Check
    assert (np.allclose(r_split[0] + 1j * r_split[1], r))

    # Get a random order
    order = tuple(np.random.randint(-2**31, 2**31, size=n_qubits))

    # Get measure
    M = Measure(qubits=np.random.choice(order, size=k, replace=False))

    # Test pickle
    assert (M == pickle.loads(pickle.dumps(M)))

    # Set numpy state
    _rng = np.random.get_state()

    # Get probabilities
    np.random.set_state(_rng)
    probs = M(r, order, get_probs_only=True)
    #
    np.random.set_state(_rng)
    probs_split = M(r_split, order, get_probs_only=True)

    # Check
    assert (np.allclose(probs, probs_split))

    # Get probabilities from projection
    _probs = [
        np.linalg.norm(
            _Projection(r,
                        axes=[order.index(q)
                              for q in M.qubits],
                        state=tuple(bin(s)[2:].zfill(M.n_qubits)),
                        renormalize=False).ravel())**2
        for s in range(2**M.n_qubits)
    ]

    # Check
    assert (np.allclose(probs, _probs, atol=1e-3))

    # Reset numpy and get state only
    np.random.set_state(_rng)
    state = M(r, order, get_state_only=True)
    #
    np.random.set_state(_rng)
    state_split = M(r_split, order, get_state_only=True)

    # Check
    assert (np.allclose(state, state_split))

    # Reset numpy and get projected state
    np.random.set_state(_rng)
    psi, new_order = M(r, order, renormalize=False)

    # Reset numpy and get normalized projected state
    np.random.set_state(_rng)
    psi_norm, new_order = M(r, order)
    #
    np.random.set_state(_rng)
    psi_norm_split, new_order_split = M(r_split, order)

    # Check
    assert (np.allclose(psi_norm,
                        to_complex(psi_norm_split[0], psi_norm_split[1])))
    assert (np.allclose(new_order, new_order_split))

    # Check order is unchanged
    assert (np.allclose(order, new_order))

    # Check normalization
    assert (np.isclose(np.linalg.norm(psi_norm.flatten()), 1, atol=1e-3))

    # Check that psi and psi_norm are the same after normalization
    assert (np.allclose(psi / np.linalg.norm(psi.flatten()),
                        psi_norm,
                        atol=1e-3))

    # Get projection
    _proj = tuple(map(int, bin(state)[2:].zfill(M.n_qubits)))
    _proj = tuple(
        _proj[M.qubits.index(x)] if x in M.qubits else slice(2) for x in order)

    # Check that the state correspond
    assert (np.allclose(r[_proj], psi[_proj], atol=1e-3))

    # Check that only projection is different from zero
    psi[_proj] = 0
    psi_norm[_proj] = 0
    assert (np.allclose(psi, 0, atol=1e-3))
    assert (np.allclose(psi_norm, 0, atol=1e-3))

    @np.vectorize
    def _get_prob(state):
        # Get state in bits
        state = bin(state)[2:].zfill(M.n_qubits)

        # Get projection
        proj = tuple(
            int(state[M.qubits.index(q)]) if q in M.qubits else slice(2)
            for q in order)

        # Return probability
        return np.linalg.norm(r[proj].flatten())**2

    # Get exact probabilities
    probs_ex = _get_prob(np.arange(2**M.n_qubits))
    probs_ex /= np.sum(probs_ex)

    # Check
    assert (np.allclose(probs_ex, probs))

    # Order shouldn't change
    assert (order == M(r, order)[1])


@pytest.mark.parametrize('n_qubits,k',
                         [(16, k) for k in range(1, 10) for _ in range(10)])
def test_gates__projection(n_qubits, k):
    from hybridq.gate import Projection
    import pickle

    # Get a random state
    r = np.random.random((2,) * n_qubits) + 1

    # Get a random order
    order = tuple(np.random.randint(-2**31, 2**31, size=n_qubits))

    # Get projection
    P = Projection(state=''.join(np.random.choice(list('01'), size=k)),
                   qubits=np.random.choice(order, size=k, replace=False))

    # Test pickle
    assert (P == pickle.loads(pickle.dumps(P)))

    # Get projected state
    psi, new_order = P(r, order, renormalize=False)

    # Order shouldn't change
    assert (order == new_order)

    # Get normalized projected state
    psi_norm, _ = P(r, order)

    # Check normalization
    assert (np.isclose(np.linalg.norm(psi_norm.flatten()), 1, atol=1e-3))

    # Check that psi and psi_norm are the same after normalization
    assert (np.allclose(psi / np.linalg.norm(psi.flatten()),
                        psi_norm,
                        atol=1e-3))

    # Get projection
    proj = tuple(
        int(P.state[P.qubits.index(q)]) if q in P.qubits else slice(2)
        for q in order)

    # Check that only elements in the projection are equal
    assert (np.allclose(r[proj], psi[proj]))

    # Check that once the projection is set to zero, everything must be zero
    psi[proj] = 0
    assert (np.allclose(psi, 0))

    # Get projected state
    psi, new_order = P(r, order, renormalize=True)

    # Check order hasn't changed
    assert (order == new_order)

    # Check normalization
    assert (np.isclose(np.linalg.norm(psi.flatten()), 1))

    # Check that only elements in the projection are equal
    assert (np.allclose(r[proj] / np.linalg.norm(r[proj].flatten()),
                        psi[proj],
                        atol=1e-3))

    # Check that once the projection is set to zero, everything must be zero
    psi[proj] = 0
    assert (np.allclose(psi, 0))


@pytest.mark.parametrize('dummy', [_ for _ in range(250)])
def test_gates__commutation(dummy):

    # Get two random qubits
    g1 = get_random_gate()
    g2 = get_random_gate()
    g1._on(np.random.choice(4, size=g1.n_qubits, replace=False))
    g2._on(np.random.choice(4, size=g2.n_qubits, replace=False))
    g12 = g1.qubits + tuple(q for q in g2.qubits if q not in g1.qubits)

    # Get corresponding matrix matrices
    U1 = utils.matrix(Circuit([g1] + [Gate('I', [q]) for q in g12]))
    U2 = utils.matrix(Circuit([g2] + [Gate('I', [q]) for q in g12]))

    # Check commutation
    assert (np.allclose(U1 @ U2, U2 @ U1,
                        atol=1e-5) == g1.commutes_with(g2, atol=1e-5))


################################ TEST GATE UTILS ################################


@pytest.mark.parametrize('n_qubits,n_ab',
                         [(9, k) for k in range(6) for _ in range(3)])
def test_gate_utils__merge_gates(n_qubits, n_ab):
    from hybridq.gate.utils import merge

    # Get sizes
    n_a = n_ab + np.random.randint(np.random.randint(1, n_qubits - n_ab - 1))
    n_b = n_qubits + n_ab - n_a

    # Get random indexes
    x_a = np.random.randint(2**32 - 1, size=(n_a - n_ab, 2))
    x_b = np.random.randint(2**32 - 1, size=(n_b - n_ab, 2))
    x_ab = np.random.randint(2**32 - 1, size=(n_ab, 2))
    x_a = tuple(
        tuple(x) for x in np.random.permutation(np.concatenate((x_a, x_ab))))
    x_b = tuple(
        tuple(x) for x in np.random.permutation(np.concatenate((x_b, x_ab))))

    # Check
    assert (len(x_a) == n_a)
    assert (len(x_b) == n_b)

    # Get random gates
    a1 = Gate('MATRIX', qubits=x_a, U=np.random.random((2**n_a, 2**n_a)))
    a2 = Gate('MATRIX', qubits=x_a, U=np.random.random((2**n_a, 2**n_a)))
    b1 = Gate('MATRIX', qubits=x_b, U=np.random.random((2**n_b, 2**n_b)))
    b2 = Gate('MATRIX', qubits=x_b, U=np.random.random((2**n_b, 2**n_b)))

    # Merge gates
    c1 = merge(a1, b1, a2, b2)
    c2 = merge(*Gate('TUPLE', gates=(a1, b1, a2, b2)))

    # Get matrix gate from utils
    _c = utils.to_matrix_gate(Circuit([a1, b1, a2, b2]))

    # Check
    assert (sort(c1.qubits) == sort(_c.qubits))
    assert (sort(c2.qubits) == sort(_c.qubits))
    assert (sort(Gate('TUPLE',
                      gates=(a1, b1, a2, b2)).qubits) == sort(_c.qubits))
    assert (np.allclose(c1.matrix(_c.qubits), _c.matrix(), atol=1e-3))
    assert (np.allclose(c2.matrix(_c.qubits), _c.matrix(), atol=1e-3))


@pytest.mark.parametrize('n_qubits,k', [(n, f)
                                        for n in range(2, 10)
                                        for f in range(2, 5) if f < n
                                        for _ in range(3)])
def test_gate_utils__decompose_gate(n_qubits, k):
    from hybridq.gate.utils import decompose, merge

    # Get random gate
    g = Gate('matrix',
             np.random.randint(2**32, size=n_qubits),
             U=np.random.random((2**n_qubits, 2**n_qubits)))

    # Get random subqubits
    qubits = np.random.choice(g.qubits, size=k, replace=False)

    # Decompose
    gd = decompose(g, qubits)

    # Merge gates and compute matrix
    W = sum((s * merge(a, b).matrix(g.qubits))
            for s, a, b in zip(gd.s, gd.gates[0], gd.gates[1]))

    # Check
    assert (np.allclose(W, g.matrix(), atol=1e-3))


################################ TEST CIRCUIT ################################


@pytest.mark.parametrize('n_qubits,n_gates', [(8, 200) for _ in range(5)])
def test_circuit__conj_T_adj_inv(n_qubits, n_gates):
    # Get random circuit
    circuit = _get_rqc_non_unitary(n_qubits, n_gates)

    # Check single gates
    assert (all(
        np.allclose(g.adj().matrix(), g.matrix().conj().T, atol=1e-3)
        for g in tqdm(circuit)))
    assert (all(
        np.allclose(g.conj().matrix(), g.matrix().conj(), atol=1e-3)
        for g in tqdm(circuit)))
    assert (all(
        np.allclose(g.T().matrix(), g.matrix().T, atol=1e-3)
        for g in tqdm(circuit)))
    assert (all(
        np.allclose(g.inv().matrix(), np.linalg.inv(g.matrix()), atol=1e-3)
        for g in tqdm(circuit)))

    assert (all(g1.adj().isclose(g2)
                for g1, g2 in tqdm(zip(reversed(circuit), circuit.adj()),
                                   total=len(circuit))))
    assert (all(
        g1.conj().isclose(g2)
        for g1, g2 in tqdm(zip(circuit, circuit.conj()), total=len(circuit))))
    assert (all(g1.T().isclose(g2)
                for g1, g2 in tqdm(zip(reversed(circuit), circuit.T()),
                                   total=len(circuit))))
    assert (all(g1.inv().isclose(g2)
                for g1, g2 in tqdm(zip(reversed(circuit), circuit.inv()),
                                   total=len(circuit))))

    # Get matrices
    U = utils.matrix(circuit)
    Ud = utils.matrix(circuit.adj())
    Uc = utils.matrix(circuit.conj())
    UT = utils.matrix(circuit.T())
    Ui = utils.matrix(circuit.inv())

    # Check
    assert (np.allclose(U.conj().T, Ud, atol=1e-3))
    assert (np.allclose(U.conj(), Uc, atol=1e-3))
    assert (np.allclose(U.T, UT, atol=1e-3))
    assert (np.allclose(U @ Ui, np.eye(U.shape[0]), atol=1e-3))
    assert (np.allclose(Ui @ U, np.eye(U.shape[0]), atol=1e-3))


@pytest.mark.parametrize('n_qubits,n_gates', [(12, 200) for _ in range(10)])
def test_circuit__projection(n_qubits, n_gates):
    import pickle

    # Generate random circuit
    circuit = _get_rqc_non_unitary(n_qubits, n_gates)

    # Get qubits
    qubits = circuit.all_qubits()
    n_qubits = len(qubits)

    # Generate random initial state
    initial_state = ''.join(np.random.choice(list('01+-'), size=n_qubits))

    # Add random projections
    for _ in range(10):
        # Get random qubits
        qs = [
            qubits[i]
            for i in np.random.choice(range(n_qubits), size=2, replace=False)
        ]

        # Get random projection
        ps = bin(np.random.randint(2**len(qs)))[2:].zfill(len(qs))

        # Add projection to circuit
        circuit.insert(np.random.choice(len(circuit)),
                       Gate('PROJ', qubits=qs, state=ps))

    # Test pickle
    circuit = pickle.loads(pickle.dumps(circuit))

    # Simulate circuit
    psi1 = simulation.simulate(circuit,
                               initial_state=initial_state,
                               verbose=True,
                               simplify=False,
                               optimize='evolution-hybridq')
    psi2 = simulation.simulate(circuit,
                               initial_state=initial_state,
                               verbose=True,
                               simplify=False,
                               optimize='evolution-einsum')

    # Check
    assert (np.allclose(psi1, psi2, atol=1e-3))


@pytest.mark.parametrize('n_qubits,n_gates', [(10, 50) for _ in range(10)])
def test_circuit__circuit(n_qubits, n_gates):

    # Generate rqc
    circuit = _get_rqc_unitary(n_qubits, n_gates)

    # Get random permutation of qubits
    _qubits = circuit.all_qubits()
    _perm_qubits = [_qubits[i] for i in np.random.permutation(len(_qubits))]

    # Get unitaries
    _U1 = utils.matrix(circuit)
    _U1b = utils.matrix(circuit, order=_perm_qubits)

    # Change back permutation using Gate('MATRIX')
    _U1b = Gate('MATRIX', qubits=_perm_qubits,
                U=_U1b).matrix(order=circuit.all_qubits())

    _U2 = utils.matrix(circuit.inv())
    _U3 = cirq.unitary(to_cirq(circuit))
    _U4 = utils.matrix(Circuit(
        utils.to_matrix_gate(c) for c in utils.compress(circuit, 4)),
                       max_compress=0)
    _U5 = utils.matrix(Circuit(
        utils.to_matrix_gate(c) for c in utils.moments(circuit)),
                       max_compress=4)

    # Check if everything mathes
    assert (np.allclose(_U1, _U2.T.conj(), atol=1e-3))
    assert (np.allclose(_U1, _U3, atol=1e-3))
    assert (np.allclose(_U1, _U4, atol=1e-3))
    assert (np.allclose(_U1, _U5, atol=1e-3))
    assert (np.allclose(_U1, _U1b, atol=1e-3))

    # Check closeness
    assert (circuit == circuit)
    assert (utils.isclose(circuit, circuit))
    try:
        assert (utils.isclose(
            circuit,
            Circuit(
                Gate('MATRIX', qubits=g.qubits, U=g.matrix())
                for g in circuit)))
        return None
    except:
        return circuit
    assert (utils.isclose(
        circuit.inv(),
        Circuit(
            Gate('MATRIX', qubits=g.qubits, U=g.matrix(), power=-1)
            for g in reversed(circuit))))
    assert (circuit != circuit + [Gate('X', [0])])
    assert (not utils.isclose(circuit, circuit + [Gate('X', [0])]))


################################ TEST CIRCUIT UTILS ################################


@pytest.mark.parametrize('n_qubits,n_gates',
                         [(n, 50) for n in range(4, 9) for _ in range(5)])
def test_circuit_utils__matrix(n_qubits, n_gates):

    # Get random circuit
    circuit = _get_rqc_unitary(n_qubits, n_gates)

    # Get random permutation
    order = np.random.permutation(circuit.all_qubits()).tolist()

    # Get matrix
    U1 = utils.matrix(circuit)
    U1b = utils.matrix(circuit, order=order)

    # Get unitary from cirq
    U2 = cirq.unitary(to_cirq(circuit))
    U2b = cirq.unitary(
        to_cirq(
            circuit,
            qubits_map={
                q: cirq.LineQubit(order.index(q)) for q in circuit.all_qubits()
            }))

    # Get matrix from matrix gate
    U3 = Gate('MATRIX', qubits=order, U=U1b).matrix(order=circuit.all_qubits())

    # Check that the two matrices are the same
    assert (np.allclose(U1, U2, atol=1e-3))
    assert (np.allclose(U1, U3, atol=1e-3))
    assert (np.allclose(U1b, U2b, atol=1e-3))


@pytest.mark.parametrize('n_qubits,depth,max_n_qubits',
                         [(10, 200, n) for n in [4, 8] for _ in range(2)])
def test_circuit_utils__compression(n_qubits, depth, max_n_qubits):
    # Get random circuit
    circuit = _get_rqc_non_unitary(n_qubits, depth)

    # Get random gate name
    gate_name = circuit[np.random.choice(len(circuit))].name

    # Compress circuit
    compr_circuit = utils.compress(circuit,
                                   max_n_qubits=max_n_qubits,
                                   verbose=True)

    # Compute unitaries
    U1 = utils.matrix(circuit, verbose=True)
    U2 = utils.matrix(Circuit(utils.to_matrix_gate(c) for c in compr_circuit),
                      verbose=True)

    # Check circuit and compr_circuit are the same
    assert (np.allclose(U1, U2, atol=1e-3))

    # Check that every compressed circuit has the right number
    # of qubits
    assert (all(len(c.all_qubits()) <= max_n_qubits for c in compr_circuit))


@pytest.mark.parametrize('n_qubits,depth,max_n_qubits',
                         [(10, 200, n) for n in [4, 8] for _ in range(2)])
def test_circuit_utils__compression_skip_name(n_qubits, depth, max_n_qubits):
    # Get random circuit
    circuit = _get_rqc_non_unitary(n_qubits, depth)

    # Get random gate name
    gate_name = circuit[np.random.choice(len(circuit))].name

    # Compress circuit
    compr_circuit = utils.compress(circuit,
                                   max_n_qubits=max_n_qubits,
                                   skip_compression=[gate_name],
                                   verbose=True)

    # Compute unitaries
    U1 = utils.matrix(circuit, verbose=True)
    U2 = utils.matrix(Circuit(utils.to_matrix_gate(c) for c in compr_circuit),
                      verbose=True)

    # Check circuit and compr_circuit are the same
    assert (np.allclose(U1, U2, atol=1e-3))

    # Check that every compressed circuit has the right number
    # of qubits
    assert (all(len(c.all_qubits()) <= max_n_qubits for c in compr_circuit))

    # Check that gates with gate.name == gate_name are not compressed
    assert (all(
        all(g.name != gate_name
            for g in c) if len(c) > 1 else True
        for c in compr_circuit))


@pytest.mark.parametrize('n_qubits,depth,max_n_qubits',
                         [(10, 200, n) for n in [4, 8] for _ in range(2)])
def test_circuit_utils__compression_skip_type(n_qubits, depth, max_n_qubits):
    # Import gate type
    from hybridq.gate.property import RotationGate

    # Get random circuit
    circuit = _get_rqc_non_unitary(n_qubits, depth)

    # Get random gate name
    gate_type = RotationGate

    # Compress circuit
    compr_circuit = utils.compress(circuit,
                                   max_n_qubits=max_n_qubits,
                                   skip_compression=[gate_type],
                                   verbose=True)

    # Compute unitaries
    U1 = utils.matrix(circuit, verbose=True)
    U2 = utils.matrix(Circuit(utils.to_matrix_gate(c) for c in compr_circuit),
                      verbose=True)

    # Check circuit and compr_circuit are the same
    assert (np.allclose(U1, U2, atol=1e-3))

    # Check that every compressed circuit has the right number
    # of qubits
    assert (all(len(c.all_qubits()) <= max_n_qubits for c in compr_circuit))

    # Check that gates which are instances of gate_type are not compressed
    assert (all(
        all(not isinstance(g, gate_type)
            for g in c) if len(c) > 1 else True
        for c in compr_circuit))


@pytest.mark.parametrize('n_qubits,n_gates', [(200, 2000) for _ in range(10)])
def test_circuit_utils__qasm(n_qubits, n_gates):

    # Generate rqc
    circuit = _get_rqc_non_unitary(n_qubits, n_gates)

    assert (Circuit(
        g.on([str(x) for x in g.qubits]) for g in circuit) == from_qasm(
            to_qasm(circuit)))


@pytest.mark.parametrize('use_matrix_commutation,max_n_qubits',
                         [(t, q) for t in [True, False] for q in range(2, 6)])
def test_circuit_utils__circuit_compress(use_matrix_commutation, max_n_qubits):

    # Generate rqc
    circuit = _get_rqc_non_unitary(20, 200)

    # Compress circuit
    compressed_circuit = utils.compress(
        circuit,
        use_matrix_commutation=use_matrix_commutation,
        max_n_qubits=max_n_qubits)

    # Check all sub-circuits have the right number of qubits
    assert (all(
        len(c.all_qubits()) <= max_n_qubits for c in compressed_circuit))

    # Two circuits should be identical
    assert (utils.isclose(Circuit(g for c in compressed_circuit for g in c),
                          circuit,
                          atol=1e-5,
                          verbose=True))


@pytest.mark.parametrize('use_matrix_commutation',
                         [t for t in [True, False] for _ in range(5)])
def test_circuit_utils__circuit_simplify_1(use_matrix_commutation):

    # Generate rqc
    circuit = _get_rqc_non_unitary(20, 200)

    # Circuit must completely simplify
    assert (not utils.simplify(circuit + circuit.inv(),
                               use_matrix_commutation=use_matrix_commutation,
                               verbose=True))

    # Generate rqc
    circuit = _get_rqc_non_unitary(10, 200)
    qubits = circuit.all_qubits()
    pinned_qubits = qubits[:1]
    circuit_pop = utils.pop(
        circuit, direction='right',
        pinned_qubits=pinned_qubits) + [Gate('X', pinned_qubits)] + utils.pop(
            circuit.inv(), direction='left', pinned_qubits=pinned_qubits)
    circuit = circuit + [Gate('X', pinned_qubits)] + circuit.inv()

    # Get matrix
    _U1 = utils.matrix(circuit, verbose=True)

    # Simplify circuit
    circuit = utils.simplify(circuit,
                             use_matrix_commutation=use_matrix_commutation,
                             verbose=True)
    # Add identities if qubits are missing
    circuit += [
        Gate('I', [q]) for q in set(qubits).difference(circuit.all_qubits())
    ]
    # Get matrix
    _U2 = utils.matrix(circuit, verbose=True)

    # Add identities if qubits are missing
    circuit_pop += [
        Gate('I', [q]) for q in set(qubits).difference(circuit_pop.all_qubits())
    ]
    # Get matrix
    _U3 = utils.matrix(circuit_pop, verbose=True)

    # Check
    assert (np.allclose(_U1, _U2, atol=1e-3))
    assert (np.allclose(_U1, _U3, atol=1e-3))


def test_circuit_utils__circuit_simplify_2(n_qubits=30):
    # Get fully connected circuit with all just phases
    circuit = Circuit(
        Gate('CPHASE', qubits=[i, j], params=[np.random.random()])
        for i in range(n_qubits)
        for j in range(i + 1, n_qubits))

    # Randomize
    circuit = Circuit(
        circuit[x]
        for x in np.random.permutation(len(circuit))) + circuit.inv()

    # Circuit must be empty
    assert (not utils.simplify(
        circuit, verbose=True, use_matrix_commutation=True))


@pytest.mark.parametrize('n_qubits,n_gates', [(20, 200) for _ in range(5)])
def test_circuit_utils__circuit_simplify_3(n_qubits, n_gates):
    # Default random get_rqc should give a unitary matrix
    circuit = get_rqc(n_qubits, n_gates)

    # Simplify
    assert (not utils.simplify(circuit + circuit.conj().T(), verbose=True))
    assert (not utils.simplify(circuit + circuit.T().conj(), verbose=True))
    assert (not utils.simplify(circuit + circuit.inv(), verbose=True))
    assert (not utils.simplify(circuit.conj().T() + circuit, verbose=True))
    assert (not utils.simplify(circuit.T().conj() + circuit, verbose=True))
    assert (not utils.simplify(circuit.inv() + circuit, verbose=True))


@pytest.mark.parametrize(
    'n_qubits', [(n_qubits) for n_qubits in range(4, 21, 4) for _ in range(10)])
def test_circuit_utils__prepare_state(n_qubits):
    _get = {
        '0': np.array([1, 0]),
        '1': np.array([0, 1]),
        '+': np.array([1, 1]) / np.sqrt(2),
        '-': np.array([1, -1]) / np.sqrt(2),
    }

    # Get random initial_state
    initial_state = ''.join(np.random.choice(list('01+-'), size=n_qubits))

    # Get state using kron
    _s1 = np.reshape(kron(*[_get[s] for s in initial_state]), (2,) * n_qubits)
    _s2 = simulation.prepare_state(initial_state)

    assert (_s1.shape == _s2.shape)
    assert (np.allclose(_s1, _s2))

    # Get random initial_state
    initial_state = ''.join(np.random.choice(list('01'), size=n_qubits))

    # Get state using kron
    _s1 = np.reshape(kron(*[_get[s] for s in initial_state]), (2,) * n_qubits)
    _s2 = simulation.prepare_state(initial_state)

    assert (_s1.shape == _s2.shape)
    assert (np.allclose(_s1, _s2))

    # Get random initial_state
    initial_state = '+' * n_qubits

    # Get state using kron
    _s1 = np.reshape(kron(*[_get[s] for s in initial_state]), (2,) * n_qubits)
    _s2 = simulation.prepare_state(initial_state)

    assert (_s1.shape == _s2.shape)
    assert (np.allclose(_s1, _s2))


################################ TEST CLIFFORD GATES/CIRCUIT ################################


def test_cliffords__check_gates():
    from hybridq.gate.utils import get_clifford_gates
    from hybridq.circuit.simulation.clifford import update_pauli_string
    from itertools import product

    # Get all available clifford gates
    gates = get_clifford_gates()

    # For each gate ..
    for gate in tqdm(gates):
        # Get the number of qubits
        n_qubits = Gate(gate).n_qubits

        # Generate circuit
        paulis = [
            Circuit(x) for x in product(*[[Gate(g, [q])
                                           for g in 'IXYZ']
                                          for q in range(n_qubits)])
        ]

        # Update pauli strings
        res = [
            update_pauli_string(Circuit([Gate(gate, range(n_qubits))]), p)
            for p in paulis
        ]

        # Check that no branches have happened
        assert (all(len(x) == 1 for x in res))


@pytest.mark.parametrize('n_qubits,n_gates,compress,parallel',
                         [(6, 12, c, p) for _ in range(5) for c in [0, 4]
                          for p in [True, False]])
def test_cliffords__circuit_1(n_qubits, n_gates, compress, parallel):

    # Get random circuit
    circuit = _get_rqc_unitary(n_qubits, n_gates)

    # Reorder accordingly to circuit
    qubits = circuit.all_qubits()

    # Get number of qubits from circuit
    n_qubits = len(qubits)

    # Get random paulis
    paulis = Circuit(
        Gate(g, [q])
        for q, g in zip(qubits, np.random.choice(list('XYZ'), size=n_qubits)))

    # Reduce Pauli operators
    all_op = clifford.update_pauli_string(circuit,
                                          paulis,
                                          simplify=False,
                                          remove_id_gates=False,
                                          parallel=parallel,
                                          max_first_breath_branches=4,
                                          sleep_time=0,
                                          compress=compress,
                                          verbose=True)

    if len(all_op) > 512:
        warn('Skipping test: too many operators.')
        pytest.skip()

    # Split the circuit in two parts
    c1 = circuit[:len(circuit) // 2]
    c2 = circuit[len(circuit) // 2:]

    # Pad circuits with identities
    c1 += Circuit(
        Gate('I', [q])
        for q in set(circuit.all_qubits()).difference(c1.all_qubits()))
    c2 += Circuit(
        Gate('I', [q])
        for q in set(circuit.all_qubits()).difference(c2.all_qubits()))

    # Check
    assert (c1.all_qubits() == c2.all_qubits())

    # Apply only the second half
    _partial_op2 = clifford.update_pauli_string(c2,
                                                paulis,
                                                verbose=True,
                                                parallel=parallel,
                                                compress=compress)

    # Apply the first half
    op2 = clifford.update_pauli_string(c1,
                                       _partial_op2,
                                       verbose=True,
                                       parallel=parallel,
                                       compress=compress)

    # Check
    assert (all(
        np.isclose(all_op[k], op2[k], atol=1e-3) for k in chain(all_op, op2)))
    # Contruct full operator
    U1 = np.zeros(shape=(2**n_qubits, 2**n_qubits), dtype='complex64')
    for op, ph in tqdm(all_op.items()):

        # Update operator
        U1 += ph * utils.matrix(
            Circuit(Gate(_op, [_q]) for _q, _op in zip(qubits, op)))

    # Get exact operator
    U2 = utils.matrix(circuit + paulis + circuit.inv())

    # Check
    assert (np.allclose(U1, U2, atol=1e-3))

    # Check identity
    all_op = clifford.update_pauli_string(
        circuit,
        Circuit(Gate('I', [q]) for q in circuit.all_qubits()),
        parallel=parallel,
        simplify=False,
        remove_id_gates=False,
        max_first_breath_branches=4,
        sleep_time=0,
        compress=compress,
        verbose=True)

    # Check
    assert (len(all_op) == 1 and 'I' * n_qubits in all_op and
            np.isclose(all_op['I' * n_qubits], 1, atol=1e-3))


@pytest.mark.parametrize('n_qubits,n_gates', [(200, 1000) for _ in range(5)])
def test_cliffords__circuit_2(n_qubits, n_gates):
    """
    Check that pure Clifford circuits do not branch.
    """

    # Get random circuit
    circuit = _get_rqc_unitary(n_qubits,
                               n_gates,
                               randomize_power=False,
                               use_clifford_only=True)

    # Get qubits
    qubits = circuit.all_qubits()

    # Get the actual number of qubits
    n_qubits = len(qubits)

    # Get random paulis
    paulis = Circuit(
        Gate(g, [q])
        for q, g in zip(qubits, np.random.choice(list('XYZ'), size=n_qubits)))

    # Get matrix without compression
    all_op, infos = clifford.update_pauli_string(circuit,
                                                 paulis,
                                                 parallel=False,
                                                 simplify=False,
                                                 sleep_time=0,
                                                 return_info=True,
                                                 compress=2,
                                                 remove_id_gates=False,
                                                 verbose=True)

    # Checks
    assert (len(all_op) == 1)
    assert (np.isclose(np.abs(next(iter(all_op.values()))), 1, atol=1e-3))
    assert (infos['n_explored_branches'] == 2)
    assert (infos['largest_n_branches_in_memory'] == 1)
    assert (infos['log2_n_expected_branches'] == 0)


################################ TEST SIMULATION ################################


@pytest.mark.parametrize('n_qubits,depth', [(12, 100) for _ in range(10)])
def test_simulation_1__tensor_trace(n_qubits, depth):
    # Get alphabet
    from string import ascii_letters
    from opt_einsum import contract

    # Get random quantum circuit
    circuit = _get_rqc_non_unitary(n_qubits, depth)

    # Initialize initial/final state
    state = bin(np.random.randint(4**n_qubits - 1))[2:].zfill(2 * n_qubits)

    # Initialize positions and letters
    pos = np.fromiter(range(2 * n_qubits), dtype='int')
    let = np.fromiter(ascii_letters, dtype='U1')

    # Add random open qubits
    _p = np.random.choice(pos, size=6, replace=False)
    pos = np.setdiff1d(pos, _p)
    state = ''.join('.' if i in _p else x for i, x in enumerate(state))

    # Add 1-qubit trace
    _p1 = np.random.choice(pos, size=5, replace=False).tolist()
    _l1 = np.random.choice(let, size=len(_p1), replace=False)
    pos = np.setdiff1d(pos, _p1)
    let = np.setdiff1d(let, _l1)
    state = ''.join(
        _l1[_p1.index(i)] if i in _p1 else x for i, x in enumerate(state))

    # Add 2-qubit trace
    _p2 = np.random.choice(pos, size=4, replace=False).tolist()
    _l2 = np.random.choice(let, size=len(_p2) // 2, replace=False)
    pos = np.setdiff1d(pos, _p2)
    let = np.setdiff1d(let, _l2)
    state = ''.join(
        _l2[_p2.index(i) // 2] if i in _p2 else x for i, x in enumerate(state))

    # Add 4-qubit trace
    _p4 = np.random.choice(pos, size=8, replace=False).tolist()
    _l4 = np.random.choice(let, size=len(_p4) // 4, replace=False)
    pos = np.setdiff1d(pos, _p4)
    let = np.setdiff1d(let, _l4)
    state = ''.join(
        _l4[_p4.index(i) // 4] if i in _p4 else x for i, x in enumerate(state))

    # Split as initial/final state
    initial_state = state[:n_qubits]
    final_state = state[n_qubits:]

    # Get matrix of the circuit
    U = utils.matrix(circuit, verbose=True)

    # Reshape and traspose matrix to be consistent with tensor
    U = np.transpose(
        np.reshape(U, (2,) * 2 * n_qubits),
        list(range(n_qubits, 2 * n_qubits)) + list(range(n_qubits)))

    # Simulate circuit using tensor contraction
    res_tn = simulation.simulate(circuit,
                                 initial_state=initial_state,
                                 final_state=final_state,
                                 optimize='tn',
                                 verbose=True)

    # Check shape of tensor is consistent with open qubits
    assert (len(res_tn.shape) == state.count('.'))

    # Properly order qubits in U
    order = [x for x, s in enumerate(state) if s in '01'
            ] + [x for x, s in enumerate(state) if s == '.'] + _p4[::4] + _p4[
                1::4] + _p4[2::4] + _p4[3::4] + _p2[::2] + _p2[1::2] + _p1
    U = np.transpose(U, order)

    # Get number of projected qubits
    n_proj = sum(s in '01' for s in state)

    # Get number of open qubits
    n_open = sum(s == '.' for s in state)

    # Get number of k-qubit traces
    n1 = len(_p1)
    n2 = len(_p2)
    n4 = len(_p4)

    # Project qubits
    U = np.reshape(U, (2**n_proj, 4**n_qubits // 2**n_proj))[int(
        ''.join(s for s in state if s in '01'), 2)]

    # Sum over the 1-qubit traces
    U = np.sum(np.reshape(U, (2**(n_open + n2 + n4), 2**n1)), axis=1)

    # Trace over the 2-qubit trace
    U = np.einsum('...ii',
                  np.reshape(U, (2**(n_open + n4),) + (2**(n2 // 2),) * 2))

    # Trace over the 4-qubit trace
    U = np.einsum('...iiii', np.reshape(U, (2**n_open,) + (2**(n4 // 4),) * 4))

    # Check that the tensor match the transformed matrix
    assert (np.allclose(U.flatten(), res_tn.flatten(), atol=1e-3))


@pytest.mark.parametrize(
    'n_qubits',
    [(n_qubits) for n_qubits in range(16, 25, 4) for _ in range(10)])
def test_simulation_1__initialize_state_1a(n_qubits):

    # Get random initial_state
    initial_state = ''.join(np.random.choice(list('01'), size=n_qubits))

    _s1 = simulation.prepare_state(initial_state)
    _s2 = simulation.simulate(
        circuit=Circuit(Gate('I', [q]) for q in range(n_qubits)),
        initial_state=initial_state,
        remove_id_gates=False,
        optimize='evolution',
        verbose=False,
    )

    assert (np.allclose(_s1, _s2))


@pytest.mark.parametrize(
    'n_qubits',
    [(n_qubits) for n_qubits in range(16, 25, 4) for _ in range(10)])
def test_simulation_1__initialize_state_1b(n_qubits):

    # Get initial_state
    initial_state = '0' * n_qubits

    _s1 = simulation.prepare_state(initial_state)
    _s2 = simulation.simulate(
        circuit=Circuit(Gate('I', [q]) for q in range(n_qubits)),
        initial_state=initial_state,
        remove_id_gates=False,
        optimize='evolution',
        verbose=False,
    )

    assert (np.allclose(_s1, _s2))

    # Get initial_state
    initial_state = '+' * n_qubits

    _s1 = simulation.prepare_state(initial_state)
    _s2 = simulation.simulate(
        circuit=Circuit(Gate('I', [q]) for q in range(n_qubits)),
        initial_state=initial_state,
        remove_id_gates=False,
        optimize='evolution',
        verbose=False,
    )

    assert (np.allclose(_s1, _s2))


@pytest.mark.parametrize(
    'n_qubits',
    [(n_qubits) for n_qubits in range(16, 25, 4) for _ in range(10)])
def test_simulation_1__initialize_state_2(n_qubits):

    # Get random initial_state
    initial_state = ''.join(np.random.choice(list('01+-'), size=n_qubits))

    _s1 = simulation.prepare_state(initial_state)
    _s2 = simulation.simulate(
        circuit=Circuit(Gate('I', [q]) for q in range(n_qubits)),
        initial_state=initial_state,
        remove_id_gates=False,
        optimize='evolution',
        verbose=False,
    )

    assert (np.allclose(_s1, _s2))


@pytest.mark.parametrize('n_qubits,depth', [(12, 200) for _ in range(3)])
def test_simulation_2__tuple(n_qubits, depth):
    from more_itertools import chunked
    from hybridq.gate.utils import merge
    import pickle

    # Generate random circuit
    circuit = _get_rqc_non_unitary(n_qubits, depth)

    # Generate random initial state
    initial_state = ''.join(np.random.choice(list('01+-'), size=n_qubits))

    # Get tuples
    c1 = Circuit(Gate('TUPLE', gates=gs) for gs in chunked(circuit, 4))

    # Merge tuples
    c2 = Circuit(merge(Gate('TUPLE', gates=gs)) for gs in chunked(circuit, 5))

    # Check pickle
    assert (c1 == pickle.loads(pickle.dumps(c1)))
    assert (c2 == pickle.loads(pickle.dumps(c2)))

    # Get single tuple
    g = Gate('TUPLE', gates=circuit)

    psi1 = simulation.simulate(circuit,
                               initial_state=initial_state,
                               verbose=True)
    psi2 = simulation.simulate(c1, initial_state=initial_state, verbose=True)
    psi3 = simulation.simulate(c2, initial_state=initial_state, verbose=True)
    psi4 = simulation.simulate(Circuit([g]),
                               initial_state=initial_state,
                               verbose=True)

    assert (sort(g.qubits) == sort(circuit.all_qubits()))
    assert (np.allclose(psi1, psi2, atol=1e-3))
    assert (np.allclose(psi1, psi3, atol=1e-3))


@pytest.mark.parametrize('n_qubits,depth', [(12, 200) for _ in range(3)])
def test_simulation_2__message(n_qubits, depth):
    from hybridq.extras.gate import Gate as ExtraGate
    from hybridq.extras.gate import MessageGate
    from more_itertools import flatten
    from io import StringIO

    # Get buffer
    file = StringIO()

    # Generate random circuit
    circuit = _get_rqc_non_unitary(n_qubits, depth)

    # Add messages
    circuit_msg = Circuit(
        flatten(
            (g, ExtraGate('MESSAGE', qubits=tuple(), message=f'{x}', file=file))
            for x, g in enumerate(circuit)))

    # Message counts when checking for equality
    assert (circuit != circuit_msg)

    # If Gate('MESSAGE') has no qubits, it shouldn't interfere with compression
    compr = utils.compress(utils.simplify(circuit), max_n_qubits=4)
    compr_msg = utils.compress(utils.simplify(circuit_msg),
                               max_n_qubits=4,
                               skip_compression=[MessageGate])

    # Check all MessageGate's are isolated
    assert (all(
        all(not isinstance(g, MessageGate)
            for g in c) if len(c) > 1 else True
        for c in compr_msg))

    # Compression should be the same once MessageGate's are removed
    assert ([
        c for c in compr_msg if len(c) > 1 or not isinstance(c[0], MessageGate)
    ] == compr)

    # Get final states
    psi = simulation.simulate(circuit, initial_state='0', verbose=True)
    psi_msg = simulation.simulate(circuit_msg, initial_state='0', verbose=True)

    # Final states should be the same
    assert (np.allclose(psi, psi_msg, atol=1e-3))

    # Wind back StringIO
    file.seek(0)

    # Get all messages
    msg = file.readlines()

    # Check all messages are printed
    assert (sorted(range(len(circuit))) == sorted(int(x.strip()) for x in msg))


@pytest.mark.parametrize('n_qubits,depth', [(14, 400) for _ in range(3)])
def test_simulation_2__fn(n_qubits, depth):
    from hybridq.utils.dot import dot
    import pickle

    # Generate FunctionalGate from Gate
    def _get_fn(gate):
        # Get qubits
        qubits = gate.qubits

        # Get matrix
        U = gate.matrix()

        # Build function
        def f(self, psi, order):
            if not isinstance(psi, np.ndarray):
                raise ValueError("Only 'numpy.ndarray' are supported.")

            # Check dimension
            if not 0 <= (psi.ndim - len(order)) <= 1:
                raise ValueError("'psi' is not consistent with order")

            # Check if psi is split in real and imaginary part
            complex_array = psi.ndim > len(order)

            # Get axes
            axes = [
                next(i for i, y in enumerate(order) if y == x) for x in qubits
            ]

            # Apply matrix
            new_psi = dot(a=U,
                          b=psi,
                          axes_b=axes,
                          b_as_complex_array=complex_array,
                          inplace=True)

            return new_psi, order

        # Return FunctionalGate
        return Gate('fn', qubits=qubits, f=f)

    # Get random circuit
    circuit = _get_rqc_non_unitary(n_qubits, depth)

    # Fix n_qubits
    n_qubits = len(circuit.all_qubits())

    # Convert to FunctionalGate
    circuit_fn = Circuit(
        _get_fn(g) if np.random.random() < 0.5 else g for g in circuit)

    # Test pickle
    assert (circuit_fn == pickle.loads(pickle.dumps(circuit_fn)))

    # Generate random initial state
    initial_state = ''.join(np.random.choice(list('01+-'), size=n_qubits))

    # Simulate the two circuits
    psi = simulation.simulate(circuit,
                              initial_state=initial_state,
                              verbose=True)
    psi_fn_1 = simulation.simulate(circuit_fn,
                                   optimize='evolution-hybridq',
                                   initial_state=initial_state,
                                   verbose=True)
    psi_fn_2 = simulation.simulate(circuit_fn,
                                   optimize='evolution-einsum',
                                   initial_state=initial_state,
                                   verbose=True)

    # Check
    assert (np.allclose(psi, psi_fn_1, atol=1e-3))
    assert (np.allclose(psi, psi_fn_2, atol=1e-3))


@pytest.mark.parametrize('n_qubits,depth,n_samples',
                         [(12, 100, 200) for _ in range(3)])
def test_simulation_2__stochastic(n_qubits, depth, n_samples):
    import pickle

    # Get first random circuits
    circuit_1 = Circuit(
        utils.to_matrix_gate(g) for g in utils.compress(utils.simplify(
            _get_rqc_non_unitary(n_qubits, depth // 2)),
                                                        max_n_qubits=4))

    # Fix number of qubits (in case not all n_qubits qubits has beed used)
    n_qubits = len(circuit_1.all_qubits())

    # Get second random circuits reusing the indexes in circuit_1
    circuit_2 = Circuit(
        utils.to_matrix_gate(g) for g in utils.compress(utils.simplify(
            _get_rqc_non_unitary(
                n_qubits, depth // 2, indexes=circuit_1.all_qubits())),
                                                        max_n_qubits=4))

    # Get random initial_state
    initial_state = ''.join(np.random.choice(list('01'), size=3)) + ''.join(
        np.random.choice(list('01+-'), size=n_qubits - 3))

    # Add a stochastic gate
    _prob = np.random.random(20)
    _prob /= np.sum(_prob)
    _gates = _get_rqc_non_unitary(n_qubits, 20, indexes=circuit_1.all_qubits())
    _stoc_gate = Gate('STOC', gates=_gates, p=_prob)

    # Check pickle
    assert (_stoc_gate == pickle.loads(pickle.dumps(_stoc_gate)))

    # Get exact result
    _psi_exact = np.zeros((2,) * n_qubits, dtype='complex64')
    for gate, p in tqdm(zip(_stoc_gate.gates, _stoc_gate.p)):
        _psi_exact += p * simulation.simulate(circuit_1 + [gate] + circuit_2,
                                              initial_state=initial_state,
                                              optimize='evolution',
                                              simplify=False,
                                              compress=0)

    # Sample
    _psi_sample = np.zeros((2,) * n_qubits, dtype='complex64')
    for _ in tqdm(range(n_samples)):
        _psi_sample += simulation.simulate(circuit_1 + [_stoc_gate] + circuit_2,
                                           initial_state=initial_state,
                                           optimize='evolution',
                                           allow_sampling=True,
                                           simplify=False,
                                           compress=0)
    _psi_sample /= n_samples

    # Check if close
    assert (np.allclose(_psi_exact, _psi_sample, atol=1 / np.sqrt(n_samples)))


@pytest.mark.parametrize('n_qubits,depth',
                         [(n_qubits, 200) for n_qubits in range(6, 10, 2)])
def test_simulation_3__simulation(n_qubits, depth):

    # Get random initial_state
    initial_state = ''.join(np.random.choice(list('01'), size=3)) + ''.join(
        np.random.choice(list('01+-'), size=n_qubits - 3))

    # Get random circuit
    circuit = _get_rqc_unitary(n_qubits, depth)

    # Get state using matrix
    _p = np.reshape(
        utils.matrix(circuit, verbose=True).dot(
            simulation.prepare_state(initial_state).flatten()), (2,) * n_qubits)

    # Get states
    _p1 = simulation.simulate(circuit,
                              optimize='evolution',
                              simplify=False,
                              initial_state=initial_state,
                              verbose=True)
    _p2 = simulation.simulate(circuit,
                              optimize='evolution-einsum-greedy',
                              simplify=False,
                              initial_state=initial_state,
                              verbose=True)
    _p2b = np.reshape(
        cirq.Simulator().simulate(to_cirq(circuit),
                                  initial_state=simulation.prepare_state(
                                      initial_state)).final_state_vector,
        (2,) * n_qubits)

    # Compress circuit
    circuit = Circuit(
        utils.to_matrix_gate(c) for c in utils.compress(circuit, 2))
    _p3 = simulation.simulate(circuit,
                              simplify=False,
                              optimize='evolution',
                              initial_state=initial_state,
                              verbose=True)

    assert (np.isclose(np.linalg.norm(_p.flatten()), 1))
    assert (np.isclose(np.linalg.norm(_p1.flatten()), 1))
    assert (np.isclose(np.linalg.norm(_p2.flatten()), 1))
    assert (np.isclose(np.linalg.norm(_p3.flatten()), 1))
    assert (np.allclose(_p, _p1, atol=1e-3))
    assert (np.allclose(_p, _p2, atol=1e-3))
    assert (np.allclose(_p, _p2b, atol=1e-3))
    assert (np.allclose(_p, _p3, atol=1e-3))

    try:
        _p4 = simulation.simulate(circuit,
                                  simplify=False,
                                  optimize='tn',
                                  initial_state=initial_state,
                                  max_n_slices=2**12,
                                  verbose=True)
    except ValueError:
        if str(sys.exc_info()[1])[:15] == "Too many slices":
            warn('Skipping test: ' + str(sys.exc_info()[1]))
            pytest.skip()
        else:
            raise sys.exc_info()[0](sys.exc_info()[1])
    except:
        raise sys.exc_info()[0](sys.exc_info()[1])

    assert (np.isclose(np.linalg.norm(_p4.flatten()), 1))
    assert (np.allclose(_p, _p4, atol=1e-3))

    # Specify some output qubits
    final_state = np.random.choice(list('01'), size=n_qubits)
    final_state[np.random.choice(n_qubits,
                                 size=int(n_qubits / 2),
                                 replace=False)] = '.'
    final_state = ''.join(final_state)
    try:
        _p5 = simulation.simulate(circuit,
                                  optimize='tn',
                                  simplify=False,
                                  max_n_slices=2**12,
                                  initial_state='...' + initial_state[3:],
                                  final_state=final_state,
                                  verbose=True)
    except ValueError:
        if str(sys.exc_info()[1])[:15] == "Too many slices":
            warn('Skipping test: ' + str(sys.exc_info()[1]))
            pytest.skip()
        else:
            raise sys.exc_info()[0](sys.exc_info()[1])
    except:
        raise sys.exc_info()[0](sys.exc_info()[1])

    # Compare with exact
    xpos = [x for x, s in enumerate(final_state) if s == '.']
    _map = ''.join([get_symbol(x) for x in range(n_qubits)])
    _map += '->'
    _map += ''.join(
        ['' if x in xpos else get_symbol(x) for x in range(n_qubits)])
    _map += ''.join([get_symbol(x) for x in xpos])
    _p5b = np.reshape(contract(_map, np.reshape(_p1, [2] * n_qubits)),
                      [2**(n_qubits - len(xpos)), 2**len(xpos)])
    _p5b = _p5b[int(
        final_state.replace('.', '').zfill(n_qubits - len(xpos)), 2)]

    assert (_p5.shape == (2,) * (3 + final_state.count('.')))
    assert (np.allclose(_p5[tuple(int(x) for x in initial_state[:3])].flatten(),
                        _p5b,
                        atol=1e-3))

    # Reduce maximum largest intermediate
    _p6_tn, (_p6_info, _p6_opt) = simulation.simulate(circuit,
                                                      optimize='tn',
                                                      simplify=False,
                                                      initial_state='...' +
                                                      initial_state[3:],
                                                      final_state=final_state,
                                                      tensor_only=True,
                                                      verbose=True)
    try:
        _p6 = simulation.simulate(_p6_tn,
                                  optimize=(_p6_info, _p6_opt),
                                  max_largest_intermediate=2**10,
                                  max_n_slices=2**12,
                                  verbose=True)
    except ValueError:
        if str(sys.exc_info()[1])[:15] == "Too many slices":
            warn('Skipping test: ' + str(sys.exc_info()[1]))
            pytest.skip()
        else:
            raise sys.exc_info()[0](sys.exc_info()[1])
    except:
        raise sys.exc_info()[0](sys.exc_info()[1])

    assert (np.allclose(_p5, _p6, atol=1e-3))


@pytest.mark.parametrize('n_qubits,depth',
                         [(n_qubits, 600) for n_qubits in range(16, 23, 2)])
def test_simulation_4__simulation_large(n_qubits, depth):

    # Get random initial_state
    initial_state = ''.join(np.random.choice(list('01'), size=3)) + ''.join(
        np.random.choice(list('01+-'), size=n_qubits - 3))

    # Get random circuit
    circuit = utils.simplify(_get_rqc_non_unitary(n_qubits, depth))

    # Get states
    _p1_c64 = simulation.simulate(circuit,
                                  optimize='evolution',
                                  compress=4,
                                  simplify=False,
                                  initial_state=initial_state,
                                  complex_type='complex64',
                                  verbose=True)
    _p1_c128 = simulation.simulate(circuit,
                                   optimize='evolution',
                                   compress=8,
                                   simplify=False,
                                   initial_state=initial_state,
                                   complex_type='complex128',
                                   verbose=True)
    _p2 = simulation.simulate(circuit,
                              optimize='evolution-einsum',
                              simplify=False,
                              initial_state=initial_state,
                              verbose=True)

    assert (_p1_c64.dtype == 'complex64')
    assert (_p1_c128.dtype == 'complex128')
    assert (np.allclose(_p1_c64, _p2, atol=1e-3))
    assert (np.allclose(_p1_c128, _p2, atol=1e-3))


@pytest.mark.parametrize('n_qubits,depth',
                         [(n_qubits, 200) for n_qubits in range(6, 13, 2)])
def test_simulation_5__expectation_value_1(n_qubits, depth):

    # Get random initial_state
    initial_state = ''.join(np.random.choice(list('01+-'), size=n_qubits))

    # Get random circuit
    circuit = _get_rqc_unitary(n_qubits, depth)

    # Get random operator
    op = _get_rqc_unitary(2, 3, indexes=circuit.all_qubits()[:2])

    v1 = simulation.expectation_value(state=simulation.simulate(
        circuit,
        initial_state,
        optimize='evolution',
        simplify=False,
        remove_id_gates=False,
        verbose=True),
                                      op=op,
                                      qubits_order=circuit.all_qubits(),
                                      remove_id_gates=False,
                                      simplify=False,
                                      verbose=True)
    v2 = simulation.simulate(circuit + op + circuit.inv(),
                             initial_state=initial_state,
                             final_state=initial_state,
                             optimize='tn',
                             simplify=False,
                             remove_id_gates=False,
                             verbose=True)

    # Check
    assert (np.isclose(v1, v2))


@pytest.mark.parametrize('n_qubits,depth',
                         [(n_qubits, 25) for n_qubits in range(6, 13, 2)])
def test_simulation_5__expectation_value_2(n_qubits, depth):

    # Get random circuit
    circuit = _get_rqc_unitary(n_qubits, depth)

    # Re-adjust number of qubits
    n_qubits = len(circuit.all_qubits())

    # Get random initial_state
    initial_state = ''.join(np.random.choice(list('01+-'), size=n_qubits))

    # Get random operator
    _qubits = circuit.all_qubits()
    op = Circuit(
        Gate(p, [q]) for q, p in zip([
            _qubits[i]
            for i in np.random.choice(len(_qubits), size=2, replace=False)
        ], np.random.choice(list('IXYZ'), size=2)))

    v1 = simulation.expectation_value(state=simulation.simulate(
        circuit,
        initial_state,
        optimize='evolution',
        simplify=False,
        remove_id_gates=False),
                                      op=op,
                                      qubits_order=circuit.all_qubits(),
                                      verbose=False)
    v2 = simulation.simulate(circuit + op + circuit.inv(),
                             initial_state=initial_state,
                             final_state=initial_state,
                             simplify=False,
                             remove_id_gates=False,
                             optimize='tn',
                             verbose=False)

    v3 = simulation.clifford.expectation_value(circuit,
                                               op,
                                               initial_state=initial_state,
                                               compress=4,
                                               verbose=True,
                                               parallel=True)

    assert (np.isclose(v1, v2, atol=1e-3))
    assert (np.isclose(v1, v3, atol=1e-3))


@pytest.mark.parametrize('n_qubits,depth',
                         [(n_qubits, 200) for n_qubits in range(6, 21, 4)])
def test_simulation_5__iswap(n_qubits, depth):

    # Get random initial_state
    initial_state = ''.join(np.random.choice(list('01+-'), size=n_qubits))

    # Get random circuit
    circuit = _get_rqc_non_unitary(n_qubits, depth, randomize_power=False)

    # Expand iswap
    circuit_exp, qubits_order = utils.remove_swap(utils.expand_iswap(circuit))

    # Get states
    _p1 = simulation.simulate(circuit_exp,
                              optimize='evolution',
                              simplify=False,
                              initial_state=initial_state,
                              verbose=True)
    _p2 = simulation.simulate(circuit,
                              optimize='evolution',
                              simplify=False,
                              initial_state=initial_state,
                              verbose=True)

    # Get qubits_map
    _qubits_map = {q: x for x, q in enumerate(circuit.all_qubits())}

    # Reorder state
    _map = ''.join([get_symbol(x) for x in range(len(qubits_order))])
    _map += '->'
    _map += ''.join([
        get_symbol(_qubits_map[x])
        for x, _ in sort(qubits_order.items(), key=lambda x: x[1])
    ])
    #
    _p1 = contract(_map, np.reshape(_p1, [2] * n_qubits))

    # Check
    assert (np.allclose(_p1, _p2, atol=1e-3))


################################ TEST DENSITY MATRICES ################################


@pytest.mark.parametrize('n_qubits,k,ndim', [(7, k, ndim) for k in range(1, 4)
                                             for ndim in range(0, 3)
                                             for _ in range(5)])
def test_dm_0__supergate_1(n_qubits, k, ndim):
    from hybridq.dm.gate.utils import to_matrix_supergate
    from hybridq.dm.gate import KrausSuperGate

    # Generate some random gates
    gates = tuple(_get_rqc_unitary(n_qubits, k))

    # Generate a random s
    if ndim == 0:
        s_1 = 1
        s_2 = s_1 * np.eye(len(gates))
    elif ndim == 1:
        s_1 = np.random.random(len(gates))
        s_1 /= np.linalg.norm(s_1)
        s_2 = np.diag(s_1)
    elif ndim == 2:
        s_1 = np.random.random((len(gates), len(gates)))
        s_1 /= np.linalg.norm(s_1)
        s_2 = s_1
    else:
        raise NotImplementedError

    # Get Kraus operator
    K = KrausSuperGate(gates=gates, s=s_1)
    K = to_matrix_supergate(K)

    # Get matrix corresponding to the operator
    M1 = K.Matrix

    # Get left/right qubits
    l_qubits, r_qubits = K.qubits

    def _merge(l_g, r_g, c):
        from hybridq.circuit.utils import to_matrix_gate
        from hybridq.circuit import Circuit
        from hybridq.gate import MatrixGate

        # Get partial left/right qubits
        l_q = [(0, q) for q in l_g.qubits]
        r_q = [(1, q) for q in r_g.qubits]

        # Get missing qubits
        m_q = tuple((0, q) for q in l_qubits if q not in l_g.qubits)
        m_q += tuple((1, q) for q in r_qubits if q not in r_g.qubits)

        # Get right order (first left qubits, then right qubits)
        order = [(0, q) for q in l_qubits] + [(1, q) for q in r_qubits]

        # Get matrix from Circuit
        g = to_matrix_gate(
            Circuit([l_g.on(l_q),
                     r_g.on(r_q),
                     Gate('I', qubits=m_q)]))

        # Get U with right order and multiplied by c
        g = MatrixGate(c * g.matrix(order), qubits=g.qubits)

        # Return matrix
        return g.Matrix

    # Get Matrix
    M2 = np.sum([
        _merge(gates[i], gates[j].conj(), s_2[i, j])
        for i in range(len(s_2))
        for j in range(len(s_2))
    ],
                axis=0)

    # Check
    assert (np.allclose(M1, M2, atol=1e-3))


@pytest.mark.parametrize('nq', [8 for _ in range(20)])
def test_dm_0__supergate_2(nq):
    from hybridq.dm.gate import KrausSuperGate, MatrixSuperGate
    from hybridq.gate import MatrixGate, SchmidtGate
    from hybridq.gate.utils import decompose

    # Get random gate
    g = utils.to_matrix_gate(_get_rqc_non_unitary(nq, 200))

    # Get random left/right qubits
    ln = np.random.randint(1, nq)
    rn = nq - ln

    # Decompose (with random left qubits)
    sg = decompose(g, [
        g.qubits[x]
        for x in np.random.choice(g.n_qubits, size=ln, replace=False)
    ])

    # Get KrausSuperGate
    K1 = KrausSuperGate(s=sg.s,
                        gates=[sg.gates[0], [g.conj() for g in sg.gates[1]]])
    K2 = MatrixSuperGate(Map=K1.Matrix,
                         l_qubits=K1.gates[0].qubits,
                         r_qubits=K1.gates[1].qubits)

    # Get matrix
    M1 = g.matrix(sg.gates[0].qubits + sg.gates[1].qubits)

    # Get matrix
    M2a = K1.Matrix
    M2b = K2.Matrix

    # Check
    assert (np.allclose(M1, M2a, atol=1e-3))
    assert (np.allclose(M1, M2b, atol=1e-3))


################################ TEST SUPERSIMULATION ################################


@pytest.mark.parametrize('n_qubits,n_gates', [(12, 200) for _ in range(3)])
def test_dm_1__simulation_1(n_qubits, n_gates):
    from hybridq.circuit.simulation.utils import prepare_state
    from hybridq.dm.gate import KrausSuperGate
    from scipy.linalg import eigvalsh

    # Get RQC
    circuit = _get_rqc_unitary(n_qubits, n_gates)

    # Get random initial state
    initial_state = ''.join(np.random.choice(list('01+-'), size=n_qubits))

    # Get state by using state evolution
    psi_1 = simulation.simulate(circuit,
                                initial_state=initial_state,
                                verbose=True)

    # Get density matrix
    rho_1 = dm_simulation.simulate(circuit,
                                   initial_state=initial_state,
                                   verbose=True)

    # Get matrix
    _rho_1 = np.reshape(rho_1, (2**n_qubits, 2**n_qubits))

    # Density matrix should be hermitian
    assert (np.allclose(_rho_1, _rho_1.conj().T, atol=1e-3))

    # Density matrix should be idempotent
    assert (np.allclose(_rho_1, _rho_1 @ _rho_1, atol=1e-3))

    # Density matrix^2 should have trace == 1
    assert (np.isclose(np.trace(_rho_1 @ _rho_1), 1, atol=1e-3))

    # Density matrix should be semi-positive definite
    assert (np.alltrue(np.round(eigvalsh(_rho_1), 5) >= 0))

    # Checks
    assert (np.allclose(np.kron(psi_1.ravel(),
                                psi_1.ravel().conj()),
                        rho_1.ravel(),
                        atol=1e-3))


@pytest.mark.parametrize('n_qubits,n_gates',
                         [(q, 60) for _ in range(4) for q in [4, 8]])
def test_dm_2__simulation_2(n_qubits, n_gates):
    from hybridq.dm.gate import KrausSuperGate, BaseSuperGate
    from hybridq.gate import BaseGate
    from scipy.linalg import eigvalsh
    from hybridq.utils import dot
    import pickle

    # Get random s
    def _get_s(n):
        s = np.random.random(size=n)
        s /= np.sum(s**2)
        return s

    # Get random circuit
    circuit = SuperCircuit(_get_rqc_unitary(n_qubits, n_gates))

    # Get qubits
    qubits = circuit.all_qubits()[0]

    # Add KrausOperators
    for _ in range(10):
        circuit.insert(
            np.random.randint(len(circuit)),
            KrausSuperGate(gates=get_rqc(
                4,
                4,
                indexes=[
                    qubits[x]
                    for x in np.random.choice(n_qubits, size=4, replace=False)
                ]),
                           s=_get_s(4)))

    # Check pickle
    assert (circuit == pickle.loads(pickle.dumps(circuit)))

    # Get left and right qubits
    l_qubits, r_qubits = circuit.all_qubits()

    # Get number of qubits
    n_l, n_r = len(l_qubits), len(r_qubits)

    # Get random initial state
    initial_state = ''.join(np.random.choice(list('01+-'), size=(n_l + n_r)))

    # Get density matrix forcing the use of SuperCircuit's
    rho_1a = dm_simulation.simulate(circuit,
                                    initial_state=initial_state,
                                    verbose=True,
                                    simplify=dict(use_matrix_commutation=False),
                                    compress=dict(max_n_qubits=4,
                                                  use_matrix_commutation=False),
                                    optimize='evolution')

    try:
        rho_1b = dm_simulation.simulate(
            circuit,
            initial_state=initial_state,
            verbose=True,
            max_n_slices=16,
            max_largest_intermediate=2**20,
            simplify=dict(use_matrix_commutation=False),
            compress=dict(max_n_qubits=2, use_matrix_commutation=False),
            optimize='tn')
    except ValueError:
        if str(sys.exc_info()[1])[:15] == "Too many slices":
            warn('Skipping test: ' + str(sys.exc_info()[1]))
            pytest.skip()
        else:
            raise sys.exc_info()[0](sys.exc_info()[1])
    except:
        raise sys.exc_info()[0](sys.exc_info()[1])

    # Checks
    assert (np.allclose(rho_1a, rho_1b, atol=1e-3))

    # Initialize state
    rho_2 = simulation.prepare_state(initial_state)

    for gate in tqdm(circuit):
        # Get qubits and map
        if isinstance(gate, BaseSuperGate):
            # Compute Kraus map
            K = gate.map()

            # Get qubits
            qubits = gate.qubits

        elif isinstance(gate, BaseGate):
            # Get matrix
            U = gate.matrix()

            # Get qubits
            qubits = gate.qubits

            # Get number of qubits
            nq = len(qubits)

            # Compute Kraus map
            K = np.reshape(np.kron(U.ravel(), U.ravel().conj()), (2**nq,) * 4)
            K = np.reshape(np.transpose(K, (0, 2, 1, 3)),
                           (np.prod(U.shape),) * 2)

            # Compute qubits
            qubits = (qubits, qubits)

        else:
            raise NotImplementedError(f"'type(gate).__name__' not supported.")

        # Get axes
        axes = [l_qubits.index(q) for q in qubits[0]
               ] + [n_l + r_qubits.index(q) for q in qubits[1]]

        # Multiply K to state
        rho_2 = dot(K, rho_2, axes_b=axes, inplace=True)

    # Checks
    assert (np.allclose(rho_1a, rho_2, atol=1e-3))
    assert (np.allclose(rho_1b, rho_2, atol=1e-3))


@pytest.mark.parametrize('n_qubits,n_gates', [(8, 60) for _ in range(4)])
def test_dm_3__simulation_3(n_qubits, n_gates):
    from hybridq.gate.measure import _Measure
    from scipy.linalg import eigvalsh
    from string import ascii_letters

    # Get RQC
    circuit = _get_rqc_unitary(n_qubits, n_gates)

    # Get random initial state
    initial_state = ''.join(np.random.choice(list('01+-'), size=n_qubits))

    # Get indexes for open qubits
    index_open_qubits = sorted(np.random.choice(n_qubits, size=2,
                                                replace=False))

    # Get final state, including the right projections
    final_state = ''.join('.' if i in index_open_qubits else c
                          for i, c in enumerate(ascii_letters[:n_qubits]))
    final_state += final_state

    # Get state by using state evolution
    psi_1 = simulation.simulate(circuit,
                                initial_state=initial_state,
                                verbose=True)

    # Get the expectation value using tensor contraction
    rho_1 = dm_simulation.simulate(circuit,
                                   initial_state=initial_state,
                                   final_state=final_state,
                                   optimize='tn',
                                   verbose=True)

    # Get matrix
    _rho_1 = np.reshape(rho_1, (2**len(index_open_qubits),) * 2)

    # Density matrix should be hermitian
    assert (np.allclose(_rho_1, _rho_1.conj().T, atol=1e-3))

    # Density matrix should be semi-positive definite
    assert (np.alltrue(np.round(eigvalsh(_rho_1), 5) >= 0))

    # Trace qubits using einsum
    _rho = np.einsum(
        ''.join(ascii_letters[i] for i in range(n_qubits)) + ',' +
        ''.join(ascii_letters[i + n_qubits if i in index_open_qubits else i]
                for i in range(n_qubits)), psi_1, psi_1.conj())

    # Check
    assert (np.allclose(_rho, rho_1, atol=1e-3))

    # Get probabilities of the projected states
    probs = _Measure(psi_1, axes=index_open_qubits, get_probs_only=True)

    # Check normalization
    assert (np.isclose(np.sum(probs), 1, atol=1e-3))

    # Check
    assert (np.allclose(np.diag(_rho_1), probs, atol=1e-3))


#########################################################################
