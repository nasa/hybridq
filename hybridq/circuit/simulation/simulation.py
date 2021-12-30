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

Types
-----
**`Array`**: `numpy.ndarray`

**`TensorNetwork`**: `quimb.tensor.TensorNetwork`

**`ContractionInfo`**: `(opt_einsum.contract.PathInfo, cotengra.hyper.HyperOptimizer)`
"""

from __future__ import annotations
from os import environ
from os.path import basename

_mpi_env = '_' in environ and basename(environ['_']) in ['mpiexec', 'mpirun']
_detect_mpi = 'DISABLE_MPI_AUTODETECT' not in environ and _mpi_env

import ctypes
from warnings import warn
from hybridq.gate import Gate
from hybridq.utils import isintegral
from hybridq.gate import property as pr
from hybridq.circuit import Circuit
from tqdm.auto import tqdm
from typing import TypeVar
from opt_einsum.contract import PathInfo
from opt_einsum import contract, get_symbol
from more_itertools import flatten
from sys import stderr
from time import time
import numpy as np
from hybridq.utils import sort, argsort, aligned
from hybridq.utils.transpose import _swap_core
from hybridq.utils.dot import _dot_core, _to_complex_core, _log2_pack_size
from hybridq.circuit.simulation.utils import prepare_state
import hybridq.circuit.utils as utils

# Define types
Array = TypeVar('Array')
TensorNetwork = TypeVar('TensorNetwork')
ContractionInfo = TypeVar('ContractionInfo')


def simulate(circuit: {Circuit, TensorNetwork},
             initial_state: any = None,
             final_state: any = None,
             optimize: any = 'evolution',
             backend: any = 'numpy',
             complex_type: any = 'complex64',
             tensor_only: bool = False,
             simplify: {bool, dict} = True,
             remove_id_gates: bool = True,
             use_mpi: bool = None,
             atol: float = 1e-8,
             verbose: bool = False,
             **kwargs) -> any:
    """
    Frontend to simulate `Circuit` using different optimization models and
    backends.

    Parameters
    ----------
    circuit: {Circuit, TensorNetwork}
        Circuit to simulate.
    initial_state: any, optional
        Initial state to use.
    final_state: any, optional
        Final state to use (only valid for `optimize='tn'`).
    optimize: any, optional
        Optimization to use. At the moment, HybridQ supports two optimizations:
        `optimize='evolution'` (equivalent to `optimize='evolution-hybridq'`)
        and `optimize='tn'` (equivalent to `optimize='cotengra'`).
        `optimize='evolution'` takes an `initial_state` (it can either be a
        string, which is processed using ```prepare_state``` or an `Array`)
        and evolve the quantum state accordingly to `Circuit`. Alternatives are:

        - `optimize='evolution-hybridq'`: use internal `C++` implementation for
          quantum state evolution that uses vectorization instructions (such as
          AVX instructions for Intel processors). This optimization method is
          best suitable for `CPU`s.
        - `optimize='evolution-einsum'`: use `einsum` to perform the evolution
          of the quantum state (via `opt_einsum`). It is possible to futher
          specify optimization for `opt_einsum` by using
          `optimize='evolution-einsum-opt'` where `opt` is one of the available
          optimization in `opt_einsum.contract` (default: `auto`). This
          optimization is best suitable for `GPU`s and `TPU`s (using
          `backend='jax'`).

        `optimize='tn'` (or, equivalently, `optimize='cotengra'`) performs the
        tensor contraction of `Circuit` given an `initial_state` and a
        `final_state` (both must be a `str`). Valid tokens for both
        `initial_state` and `final_state` are:

        - `0`: qubit is set to `0` in the computational basis,
        - `1`: qubit is set to `1` in the computational basis,
        - `+`: qubit is set to `+` state in the computational basis,
        - `-`: qubit is set to `-` state in the computational basis,
        - `.`: qubit is left uncontracted.

        Before the actual contraction, `cotengra` is called to identify an
        optimal contraction. Such contraction is then used to perform the tensor
        contraction.

        If `Circuit` is a `TensorNetwork`, `optimize` must be a
        valid contraction (see `tensor_only` parameter).
    backend: any, optional
        Backend used to perform the simulation. Backend must have `tensordot`,
        `transpose` and `einsum` methods.
    complex_type: any, optional
        Complex type to use for the simulation.
    tensor_only: bool, optional
        If `True` and `optimize=None`, `simulate` will return a
        `TensorNetwork` representing `Circuit`. Otherwise, if
        `optimize='cotengra'`, `simulate` will return the `tuple`
        ```(TensorNetwork```, ```ContractionInfo)```. ```TensorNetwork``` and
        and ```ContractionInfo``` can be respectively used as values for
        `circuit` and `optimize` to perform the actual contraction.
    simplify: {bool, dict}, optional
        Circuit is simplified before the simulation using
        `circuit.utils.simplify`. If non-empty `dict` is provided, `simplify`
        is passed as arguments for `circuit.utils.simplity`.
    remove_id_gates: bool, optional
        Identity gates are removed before to perform the simulation.
        If `False`, identity gates are kept during the simulation.
    use_mpi: bool, optional
        Use `MPI` if available. Unless `use_mpi=False`, `MPI` will be used if
        detected (for instance, if `mpiexec` is used to called HybridQ). If
        `use_mpi=True`, force the use of `MPI` (in case `MPI` is not
        automatically detected).
    atol: float, optional
        Use `atol` as absolute tollerance.
    verbose: bool, optional
        Verbose output.

    Returns
    -------
    Output of `simulate` depends on the chosen parameters.

    Other Parameters
    ----------------
    parallel: int (default: False)
        Parallelize simulation (where possible). If `True`, the number of
        available cpus is used. Otherwise, a `parallel` number of threads is
        used.
    compress: {int, dict} (default: auto)
        Select level of compression for ```circuit.utils.compress```, which is
        run on `Circuit` prior to perform the simulation. If non-empty `dict`
        is provided, `compress` is passed as arguments for
        `circuit.utils.compress`. If `optimize=evolution`, `compress` is set to
        `4` by default. Otherwise, if `optimize=tn`, `compress` is set to `2`
        by default.
    allow_sampling: bool (default: False)
        If `True`, `Gate`s that provide the method `sample` will not be sampled.
    sampling_seed: int (default: None)
        If provided, `numpy.random` state will be saved before sampling and
        `sampling_seed` will be used to sample `Gate`s. `numpy.random` state will
        be restored after sampling.
    block_until_ready: bool (default: True)
        When `backend='jax'`, wait till the results are ready before returning.
    return_numpy_array: bool (default: True)
        When `optimize='hybridq'` and `return_numpy_array` is `False, a `tuple`
        of two `np.ndarray` is returned, corresponding to the real and
        imaginary part of the quantu state. If `True`, the real and imaginary
        part are copied to a single `np.ndarray` of complex numbers.
    return_info: bool (default: False)
        Return extra information collected during the simulation.
    simplify_tn: str (default: 'RC')
        Simplification to apply to `TensorNetwork`. Available simplifications as
        specified in `quimb.tensor.TensorNetwork.full_simplify`.
    max_largest_intermediate: int (default: 2**26)
        Largest intermediate which is allowed during simulation. If
        `optimize='evolution'`, `simulate` will raise an error if the
        largest intermediate is larger than `max_largest_intermediate`.  If
        `optimize='tn'`, slicing will be applied to fit the contraction in
        memory.
    target_largest_intermediate: int (default: 0)
        Stop `cotengra` if a contraction having the largest intermediate smaller
        than `target_largest_intermediate` is found.
    max_iterations: int (default: 1)
        Number of `cotengra` iterations to find optimal contration.
    max_time: int (default: 120)
        Maximum number of seconds allowed to `cotengra` to find optimal
        contraction for each iteration.
    max_repeats: int (default: 16)
        Number of `cotengra` steps to find optimal contraction for each
        iteration.
    temperatures: list[float] (default: [1.0, 0.1, 0.01])
        Temperatures used by `cotengra` to find optimal slicing of the tensor
        network.
    max_n_slices: int (default: None)
        If specified, `simulate` will raise an error if the number of
        slices to fit the tensor contraction in memory is larger than
        `max_n_slices`.
    minimize: str (default: 'combo')
        Cost function to minimize while looking for the best contraction (see
        `cotengra` for more information).
    methods: list[str] (default: ['kahypar', 'greedy'])
        Heuristics used by `cotengra` to find optimal contraction.
    optlib: str (default: 'baytune')
        Library used by `cotengra` to tune hyper-parameters while looking for
        the best contraction.
    sampler: str (default: 'GP')
        Sampler used by `cotengra` while looking for the contraction.
    cotengra: dict[any, any] (default: {})
        Extra parameters to pass to `cotengra`.
    """

    # Set defaults
    kwargs.setdefault('allow_sampling', False)
    kwargs.setdefault('sampling_seed', None)

    # Convert simplify
    simplify = simplify if isinstance(simplify, bool) else dict(simplify)

    # Checks
    if tensor_only and type(optimize) == str and 'evolution' in optimize:
        raise ValueError(
            f"'tensor_only' is not support for optimize={optimize}")

    # Try to convert to circuit
    try:
        circuit = Circuit(circuit)
    except:
        pass

    # Simplify circuit
    if isinstance(circuit, Circuit):
        # Flatten circuit
        circuit = utils.flatten(circuit)

        # If 'sampling_seed' is provided, use it
        if kwargs['sampling_seed'] is not None:
            # Store numpy.random state
            __state = np.random.get_state()

            # Set seed
            np.random.seed(int(kwargs['sampling_seed']))

        # If stochastic gates are present, randomly sample from them
        circuit = Circuit(g.sample() if isinstance(g, pr.StochasticGate) and
                          kwargs['allow_sampling'] else g for g in circuit)

        # Restore numpy.random state
        if kwargs['sampling_seed'] is not None:
            np.random.set_state(__state)

        # Get qubits
        qubits = circuit.all_qubits()
        n_qubits = len(qubits)

        # Prepare state
        def _prepare_state(state):
            if isinstance(state, str):
                if len(state) == 1:
                    state *= n_qubits
                if len(state) != n_qubits:
                    raise ValueError(
                        "Wrong number of qubits for initial/final state.")
                return state
            else:
                # Convert to np.ndarray
                state = np.asarray(state)
                # For now, it only supports "qubits" ...
                if any(x != 2 for x in state.shape):
                    raise ValueError(
                        "Only qubits of dimension 2 are supported.")
                # Check number of qubits is consistent
                if state.ndim != n_qubits:
                    raise ValueError(
                        "Wrong number of qubits for initial/final state.")
                return state

        # Prepare initial/final state
        initial_state = None if initial_state is None else _prepare_state(
            initial_state)
        final_state = None if final_state is None else _prepare_state(
            final_state)

        # Strip Gate('I')
        if remove_id_gates:
            circuit = Circuit(gate for gate in circuit if gate.name != 'I')
        # Simplify circuit
        if simplify:
            circuit = utils.simplify(
                circuit,
                remove_id_gates=remove_id_gates,
                atol=atol,
                verbose=verbose,
                **(simplify if isinstance(simplify, dict) else {}))

        # Stop if qubits have changed
        if circuit.all_qubits() != qubits:
            raise ValueError(
                "Active qubits have changed after simplification. Forcing stop."
            )

    # Simulate
    if type(optimize) == str and 'evolution' in optimize:

        # Set default parameters
        optimize = '-'.join(optimize.split('-')[1:])
        if not optimize:
            optimize = 'hybridq'
        kwargs.setdefault('compress', 4)
        kwargs.setdefault('max_largest_intermediate', 2**26)
        kwargs.setdefault('return_info', False)
        kwargs.setdefault('block_until_ready', True)
        kwargs.setdefault('return_numpy_array', True)

        return _simulate_evolution(circuit, initial_state, final_state,
                                   optimize, backend, complex_type, verbose,
                                   **kwargs)
    else:

        # Set default parameters
        kwargs.setdefault('compress', 2)
        kwargs.setdefault('simplify_tn', 'RC')
        kwargs.setdefault('max_iterations', 1)
        try:
            import kahypar as __kahypar__
            kwargs.setdefault('methods', ['kahypar', 'greedy'])
        except ModuleNotFoundError:
            warn("Cannot find module kahypar. Remove it from defaults.")
            kwargs.setdefault('methods', ['greedy'])
        except ImportError:
            warn("Cannot import module kahypar. Remove it from defaults.")
            kwargs.setdefault('methods', ['greedy'])
        kwargs.setdefault('max_time', 120)
        kwargs.setdefault('max_repeats', 16)
        kwargs.setdefault('minimize', 'combo')
        kwargs.setdefault('optlib', 'baytune')
        kwargs.setdefault('sampler', 'GP')
        kwargs.setdefault('target_largest_intermediate', 0)
        kwargs.setdefault('max_largest_intermediate', 2**26)
        kwargs.setdefault('temperatures', [1.0, 0.1, 0.01])
        kwargs.setdefault('parallel', None)
        kwargs.setdefault('cotengra', {})
        kwargs.setdefault('max_n_slices', None)
        kwargs.setdefault('return_info', False)

        # If use_mpi==False, force the non-use of MPI
        if not (use_mpi == False) and (use_mpi or _detect_mpi):

            # Warn that MPI is used because detected
            if not use_mpi:
                warn("MPI has been detected. Using MPI.")

            from hybridq.circuit.simulation.simulation_mpi import _simulate_tn_mpi
            return _simulate_tn_mpi(circuit, initial_state, final_state,
                                    optimize, backend, complex_type,
                                    tensor_only, verbose, **kwargs)
        else:

            if _detect_mpi and use_mpi == False:
                warn(
                    "MPI has been detected but use_mpi==False. Not using MPI as requested."
                )

            return _simulate_tn(circuit, initial_state, final_state, optimize,
                                backend, complex_type, tensor_only, verbose,
                                **kwargs)


def _simulate_evolution(circuit: iter[Gate], initial_state: any,
                        final_state: any, optimize: any, backend: any,
                        complex_type: any, verbose: bool, **kwargs):
    """
    Perform simulation of the circuit by using the direct evolution of the quantum state.
    """

    if _detect_mpi:
        warn("Detected MPI but optimize='evolution' does not support MPI.")

    # Initialize info
    _sim_info = {}

    # Convert iterable to circuit
    circuit = Circuit(circuit)

    # Get number of qubits
    qubits = circuit.all_qubits()
    n_qubits = len(qubits)

    # Check if core libraries have been loaded properly
    if any(not x
           for x in [_swap_core, _dot_core, _to_complex_core, _log2_pack_size]):
        warn("Cannot find C++ HybridQ core. "
             "Falling back to optimize='evolution-einsum' instead.")
        optimize = 'einsum'

    # If the system is too small, fallback to einsum
    if optimize == 'hybridq' and n_qubits <= max(10, _log2_pack_size):
        warn("The system is too small to use optimize='evolution-hybridq'. "
             "Falling back to optimize='evolution-einsum'")
        optimize = 'einsum'

    if verbose:
        print(f'# Optimization: {optimize}', file=stderr)

    # Check memory
    if 2**n_qubits > kwargs['max_largest_intermediate']:
        raise MemoryError(
            "Memory for the given number of qubits exceeds the 'max_largest_intermediate'."
        )

    # If final_state is specified, warn user
    if final_state is not None:
        warn(
            f"'final_state' cannot be specified in optimize='{optimize}'. Ignoring 'final_state'."
        )

    # Initial state must be provided
    if initial_state is None:
        raise ValueError(
            "'initial_state' must be specified for optimize='evolution'.")

    # Convert complex_type to np.dtype
    complex_type = np.dtype(complex_type)

    # Print info
    if verbose:
        print(f"Compress circuit (max_n_qubits={kwargs['compress']}): ",
              end='',
              file=stderr)
        _time = time()

    # Compress circuit
    circuit = utils.compress(
        circuit,
        kwargs['compress']['max_n_qubits'] if isinstance(
            kwargs['compress'], dict) else kwargs['compress'],
        verbose=verbose,
        skip_compression=[pr.FunctionalGate],
        **({k: v for k, v in kwargs['compress'].items() if k != 'max_n_qubits'}
           if isinstance(kwargs['compress'], dict) else {}))

    # Check that FunctionalGate's are not compressed
    assert (all(not isinstance(g, pr.FunctionalGate) if len(x) > 1 else True
                for x in circuit
                for g in x))

    # Compress everything which is not a FunctionalGate
    circuit = Circuit(g for c in (c if any(
        isinstance(g, pr.FunctionalGate)
        for g in c) else [utils.to_matrix_gate(c, complex_type=complex_type)]
                                  for c in circuit) for g in c)

    # Get state
    initial_state = prepare_state(initial_state,
                                  complex_type=complex_type) if isinstance(
                                      initial_state, str) else initial_state

    if verbose:
        print(f"Done! ({time()-_time:1.2f}s)", file=stderr)

    if optimize == 'hybridq':

        if complex_type not in ['complex64', 'complex128']:
            warn(
                "optimize=evolution-hybridq only support ['complex64', 'complex128']. Using 'complex64'."
            )
            complex_type = np.dtype('complex64')

        # Get float_type
        float_type = np.real(np.array(1, dtype=complex_type)).dtype

        # Get C float_type
        c_float_type = {
            np.dtype('float32'): ctypes.c_float,
            np.dtype('float64'): ctypes.c_double
        }[float_type]

        # Load libraries
        _apply_U = _dot_core[float_type]

        # Get swap core
        _swap = _swap_core[float_type]

        # Get to_complex core
        _to_complex = _to_complex_core[complex_type]

        # Get states
        _psi = aligned.empty(shape=(2,) + initial_state.shape,
                             dtype=float_type,
                             order='C',
                             alignment=32)
        # Split in real and imaginary part
        _psi_re = _psi[0]
        _psi_im = _psi[1]

        # Check alignment
        assert (_psi_re.ctypes.data % 32 == 0)
        assert (_psi_im.ctypes.data % 32 == 0)

        # Get C-pointers
        _psi_re_ptr = _psi_re.ctypes.data_as(ctypes.POINTER(c_float_type))
        _psi_im_ptr = _psi_im.ctypes.data_as(ctypes.POINTER(c_float_type))

        # Initialize
        np.copyto(_psi_re, np.real(initial_state))
        np.copyto(_psi_im, np.imag(initial_state))

        # Create index maps
        _map = {q: n_qubits - x - 1 for x, q in enumerate(qubits)}
        _inv_map = [q for q, _ in sort(_map.items(), key=lambda x: x[1])]

        # Set largest swap_size
        _max_swap_size = 0

        # Start clock
        _ini_time = time()

        # Apply all gates
        for gate in tqdm(circuit, disable=not verbose):

            # FunctionalGate
            if isinstance(gate, pr.FunctionalGate):
                # Get order
                order = tuple(
                    q
                    for q, _ in sorted(_map.items(), key=lambda x: x[1])[::-1])

                # Apply gate to state
                new_psi, new_order = gate.apply(psi=_psi, order=order)

                # Copy back if needed
                if new_psi is not _psi:
                    # Align if needed
                    _psi = aligned.asarray(new_psi,
                                           order='C',
                                           alignment=32,
                                           dtype=_psi.dtype)

                    # Redefine real and imaginary part
                    _psi_re = _psi[0]
                    _psi_im = _psi[1]

                    # Get C-pointers
                    _psi_re_ptr = _psi_re.ctypes.data_as(
                        ctypes.POINTER(c_float_type))
                    _psi_im_ptr = _psi_im.ctypes.data_as(
                        ctypes.POINTER(c_float_type))

                # This can be eventually fixed ...
                if any(x != y for x, y in zip(order, new_order)):
                    raise RuntimeError("'order' has changed.")

            elif gate.provides(['qubits', 'matrix']):

                # Check if any qubits is withing the pack_size
                if any(q in _inv_map[:_log2_pack_size] for q in gate.qubits):

                    #@@@ Alternative way to always use the smallest swap size
                    #@@@
                    #@@@ # Get positions
                    #@@@ _pos = np.fromiter((_map[q] for q in gate.qubits),
                    #@@@                    dtype=int)

                    #@@@ # Get smallest swap size
                    #@@@ _swap_size = 0 if np.all(_pos >= _log2_pack_size) else next(
                    #@@@     k
                    #@@@     for k in range(_log2_pack_size, 2 *
                    #@@@                    max(len(_pos), _log2_pack_size) + 1)
                    #@@@     if sum(_pos < k) <= k - _log2_pack_size)

                    #@@@ # Get new order
                    #@@@ _order = [
                    #@@@     x for x, q in enumerate(_inv_map[:_swap_size])
                    #@@@     if q not in gate.qubits
                    #@@@ ]
                    #@@@ _order += [
                    #@@@     x for x, q in enumerate(_inv_map[:_swap_size])
                    #@@@     if q in gate.qubits
                    #@@@ ]

                    if len(gate.qubits) <= 4:

                        # Get new order
                        _order = [
                            x for x, q in enumerate(_inv_map[:8])
                            if q not in gate.qubits
                        ]
                        _order += [
                            x for x, q in enumerate(_inv_map[:8])
                            if q in gate.qubits
                        ]

                    else:

                        # Get qubit indexes for gate
                        _gate_idxs = [_inv_map.index(q) for q in gate.qubits]

                        # Get new order
                        _order = [
                            x for x in range(n_qubits) if x not in _gate_idxs
                        ][:_log2_pack_size]
                        _order += [x for x in _gate_idxs if x < max(_order)]

                    # Get swap size
                    _swap_size = len(_order)

                    # Update max swap size
                    if _swap_size > _max_swap_size:
                        _max_swap_size = _swap_size

                    # Update maps
                    _inv_map[:_swap_size] = [
                        _inv_map[:_swap_size][x] for x in _order
                    ]
                    _map.update(
                        {q: x for x, q in enumerate(_inv_map[:_swap_size])})

                    # Apply swap
                    _order = np.array(_order, dtype='uint32')
                    _swap(
                        _psi_re_ptr,
                        _order.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                        n_qubits, len(_order))
                    _swap(
                        _psi_im_ptr,
                        _order.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                        n_qubits, len(_order))

                # Get positions
                _pos = np.array([_map[q] for q in reversed(gate.qubits)],
                                dtype='uint32')

                # Get matrix
                _U = np.asarray(gate.matrix(), dtype=complex_type, order='C')

                # Apply matrix
                if _apply_U(
                        _psi_re_ptr, _psi_im_ptr,
                        _U.ctypes.data_as(ctypes.POINTER(c_float_type)),
                        _pos.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                        n_qubits, len(_pos)):

                    raise RuntimeError('something went wrong')

            else:
                raise RuntimeError(f"'{gate}' not supported")

        # Check maps are still consistent
        assert (all(_inv_map[_map[q]] == q for q in _map))

        # Swap back to the correct order
        _order = np.array([_inv_map.index(q) for q in reversed(qubits)
                          ][:_max_swap_size],
                          dtype='uint32')
        _swap(_psi_re_ptr,
              _order.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), n_qubits,
              len(_order))
        _swap(_psi_im_ptr,
              _order.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), n_qubits,
              len(_order))

        # Stop clock
        _end_time = time()

        # Copy the results
        if kwargs['return_numpy_array']:
            _complex_psi = np.empty(_psi.shape[1:], dtype=complex_type)
            _to_complex(
                _psi_re_ptr, _psi_im_ptr,
                _complex_psi.ctypes.data_as(ctypes.POINTER(c_float_type)),
                2**n_qubits)
            _psi = _complex_psi

        # Update info
        _sim_info['runtime (s)'] = _end_time - _ini_time

    elif optimize.split('-')[0] == 'einsum':

        optimize = '-'.join(optimize.split('-')[1:])
        if not optimize:
            optimize = 'auto'

        # Split circuits to separate FunctionalGate's
        circuit = utils.compress(
            circuit,
            max_n_qubits=len(qubits),
            skip_compression=[pr.FunctionalGate],
            **({
                k: v
                for k, v in kwargs['compress'].items()
                if k != 'max_n_qubits'
            } if isinstance(kwargs['compress'], dict) else {}))

        # Check that FunctionalGate's are not compressed
        assert (all(not isinstance(g, pr.FunctionalGate) if len(x) > 1 else True
                    for x in circuit
                    for g in x))

        # Prepare initial_state
        _psi = initial_state

        # Initialize time
        _ini_time = time()
        for circuit in circuit:

            # Check
            assert (all(not isinstance(g, pr.FunctionalGate) for g in circuit)
                    or len(circuit) == 1)

            # Apply gate if functional
            if len(circuit) == 1 and isinstance(circuit[0], pr.FunctionalGate):

                # Apply gate to state
                _psi, qubits = circuit[0].apply(psi=_psi, order=qubits)

            else:
                # Get gates and corresponding qubits
                _qubits, _gates = zip(
                    *((c.qubits,
                       np.reshape(c.matrix().astype(complex_type), (2,) *
                                  (2 * len(c.qubits)))) for c in circuit))

                # Initialize map
                _map = {q: get_symbol(x) for x, q in enumerate(qubits)}
                _count = n_qubits
                _path = ''.join((_map[q] for q in qubits))

                # Generate map
                for _qs in _qubits:

                    # Initialize local paths
                    _path_in = _path_out = ''

                    # Add incoming legs
                    for _q in _qs:
                        _path_in += _map[_q]

                    # Add outcoming legs
                    for _q in _qs:
                        _map[_q] = get_symbol(_count)
                        _count += 1
                        _path_out += _map[_q]

                    # Update path
                    _path = _path_out + _path_in + ',' + _path

                # Make sure that qubits order is preserved
                _path += '->' + ''.join([_map[q] for q in qubits])

                # Contracts
                _psi = contract(_path,
                                *reversed(_gates),
                                _psi,
                                backend=backend,
                                optimize=optimize)

                # Block JAX until result is ready (for a more precise runtime)
                if backend == 'jax' and kwargs['block_until_ready']:
                    _psi.block_until_ready()

        # Stop time
        _end_time = time()

        # Update info
        _sim_info['runtime (s)'] = _end_time - _ini_time

    else:

        raise ValueError(f"optimize='{optimize}' not implemented.")

    if verbose:
        print(f'# Runtime (s): {_sim_info["runtime (s)"]:1.2f}', file=stderr)

    # Return state
    if kwargs['return_info']:
        return _psi, _sim_info
    else:
        return _psi


def _simulate_tn(circuit: any, initial_state: any, final_state: any,
                 optimize: any, backend: any, complex_type: any,
                 tensor_only: bool, verbose: bool, **kwargs):
    import quimb.tensor as tn
    import cotengra as ctg

    # Get random leaves_prefix
    leaves_prefix = ''.join(
        np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), size=20))

    # Initialize info
    _sim_info = {}

    # Alias for tn
    if optimize == 'tn':
        optimize = 'cotengra'

    if isinstance(circuit, Circuit):

        # Get number of qubits
        qubits = circuit.all_qubits()
        n_qubits = len(qubits)

        # If initial/final state is None, set to all .'s
        initial_state = '.' * n_qubits if initial_state is None else initial_state
        final_state = '.' * n_qubits if final_state is None else final_state

        # Initial and final states must be valid strings
        for state, sname in [(initial_state, 'initial_state'),
                             (final_state, 'final_state')]:
            # Get alphabet
            from string import ascii_letters

            # Check if string
            if not isinstance(state, str):
                raise ValueError(f"'{sname}' must be a valid string.")

            # Deprecated error
            if any(x in 'xX' for x in state):
                from hybridq.utils import DeprecationWarning
                from warnings import warn

                # Warn the user that '.' is used to represent open qubits
                warn(
                    "Since '0.6.3', letters in the alphabet are used to "
                    "trace selected qubits (including 'x' and 'X'). "
                    "Instead, '.' is used to represent an open qubit.",
                    DeprecationWarning)

            # Check only valid symbols are present
            if set(state).difference('01+-.' + ascii_letters):
                raise ValueError(f"'{sname}' contains invalid symbols.")

            # Check number of qubits
            if len(state) != n_qubits:
                raise ValueError(f"'{sname}' has the wrong number of qubits "
                                 f"(expected {n_qubits}, got {len(state)})")

        # Check memory
        if 2**(initial_state.count('.') +
               final_state.count('.')) > kwargs['max_largest_intermediate']:
            raise MemoryError("Memory for the given number of open qubits "
                              "exceeds the 'max_largest_intermediate'.")

        # Compress circuit
        if kwargs['compress']:
            if verbose:
                print(f"Compress circuit (max_n_qubits={kwargs['compress']}): ",
                      end='',
                      file=stderr)
                _time = time()

            circuit = utils.compress(
                circuit,
                kwargs['compress']['max_n_qubits'] if isinstance(
                    kwargs['compress'], dict) else kwargs['compress'],
                verbose=verbose,
                **({
                    k: v
                    for k, v in kwargs['compress'].items()
                    if k != 'max_n_qubits'
                } if isinstance(kwargs['compress'], dict) else {}))

            circuit = Circuit(
                utils.to_matrix_gate(c, complex_type=complex_type)
                for c in circuit)
            if verbose:
                print(f"Done! ({time()-_time:1.2f}s)", file=stderr)

        # Get tensor network representation of circuit
        tensor, tn_qubits_map = utils.to_tn(circuit,
                                            return_qubits_map=True,
                                            leaves_prefix=leaves_prefix)

        # Define basic MPS
        _mps = {
            '0': np.array([1, 0]),
            '1': np.array([0, 1]),
            '+': np.array([1, 1]) / np.sqrt(2),
            '-': np.array([1, -1]) / np.sqrt(2)
        }

        # Attach initial/final state
        for state, ext in [(initial_state, 'i'), (final_state, 'f')]:
            for s, q in ((s, q) for s, q in zip(state, qubits) if s in _mps):
                inds = [f'{leaves_prefix}_{tn_qubits_map[q]}_{ext}']
                tensor &= tn.Tensor(_mps[s], inds=inds, tags=inds)

        # For each unique letter, apply trace
        for x in set(initial_state + final_state).difference(''.join(_mps) +
                                                             '.'):
            # Get indexes
            inds = [
                f'{leaves_prefix}_{tn_qubits_map[q]}_i'
                for s, q in zip(initial_state, qubits)
                if s == x
            ]
            inds += [
                f'{leaves_prefix}_{tn_qubits_map[q]}_f'
                for s, q in zip(final_state, qubits)
                if s == x
            ]

            # Apply trace
            tensor &= tn.Tensor(np.reshape([1] + [0] * (2**len(inds) - 2) + [1],
                                           (2,) * len(inds)),
                                inds=inds)

        # Simplify if requested
        if kwargs['simplify_tn']:
            tensor.full_simplify_(kwargs['simplify_tn']).astype_(complex_type)
        else:
            # Otherwise, just convert to the given complex_type
            tensor.astype_(complex_type)

        # Get contraction from heuristic
        if optimize == 'cotengra' and kwargs['max_iterations'] > 0:

            # Create local client if MPI has been detected (not compatible with Dask at the moment)
            if _mpi_env and kwargs['parallel']:

                from distributed import Client, LocalCluster
                _client = Client(LocalCluster(processes=False))

            else:

                _client = None

            # Set cotengra parameters
            cotengra_params = lambda: ctg.HyperOptimizer(
                methods=kwargs['methods'],
                max_time=kwargs['max_time'],
                max_repeats=kwargs['max_repeats'],
                minimize=kwargs['minimize'],
                optlib=kwargs['optlib'],
                sampler=kwargs['sampler'],
                progbar=verbose,
                parallel=kwargs['parallel'],
                **kwargs['cotengra'])

            # Get optimized path
            opt = cotengra_params()
            info = tensor.contract(all, optimize=opt, get='path-info')

            # Get target size
            tli = kwargs['target_largest_intermediate']

            # Repeat for the requested number of iterations
            for _ in range(1, kwargs['max_iterations']):

                # Break if largest intermediate is equal or smaller than target
                if info.largest_intermediate <= tli:
                    break

                # Otherwise, restart
                _opt = cotengra_params()
                _info = tensor.contract(all, optimize=_opt, get='path-info')

                # Store the best
                if kwargs['minimize'] == 'size':

                    if _info.largest_intermediate < info.largest_intermediate or (
                            _info.largest_intermediate
                            == info.largest_intermediate and
                            _opt.best['flops'] < opt.best['flops']):
                        info = _info
                        opt = _opt

                else:

                    if _opt.best['flops'] < opt.best['flops'] or (
                            _opt.best['flops'] == opt.best['flops'] and
                            _info.largest_intermediate <
                            info.largest_intermediate):
                        info = _info
                        opt = _opt

            # Close client if exists
            if _client:

                _client.shutdown()
                _client.close()

        # Just return tensor if required
        if tensor_only:
            if optimize == 'cotengra' and kwargs['max_iterations'] > 0:
                return tensor, (info, opt)
            else:
                return tensor

    else:

        # Set tensor
        tensor = circuit

        if len(optimize) == 2 and isinstance(
                optimize[0], PathInfo) and isinstance(optimize[1],
                                                      ctg.hyper.HyperOptimizer):

            # Get info and opt from optimize
            info, opt = optimize

            # Set optimization
            optimize = 'cotengra'

        else:

            # Get tensor and path
            tensor = circuit

    # Print some info
    if verbose:
        print(
            f'Largest Intermediate: 2^{np.log2(float(info.largest_intermediate)):1.2f}',
            file=stderr)
        print(
            f'Max Largest Intermediate: 2^{np.log2(float(kwargs["max_largest_intermediate"])):1.2f}',
            file=stderr)
        print(f'Flops: 2^{np.log2(float(info.opt_cost)):1.2f}', file=stderr)

    if optimize == 'cotengra':

        # Get indexes
        _inds = tensor.outer_inds()

        # Get input indexes and output indexes
        _i_inds = sort([x for x in _inds if x[-2:] == '_i'],
                       key=lambda x: int(x.split('_')[1]))
        _f_inds = sort([x for x in _inds if x[-2:] == '_f'],
                       key=lambda x: int(x.split('_')[1]))

        # Get order
        _inds = [_inds.index(x) for x in _i_inds + _f_inds]

        # Get slice finder
        sf = ctg.SliceFinder(info,
                             target_size=kwargs['max_largest_intermediate'])

        # Find slices
        with tqdm(kwargs['temperatures'], disable=not verbose,
                  leave=False) as pbar:
            for _temp in pbar:
                pbar.set_description(f'Find slices (T={_temp})')
                ix_sl, cost_sl = sf.search(temperature=_temp)

        # Get slice contractor
        sc = sf.SlicedContractor([t.data for t in tensor])

        # Update infos
        _sim_info.update({
            'flops': info.opt_cost,
            'largest_intermediate': info.largest_intermediate,
            'n_slices': cost_sl.nslices,
            'total_flops': cost_sl.total_flops
        })

        # Print some infos
        if verbose:
            print(f'Number of slices: 2^{np.log2(float(cost_sl.nslices)):1.2f}',
                  file=stderr)
            print(f'Flops+Cuts: 2^{np.log2(float(cost_sl.total_flops)):1.2f}',
                  file=stderr)

        if kwargs['max_n_slices'] and sc.nslices > kwargs['max_n_slices']:
            raise RuntimeError(
                f'Too many slices ({sc.nslices} > {kwargs["max_n_slices"]})')

        # Contract tensor
        _li = np.log2(float(info.largest_intermediate))
        _mli = np.log2(float(kwargs["max_largest_intermediate"]))
        _tensor = sc.gather_slices(
            (sc.contract_slice(i, backend=backend) for i in tqdm(
                range(sc.nslices),
                desc=f'Contracting tensor (li=2^{_li:1.0f}, mli=2^{_mli:1.1f})',
                leave=False)))

        # Create map
        _map = ''.join([get_symbol(x) for x in range(len(_inds))])
        _map += '->'
        _map += ''.join([get_symbol(x) for x in _inds])

        # Reorder tensor
        tensor = contract(_map, _tensor)

        # Deprecated
        ## Reshape tensor
        #if _inds:
        #    if _i_inds and _f_inds:
        #        tensor = np.reshape(tensor, (2**len(_i_inds), 2**len(_f_inds)))
        #    else:
        #        tensor = np.reshape(tensor,
        #                            (2**max(len(_i_inds), len(_f_inds)),))

    else:

        # Contract tensor
        tensor = tensor.contract(optimize=optimize, backend=backend)

        if hasattr(tensor, 'inds'):

            # Get input indexes and output indexes
            _i_inds = sort([x for x in tensor.inds if x[-2:] == '_i'],
                           key=lambda x: int(x.split('_')[1]))
            _f_inds = sort([x for x in tensor.inds if x[-2:] == '_f'],
                           key=lambda x: int(x.split('_')[1]))

            # Transpose tensor
            tensor.transpose(*(_i_inds + _f_inds), inplace=True)

            # Deprecated
            ## Reshape tensor
            #if _i_inds and _f_inds:
            #    tensor = np.reshape(tensor, (2**len(_i_inds), 2**len(_f_inds)))
            #else:
            #    tensor = np.reshape(tensor,
            #                        (2**max(len(_i_inds), len(_f_inds)),))

    if kwargs['return_info']:
        return tensor, _sim_info
    else:
        return tensor


def expectation_value(state: Array,
                      op: Circuit,
                      qubits_order: iter[any],
                      complex_type: any = 'complex64',
                      backend: any = 'numpy',
                      verbose: bool = False,
                      **kwargs) -> complex:
    """
    Compute expectation value of an operator given a quantum state.

    Parameters
    ----------
    state: Array
        Quantum state to use to compute the expectation value of the operator
        `op`.
    op: Circuit
        Quantum operator to use to compute the expectation value.
    qubits_order: iter[any]
        Order of qubits used to map `Circuit.qubits` to `state`.
    complex_type: any, optional
        Complex type to use to compute the expectation value.
    backend: any, optional
        Backend used to compute the quantum state. Backend must have
        `tensordot`, `transpose` and `einsum` methods.
    verbose: bool, optional
        Verbose output.

    Returns
    -------
    complex
        The expectation value of the operator `op` given `state`.

    Other Parameters
    ----------------
    `expectation_value` accepts all valid parameters for `simulate`.

    See Also
    --------
    `simulate`

    Example
    -------
    >>> op = Circuit([
    >>>     Gate('H', qubits=[32]),
    >>>     Gate('CX', qubits=[32, 42]),
    >>>     Gate('RX', qubits=[12], params=[1.32])
    >>> ])
    >>> expectation_value(
    >>>     state=prepare_state('+0-'),
    >>>     op=op,
    >>>     qubits_order=[12, 42, 32],
    >>> )
    array(0.55860883-0.43353909j)
    """

    # Fix remove_id_gates
    kwargs['remove_id_gates'] = False

    # Get number of qubits
    state = np.asarray(state)

    # Get number of qubits
    n_qubits = state.ndim

    # Convert qubits_order to a list
    qubits_order = list(qubits_order)

    # Check lenght of qubits_order
    if len(qubits_order) != n_qubits:
        raise ValueError(
            "'qubits_order' must have the same number of qubits of 'state'.")

    # Check that qubits in op are a subset of qubits_order
    if set(op.all_qubits()).difference(qubits_order):
        raise ValueError("'op' has qubits not included in 'qubits_order'.")

    # Add Gate('I') to op for not used qubits
    op = op + [
        Gate('I', qubits=[q])
        for q in set(qubits_order).difference(op.all_qubits())
    ]

    # Simulate op with given state
    _state = simulate(op,
                      initial_state=state,
                      optimize='evolution',
                      complex_type=complex_type,
                      backend=backend,
                      verbose=verbose,
                      **kwargs)

    # Return expectation value
    return np.real_if_close(np.sum(_state * state.conj()))
