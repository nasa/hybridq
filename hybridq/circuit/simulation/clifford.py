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
from os import environ
from os.path import basename

_detect_mpi = 'DISABLE_MPI_AUTODETECT' not in environ and '_' in environ and basename(
    environ['_']) in ['mpiexec', 'mpirun']

from more_itertools import distribute, chunked, flatten
from psutil import virtual_memory, getloadavg
from functools import partial as partial_func
from hybridq.utils import globalize, kron
import hybridq.circuit.utils as utils
from collections import defaultdict
from hybridq.circuit import Circuit
from multiprocessing import Pool
from itertools import product
from hybridq.gate import Gate
from time import sleep, time
from tqdm.auto import tqdm
from warnings import warn
from os import cpu_count
import numpy as np
import numba
import sys

# Index multiplier
_gate_mul = 20

# 1q gates (0 <= index < 20)
_I = 0
_X = 1
_Y = 2
_Z = 3
_H = 4
_RX = 10
_RY = 11
_RZ = 12
_MATRIX_1 = 19

# 2q gates (20 <= index < 40)
_CZ = 20
_ISWAP = 21
_SWAP = 22
_MATRIX_2 = 39

# 3q gates (40 <= index < 60)
_MATRIX_3 = 59

# 4q gates (60 <= index < 80)
_MATRIX_4 = 79

# 5q gates (80 <= index < 100)
_MATRIX_5 = 99

_MATRIX_SET = [
    _MATRIX_1,
    _MATRIX_2,
    _MATRIX_3,
    _MATRIX_4,
    _MATRIX_5,
]


@numba.njit(fastmath=True, cache=True)
def _update_pauli_string(gates, qubits, params, pauli_string: list[int],
                         phase: float, pos_shift: int, eps: float,
                         atol: float) -> float:

    # Get branches
    _branches = []

    for pos in range(pos_shift, len(gates)):

        # Get gate
        _name = gates[pos]

        # Get number of qubits
        _n_qubits = (_name // _gate_mul) + 1

        # Get qubits and paulis
        if _n_qubits == 5:
            q1, q2, q3, q4, q5 = qubits[pos][0], qubits[pos][1], qubits[pos][
                2], qubits[pos][3], qubits[pos][4]
            s1, s2, s3, s4, s5 = pauli_string[q1], pauli_string[
                q2], pauli_string[q3], pauli_string[q4], pauli_string[q5]
            if s1 == s2 == s3 == s4 == s5 == _I:
                continue
        elif _n_qubits == 4:
            q1, q2, q3, q4 = qubits[pos][0], qubits[pos][1], qubits[pos][
                2], qubits[pos][3]
            s1, s2, s3, s4 = pauli_string[q1], pauli_string[q2], pauli_string[
                q3], pauli_string[q4]
            if s1 == s2 == s3 == s4 == _I:
                continue
        elif _n_qubits == 3:
            q1, q2, q3 = qubits[pos][0], qubits[pos][1], qubits[pos][2]
            s1, s2, s3 = pauli_string[q1], pauli_string[q2], pauli_string[q3]
            if s1 == s2 == s3 == _I:
                continue
        elif _n_qubits == 2:
            q1, q2 = qubits[pos][0], qubits[pos][1]
            s1, s2 = pauli_string[q1], pauli_string[q2]
            if s1 == s2 == _I:
                continue
        else:
            q1 = qubits[pos][0]
            s1 = pauli_string[q1]
            if s1 == _I:
                continue

        # I
        if _name == _I:

            # Just ignore
            pass

        # X, Y, Z
        elif _name in [_X, _Y, _Z]:
            # Update phase
            if s1 != _name:
                phase *= -1

        # H
        elif _name == _H:

            if s1 == _X:
                pauli_string[q1] = _Z
            elif s1 == _Z:
                pauli_string[q1] = _X
            elif s1 == _Y:
                phase *= -1

        # ISWAP
        elif _name == _ISWAP:

            if s1 == _X and s2 == _I:
                pauli_string[q1] = _Z
                pauli_string[q2] = _Y
                phase *= -1

            elif s1 == _Y and s2 == _I:
                pauli_string[q1] = _Z
                pauli_string[q2] = _X

            elif s1 == _Z and s2 == _I:
                pauli_string[q1] = _I
                pauli_string[q2] = _Z

            elif s1 == _I and s2 == _X:
                pauli_string[q1] = _Y
                pauli_string[q2] = _Z
                phase *= -1

            elif s1 == _Y and s2 == _X:
                pauli_string[q1] = _X
                pauli_string[q2] = _Y

            elif s1 == _Z and s2 == _X:
                pauli_string[q1] = _Y
                pauli_string[q2] = _I
                phase *= -1

            elif s1 == _I and s2 == _Y:
                pauli_string[q1] = _X
                pauli_string[q2] = _Z

            elif s1 == _X and s2 == _Y:
                pauli_string[q1] = _Y
                pauli_string[q2] = _X

            elif s1 == _Z and s2 == _Y:
                pauli_string[q1] = _X
                pauli_string[q2] = _I

            elif s1 == _I and s2 == _Z:
                pauli_string[q1] = _Z
                pauli_string[q2] = _I

            elif s1 == _X and s2 == _Z:
                pauli_string[q1] = _I
                pauli_string[q2] = _Y
                phase *= -1

            elif s1 == _Y and s2 == _Z:
                pauli_string[q1] = _I
                pauli_string[q2] = _X

        # CZ
        elif _name == _CZ:

            # Combine indexes
            if s1 in [_X, _Y] and s2 == _I:
                pauli_string[q2] = _Z

            elif s1 == _I and s2 in [_X, _Y]:
                pauli_string[q1] = _Z

            elif s1 in [_X, _Y] and s2 == _Z:
                pauli_string[q2] = _I

            elif s1 == _Z and s2 in [_X, _Y]:
                pauli_string[q1] = _I

            elif s1 == _X and s2 == _X:
                pauli_string[q1] = _Y
                pauli_string[q2] = _Y

            elif s1 == _X and s2 == _Y:
                pauli_string[q1] = _Y
                pauli_string[q2] = _X
                phase *= -1

            elif s1 == _Y and s2 == _X:
                pauli_string[q1] = _X
                pauli_string[q2] = _Y
                phase *= -1

            elif s1 == _Y and s2 == _Y:
                pauli_string[q1] = _X
                pauli_string[q2] = _X

        # SWAP
        elif _name == _SWAP:

            pauli_string[q1], pauli_string[q2] = pauli_string[q2], pauli_string[
                q1]

        elif _name == _MATRIX_1:

            # Get weights
            _ws = np.reshape(params[pos][:16], (4, 4))[s1]

            # Get indexes where weights are different from zeros
            _idxs = np.where(np.abs(_ws) > eps)[0]

            # Get only weights different from zeros
            _ws = _ws[_idxs]

            # Sort weights
            _pos = np.argsort(np.abs(_ws))[::-1]
            _ws = _ws[_pos]
            _idxs = _idxs[_pos]

            for _i in range(1, len(_idxs)):

                # Get position and weight
                _p = _idxs[_i]
                _w = _ws[_i]

                # Get new phase
                _phase = _w * phase

                # Check phase is large enough
                if abs(_phase) > atol:

                    # Branch
                    pauli_string[q1] = _p
                    _branches.append((np.copy(pauli_string), _phase, pos + 1))

            # Keep going with the largest weight
            pauli_string[q1] = _idxs[0]
            phase *= _ws[0]

        elif _name == _MATRIX_2:

            # Get weights
            _ws = np.reshape(params[pos][:256], (4, 4, 16))[s1, s2]

            # Get indexes where weights are different from zeros
            _idxs = np.where(np.abs(_ws) > eps)[0]

            # Get only weights different from zeros
            _ws = _ws[_idxs]

            # Sort weights
            _pos = np.argsort(np.abs(_ws))[::-1]
            _ws = _ws[_pos]
            _idxs = _idxs[_pos]

            for _i in range(1, len(_idxs)):

                # Get position and weight
                _p = _idxs[_i]
                _w = _ws[_i]

                # Get new phase
                _phase = _w * phase

                # Check phase is large enough
                if abs(_phase) > atol:

                    # Get gates
                    _g2 = _p % 4
                    _g1 = _p // 4

                    # Branch
                    pauli_string[q1] = _g1
                    pauli_string[q2] = _g2
                    _branches.append((np.copy(pauli_string), _phase, pos + 1))

            # Keep going with the largest weight
            _p = _idxs[0]
            _g2 = _p % 4
            _g1 = _p // 4
            pauli_string[q1] = _g1
            pauli_string[q2] = _g2
            phase *= _ws[0]

        elif _name == _MATRIX_3:

            # Get weights
            _ws = np.reshape(params[pos][:4096], (4, 4, 4, 64))[s1, s2, s3]

            # Get indexes where weights are different from zeros
            _idxs = np.where(np.abs(_ws) > eps)[0]

            # Get only weights different from zeros
            _ws = _ws[_idxs]

            # Sort weights
            _pos = np.argsort(np.abs(_ws))[::-1]
            _ws = _ws[_pos]
            _idxs = _idxs[_pos]

            for _i in range(1, len(_idxs)):

                # Get position and weight
                _p = _idxs[_i]
                _w = _ws[_i]

                # Get new phase
                _phase = _w * phase

                # Check phase is large enough
                if abs(_phase) > atol:

                    # Get gates
                    _g3 = _p % 4
                    _g2 = (_p // 4) % 4
                    _g1 = _p // 16

                    # Branch
                    pauli_string[q1] = _g1
                    pauli_string[q2] = _g2
                    pauli_string[q3] = _g3
                    _branches.append((np.copy(pauli_string), _phase, pos + 1))

            # Keep going with the largest weight
            _p = _idxs[0]
            _g3 = _p % 4
            _g2 = (_p // 4) % 4
            _g1 = _p // 16
            pauli_string[q1] = _g1
            pauli_string[q2] = _g2
            pauli_string[q3] = _g3
            phase *= _ws[0]

        elif _name == _MATRIX_4:

            # Get weights
            _ws = np.reshape(params[pos][:65536], (4, 4, 4, 4, 256))[s1, s2, s3,
                                                                     s4]

            # Get indexes where weights are different from zeros
            _idxs = np.where(np.abs(_ws) > eps)[0]

            # Get only weights different from zeros
            _ws = _ws[_idxs]

            # Sort weights
            _pos = np.argsort(np.abs(_ws))[::-1]
            _ws = _ws[_pos]
            _idxs = _idxs[_pos]

            for _i in range(1, len(_idxs)):

                # Get position and weight
                _p = _idxs[_i]
                _w = _ws[_i]

                # Get new phase
                _phase = _w * phase

                # Check phase is large enough
                if abs(_phase) > atol:

                    # Get gates
                    _g4 = _p % 4
                    _g3 = (_p // 4) % 4
                    _g2 = (_p // 16) % 4
                    _g1 = _p // 64

                    # Branch
                    pauli_string[q1] = _g1
                    pauli_string[q2] = _g2
                    pauli_string[q3] = _g3
                    pauli_string[q4] = _g4
                    _branches.append((np.copy(pauli_string), _phase, pos + 1))

            # Keep going with the largest weight
            _p = _idxs[0]
            _g4 = _p % 4
            _g3 = (_p // 4) % 4
            _g2 = (_p // 16) % 4
            _g1 = _p // 64
            pauli_string[q1] = _g1
            pauli_string[q2] = _g2
            pauli_string[q3] = _g3
            pauli_string[q4] = _g4
            phase *= _ws[0]

        elif _name == _MATRIX_5:

            # Get weights
            _ws = np.reshape(params[pos][:1048576],
                             (4, 4, 4, 4, 4, 1024))[s1, s2, s3, s4, s5]

            # Get indexes where weights are different from zeros
            _idxs = np.where(np.abs(_ws) > eps)[0]

            # Get only weights different from zeros
            _ws = _ws[_idxs]

            # Sort weights
            _pos = np.argsort(np.abs(_ws))[::-1]
            _ws = _ws[_pos]
            _idxs = _idxs[_pos]

            for _i in range(1, len(_idxs)):

                # Get position and weight
                _p = _idxs[_i]
                _w = _ws[_i]

                # Get new phase
                _phase = _w * phase

                # Check phase is large enough
                if abs(_phase) > atol:

                    # Get gates
                    _g5 = _p % 4
                    _g4 = (_p // 4) % 4
                    _g3 = (_p // 16) % 4
                    _g2 = (_p // 64) % 4
                    _g1 = _p // 256

                    # Branch
                    pauli_string[q1] = _g1
                    pauli_string[q2] = _g2
                    pauli_string[q3] = _g3
                    pauli_string[q4] = _g4
                    pauli_string[q5] = _g5
                    _branches.append((np.copy(pauli_string), _phase, pos + 1))

            # Keep going with the largest weight
            _p = _idxs[0]
            _g5 = _p % 4
            _g4 = (_p // 4) % 4
            _g3 = (_p // 16) % 4
            _g2 = (_p // 64) % 4
            _g1 = _p // 256
            pauli_string[q1] = _g1
            pauli_string[q2] = _g2
            pauli_string[q3] = _g3
            pauli_string[q4] = _g4
            pauli_string[q5] = _g5
            phase *= _ws[0]

    return (pauli_string, phase), _branches


# Pre-process circuit
def _process_gate(gate, **kwargs):

    # Set default
    kwargs.setdefault('LS_cache', {})
    kwargs.setdefault('P_cache', {})

    def _GenerateLinearSystem(n_qubits):

        # Check if number of qubits is supported
        if f'_MATRIX_{n_qubits}' not in globals():
            raise ValueError('Too many qubits')

        if n_qubits not in kwargs['LS_cache']:

            I = Gate('I').matrix().astype('complex128')
            X = Gate('X').matrix().astype('complex128')
            Y = Gate('Y').matrix().astype('complex128')
            Z = Gate('Z').matrix().astype('complex128')

            W = [I, X, Y, Z]
            for _ in range(n_qubits - 1):
                W = [kron(g1, g2) for g1 in W for g2 in [I, X, Y, Z]]

            W = np.linalg.inv(np.reshape(W, (2**(2 * n_qubits),) * 2).T)

            kwargs['LS_cache'][n_qubits] = W

        return kwargs['LS_cache'][n_qubits]

    def _GetPauliOperator(*ps):

        # Check if number of qubits is supported
        if f'_MATRIX_{len(ps)}' not in globals():
            raise ValueError('Too many qubits')

        ps = ''.join(ps)
        if ps not in kwargs['P_cache']:
            kwargs['P_cache'][ps] = kron(*(Gate(g).matrix() for g in ps))
        return kwargs['P_cache'][ps]

    # Get matrix
    _U = gate.matrix()

    # Get qubits
    _q = gate.qubits

    # Get linear system
    _LS = _GenerateLinearSystem(len(_q))

    # Decompose
    _params = np.real(
        np.array([
            _LS.dot(_U.conj().T.dot(_GetPauliOperator(*_gs)).dot(_U).flatten())
            for _gs in product(*(('IXYZ',) * len(_q)))
        ]).flatten())
    return [(f'MATRIX_{len(_q)}', gate.qubits, _params)]


def _breadth_first_search(_update, db, branches, max_n_branches, infos, verbose,
                          **kwargs):

    # Explore all branches (breadth-first search)
    with tqdm(desc="Collect branches", disable=not verbose) as pbar:

        # Explore all branches
        while branches and len(branches) < max_n_branches:

            # Get new branches
            (_new_ps, _new_ph), _new_branches = _update(*branches.pop())

            # Collect results
            kwargs['collect'](db, kwargs['transform'](_new_ps), _new_ph)

            # Update branches
            branches.extend(_new_branches)

            # Update infos
            infos['largest_n_branches_in_memory'] = max(
                len(branches), infos['largest_n_branches_in_memory'])

            # Sort
            branches = sorted(branches, key=lambda x: -x[2])

            # Update infos
            infos['n_explored_branches'] += 1

            # Update progressbar
            pbar.set_description(
                f'Collect branches ({len(branches)}/{max_n_branches})')

    # Sort branches accordingly to the position shift
    branches = sorted(branches, key=lambda x: x[2])

    return branches


def _depth_first_search(_update, db, branches, parallel, infos, info_init,
                        verbose, mpi_rank, mpi_size, **kwargs):

    # Define parallel core
    def _parallel_core(branches, db=None):

        # Initialize db
        if db is None:
            db = kwargs['db_init']()

        # Convert to list
        branches = list(branches)

        # Initialize infos
        infos = info_init()

        # Explore all branches
        while branches:

            # Get new branches
            (_new_ps, _new_ph), _new_branches = _update(*branches.pop())

            # Collect results
            kwargs['collect'](db, kwargs['transform'](_new_ps), _new_ph)

            # Update branches
            branches.extend(_new_branches)

            # Update infos
            infos['largest_n_branches_in_memory'] = max(
                len(branches), infos['largest_n_branches_in_memory'])

            # Update infos
            infos['n_explored_branches'] += 1

        return db, infos

    # If no parallelization is requires, explore branchces one by one
    if parallel == 1:
        from more_itertools import ichunked

        # Get number of chunks
        chunk_size = max(1, len(branches) // 100)

        for _bs in tqdm(ichunked(branches, chunk_size),
                        total=len(branches) // chunk_size,
                        desc=f'Mem={virtual_memory().percent}%',
                        disable=not verbose):
            # Update database and infos
            db, _infos = _parallel_core(_bs, db)

            # Update infos
            infos['n_explored_branches'] += _infos['n_explored_branches']
            infos['largest_n_branches_in_memory'] = max(
                _infos['largest_n_branches_in_memory'],
                infos['largest_n_branches_in_memory'])

    # Otherwise, distribute workload among different cores
    else:
        with globalize(_parallel_core) as _parallel_core, Pool(
                parallel) as pool:

            # Apply async
            _fps = [
                pool.apply_async(_parallel_core, (_branches,))
                for _branches in distribute(kwargs['n_chunks'], branches)
            ]
            _status = [False] * len(_fps)

            with tqdm(total=len(_fps),
                      desc=f'Mem={virtual_memory().percent}%',
                      disable=not verbose) as pbar:

                _pending = len(_fps)
                while _pending:

                    # Wait
                    sleep(kwargs['sleep_time'])

                    # Activate/disactivate
                    if verbose:
                        pbar.disable = int(time()) % mpi_size != mpi_rank

                    # Get virtual memory
                    _vm = virtual_memory()

                    _pending = 0
                    for _x, (_p, _s) in enumerate(zip(_fps, _status)):
                        if not _p.ready():
                            _pending += 1
                        elif not _s:
                            # Collect data
                            _new_db, _infos = _p.get()

                            # Merge datasets
                            kwargs['merge'](db, _new_db)

                            # Clear dataset
                            _new_db.clear()

                            # Update infos
                            infos['n_explored_branches'] += _infos[
                                'n_explored_branches']
                            infos['largest_n_branches_in_memory'] = max(
                                _infos['largest_n_branches_in_memory'],
                                infos['largest_n_branches_in_memory'])

                            # Set status
                            _status[_x] = True

                    # Update pbar
                    if verbose:
                        pbar.set_description(
                            (f'[{mpi_rank}] ' if mpi_size > 1 else '') + \
                            f'Mem={_vm.percent}%, ' + \
                            f'NThreads={infos["n_threads"]}, ' + \
                            f'NCPUs={infos["n_cpus"]}, ' + \
                            f'LoadAvg={getloadavg()[0]/infos["n_cpus"]*100:1.2f}%, ' + \
                            f'NBranches={infos["n_explored_branches"]}'
                        )
                        pbar.n = len(_fps) - _pending
                        pbar.refresh()

                    # Update infos
                    infos['average_virtual_memory (GB)'] = (
                        infos['average_virtual_memory (GB)'][0] +
                        _vm.used / 2**30,
                        infos['average_virtual_memory (GB)'][1] + 1)
                    infos['peak_virtual_memory (GB)'] = max(
                        infos['peak_virtual_memory (GB)'], _vm.used / 2**30)

                    # If memory above threshold, raise error
                    if _vm.percent > kwargs['max_virtual_memory']:
                        raise MemoryError(
                            f'Memory above threshold: {_vm.percent}% > {kwargs["max_virtual_memory"]}%'
                        )

                # Last refresh
                if verbose:
                    pbar.refresh()

        # Check all chunks have been explored
        assert (np.alltrue(_status))


def update_pauli_string(circuit: Circuit,
                        pauli_string: {Circuit, dict[str, float]},
                        phase: float = 1,
                        parallel: {bool, int} = False,
                        return_info: bool = False,
                        use_mpi: bool = None,
                        compress: int = 4,
                        simplify: bool = True,
                        remove_id_gates: bool = True,
                        float_type: any = 'float32',
                        verbose: bool = False,
                        **kwargs) -> defaultdict:
    """
    Evolve density matrix accordingly to `circuit` using `pauli_string` as
    initial product state. The evolved density matrix will be represented as a
    set of different Pauli strings, each of them with a different phase, such
    that their sum corresponds to the evolved density matrix. The number of
    branches depends on the number of non-Clifford gates in `circuit`.

    Parameters
    ----------
    circuit: Circuit
        Circuit to use to evolve `pauli_string`.
    pauli_string: {Circuit, dict[str, float]}
        Pauli string to be evolved. `pauli_string` must be a `Circuit` composed
        of single qubit Pauli `Gate`s (that is, either `Gate('I')`, `Gate('X')`,
        `Gate('Y')` or `Gate('Z')`), each one acting on every qubit of
        `circuit`. If a dictionary is provided, every key of `pauli_string` must
        be a valid Pauli string. The size of each Pauli string must be equal to
        the number of qubits in `circuit`. Values in `pauli_string` will be
        used as inital phase for the given string.
    phase: float, optional
        Initial phase for `pauli_string`.
    atol: float, optional
        Discard all Pauli strings that have an absolute amplitude smaller than
        `atol`.
    parallel: int, optional
        Parallelize simulation (where possible). If `True`, the number of
        available cpus is used. Otherwise, a `parallel` number of threads is
        used.
    return_info: bool
        Return extra information collected during the evolution.
    use_mpi: bool, optional
        Use `MPI` if available. Unless `use_mpi=False`, `MPI` will be used if
        detected (for instance, if `mpiexec` is used to called HybridQ). If
        `use_mpi=True`, force the use of `MPI` (in case `MPI` is not
        automatically detected).
    compress: int, optional
        Compress `Circuit` using `utils.compress` prior the simulation.
    simplify: bool, optional
        Simplify `Circuit` using `utils.simplify` prior the simulation.
    remove_id_gates: bool, optional
        Remove `ID` gates prior the simulation.
    float_type: any, optional
        Float type to use for the simulation.
    verbose: bool, optional
        Verbose output.

    Returns
    -------
    dict[str, float] [, dict[any, any]]
        If `return_info=False`, `update_pauli_string` returns a `dict` of Pauli
        strings and the corresponding amplitude. The full density matrix can be
        reconstructed by resumming over all the Pauli string, weighted with the
        corresponding amplitude. If `return_info=True`, information gathered
        during the simulation are also returned.

    Other Parameters
    ----------------
    eps: float, optional (default: auto)
        Do not branch if the branch weight for the given non-Clifford operation
        is smaller than `eps`. `atol=1e-7` if `float_type=float32`, otherwise `atol=1e-8`
        if `float_type=float64`.
    atol: float, optional (default: auto)
        Remove elements from final state if such element as an absolute amplitude
        smaller than `atol`. `atol=1e-8` if `float_type=float32`, otherwise `atol=1e-12`
        if `float_type=float64`.
    branch_atol: float, optional
        Stop branching if the branch absolute amplitude is smaller than
        `branch_atol`. If not specified, it will be equal to `atol`.
    max_breadth_first_branches: int (default: auto)
        Max number of branches to collect using breadth first search. The number
        of branches collect during the breadth first phase will be split among
        the different threads (or nodes if using `MPI`).
    n_chunks: int (default: auto)
        Number of chunks to divide the branches obtained during the breadth
        first phase. The default value is twelve times the number of threads.
    max_virtual_memory: float (default: 80)
        Max virtual memory (%) that can be using during the simulation. If the
        used virtual memory is above `max_virtual_memory`, `update_pauli_string`
        will raise an error.
    sleep_time: float (default: 0.1)
        Completition of parallel processes is checked every `sleep_time`
        seconds.

    Example
    -------
    >>> from hybridq.circuit import utils
    >>> import numpy as np
    >>>
    >>> # Define circuit
    >>> circuit = Circuit(
    >>>     [Gate('X', qubits=[0])**1.2,
    >>>      Gate('ISWAP', qubits=[0, 1])**2.3])
    >>>
    >>> # Define Pauli string
    >>> pauli_string = Circuit([Gate('Z', qubits=[1])])
    >>>
    >>> # Get density matrix decomposed in Pauli strings
    >>> dm = clifford.update_pauli_string(circuit=circuit,
    >>>                                   pauli_string=pauli_string,
    >>>                                   float_type='float64')
    >>>
    >>> dm
    defaultdict(<function hybridq.circuit.simulation.clifford.update_pauli_string.<locals>._db_init.<locals>.<lambda>()>,
                {'IZ': 0.7938926261462365,
                 'YI': -0.12114687473997318,
                 'ZI': -0.166744368113685,
                 'ZX': 0.2377641290737882,
                 'YX': -0.3272542485937367,
                 'XY': -0.40450849718747345})
    >>> # Reconstruct density matrix
    >>> U = sum(phase * np.kron(Gate(g1).matrix(),
    >>>                         Gate(g2).matrix()) for (g1, g2), phase in dm.items())
    >>>
    >>> U
    array([[ 0.62714826+0.j        ,  0.23776413+0.j        ,
             0.        +0.12114687j,  0.        +0.73176275j],
           [ 0.23776413+0.j        , -0.96063699+0.j        ,
             0.        -0.07725425j,  0.        +0.12114687j],
           [ 0.        -0.12114687j,  0.        +0.07725425j,
             0.96063699+0.j        , -0.23776413+0.j        ],
           [ 0.        -0.73176275j,  0.        -0.12114687j,
            -0.23776413+0.j        , -0.62714826+0.j        ]])
    >>> np.allclose(utils.matrix(circuit + pauli_string + circuit.inv()),
    >>>             U,
    >>>             atol=1e-8)
    True
    >>> U[0b11, 0b11]
    (-0.6271482580325515+0j)
    """

    # ==== Set default parameters ====

    # If use_mpi==False, force the non-use of MPI
    if use_mpi is None and _detect_mpi:

        # Warn that MPI is used because detected
        warn("MPI has been detected. Using MPI.")

        # Set MPI to true
        use_mpi = True

    # If parallel==True, use number of cpus
    if type(parallel) is bool:
        parallel = cpu_count() if parallel else 1
    else:
        parallel = int(parallel)
        if parallel <= 0:
            warn("'parallel' must be a positive integer. Setting parallel=1")
            parallel = 1

    # utils.globalize may not work properly on MacOSX systems .. for now, let's
    # disable parallelization for MacOSX
    if parallel > 1:
        from platform import system
        from warnings import warn

        if system() == 'Darwin':
            warn(
                "'utils.globalize' may not work on MacOSX. Disabling parallelization."
            )
            parallel = 1

    # Fix atol
    if 'atol' in kwargs:
        atol = kwargs['atol']
        del (kwargs['atol'])
    else:
        float_type = np.dtype(float_type)
        if float_type == np.float64:
            atol = 1e-12
        elif float_type == np.float32:
            atol = 1e-8
        else:
            raise ValueError(f'Unsupported array dtype: {float_type}')

    # Fix branch_atol
    if 'branch_atol' in kwargs:
        branch_atol = kwargs['branch_atol']
        del (kwargs['branch_atol'])
    else:
        branch_atol = atol

    # Fix eps
    if 'eps' in kwargs:
        eps = kwargs['eps']
        del (kwargs['eps'])
    else:
        float_type = np.dtype(float_type)
        if float_type == np.float64:
            eps = 1e-8
        elif float_type == np.float32:
            eps = 1e-7
        else:
            raise ValueError(f'Unsupported array dtype: {float_type}')

    # Set default db initialization
    def _db_init():
        return defaultdict(int)

    # Set default transform
    def _transform(ps):

        # Join bitstring
        return ''.join({_X: 'X', _Y: 'Y', _Z: 'Z', _I: 'I'}[op] for op in ps)

    # Set default collect
    def _collect(db, ps, ph):

        # Update final paulis
        db[ps] += ph

        # Remove elements close to zero
        if abs(db[ps]) < atol:
            del (db[ps])

    # Set default merge
    def _merge(db, db_new, use_tuple=False):

        # Update final paulis
        for ps, ph in db_new if use_tuple else db_new.items():

            # Collect results
            kwargs['collect'](db, ps, ph)

    kwargs.setdefault('max_breadth_first_branches', min(4 * 12 * parallel,
                                                        2**14))
    kwargs.setdefault('n_chunks', 12 * parallel)
    kwargs.setdefault('max_virtual_memory', 80)
    kwargs.setdefault('sleep_time', 0.1)
    kwargs.setdefault('collect', _collect)
    kwargs.setdefault('transform', _transform)
    kwargs.setdefault('merge', _merge)
    kwargs.setdefault('db_init', _db_init)

    # Get MPI info
    if use_mpi:
        from mpi4py import MPI
        _mpi_comm = MPI.COMM_WORLD
        _mpi_size = _mpi_comm.Get_size()
        _mpi_rank = _mpi_comm.Get_rank()
        kwargs.setdefault('max_breadth_first_branches_mpi',
                          min(_mpi_size * 2**9, 2**14))
        kwargs.setdefault('mpi_chunk_max_size', 2**20)
        kwargs.setdefault('mpi_merge', True)

    # Get complex_type from float_type
    complex_type = (np.array([1], dtype=float_type) +
                    1j * np.array([1], dtype=float_type)).dtype

    # Local verbose
    _verbose = verbose and (not use_mpi or _mpi_rank == 0)

    # =========== CHECKS =============

    if type(pauli_string) == Circuit:
        from collections import Counter

        # Initialize error message
        _err_msg = "'pauli_string' must contain only I, X, Y and Z gates acting on different qubits."

        # Check qubits match with circuit
        if any(g.n_qubits != 1 or not g.qubits for g in pauli_string) or set(
                pauli_string.all_qubits()).difference(
                    circuit.all_qubits()) or set(
                        Counter(gate.qubits[0]
                                for gate in pauli_string).values()).difference(
                                    [1]):
            raise ValueError(_err_msg)

        # Get ideal paulis
        _ig = list(map(lambda n: Gate(n).matrix(), 'IXYZ'))

        # Get the correct pauli
        def _get_pauli(gate):
            # Get matrix
            U = gate.matrix()

            # Get right pauli
            p = next(
                (p for x, p in enumerate('IXYZ') if np.allclose(_ig[x], U)),
                None)

            # If not found, raise error
            if not p:
                raise ValueError(_err_msg)

            # Otherwise, return pauli
            return Gate(p, qubits=gate.qubits)

        # Reconstruct paulis
        pauli_string = Circuit(map(_get_pauli, pauli_string))

    else:

        # Check that all strings only have I,X,Y,Z tokens
        _n_qubits = len(circuit.all_qubits())
        if any(
                set(p).difference('IXYZ') or len(p) != _n_qubits
                for p in pauli_string):
            raise ValueError(
                f"'pauli_string' must contain only I, X, Y and Z gates acting on different qubits."
            )

    # ================================

    # Start pre-processing time
    _prep_time = time()

    # Get qubits
    _qubits = circuit.all_qubits()

    # Remove ID gates
    if remove_id_gates:
        circuit = Circuit(gate for gate in circuit if gate.name != 'I')

    # Simplify circuit
    if simplify:
        # Get qubits to pin
        if type(pauli_string) == Circuit:
            # Pinned qubits
            _pinned_qubits = pauli_string.all_qubits()

        else:
            # Find qubits to pin
            _pinned_qubits = set.union(
                *({q
                   for q, g in zip(_qubits, p)
                   if g != 'I'}
                  for p in pauli_string))

        # Simplify
        circuit = utils.simplify(circuit,
                                 remove_id_gates=remove_id_gates,
                                 verbose=_verbose)
        circuit = utils.popright(utils.simplify(circuit),
                                 pinned_qubits=set(_pinned_qubits).intersection(
                                     circuit.all_qubits()),
                                 verbose=_verbose)

    # Compress circuit
    circuit = Circuit(
        utils.to_matrix_gate(c, complex_type=complex_type)
        for c in tqdm(utils.compress(circuit, max_n_qubits=compress),
                      disable=not _verbose,
                      desc=f"Compress ({int(compress)})"))

    # Pad missing qubits
    circuit += Circuit(
        Gate('MATRIX', [q], U=np.eye(2))
        for q in set(_qubits).difference(circuit.all_qubits()))

    # Get qubits map
    qubits_map = kwargs['qubits_map'] if 'qubits_map' in kwargs else {
        q: x for x, q in enumerate(circuit.all_qubits())
    }

    # Pre-process circuit
    _LS_cache = {}
    _P_cache = {}
    circuit = [
        g for gate in tqdm(reversed(circuit),
                           total=len(circuit),
                           disable=not _verbose,
                           desc='Pre-processing')
        for g in _process_gate(gate, LS_cache=_LS_cache, P_cache=_P_cache)
    ]
    _LS_cache.clear()
    _P_cache.clear()
    del (_LS_cache)
    del (_P_cache)

    # Get maximum number of qubits and parameters
    _max_n_qubits = max(max(len(gate[1]) for gate in circuit), 2)
    _max_n_params = max(len(gate[2]) for gate in circuit)

    # Get qubits
    qubits = np.array([
        np.pad([qubits_map[q]
                for q in gate[1]], (0, _max_n_qubits - len(gate[1])))
        for gate in circuit
    ],
                      dtype='int32')

    # Get parameters
    params = np.round(
        np.array([
            np.pad(gate[2], (0, _max_n_params - len(gate[2])))
            for gate in circuit
        ],
                 dtype=float_type),
        -int(np.floor(np.log10(atol))) if atol < 1 else 0)

    # Remove -0
    params[np.abs(params) == 0] = 0

    # Quick check
    assert (all('_' + gate[0] in globals() for gate in circuit))

    # Get gates
    gates = np.array([globals()['_' + gate[0]] for gate in circuit],
                     dtype='int')

    # Compute expected number of paths
    _log2_n_expected_branches = 0
    for _idx in np.where(np.isin(gates, _MATRIX_SET))[0]:
        _nq = (gates[_idx] // _gate_mul) + 1
        _p = params[_idx][:4**(2 * _nq)]
        _log2_n_expected_branches += np.sum(
            np.log2(
                np.sum(np.abs(np.reshape(_p, (4**_nq, 4**_nq))) > eps,
                       axis=1))) / 4**_nq

    # Check
    assert (len(gates) == len(qubits) and len(gates) == len(params))

    # Initialize branches
    if type(pauli_string) == Circuit:

        # Convert Pauli string
        _pauli_string = np.array([_I] * len(qubits_map), dtype='int')
        for gate in pauli_string:
            if gate.name != 'I':
                _pauli_string[qubits_map[gate.qubits[0]]] = {
                    'X': _X,
                    'Y': _Y,
                    'Z': _Z
                }[gate.name]

        # Initialize branches
        branches = [(_pauli_string, phase, 0)]

    else:

        # Initialize branches
        branches = [(np.array([{
            'I': _I,
            'X': _X,
            'Y': _Y,
            'Z': _Z
        }[g]
                               for g in p],
                              dtype='int'), phase, 0)
                    for p, phase in pauli_string.items()
                    if abs(phase) > atol]

    # Initialize final Pauli strings
    db = kwargs['db_init']()

    # Define update function
    _update = partial_func(_update_pauli_string,
                           gates,
                           qubits,
                           params,
                           eps=eps,
                           atol=branch_atol)

    # End pre-processing time
    _prep_time = time() - _prep_time

    # Initialize infos
    _info_init = lambda: {
        'n_explored_branches': 0,
        'largest_n_branches_in_memory': 0,
        'peak_virtual_memory (GB)': virtual_memory().used / 2**30,
        'average_virtual_memory (GB)': (virtual_memory().used / 2**30, 1),
        'n_threads': parallel,
        'n_cpus': cpu_count(),
        'eps': eps,
        'atol': atol,
        'branch_atol': branch_atol,
        'float_type': str(float_type),
        'log2_n_expected_branches': _log2_n_expected_branches
    }
    infos = _info_init()
    infos['memory_baseline (GB)'] = virtual_memory().used / 2**30
    if not use_mpi or _mpi_rank == 0:
        infos['n_explored_branches'] = 1
        infos['largest_n_branches_in_memory'] = 1

    # Start clock
    _init_time = time()

    # Scatter first batch of branches to different MPI nodes
    if use_mpi and _mpi_size > 1:

        if _mpi_rank == 0:

            # Explore branches (breadth-first search)
            branches = _breadth_first_search(
                _update,
                db,
                branches,
                max_n_branches=kwargs['max_breadth_first_branches_mpi'],
                infos=infos,
                verbose=verbose,
                mpi_rank=_mpi_rank,
                **kwargs)

        # Distribute branches
        branches = _mpi_comm.scatter(
            [list(x) for x in distribute(_mpi_size, branches)], root=0)

    # Explore branches (breadth-first search)
    branches = _breadth_first_search(
        _update,
        db,
        branches,
        max_n_branches=kwargs['max_breadth_first_branches'],
        infos=infos,
        verbose=verbose if not use_mpi or _mpi_rank == 0 else False,
        **kwargs)

    # If there are remaining branches, use depth-first search
    if branches:

        _depth_first_search(_update,
                            db,
                            branches,
                            parallel=parallel,
                            infos=infos,
                            info_init=_info_init,
                            verbose=verbose,
                            mpi_rank=_mpi_rank if use_mpi else 0,
                            mpi_size=_mpi_size if use_mpi else 1,
                            **kwargs)

    # Update infos
    infos['average_virtual_memory (GB)'] = infos['average_virtual_memory (GB)'][
        0] / infos['average_virtual_memory (GB)'][1] - infos[
            'memory_baseline (GB)']
    infos['peak_virtual_memory (GB)'] -= infos['memory_baseline (GB)']

    # Update branching time
    infos['branching_time (s)'] = time() - _init_time

    # Collect results
    if use_mpi and _mpi_size > 1 and kwargs['mpi_merge']:

        for _k in infos:
            infos[_k] = [infos[_k]]

        # Initialize pbar
        if _mpi_rank == 0:
            pbar = tqdm(total=int(np.ceil(np.log2(_mpi_size))),
                        disable=not verbose,
                        desc='Collect results')

        # Initialize tag and size
        _tag = 0
        _size = _mpi_size
        while _size > 1:
            # Update progressbar
            if _mpi_rank == 0:
                pbar.set_description(
                    f'Collect results (Mem={virtual_memory().percent}%)')

            # Get shift
            _shift = (_size // 2) + (_size % 2)

            if _mpi_rank < (_size // 2):
                # Get infos
                _infos = _mpi_comm.recv(source=_mpi_rank + _shift, tag=_tag)

                # Update infos
                for _k in infos:
                    infos[_k].extend(_infos[_k])

                # Get number of chunks
                _n_chunks = _mpi_comm.recv(source=_mpi_rank + _shift,
                                           tag=_tag + 1)

                if _n_chunks > 1:
                    # Initialize _process
                    with tqdm(range(_n_chunks),
                              desc='Get db',
                              leave=False,
                              disable=_mpi_rank != 0) as pbar:
                        for _ in pbar:
                            # Receive db
                            _db = _mpi_comm.recv(source=_mpi_rank + _shift,
                                                 tag=_tag + 2)

                            # Merge datasets
                            kwargs['merge'](db, _db, use_tuple=True)

                            # Update description
                            pbar.set_description(
                                f'Get db (Mem={virtual_memory().percent}%)')

                            # Clear dataset
                            _db.clear()

                else:
                    # Receive db
                    _db = _mpi_comm.recv(source=_mpi_rank + _shift,
                                         tag=_tag + 2)

                    # Merge datasets
                    kwargs['merge'](db, _db)

                    # Clear dataset
                    _db.clear()

            elif _shift <= _mpi_rank < _size:
                # Remove default_factory because pickle is picky regarding local objects
                db.default_factory = None

                # Send infos
                _mpi_comm.send(infos, dest=_mpi_rank - _shift, tag=_tag)

                # Compute chunks
                _n_chunks = kwargs['mpi_chunk_max_size']
                _n_chunks = (len(db) // _n_chunks) + (
                    (len(db) % _n_chunks) != 0)

                # Send number of chunks
                _mpi_comm.send(_n_chunks, dest=_mpi_rank - _shift, tag=_tag + 1)

                if _n_chunks > 1:
                    # Split db in chunks
                    for _db in chunked(db.items(),
                                       kwargs['mpi_chunk_max_size']):
                        _mpi_comm.send(_db,
                                       dest=_mpi_rank - _shift,
                                       tag=_tag + 2)
                else:
                    # Send db
                    _mpi_comm.send(db, dest=_mpi_rank - _shift, tag=_tag + 2)

                # Reset db and infos
                db.clear()
                infos.clear()

            # update size
            _tag += 3
            _size = _shift
            _mpi_comm.barrier()

            # Update progressbar
            if _mpi_rank == 0:
                pbar.set_description(
                    f'Collect results (Mem={virtual_memory().percent}%)')
                pbar.update()

    # Update runtime
    if not use_mpi or _mpi_rank == 0 or not kwargs['mpi_merge']:
        infos['runtime (s)'] = time() - _init_time
        infos['pre-processing (s)'] = _prep_time

    # Check that all the others dbs/infos (excluding rank==0) has been cleared up
    if use_mpi and _mpi_rank > 0 and kwargs['mpi_merge']:
        assert (not len(db) and not len(infos))

    if return_info:
        return db, infos
    else:
        return db


def expectation_value(circuit: Circuit, op: Circuit, initial_state: str,
                      **kwargs) -> float:
    """
    Compute the expectation value of `op` for the given `circuit`, using
    `initial_state` as initial state.

    Parameters
    ----------
    circuit: Circuit
        Circuit to simulate.
    op: Circuit
        Operator used to compute the expectation value. `op` must be a valid
        `Circuit` containing only Pauli gates (that is, either `I`, `X`, `Y` or
        `Z` gates) acting on different qubits.
    initial_state: str
        Initial state used to compute the expectation value. Valid tokens for
        `initial_state` are:

        - `0`: qubit is set to `0` in the computational basis,
        - `1`: qubit is set to `1` in the computational basis,
        - `+`: qubit is set to `+` state in the computational basis,
        - `-`: qubit is set to `-` state in the computational basis.

    Returns
    -------
    float [, dict[any, any]]
        The expectation value of the operator `op`. If `return_info=True`,
        information gathered during the simulation are also returned.

    Other Parameters
    ----------------
    `expectation_value` uses all valid parameters for `update_pauli_string`.

    See Also
    --------
    `update_pauli_string`

    Example
    -------
    >>> # Define circuit
    >>> circuit = Circuit(
    >>>     [Gate('X', qubits=[0])**1.2,
    >>>      Gate('ISWAP', qubits=[0, 1])**2.3])
    >>>
    >>> # Define operator
    >>> op = Circuit([Gate('Z', qubits=[1])])
    >>>
    >>> # Get expectation value
    >>> clifford.expectation_value(circuit=circuit,
    >>>                            op=op,
    >>>                            initial_state='11',
    >>>                            float_type='float64')
    -0.6271482580325515
    """

    # ==== Set default parameters ====

    def _db_init():
        return [0]

    def _collect(db, ops, ph):
        # Compute expectation value given pauli string
        if next((False for x in ops if x in 'XY'), True):
            db[0] += ph

    def _merge(db, db_new):
        db[0] += db_new[0]

    def _prepare_state(state, qubits):
        c = Circuit()
        for q, s in zip(qubits, state):
            if s == '0':
                pass
            elif s == '1':
                c.append(Gate('X', [q]))
            elif s == '+':
                c.append(Gate('H', [q]))
            elif s == '-':
                c.extend([Gate('X', [q]), Gate('H', [q])])
            else:
                raise ValueError(f"Unexpected token '{s}'")
        return c

    kwargs.setdefault('use_mpi', None)
    kwargs.setdefault('return_info', False)

    # If use_mpi==False, force the non-use of MPI
    if kwargs['use_mpi'] is None and _detect_mpi:

        # Warn that MPI is used because detected
        warn("MPI has been detected. Using MPI.")

        # Set MPI to true
        kwargs['use_mpi'] = True

    # Get MPI info
    if kwargs['use_mpi']:
        from mpi4py import MPI
        _mpi_comm = MPI.COMM_WORLD
        _mpi_size = _mpi_comm.Get_size()
        _mpi_rank = _mpi_comm.Get_rank()

    # ================================

    # Prepare initial state
    if type(initial_state) == str:
        # Check
        if len(initial_state) != len(circuit.all_qubits()):
            raise ValueError(
                f"'initial_state' has the wrong number of qubits "
                f"(expected {len(circuit.all_qubits())}, got {len(initial_state)})."
            )
        # Get state
        initial_state = _prepare_state(initial_state, circuit.all_qubits())
    else:
        raise ValueError(
            f"'{type(initial_state)}' not supported for 'initial_state'.")

    # Get expectation value
    _res = update_pauli_string(initial_state + circuit,
                               op,
                               db_init=_db_init,
                               collect=_collect,
                               merge=_merge,
                               mpi_merge=False,
                               **kwargs)

    # Collect results
    if kwargs['use_mpi'] and _mpi_size > 1:
        _all_res = _mpi_comm.gather(_res, root=0)
        if _mpi_rank == 0:
            if kwargs['return_info']:
                infos = {}
                for _, _infos in _all_res:
                    for _k, _v in _infos.items():
                        if _k not in infos:
                            infos[_k] = [_v]
                        else:
                            infos[_k].append(_v)
                exp_value = sum(ev[0] for ev, _ in _all_res)
            else:
                exp_value = sum(ev[0] for ev in _all_res)
        else:
            exp_value = infos = None
    else:
        if kwargs['return_info']:
            exp_value = _res[0][0]
            infos = _res[1]
        else:
            exp_value = _res[0]

    # Return expectation value
    if kwargs['return_info']:
        return exp_value, infos
    else:
        return exp_value
