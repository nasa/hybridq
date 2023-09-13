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

from collections import defaultdict
from multiprocessing import Pool
import more_itertools as mit
from threading import Timer
import functools as fts
import itertools as its
from time import time
import numpy as np
import numba
import sys
import os

__all__ = ['DecomposeOperator', 'FromPauliString', 'ToPauliString', 'simulate']

# Initialize Pauli matrices
Paulis_ = dict(I=np.eye(2),
               X=np.array([[0, 1], [1, 0]]),
               Y=np.array([[0, -1j], [1j, 0]]),
               Z=np.array([[1, 0], [0, -1]]))


@fts.lru_cache
def GetPauliOperator_(op, dtype='complex64'):
    return fts.reduce(np.kron, map(lambda g: Paulis_[g].astype(dtype), op))


@fts.lru_cache
def GenerateLinearSystem_(n_qubits, dtype='complex64'):
    return np.reshape(
        np.array(
            list(
                map(
                    fts.partial(GetPauliOperator_, dtype=dtype),
                    map(fts.partial(ToPauliString, n_paulis=n_qubits),
                        range(4**n_qubits))))),
        (4**n_qubits, 4**n_qubits)).conj() / 2**n_qubits


def ToPauliString(x, n_paulis):
    return ''.join(map(lambda i: 'IXYZ'[(x // 4**i) % 4], range(n_paulis)))


def FromPauliString(pstr):
    return sum(4**i_ * dict(I=0, X=1, Y=2, Z=3)[p_]
               for i_, p_ in enumerate(pstr.upper()))


def GetComplexType_(dtype):
    dtype = np.dtype(dtype)
    if np.iscomplexobj(dtype.type(1)):
        return dtype
    else:
        return (np.ones(1, dtype=dtype) + 1j * np.ones(1, dtype=dtype)).dtype


# Get operator decomposition
def DecomposeOperator(gate):

    # Gate must be a quadratic matrix of size 2^n
    if gate.ndim != 2 or gate.shape[0] != gate.shape[1] or int(
            np.log2(gate.shape[0])) != np.log2(gate.shape[0]):
        raise ValueError("'gate' is not a valid")

    # Get number of qubits
    n_ = int(np.log2(gate.shape[0]))

    # Generate linear system
    LS_ = GenerateLinearSystem_(n_qubits=n_, dtype=GetComplexType_(gate.dtype))

    # Return decomposition
    return np.real_if_close((LS_ @ np.array(
        list(
            map(
                lambda op: (gate @ op @ gate.conj().T).ravel(),
                map(
                    fts.partial(GetPauliOperator_,
                                dtype=GetComplexType_(gate.dtype)),
                    map(fts.partial(ToPauliString, n_paulis=n_), range(
                        4**n_)))))).T)).T


def StateFromPauliString_(pstr):
    return np.fromiter(
        (map(FromPauliString, map(''.join, mit.chunked(pstr, 4)))),
        dtype='uint8')


def ToPauliStringFromState_(state, n_paulis):
    return ''.join(map(fts.partial(ToPauliString, n_paulis=4),
                       state))[:n_paulis]


@numba.njit
def GetSubState_(state, qubits):
    return sum([
        4**i_ * ((state[q_ // 4] >> (2 * (q_ % 4))) & 0b11)
        for i_, q_ in enumerate(qubits)
    ])


@numba.njit
def UpdateState_(state, sub_state, qubits):
    for p_, q_ in zip([(sub_state // 4**i_) % 4 for i_ in range(len(qubits))],
                      qubits):
        state[q_ // 4] &= ~(0b11 << (2 * (q_ % 4)))
        state[q_ // 4] ^= p_ << (2 * (q_ % 4))
    return state


@numba.njit
def UpdateBranch_(branch, gates, gate_qubits, norm_atol=1e-8, atol=1e-8):
    # Get current branch
    state_, phase_, norm_phase_, gate_idx_ = branch

    # Get qubits and phases associated with gate
    qs_, phases_ = gate_qubits[gate_idx_], gates[gate_idx_]

    # Get new phases
    ph_ = phases_[GetSubState_(state_, qs_)]

    # Get absolute value of phases
    abs_ph_ = np.abs(ph_)

    # Renormalize phase
    norm_phase_ = np.abs(norm_phase_) / np.max(abs_ph_)

    # Get only relevant positions
    ps_ = np.where((np.abs(phase_) * abs_ph_ > atol) &
                   (norm_phase_ * abs_ph_ > norm_atol))[0]

    # Get new states
    new_states_ = [UpdateState_(state_.copy(), p_, qs_) for p_ in ps_]

    # Get new phases
    new_phases_ = ph_[ps_] * phase_

    # Get new normalized phases
    new_norm_phases_ = ph_[ps_] * norm_phase_

    # Get new gate idxs
    new_gate_idxs_ = [gate_idx_ + 1] * len(new_states_)

    # Zip all together
    return list(zip(new_states_, new_phases_, new_norm_phases_, new_gate_idxs_))


def UpdateBranches_(branches,
                    gates,
                    gate_qubits,
                    depth_first=True,
                    *,
                    max_time=None,
                    max_n_branches=None,
                    norm_atol=1e-8,
                    atol=1e-8,
                    verbose=False):

    # Define how to print status bar
    def status_bar_(n_branches, n_completed_branches, n_explored_branches,
                    elapsed):
        from datetime import timedelta
        if n_explored_branches:
            bt_ = elapsed / n_explored_branches
            if bt_ < 1e-3:
                bt_ = f'{bt_*1e6:1.2f}μs/branch'
            elif bt_ < 1:
                bt_ = f'{bt_*1e3:1.2f}ms/branch'
            else:
                bt_ = f'{bt_:1.2f}s/branch'
        else:
            bt_ = 'n/a'

        s_ = f'PB:{n_branches:,}/CB:{n_completed_branches:,} '
        s_ += f'[{timedelta(seconds=round(elapsed))}, {bt_}]'
        if len(s_) > 100:
            s_ = s_[:-3] + '...'
        s_ += ' ' * (100 - len(s_))
        return s_

    # Initialize stop flag
    flags_ = [1]

    def set_flag_():
        flags_[0] = 0

    # Initialize timer
    if max_time is not None:
        Timer(max_time, set_flag_).start()

    # Convert gates and gate_qubits to numba.typed.List
    gates = numba.typed.List(gates)
    gate_qubits = numba.typed.List(map(numba.typed.List, gate_qubits))

    # Initialize completed branches
    completed_ = defaultdict(float)

    # Initialize number of total branches
    n_completed_branches_ = 0

    # Create core to call
    UB_ = fts.partial(UpdateBranch_,
                      gates=gates,
                      gate_qubits=gate_qubits,
                      norm_atol=norm_atol,
                      atol=atol)

    # Get initial time
    it_ = time()

    # Initialize number of explored branches
    n_explored_branches_ = 0

    # While there are still some branches ...
    while len(branches) and flags_[0] and (max_n_branches is None or
                                           len(branches) < max_n_branches):
        # Get new branches
        new_branches_ = UB_(branches.pop(-1 if depth_first else 0))

        # If new branches are leaves, append to completed_
        if new_branches_ and new_branches_[-1][-1] == len(gates):
            # Update number of total branches
            n_completed_branches_ += len(new_branches_)

            # Dump to db
            for br_, ph_, norm_ph_, _ in new_branches_:
                completed_[tuple(br_)] += ph_

        # Otherwise, add them to branches
        else:
            branches += new_branches_

        # Increment counter
        n_explored_branches_ += 1

        # Print status
        if verbose and (n_explored_branches_ % 10_000 == 0):
            print(status_bar_(len(branches), n_completed_branches_,
                              n_explored_branches_,
                              time() - it_),
                  file=sys.stderr,
                  end='\r',
                  flush=True)

    # Get total time
    total_time_ = time() - it_

    # Get branching time
    branching_time_ = total_time_ / n_explored_branches_ if n_explored_branches_ else np.nan

    # Print some extra infos
    if verbose:
        print(f"\nTotal Time: {total_time_:1.2g}s", file=sys.stderr, flush=True)
        print(f"Number Explored Branches: {n_explored_branches_:,}",
              file=sys.stderr,
              flush=True)
        print(f"Number Completed Branches: {n_completed_branches_:,}",
              file=sys.stderr,
              flush=True)
        print(f"Branching Time: {branching_time_ * 1e6:1.2g}μs",
              file=sys.stderr,
              flush=True)

    # Return completed and non-completed branches
    return completed_, branches, dict(
        n_completed_branches=n_completed_branches_,
        n_explored_branches=n_explored_branches_,
        total_time=total_time_)


# Define how to merge branches from threads
def merge_branches_(all_branches, branches):
    # For all partial completed branches ...
    for x_ in branches:
        # Update db of completed branches
        for k_, v_ in x_.items():
            all_branches[k_] += v_

    # Return merged data
    return all_branches


# Define how to merge infos from threads
def merge_info_(all_info, info):
    # Merge total number of branches
    all_info['n_completed_branches'] += sum(
        map(lambda x: x['n_completed_branches'], info))
    all_info['n_explored_branches'] += sum(
        map(lambda x: x['n_explored_branches'], info))

    # Return merged info
    return all_info


def UpdateBranchesParallel_(branches,
                            gates,
                            gate_qubits,
                            *,
                            n_threads=None,
                            max_time=None,
                            thread_max_time=5,
                            norm_atol=1e-8,
                            atol=1e-8,
                            verbose=False,
                            completed_=None,
                            info_=None,
                            merge_branches_=merge_branches_,
                            merge_info_=merge_info_):

    # Create partial function to parallelize
    UB_ = fts.partial(UpdateBranches_,
                      gates=gates,
                      gate_qubits=gate_qubits,
                      norm_atol=norm_atol,
                      atol=atol,
                      max_time=thread_max_time)

    # Get number of threads
    n_threads = len(os.sched_getaffinity(
        os.getpid())) if n_threads is None else int(n_threads)

    # Initialize db of all completed branches
    all_ = defaultdict(float)
    all_info_ = dict(n_completed_branches=0, n_explored_branches=0)

    # Initialize branches
    branches_ = [branches]
    completed_ = [] if None else [completed_]
    info_ = [] if None else [info_]

    # Initialize stop flag
    flags_ = [1]

    def set_flag_():
        flags_[0] = 0

    # Initialize timer
    if max_time is not None:
        Timer(max_time, set_flag_).start()

    # Initialize time
    it_ = time()

    # Initialize remaining branches
    n_rem_branches_ = sum(map(len, branches_))

    # Define how to print status bar
    def status_bar_(n_branches, n_completed_branches, n_explored_branches,
                    elapsed):
        from psutil import cpu_percent, getloadavg, virtual_memory
        from datetime import timedelta
        if n_explored_branches:
            bt_ = elapsed / n_explored_branches
            if bt_ < 1e-3:
                bt_ = f'{bt_*1e6:1.2f}μs/branch'
            elif bt_ < 1:
                bt_ = f'{bt_*1e3:1.2f}ms/branch'
            else:
                bt_ = f'{bt_:1.2f}s/branch'
        else:
            bt_ = 'n/a'

        s_ = f'PB:{n_branches:,}/CB:{n_completed_branches:,} '
        s_ += f'CPU:{cpu_percent()}%, LAVG:{getloadavg()[0]:1.2f}, '
        s_ += f'MEM:{virtual_memory()[2]:1.1f}% '
        s_ += f'[{timedelta(seconds=round(elapsed))}, {bt_}]'
        if len(s_) > 100:
            s_ = s_[:-3] + '...'
        s_ += ' ' * (100 - len(s_))
        return s_

    # Print number of used threads
    if verbose:
        print(f'Starting simulation with {n_threads} threads.',
              file=sys.stderr,
              flush=True)

    # Initialize parallel pool
    with Pool(n_threads) as pool_:

        # While there are still branches ...
        while n_rem_branches_ and flags_[0]:

            # Print status bar
            if verbose:
                print(status_bar_(n_rem_branches_,
                                  all_info_['n_completed_branches'],
                                  all_info_['n_explored_branches'],
                                  time() - it_),
                      file=sys.stderr,
                      end='\r',
                      flush=True)

            # Get handlers for parallel threads
            h_ = list(
                map(
                    lambda x: pool_.apply_async(UB_, args=(x,)),
                    map(list, mit.distribute(n_threads,
                                             mit.flatten(branches_)))))

            # Merge info
            all_info_ = merge_info_(all_info_, info_)

            # Merge completed branches
            all_ = merge_branches_(all_, completed_)

            # Wait till threads are complete
            completed_, branches_, info_ = zip(*map(lambda x: x.get(), h_))

            # Update number of remaining branches
            n_rem_branches_ = sum(map(len, branches_))

        # Last merge of completed branches
        all_ = merge_branches_(all_, completed_)

        # Last merge of info
        all_info_ = merge_info_(all_info_, info_)

    # Get final info
    all_info_['total_time_s'] = time() - it_
    all_info_['branching_time_s'] = all_info_['total_time_s'] / all_info_[
        'n_explored_branches'] if all_info_['n_explored_branches'] else np.nan

    # Print some extra infos
    if verbose:
        print(f"\nTotal Time: {all_info_['total_time_s']:1.2g}s",
              file=sys.stderr,
              flush=True)
        print(f"Number Explored Branches: {all_info_['n_explored_branches']:,}",
              file=sys.stderr,
              flush=True)
        print(
            f"Number Completed Branches: {all_info_['n_completed_branches']:,}",
            file=sys.stderr,
            flush=True)
        print(f"Branching Time: {all_info_['branching_time_s'] * 1e6:1.2g}μs",
              file=sys.stderr,
              flush=True)

    # Return results
    return all_, list(mit.flatten(branches_)), all_info_


def simulate(circuit,
             paulis=None,
             branches=None,
             *,
             parallel=True,
             norm_atol=1e-8,
             atol=1e-8,
             max_time=None,
             thread_max_time=1,
             verbose=False,
             **kwargs):

    # Provide either 'paulis' or 'branches', but not both
    if not ((paulis is not None) ^ (branches is not None)):
        raise ValueError("Provide either 'paulis' or 'branches', but not both")

    # Convert parallel to number of threads
    parallel = (len(os.sched_getaffinity(os.getpid())) if parallel else
                0) if isinstance(parallel, bool) else int(parallel)

    # Convert paulis
    if isinstance(paulis, str):
        paulis = {paulis: 1}

    # Split matrices and qubits
    gates_, gate_qubits_ = zip(*map(
        lambda x: (DecomposeOperator(x[0]), tuple(map(int, x[1]))), circuit))

    # Initialize branches
    branches_ = list(
        map(lambda x: (StateFromPauliString_(x[0]), x[1], x[1], 0),
            paulis.items())) if paulis is not None else branches

    # Run in parallel if needed ...
    if parallel:

        # Get first shallow batch of branches
        completed_, branches_, info_ = UpdateBranches_(
            branches_,
            gates_,
            gate_qubits_,
            depth_first=False,
            max_n_branches=kwargs.get('breadth_first_max_n_branches',
                                      10 * parallel),
            max_time=kwargs.get('breadth_first_max_time', 10),
            norm_atol=norm_atol,
            atol=atol,
            verbose=False)

        # Get all branches
        all_, branches_, all_info_ = UpdateBranchesParallel_(
            branches=branches_,
            gates=gates_,
            gate_qubits=gate_qubits_,
            completed_=completed_,
            info_=info_,
            norm_atol=norm_atol,
            atol=atol,
            max_time=max_time,
            thread_max_time=thread_max_time,
            n_threads=parallel,
            verbose=verbose,
            **kwargs)

        # All branches should have been explored
        if max_time is None:
            assert (not len(branches_))

    # Otherwise, run sequentially
    else:
        # Get all branches
        all_, branches_, all_info_ = UpdateBranches_(branches_,
                                                     gates_,
                                                     gate_qubits_,
                                                     norm_atol=norm_atol,
                                                     atol=atol,
                                                     max_time=max_time,
                                                     verbose=verbose,
                                                     **kwargs)

        # All branches should have been explored
        if max_time is None:
            assert (not len(branches_))

    # Append results to all_info_
    all_info_['branches'] = all_
    all_info_['partial_branches'] = branches_

    # Return results
    return all_info_