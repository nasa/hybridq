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
from datetime import timedelta
import more_itertools as mit
from tqdm.auto import tqdm
import functools as fts
import itertools as its
import numpy as np
import psutil
import time
import sys
import os

__all__ = []

# Initialize Pauli matrices
Paulis_ = dict(I=np.eye(2),
               X=np.array([[0, 1], [1, 0]]),
               Y=np.array([[0, -1j], [1j, 0]]),
               Z=np.array([[1, 0], [0, -1]]))
Paulis_.update((i_, Paulis_[x_]) for i_, x_ in enumerate('IXYZ'))


def GetComplexType(dtype: np.dtype) -> np.dtype:
    """
    Convert `dtype` to a complex `dtype`.
    """
    dtype = np.dtype(dtype)
    if np.iscomplexobj(dtype.type(1)):
        return dtype
    else:
        return (np.ones(1, dtype=dtype) + 1j * np.ones(1, dtype=dtype)).dtype


def ToPauliString(x: int, n: int) -> str:
    """
    Convert an integer to a Pauli string.
    """
    return ''.join(map(lambda i: 'IXYZ'[(x // 4**i) % 4], range(n)))


@fts.lru_cache
def GetPauliOperator(op: iter[int] | str, dtype='complex64') -> np.array:
    """
    Given a Pauli string, return its matrix representation.
    """
    return fts.reduce(np.kron, map(lambda g: Paulis_[g].astype(dtype), op))


@fts.lru_cache
def GenerateLinearSystem(n: int, dtype: np.dtype = 'complex64'):
    """
    Generate Pauli's linear system of `n` qubits.
    """
    GPO_ = fts.partial(GetPauliOperator, dtype=dtype)
    TPS_ = fts.partial(ToPauliString, n=n)
    return np.reshape(
        list(map(lambda x: GPO_(TPS_(x)).conj() / 2**n, range(4**n))),
        (4**n, 4**n))


# Get operator decomposition
def DecomposeOperator(gate: np.array,
                      atol: float = 1e-8,
                      dtype: np.dtype = 'float32'):

    # Gate must be a quadratic matrix of size 2^n
    if gate.ndim != 2 or gate.shape[0] != gate.shape[1] or int(
            np.log2(gate.shape[0])) != np.log2(gate.shape[0]):
        raise ValueError("'gate' is not a valid")

    # Get number of qubits
    n_ = int(np.log2(gate.shape[0]))

    # Generate linear system
    LS_ = GenerateLinearSystem(n=n_, dtype=GetComplexType(dtype))

    # Specialize functions
    TPS_ = fts.partial(ToPauliString, n=n_)
    GPO_ = fts.partial(GetPauliOperator, dtype=GetComplexType(dtype))

    # Get decomposition
    gate_ = np.real_if_close(LS_ @ np.array(
        list(
            map(lambda op: (gate @ op @ gate.conj().T).ravel(),
                map(GPO_, map(TPS_, range(4**n_)))))).T).T

    # Find values which are above the given threshold
    pos_ = list(map(lambda x: np.where(np.abs(x) > atol)[0], gate_))

    # Extract
    dec_ = list(map(lambda g, p: g[p], gate_, pos_))

    # Sort so that first state is always the largest
    dec_, pos_ = zip(
        *((d_[s_], p_[s_]) for d_, p_, s_ in ((d_, p_,
                                               np.argsort(np.abs(d_))[::-1])
                                              for d_, p_ in zip(dec_, pos_))))

    # Return
    return dec_, pos_


def simulate(circuit: list[tuple[U, qubits]],
             paulis: str | dict[str, float] = None,
             branches=None,
             *,
             parallel: bool | int = True,
             norm_atol: float = 1e-8,
             atol: float = 1e-8,
             dec_atol: float = 1e-8,
             log2_n_buckets: int = 12,
             verbose: bool = False,
             core_: str = None,
             **kwargs):
    import importlib

    # Load core
    core_ = importlib.import_module(
        'hybridq_clifford_core' if core_ is None else core_)

    # Load object
    Branch, StateFromPauli, Simulator = map(
        lambda x: getattr(core_, x), ['Branch', 'StateFromPauli', 'Simulator'])

    # Provide either 'paulis' or 'branches', but not both
    if not ((paulis is not None) ^ (branches is not None)):
        raise ValueError("Provide either 'paulis' or 'branches', but not both")

    # Convert parallel to number of threads
    n_threads_ = 1 if not parallel else 0 if isinstance(parallel,
                                                        bool) else int(parallel)

    # Convert paulis
    if isinstance(paulis, str):
        paulis = {paulis: 1}

    # Initialize branches
    branches_ = [
        Branch(StateFromPauli(p_), ph_, ph_, 0) for p_, ph_ in paulis.items()
    ]

    # Decompose circuit
    phases_, positions_, qubits_ = zip(
        *map(lambda x: (*DecomposeOperator(x[0]), x[1]),
             tqdm(circuit, disable=not verbose, desc="Decomposing")))

    # Get number of qubits
    n_qubits_ = max(mit.flatten(qubits_)) + 1

    # Check that all paulis have the right number of qubits
    if paulis is not None:
        if any(map(lambda p: len(p) < n_qubits_, paulis)):
            raise ValueError("Number of qubits in 'paulis' is not "
                             "consistent with circuit")

    # Get simulator
    sim_ = Simulator(phases_,
                     positions_,
                     qubits_,
                     atol=atol,
                     norm_atol=norm_atol,
                     n_threads=n_threads_,
                     log2_n_buckets=log2_n_buckets,
                     verbose=verbose,
                     **kwargs)

    # Get initial time
    init_time_ = time.time()

    def print_info_(mem_used_gb, *, end='\r', n=150):
        # Get time
        time_ = time.time()

        # Collect info
        n_explored_branches_ = sum(x_.n_explored_branches for x_ in sim_.infos)
        n_remaining_branches_ = sum(
            x_.n_remaining_branches for x_ in sim_.infos)
        n_completed_branches_ = sum(
            x_.n_completed_branches for x_ in sim_.infos)
        branching_time_ = (time_ - init_time_) / n_explored_branches_ * 1e6

        # Build message
        msg_ = f"NT={sim_.n_threads}, "
        msg_ += f"EB={n_explored_branches_:,}, "
        msg_ += f"RB={n_remaining_branches_:,}, "
        msg_ += f"CB={n_completed_branches_:,} "
        msg_ += f"(BT={branching_time_:1.1f}μs, "
        msg_ += f"ET={timedelta(seconds=round(time_ - init_time_))}, "
        msg_ += f"LV={psutil.cpu_percent(interval=None)}%, "
        msg_ += f"FM={psutil.virtual_memory().available / psutil.virtual_memory().total * 100:1.2f}%, "
        msg_ += f"UM={mem_used_gb:1.2f}GB)"

        # Pad message
        if len(msg_) > n:
            msg_ = msg_[:n - 3] + '...'
        else:
            msg_ += ' ' * (n - len(msg_))

        # Dump message
        print(msg_, end=end, flush=True, file=sys.stderr)

    def get_mem_gb_():
        return psutil.Process(os.getpid()).memory_info().rss / 2**30

    # Initialize memory peak
    mem_peak_gb_ = 0

    # Start simulation
    sim_.start(branches_)
    while (not sim_.ready(1000)):
        # Update memory used
        mem_used_gb_ = get_mem_gb_()

        # Save peak
        if mem_used_gb_ > mem_peak_gb_:
            mem_peak_gb_ = mem_used_gb_

        # Print info
        if verbose:
            print_info_(mem_used_gb_)

    # Cleanup output
    if verbose:
        print_info_(get_mem_gb_(), end='\n')

    # Join to get results
    sim_info_, partial_branches_, *rest_ = sim_.join()

    # Partial branches should be empty
    assert (not len(partial_branches_))

    # Convert to dict
    sim_info_ = sim_info_.dict()

    # Update output
    sim_info_['mem_peak_gb'] = mem_peak_gb_
    sim_info_['n_branches/us'] = sim_info_['n_explored_branches'] / (
        sim_info_['runtime_ms'] * 1e3)

    # Return results
    return sim_info_, *rest_
