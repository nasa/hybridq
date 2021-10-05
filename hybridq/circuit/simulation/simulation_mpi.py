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

MPI implementation of hybridq.circuit.simulation.simulate.

See Also
--------
hybridq.circuit.simulate
    Simulate quantum circuit.
"""

from __future__ import annotations
from os import cpu_count
from mpi4py import MPI
from multiprocessing import Pool
from warnings import warn
from hybridq.gate import Gate
from hybridq.circuit import Circuit
from typing import TypeVar
from tqdm.auto import tqdm
from opt_einsum.contract import PathInfo
from opt_einsum import contract, get_symbol
from sys import stderr
from time import time, sleep
import numpy as np
import hybridq.circuit.utils as utils
from hybridq.utils import sort, argsort

# Define types
TensorNetwork = TypeVar('TensorNetwork')


def _simulate_tn_mpi(circuit: Circuit, initial_state: any, final_state: any,
                     optimize: any, backend: any, complex_type: any,
                     tensor_only: bool, verbose: bool, **kwargs):
    import quimb.tensor as tn
    import cotengra as ctg

    # Get MPI
    _mpi_comm = MPI.COMM_WORLD
    _mpi_size = _mpi_comm.Get_size()
    _mpi_rank = _mpi_comm.Get_rank()

    # Set default parameters
    kwargs.setdefault('compress', 2)
    kwargs.setdefault('simplify_tn', 'RC')
    kwargs.setdefault('max_iterations', 1)
    kwargs.setdefault('methods', ['kahypar', 'greedy'])
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

    # Get random leaves_prefix
    leaves_prefix = ''.join(
        np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), size=20))

    # Initialize info
    _sim_info = {}

    # Alias for tn
    if optimize == 'tn':
        optimize = 'cotengra'

    if isinstance(circuit, Circuit):

        if not kwargs['parallel']:
            kwargs['parallel'] = 1
        else:
            # If number of threads not provided, just use half of the number of available cpus
            if isinstance(kwargs['parallel'],
                          bool) and kwargs['parallel'] == True:
                kwargs['parallel'] = cpu_count() // 2

        if optimize is not None and kwargs['parallel'] and kwargs[
                'max_iterations'] == 1:
            warn("Parallelization for MPI works for multiple iterations only. "
                 "For a better performance, use: 'max_iterations' > 1")

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
                from warnings import warn

                # Define new DeprecationWarning (to always print the warning
                # signal)
                class DeprecationWarning(Warning):
                    pass

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

            # Set cotengra parameters
            cotengra_params = lambda: ctg.HyperOptimizer(
                methods=kwargs['methods'],
                max_time=kwargs['max_time'],
                max_repeats=kwargs['max_repeats'],
                minimize=kwargs['minimize'],
                optlib=kwargs['optlib'],
                sampler=kwargs['sampler'],
                progbar=False,
                parallel=False,
                **kwargs['cotengra'])

            # Get target size
            tli = kwargs['target_largest_intermediate']

            with Pool(kwargs['parallel']) as pool:

                # Sumbit jobs
                _opts = [
                    cotengra_params() for _ in range(kwargs['max_iterations'])
                ]
                _map = [
                    pool.apply_async(tensor.contract, (all,),
                                     dict(optimize=_opt, get='path-info'))
                    for _opt in _opts
                ]

                with tqdm(total=len(_map),
                          disable=not verbose,
                          desc='Collecting contractions') as pbar:

                    _old_completed = 0
                    while 1:

                        # Count number of completed
                        _completed = 0
                        for _w in _map:
                            _completed += _w.ready()
                            if _w.ready() and not _w.successful():
                                _w.get()

                        # Update pbar
                        pbar.update(_completed - _old_completed)
                        _old_completed = _completed

                        if _completed == len(_map):
                            break

                        # Wait
                        sleep(1)

                # Collect results
                _infos = [_w.get() for _w in _map]

            if kwargs['minimize'] == 'size':
                opt, info = sort(
                    zip(_opts, _infos),
                    key=lambda w:
                    (w[1].largest_intermediate, w[0].best['flops']))[0]
            else:
                opt, info = sort(
                    zip(_opts, _infos),
                    key=lambda w:
                    (w[0].best['flops'], w[1].largest_intermediate))[0]

        if optimize == 'cotengra':

            # Gather best contractions
            _cost = _mpi_comm.gather(
                (info.largest_intermediate, info.opt_cost, _mpi_rank), root=0)
            if _mpi_rank == 0:
                if kwargs['minimize'] == 'size':
                    _best_rank = sort(_cost, key=lambda x: (x[0], x[1]))[0][-1]
                else:
                    _best_rank = sort(_cost, key=lambda x: (x[1], x[0]))[0][-1]
            else:
                _best_rank = None
            _best_rank = _mpi_comm.bcast(_best_rank, root=0)

            if hasattr(opt, '_pool'):
                del (opt._pool)

            # Distribute opt/info
            tensor, info, opt = _mpi_comm.bcast((tensor, info, opt),
                                                root=_best_rank)

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
    if verbose and _mpi_rank == 0:
        print(
            f'Largest Intermediate: 2^{np.log2(float(info.largest_intermediate)):1.2f}',
            file=stderr)
        print(
            f'Max Largest Intermediate: 2^{np.log2(float(kwargs["max_largest_intermediate"])):1.2f}',
            file=stderr)
        print(f'Flops: 2^{np.log2(float(info.opt_cost)):1.2f}', file=stderr)

    if optimize == 'cotengra':

        if _mpi_rank == 0:

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
                                 target_size=kwargs['max_largest_intermediate'],
                                 allow_outer=False)

            # Find slices
            with tqdm(kwargs['temperatures'], disable=not verbose,
                      leave=False) as pbar:
                for _temp in pbar:
                    pbar.set_description(f'Find slices (T={_temp})')
                    ix_sl, cost_sl = sf.search(temperature=_temp)

            # Get slice contractor
            sc = sf.SlicedContractor([t.data for t in tensor])

            # Make sure that no open qubits are sliced
            assert (not {
                ix: i for i, ix in enumerate(sc.output) if ix in sc.sliced
            })

            # Print some infos
            if verbose:
                print(
                    f'Number of slices: 2^{np.log2(float(cost_sl.nslices)):1.2f}',
                    file=stderr)
                print(
                    f'Flops+Cuts: 2^{np.log2(float(cost_sl.total_flops)):1.2f}',
                    file=stderr)

            # Update infos
            _sim_info.update({
                'flops': info.opt_cost,
                'largest_intermediate': info.largest_intermediate,
                'n_slices': cost_sl.nslices,
                'total_flops': cost_sl.total_flops
            })

            # Get slices
            slices = list(range(cost_sl.nslices + 1)) + [None] * (
                _mpi_size -
                cost_sl.nslices) if cost_sl.nslices < _mpi_size else [
                    cost_sl.nslices / _mpi_size * i for i in range(_mpi_size)
                ] + [cost_sl.nslices]
            if not np.alltrue([
                    int(x) == x for x in slices if x is not None
            ]) or not np.alltrue([
                    slices[i] < slices[i + 1]
                    for i in range(_mpi_size)
                    if slices[i] is not None and slices[i + 1] is not None
            ]):
                raise RuntimeError('Something went wrong')

            # Convert all to integers
            slices = [int(x) if x is not None else None for x in slices]

        else:

            sc = slices = None

        # Distribute slicer and slices
        sc, slices = _mpi_comm.bcast((sc, slices), root=0)

        _n_slices = max(x for x in slices if x)
        if kwargs['max_n_slices'] and _n_slices > kwargs['max_n_slices']:
            raise RuntimeError(
                f'Too many slices ({_n_slices} > {kwargs["max_n_slices"]})')

        # Contract slices
        _tensor = None
        if slices[_mpi_rank] is not None and slices[_mpi_rank + 1] is not None:
            for i in tqdm(range(slices[_mpi_rank], slices[_mpi_rank + 1]),
                          desc='Contracting slices',
                          disable=not verbose,
                          leave=False):
                if _tensor is None:
                    _tensor = np.copy(sc.contract_slice(i, backend=backend))
                else:
                    _tensor += sc.contract_slice(i, backend=backend)

        # Gather tensors
        if _mpi_rank != 0:
            _mpi_comm.send(_tensor, dest=0, tag=11)
        elif _mpi_rank == 0:
            for i in tqdm(range(1, _mpi_size),
                          desc='Collecting tensors',
                          disable=not verbose):
                _p_tensor = _mpi_comm.recv(source=i, tag=11)
                if _p_tensor is not None:
                    _tensor += _p_tensor

        if _mpi_rank == 0:

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
            #        tensor = np.reshape(tensor,
            #                            (2**len(_i_inds), 2**len(_f_inds)))
            #    else:
            #        tensor = np.reshape(tensor,
            #                            (2**max(len(_i_inds), len(_f_inds)),))

        else:

            tensor = None

    else:

        if _mpi_rank == 0:

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
                #    tensor = np.reshape(tensor,
                #                        (2**len(_i_inds), 2**len(_f_inds)))
                #else:
                #    tensor = np.reshape(tensor,
                #                        (2**max(len(_i_inds), len(_f_inds)),))

        else:

            tensor = None

    if kwargs['return_info']:
        return tensor, _sim_info
    else:
        return tensor
