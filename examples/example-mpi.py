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

from opt_einsum import contract, get_symbol
from hybridq.extras.random import get_rqc
from hybridq.circuit import simulation
import hybridq.circuit.utils as utils
from hybridq.circuit import Circuit
from mpi4py import MPI
import numpy as np

if __name__ == '__main__':

    # Get MPI info
    _mpi_comm = MPI.COMM_WORLD
    _mpi_size = _mpi_comm.Get_size()
    _mpi_rank = _mpi_comm.Get_rank()

    # Set number of qubits
    n_qubits = 12

    # Set depth
    depth = 100

    # Get circuit and initial/final states
    if _mpi_rank == 0:

        # Get random initial_state
        initial_state = ''.join(np.random.choice(list('01'), size=3)) + ''.join(
            np.random.choice(list('01+-'), size=n_qubits - 3))

        # Specify some output qubits
        final_state = np.random.choice(list('01'), size=n_qubits)
        final_state[np.random.choice(n_qubits,
                                     size=int(n_qubits / 2),
                                     replace=False)] = '.'
        final_state = ''.join(final_state)

        # Get random circuit
        circuit = get_rqc(n_qubits, depth, use_random_indexes=True)

    else:

        initial_state = final_state = circuit = None

    # Broadcast
    circuit, initial_state, final_state = _mpi_comm.bcast(
        (circuit, initial_state, final_state), root=0)

    # Get state using matrix
    _states_using_matrix = np.reshape(
        utils.matrix(circuit, verbose=True)
        @ simulation.prepare_state(initial_state).flatten(), (2,) * n_qubits)

    # Get states using time evolution
    _states_evolution = simulation.simulate(circuit,
                                            optimize='evolution',
                                            initial_state=initial_state,
                                            verbose=True)

    # Get states using time evolution
    _states_evolution_einsum = simulation.simulate(circuit,
                                                   optimize='evolution-einsum',
                                                   initial_state=initial_state,
                                                   verbose=True)

    # Get states using tensor contraction
    _states_tn_1 = simulation.simulate(circuit,
                                       optimize='tn',
                                       initial_state='...' + initial_state[3:],
                                       final_state=final_state,
                                       max_iterations=4,
                                       max_n_slices=2**11,
                                       parallel=True,
                                       verbose=True)

    # Reduce maximum largest intermediate
    _states_tn_2_tensor, (_states_tn_2_info,
                          _states_tn_2_opt) = simulation.simulate(
                              circuit,
                              optimize='tn',
                              initial_state='...' + initial_state[3:],
                              final_state=final_state,
                              max_iterations=4,
                              parallel=True,
                              tensor_only=True,
                              use_mpi=False,
                              verbose=True)
    _states_tn_2 = simulation.simulate(_states_tn_2_tensor,
                                       optimize=(_states_tn_2_info,
                                                 _states_tn_2_opt),
                                       max_largest_intermediate=2**10,
                                       max_n_slices=2**11,
                                       verbose=True)

    # Broadcast
    _states_tn_1, _states_tn_2 = _mpi_comm.bcast((_states_tn_1, _states_tn_2),
                                                 root=0)

    # Compare with exact
    _xpos = [x for x, s in enumerate(final_state) if s == '.']
    _map = ''.join([get_symbol(x) for x in range(n_qubits)])
    _map += '->'
    _map += ''.join(
        ['' if x in _xpos else get_symbol(x) for x in range(n_qubits)])
    _map += ''.join([get_symbol(x) for x in _xpos])
    _states_expected_tn = contract(
        _map, np.reshape(_states_evolution, [2] * n_qubits))
    _states_expected_tn = _states_expected_tn[tuple(
        map(int,
            final_state.replace('.', '').zfill(n_qubits - len(_xpos))))]

    assert (np.isclose(np.linalg.norm(_states_using_matrix.flatten()), 1))
    assert (np.isclose(np.linalg.norm(_states_evolution.flatten()), 1))
    assert (_states_tn_1.shape == (2,) * (3 + final_state.count('.')))
    assert (np.allclose(_states_using_matrix, _states_evolution, atol=1e-5))
    assert (np.allclose(_states_using_matrix,
                        _states_evolution_einsum,
                        atol=1e-5))
    assert (np.allclose(_states_tn_1[tuple(map(int, initial_state[:3]))],
                        _states_expected_tn,
                        atol=1e-5))
    assert (np.allclose(_states_tn_1, _states_tn_2, atol=1e-5))
