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
        # Get alphabet
        from string import ascii_letters

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
        state = ''.join(_l2[_p2.index(i) // 2] if i in _p2 else x
                        for i, x in enumerate(state))

        # Add 4-qubit trace
        _p4 = np.random.choice(pos, size=8, replace=False).tolist()
        _l4 = np.random.choice(let, size=len(_p4) // 4, replace=False)
        pos = np.setdiff1d(pos, _p4)
        let = np.setdiff1d(let, _l4)
        state = ''.join(_l4[_p4.index(i) // 4] if i in _p4 else x
                        for i, x in enumerate(state))

        # Split as initial/final state
        initial_state = state[:n_qubits]
        final_state = state[n_qubits:]

        # Get random circuit
        circuit = get_rqc(n_qubits, depth, use_random_indexes=True)

    else:

        initial_state = final_state = circuit = None

    # Broadcast
    circuit, initial_state, final_state = _mpi_comm.bcast(
        (circuit, initial_state, final_state), root=0)

    # Get states using tensor contraction
    res_tn = simulation.simulate(circuit,
                                 optimize='tn',
                                 initial_state=initial_state,
                                 final_state=final_state,
                                 max_iterations=4,
                                 max_n_slices=2**11,
                                 max_largest_intermediate=2**8,
                                 parallel=True,
                                 verbose=True)

    # Check
    if _mpi_rank == 0:

        # Check shape of tensor is consistent with open qubits
        assert (len(res_tn.shape) == state.count('.'))

        # Get matrix of the circuit
        U = utils.matrix(circuit, verbose=True)

        # Reshape and traspose matrix to be consistent with tensor
        U = np.transpose(
            np.reshape(U, (2,) * 2 * n_qubits),
            list(range(n_qubits, 2 * n_qubits)) + list(range(n_qubits)))

        # Properly order qubits in U
        order = [x for x, s in enumerate(state) if s in '01'] + [
            x for x, s in enumerate(state) if s == '.'
        ] + _p4[::4] + _p4[1::4] + _p4[2::4] + _p4[3::4] + _p2[::2] + _p2[
            1::2] + _p1
        U = np.transpose(U, order)

        # Get number of projected qubits
        n_proj = sum(s in '01' for s in state)

        # Get number of open qubits
        n_open = sum(s == '.' for s in state)

        # Get number of 1-qubit and 2-qubits trace
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
        U = np.einsum('...iiii',
                      np.reshape(U, (2**n_open,) + (2**(n4 // 4),) * 4))

        # Check that the tensor match the transformed matrix
        assert (np.allclose(U.flatten(), res_tn.flatten(), atol=1e-5))
