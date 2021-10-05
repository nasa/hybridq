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

from hybridq.circuit.simulation import simulate
from hybridq.extras.random import get_rqc
import hybridq.circuit.utils as utils
from hybridq.circuit import Circuit
import numpy as np

if __name__ == '__main__':

    # Set number of qubits
    n_qubits = 23

    # Set number of gates
    n_gates = 2000

    # Generate RQC
    print('# Generate RQC: ', end='')
    circuit = get_rqc(n_qubits, n_gates, use_random_indexes=True)
    print('Done!')

    # Compress circuit
    print('# Compress RQC: ', end='')
    circuit = Circuit(
        utils.to_matrix_gate(c) for c in utils.compress(circuit, 2))
    print('Done!')

    # Get final state
    print('# Compute quantum evolution: ', end='')
    psi = simulate(circuit,
                   optimize='evolution',
                   initial_state='0' * n_qubits,
                   verbose=True)
    print('Done!')

    # Print final state
    for i in np.random.choice(2**n_qubits, size=20, replace=False):
        x = bin(i)[2:].zfill(n_qubits)
        print(f'|{x}> = {psi[tuple(map(int, x))]:+1.5f}')
    print('...')
