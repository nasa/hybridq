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

from hybridq.circuit.simulation import clifford
from hybridq.extras.random import get_rqc
from hybridq.circuit import Circuit
from hybridq.gate import Gate
import numpy as np

if __name__ == '__main__':

    from mpi4py import MPI
    _mpi_comm = MPI.COMM_WORLD
    _mpi_size = _mpi_comm.Get_size()
    _mpi_rank = _mpi_comm.Get_rank()

    # Generate random circuit
    if _mpi_rank == 0:

        # Get random circuit
        c = get_rqc(20, 40, use_random_indexes=True)

        # Get random observable
        b = Circuit(
            Gate(np.random.choice(list('XYZ')), [q])
            for q in c.all_qubits()[:2])

    else:
        c = b = None

    # Broadcast circuits
    c = _mpi_comm.bcast(c, root=0)
    b = _mpi_comm.bcast(b, root=0)

    # Compute expectation value with mpi
    res1 = clifford.expectation_value(
        c,
        b,
        initial_state='+' * len(c.all_qubits()),
        return_info=True,
        parallel=True,
        verbose=True,
    )

    # Compute expectation value without mpi
    if _mpi_rank == 0:
        res2 = clifford.expectation_value(
            c,
            b,
            initial_state='+' * len(c.all_qubits()),
            return_info=True,
            verbose=True,
            parallel=True,
            use_mpi=False,
        )

    # Compare results
    if _mpi_rank == 0:
        assert (np.isclose(res1[0], res2[0], atol=1e-5))
