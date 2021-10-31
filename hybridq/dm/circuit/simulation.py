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
from hybridq.dm.circuit import Circuit as SuperCircuit
from hybridq.circuit import Circuit
import numpy as np


def __transform(gate):
    """
    Convert SuperCircuit -> Circuit
    """
    from hybridq.dm.gate import BaseSuperGate
    from hybridq.gate import BaseGate
    from hybridq.gate import MatrixGate

    # If gate is a BaseSuperGate
    if isinstance(gate, BaseSuperGate):
        # Get left and right qubits
        l_qubits, r_qubits = (gate.qubits, gate.qubits) if isinstance(
            gate, BaseGate) else gate.qubits

        # Return a MatrixGate which uses map as Matrix and left/right
        # qubits as qubits
        return MatrixGate(gate.map(), [(0, q) for q in l_qubits] +
                          [(1, q) for q in r_qubits])

    # If gate is a regular BaseGate
    elif isinstance(gate, BaseGate):
        # Apply the gate to both left and right qubits
        return (gate.on((0, q) for q in gate.qubits), gate.conj().on(
            (1, q) for q in gate.qubits))

    # Only BaseGate's and BaseSuperGate's are supported
    else:
        raise TypeError(f"{type(gate).__name__} not supported.")


def __convert(circuit: iter,
              parallel: {bool, int} = False,
              verbose: bool = False) -> Circuit:
    from tqdm.auto import tqdm

    # Get lenght
    try:
        total = len(circuit)
    except:
        total = None

    # Flatten circuit
    circuit = tuple(
        g for w in circuit for g in (w if isinstance(w, tuple) else [w]))

    # Parallelize if requested
    if parallel:
        from hybridq.utils import isintegral
        from multiprocessing import Pool
        from time import sleep

        # Get number of parallel threads
        if isinstance(parallel, bool):
            from os import cpu_count
            parallel = cpu_count()
        elif isintegral(parallel) and parallel > 0:
            parallel = int(parallel)
        else:
            raise ValueError("'parallel' must be a positive integer")

        with Pool(parallel) as pool, tqdm(total=len(circuit),
                                          desc='Converting circuit',
                                          disable=not verbose) as pbar:
            # Get map
            _map = [pool.apply_async(__transform, (g,)) for g in circuit]

            # Wait till ready
            while 1:
                # Count number of complete
                c = sum(m.ready() for m in _map)

                # Update pbar
                pbar.n = c
                pbar.update()

                # Break if all complete
                if c == len(_map):
                    break

                # Sleep for a while
                sleep(0.1)

            # Return circuit
            return Circuit(m.get() for m in _map)

    # Otherwise, transform one by one
    else:
        return Circuit(
            tqdm(map(__transform, circuit),
                 total=total,
                 desc='Converting circuit',
                 disable=not verbose))


def simulate(circuit: SuperCircuit,
             initial_state: any,
             final_state: any = None,
             optimize: any = 'evolution',
             parallel: {bool, int} = False,
             verbose: bool = False,
             **kwargs):
    """
    Frontend to simulate `rho` using different optimization models and
    backends.

    Parameters
    ----------
    circuit: Circuit
        Circuit to simulate.
    initial_state: {str, Circuit, array_like}
        Initial density matrix to evolve.
    final_state: {str, Circuit, array_like}
        Final density matrix to project to.
    optimize: any
        Method to use to perform the simulation. The available methods are:
        - `evolution`: Evolve the density matrix using state vector evolution
        - `tn`: Evolve the density matrix using tensor contraction
        - `clifford`: Evolve the density matrix using Clifford expansion

    See Also
    --------
    `hybridq.circuit.simulation` and `hybridq.circuit.simulation.clifford`
    """
    from hybridq.circuit import Circuit

    # Convert circuit to list
    circuit = list(circuit)

    if optimize == 'clifford':
        from hybridq.circuit.simulation.clifford import update_pauli_string
        from hybridq.gate import BaseGate

        # Check that all gates are BaseGate
        if any(not isinstance(gate, BaseGate) for gate in circuit):
            raise NotImplementedError(
                "'optimize=clifford' only supports 'BaseGate's")

        # Final state cannot be provided if optimize='clifford'
        if final_state is not None:
            raise ValueError(
                "'final_state' cannot be provided if optimize='clifford'.")

        # Get circuit
        circuit = Circuit(circuit)

        # Try to convert to circuit
        try:
            initial_state = Circuit(initial_state)
        except:
            pass

        # Return Clifford expandion
        return update_pauli_string(circuit=circuit,
                                   pauli_string=initial_state,
                                   parallel=parallel,
                                   verbose=verbose,
                                   **kwargs)

    else:
        from hybridq.circuit.simulation import simulate
        from hybridq.utils import sort

        # Convert to SuperCircuit
        circuit = SuperCircuit(circuit)

        # Get left/right qubits
        l_qubits, r_qubits = circuit.all_qubits()

        # Convert SuperCircuit to Circuit
        circuit = __convert(circuit, parallel=parallel, verbose=verbose)

        # Get number of qubits
        nl = len(l_qubits)
        nr = len(r_qubits)

        # Check states
        def _get_state(state, name):
            # If None, return None
            if state is None:
                return None

            # Check if string
            elif isinstance(state, str):
                # If single char, extend to full size
                state = state * (nl + nr) if len(state) == 1 else state

                # Check that state has the right number of chars
                if not (len(state) == (nl + nr) or
                        (l_qubits == r_qubits and len(state) == nl)):
                    raise ValueError(
                        f"'{name}' has the wrong number of qubits.")

                # Extend if needed
                state = state + state if len(state) == nl else state

                # Return
                return state

            # Check if Circuit
            elif isinstance(state, Circuit):
                from hybridq.circuit.utils import matrix

                # Check that qubits are consistent
                if l_qubits != r_qubits or sort(l_qubits) != sort(
                        state.all_qubits()):
                    raise ValueError(
                        f"Qubits in '{name}' are not consistent with 'circuit'."
                    )

                # Get matrix
                U = matrix(state, order=l_qubits)

                # Swap input/output
                return np.transpose(np.reshape(U, (2,) * 2 * nl),
                                    list(range(nl, 2 * nl)) + list(range(nl)))

            else:
                # Try to convert to numpy array
                state = np.asarray(state)

                # At the moment, only 2-dimensional qubits are allowed
                if set(state.shape) != {2}:
                    raise NotImplementedError(
                        "Only 2-dimensional qubits are allowed.")

                # Check if the number of dimensions matches
                if not (state.ndim == (nl + nr) or
                        (l_qubits == r_qubits and state.ndim == nl)):
                    raise ValueError(
                        f"'{name}' has the wrong number of qubits.")

                # Extend if needed
                if state.ndim == nl:
                    state = np.reshape(np.kron(state.ravel(), state.ravel()),
                                       (2,) * 2 * nl)

                # Return state
                return state

        # Get states
        initial_state = _get_state(initial_state, 'initial_state')
        final_state = _get_state(final_state, 'final_state')

        # Return results
        return simulate(circuit=circuit,
                        initial_state=initial_state,
                        final_state=final_state,
                        optimize=optimize,
                        parallel=parallel,
                        verbose=verbose,
                        **kwargs)
