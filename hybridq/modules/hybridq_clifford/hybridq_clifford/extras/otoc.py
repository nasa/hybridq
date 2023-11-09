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
from ..simulation import simulate as simulate_


def simulate(circ_U: list[tuple[U, qubits]],
             initial_state: str,
             butterfly: str | dict[int, str],
             target_position: int,
             log10_dnorm_atol: float = 0.1,
             **kwargs):
    """
    Compute OTOC value for the operator `circ_U`.

    See also
    --------
    hybridq_clifford.simulation.simulate
    """

    # Check if initial state is valid
    if set(initial_state).difference('01+-'):
        raise ValueError(
            "'initial_state' must be a string containing '01+-' only")

    # If butterfly is dict, build string
    if isinstance(butterfly, dict):
        # Check butterfly
        if any(not 0 <= x_ < len(initial_state)
               for x_ in butterfly.keys()) or any(
                   x_.upper() not in 'IXYZ' for x_ in butterfly.values()):
            raise ValueError("'butterfly' is not valid")
        butterfly = ''.join(
            butterfly.get(x_, 'I') for x_ in range(len(initial_state)))

    # Check if butterfly is valid
    if set(butterfly.upper()).difference('IXYZ'):
        raise ValueError("'butterfly' must be a string containing 'IXYZ' only")
    if len(butterfly) != len(initial_state):
        raise ValueError("'butterfly' has the wrong number of qubits")

    # Check target position
    if not 0 <= target_position < len(initial_state):
        raise ValueError("'target_position' is not valid")

    # Run simulation
    return simulate_(circuit=[(U_.conj().T, qs_) for U_, qs_ in reversed(circ_U)
                             ],
                     paulis=butterfly.upper(),
                     initial_state=['01+-'.index(x_) for x_ in initial_state],
                     target_position=target_position,
                     log10_dnorm_atol=log10_dnorm_atol,
                     core_='hybridq_clifford.core.extras.otoc',
                     **kwargs)
