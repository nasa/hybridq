"""
Authors: Salvatore Mandra (salvatore.mandra@nasa.gov),
         Jeffrey Marshall (jeffrey.s.marshall@nasa.gov)

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
import numpy as np
from hybridq.noise.channel.channel import BaseChannel, GlobalDepolarizingChannel
from hybridq.dm.circuit import Circuit as SuperCircuit


def add_depolarizing_noise(c: Circuit, probs=(0., 0.)):
    """
    Given a `Circuit`, add depolarizing noise after each instance of
    a `Gate`, with the same locality as the gate.
    Note, noise will not be added after an instance of `BaseChannel`

    c: Circuit
        The `Circuit` which will be modified. Note, a new `Circuit` is
        returned (this is not in place).
    probs: tuple[float, ...]
        depolarizing probabilities as a tuple, where the k'th value corresponds
        to the probability of depolarizing for a k-local gate.
        `probs` should be the size of the largest locality `Gate`
        in the circuit.
    """
    c2 = SuperCircuit()
    for g in c:
        c2 += [g]
        if isinstance(g, BaseChannel):
            continue
        k = len(g.qubits)  # locality
        if k > len(probs):
            raise ValueError("`probs` does not have sufficient "
                             f"entries for {k}-local Gate")
        c2 += [GlobalDepolarizingChannel(g.qubits, probs[k - 1])]
    return c2
