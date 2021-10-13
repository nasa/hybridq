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
from hybridq.circuit import Circuit
from hybridq.noise.channel.channel import BaseChannel, GlobalDepolarizingChannel
from hybridq.dm.circuit import Circuit as NoiseCircuit


def is_dm(rho: np.ndarray, atol=1e-6) -> bool:
    """
    check if the given input a valid density matrix.
    """
    rho = np.asarray(rho)
    d = int(np.sqrt(np.prod(rho.shape)))
    rho_full = np.reshape(rho, (d, d))

    hc = np.allclose(rho_full, rho_full.T.conj(), atol=atol)
    tp = np.isclose(np.trace(rho_full), 1, atol=atol)

    apprx_gtr = lambda y, x: np.real(y) >= x or np.isclose(y, x, atol=atol)
    ev = np.linalg.eigvals(rho_full)
    psd = np.all([apprx_gtr(e, 0) for e in ev])

    return (hc and tp and psd)


def is_channel():
    raise NotImplementedError("working on this")


def add_depolarizing_noise(c: Circuit, probs=(0., 0.)):
    """
    Given a `Circuit`, add depolarizing noise after each instance of
    a `BaseGate`, with the same locality as the gate.

    Noise will not be added after a `BaseChannel`

    probs: tuple[float, ...]
        depolarizing probabilities as a tuple, where the k'th value corresponds
        to the probability of depolarizing for a k-local gate.
        probs should be the size of the largest locality `Gate` in the circuit.
    """
    c2 = NoiseCircuit()
    for g in c:
        c2 += [g]
        if isinstance(c2, BaseChannel):
            continue
        c2 += [GlobalDepolarizingChannel(g.qubits, probs[len(g.qubits)-1])]
    return c2




