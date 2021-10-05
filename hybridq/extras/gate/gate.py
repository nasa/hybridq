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
from hybridq.gate.property import generate, staticvars, compare
import hybridq.gate.property as pr
from hybridq.gate import BaseGate
import sys


def _MessageGateApply(self, psi, order, *args, **kwargs):
    print(self.message, file=self.file)
    return psi, order


@staticvars('message,file', check=dict(message=lambda m: isinstance(m, str)))
@compare('message,file')
class MessageGate(pr.FunctionalGate,
                  pr.TagGate,
                  pr.NameGate,
                  n_qubits=any,
                  apply=_MessageGateApply,
                  name='MESSAGE'):

    def __init_subclass__(cls, **kwargs):
        # Get name
        name = cls.__get_staticvar__('name')

        # Get apply
        apply = cls.__get_staticvar__('apply')

        # Continue
        super().__init_subclass__(name=name, apply=apply, **kwargs)

    def __print__(self):
        return {
            'message': (400, f"message='{self.message}'", 0),
            'file':
                (401,
                 f"file={self.file}" if self.file is not sys.stdout else "", 0)
        }


_gate_aliases = {'MSG': 'MESSAGE'}


def Gate(name: str,
         message: str,
         qubits: iter[any] = None,
         file: any = sys.stdout):

    # Convert to upper string
    name = str(name).upper()

    # Check if name is an alias
    name = _gate_aliases.get(name, name)

    # Get the correct gate
    if name == 'MESSAGE':

        # Convert message to str
        try:
            message = str(message)
        except:
            raise ValueError("'message' must be convertible to 'str'.")

        # Convert qubits to tuple
        if qubits is not None:
            try:
                qubits = tuple(qubits)
            except:
                raise ValueError("'qubits' must be convertible to 'tuple'.")

        # Get n_qubits
        n_qubits = 0 if qubits is None else len(qubits)

        return generate(class_name='MessageGate',
                        mro=(BaseGate, MessageGate),
                        n_qubits=n_qubits,
                        message=message,
                        file=file)(qubits=qubits)

    else:

        raise ValueError(f"Gate('{name}') not supported.")
