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
from hybridq.circuit import Circuit
from hybridq.gate import Gate
from tqdm.auto import tqdm
from warnings import warn
import numpy as np
import json
import re


def _isint(x: any) -> bool:
    """
    Check if x is convertible to integer.
    """
    try:
        int(x)
        return True
    except:
        return False


def _isnumber(x: any) -> bool:
    """
    Check if x is convertible to a number.
    """
    try:
        float(x)
        return True
    except:
        return False


def _isstring(x: any) -> bool:
    """
    Check if x is a string.
    """
    return isinstance(x, str)


def to_qasm(circuit: Circuit, qubits_map: dict[any, int] = None) -> str:
    """
    Convert a `Circuit` to QASM language.

    Parameters
    ----------
    circuit: Circuit
        `Circuit` to convert to QASM language.
    qubits_map: dict[any, int], optional
        If provided, `qubits_map` map qubit indexes in `Circuit` to qubit
        indexes in QASM. Otherwise, indexes are assigned to QASM qubits by using
        `Circuit.all_qubits()` order.

    Returns
    -------
    str
        String representing the QAMS circuit.

    Notes
    -----

    The QASM language used in HybridQ is compatible with the standard QASM.
    However, HybridQ introduces few extensions, which are recognized by the
    parser using ``#@`` at the beginning of the line (``#`` at the beginning of
    the line represent a general comment in QASM). At the moment, the following
    QAMS extensions are supported:

    * **qubits**, used to store `qubits_map`,
    * **power**, used to store the power of the gate,
    * **tags**, used to store the tags associated to the gate,
    * **U**, used to store the matrix reprentation of the gate if gate name is `MATRIX`

    If `Gate.qubits` are not specified, a single `.` is used to represent the
    missing qubits. If `Gate.params` are missing, parameters are just omitted.

    Example
    -------
    >>> from hybridq.extras.qasm import to_qasm
    >>> print(to_qasm(Circuit(Gate('H', [q]) for q in range(10))))
    10
    #@ qubits =
    #@ {
    #@   "0": "0",
    #@   "1": "1",
    #@   "2": "2",
    #@   "3": "3",
    #@   "4": "4",
    #@   "5": "5",
    #@   "6": "6",
    #@   "7": "7",
    #@   "8": "8",
    #@   "9": "9"
    #@ }
    h 0
    h 1
    h 2
    h 3
    h 4
    h 5
    h 6
    h 7
    h 8
    h 9
    >>> c = Circuit()
    >>> c.append(Gate('RX', tags={'params': False, 'qubits': False}))
    >>> c.append(Gate('RY', params=[1.23], tags={'params': True, 'qubits': False}))
    >>> c.append(Gate('RZ', qubits=[42], tags={'params': False, 'qubits': True})**1.23)
    >>> c.append(Gate('MATRIX', U=Gate('H').matrix()))
    >>> print(to_qasm(c))
    1
    #@ qubits =
    #@ {
    #@   "0": "42"
    #@ }
    #@ tags =
    #@ {
    #@   "params": false,
    #@   "qubits": false
    #@ }
    rx .
    #@ tags =
    #@ {
    #@   "params": true,
    #@   "qubits": false
    #@ }
    ry . 1.23
    #@ tags =
    #@ {
    #@   "params": false,
    #@   "qubits": true
    #@ }
    #@ power = 1.23
    rz 0
    #@ U =
    #@ [
    #@   [
    #@     "0.7071067811865475",
    #@     "0.7071067811865475"
    #@   ],
    #@   [
    #@     "0.7071067811865475",
    #@     "-0.7071067811865475"
    #@   ]
    #@ ]
    matrix .
    """

    # Initialize output
    _out = ''

    # Get qubits map
    if qubits_map is None:
        qubits_map = {
            q: x
            for x, q in enumerate(circuit.all_qubits(
                ignore_missing_qubits=True))
        }

    # Get inverse map
    inv_qubits_map = {x: str(q) for q, x in qubits_map.items()}

    # Dump number of qubits
    _out += f'{len(qubits_map)}\n'

    # Dump map
    _out += '#@ qubits = \n'
    _out += '\n'.join(
        ['#@ ' + x for x in json.dumps(inv_qubits_map, indent=2).split('\n')])
    _out += '\n'

    for gate in circuit:

        # Store matrix
        if gate.name == 'MATRIX':
            _out += '#@ U = \n'
            _out += '\n'.join(
                '#@ ' + x
                for x in json.dumps([[str(y) for y in x] for x in gate.Matrix],
                                    indent=2).split('\n'))
            _out += '\n'

        # Dump tags
        if gate.provides('tags') and gate.tags is not None:
            _out += '#@ tags = \n'
            _out += '\n'.join([
                '#@ ' + x for x in json.dumps(gate.tags, indent=2).split('\n')
            ])
            _out += '\n'

        # Dump power
        if gate.provides('power') and gate.power != 1:
            _out += f'#@ power = {gate.power}\n'

        # Dump conjugation
        if gate.provides('is_conjugated') and gate.is_conjugated():
            _out += f'#@ conj\n'

        # Dump transposition
        if gate.provides('is_transposed') and gate.is_transposed():
            _out += f'#@ T\n'

        # Dump name
        _out += gate.name.lower()

        # Dump gates
        if gate.provides('qubits') and gate.qubits is not None:
            _out += ' ' + ' '.join((str(qubits_map[x]) for x in gate.qubits))
        else:
            _out += ' .'

        # Dump params
        if gate.provides('params'):
            if gate.params is not None:
                _out += ' ' + ' '.join((str(x) for x in gate.params))
            else:
                raise ValueError('Not yet implemented.')

        # Add newline
        _out += '\n'

    return _out


def from_qasm(qasm_string: str) -> Circuit:
    """
    Convert a QASM circuit to `Circuit`.

    Parameters
    ----------
    qasm_string: str
        QASM circuit to convert to `Circuit`.

    Returns
    -------
    Circuit
        QAMS circuit converted to `Circuit`.

    Notes
    -----

    The QASM language used in HybridQ is compatible with the standard QASM.
    However, HybridQ introduces few extensions, which are recognized by the
    parser using ``#@`` at the beginning of the line (``#`` at the beginning of
    the line represent a general comment in QASM). At the moment, the following
    QAMS extensions are supported:

    * **qubits**, used to store `qubits_map`,
    * **power**, used to store the power of the gate,
    * **tags**, used to store the tags associated to the gate,
    * **U**, used to store the matrix reprentation of the gate if gate name is `MATRIX`

    If `Gate.qubits` are not specified, a single `.` is used to represent the
    missing qubits. If `Gate.params` are missing, parameters are just omitted.

    Example
    -------
    >>> from hybridq.extras.qasm import from_qasm
    >>> qasm_str = \"\"\"
    >>> 1
    >>> #@ qubits =
    >>> #@ {
    >>> #@   "0": "42"
    >>> #@ }
    >>> #@ tags =
    >>> #@ {
    >>> #@   "params": false,
    >>> #@   "qubits": false
    >>> #@ }
    >>> rx .
    >>> #@ tags =
    >>> #@ {
    >>> #@   "params": true,
    >>> #@   "qubits": false
    >>> #@ }
    >>> ry . 1.23
    >>> #@ tags =
    >>> #@ {
    >>> #@   "params": false,
    >>> #@   "qubits": true
    >>> #@ }
    >>> #@ power = 1.23
    >>> rz 0
    >>> #@ U =
    >>> #@ [
    >>> #@   [
    >>> #@     "0.7071067811865475",
    >>> #@     "0.7071067811865475"
    >>> #@   ],
    >>> #@   [
    >>> #@     "0.7071067811865475",
    >>> #@     "-0.7071067811865475"
    >>> #@   ]
    >>> #@ ]
    >>> matrix .
    >>> \"\"\"
    >>> from_qasm(qasm_str)
    Circuit([
            Gate(name=RX, tags={'params': False, 'qubits': False})
            Gate(name=RY, params=[1.23], tags={'params': True, 'qubits': False})
            Gate(name=RZ, qubits=[42], tags={'params': False, 'qubits': True})**1.23
            Gate(name=MATRIX, U=np.array(shape=(2, 2), dtype=float64))
    ])
    """

    # Initialize circuit
    circuit = Circuit()

    # Initialize tags
    _extra = None
    _power = None
    _conj = False
    _T = False
    _tags = None
    _qubits_map = None
    _U = None

    for line in (line for line in qasm_string.split('\n')
                 if line and (line[0] != '#' or line[:2] == '#@')):

        if line[:2] == '#@':

            # Strip line
            _line = re.sub(r'\s+', '', line)

            if '#@tags=' in _line:

                if _tags is not None:
                    raise ValueError('Format error.')

                # Initialize tags
                _tags = line.split('=')[-1]
                _extra = 'tags'

            elif '#@U=' in _line:

                if _U is not None:
                    raise ValueError('Format error.')

                # Initialize matrix
                _U = line.split('=')[-1]
                _extra = 'U'

            elif '#@power=' in _line:

                if _power is not None:
                    raise ValueError('Format error.')

                # Initialize power
                _power = line.split('=')[-1]
                _extra = 'power'

            elif '#@conj' in _line:

                if _conj is not False:
                    raise ValueError('Format error.')

                # Initialize conjugation
                _conj = True

            elif '#@T' in _line:

                if _T is not False:
                    raise ValueError('Format error.')

                # Initialize transposition
                _T = True

            elif '#@qubits=' in _line:

                if _qubits_map is not None:
                    raise ValueError('Format error.')

                # Initialize qubits
                _qubits_map = line.split('=')[-1]
                _extra = 'qubits'

            elif _extra:

                # Update tags
                if _extra == 'tags':
                    _tags += line.replace('#@', '')

                # Update matrix
                elif _extra == 'U':
                    _U += line.replace('#@', '')

                # Update power
                elif _extra == 'power':
                    _power += line.replace('#@', '')

                # Update qubits_map
                elif _extra == 'qubits':
                    _qubits_map += line.replace('#@', '')

                # Otherwise, error
                else:
                    raise ValueError('Format error.')

        else:

            # Restart _extra
            _extra = None

            # Strip everything after the first #
            line = line.split('#')[0].split()

            # Make few guesses about format
            if len(line) == 1:
                if _isint(line[0]):
                    warn(
                        f"Skipping '{' '.join(line)}' (most likely the number of qubits in the circuit)."
                    )
                    continue
                else:
                    warn(
                        f"Skipping '{' '.join(line)}' (format is not understood)."
                    )
                    continue

            # Make few guesses about format
            if _isint(line[0]):
                warn(
                    f"Skipping {line[0]} in '{' '.join(line)}' (most likely the circuit layer)."
                )
                del (line[0])

            # Make few guesses about format
            if not _isstring(line[0]):
                warn(f"Skipping '{' '.join(line)}' (format is not understood).")
                continue

            if line[0].upper() == 'MATRIX':

                # Remove name from line
                del (line[0])

                # Add tags
                if not _U:
                    raise ValueError('Format error.')

                # Set gate
                _U = np.real_if_close(
                    np.array([[complex(y) for y in x] for x in json.loads(_U)]))
                gate = Gate('MATRIX', U=_U)

                # Set qubits if present
                if line[0] != '.':

                    # Set qubits
                    gate.on([int(x) for x in line], inplace=True)

                # Reset
                _U = None

            else:

                # Set position
                _p = 0

                # Initialize gate with name
                gate = Gate(line[_p])

                # Check if qubits are provided
                _p += 1
                if line[_p] != '.':

                    # Set qubits
                    gate.on([int(x) for x in line[_p:_p + gate.n_qubits]],
                            inplace=True)
                    _p += gate.n_qubits

                else:

                    # Skip qubits
                    _p += 1

                if _p != len(line):

                    if not gate.provides('params') and (
                            _p + gate.n_params) != len(line):
                        raise ValueError('Format error.')

                    gate.set_params(
                        [float(x) for x in line[_p:_p + gate.n_params]],
                        inplace=True)
                    _p += gate.n_params

                if len(line) != _p:
                    print(line, gate)

            # Add tags
            if _tags:
                gate._set_tags(json.loads(_tags))

            # Apply power
            if _power:
                gate._set_power(float(_power))

            # Add conjugation
            if _conj:
                gate._conj()

            # Add transposition
            if _T:
                gate._T()

            # Append gate to circuit
            circuit.append(gate)

            # Reset
            _tags = None

            # Reset power
            _power = None

            # Reset conjugation
            _conj = False

            # Reset transposition
            _T = False

    # Remap qubits
    if _qubits_map is not None:

        def _int(x):
            try:
                return int(x)
            except:
                return x

        _qubits_map = json.loads(_qubits_map)
        _qubits_map = {int(x): _int(y) for x, y in _qubits_map.items()}
        for gate in circuit:
            if gate.provides('qubits') and gate.qubits is not None:
                gate._on([_qubits_map[x] for x in gate.qubits])

    return circuit
