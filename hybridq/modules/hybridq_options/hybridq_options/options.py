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
from benedict import benedict

__all__ = ['Options']


class Options(benedict):
    """
    Class to manage options. Each option has the format
    'key1.key2.[...].opt_name' with 'key1', 'key2', ..., 'opt_name' being valid
    strings. Options can be set and retrieved using the square brackets:

    opts = Options()
    opts['key1.key2', 'opt1'] = 1
    opts['key1.key2', 'opt2'] = 2
    opts['key1.key2.key3', 'opt1'] = 3

    opts['key1']
    > {'key2': {'opt1': 1, 'opt2': 2, 'key3': {'opt1': 3}}}

    opts['key1.key2']
    > {'opt1': 1, 'opt2': 2, 'key3': {'opt1': 3}}

    opts['key1.key2', 'opt1']
    > 1

    Options can also be retrieved by using the '.' notation:

    opts.key1.key2
    > {'opt1': 1, 'opt2': 2, 'key3': {'opt1': 3}}

    Options can be retrieved by looking at the closest matching:

    # The closest option is opts['key1.key2', 'opt1']
    opts.match('key1.key2', 'opt1')
    > 1

    # The closest option is opts['key1.key2', 'opt2']
    opts.match('key1.key2', 'opt2')
    > 2

    # The closest option is opts['key1.key2', 'opt1']
    opts.match('key1.key2.key4', 'opt1')
    > 1

    # The closest option is opts['key1.key2.key3', 'opt1']
    opts.match('key1.key2.key3', 'opt1')
    > 3

    An error is raised if no matches are found

    # No matches exist
    opts.match('key1.key3', 'opt1')
    > KeyError: "Not match for keys: '['key1', 'key3']' and option name 'opt1'"

    See Also
    --------
    python-benedict
    """

    def to_xls(self, *args, **kwargs) -> str:
        """
        Encode the current dict instance in XML format. Encoder specific
        options can be passed using kwargs: https://github.com/martinblech/xmltodict

        Returns
        -------
        str:
            Return the encoded string and optionally save it at `filepath`. A
            `ValueError` is raised in case of failure.

        See Also
        --------
        benedict.benedict.to_xls
        """
        return super().to_xls(*args, **kwargs)

    def match(self, *keys: tuple[str, ...]) -> any:
        """
        Find the closest match. If there are no matches, raises a `KeyError`.

        Parameters
        ----------
        *keys: tuple[str, ...]
            Keys to use to find the closest match. If multiple keys are
            provided, such keys are joint together using the keypath separator
            `,`.

        Returns
        -------
        any:
            The closest match.

        Example
        -------

        opts = Options()
        opts['key1.key2', 'opt1'] = 1
        opts['key1.key2', 'opt2'] = 2
        opts['key1.key2.key3', 'opt1'] = 3

        # The closest option is opts['key1.key2', 'opt1']
        opts.match('key1.key2', 'opt1')
        > 1

        # The closest option is opts['key1.key2', 'opt2']
        opts.match('key1.key2', 'opt2')
        > 2

        # The closest option is opts['key1.key2', 'opt1']
        opts.match('key1.key2.key4', 'opt1')
        > 1

        # The closest option is opts['key1.key2.key3', 'opt1']
        opts.match('key1.key2.key3', 'opt1')
        > 3

        # Multiple keys are joint together using the keypath separator '.'
        assert (opts.match('key1.key2.opt1') == opts.match('key1.key2', 'opt1') ==
                opts.match('key1', 'key2', 'opt1'))

        # No matches exist
        opts.match('key1.key3', 'opt1')
        > KeyError: "Not match for keys: '['key1', 'key3']' and
        > option name 'opt1'"
        """

        # Split keys
        keys = [y for x in keys for y in x.split(self.keypath_separator)]

        # Get option name
        _name = keys[-1]

        # Initialize node
        _node = self

        # Initialize value
        _value = _node.get(_name, NotImplemented)

        # Navigate dict
        for _k in keys[:-1]:

            # If _node can be navigate with key _k, get new _node
            if _k in _node and isinstance(_node[_k], Options):
                _node = _node[_k]

            # Otherwise break
            else:
                break

            # If a new _node is obtained, check if _name exists and get
            # a new _value
            if _name in _node and not isinstance(_node[_name], Options):
                _value = _node[_name]

        # If _value is not found, raise
        if _value is NotImplemented:
            raise KeyError(f"Not match for keys: '{keys[:-1]}' "
                           f"and option name '{_name}'")

        # Otherwise, return value
        return _value

    # Overload __setitem__ to accept only strings
    def __setitem__(self, keys: tuple[str, ...], value: any) -> None:
        # Check if keys are all strings
        if not all(isinstance(_k, str) for _k in keys):
            raise AttributeError("keys must be valid strings")

        # Set value
        super().__setitem__(keys, value)

    def __getattr__(self, key: str) -> any:
        if key in self:
            return self[key]

        # Otherwise, raise
        raise AttributeError(f"'{type(self).__name__}' object has "
                             f"no attribute '{key}'")
