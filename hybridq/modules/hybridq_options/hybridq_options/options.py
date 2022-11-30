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

__all__ = ['Options']


class Options:
    """
    Class to manage options. Each option has the format 'node1.node2.[...].value'
    with 'node1', 'node2', ..., 'value' being valid identifier. Options can be
    set and retrieved using the square brackets:

    opts = Options()
    opts.node1['key1'] = 'hello'
    opts
    > Options(node1.key1 = hello)
    opts.node1['key1']
    > 'hello'
    opts['node1.key1']
    > 'hello'

    Options can be retrieved by looking at the closest matching:

    opts.node1.node2['key1']
    > KeyError: 'node1.node2.key1'
    opts.match('node1.node2.key1')
    > 'hello'

    An error is raised if no matches are found

    a.match('node1.node2.key2')
    > KeyError: 'node1.node2.key2'
    """

    __slots__ = ('_opts', '_node')

    @staticmethod
    def _concat(*keys):
        return '.'.join(k for k in keys if k)

    # Get node from node.key
    @staticmethod
    def _get_node(key):
        return '.'.join(key.split('.')[:-1])

    # Get key from node.key
    @staticmethod
    def _get_key(key):
        return key.split('.')[-1]

    def __init__(self, *, _opts=None, _node='', **kwargs):
        # Check
        if kwargs and _opts is not None:
            raise ValueError("Cannot use '_opts' and provide keywords "
                             "arguments")

        # Initialize
        self._opts = kwargs if _opts is None else _opts
        self._node = _node

    def __getitem__(self, key):
        return self._opts[self._concat(self._node, key)]

    def __setitem__(self, key, value):
        self._opts[self._concat(self._node, key)] = value

    def __iter__(self):
        return self.keys()

    def __str__(self):
        _sp = ',\n' + ' ' * (len(type(self).__name__) + 1)
        return type(self).__name__ + '(' + _sp.join(
            f'{k} = {v}' for k, v in self.items()) + ')'

    def __repr__(self):
        return str(self)

    def __getattr__(self, key):
        if key in self.__slots__:
            return object.__getattribute__(self, key)
        else:
            return Options(_opts=self._opts,
                           _node=self._concat(self._node, key))

    def items(self):
        # Get shift for node
        _shift = len(self._node) + 1 if self._node else 0

        # Return items
        return ((k[_shift:], v)
                for k, v in self._opts.items()
                if self._get_node(k).startswith(self._node))

    def values(self):
        # Return items
        return (v for k, v in self._opts.items()
                if self._get_node(k).startswith(self._node))

    def keys(self):
        # Get shift for node
        _shift = len(self._node) + 1 if self._node else 0

        # Return items
        return (k[_shift:]
                for k in self._opts.keys()
                if self._get_node(k).startswith(self._node))

    def match(self,
              key: str,
              how: 'all' | 'closest' | 'top' = 'closest') -> list[Key] | Key:
        """
        Get matches given `key`. If there are no matches, raise a `KeyError`.

        Parameters
        ----------
        key: str
            Key to match.
        how: 'all' | 'closest' | 'top'
            If `how == 'all'`, all possible matches are returned. If
            `how == 'closest'`, return `Key` which shares the largest number of
            nodes with `key`. Otherwise, if `how == 'top'`, return `Key` which
            shares the least amount of nodes with `key`.
        """
        # Split in actual node and key
        _node = self._get_node(key)
        _key = self._get_key(key)

        # Get matching keys
        o = sorted(filter(
            lambda x: _node.startswith(self._get_node(x[0])) and self._get_key(
                x[0]) == _key, self.items()),
                   key=lambda x: len(x[0].split('.')))

        # If no matches are found, raises
        if not len(o):
            raise KeyError(key)

        # Get all matches
        if how == 'all':
            return Options(_opts=dict(o))

        # Get closest match
        elif how == 'closest':
            return o[-1][1]

        # Get the most general key
        elif how == 'top':
            return o[0][1]

        # Raise
        else:
            raise NotImplemented
