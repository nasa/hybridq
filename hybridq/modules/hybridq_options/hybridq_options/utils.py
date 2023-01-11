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

__all__ = ['Default', 'parse_default']


# Define class for default values
class DefaultType:
    __slots__ = ()

    def __str__(self):
        return 'Default'

    def __repr__(self):
        return str(self)


# Define default
Default = DefaultType()


class DefaultException(Exception):

    def __init__(self, module, param):
        self._message = f"No default value for param '{param}' " \
                        f"in module '{module}'"
        super().__init__(self._message)


class _Function:
    __slots__ = ('_f', '_opts', '_module', '_pickler', '_args', '_kwargs')

    def __init__(self,
                 f: callable,
                 /,
                 opts: Options,
                 *,
                 module: str = None,
                 pickler: str = 'dill'):
        from inspect import stack, getmodule, signature, _ParameterKind
        from .options import Options

        # Set function
        self._f = f

        # Set options
        self._opts = Options(opts)

        # Check that keypath separator is '.'
        if self._opts._keypath_separator != '.':
            raise ValueError("Keypath separator for 'Option' must be '.'")

        # Set module
        self._module = getattr(getmodule(stack()[0][0]), '__name__',
                               '') if module is None else str(module)

        # Set pickle
        self._pickler = str(pickler)

        # Get parameters
        _params = signature(f).parameters.values()

        # Get positional parameters
        self._args = tuple(
            filter(lambda x: x.kind != _ParameterKind.KEYWORD_ONLY, _params))

        # Get kw parameters onlyt
        self._kwargs = tuple(
            filter(lambda x: x.kind == _ParameterKind.KEYWORD_ONLY, _params))

    def __getstate__(self):
        from importlib import import_module

        # Load pickler
        pickler = import_module(self._pickler)

        # Dump state
        return pickler.dumps(tuple(getattr(self, x) for x in self.__slots__))

    def __setstate__(self, state):
        from importlib import import_module

        # Load pickler
        pickler = import_module(self._pickler)

        # Load state
        for _name, _val in zip(self.__slots__, pickler.loads(state)):
            setattr(self, _name, _val)

    def _get_default(self, name):
        try:
            return self._opts.match(self._module +
                                    self._opts._keypath_separator + name)
        except KeyError:
            raise DefaultException(module=self._module, param=name)

    def __call__(self, *args, **kwargs):

        # Fill positional arguments
        args = tuple(
            kwargs.pop(_par.name,
                       args[_pos] if _pos < len(args) else _par.default)
            for _pos, _par in enumerate(self._args))

        # Substitute default values
        args = tuple(
            self._get_default(_par.name) if _v == Default else _v
            for _v, _par in zip(args, self._args))

        # Fill kw arguments only
        kwargs = {
            _par.name: kwargs.get(_par.name, _par.default)
            for _par in self._kwargs
        }

        # Substitute default values
        kwargs = {
            k: self._get_default(k) if v == Default else v
            for k, v in kwargs.items()
        }

        # Call function
        return self._f(*args, **kwargs)


def parse_default(opts: Options,
                  /,
                  *,
                  module: str = None,
                  pickler: str = 'dill'):
    """
    Decorate function to automatically substitute `Default` values with values
    provided in `opts`.

    Parameters
    ----------
    opts: Options
        Values to use as default arguments.
    module: str, optional
        Prefix to use to match variable in `opts`. If not provided, the name of
        the calling module is used.
    pickler: str, optional
        Module to use to pickle the decorated function.

    Example
    -------

    opts = Options()
    opts['key1', 'v'] = 1
    opts['key1.key2', 'v'] = 'hello!'

    @parse_default(opts, module='key1')
    def f(v = Default):
        return v

    # The closest match is 'key1.v'
    f()
    > 1

    @parse_default(opts, module='key1.key2')
    def f(v = Default):
        return v

    # The closest match is 'key1.key2.v'
    f()
    > 'hello!'

    @parse_default(opts, module='key1.key3')
    def f(v = Default):
        return v

    # The closest match is 'key1.v'
    f()
    > 1

    # Options can be updated on-the-fly
    opts['key1.v'] = 42

    # The closest match is 'key1.v'
    f()
    > 42
    """

    # Define the actual decorator
    def _parse_default(f: callable):
        return _Function(f, opts, module=module, pickler=pickler)

    # Return decorator
    return _parse_default
