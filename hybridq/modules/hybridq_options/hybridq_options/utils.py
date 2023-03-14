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
from inspect import stack, getmodule, signature, _ParameterKind
from functools import wraps
from os import environ

from .options import Options

__all__ = ['Default', 'parse_default']


# Define class for default values
class DefaultType:
    """
    Default type for default values.
    """

    __slots__ = ()

    def __str__(self):
        return 'Default'

    def __repr__(self):
        return str(self)


# Define default
Default = DefaultType()


class DefaultException(Exception):
    """
    Exception to raise if no default values are found for a given parameter.
    """

    def __init__(self, module, param):
        self._message = f"No default value for param '{param}' " \
                        f"in module '{module}'"
        super().__init__(self._message)


class _DynamicDoc(str):
    _opts = None
    _module = None
    _defaults = None

    def expandtabs(self, *args, **kwargs):
        # Get original docstring
        _str = str(self)

        # Add default values
        _str += '\nDefault values:'
        for _p in self._defaults:
            _str += f'\n\t{_p} = {self._opts.match(self._module, _p)}'

        # Expand tabs
        return _str.expandtabs(*args, **kwargs)


def parse_default(opts: Options,
                  /,
                  *,
                  module: str = None,
                  env_prefix: str = None):
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
    env_prefix: str, optional
        If provided, `parse_default` looks for an environment variables named
        `[env_prefix]_OPTIONNAME` (all uppercase) to override the default
        values in `opts`.

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

    # Check if opts is an instance of 'Options'
    if not isinstance(opts, Options):
        raise ValueError("'opts' must be a valid instance of 'Options'")

    # Check that keypath separator is '.'
    if opts.keypath_separator != '.':
        raise ValueError("Keypath separator for 'Option' must be '.'")

    # Get module name
    _module = getattr(getmodule(stack()[1][0]), '__name__',
                      '') if module is None else str(module)

    # Convert prefix
    _env_prefix = None if env_prefix is None else str(env_prefix).upper()

    # Define how to get default values
    def _get_default(name):

        # Check if an env variable is present
        if _env_prefix is not None and _env_prefix + '_' + name.upper(
        ) in environ:
            return environ[_env_prefix + '_' + name.upper()]

        # Otherwise, check for default values
        try:
            return opts.match(_module + opts.keypath_separator + name)
        except KeyError:
            # pylint: disable=raise-missing-from
            raise DefaultException(module=_module, param=name)

    # Define the actual decorator
    def _parse_default(func: callable):

        # Get parameters
        _params = signature(func).parameters.values()

        # Get positional parameters
        _args = tuple(
            filter(
                lambda x: x.kind != _ParameterKind.KEYWORD_ONLY and x.kind
                not in
                [_ParameterKind.VAR_POSITIONAL, _ParameterKind.VAR_KEYWORD],
                _params))

        # Get kw only parameters
        _kwargs = tuple(
            filter(
                lambda x: x.kind == _ParameterKind.KEYWORD_ONLY and x.kind
                not in
                [_ParameterKind.VAR_POSITIONAL, _ParameterKind.VAR_KEYWORD],
                _params))

        # Get default parameters
        _defaults = tuple(x.name for x in filter(
            lambda x: isinstance(x.default, DefaultType), _params))

        @wraps(func)
        def _f(*args, **kwargs):
            # Fill positional arguments
            args = tuple(
                kwargs.pop(_par.name,
                           args[_pos] if _pos < len(args) else _par.default)
                for _pos, _par in enumerate(_args))

            # Substitute default values
            args = tuple(
                _get_default(_par.name) if isinstance(_v, DefaultType) else _v
                for _v, _par in zip(args, _args))

            # Fill kw arguments only
            kwargs = {
                _par.name: kwargs.get(_par.name, _par.default)
                for _par in _kwargs
            }

            # Substitute default values
            kwargs = {
                k: _get_default(k) if isinstance(v, DefaultType) else v
                for k, v in kwargs.items()
            }

            # Call function
            return func(*args, **kwargs)

        _f.__doc__ = _DynamicDoc('' if _f.__doc__ is None else _f.__doc__)

        # pylint: disable=protected-access
        _f.__doc__._opts = opts

        # pylint: disable=protected-access
        _f.__doc__._module = _module

        # pylint: disable=protected-access
        _f.__doc__._defaults = _defaults

        # Return wrapper function
        return _f

    # Return decorator
    return _parse_default
