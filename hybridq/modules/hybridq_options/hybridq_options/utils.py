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


def parse_default(opts):
    """
    Decorate `fn` to parse `Default` values.
    """
    from . import Options

    # Check
    if not isinstance(opts, Options):
        raise AttributeError("'opts' must be an instance "
                             f"of '{Options.__name__}'")

    def _parse_default(fn: callable):
        from inspect import signature
        from functools import wraps

        # Get default parameters
        _default = frozenset(k for k, v in signature(fn).parameters.items()
                             if v.default == Default)

        @wraps(fn)
        def _fn(*args, **kwargs):
            from inspect import stack, getmodule
            # Get module name
            _module = getattr(getmodule(stack()[0][0]), '__name__', '')

            # Get default values
            _values = get_defaults(
                opts,
                **{k: kwargs.get(k, Default) for k in _default},
                module=_module)

            # Update
            kwargs.update(zip(_default, _values))

            # Return result
            return fn(*args, **kwargs)

        return _fn

    # Return wrap
    return _parse_default


# Get default value
def get_defaults(opts: Options, module: str, **kw) -> iter[any, ...]:
    """
    Return default values for HybridQ. Default values are stored in
    `hybridq.opts`.

    Parameters
    ----------
    opts: Options
        `Options` to use to get default values.
    module: str
        Module to use as part of the key.

    Returns
    -------
    iter[any, ...]
        A `tuple` of values. If the value of a given key is `Default`, then
        `hybridq.opts[key]` is returned. Otherwise, the provided value is
        returned.
    """

    def _get(key, value: any = Default):
        # If value is Default, look for default value ...
        if isinstance(value, DefaultType):

            # Return default value
            return opts.match(module + '.' + key)

        # Otherwise, just return value
        else:
            return value

    # Return values
    return map(lambda x: _get(*x), kw.items())
