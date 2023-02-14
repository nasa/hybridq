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
from functools import partialmethod
from .utils import split_keys

__all__ = ['compare']


# pylint: disable=redefined-outer-name
def __eq_compare__(self, other, __eq__=None, compare=None):
    # Return false if original __eq__ return False
    if __eq__ is not None and not __eq__(self, other):
        return False

    # For each key in compare
    for key, cmp in compare.items():
        _x = getattr(self, key, NotImplemented)
        _y = getattr(other, key, NotImplemented)
        if _x is NotImplemented or _y is NotImplemented or not cmp(_x, _y):
            return False

    # If all checks pass, return True
    return True


def compare(keys='', **compare):
    """
    Allow to compare two objects.

    Parameters
    ----------
    keys: str
        A string containing valid keys (separated by a comma). All members of
        objects in `keys` will be checked for equality.

    compare: dict[str, callable[any, any]], optional
        If provided, the corresponding method will be tested using
        `callable[x, y]`.

    Example
    -------

    @compare('a', b=lambda x, y: abs(x) == abs(y))
    class A:
        def __init__(self, a, b):
            self.a = a
            self.b = b

    assert(A(a=1, b=2) == A(a=1, b=-2))
    assert(A(a=1, b=2) == A(a=1, b=2))
    assert(A(a=1, b=2) != A(a=-1, b=-2))

    # Keys cannot be specified in both `keys` and `compare`
    try:
        @compare('a,b', b=lambda x, y: abs(x) == abs(y))
        class A:
    except TypeError as e:
        print(e)

    > compare() got multiple values for argument(s) 'b'
    """

    # Get keys
    keys = frozenset(split_keys(keys))

    # Check if there are any duplicates
    if _k := ', '.join(map(lambda x: f"'{x}'", keys.intersection(compare))):
        raise TypeError(f"compare() got multiple values for argument(s) {_k}")

    # Update compare
    compare.update({k: lambda x, y: x == y for k in keys})

    def _compare(cls):

        # Overload __eq__
        cls.__eq__ = partialmethod(
            __eq_compare__,
            __eq__=None if cls.__eq__ == object.__eq__ else cls.__eq__,
            compare=compare)

        # Return type
        return cls

    # Return decorator
    return _compare
