"""
Author: Salvatore Mandra (salvatore.mandra@nasa.gov)

Copyright © 2021, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The Decorama: Useful Decorators For Classes is licensed under the Apache
License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

__all__ = ['hasher']


def __hash__(self, keys=()):
    from builtins import hash
    from pickle import dumps

    # Return hash value of the pickled object
    if keys:
        return hash(tuple(map(__hash__, map(lambda x: getattr(self, x), keys))))
    else:
        return hash(dumps(self))


def hasher(_cls=None, *, keys='', method=None):
    """
    Enable `hash` for arbitrary objects.

    Parameters
    ----------
    keys: str, optional
        A string containing valid keys (separated by a comma) to add to the
        type object. If provided, only `keys` will be used to compute the hash.
    method: callable, optional
        Method to use to hash object. By default, the hash value of the pickled
        object is used.

    Example
    -------
    import pickle

    @hasher
    class A(list):
        pass

    hash([1, 2, 3])
    > TypeError: unhashable type: 'list'

    a = A([1, 2, 3])
    hash(a)
    > 5473906009225658085

    # By default, the hash value is equal to the hash value of the
    # pickled object
    hash(pickle.dumps(a)) == hash(a)
    > True

    @hasher(method=lambda x: sum(x))
    class B(list):
        pass

    b = B([1, 2, 3])
    hash(b)
    > 6

    hash(pickle.dumps(b)) == hash(b)
    > False

    hash(sum(b)) == hash(b)
    > True
    """

    # If method is not defined, use default
    if method is None:
        from functools import partialmethod
        from .utils import split_keys

        # Initialize method
        method = partialmethod(__hash__, keys=frozenset(split_keys(keys)))

    # Initialize decorator
    def _hash(cls):
        # Add hash to class
        cls.__hash__ = method

        # Return class
        return cls

    # Return decorator or decorated class
    return _hash if _cls is None else _hash(_cls)
