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

__all__ = ['requires', 'provides']


def __init_requires__(self, *args, __init__=None, **kwargs):
    # Call init
    __init__(self, *args, **kwargs)

    # Find first fail
    if _k := next(
        (x for x in self.__get_requires__().difference(self.__get_provides__())
         if not hasattr(self, x)), None):
        raise AttributeError(
            f"type object '{type(self).__name__}' requires '{_k}'")


def __get_requires__(cls):
    return frozenset(
        k for c in cls.mro() for k in getattr(c, '__requires__', ()))


def __get_provides__(cls):
    return frozenset(
        k for c in cls.mro() for k in getattr(c, '__provides__', ()))


def requires(keys):
    """
    Add required variables to a class.

    Parameters
    ----------
    keys: str
        A string containing valid keys (separated by a comma). Keys provided in
        this way will raise an `AttributeError` if the class not implements
        methods with such names.
    """

    keys = frozenset(split_keys(keys))

    def _requires(cls):
        cls.__requires__ = getattr(cls, '__requires__', frozenset()).union(keys)
        cls.__provides__ = getattr(cls, '__provides__', frozenset())
        cls.__get_requires__ = classmethod(__get_requires__)
        cls.__get_provides__ = classmethod(__get_provides__)
        cls.__init__ = partialmethod(__init_requires__, __init__=cls.__init__)
        return cls

    return _requires


def provides(keys):
    """
    Add a list of variables provided by the class.

    Parameters
    ----------
    keys: str
        A string containing valid keys (separated by a comma). Keys provided in
        this way and that also appear in `requires` will not trigger an
        `AttributeError`.
    """

    keys = frozenset(split_keys(keys))

    def _provides(cls):
        cls.__provides__ = getattr(cls, '__provides__', frozenset()).union(keys)
        return cls

    return _provides
