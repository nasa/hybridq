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

__all__ = ['attributes']


def __init_attributes__(
        self,
        *args,
        __init__=None,
        _pname=None,
        attributes=None,
        # pylint: disable=redefined-outer-name
        transform=None,
        check=None,
        **kwargs):
    # Initialize
    transform = {} if transform is None else transform
    check = {} if check is None else check

    # For each key and value in attributes ...
    for key, value in attributes.items():

        # Get private key
        _pkey = _pname.format(key=key)

        # Get tranform
        _transform = transform.get(key)

        # Get check
        _check = check.get(key)

        # Get value
        _v = kwargs.pop(key, value)

        # If NotImplemented, raise
        if _v is NotImplemented:
            raise TypeError(f"{type(self).__qualname__}.__init__() missing "
                            f"required keyword-only argument: '{key}'")

        # Transform if needed
        if _transform is not None:
            _v = _transform(_v)

        # Check if needed
        if _check is not None and not _check(_v):
            raise ValueError(f"Check failed for variable '{key}'")

        # Set value
        setattr(self, _pkey, _v)

    # Call init
    __init__(self, *args, **kwargs)


# pylint: disable=redefined-outer-name
def attributes(keys='',
               *,
               transform=None,
               check=None,
               _pname='_{key}',
               **attributes):
    """
    Add read-only attributes to objects.

    Parameters
    ----------
    keys: str
        A string containing valid keys (separated by a comma) to add to the
        type object. Attributes provided in this way are set to
        `NotImplemented`, which would raise an error using `getattr`. To set
        these attributes, a keyword corresponding the the attribute must be
        provided while creating the object:

        ```
        @attributes('a', b=1)
        class A:
            def __init__(self, c):
                self.c = c

        obj = A(a=2, c=1)
        obj.a, obj.b, obj.c
        > (2, 1, 1)

        obj = A(a='a', b='b', c='c')
        obj.a, obj.b, obj.c
        > ('a', 'b', 'c')
        ```
        This is a shorthand for `staticvars(a=NotImplemented)`
    transform: dict[str, callable], optional
        Transform the value of a given key before storing it.
    check: dict[str, callable], optional
        Check the value of a given key before storing it. If `transform` is
        provided for a given key, the check is performed after the
        transformation.
    attributes:
        Attributes/values to add to the object.

    Other Parameters
    ----------------
    _p_name: str, optional
        The private name used to store the actual value of the attribute.
    """

    # Get keys
    keys = split_keys(keys)

    # Check if there are any duplicates
    if _k := ', '.join(
            map(lambda x: f"'{x}'",
                set(keys).intersection(attributes))):
        raise TypeError(
            f"attributes() got multiple values for argument(s) {_k}")

    # Update attributes
    attributes.update({k: NotImplemented for k in keys})

    def _attributes(cls):

        # Check keys are not already in use
        if _k := next((x for x in attributes if hasattr(cls, x)), None):
            raise AttributeError(f"type object '{cls.__name__}' has already "
                                 f"an attribute '{_k}'")

        # Overload __init__
        cls.__init__ = partialmethod(__init_attributes__,
                                     __init__=cls.__init__,
                                     _pname=_pname,
                                     attributes=attributes,
                                     transform=transform,
                                     check=check)

        # Set keys
        for key in attributes:
            _pkey = _pname.format(key=key)
            setattr(
                cls,
                key,
                # pylint: disable=unnecessary-direct-lambda-call
                (lambda key: property(lambda self: getattr(self, key)))(_pkey))

        # Return type
        return cls

    # Return decorator
    return _attributes
