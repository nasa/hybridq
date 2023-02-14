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
from string import ascii_letters
from functools import partial
from random import choices
from .utils import split_keys

__all__ = ['classproperty', 'ClassProperty', 'staticvars']


class classproperty(property):  # pylint: disable=invalid-name
    """
    Property attribute for classes.

    Parameters
    ----------
    fget: callable
        function to be used for getting an attribute value
    fset: callable, optional
        function to be used for setting an attribute value
    fdel: callable, optional
        function to be used for del'ing an attribute
    fmut: callable, optional
        function to be used for setting an attribute value for new types
    doc: str
        docstring

    Example
    -------

    class A(ClassProperty):
        @classproperty
        def x(cls):
            return 0

    A.x
    > 0
    """

    # pylint: disable=too-many-arguments
    def __init__(
            self,
            # pylint: disable=redefined-outer-name
            fget=None,
            fset=None,
            fdel=None,
            # pylint: disable=redefined-outer-name
            fmut=None,
            doc=None,
            *,
            _id=None):

        # Call __init__ from 'property' ...
        super().__init__(fget=fget, fset=fset, fdel=fdel, doc=doc)

        # ... set 'fmut'
        self.fmut = fmut

        # ... add unique id
        self._id = ''.join(choices(ascii_letters,
                                   k=64)) if _id is None else str(_id)

    @staticmethod
    def __generate__(state):
        # Create a new object from state
        return classproperty(**state)

    def __reduce__(self):
        # Dump state
        state = {
            'fget': self.fget,
            'fset': self.fset,
            'fdel': self.fdel,
            'fmut': self.fmut,
            'doc': self.__doc__,
            '_id': self._id
        }

        # Return reduction
        return (self.__generate__, (state,))


def fget(cls, _pkey):
    """
    Prototype of `fget`.
    """
    return getattr(cls, _pkey)


def fmut(cls, arg, _pkey, transform=None, check=None, check_msg='Check failed'):
    """
    Prototype of `fmut`.
    """
    # Transform if required
    if transform is not None:
        arg = transform(arg)

    # Check if required
    if check is not None and not check(arg):
        raise ValueError(check_msg)

    # Set
    return setattr(cls, _pkey, arg)


def staticvars(keys='',
               _pname='__{cls}_static_{key}',
               mutable=False,
               transform=None,
               check=None,
               **svars):
    """
    Add read-only attributes to type objects.

    Parameters
    ----------
    keys: str
        A string containing valid keys (separated by a comma) to add to the
        type object. Attributes provided in this way are set to
        `NotImplemented`, which would raise an error using `getattr`. To set
        these attributes, create a new type object as:

        ```
        @staticvars('a', mutable=True)
        class A(ClassProperty):
            ...

        B = type('B', (A, ), {}, static_vars=dict(a=1))
        B.a
        > 1
        ```
        This is a shorthand for `staticvars(a=NotImplemented)`
    mutable: bool, optional
        If `True`, attributes provided in both `keys` and `svars` can be
        updated while creating a new type object. Otherwise, an error would be
        raised.
        ```
        @staticvars('a')
        class A(ClassProperty):
            ...

        B = type('B', (A, ), {}, static_vars=dict(a=1))
        > AttributeError: can't set attribute 'a'

        @staticvars('a', mutable=True)
        class A(ClassProperty):
            ...

        B = type('B', (A, ), {}, static_vars=dict(a=1))
        B.a
        ```
    transform: dict[str, callable], optional
        Transform the value of a given key before storing it.
    check: dict[str, callable], optional
        Check the value of a given key before storing it. If `transform` is
        provided for a given key, the check is performed after the
        transformation.
    svars:
        Attributes/values to add to the type object.
    """

    # Initialize
    transform = {} if transform is None else transform
    check = {} if check is None else check

    # Split keys
    keys = set(split_keys(keys))

    # Check if there are any duplicates
    if _k := ', '.join(map(lambda x: f"'{x}'", keys.intersection(svars))):
        raise TypeError(
            f"staticvars() got multiple values for argument(s) {_k}")

    # Update static vars
    svars.update({k: NotImplemented for k in keys})

    # Return decorator
    def _staticvars(cls):

        # Check cls is inheriting from ClassProperty
        if not issubclass(cls, ClassProperty):
            raise TypeError(f"type object '{cls.__name__}' must inherit from "
                            "'ClassProperty'")

        # Check keys are not already in use
        if _k := next((x for x in svars if hasattr(cls, x)), None):
            raise AttributeError(f"type object '{cls.__name__}' has already "
                                 f"an attribute '{_k}'")

        # For each key and value ...
        for key, arg in svars.items():
            # Get transform
            _transform = transform.get(key)

            # Get check
            _check = check.get(key)
            _check_msg = f"Check failed for variable '{key}'"

            # If arg has been provided ...
            if arg is not NotImplemented:
                # Transform if needed
                if _transform is not None:
                    arg = _transform(arg)

                # Check if needed
                if _check is not None and not _check(arg):
                    raise ValueError(_check_msg)

            # Get private key
            _pkey = _pname.format(cls=cls.__name__, key=key)

            # Get fget
            _fget = partial(fget, _pkey=_pkey)

            # Get fmut or set to None if 'key' is not mutable
            _fmut = partial(fmut,
                            _pkey=_pkey,
                            transform=_transform,
                            check=_check,
                            check_msg=_check_msg) if mutable else None

            # Set value to private key
            setattr(cls, _pkey, arg)

            # Set classproperty to key
            setattr(cls, key, classproperty(_fget, fmut=_fmut))

        # Return new type
        return cls

    # Return decorator
    return _staticvars


class MetaClassProperty(type):
    """
    Enable `classproperty`.

    See Also
    --------
    classproperty
    """

    def __new__(mcs, name, bases, ns, static_vars=None):
        # Get new type
        cls = type.__new__(mcs, name, bases, ns)

        # If some static vars are provided ...
        if static_vars is not None:

            # For each key and value ...
            for key, arg in static_vars.items():

                # Check if the key is a classproperty
                # pylint: disable=bad-super-call
                if isinstance(
                        _arg := super(type(cls), cls).__getattribute__(key),
                        classproperty):

                    # If the classproperty is mutable
                    if _arg.fmut:

                        # Assign
                        _arg.fmut(cls, arg)

                    # Otherwise raise
                    else:
                        raise AttributeError(f"can't set attribute '{key}'")

                # Otherwise raise
                else:
                    raise AttributeError(f"can't set attribute '{key}'")

        # Return type
        return cls

    def __getattribute__(cls, key):
        # If key is a classproperty
        if isinstance(arg := super().__getattribute__(key), classproperty):

            # If arg is NotImplemented, raise
            if (arg := arg.fget(cls)) is NotImplemented:
                raise AttributeError(
                    f"type object '{cls.__name__}' has no attribute '{key}'")

            # Otherwise, return value
            return arg

        # Otherwise, return value
        return arg

    def __setattr__(cls, key, arg):
        # Try to get value ...
        try:
            _arg = super().__getattribute__(key)

        # ... if it fails, set to None.
        except:  # pylint: disable=bare-except
            _arg = None

        # Once _arg is obtained, ...
        finally:

            # Check if _arg is a classproperty
            if isinstance(_arg, classproperty):  # pylint: disable=used-before-assignment

                # If 'arg' is a classproperty and 'arg' and '_arg' share the same id,
                # we can skip the error
                if isinstance(arg, classproperty) and arg._id == _arg._id:
                    pass

                # If it cannot be set, raise
                elif _arg.fset is None:
                    raise AttributeError(f"can't set attribute '{key}'")

                # Otherwise, set.
                else:
                    _arg.fset(cls, arg)

            # If regular key, use __setattr__
            else:
                super().__setattr__(key, arg)

    def __delattr__(cls, key):
        # Try to get value ...
        try:
            _arg = super().__getattribute__(key)

        # ... if it fails, set to None.
        except:  # pylint: disable=bare-except
            _arg = None

        # Once _arg is obtained, ...
        finally:

            # Check if _arg is a classproperty
            if isinstance(_arg, classproperty):  # pylint: disable=used-before-assignment

                # If it cannot be deleted, raise
                if _arg.fdel is None:
                    raise AttributeError(f"can't delete attribute '{key}'")

                # Otherwise, delete.
                _arg.fdel(cls)

            # If regular key, use __delattr__
            else:
                super().__delattr__(key)


class ClassProperty(metaclass=MetaClassProperty):  # pylint: disable=too-few-public-methods
    """
    Enable `classproperty`.

    See Also
    --------
    classproperty
    """

    __slots__ = ()

    def __getattribute__(self, key):
        # If 'key' is NotImplemented, raise
        if (arg := super().__getattribute__(key)) is NotImplemented:
            # TODO: FIX error message when key is a classproperty and its value
            # is set to NotImplemented
            raise AttributeError(
                f"type object '{type(self).__name__}' has no attribute '{key}'")

        # Otherwise, return value
        return arg
