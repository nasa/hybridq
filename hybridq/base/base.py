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
from copy import deepcopy
import dill as pickle
import numpy as np


def _split_names(names: {str, tuple[str, ...]}):
    return tuple(''.join(x.split()) for x in names.split(',')) if isinstance(
        names, str) else tuple(map(str, names))


def staticvars(staticvars: {str, tuple[str, ...]},
               check: dict[str, any] = None,
               transform: dict[str, any] = None,
               **defaults):
    """
    Decorator for `class`es to add static variables to them.
    """
    from inspect import isbuiltin

    # Get static vars
    staticvars = _split_names(staticvars)

    # Set defauls
    check = {} if check is None else dict(check)
    transform = {} if transform is None else dict(transform)

    # Get decorator
    def decorator(cls):
        # If __init_subclass__ is already defined, store it
        __cls_isc__ = None if isbuiltin(
            cls.__init_subclass__) or cls.__init_subclass__.__qualname__ in [
                'staticvars.<locals>.decorator.<locals>.__isc__',
                '__Base__.__init_subclass__'
            ] else cls.__init_subclass__.__func__

        # Generate __init_subclass__
        def __isc__(subcls, **kwargs):
            # Add all static variables
            for k in staticvars:
                # Add each static variable
                if k in kwargs or k in defaults:
                    # Get value (kwargs has precedence)
                    v = kwargs[k] if k in kwargs else defaults[k]

                    # Transform if needed
                    if k in transform:
                        v = transform[k](v)

                    # Check if needed
                    if k in check:
                        # Get function and message
                        _chk, _msg = (check[k],
                                      f"Check failed for '{k}'") if callable(
                                          check[k]) else check[k]

                        # Check
                        if not _chk(v):
                            raise ValueError(_msg)

                    # Set static variable
                    setattr(subcls, f'_{subcls.__name__}__{k}', v)

                    # Remove from kwargs
                    if k in kwargs:
                        del (kwargs[k])

                # If not present, raise error
                else:
                    raise ValueError(f"Static variable '{k}' must be provided.")

            # If __isc__ was provided, call it
            if __cls_isc__:
                __cls_isc__(subcls, **kwargs)

            # Otherwise, call super
            else:
                super(cls, subcls).__init_subclass__(**kwargs)

        # Add to class
        cls.__staticvars__ = staticvars
        cls.__init_subclass__ = classmethod(__isc__)
        return cls

    return decorator


def compare(staticvars: {str, tuple[str, ...]}, cmp: dict[str, any] = None):
    # Split staticvars
    staticvars = _split_names(staticvars)

    # Get default cmp
    cmp = {} if cmp is None else dict(cmp)

    # If no 'cmp' is provided for k in 'staticvars', add default
    cmp.update(
        (k, lambda x, y: x == y) for k in set(staticvars).difference(cmp))

    # Get decorator
    def decorator(cls):
        # Generate comparing function
        def __gen_compare__():
            # Get comparison function
            def __compare__(self, other):
                # Check that all cmp's are satisfied
                return isinstance(other, cls) and all(
                    np.all(c(getattr(self, k), getattr(other, k)))
                    for k, c in cmp.items())

            # Return comparison function
            return __compare__

        # Assign compare
        cls.__compare__ = __gen_compare__()

        # Return class
        return cls

    # Return decorator
    return decorator


def requires(names: {str, tuple[str, ...]}):
    """
    Add requires static variables.
    """

    # Get decorator
    def decorator(cls):
        # Add requires
        cls.__required__ = _split_names(names)

        # Return class
        return cls

    # Return decorator
    return decorator


class __Base__:
    """
    Basic features.
    """

    ##################################### INITIALIZE ClASS #####################################

    @classmethod
    def __init_subclass__(cls: type, skip_requirements: bool = False, **kwargs):
        # Get provided
        pr = cls.__get_provided__()

        # Get required
        rq = cls.__get_required__()

        # Check all required attributes are provided
        if not skip_requirements:
            d = set(rq).difference(pr)
            if d:
                raise AttributeError("The following required attributes are "
                                     f"not provided: {tuple(d)}")

        # Continue
        super().__init_subclass__(**kwargs)

    #################################### SET/GET ATTRIBUTES ####################################

    @classmethod
    def __get_staticvar__(cls: type, name: str, c: type = None) -> any:
        # If c is provided, check c only
        if c:
            # Get class
            _c = next((_c for _c in cls.mro() if _c == c), None)

            # If not found, raise error
            if _c is None:
                raise AttributeError(
                    f"type object '{c.__name__}' has no attribute '{name}'")

            # Return variable
            return getattr(_c, f'_{c.__name__}__{name}')

        # Otherwise, try every class
        else:
            # Find the first class which provides 'name'
            v = next((
                x for x in (getattr(c, f'_{c.__name__}__{name}', NotImplemented)
                            for c in cls.mro()) if x is not NotImplemented),
                     NotImplemented)

            # If v is not found, raise error
            if v is NotImplemented:
                raise AttributeError(
                    f"type object '{cls.__name__}' has no attribute '{name}'")
            else:
                return v

    @classmethod
    def __has_staticvar__(cls: type, name: str, c: type = None) -> bool:
        # Try to get static variable
        try:
            cls.__get_staticvar__(name, c)

        # If fails, return False
        except AttributeError:
            return False

        # Otherwise, return True
        else:
            return True

    @classmethod
    def __get_staticvars__(cls: type,
                           *,
                           force: bool = False) -> tuple[str, ...]:
        # Get values
        vs = None if force else getattr(cls, '__all_staticvars__', None)

        # If '__all_staticvars__' is present, return it
        if vs is not None:
            return vs

        # Otherwise, recompute from scratch
        else:
            return tuple(
                set(x for c in cls.mro()
                    for x in getattr(c, '__staticvars__', [])))

    @classmethod
    def __get_static_dict__(cls: type) -> dict[str, any]:
        # Return dictionary of static variables
        return {k: cls.__get_staticvar__(k) for k in cls.__get_staticvars__()}

    @property
    def __static_dict__(self):
        # Return dictionary of static variables
        return self.__get_static_dict__()

    @classmethod
    def __getattr__(cls: type, name: str) -> any:
        # Try to retrieve a static variable
        return cls.__get_staticvar__(name)

    def __setattr__(self, name: str, value: any) -> None:
        # Check if name is among the static variables
        if name in self.__get_staticvars__():
            raise AttributeError('Cannot set static variable')
        else:
            super().__setattr__(name, value)

    ###################################### SET/GET METHODS #####################################

    @classmethod
    def __get_methods__(cls, *, force: bool = False) -> tuple[str, ...]:
        # Get values
        vs = None if force else getattr(cls, '__all_methods__', None)

        # If class provides '__all_methods', just return it
        if vs is not None:
            return vs

        # Otherwise, recompute all methods
        else:
            return tuple(
                set(x for c in cls.mro()
                    for x in getattr(c, '__methods__', [])))

    @classmethod
    def __get_methods_dict__(cls) -> dict[str, any]:
        return {k: getattr(cls, k) for k in cls.__get_methods__()}

    @property
    def __methods_dict__(self) -> dict[str, any]:
        return type(self).__get_methods_dict__()

    ############################################# MRO ##########################################

    @classmethod
    def __get_mro__(cls):
        return tuple(
            c for c in cls.mro() if not getattr(c, '__virtual__', False))

    ######################################## COPY/DEEPCOPY #####################################

    def __copy__(self):
        _new = type(self)()
        _new.__dict__ = copy(self.__dict__)
        return _new

    def __deepcopy__(self, memo):
        _new = type(self)()
        _new.__dict__ = deepcopy(self.__dict__, memo)
        return _new

    ####################################### PROVIDED/REQUIRED #################################

    @classmethod
    def __get_provided__(cls, *, force: bool = False):
        # Get values
        vs = None if force else getattr(cls, '__all_provided__', None)

        # If '__all_provided__' is present, return it
        if vs is not None:
            return vs

        # Otherwise, recompute it
        else:
            # Get all statics
            st = cls.__get_staticvars__()

            # Get all methods
            mt = cls.__get_methods__()

            # Get all provided
            pr = tuple(x for c in cls.mro() for x in c.__dict__)

            # Return everything
            return tuple(set(st).union(mt + pr))

    @classmethod
    def __get_required__(cls, *, force: bool = False):
        # Get values
        vs = None if force else getattr(cls, '__all_required__', None)

        # If '__all_required__' is present, return it
        if vs is not None:
            return vs

        # Otherwise, recompute it
        else:
            return tuple(
                set(x for c in cls.mro()
                    for x in getattr(c, '__required__', [])))

    ######################################## REQUIREMENTS ######################################

    @classmethod
    def __get_requirements__(cls):
        return tuple(
            {x for c in cls.mro() for x in getattr(c, '__required__', [])})

    ########################################## PROVIDES ########################################

    def provides(self,
                 method: {str, iter[str]},
                 *,
                 which: {all, any} = all) -> bool:
        # Split methods
        method = _split_names(method)

        # Check which
        if which not in [all, any]:
            raise ValueError(f"'which={which}' not supported")

        # Check
        return which(
            getattr(self, m, NotImplemented) is not NotImplemented
            for m in method)

    ##################################### METHODS FOR PICKLE ###################################

    def __getstate__(self):
        return pickle.dumps(self.__dict__)

    def __setstate__(self, state):
        self.__dict__ = pickle.loads(state)

    @staticmethod
    def __generate__(class_name: str, mro: list[type], staticvars, methods):
        return generate(class_name=class_name,
                        mro=mro,
                        methods=dict(pickle.loads(methods)),
                        **pickle.loads(staticvars))()

    def __reduce__(self):
        # Get class
        cls = type(self)

        # Return reduction
        return (cls.__generate__, (cls.__name__, cls.__get_mro__(),
                                   pickle.dumps(self.__static_dict__),
                                   pickle.dumps(self.__methods_dict__)),
                self.__getstate__())

    #################################### STRING REPRESENTATION #################################

    def __str__(self) -> str:
        """
        Return string representation of Gate.
        """
        # Get all elements to print
        _all_collect = {
            str(k): (int(x), str(v), int(p))
            for c in reversed(type(self).mro()) for k, (
                x, v,
                p) in getattr(c, '__print__', lambda self: {})(self).items()
        }

        # Print elements at the left of Gate()
        _collect = {
            k: (x, v) for k, (x, v, p) in _all_collect.items() if p == -1
        }
        _str = ''.join(
            str(v)
            for _, v in sorted(_collect.values(), key=lambda x: x[0])
            if v)

        # Print name
        _str += f'{type(self).__name__}'

        # Print elements between name and ()
        _collect = {
            k: (x, v) for k, (x, v, p) in _all_collect.items() if p == 2
        }
        _str += ''.join(
            str(v)
            for _, v in sorted(_collect.values(), key=lambda x: x[0])
            if v)

        # Print elements inside Gate()
        _collect = {
            k: (x, v) for k, (x, v, p) in _all_collect.items() if p == 0
        }
        _str += '('
        _str += ', '.join(
            str(v)
            for _, v in sorted(_collect.values(), key=lambda x: x[0])
            if v)
        _str += ')'

        # Print elements at the right of Gate()
        _collect = {
            k: (x, v) for k, (x, v, p) in _all_collect.items() if p == 1
        }
        _str += ''.join(
            str(v)
            for _, v in sorted(_collect.values(), key=lambda x: x[0])
            if v)

        # Return string
        return _str

    def __repr__(self) -> str:
        """
        Return string representation of Gate.
        """

        return self.__str__()

    ######################################### COMPARISON #######################################

    def __eq__(self, other) -> bool:
        return all(
            getattr(cls, '__compare__', lambda self, other: True)(self, other)
            for cls in type(self).mro())


def generate(class_name: str,
             mro: iter[type],
             methods: dict[str, any] = None,
             **staticvars):
    """
    Generate new `type`.

    Parameters
    ----------
    class_name: str
        Name of the new `type`. It must be a valid identified.
    mro: iter[type]
        A series of `type`s to derive from.
    methods: dict[str, any]
        Extra method to add to class.

    Returns
    -------
    type
        The new `type`.
    """
    if not isinstance(class_name, str) or not class_name.isidentifier():
        raise ValueError("'class_name' is not a valid identifier.")

    # Get methods
    methods = {} if methods is None else dict(methods)

    # Add __Base__ if nor already included
    mro = tuple(mro)
    if __Base__ not in mro:
        mro += (__Base__,)

    # Get new_type
    new_type = type(
        class_name, mro, {
            **methods, '__methods__': tuple(methods),
            '__staticvars__': tuple(staticvars)
        }, **staticvars)

    # Add all static variables
    new_type.__all_staticvars__ = new_type.__get_staticvars__(force=True)

    # Add all methods
    new_type.__all_methods__ = new_type.__get_methods__(force=True)

    # Add all provided variables
    new_type.__all_provided__ = new_type.__get_provided__(force=True)

    # Add all required variables
    new_type.__all_required__ = new_type.__get_required__(force=True)

    # Add virtual flag
    new_type.__virtual__ = True

    # Return new_type
    return new_type
