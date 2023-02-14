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

__all__ = ['PrintObject', 'Printer', 'printer']


class PrintObject:
    """
    `PrintObject` is used to set the format `self`.

    Parameters
    ----------
    func: callable | str
        Function used to format `self`. If `callable`, it must accept a single
        argument, `self`, and return an object convertible to `str`. String
        will instead be evaluated using `str.format(self)`.
    pos: 'pre' | 'name' | 'bulk' | 'post', optional
        Where to locate the argument. The output of `str(obj`) is divided in
        four parts:
                            [pre][name]([bulk])[post]
        All four parts are processed in the same way, and then placed in the
        right position.
    sep: str, optional
        If this argument is followed by another within the same `pos`, use `sep`
        to separate them.
    order: int, optional
        Arguments are ordered accordingly to `order`.
    """
    __slots__ = ('_func', '_pos', '_sep', '_order')

    def __init__(self,
                 func: callable | str,
                 pos: 'pre' | 'name' | 'bulk' | 'post' = 'bulk',
                 sep: str = ', ',
                 order: int = 100):
        # Convert
        pos = str(pos)
        sep = str(sep)
        order = int(order)

        # Checks
        if not (isinstance(func, str) or callable(func)):
            raise ValueError("'func' must be a string or a callable")
        if pos not in ['pre', 'name', 'bulk', 'post']:
            raise ValueError("'pos' must be either 'pre', "
                             "'name', 'bulk' or 'post'")

        # Assign
        self._func = func
        self._pos = pos
        self._sep = sep
        self._order = order

    @property
    # pylint: disable=missing-function-docstring
    def func(self):
        return self._func

    @property
    # pylint: disable=missing-function-docstring
    def pos(self):
        return self._pos

    @property
    # pylint: disable=missing-function-docstring
    def sep(self):
        return self._sep

    @property
    # pylint: disable=missing-function-docstring
    def order(self):
        return self._order


def printer(**args) -> type:
    """
    Help setting the printing output of an object.

    Parameters
    ----------
    args:
        Arguments to include for `str(obj)`. Every argument must be either a
        `str` or a `PrintObject`. If a given argument is repeated multiple
        times while inheriting, the latest one will be used.

    See Also
    --------
    PrintObject

    Example
    -------
    @printer(a='a={self.a}',
             b=lambda self: 'b=' + str(self.b**2),
             s=PrintObject('...', order=1000),
             c=PrintObject('^{self.c}', pos='post'),
             name=PrintObject('ClassName', pos='name'))
    class A(Printer):
        def __init__(self):
            self.a = 1
            self.b = 2
            self.c = 3

    @printer(name=PrintObject('NewClass', pos='name'),
             dev=PrintObject(lambda self: type(self).mro()[1].__name__ + ' --> ',
                             pos='pre'))
    class B(A):
        ...

    A()
    > ClassName(a=1, b=4, ...)^3

    B()
    > A --> NewClass(a=1, b=4, ...)^3
    """

    # Get decorator
    def _printer(cls):
        # Check cls is inheriting from ClassProperty
        if not issubclass(cls, Printer):
            raise TypeError(f"type object '{cls.__name__}' must inherit from "
                            "'Printer'")

        # Add __printer__ to type
        cls.__printer__ = args

        # Return type
        return cls

    # Return decorator
    return _printer


class Printer:  # pylint: disable=too-few-public-methods
    """
    Enable `printer` for object.

    See Also
    --------
    printer, PrintObject
    """

    __slots__ = ()
    __printer__ = {}

    def __str__(self):
        # Collect arguments from all types
        dct = {
            k: v if isinstance(v, PrintObject) else PrintObject(v)
            for c in reversed(type(self).mro())
            for k, v in getattr(c, '__printer__', {}).items()
        }

        # Define how to join arguments
        def _join(args):
            # Sort
            args = sorted(args, key=lambda x: x.order)

            # Initialize string
            return ''.join(
                (arg.func.format(self=self) if isinstance(arg.func, str) else
                 str(arg.func(self))) + (arg.sep if i < len(args) - 1 else '')
                for i, arg in enumerate(args))

        # Initialize output
        _pre = _join(filter(lambda x: x.pos == 'pre', dct.values()))
        _name = _join(filter(lambda x: x.pos == 'name', dct.values()))
        _bulk = _join(filter(lambda x: x.pos == 'bulk', dct.values()))
        _post = _join(filter(lambda x: x.pos == 'post', dct.values()))

        # If _name is empty use class name
        _name = _name if _name else type(self).__name__

        # Return
        return _pre + _name + '(' + _bulk + ')' + _post

    def _repr_pretty_(self, arg, cycle):
        arg.text(str(self) if not cycle else '...')
