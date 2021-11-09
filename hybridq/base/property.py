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
from hybridq.base import __Base__, generate
from hybridq.base import staticvars, compare, requires
from hybridq.utils import isintegral, isnumber
from copy import copy, deepcopy


@staticvars('docstring',
            docstring="",
            transform=dict(docstring=lambda x: str(x)))
class DocString(__Base__):
    """
    Add docstring to an object.
    """

    def __init_subclass__(cls, **kwargs):
        # Call super
        super().__init_subclass__(**kwargs)

        # Update docstring
        cls.__doc__ = cls.__get_staticvar__('docstring')


class Tags(__Base__):
    """
    Add tags to a object.

    Attributes
    ----------
    tags: dict[any, any], optional
        Dictionary of tags.
    """

    def __init__(self, tags: dict[any, any] = None, **kwargs) -> None:
        # Call super
        super().__init__(**kwargs)

        # Set tags
        self.__tags = {} if tags is None else dict(tags)

    def __print__(self) -> dict[str, tuple[int, str, int]]:
        return {'tags': (999, f'tags={self.tags}' if self.tags else '', 0)}

    @property
    def tags(self) -> dict[any, any]:
        return self.__tags

    def _set_tags(self, tags: dict[any, any]) -> None:
        """
        Set `tags` to `Tags`.
        """

        self.set_tags(tags, inplace=True)

    def set_tags(self,
                 tags: dict[any, any] = None,
                 *,
                 inplace: bool = False) -> Tags:
        """
        Return `Tags` with given `tags`. All previous tags are removed and
        substituted with `tags`. If `inplace` is `True`, `Tags` is modified
        in place.

        Parameters
        ----------
        tags: dict[any, any]
            Parameters used to define the new `Tags`.
        inplace: bool, optional
            If `True`, `Tags` is modified in place. Otherwise, a new
            `Tags` is returned.

        Returns
        -------
        Tags
            New `Tags` with `tags`. If `inplace` is `True`, `Tags` is
            modified in place.
        """

        # Make a copy if needed
        if inplace:
            _g = self
        else:
            _g = deepcopy(self)

        # If tags is not None, set tags
        if tags is not None:

            # Assign tags
            _g.__tags = dict(tags)

        # Otherwise, remove tags
        else:

            # Clear tags
            _g.__tags.clear()

        return _g

    def _update_tags(self, *args, **kwargs) -> None:
        """
        Update `Tags`'s `tags`.
        """

        self.update_tags(*args, **kwargs, inplace=True)

    def update_tags(self, *args, inplace: bool = False, **kwargs) -> Tags:
        """
        Return `Tags` with updated tags. If `inplace` is `True`, `Tags` is
        modified in place.

        Parameters
        ----------
        inplace: bool, optional
            If `True`, `Tags` is modified in place. Otherwise, a new `Tags` is
            returned.

        Returns
        -------
        Tags
            New `Tags` with updated tags. If `inplace` is `True`, `Tags` is
            modified in place.
        """

        # Make a copy if needed
        if inplace:
            _g = self
        else:
            _g = deepcopy(self)

        # Update tags
        _g.tags.update(*args, **kwargs)

        return _g

    def _remove_tags(self, keys: iter[any]) -> None:
        """
        Remove tags matching `keys`.
        """

        self.remove_tags(keys, inplace=True)

    def remove_tags(self, keys: iter[any], *, inplace: bool = False) -> Tags:
        """
        Return `Tags` with removed tags matching `keys`. If `inplace` is
        `True`, `Tags` is modified in place.

        Parameters
        ----------
        keys: iter[any]
            Keys to remove from tags.
        inplace: bool, optional
            If `True`, `Tags` is modified in place. Otherwise, a new
            `Tags` is returned.

        Returns
        -------
        Tags
            New `Tags` with `keys` in tags removed. If `inplace` is `True`,
            `Tags` is modified in place.
        """

        # Make a copy if needed
        if inplace:
            _g = self
        else:
            _g = deepcopy(self)

        # Convert to set
        keys = set(keys)

        # Remove tags
        _g._set_tags({k: v for k, v in _g.tags.items() if k not in keys})

        return _g

    def _remove_tag(self, key: any) -> None:
        """
        Remove tag matching `key`.
        """

        self.remove_tag(key, inplace=True)

    def remove_tag(self, key: any, *, inplace: bool = False) -> Tags:
        """
        Return `Tags` with removed tag mathcing `key`. If `inplace` is
        `True`, `Tags` is modified in place.

        Parameters
        ----------
        key: any
            Key to remove from tags.
        inplace: bool, optional
            If `True`, `Tags` is modified in place. Otherwise, a new
            `Tags` is returned.

        Returns
        -------
        Tags
            New `Tags` with `key` in tags removed. If `inplace` is `True`,
            `Tags` is modified in place.
        """

        return self.remove_tags([key], inplace=inplace)


@compare('name')
@staticvars('name',
            check=dict(name=(lambda s: isinstance(s, str),
                             "'name' must be 'str'")))
class Name(__Base__):
    """
    Add name to a object.
    """

    def __print__(self) -> dict[str, tuple[int, str, int]]:
        return {
            'name': (0, f"name='{self.name}'" if self.name else '', 0),
        }


@compare('n_params,params')
@staticvars(
    'n_params',
    check=dict(n_params=(lambda n: n is any or (isintegral(n) and n >= 0),
                         "'n_params' must be a non-negative integer")))
class Params(__Base__):
    """
    Add parameters to class.

    Attributes
    ----------
    params: iter[any], optional
    """

    def __init__(self, params: iter[any] = None, **kwargs) -> None:
        # Call super
        super().__init__(**kwargs)

        # Set params
        self._set_params(params)

    @property
    def params(self) -> tuple[any]:
        return self.__params

    def __print__(self) -> dict[str, tuple[int, str, int]]:
        return {
            'n_params':
                (11, f"n_params={self.n_params}" if self.params is None else '',
                 0),
            'params': (101, f"params={self.params}" if self.params else "", 0),
        }

    def _set_params(self, params: iter[any]) -> None:
        """
        Set `params` to `Params`.
        """

        self.set_params(params, inplace=True)

    def set_params(self, params: iter[any], *, inplace: bool = False) -> Params:
        """
        Return `Params` with given `params`. If `inplace` is `True`,
        `Params` is modified in place.

        Parameters
        ----------
        params: iter[any]
            Parameters used to define the new Params.
        inplace: bool, optional
            If `True`, `Params` is modified in place. Otherwise, a new
            `Params` is returned.

        Returns
        -------
        Params
            New `Params` with `params`. If `inplace` is `True`, `Params`
            is modified in place.
        """

        # Set default
        if params is None and self.n_params == 0:
            params = tuple()

        # Check params is tuple convertible
        if params is not None:
            try:
                params = tuple(params)
            except:
                raise ValueError("'params' must be 'tuple' convertible.")

        # Check lenght
        if params is not None and len(params) != self.n_params:
            raise ValueError(f"Wrong number of 'params' "
                             f"(expected {self.n_params}, got {len(params)})")

        # Make a copy if needed
        if inplace:
            _g = self
        else:
            _g = deepcopy(self)

        # Set
        _g.__params = params

        # Return
        return _g


@compare('elements')
@staticvars('_base_check',
            _base_check=None,
            check=dict(_base_check=lambda ts: ts is None or all(
                callable(k) and all(type(t) == type for t in ts) for k, ts in ts
                .items())),
            transform=dict(_base_check=lambda ts: None if ts is None else dict(
                (k, tuple(ts)) for k, ts in dict(ts).items())))
class Tuple(Tags, __Base__):
    """
    Tuple class for `__Base__`.
    """

    def __init__(self, elements=tuple(), tags=None, **kwargs):
        # Call super
        super().__init__(tags=tags, **kwargs)

        # Convert elements to tuple
        elements = tuple(elements)

        # Check that all elements are __Base__
        if not all(isinstance(el, __Base__) for el in elements):
            raise TypeError(f"Only '__Base__' elements are supported")

        # Get possible basis
        _base_check = self.__get_staticvar__('_base_check')

        if _base_check and not all(
                f(isinstance(el, t)
                  for t in ts)
                for el in elements
                for f, ts in _base_check.items()):
            raise TypeError(
                f"Only {self.__get_staticvar__('_base_check')} elements are supported"
            )

        # Initialize elements
        self.__elements = elements

    @property
    def elements(self):
        return self.__elements

    def __len__(self) -> int:
        return len(self.elements)

    def __print__(self) -> dict[str, tuple[int, str, int]]:
        # Get string representation
        _el = ', '.join(str(e) for e in self.elements)

        # Return representation
        return dict(elements=(0, _el if len(self) else '', 0))

    def __getitem__(self, *args, **kwargs):
        # Get elements
        return self.elements.__getitem__(*args, **kwargs)

    def index(self, *args, **kwargs):
        # Get index
        return self.elements.index(*args, **kwargs)

    def __radd__(self, other: Tuple):
        # If 'other' is 'tuple' try to convert ..
        if isinstance(other, tuple):
            # .. to type(self) or ..
            try:
                other = type(self)(other)

            # .. to Tuple.
            except:
                other = Tuple(other)

        # Check that other is a Tuple
        if not isinstance(other, Tuple):
            raise TypeError(f"Type '{type(other).__name__}' not supported")

        # Return Tuple. If self and other have different types, fallback to Type.
        return (type(self) if type(self) == type(other) else
                Tuple)(other.elements + self.elements)

    def __add__(self, other: Tuple) -> Tuple:
        # If 'other' is 'tuple' try to convert ..
        if isinstance(other, tuple):
            # .. to type(self) or ..
            try:
                other = type(self)(other)

            # .. to Tuple.
            except:
                other = Tuple(other)

        # 'other' must be Tuple
        if not isinstance(other, Tuple):
            raise TypeError(f"Type '{type(other).__name__}' not supported")

        # Get left and right tags
        l_tags = self.tags if self.provides('tags') else {}
        r_tags = other.tags if other.provides('tags') else {}

        # Get common keys
        ckeys = set(l_tags).intersection(r_tags)

        # Create new tags
        tags = {}
        tags.update({
            (str(k) + '_x' if k in ckeys else k): v for k, v in l_tags.items()
        })
        tags.update({
            (str(k) + '_y' if k in ckeys else k): v for k, v in r_tags.items()
        })

        # If self and other have different types, fallback to Tuple
        other = (type(self) if type(self) == type(other) else
                 Tuple)(self.elements + other.elements)

        # Update tags and return
        return other.update_tags(tags, inplace=True)

    def flatten(self) -> Tuple:
        """
        Return a flattend `Tuple`.
        """
        return type(self)(y for x in (g.flatten(
        ) if isinstance(g, Tuple) and g.provides('flatten') else [g]
                                      for g in self) for y in x)
