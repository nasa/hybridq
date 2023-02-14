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
from importlib import import_module

__all__ = ['Pickler', 'pickler']


def pickler(module: str):
    """
    SWitch pickler.

    Parameters
    ----------
    pickler: str
        Pickler to use.
    """

    # Get decorator
    def _pickler(cls):
        # Check type
        if not issubclass(cls, Pickler):
            raise TypeError(f"type object '{cls.__name__}' must inherit from "
                            "'Pickler'")

        # Update pickler
        cls.__pickler__ = str(module)

        # Return type
        return cls

    # Return decorator
    return _pickler


class Pickler:
    """
    Enable `pickle` for arbitrary classes.
    """

    __slots__ = ()

    def __getstate__(self):

        # Import pickler
        _module = getattr(self, '__pickler__', 'pickle')
        _pickler = import_module(_module)

        # Get dict
        _dict = getattr(self, '__dict__', {})

        # Get slots
        _slots = {k: getattr(self, k) for k in getattr(self, '__slots__', ())}

        # Dump
        return _pickler.dumps((_dict, _slots)), _module

    def __setstate__(self, obj):

        # Import pickler
        _pickler = import_module(obj[1])

        # Get dict and slots
        _dict, _slots = _pickler.loads(obj[0])

        # If dict is present, assign it
        if _dict:
            self.__dict__ = _dict

        # If slots is present, assign it
        for key, val in _slots.items():
            setattr(self, key, val)

    @staticmethod
    def __generate__(obj, module):

        # Import pickler
        _pickler = import_module(module)

        # Get type
        cls = _pickler.loads(obj)

        # Get new type
        cls = cls.__new__(cls)

        # Return
        return cls

    def __reduce__(self):
        # pylint: disable=import-outside-toplevel
        from ._pickler import _generate

        # Get pickler
        _module = getattr(self, '__pickler__', 'pickle')
        _pickler = import_module(_module)

        # Dump
        return (_generate, (_pickler.dumps(type(self)), _module),
                self.__getstate__())
