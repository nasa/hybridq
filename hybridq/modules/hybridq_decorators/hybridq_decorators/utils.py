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

__all__ = ['split_keys']


def split_keys(keys: str | iter[str, ...]):
    from re import sub

    # Remove spaces
    keys = sub(r"\s+", "", keys)

    # If keys is empty, return empty tuple
    if not keys:
        return ()

    # Get keys
    keys = tuple(keys.split(',')) if isinstance(keys, str) else tuple(
        map(str, keys))

    # Check keys
    if _keys := ', '.join(
            map(lambda x: f"'{x}'", filter(lambda x: not x.isidentifier(),
                                           keys))):
        _plr = _keys.count(',')
        _err = "are not valid identifiers" if _keys.count(
            ',') else "is not a valid identifier"
        raise ValueError(f"{_keys} " + _err)

    # Return keys
    return keys
