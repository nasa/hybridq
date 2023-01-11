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

import pytest


@pytest.fixture(autouse=True)
def set_seed():
    from os import environ
    from sys import stderr
    import random

    # Get random seed
    seed = random.randint(0, 2**32 - 1)

    # Get state
    state = random.getstate()

    # Set seed
    random.seed(seed)

    # Print seed
    print(f"# Used seed [{environ['PYTEST_CURRENT_TEST']}]: {seed}",
          file=stderr)

    # Wait for PyTest
    yield

    # Set state
    random.setstate(state)


@pytest.mark.parametrize('dummy', [_ for _ in range(10)])
def test__Options(dummy):
    from hybridq_options import Options

    # Define how to get a random string
    def _random_str(n=20):
        from string import ascii_letters
        from random import randint
        return ''.join(ascii_letters[randint(0,
                                             len(ascii_letters) - 1)]
                       for _ in range(n))

    # Define how to join keys
    def _join(*args):
        return '.'.join(args)

    # Get Options
    opts = Options()

    # Get keys
    _k1 = _random_str()
    _k2 = _random_str()
    _k3 = _random_str()
    _k4 = _random_str()

    # Get values
    _v1 = _random_str()
    _v2 = _random_str()
    _v3 = _random_str()
    _v4 = _random_str()
    _v5 = _random_str()

    # Add options
    opts[_join(_k1, _k2, _k3, _k4)] = _v1
    opts[_join(_k1, _k2, _k4)] = _v2
    opts[_join(_k1, _k4)] = _v3
    opts[_join(_k4)] = _v4
    opts[_join(_k2, _k4)] = _v5

    # Check a few keys
    assert (opts.match(_join(_k1, _k2, _k3, _k2, _k2, _k4)) == _v1)
    assert (opts.match(_join(_k1, _k2, _k4, _k2, _k2, _k4)) == _v2)
    assert (opts.match(_join(_k1, _k3, _k2, _k4)) == _v3)
    assert (opts.match(_join(_k3, _k4)) == _v4)
    assert (opts.match(_join(_k4, _k4)) == _v4)
    assert (opts.match(_join(_k3, _k2, _k2, _k4)) == _v4)
    assert (opts.match(_join(_k2, _k2, _k2, _k4)) == _v5)
    assert (opts.match(_join(_k2, _k1, _k4)) == _v5)

    # Add keys
    opts['a.b.e', 'f'] = 1

    # Check
    assert (opts.a.b.e['f'] == 1)
    assert (opts.match('a.b.e.r.f.g.t.h.u.f') == 1)


@pytest.mark.parametrize('dummy', [_ for _ in range(10)])
def test(dummy):
    from hybridq_options import Options, parse_default, Default
    from string import ascii_letters
    from random import choices

    # Initialize options
    opts = Options()

    # Set options
    opts['key0', 'v'] = 0
    opts['key0.key1', 'v'] = 1
    opts['key0.key2', 'v'] = 2

    # The closest match should be key0.v
    assert (parse_default(opts, module='key0')(lambda v=Default: v)() == 0)

    # The closest match should be key0.key1.v
    assert (parse_default(opts, module='key0.key1')(lambda v=Default: v)() == 1)

    # The closest match should be key0.key2.v
    assert (parse_default(opts, module='key0.key2')(lambda v=Default: v)() == 2)

    # The closest match should be key0.v
    assert (parse_default(opts, module='key0.key3')(lambda v=Default: v)() == 0)

    # The closest match should be key0.key1.v
    assert (parse_default(opts,
                          module='key0.key1.key3')(lambda v=Default: v)() == 1)

    # The closest match should be key0.key2.v
    assert (parse_default(opts,
                          module='key0.key2.key3')(lambda v=Default: v)() == 2)

    # Reset options
    opts.clear()

    # Set options
    opts['key0', 'a'] = ''.join(choices(ascii_letters, k=20))
    opts['key0', 'b'] = ''.join(choices(ascii_letters, k=20))
    opts['key0', 'c'] = ''.join(choices(ascii_letters, k=20))
    opts['key0', 'd'] = ''.join(choices(ascii_letters, k=20))

    @parse_default(opts, module='key0')
    def f(A=1,
          a=Default,
          /,
          B=2,
          b=Default,
          C=3,
          *,
          D=4,
          c=Default,
          E=5,
          d=Default):
        return A, a, B, b, C, D, c, E, d

    # Check
    assert (f() == (1, opts['key0.a'], 2, opts['key0.b'], 3, 4, opts['key0.c'],
                    5, opts['key0.d']))
