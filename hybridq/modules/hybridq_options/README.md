# HybridQ-Options: Default values simplified

**HybridQ-Options** is a function decorator library to automatically retrieve
default values. Default values can be updated on-the-fly without changing the
function signature.

## Installation

**HybridQ-Options** can be installed as stand-alone library using `pip`:
```
pip install 'git+https://github.com/nasa/hybridq#egg=hybridq-options&subdirectory=hybridq/modules/hybridq_options'
```

## Getting Started

Tutorials on how to use **HybridQ-Options** can be found in
[hybridq-options/tutorials](https://github.com/nasa/hybridq/tree/main/hybridq/modules/hybridq_options/tutorials).

## How to Use


**HybridQ-Options** is a library to easily manage default options for
functions.  Each option has the format `key1.key2.[...].opt_name` with `key1`,
`key2`, ..., `opt_name` being valid strings. Options can be set and retrieved
using the square brackets:

```
from hybridq_options import Options, parse_default, Default

opts = Options()
opts['key1.key2', 'opt1'] = 1
opts['key1.key2', 'opt2'] = 2
opts['key1.key2.key3', 'opt1'] = 3

opts['key1']
> {'key2': {'opt1': 1, 'opt2': 2, 'key3': {'opt1': 3}}}

opts['key1.key2']
> {'opt1': 1, 'opt2': 2, 'key3': {'opt1': 3}}

opts['key1.key2.opt1']
> 1
```
Keys can be split at the keypath separator `.` while using square brackets
```
assert (opts['key1.key2.opt1'] == opts['key1', 'key2', 'opt1'])
assert (opts['key1.key2.opt1'] == opts['key1.key2', 'opt1'])
```
Options can also be retrieved by using the `.` notation:
```
opts.key1.key2
> {'opt1': 1, 'opt2': 2, 'key3': {'opt1': 3}}
```
The class `Options` provides the method `match` to find the closest match for a
given option. This is useful to provide a common default value for all subpaths
that share a common path:
```
# The closest option is 'key1.key2.opt1'
print('key1.key2.opt1 =', opts.match('key1.key2.opt1'))

# The closest option is 'key1.key2.opt2'
print('key1.key2.opt2 =', opts.match('key1.key2.opt2'))

# The closest option is 'key1.key2.opt1'
print('key1.key2.key4.opt1 =', opts.match('key1.key2.key4.opt1'))

# The closest option is 'key1.key2.key3.opt1'
print('key1.key2.key3.opt1 =', opts.match('key1.key2.key3.opt1'))
> key1.key2.opt1 = 1
> key1.key2.opt2 = 2
> key1.key2.key4.opt1 = 1
> key1.key2.key3.opt1 = 3
```
A `KeyError` is raised if no matches are found
```
try:
    opts.match('key1.key3', 'opt1')
except KeyError as e:
    print(e)
> "Not match for keys: '['key1', 'key3']' and option name 'opt1'"
```
The class `Options` is based on `python-benedict`:
```
print(type(opts).mro()[1].__name__)

# See 'python-benedict' for any further use of the class 'Options'
help(type(opts).mro()[1])
> benedict
> Help on class benedict in module benedict.dicts:
> [...]
```
The library **HybridQ-Options** also provides `parse_default` to automatically
parse default values for any function:
```
opts = Options()
opts['v'] = 1
opts['key1.v'] = 2

# By default, 'parse_default' uses the name of the current module as path.
# In this case, the module name is an empty string:
@parse_default(opts)
def f(v=Default):
    return v

# The closest match is 'v'
print(f'{f() = }')

# If specified, 'parse_default' will use the provided module name:
@parse_default(opts, module='key1')
def f(v=Default):
    return v

# The closest match is 'key1.v'
print(f'{f() = }')

# If specified, 'parse_default' will use the provided module name:
@parse_default(opts, module='key2')
def f(v=Default):
    return v

# The closest match is 'v'
print(f'{f() = }')
> f() = 1
> f() = 2
> f() = 1
```
Options can be changed on-the-fly:
```
opts['v'] = 'hello!'

# The closest match is 'v'
print(f'{f() = }')
> f() = 'hello!'
```
`parse_default` can parse default values for all kind of parameters:
```
# Reset options
opts.clear()

# Set options
from string import ascii_letters
from random import choices

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
assert (f() == (1, opts['key0.a'], 2, opts['key0.b'], 3, 4, opts['key0.c'], 5,
                opts['key0.d']))
```
Functions decorated using `parse_default` can be pickled as usual:
```
import pickle

# Dump binary
pickle.dumps(f)
> b"\x80..."
```
By default, the module `dill` is used to pickle the decorated function.
Alternative modules compatible with `pickle` can be also used:
```
import cloudpickle

@parse_default(opts, pickler='cloudpickle')
def f():
    ...

# Dump binary
pickle.dumps(f)
```

## How To Cite

[1] S. Mandrà, J. Marshall, E. Rieffel, and R. Biswas, [*"HybridQ: A Hybrid
Simulator for Quantum Circuits"*](https://doi.org/10.1109/QCS54837.2021.00015),
IEEE/ACM Second International Workshop on Quantum Computing Software (QCS)
(2021)

## Licence

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
