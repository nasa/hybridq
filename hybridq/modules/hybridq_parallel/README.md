# HybridQ-Parallel: Light-Weight Parallelization

**HybridQ-Parallel** is light-weight library to parallelize processes based on
`concurrent.futures.ProcessPoolExecutor`.

## Installation

`HybridQ-Parallel` can be installed as stand-alone library using `pip`:
```
pip install 'git+https://github.com/nasa/hybridq#egg=hybridq-parallel&subdirectory=hybridq/modules/hybridq_parallel'
```

## Getting Started

Tutorials on how to use **HybridQ-Parallel** can be found in
[hybridq-parallel/tutorials](https://github.com/nasa/hybridq/tree/main/hybridq/modules/hybridq_parallel/tutorials).

## How to Use

**HybridQ-Parallel** provide the equivalent of `map` to run on multiple thread:
```
from hybridq_parallel import map as pmap

list(pmap(lambda x: x**2, range(10)))
> [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```
The pool executor is automatically started when requires and its number of
workers is initialized by default to the number of available cores. The default
behavior can be changed by setting the environment variable
`HYBRIDQ_PARALLEL_MAX_WORKERS` to the desired number before starting the pool
executor. Otherwise, the pool executor can be manually started with the desired
number of workers:
```
from hybridq_parallel import map as pmap
from hybridq_parallel import restart, get_n_workers

restart(max_workers=2)
get_n_workers()
> 2
```
**HybridQ-Parallel** also provide the equivalent of `multiprocessing.starmap`:
```
from hybridq_parallel import starmap

list(starmap(lambda x, y: x * y**2, zip(range(10), range(10))))
> [0, 1, 8, 27, 64, 125, 216, 343, 512, 729]
```
**HybridQ-Parallel** can pickle any callable by wrapping it using
`hybridq_parallel.Function`:
```
from hybridq_parallel import Function
import pickle

fn = lambda x: x

try:
    pickle.dumps(fn)
except pickle.PickleError as e:
    print(e)

pickle.dumps(Function(fn))
> Can't pickle <function <lambda> at 0x7f34d0e1a170>: attribute lookup <lambda> on __main__ failed
> b'\x80...'
```
By default **HybridQ-Parallel** uses `dill` as pickler. The default behavior
can be changed by setting the environment variable `HYBRIDQ_PARALLEL_PICKLER`
to the desired pickler module. Otherwise, a valid pickler module can be
provided while calling `map` and `starmap`:
```
from hybridq_parallel import map as pmap

list(pmap(lambda x: x**2, range(10), pickler='cloudpickle'))
> [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```
Finally, **HybridQ-Parallel** uses `tqdm` to provide a progressbar:
```
from hybridq_parallel import map as pmap

list(pmap(lambda x, y: y * x**2, range(10), range(10), verbose=True))
> [0, 1, 8, 27, 64, 125, 216, 343, 512, 729]
```
Extra parameters can be passed to `tqdm` through `verbose`:
```
from hybridq_parallel import starmap

list(
    starmap(lambda x, y: x * y**2,
            zip(range(10), range(10)),
            verbose=dict(total=10)))
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
