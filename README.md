# HybridQ: A Hybrid Simulator for Quantum Circuits

**HybridQ** is a highly extensible platform designed to provide a common framework
to integrate multiple state-of-the-art techniques to simulate large scale
quantum circuits on a variety of hardware. **HybridQ** provides tools to manipulate,
develop, and extend noiseless and noisy circuits for different hardware
architectures. **HybridQ** also supports large-scale high-performance computing (HPC)
simulations, automatically balancing workload among different processor nodes
and enabling the use of multiple backends to maximize parallel efficiency.
Everything is then glued together by a simple and expressive language that
allows seamless switching from one technique to another as well as from one
hardware to the next, without the need to write lengthy translations, thus
greatly simplifying the development of new hybrid algorithms and techniques.

## Contributors

[Salvatore Mandrà](https://github.com/s-mandra)<br>
[Jeffrey Marshall](https://github.com/jsmarsha11) (noise models)<br>

## Publications

[1] S. Mandrà, J. Marshall, E. Rieffel, and R. Biswas, *"HybridQ: A Hybrid Simulator for Quantum Circuits"*, arXiv (2021)

## Documentation

For technical documentation on how to use **HybridQ**, see [hybridq/docs](https://github.com/nasa/hybridq/tree/main/docs).

## Installation

**HybridQ** can be installed by either using `pip`:
```
pip install hybridq
```
directly from the `GitHub` repository (to locally compile **HybridQ** C++
libraries):
```
pip install git+https://github.com/nasa/hybridq
```
using a `zip` file:
```
pip install hybridq.zip
```
or by using `conda`:
```
conda env create -f envinronment.yml
```

**HybridQ** is also available as `docker` container (compiled for a generic
`x86-64` architecture):
```
docker pull smandra/hybridq
```

## Installation Troubleshoots

**HybridQ** depends on [KaHyPar](https://github.com/kahypar/kahypar), which
requires the [Boost C++ Library](https://www.boost.org/) installed in the
system. To properly install KaHyPar, the following steps usually work:

1. Download the boost library: https://dl.bintray.com/boostorg/release/1.73.0/source/boost_1_73_0.tar.bz2
2. Decompress the library in a temporary folder using: `tar xvjf boost_1_73_0.tar.bz2`
3. Execute: `mkdir -p $HOME/local/boost/1.73.0`
4. In the folder where the boost library has been extracted, execute: `./bootstrap.sh --prefix=$HOME/local/boost/1.73.0`
5. Once finished, execute: `./b2`
6. Once finished, execute: `./b2 install`

To install KaHyPar through `pip`:

1. Export Boost library: `export BOOST_ROOT=$HOME/local/boost/1.73.0`
2. Reinstall KaHyPar: `pip install -U git+https://github.com/kahypar/kahypar@1.2.1 --force-reinstall`

Depending on the system, the user may need to install updated versions of
`cmake` and/or `gcc/g++` to complete the installation of KaHyPar.

Alternatively, it is possible to use Conda to properly install KaHyPar:

1. Clone/unzip **HybridQ** repository and enter local copy of repository
2. Create new Conda environment: `conda env create -f environment.yml`
3. Activate environment: `conda activate hybridq`
4. Install dependencies: `conda install make cmake libboost`
5. Export Boost library: `export BOOST_ROOT=$CONDA_PREFIX`
6. Reinstall KaHyPar: `pip install -U git+https://github.com/kahypar/kahypar@1.2.1 --force-reinstall`

Depending on the system, the user may still need to install an updated version
of `gcc/g++` to complete the installation of KaHyPar. On MacOSX, it is suggested to use
`clang++` as compiler for KaHyPar because of potential linking problems. To force
the use of `clang++` to compile KaHyPar, run `export CXX=clang++` before point 6.

### OpenMP

**HybridQ** uses `OpenMP` to parallelize the C++ evolution core. However,
`clang` on MacOSX does not directly support `OpenMP`. The easiest way to
overcome this limitation is to install a version of `g++` which support the
standard C++17. Otherwise, **HybridQ** will be installed without `OpenMP`
support.

### MPI Auto-detection

**HybridQ** is able to auto-detect the use of `MPI` by checking if **HybridQ**
has been launched by either using `mpiexec` or `mpirun`. However, auto-detection
may be interfere with other libraries/software. To this end, **HybridQ** will ignore
the auto-detection of `MPI` if `DISABLE_MPI_AUTODETECT` is set to any values, that
is `export DISABLE_MPI_AUTODETECT=1`.

### `RuntimeError: Cannot set NUMBA_NUM_THREADS to a different value once the threads have been launched`

After its first launch, `quimb` pre-compiles some parts of its library using
`numba`. Such error arises when `NUMBA_NUM_THREADS` is changed after the `quimb`
library is pre-cached. The error may be removed by clearing `quimb` cache as:

1. Locate `quimb` installation folder: `python -m pip show quimb`
2. Clear cache: `rm -fr /path/to/quimb/__pycache__`

If the problem persists, try to clean `quimb` cache and set `NUMBA_NUM_THREADS`
to the desired number (typically, the number of physical cores).

## Run HybridQ in a Docker Container

**HybridQ** supports the installation in Docker containers. To create a Docker
container, it is sufficient to run:
```
docker-compose build
```
which will install **HybridQ** in the `hybridq` container (source files will be
stored in `/opt/hybridq/`). The baseline for the Docker container is
[`quay.io/pypa/manylinux2014_x86_64`](https://github.com/pypa/manylinux).  By
default, `hybridq` container is built by using the general `native`
architecture.  To use a different architecture, run for instance:
```
docker-compose build --build-arg ARCH=x86-64
```
with `ARCH` being any available `gcc` architecture. The container is built using
`python3.7`. If a different version of `python` is needed, it is possible to
specify its version in building time:
```
docker-compose build --build-arg PYTHON=cp38-cp38
```
Available versions are:
* `cp37-cp37m`
* `cp38-cp38`
* `cp39-cp39`

Once the container is built, **HybridQ** can be directly used as follows:
```
docker run --rm hybridq -c 'hybridq /opt/hybridq/examples/circuit.qasm /dev/null --verbose'
```
and tests can be run as follows:
```
docker run --rm hybridq -c 'pytest -vx /opt/hybridq/tests/tests.py'
```

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
