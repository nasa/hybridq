# HybridQ: A Hybrid Simulator for Quantum Circuits

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nasa/hybridq/main)
[![GitHub License](https://img.shields.io/badge/License-Apache-green)](https://github.com/nasa/hybridq/blob/main/LICENSE)
[![Python 3.7+](https://img.shields.io/badge/Python-3.7%2B-success)]()
[![PyTest](https://github.com/nasa/hybridq/actions/workflows/python-pytest.yml/badge.svg)](https://github.com/nasa/hybridq/actions/workflows/python-pytest.yml)
[![Tutorials](https://github.com/nasa/hybridq/actions/workflows/python-tutorials.yml/badge.svg)](https://github.com/nasa/hybridq/tree/main/tutorials)<br>
[![PyPI](https://img.shields.io/pypi/v/hybridq.svg)](https://pypi.org/project/hybridq/#description)
[![Downloads](https://static.pepy.tech/personalized-badge/hybridq?period=total&units=international_system&left_color=grey&right_color=orange&left_text=downloads)](https://pepy.tech/project/hybridq)
[![Downloads-week](https://static.pepy.tech/personalized-badge/hybridq?period=week&units=international_system&left_color=grey&right_color=orange&left_text=downloads/week)](https://pepy.tech/project/hybridq)<br>
[![Documentation](https://img.shields.io/static/v1?label=documentation&message=https://nasa.github.io/hybridq&color=success)](https://nasa.github.io/hybridq)
[![Cite](https://img.shields.io/static/v1?label=cite&message=IEEE/ACM%20(QCS)&color=success)](https://doi.org/10.1109/QCS54837.2021.00015)

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

## Getting Started

Tutorials on how to use **HybridQ** can be found in
[hybridq/tutorials](https://github.com/nasa/hybridq/tree/main/tutorials).

## Contributors

[Salvatore Mandrà](https://github.com/s-mandra)<br>
[Jeffrey Marshall](https://github.com/jsmarsha11) (noise models)<br>

## How To Cite

[1] S. Mandrà, J. Marshall, E. Rieffel, and R. Biswas, [*"HybridQ: A Hybrid
Simulator for Quantum Circuits"*](https://doi.org/10.1109/QCS54837.2021.00015), 
IEEE/ACM Second International Workshop on Quantum Computing Software (QCS) (2021)

## Publications Using **HybridQ**

[1] X. Mi, P. Roushan, C. Quintana, S. Mandrà, J. Marshall, *et al.*, 
[*"Information scrambling in quantum circuits"*](https://doi.org/10.1126/science.abg5029),
Science 374, 6574 (2021)

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

1. Clone KaHyPar: 
```
git clone git@github.com:SebastianSchlag/kahypar.git /tmp/kahypar \
    --depth=1 \
    --recursive \
    --branch 1.2.1
```
2. Force installation of minimal Boost library:
* BSD:
```
sed -i '' -e "$(echo -e '/option(KAHYPAR_USE_MINIMAL_BOOST/,/)/c\' \
                "\noption(KAHYPAR_USE_MINIMAL_BOOST \"\" ON)")" \
          /tmp/kahypar/CMakeLists.txt
```
* GNU:
```
sed -i '/option(KAHYPAR_USE_MINIMAL_BOOST/,/)/c\option(KAHYPAR_USE_MINIMAL_BOOST \"\" ON)'  \
          /tmp/kahypar/CMakeLists.txt
```
3. Install KaHyPar:
```
export CXXFLAGS='-fPIC' && pip install -U /tmp/kahypar/ --force-reinstall
```

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
