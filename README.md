# HybridQ: A Hybrid Simulator for Quantum Circuits

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nasa/hybridq/main)
[![GitHub License](https://img.shields.io/badge/License-Apache-green)](https://github.com/nasa/hybridq/blob/main/LICENSE)
[![Python 3.7+](https://img.shields.io/badge/Python-3.7%2B-success)]()
[![PyTest](https://github.com/nasa/hybridq/actions/workflows/python-pytest.yml/badge.svg)](https://github.com/nasa/hybridq/actions/workflows/python-pytest.yml)
[![Tutorials](https://github.com/nasa/hybridq/actions/workflows/python-tutorials.yml/badge.svg)](https://github.com/nasa/hybridq/tree/main/tutorials)<br>
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/nasa/hybridq.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/nasa/hybridq/context:python)
[![Language grade: C++](https://img.shields.io/lgtm/grade/cpp/g/nasa/hybridq.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/nasa/hybridq/context:cpp)<br>
[![PyPI](https://img.shields.io/pypi/v/hybridq.svg)](https://pypi.org/project/hybridq/#description)
[![Downloads](https://static.pepy.tech/personalized-badge/hybridq?period=total&units=international_system&left_color=gray&right_color=orange&left_text=downloads)](https://pepy.tech/project/hybridq)
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
