# Get base
ARG BASE

# Get baseline
FROM quay.io/pypa/${BASE}

# Get base
ARG BASE

# Get Python version
ARG PYTHON

# Set default if needed
ENV PYTHON=${PYTHON:-cp37-cp37m}

# Update PATH
ENV PATH=/opt/python/${PYTHON}/bin/:$PATH

# Copy HybridQ
COPY ./ /opt/hybridq

# Add x64-64 warning
RUN sed -i '/__future__/afrom warnings import warn\nwarn("""The C++ simulator for this package has been compiled for a\ngeneral x86-64 architecture. For a better performance, try\n`pip install -U git+https://github.com/nasa/hybridq`.""")' /opt/hybridq/hybridq/circuit/simulation/simulation.py

# Create wheel
RUN mkdir /tmp/hybridq && \
    ARCH=x86-64 pip wheel --no-dependencies -v /opt/hybridq -w /tmp/hybridq

# Rename wheel
RUN mv /tmp/hybridq/hybridq*.whl /hybridq-$(python /opt/hybridq/setup.py --version)-py3-none-${BASE}.whl
