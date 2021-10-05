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

# Create wheel
RUN mkdir /tmp/hybridq && \
    ARCH=core-avx2 pip wheel --no-dependencies -v /opt/hybridq -w /tmp/hybridq

# Rename wheel
RUN mv /tmp/hybridq/hybridq*.whl /hybridq-$(python /opt/hybridq/setup.py --version)-py3-none-${BASE}.whl
