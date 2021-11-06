# Get baseline
FROM smandra/hybridq-baseline

# Get ENV variables
ARG ARCH
ARG OMP_NUM_THREADS
ARG SKIP_PRE_CACHING
ARG HYBRIDQ_DISABLE_CPP_CORE

# Copy HybridQ
COPY ./ /opt/hybridq

# Install HybridQ
RUN cd /opt/hybridq && \
    source /opt/rh/devtoolset-10/enable && \
    $PYTHON -m pip install --no-cache --upgrade pip && \
    $PYTHON -m pip install --no-cache -v .

# Install JupyterLab
RUN $PYTHON -m pip install --no-cache notebook jupyterlab

# Create User with a Home Directory
ARG NB_USER
ARG NB_UID
ENV USER ${NB_USER:-user}
ENV HOME /home/${NB_USER:-user}
#
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

# Install alternatives
RUN alternatives --install /usr/bin/hybridq hybridq /opt/python/${PYTHON_VERSION}/bin/hybridq 20
RUN alternatives --install /usr/bin/hybridq-dm hybridq-dm /opt/python/${PYTHON_VERSION}/bin/hybridq-dm 20

# Run example to cache numba functions
RUN ${SKIP_PRE_CACHING:-false} || (cd /opt/hybridq && \
    hybridq examples/circuit.qasm /dev/null --verbose --optimize=evolution && \
    hybridq-dm examples/circuit_simple.qasm XIZXYZXZYZIXIZXXXZIZXIYY /dev/null --verbose --parallel)

# Change workdir and user
WORKDIR ${HOME}
USER ${USER}

# Add entry point
ENTRYPOINT ["/bin/bash"]
