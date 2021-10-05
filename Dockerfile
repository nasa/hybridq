# Get baseline
FROM hybridq-baseline

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
    python -m pip install -v . && \
    (python -m pip cache purge || true)

# Update LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/opt/python/${PYTHON}/lib/:$LD_LIBRARY_PATH

# Run example to cache numba functions
RUN ${SKIP_PRE_CACHING:-false} || (cd /opt/hybridq && \
    hybridq examples/circuit.qasm /dev/null --verbose --optimize=evolution && \
    hybridq-dm examples/circuit_simple.qasm XIZXYZXZYZIXIZXXXZIZXIYY /dev/null --verbose --parallel)

# Add entry point
ENTRYPOINT ["/bin/bash"]
