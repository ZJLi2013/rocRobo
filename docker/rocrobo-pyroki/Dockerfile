# rocRobo JAX sidecar — PyRoki on ROCm (jax 0.8.2 / ROCm 7.2 line).
#
# Base: rocm/jax:rocm7.2.4-jax0.8.2-py3.12. PyRoki is copied from the `pyroki/` git submodule
# at build time; bind-mount the RocRobSim repo at runtime for editable overrides (see rocRobo/readme.md §0).
#
# gfx950 / MI350X: pass AMD_COMGR_NAMESPACE=1 at docker run — see https://github.com/ROCm/jax/issues/800
#
# Build from RocRobSim repo root (submodule must be present: `git submodule update --init --recursive`):
#   docker build -f docker/rocrobo-pyroki/Dockerfile -t rocrobo/pyroki:rocm7.2.4-jax0.8.2 .
ARG JAX_BASE=rocm/jax:rocm7.2.4-jax0.8.2-py3.12
FROM ${JAX_BASE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
COPY pyroki /opt/pyroki

WORKDIR /opt/pyroki
RUN pip install --no-cache-dir -e .

CMD ["sleep", "infinity"]
