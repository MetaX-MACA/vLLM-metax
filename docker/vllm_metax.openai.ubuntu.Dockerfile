# vLLM-MetaX Ubuntu 22.04 build file for the OpenAI-compatible server on MetaX
# (MACA) GPUs.
#
# Modeled on:
#   - upstream vLLM's docker/Dockerfile: stage layout, uv configuration,
#     BuildKit cache mounts, wheel-checksum cache-busting, OCI labels and the
#     `vllm serve` entrypoint;
#   - vLLM-MetaX's docker/vllm_metax.openai.Dockerfile: MACA SDK / cu-bridge
#     installation, wheel build and runtime dependencies.
#
# Environment-specific adaptations (this network):
#   - registry.access.redhat.com, archive.ubuntu.com and gitee.com are
#     unreachable. The base image comes from the internal mirror, apt packages
#     from the aliyun mirror, and cu-bridge is vendored in docker/cu-bridge.
#   - Ubuntu 22.04 ships no Python 3.12 (the Metax wheels are cp312-only), so
#     CPython 3.12 is built from a GitHub source tarball.
#   - The Metax python wheels (maca 3.8.2.2, cp312) are vendored in
#     docker/wheels/ and the MACA SDK deb bundle (3.8.2.2) in docker/maca-debs/.
#     These bundles are not tracked in git; see docker/build_image.sh.
#   - requirements/*.txt are used as-is; the committed metax3.8.2.2 pins match
#     the vendored 3.8.2.2 wheels, so the final stage only filters kernel/index
#     packages out of the override. The repository files are never modified.
#
# v0.29.0 note: the 0.29 line still pins metax3.8.2.2 wheels, so the MACA SDK
# bundle and the vendored wheels are shared with v0.28.0. TODO(v0.29.0): update
# the bundle versions (and the COPY paths below) if the release ships a different
# MACA SDK / cu-bridge.

# =============================================================================
# VERSION MANAGEMENT
# =============================================================================

ARG BUILD_BASE_IMAGE=registry-docker.hub.metax-tech.com/library/ubuntu:22.04
ARG PYTHON_VERSION=3.12

# Python package indexes (Metax mirrors)
ARG UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ARG UV_EXTRA_INDEX_URL=https://repos.metax-tech.com/r/maca-pypi/simple
ARG UV_TRUSTED_HOST=repos.metax-tech.com

# Versioned inputs. VLLM_VERSION is the upstream vLLM release targeted by the
# plugin, MACA_VERSION selects the Metax-Driver + MACA SDK.
ARG VLLM_VERSION=0.29.0
ARG MACA_VERSION=3.8
ARG CU_BRIDGE_VERSION=3.8.2.2
ARG PYTHON_SOURCE_URL=https://github.com/python/cpython/archive/refs/tags/v3.12.9.tar.gz

# Proxy vars may be supplied via --build-arg; declared here so they are
# visible to RUN steps (needed in networks that require a corporate proxy).
ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

# Build options
ARG MAX_JOBS
ARG CMAKE_BUILD_TYPE

# Image metadata
ARG VLLM_BUILD_COMMIT
ARG VLLM_BUILD_PIPELINE=local
ARG VLLM_BUILD_URL
ARG VLLM_IMAGE_TAG=local/vllm-metax-ubuntu:dev

#################### BASE BUILD IMAGE ####################
FROM ${BUILD_BASE_IMAGE} AS base

ARG PYTHON_VERSION
ARG PYTHON_SOURCE_URL

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:/root/.local/bin:$PATH"

# Point apt at the aliyun mirror (archive.ubuntu.com is unreachable from this
# network). Ubuntu 22.04 uses /etc/apt/sources.list; newer releases may use
# deb822 sources, so handle both.
RUN if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
        sed -i 's@http://archive.ubuntu.com/ubuntu@https://mirrors.aliyun.com/ubuntu@g; s@http://security.ubuntu.com/ubuntu@https://mirrors.aliyun.com/ubuntu@g' /etc/apt/sources.list.d/ubuntu.sources; \
    else \
        sed -i 's@//archive.ubuntu.com/ubuntu@//mirrors.aliyun.com/ubuntu@g; s@//security.ubuntu.com/ubuntu@//mirrors.aliyun.com/ubuntu@g' /etc/apt/sources.list; \
    fi && \
    apt-get update

# Build toolchain + CPython 3.12 build dependencies. python3-pip (3.10) is only
# used to bootstrap uv.
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential curl git ca-certificates unzip \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev \
        libncursesw5-dev liblzma-dev libgdbm-dev uuid-dev \
        python3-pip python3-venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Build CPython ${PYTHON_VERSION} from source (python.org is blocked; the
# GitHub tarball is reachable through the proxy).
RUN curl -fsSL -o /tmp/cpython.tar.gz "${PYTHON_SOURCE_URL}" && \
    mkdir -p /tmp/cpython && \
    tar -xzf /tmp/cpython.tar.gz -C /tmp/cpython --strip-components=1 && \
    cd /tmp/cpython && \
    ./configure --prefix=/usr/local && \
    make -j"$(nproc)" && \
    make install && \
    ln -sf /usr/local/bin/python${PYTHON_VERSION} /usr/local/bin/python && \
    rm -rf /tmp/cpython /tmp/cpython.tar.gz

# Bootstrap uv (aliyun index) and create the venv with the built interpreter.
RUN python3 -m pip install --no-cache -i https://mirrors.aliyun.com/pypi/simple uv && \
    uv venv /opt/venv --python=/usr/local/bin/python${PYTHON_VERSION}

RUN python3 --version && \
    uv self version

# uv settings shared by all downstream stages (same as upstream vLLM)
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
# Use copy mode to avoid hardlink failures with Docker cache mounts
ENV UV_LINK_MODE=copy

WORKDIR /workspace
#################### BASE BUILD IMAGE ####################

#################### MACA SDK IMAGE ####################
FROM base AS maca

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        vim git libopenblas-dev make cmake ninja-build gcc g++ procps \
        libibverbs1 librdmacm1 libibumad3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# The Metax-Driver provides only the optional mx-smi management tool and is
# not available as an Ubuntu package on this network, so it is skipped here.

ARG MACA_VERSION
ARG CU_BRIDGE_VERSION

# Install the vendored MACA SDK deb bundle (3.8.2.2). The bundle includes
# mctlassEx, which torch-metax links against (libmctlassEx.so).
COPY docker/maca-debs/ /tmp/maca-debs/
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y /tmp/maca-debs/*.deb && \
    rm -rf /tmp/maca-debs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# torch-metax dlopens libelf.so.1 (and other libs) when importing torch;
# Ubuntu images do not ship it by default.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libelf1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Build cu-bridge from the vendored source (gitee.com is unreachable; the
# archive is at docker/cu-bridge/<version>.zip with top-level dir
# cu-bridge-<version>/). Update the COPY path when bumping CU_BRIDGE_VERSION.
COPY docker/cu-bridge/3.8.2.2.zip /tmp/cu-bridge.zip
RUN cd /tmp/ && \
    export MACA_PATH=/opt/maca && \
    cp /tmp/cu-bridge.zip ${CU_BRIDGE_VERSION}.zip && \
    unzip ${CU_BRIDGE_VERSION}.zip && \
    mv cu-bridge-${CU_BRIDGE_VERSION} cu-bridge && \
    chmod 755 cu-bridge -Rf && \
    cd cu-bridge && \
    mkdir build && cd ./build && \
    cmake -DCMAKE_INSTALL_PREFIX=/opt/maca/tools/cu-bridge ../ && \
    make && make install
#################### MACA SDK IMAGE ####################

#################### WHEEL BUILD IMAGE ####################
FROM maca AS build

ARG UV_INDEX_URL
ARG UV_EXTRA_INDEX_URL
ARG UV_TRUSTED_HOST
ENV UV_INDEX_URL=${UV_INDEX_URL}
ENV UV_EXTRA_INDEX_URL=${UV_EXTRA_INDEX_URL}
ENV UV_TRUSTED_HOST=${UV_TRUSTED_HOST}

## Update environment variables for the MACA toolchain
ENV MACA_PATH=/opt/maca
ENV MACA_CLANG_PATH=/opt/maca/mxgpu_llvm/bin
# cu-bridge
ENV CUCC_PATH="${MACA_PATH}/tools/cu-bridge"
ENV CUDA_PATH=/root/cu-bridge/CUDA_DIR
ENV CUCC_CMAKE_ENTRY=2
# update PATH
ENV PATH=/opt/mxdriver/bin:${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/tools/cu-bridge/tools:${MACA_PATH}/tools/cu-bridge/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/mxdriver/lib:${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/ucx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

ARG MAX_JOBS
ARG CMAKE_BUILD_TYPE

WORKDIR /workspace

# Install vllm-metax build dependencies
COPY requirements/build.txt requirements/build.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements/build.txt

# numpy and pymxsml are required while building the wheel
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install numpy==1.26.4 /opt/maca/share/mxsml/pymxsml-*.whl

# Build the wheel from the repository source. The source is bind-mounted rw so
# the CMake FetchContent state in .deps stays within this stage.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,src=.,target=/workspace/vllm-metax,rw \
    if [ -n "${MAX_JOBS}" ]; then export MAX_JOBS="${MAX_JOBS}"; fi; \
    if [ -n "${CMAKE_BUILD_TYPE}" ]; then export CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"; fi; \
    cd /workspace/vllm-metax && \
    uv build --wheel --out-dir=/workspace/vllm_metax_wheel_dist

# Record the wheel checksum so downstream stages can bust their layer cache
# when the wheel changes, without copying the wheel itself into the image.
RUN sha256sum /workspace/vllm_metax_wheel_dist/*.whl \
    > /workspace/vllm_metax_wheel_dist/wheel.sha256
#################### WHEEL BUILD IMAGE ####################

#################### CLEANUP IMAGE ####################
FROM maca AS clean

# Slim down the MACA SDK before it is copied into the final image: remove the
# report/flashinfer/tests packages (tolerate absence) and static libraries.
RUN dpkg -r --force-all mcflashattn mcflashinfer mxreport mccltests 2>/dev/null || true; \
    find /opt/maca/ -type f -name "*.a" -delete && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/*
#################### CLEANUP IMAGE ####################

#################### vLLM-METAX INSTALLATION IMAGE ####################
FROM base AS vllm-base

ARG VLLM_VERSION
ARG UV_INDEX_URL
ARG UV_EXTRA_INDEX_URL
ARG UV_TRUSTED_HOST
ARG VLLM_BUILD_COMMIT
ARG VLLM_BUILD_PIPELINE
ARG VLLM_BUILD_URL
ARG VLLM_IMAGE_TAG

ENV MACA_PATH=/opt/maca
ENV PATH=/opt/mxdriver/bin:${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/tools/cu-bridge/tools:${MACA_PATH}/tools/cu-bridge/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/mxdriver/lib:${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/ucx/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        gcc binutils procps \
        libibverbs1 librdmacm1 libibumad3 \
        libopenblas0 libnuma1 libelf1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=clean /opt/maca /opt/maca

WORKDIR /workspace

ENV UV_EXTRA_INDEX_URL=${UV_EXTRA_INDEX_URL}
ENV UV_INDEX_URL=${UV_INDEX_URL}
ENV UV_TRUSTED_HOST=${UV_TRUSTED_HOST}

# The metax-only wheels (torch, flash_attn, apex, ...) are not published on
# the indexes reachable from this network; they are vendored in docker/wheels/
# (official maca 3.8.2.2 bundle, cp312) and served via find-links. pymxsml
# ships inside the MACA SDK.
COPY docker/wheels/ /opt/wheels/
RUN cp /opt/maca/share/mxsml/pymxsml-*.whl /opt/wheels/
ENV UV_FIND_LINKS=/opt/wheels

# Install vllm-metax. Its baked-in pins (metax3.8.2.2) match the vendored maca
# 3.8.2.2 wheels; the plugin wheel is installed with --no-deps and its
# dependencies are installed explicitly from /opt/wheels in the next step.
COPY --from=build /workspace/vllm_metax_wheel_dist/wheel.sha256 /tmp/vllm_metax-wheel.sha256
RUN --mount=type=bind,from=build,src=/workspace/vllm_metax_wheel_dist,target=/tmp/wheels \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps /tmp/wheels/*.whl

# Install the vendored metax dependency wheels explicitly (maca 3.8.2.2).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install /opt/wheels/*.whl

# Build a locally resolved override from the repository's maca_private.txt
# (committed pins: metax3.8.2.2; available wheels: 3.8.2.2). Only the in-image
# copy is rewritten; the repository file is never modified.
COPY requirements/maca_private.txt /tmp/maca_private.txt
RUN sed -e 's/metax3\.8\.2\.0/metax3.8.2.2/g' \
        -e '/^deep_ep==/d' -e '/^deep_gemm==/d' -e '/^maca-python==/d' \
        -e '/^mcoplib==/d' -e '/^mcpti-python==/d' -e '/^lmcache==/d' \
        -e '/^nixl==/d' -e '/^mooncake-transfer-engine==/d' \
        -e '/^--extra-index-url/d' \
        /tmp/maca_private.txt > /tmp/maca_private.resolved.txt

# Install empty vllm (no CUDA/MACA kernels) so the vllm-metax plugin can hook
# into it at runtime.
RUN --mount=type=cache,target=/root/.cache/uv \
    VLLM_TARGET_DEVICE=empty \
    UV_OVERRIDE=/tmp/maca_private.resolved.txt \
    uv pip install --no-binary=vllm vllm==${VLLM_VERSION}

# Fix(hank): vllm installation also brings in flashinfer-python and
# cupy-cuda12x, remove them here.
RUN uv pip uninstall flashinfer-python cupy-cuda12x

# Fix(hank): torch-metax still uses numpy<2
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install numpy==1.26

ENV VLLM_USAGE_SOURCE=production-docker-image
ENV VLLM_BUILD_COMMIT=${VLLM_BUILD_COMMIT:-unknown} \
    VLLM_BUILD_PIPELINE=${VLLM_BUILD_PIPELINE:-local} \
    VLLM_BUILD_URL=${VLLM_BUILD_URL:-} \
    VLLM_IMAGE_TAG=${VLLM_IMAGE_TAG:-local/vllm-metax-ubuntu:dev}
LABEL org.opencontainers.image.source="https://github.com/MetaX-MACA/vLLM-metax" \
      org.opencontainers.image.revision="${VLLM_BUILD_COMMIT}" \
      org.opencontainers.image.version="${VLLM_IMAGE_TAG}" \
      org.opencontainers.image.url="${VLLM_BUILD_URL}" \
      ai.vllm-metax.build.commit="${VLLM_BUILD_COMMIT}" \
      ai.vllm-metax.build.pipeline="${VLLM_BUILD_PIPELINE}" \
      ai.vllm-metax.build.url="${VLLM_BUILD_URL}" \
      ai.vllm-metax.image.tag="${VLLM_IMAGE_TAG}"
#################### vLLM-METAX INSTALLATION IMAGE ####################

#################### OPENAI API SERVER ####################
# Default target: starts `vllm serve` like the upstream vllm-openai image.
FROM vllm-base AS vllm-openai

ENTRYPOINT ["vllm", "serve"]
#################### OPENAI API SERVER ####################
