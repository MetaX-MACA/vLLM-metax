ARG BUILD_BASE_IMAGE=registry.access.redhat.com/ubi9/ubi:9.6
ARG PYTHON_VERSION=3.12
ARG UV_EXTRA_INDEX_URL=https://console.redhat.com/api/pypi/public-rhai/rhoai/3.4/cpu-ubi9/simple
ARG UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
# may need passing a particular vllm version during build
ARG VLLM_VERSION

#################### BASE IMAGE ####################
FROM ${BUILD_BASE_IMAGE} AS base

WORKDIR /workspace

# Install minimal dependencies and uv
RUN --mount=type=cache,target=/var/cache/dnf,sharing=locked \
    yum makecache && \
    yum install -y --setopt=install_weak_deps=False \
        sudo \
        git \
        wget \
        curl \
        ca-certificates \
        gcc \
        gcc-c++ \
        gcc-gfortran \
        make \
        ccache \
        python3 \
        python3-devel \
        openssl-devel \
        libffi-devel \
        zlib-devel \
        numactl-libs \
        numactl-devel \
        libSM \
        libXext \
        mesa-libGL \
        jq \
        lsof \
        which \
        findutils \
        tar \
        gzip \
        bzip2 \
        patch \
        procps-ng \
        pkgconf-pkg-config \
        openblas openblas-devel \
        libtiff libtiff-devel \
        openjpeg2 openjpeg2-devel \
        zeromq zeromq-devel \
        ffmpeg ffmpeg-libs \
        xz \
    && yum clean all && \
    curl -LsSf https://astral.sh/uv/install.sh | sh

ENV CC=/usr/bin/gcc CXX=/usr/bin/g++

ENV PATH="/root/.local/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ARG PYTHON_VERSION
RUN uv venv --python ${PYTHON_VERSION} --seed ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ENV UV_HTTP_TIMEOUT=500

ENV LD_PRELOAD="libtcmalloc_minimal.so.4:/opt/venv/lib/libiomp5.so"

RUN echo 'ulimit -c 0' >> ~/.bashrc

###################### BUILD IMAGE ####################
FROM base AS vllm-cpu-build


# Install Python dependencies
ARG UV_EXTRA_INDEX_URL
ARG UV_INDEX_URL
ENV UV_EXTRA_INDEX_URL=${UV_EXTRA_INDEX_URL}
ENV UV_INDEX_URL=${UV_INDEX_URL}
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE="copy"

ARG VLLM_VERSION
RUN uv pip install vllm==${VLLM_VERSION}

