ARG BUILD_BASE_IMAGE=registry.access.redhat.com/ubi9/ubi:9.6
ARG PYTHON_VERSION=3.10
ARG PIP_INDEX_URL
ARG PIP_EXTRA_INDEX_URL=https://repos.metax-tech.com/r/maca-pypi/simple
ARG UV_TRUSTED_HOST=repos.metax-tech.com
ARG UV_INDEX_URL=${PIP_INDEX_URL}
ARG UV_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}
ARG CU_BRIDGE_VERSION=3.1.0

#################### BASE BUILD IMAGE ####################
FROM ${BUILD_BASE_IMAGE} AS base

ARG PYTHON_VERSION
ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG UV_INDEX_URL=${PIP_INDEX_URL}
ARG UV_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}
ARG UV_TRUSTED_HOST

# Install system dependencies and uv, then create Python virtual environment
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    $HOME/.local/bin/uv venv /opt/venv --python ${PYTHON_VERSION} && \
    rm -f /usr/bin/python3 /usr/bin/python3-config /usr/bin/pip && \
    ln -s /opt/venv/bin/python3 /usr/bin/python3 && \
    ln -s /opt/venv/bin/python3-config /usr/bin/python3-config && \
    ln -s /opt/venv/bin/pip /usr/bin/pip && \
    python3 --version && python3 -m pip --version

# Activate virtual environment and add uv to PATH
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="$VIRTUAL_ENV/bin:/root/.local/bin:$PATH"

ENV UV_INDEX_STRATEGY="unsafe-best-match"

# Use copy mode to avoid hardlink failures with Docker cache mounts
ENV UV_LINK_MODE=copy


WORKDIR /workspace

# install build and runtime dependencies
COPY requirements/common.txt requirements/common.txt
COPY requirements/maca.txt requirements/maca.txt
COPY requirements/maca_private.txt requirements/maca_private.txt
COPY requirements/constraints.txt requirements/constraints.txt

RUN uv pip install --python /opt/venv/bin/python3 -r requirements/maca.txt \
    --extra-index-url ${UV_EXTRA_INDEX_URL} --trusted-host ${UV_TRUSTED_HOST}

# The following packages need subscription, so we just SKIP them. 
# `lapack-devel librdmacm-utils libibverbs-utils`

RUN yum makecache && yum install -y \
    wget zip unzip tar tzdata vim git \
    openblas-devel make cmake patch \
    ninja-build gcc gcc-c++ \
    procps-ng libxml2 libXau \
    libibverbs librdmacm libibumad \
    && yum clean all

# Installing Metax-Driver
RUN printf "[metax-centos]\n\
name=Maca Driver Yum Repository\n\
baseurl=https://repos.metax-tech.com/r/metax-driver-centos-$(uname -m)/\n\
enabled=1\n\
gpgcheck=0" > /etc/yum.repos.d/metax-driver-centos.repo

# would install the newest 3.1.0.x release
# Metax-Driver mainly contains vbios and kmd file, which are not needed in a container.
# Here we want to get the mx-smi management tool. 
# kernel version mismatch errors are ignored
RUN yum makecache && \
    yum install -y metax-driver mxgvm && \
    yum clean all && rm -rf /var/cache/yum /tmp/*

# Installing MACA SDK
RUN printf "[maca-sdk]\n\
name=Maca Sdk Yum Repository\n\
baseurl=https://repos.metax-tech.com/r/maca-sdk-rpm-$(uname -m)/\n\
enabled=1\n\
gpgcheck=0" > /etc/yum.repos.d/maca-sdk-rpm.repo

RUN yum makecache && \
    yum install -y maca_sdk && \
    yum clean all && rm -rf /var/cache/yum /tmp/*

## Install cu-bridge
ARG CU_BRIDGE_VERSION
RUN cd /tmp/ && \
    export MACA_PATH=/opt/maca && \
    curl -o ${CU_BRIDGE_VERSION}.zip -LsSf https://gitee.com/metax-maca/cu-bridge/repository/archive/${CU_BRIDGE_VERSION}.zip && \
    unzip ${CU_BRIDGE_VERSION}.zip && \
    mv cu-bridge-${CU_BRIDGE_VERSION} cu-bridge && \
    chmod 755 cu-bridge -Rf && \
    cd cu-bridge && \
    mkdir build && cd ./build && \
    cmake -DCMAKE_INSTALL_PREFIX=/opt/maca/tools/cu-bridge ../ && \
    make && make install

## Update environment variables
# setup MACA path
ENV MACA_PATH=/opt/maca
ENV MACA_CLANG_PATH=/opt/maca/mxgpu_llvm/bin 
# cu-bridge
ENV CUCC_PATH="${MACA_PATH}/tools/cu-bridge"
ENV CUDA_PATH=/root/cu-bridge/CUDA_DIR
ENV CUCC_CMAKE_ENTRY=2
# update PATH
ENV PATH=/opt/mxdriver/bin:${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/tools/cu-bridge/tools:${MACA_PATH}/tools/cu-bridge/bin:${PATH} 
ENV LD_LIBRARY_PATH=/opt/mxdriver/lib:${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/ucx/lib:${LD_LIBRARY_PATH}
# vllm compile option
ENV VLLM_INSTALL_PUNICA_KERNELS=1
#################### BASE BUILD IMAGE ####################


#################### WHEEL BUILD IMAGE ####################
FROM base AS build

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL


RUN uv pip install numpy==1.26.4
RUN uv pip install /opt/maca/share/mxsml/pymxsml-*.whl

# install vllm (or build from source)
RUN git clone --depth 1 --branch v0.11.0 https://github.com/vllm-project/vllm && \
    cd vllm && \
    python use_existing_torch.py && \
    uv pip install -r requirements/build.txt && \
    VLLM_TARGET_DEVICE=empty python -m build -w -n && \
    mkdir -p /workspace/wheels/ && \
    cp dist/vllm-*.whl /workspace/wheels/ && \
    cd .. && rm -rf vllm

# install vllm-metax
COPY requirements/build.txt requirements/build.txt

COPY . vllm-metax
WORKDIR /workspace/vllm-metax
RUN uv pip install -r requirements/build.txt && \
    python -m build -w && \
    mkdir -p /workspace/wheels/ && \
    cp dist/vllm_metax-*.whl /workspace/wheels/

# We need this to copy .so files to vllm's location
# Remove when master support (might be v0.11.1)
RUN vllm_metax_init


## Install ray

# Currently, skipped ray patch

# COPY ray-patch.zip /tmp/
# RUN cd /tmp && unzip ray-patch.zip && \
#     mv ray-patch /workspace && \
#     cd /workspace/ray-patch/ray_patch && \
#     source $HOME/.local/bin/env && \
#     source /opt/venv/bin/activate && \
#     python apply_ray_patch.py mx_ray_2.46.batch && \
#     if [ -f "/opt/conda/bin/ray" ]; then ln -sf /opt/conda/bin/ray /bin/ray; fi
#################### WHEEL BUILD IMAGE ####################


#################### FINAL IMAGE ####################
FROM base AS final

WORKDIR /workspace

COPY --from=build /workspace/wheels/ /workspace/wheels/
RUN uv pip install wheels/vllm-*.whl && \
    uv pip install wheels/vllm_metax-*.whl
#################### FINAL IMAGE ####################