ARG BUILD_BASE_IMAGE=registry.access.redhat.com/ubi9/ubi:9.6
ARG PYTHON_VERSION=3.10
# ARG UV_INDEX_URL=https://repos.metax-tech.com/r/maca-pypi/simple
ARG UV_EXTRA_INDEX_URL=https://repos.metax-tech.com/r/maca-pypi/simple
ARG UV_TRUSTED_HOST=repos.metax-tech.com

# may need passing a particular vllm version during build
ARG VLLM_VERSION
ARG MACA_VERSION
ARG CU_BRIDGE_VERSION=${MACA_VERSION}

#################### BASE BUILD IMAGE ####################
FROM ${BUILD_BASE_IMAGE} AS base

ARG PYTHON_VERSION
# ARG UV_INDEX_URL
ARG UV_TRUSTED_HOST

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:/root/.local/bin:$PATH"
RUN dnf -y install python3-pip && \
    dnf clean all

RUN python3 -m pip install --no-cache uv && \
    uv venv /opt/venv --python=${PYTHON_VERSION}

RUN python3 --version && \
    uv self version


ENV UV_INDEX_STRATEGY="unsafe-best-match"

# Use copy mode to avoid hardlink failures with Docker cache mounts
ENV UV_LINK_MODE=copy
# ENV UV_INDEX_URL=${UV_INDEX_URL}
ENV UV_EXTRA_INDEX_URL=${UV_EXTRA_INDEX_URL}

WORKDIR /workspace

# install build and runtime dependencies and cache them
COPY requirements/common.txt requirements/common.txt
COPY requirements/maca.txt requirements/maca.txt
COPY requirements/maca_private.txt requirements/maca_private.txt
COPY requirements/constraints.txt requirements/constraints.txt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements/maca.txt \
    --trusted-host ${UV_TRUSTED_HOST}

# install empty vllm
ARG VLLM_VERSION
RUN --mount=type=cache,target=/root/.cache/uv \
    VLLM_TARGET_DEVICE=empty \
    UV_OVERRIDE=requirements/maca_private.txt \
    uv pip install --no-binary=vllm vllm==${VLLM_VERSION} \
    --trusted-host ${UV_TRUSTED_HOST}
#################### BASE BUILD IMAGE ####################

#################### Install MACA SDK and Metax-Driver ####################
FROM base AS full_maca

RUN yum makecache && yum install -y \
    unzip vim git openblas-devel make cmake \
    ninja-build gcc g++ procps-ng \
    libibverbs librdmacm libibumad \
    && yum clean all

ARG MACA_VERSION

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
    yum install -y metax-driver-${MACA_VERSION}* mxgvm && \
    yum clean all && rm -rf /var/cache/yum /tmp/*

# Installing MACA SDK
RUN printf "[maca-sdk]\n\
name=Maca Sdk Yum Repository\n\
baseurl=https://repos.metax-tech.com/r/maca-sdk-rpm-$(uname -m)/\n\
enabled=1\n\
gpgcheck=0" > /etc/yum.repos.d/maca-sdk-rpm.repo

RUN yum makecache && \
    yum install -y maca_sdk-${MACA_VERSION}* && \
    yum clean all && rm -rf /var/cache/yum /tmp/*

## Install cu-bridge
# CU_BRIDGE 3.2.1 has some bugs and can't work with MACA SDK 3.2.1 properly.
# So here we install CU_BRIDGE 3.1.0 instead.
ARG CU_BRIDGE_VERSION=3.1.0
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

#################### Install MACA SDK and Metax-Driver ####################


#################### WHEEL BUILD IMAGE ####################
FROM full_maca AS wheel_build

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

WORKDIR /workspace
ARG UV_EXTRA_INDEX_URL
ENV UV_EXTRA_INDEX_URL=${UV_EXTRA_INDEX_URL}

# install vllm-metax build dependencies
COPY requirements/build.txt requirements/build.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements/build.txt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install numpy==1.26.4 /opt/maca/share/mxsml/pymxsml-*.whl

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,src=.,target=/workspace/vllm-metax,rw \
    cd /workspace/vllm-metax && \
    uv build --wheel --out-dir=/workspace/vllm_metax_wheel_dist

#################### WHEEL BUILD IMAGE ####################


#################### CLEANUP ####################
FROM full_maca AS clean_maca

RUN rpm -e --nodeps \
        mcflashattn_${MACA_VERSION} \
        mcflashinfer_${MACA_VERSION} \
        mxreport-${MACA_VERSION} \
        mccltests-${MACA_VERSION} && \
    find /opt/maca/ -type f -name "*.a" -delete && \
    yum clean all && rm -rf /var/cache/yum /tmp/*

#################### CLEANUP ####################

#################### FINAL IMAGE ####################
FROM base AS final
ARG MACA_VERSION

ENV MACA_PATH=/opt/maca
ENV PATH=/opt/mxdriver/bin:${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/tools/cu-bridge/tools:${MACA_PATH}/tools/cu-bridge/bin:${PATH} 
ENV LD_LIBRARY_PATH=/opt/mxdriver/lib:${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/ucx/lib:${LD_LIBRARY_PATH}

RUN yum makecache && yum install -y \
    gcc \
    binutils \
    procps-ng \
    libibverbs \
    librdmacm \
    libibumad \
    openblas \
    numactl-libs \
    && yum clean all && rm -rf /var/cache/yum /tmp/*

COPY --from=clean_maca /opt/maca /opt/maca
COPY --from=clean_maca /opt/mxdriver /opt/mxdriver

WORKDIR /workspace
ARG UV_EXTRA_INDEX_URL
ENV UV_EXTRA_INDEX_URL=${UV_EXTRA_INDEX_URL}

# install vllm-metax from built wheels
COPY --from=wheel_build /workspace/vllm_metax_wheel_dist /tmp/wheels
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install /tmp/wheels/* 

# Fix(hank): don't know why vllm installation also brings in flashinfer-python, remove it here.
RUN uv pip uninstall flashinfer-python cupy-cuda12x

RUN ray-patch
#################### FINAL IMAGE ####################