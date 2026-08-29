ARG BUILD_BASE_IMAGE=registry.redhat.io/rhai/base-image-cpu-rhel9:3.4.0-1777399554
ARG PYTHON_VERSION=3.12
# ARG UV_EXTRA_INDEX_URL=https://console.redhat.com/api/pypi/public-rhai/rhoai/3.4/cpu-ubi9/simple
ARG UV_INDEX_URL=https://console.redhat.com/api/pypi/public-rhai/rhoai/3.4/cpu-ubi9/simple
# may need passing a particular vllm version during build
ARG VLLM_VERSION

ARG RHSM_USER
ARG RHSM_PASS

#################### BASE IMAGE ####################
FROM ${BUILD_BASE_IMAGE} AS base


# # TODO: we most likely do not need all of the dnf installs below: they're already in the base image
# RUN --mount=type=secret,id=rhsm \
#     --mount=type=cache,target=/var/cache/dnf,sharing=locked \
#     bash -euo pipefail -c '\
#       source /run/secrets/rhsm; \
#       cleanup() { \
#         subscription-manager unregister >/dev/null 2>&1 || true; \
#         subscription-manager clean >/dev/null 2>&1 || true; \
#       }; \
#       trap cleanup EXIT; \
#       subscription-manager register \
#         --username "$RHSM_USER" \
#         --password "$RHSM_PASS" \
#         --auto-attach; \
#       subscription-manager repos \
#         --enable codeready-builder-for-rhel-9-x86_64-rpms; \
#       yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm; \
#       /usr/bin/crb enable; \
#       yum makecache; \
#       yum install -y --setopt=install_weak_deps=False \
#         python3 \
#         python3-devel \
#         zeromq \
#         bzip2 \
#         cpio \
#         elfutils-debuginfod-client \
#         ffmpeg-free \
#         fftw \
#         file \
#         freetype \
#         gcc \
#         gcc-c++ \
#         gdal-libs \
#         gdb \
#         geos \
#         git-core \
#         glibc-langpack-en \
#         glog \
#         gmp \
#         gzip \
#         hdf5 \
#         jemalloc \
#         jq \
#         krb5-libs \
#         lcms2 \
#         libaio \
#         libev \
#         libjpeg \
#         libmpc \
#         libomp \
#         libpng \
#         libpq \
#         libqhull_r \
#         libsndfile \
#         libtiff \
#         libunwind \
#         libva \
#         libwebp \
#         libxml2 \
#         libxslt \
#         libzip \
#         libzstd \
#         loguru \
#         lz4 \
#         make \
#         mariadb-connector-c \
#         mpfr \
#         netcdf \
#         numactl \
#         nvtop \
#         openblas openblas-openmp openblas-openmp64 openblas-serial openblas-serial64 openblas-threads openblas-threads64 \
#         openjpeg2 \
#         openmpi \
#         proj \
#         protobuf \
#         qpdf \
#         re2 \
#         snappy \
#         spatialindex \
#         tbb \
#         tesseract \
#         thrift \
#         unixODBC \
#         utf8proc \
#         wget \
#         xz \
#         xz-libs \
#         zlib \
#         zstd; \
#       yum clean all'

WORKDIR /workspace

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV CC=/usr/bin/gcc CXX=/usr/bin/g++

ENV PATH="/root/.local/bin:$PATH"
ENV VIRTUAL_ENV="/opt/app-root/"
ENV UV_PYTHON_INSTALL_DIR=/opt/app-root/python
ARG PYTHON_VERSION
# RUN uv venv  ${VIRTUAL_ENV} --python ${PYTHON_VERSION} --seed
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ENV UV_HTTP_TIMEOUT=500

ENV LD_PRELOAD="/opt/app-root/lib/libiomp5.so"

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

ENV VLLM_LOGGING_LEVEL=DEBUG

ARG VLLM_VERSION
RUN uv pip install --torch-backend=cpu vllm==${VLLM_VERSION}
