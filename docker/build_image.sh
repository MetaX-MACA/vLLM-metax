#!/usr/bin/env bash
# Build the Ubuntu 22.04 based vLLM-MetaX OpenAI image.
# Current directory should be the root of the repository.
#
# Environment notes:
#   - Uses the per-user `lli-build` builder (docker-container driver) which is
#     configured to trust the internal image mirror.
#   - The corporate proxy is passed by IP because build containers cannot
#     resolve the proxy hostname.
#   - The MACA SDK deb bundle, the metax python wheels (cp312) and cu-bridge
#     are vendored under docker/maca-debs/, docker/wheels/ and docker/cu-bridge/.
#     Those bundles are NOT tracked in git (large vendor artifacts), so they must
#     be provided locally before building. Bump the versions below together with
#     the bundles.
#   - TODO(v0.29.0): confirm the MACA SDK / cu-bridge bundle versions for the
#     v0.29.0 release; 3.8.2.2 is the v0.28.0 bundle and is shared with the
#     0.29 line's metax3.8.2.2 wheels (see requirements/maca_private.txt).

set -euo pipefail

IMAGE_TAG="${1:-vllm_metax:v0.29.0-maca3.8-ubuntu}"

docker buildx build \
    --builder lli-build \
    --platform linux/amd64 \
    -f docker/vllm_metax.openai.ubuntu.Dockerfile \
    -t "${IMAGE_TAG}" \
    --load \
    --build-arg VLLM_VERSION=0.29.0 \
    --build-arg MACA_VERSION=3.8 \
    --build-arg CU_BRIDGE_VERSION=3.8.2.2 \
    --build-arg PYTHON_VERSION=3.12 \
    --build-arg BUILD_BASE_IMAGE=registry-docker.hub.metax-tech.com/library/ubuntu:22.04 \
    .

    # proxy for metax
    # --build-arg http_proxy=http://10.2.192.22:1080 \
    # --build-arg https_proxy=http://10.2.192.22:1080 \
    # --build-arg HTTP_PROXY=http://10.2.192.22:1080 \
    # --build-arg HTTPS_PROXY=http://10.2.192.22:1080 \
    # --build-arg no_proxy=localhost,127.0.0.1,::1 \
    # --build-arg NO_PROXY=localhost,127.0.0.1,::1 \

echo
echo "Image ready: ${IMAGE_TAG}"
echo "Run with: docker run --gpus all -p 8000:8000 ${IMAGE_TAG} --model <model_dir>"
