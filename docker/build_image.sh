# Current directory should be the root of the repository

# You may need to pass VLLM_VERSION to build a specific version
docker build \
    --network host \
    -f docker/vllm_metax.Dockerfile \
    -t vllm_metax:v0 \
    --build-arg VLLM_VERSION=v0.10.2 \
    --build-arg MACA_VERSION=3.2.1 \
     .

# BUILDX_EXPERIMENTAL=1 docker buildx debug --invoke /bin/bash --on=error build \
#     --network host \
#     -f docker/vllm_metax.Dockerfile \
#     -t vllm_metax:v1 \
#     --build-arg VLLM_VERSION=v0.10.2 \
#     --build-arg MACA_VERSION=3.2.1 \
#     .
