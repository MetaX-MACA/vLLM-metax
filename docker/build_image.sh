# Current directory should be the root of the repository

# You may need to pass 
#   - VLLM_VERSION
#   - MACA_VERSION
# to build a specific version

docker build \
    --network host \
    -f docker/vllm_metax.Dockerfile \
    -t vllm_metax:v0 \
    --build-arg VLLM_VERSION=v0.14.0 \
    --build-arg MACA_VERSION=3.3.0 \
     .

# debug dockerfile and run into shell with buildx:
# ddocker () {
#     BUILDX_EXPERIMENTAL=1 docker buildx debug --invoke /bin/bash --on=error $@
# }

