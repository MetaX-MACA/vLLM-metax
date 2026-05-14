# Current directory should be the root of the repository

# You may need to pass 
#   - VLLM_VERSION
#   - MACA_VERSION
# to build a specific version

RHSM_SECRET_FILE=${RHSM_SECRET_FILE:-rhsm.env}
if [ ! -f "$RHSM_SECRET_FILE" ] && [ -f docker/rhsm.env ]; then
    RHSM_SECRET_FILE=docker/rhsm.env
fi

DOCKER_BUILDKIT=1 docker build \
    --network host \
    --secret id=rhsm,src=${RHSM_SECRET_FILE} \
    -f docker/vllm-cpu.Dockerfile \
    -t vllm_cpu:v0 \
    --build-arg VLLM_VERSION=v0.18.0 \
     .

# debug dockerfile and run into shell with buildx:
# ddocker () {
#     BUILDX_EXPERIMENTAL=1 docker buildx debug --invoke /bin/bash --on=error $@
# }

