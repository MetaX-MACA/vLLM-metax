# Current directory should be the root of the repository

# You may need to pass VLLM_VERSION to build a specific version
docker build --network host -f docker/vllm_metax.Dockerfile -t vllm_metax:v0 --build-arg VLLM_VERSION=v0.11.0 .