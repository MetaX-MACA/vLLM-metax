# Current directory should be the root of the repository
docker build --network host -f docker/vllm_metax.Dockerfile -t vllm_metax:v0 .