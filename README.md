# vLLM-MetaX

The MXMACA backend plugin for vLLM.

## Prepare Building Environments

vllm-metax plugin needs to be built with corresponding **Maca Toolkits**.

### manually

Checking and install all the vLLM environment requirements [here](https://developer.metax-tech.com/softnova/category?package_kind=AI&dimension=metax&chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85&ai_frame=vllm&ai_label=vLLM):

<img width="1788" height="183" alt="image" src="https://github.com/user-attachments/assets/df1c30bd-e2f9-41a9-a1b2-256291edc618" />

You could also update *Maca Toolkits* separately with specific version by:

- *MetaX Driver*: [*online*](https://developer.metax-tech.com/softnova/download?package_kind=Driver&dimension=metax&chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85) or [*offline*](https://developer.metax-tech.com/softnova/download?package_kind=Driver&dimension=metax&chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85)

- *Maca SDK*: [*online*](https://developer.metax-tech.com/softnova/download?package_kind=SDK&dimension=metax&chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85) or [*offline*](https://developer.metax-tech.com/softnova/download?package_kind=SDK&dimension=metax&chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&deliver_type=%E5%88%86%E5%B1%82%E5%8C%85)

### or... docker images

Directly using docker images released on [*MetaX Develop Community*](https://developer.metax-tech.com/softnova/docker).
All the required components are pre-installed in docker image.

> Note: You may need to search docker images for `vllm` distribution.

***Belows is version mapping to released plugin and maca***:

| plugin version | maca version | docker distribution tag |
|:--------------:|:------------:|:-----------------------:|
|v0.8.5          |maca2.33.1.13 | vllm:maca.ai2.33.1.13-torch2.6-py310-ubuntu22.04-amd64 |
|v0.9.1          |maca3.0.0.5   | vllm:maca.ai3.0.0.5-torch2.6-py310-ubuntu22.04-amd64 |
|v0.10.1.1 (dev only)|maca3.0.0.5(dev only)| vllm:maca.ai3.0.0.5-torch2.6-py310-ubuntu22.04-amd64 (dev only)|

> Note: All the vllm tests are based on the related maca version. Using incorresponding version of maca for vllm may cause unexpected bugs or errors. This is not garanteed.

## Install plugin

### install vllm
```bash
pip install vllm==0.10.1.1 --no-deps
```

### install vllm-metax

> Currently we only support build from source.

**clone repository**:
```bash
# clone vllm-metax
git clone  --depth 1 --branch [branch-name] [vllm-metax-repo-url] && cd vllm-metax
```

**install cuda toolkit**:

```bash
# build vllm on maca needs cuda 11.6
wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run && \
    sh cuda_11.6.0_510.39.01_linux.run --silent --toolkit && \
    rm cuda_11.6.0_510.39.01_linux.run
```

**setup env variables**:

```
# setup MACA path
DEFAULT_DIR="/opt/maca"
export MACA_PATH=${1:-$DEFAULT_DIR}

# setup CUDA && cu-bridge
export CUDA_PATH=/usr/local/cuda
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge

# update PATH
export PATH=${CUDA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}

export VLLM_INSTALL_PUNICA_KERNELS=1
```

There are two ways to build the plugin:

- if you want to build the binary distribution :

```bash
# install requirements for building
pip install -r requirements/build.txt
# build wheels
python setup.py bdist_wheel
# install wheels
pip install dist/*.whl
```

- Or, you could *build and install* the plugin via `pip`:

```bash
# since we use our local pytorch, add the --no-build-isolation flag 
# to avoid the conflict with the official pytorch
pip install . -v --no-build-isolation
```

> ***Note***: plugin would copy the `.so` files to the vllm_dist_path, which is the `vllm` under `pip show vllm | grep Location` by default.
>
> If you :
>
> - ***Skipped the building step*** and installed the binary distribution `.whl` from somewhere else(e.g. pypi).
>
> - Or ***reinstalled*** the official vllm
>
> You need **manually** executing the following command to initialize the plugin after the plugin installation:

```bash
$ vllm_metax_init

```
