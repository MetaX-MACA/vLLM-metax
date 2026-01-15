Here we demonstrate how to build vLLM manually starting from a ubi9 docker image.

# 0. start a base image container

## start container
```bash
# you may modify privileged option and mount only specific GPU cards.
# please refer to our docucments on https://developer.metax-tech.com
docker run -d -it --net=host --uts=host --ipc=host --privileged=true --group-add video  \
    --shm-size 100gb --ulimit memlock=-1 \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    --device=/dev/dri --device=/dev/mxcd \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime:ro \
    --name base_image \
    registry.access.redhat.com/ubi9/ubi:9.6 bash
```
Some packages needs subscription, in the following steps we ignore these packages.



# 1. Installing Metax-Driver

Reference: [MACA Repo](https://repos.metax-tech.com/gitea/repos/index/wiki/MACA.md#metax-driver)

```bash
# add repo source
cat <<EOF> /etc/yum.repos.d/metax-driver-centos.repo 
[metax-centos]
name=Maca Driver Yum Repository
baseurl=https://repos.metax-tech.com/r/metax-driver-centos-$(uname -m)/
enabled=1
gpgcheck=0
EOF

# would install the newest 3.1.0.x release
# Metax-Driver mainly contains vbios and kmd file, which are not needed in a container.
# Here we want to get the mx-smi management tool. 
# kernel version mismatch errors are ignored
yum makecache
yum install -y metax-driver mxgvm

# check
rpm -qa | egrep "(metax|mxsmt|mxfw)"
```


# 2. Installing MACA SDK

Reference: [MACA Repo](https://repos.metax-tech.com/gitea/repos/index/wiki/MACA.md#metax-driver)

may need some time according to your network speed

```bash
cat <<EOF> /etc/yum.repos.d/maca-sdk-rpm.repo 
[maca-sdk]
name=Maca Sdk Yum Repository
baseurl=https://repos.metax-tech.com/r/maca-sdk-rpm-$(uname -m)/
enabled=1
gpgcheck=0
EOF

yum makecache

yum install -y maca_sdk

rpm -qa | egrep "maca_sdk"

# you may install specific MACA SDK version like this
yum --showduplicates list |grep maca_sdk
yum install maca_sdk-<version>-<release>
```


# 3. Install torch

Reference: [MACA Repo](https://repos.metax-tech.com/gitea/repos/index/wiki/MACA.md#metax-driver)


## 3.1 Setup a python environment using uv

Our internal build pipeline uses conda. Here we use uv instead. 

Attention: NOT fully tested!

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv /opt/venv --python 3.10
source /opt/venv/bin/activate
```


## 3.2 Install dependent yum packages

The following packages need subscription, so we just SKIP them. It is about lapack and IB/RDMA utils, NOT fully tested, but should not cause problems.

* `lapack-devel librdmacm-utils libibverbs-utils`

```bash
yum makecache && yum install -y \
    openblas-devel \
    gcc-c++ \
    libibverbs librdmacm libibumad openssh-server \
    && yum clean all
```

## 3.3 Install cu-bridge

Cu-bridge is our cuda compatiable package used to compile cuda code. Before installing torch, cu-bridge need to be installed.

Please refer to [cu-bridge/02_User_Manual](https://gitee.com/metax-maca/cu-bridge/tree/master/docs/02_User_Manual)


```bash
# you may separate the building process in a single stage.

yum install -y wget unzip cmake

export MACA_PATH=/opt/maca

wget https://gitee.com/metax-maca/cu-bridge/repository/archive/3.1.0.zip
unzip 3.1.0.zip
mv cu-bridge-3.1.0 cu-bridge
chmod 755 cu-bridge -Rf
cd cu-bridge
mkdir build && cd ./build
cmake -DCMAKE_INSTALL_PREFIX=/opt/maca/tools/cu-bridge ../
make && make install
```


## 3.4 Some import environment settings

You need to set the following envs to make sure maca-pytorch running properly. The cucc parts are manily used for compiling.

```bash
export MACA_PATH=/opt/maca

export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin

export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
export CUDA_PATH=${CUCC_PATH}
export PATH=$PATH:${CUCC_PATH}/tools:${CUCC_PATH}/bin

export PATH=/opt/mxdriver/bin:${MACA_PATH}/bin:${MACA_CLANG_PATH}:${PATH}
export LD_LIBRARY_PATH=/opt/mxdriver/lib:${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/ucx/lib:${LD_LIBRARY_PATH}
```


## 3.5 Install torch using uv 

1. Currently in our internal CI/CD flow, we install from local wheel packages. Here we install packages from metax pypi source.
2. Metax maca version is added to package's local version, e.g., "+metax3.1.0.4torch2.6", and packages's version should match with the installed maca SDK version.
3. The only way to specify maca version is giving the full version name. It would be better to use seperate channels for different maca versions, which is in our plan.


```bash

# `datasets` only has 3.1.0 in metax's pypi repo. first install from pulbic source.
uv pip install datasets==4.1.1

cat <<EOF>  ./requirements.txt
apex==0.1+metax3.1.0.4
causal_conv1d==1.5.0.post8+metax3.1.0.4torch2.6
dropout_layer_norm==0.1+metax3.1.0.4torch2.6
flash_attn==2.6.3+metax3.1.0.4torch2.6
flash_linear_attention==0.1+metax3.1.0.4torch2.6
flash_mla==1.0.1+metax3.1.0.4torch2.6
flashinfer==0.2.2.post1+metax3.1.0.4torch2.6
fused_dense_lib==2.6.3+metax3.1.0.4torch2.6
mamba_ssm==2.2.4+metax3.1.0.4torch2.6
mctlassEx==0.1.1+metax3.1.0.4torch2.6
rotary_emb==0.1+metax3.1.0.4torch2.6
sageattention==2.0.1+metax3.1.0.4torch2.6
spconv==2.1.0+metax3.1.0.4torch2.6
torch==2.8.0+metax3.1.0.4
torchaudio==2.4.1+metax3.1.0.4
torchvision==0.15.1+metax3.1.0.4
triton==3.0.0+metax3.1.0.4
xentropy_cuda_lib==0.1+metax3.1.0.4torch2.6
xformers==0.0.22+metax3.1.0.4torch2.6
EOF

uv pip install -r ./requirements.txt -i https://repos.metax-tech.com/r/maca-pypi/simple --trusted-host repos.metax-tech.com

# torch need to use numpy < 2.0
uv pip install numpy==1.26.4
```

You may try the following commands:

```bash
# Search for avialable versions:
pip index versions mcspconv -i https://repos.metax-tech.com/r/maca-pypi/simple --trusted-host repos.metax-tech.com

# install a package
uv pip install torch torchaudio torchvision -i https://repos.metax-tech.com/r/maca-pypi/simple  --trusted-host repos.metax-tech.com
```

Check torch installing result:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count());"
```



# 4. Build and install vllm

Reference: [vLLM-metax](https://github.com/MetaX-MACA/vLLM-metax)


```bash
yum install -y vim zip wget tar tzdata \
    make cmake ninja-build gcc gcc-c++ procps-ng libxml2 openssh-server libXau \
    openblas-devel \
    libibverbs librdmacm libibumad \
    && yum clean all

uv pip install /opt/maca/share/mxsml/pymxsml-*.whl
uv pip install tokenizers==0.20.3 orjson==3.10.6


yum install -y git

# build or install vllm
uv pip install vllm==0.10.2 --no-deps


git clone  --depth 1 --branch v0.10.2 https://github.com/MetaX-MACA/vLLM-metax.git
cd vLLM-metax

# build vllm on maca needs cuda 11.6
wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run && \
    sh cuda_11.6.0_510.39.01_linux.run --silent --toolkit && \
    rm cuda_11.6.0_510.39.01_linux.run


# setup MACA path
export MACA_PATH="/opt/maca"

# setup CUDA && cu-bridge
export CUDA_PATH="/usr/local/cuda"
export CUCC_PATH="${MACA_PATH}/tools/cu-bridge"

# update PATH
export PATH=${CUDA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}

export VLLM_INSTALL_PUNICA_KERNELS=1


# install requirements for building
uv pip install -r requirements/build.txt
# build wheels
python setup.py bdist_wheel
# install wheels
uv pip install dist/*.whl

```


## 5. Install ray


```bash
yum install -y patch

uv pip install click==8.2.1

# the following packages cannot be installed. SKIP
# uv pip install mcpy==2.1.9.4+b3.1.0.14 numbax==2.1.9.4+b3.1.0.14 -i https://repos.metax-tech.com/r/maca-pypi/simple  --trusted-host repos.metax-tech.com

pip install ray==2.46.0

unzip ray-patch.zip
cp -rd ray-patch /workspace
cd /workspace/ray-patch/ray_patch
python apply_ray_patch.py mx_ray_2.46.batch

if [ -f "/opt/conda/bin/ray" ]; then
    ln -sf /opt/conda/bin/ray /bin/ray
fi
```
