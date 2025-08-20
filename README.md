# vLLM-MetaX

The vLLM-MetaX backend plugin for vLLM.

## Installation

Currently we only support build from source:

### install vllm
```bash
pip install vllm==0.9.1 --no-deps
```

### install vllm-metax

preparations for vllm-metax initialization:
```bash
# clone vllm-metax
git clone  --depth 1 --branch [branch-name] [vllm-metax-repo-url]
cd vllm-metax

# setup env
source env.sh
```

There are two ways to build the plugin, you could *build and install* the plugin, use pip as:

```bash
pip install -r requirements/build.txt
# since we use our local pytorch, add the --no-build-isolation flag 
# to avoid the conflict with the official pytorch
pip install . -v --no-build-isolation
```

Or, if you want to build the binary distribution `.whl`:

```bash
python setup.py bdist_wheel
pip install dist/*.whl
```

> ***Note***: plugin would copy the `.so` files to the vllm_dist_path, which is the `vllm` under `pip show vllm | grep Location` by default.
>
> If you :
>
> - ***Skipped the build step*** and installed the binary distribution `.whl` from somewhere else(e.g. pypi).
>
> - Or ***reinstalled*** the official vllm
>
> You need **manually** executing the following command to initialize the plugin after the plugin installation:

```bash
$ vllm_metax_init
```