# Recipes for installing packages necessary for AI on Frontier

## Megatron-DeepSpeed ported to Frontier
### FA2 Branch with FlashAttention 2
```
https://github.com/sajal-vt/Megatron-DeepSpeed-ORNL/tree/FA2
```
### Conda environment with APEX + FlashAttention 2
```
/lustre/orion/stf006/world-shared/irl1/flash2
```
### Training script for a GPT-like model
```
https://github.com/sajal-vt/Megatron-DeepSpeed-ORNL/blob/FA2/launch_gpt22b_bf16.slurm
```

## Recipe for installing APEX on Frontier
```
#!/bin/bash
module load rocm/5.7.0
module load cray-mpich/8.1.23
# source environment created with torch==2.2.0.dev20231002+rocm5.7 (current nightly pip install for rocm/5.7)
source megatron_env/bin/activate
# cache pip downloads in /usr/workspace to avoid filling home directory quota
pkgdir=$(pwd)/pkgs
mkdir -p $pkgdir
export PIP_CACHE_DIR=$pkgdir
git clone --recursive
https://github.com/ROCmSoftwarePlatform/apex.git
cd apex
git checkout torch_2.1_higher
git submodule init
git submodule update
export DISTUTILS_DEBUG=1
export __HIP_PLATFORM_HCC__
export __HIP_PLATFORM_AMD__
export HCC_AMDGPU_TARGET=gfx90a
export PYTORCH_ROCM_ARCH=gfx90a
export ROCM_HOME=/opt/rocm-5.7.0
export CC=hipcc
export CXX=hipcc
pip3 wheel -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
pip3 install apex-0.1*whl
```

## Recipe for installing FlashAttention 2 on Frontier
```
pip3 install --pre torch torchvision torchaudio --index-url
https://download.pytorch.org/whl/nightly/rocm5.7

git clone
https://github.com/irlyngaas/flash-attention.git
cd flash-attention
git checkout flash_11-8-23
git submodule init
git submodule update
module load rocm/5.7.0
patch /path/to/new/env/lib/python3.10/site-packages/torch/utils/hipify/hipify_python.py hipify_patch.patch
python setup.py install
```
