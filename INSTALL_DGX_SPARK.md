# Installation on NVIDIA DGX Spark (GB10, aarch64)

This document adapts [INSTALL.md](https://github.com/nv-tlabs/lyra/blob/main/Lyra-2/INSTALL.md) to a DGX Spark. Every step keeps the numbering of
the original and ends with a **What changed** note. Steps without changes are marked as such.

**Status: researched and dry-run checked, not yet executed.** No package has been installed. Only
the empty `lyra2` conda environment was created (Python 3.10.22, pip 26.2.1) to confirm that
conda resolves the `linux-aarch64` packages below. Every conda command in this document was
verified with `--dry-run`, every pip package was checked for an `aarch64` wheel or source
distribution on PyPI, and the PyTorch index was checked for the exact wheel filenames. Items that
can only be confirmed by actually compiling are listed in [Known risks](#known-risks-unverified-until-the-build-runs).

## Host baseline (inspected 2026-10-19)

| Item | Original INSTALL.md assumes | This host |
|---|---|---|
| CPU / arch | x86_64 | 20-core Arm (Cortex-X925 / A725), `aarch64` |
| GPU | H100 (compute capability 9.0) | GB10, compute capability **12.1** (`sm_121`) |
| Memory | discrete 80 GB HBM | 128 GB **unified** CPU+GPU memory, 15 GB swap |
| OS | Ubuntu 22.04 | Ubuntu 24.04.5, kernel 7.0.0-1019-nvidia |
| Driver / CUDA | CUDA 12.8 | driver 580.178.04, system CUDA **13.0** (`/usr/local/cuda-13.0`, nvcc 13.0.88) |
| System gcc | — | gcc 13.3.0 |
| conda | — | conda 26.7.1 at `~/miniconda3`, platform `linux-aarch64` |

Two facts drive almost every change:

1. **The GB10 is `sm_121`.** CUDA 12.8 does not know this architecture. Its `ptxas` and NVRTC
   reject `sm_121`, so CUDA 12.8 wheels can only run pre-compiled `sm_120` code and fail on any
   JIT path (Triton, `torch.compile`, NVRTC-compiled kernels). `sm_121` was added in CUDA 12.9; the
   driver on this host reports CUDA 13.0. `sm_120` binaries are forward compatible with `sm_121`
   (same major version), which is what every build flag below relies on.
2. **Several pinned packages have no `aarch64` wheels.** `decord`, `pycolmap`, `usd-core`,
   `embreex` and `xformers` have no Arm wheel at all; `OpenEXR==3.2.3`, `tensorstore==0.1.45`,
   `numcodecs==0.13.1` and `flash-attn` have x86-only wheels.

## Summary of changes

| Step | Original | DGX Spark | Reason |
|---|---|---|---|
| 1 | `eigen` unpinned | `eigen=3.4.0` | conda-forge now resolves Eigen 5.0.1; VIPE targets 3.4 |
| 1 | `x86_64-conda-linux-gnu-*` compilers | `aarch64-conda-linux-gnu-*` | compiler triplet on Arm |
| 2 | CUDA 12.8 (`nvidia/label/cuda-12.8.0`) | **CUDA 13.0.2** (`nvidia/label/cuda-13.0.2`) | `sm_121` needs CUDA ≥ 12.9; matches driver and torch wheels |
| 3 | torch 2.7.1 / torchvision 0.22.1 `cu128` | **torch 2.9.1 / torchvision 0.24.1 `cu130`** | cu128 lacks `sm_121` tooling; cu130 aarch64 wheels exist |
| 4 | `nvidia/cuda_runtime/lib` in paths | `nvidia/cu13/lib` | CUDA 13 pip wheels use a new directory layout |
| 4 | — | `TORCH_CUDA_ARCH_LIST=12.1`, `FLASH_ATTN_CUDA_ARCHS=120`, `TRITON_PTXAS_PATH` | pin architectures; fix Triton's bundled ptxas |
| 5 | `OpenEXR==3.2.3`, `tensorstore==0.1.45`, `decord==0.6.0`, `numcodecs` wheel | `OpenEXR==3.2.5`, `tensorstore==0.1.78`, decord deferred, numcodecs from source | no aarch64 wheels for the original pins |
| 5 | `pip install git+MoGe` with deps | `--no-deps` + two pinned git deps | gradio 6 replaces `huggingface-hub` 0.35.3 with 2.x, breaking transformers/tokenizers |
| 5 | — | new 5d/5e: install skipped transitive deps, re-pin, `pip check` | `--no-deps` installs leave hydra/omegaconf/ftfy/fvcore unimportable |
| 5 | `transformer_engine[pytorch]` unpinned + `cudart` symlink | `transformer_engine[pytorch]==2.20.2`, no symlink | TE 2.20 is CUDA-13 native and finds `nvidia/cu13` itself |
| 5 | `protobuf==4.25.3`, `typing-extensions==4.14.1` | `>=6.31.1`, `>=4.16.0` | TE 2.20 imports onnx 1.23 (needs protobuf ≥ 6.31); anyio 4.15 needs typing-extensions ≥ 4.16 |
| 6 | `flash-attn==2.6.3` | **`flash-attn==2.8.3.post2`** built for `sm_120` | 2.6.3 only emits `sm_80`/`sm_90` kernels |
| 7 | `pip install -e vipe` (with deps) | `--no-deps` + explicit deps | `python-pycg` hard-requires `usd-core` (no Arm wheel) |
| 7 | `pip install -e depth_anything_3[gs]` | `--no-deps` + explicit deps + lazy `pycolmap` import | `pycolmap` has no Arm wheel; avoids `numpy<2` downgrade |
| GUI | build as-is | patch CMake arch cap | CMake clamps any arch > 90 down to 90, which cannot run on GB10 |
| Training | 8 GPUs | 1 GPU; `decord` built from source | single-GPU host |

## Installation

Run everything from the repository root, in a fresh shell.

This host's conda has the `defaults` channel configured. Every conda command below names its
channels explicitly and passes `--override-channels`, so that `defaults` cannot supply a
conflicting package (see Step 2 for the failure this prevents).

### Step 0. Clone repository

```bash
git clone --recursive git@github.com:nv-tlabs/lyra.git
cd Lyra-2
git submodule update --init --recursive   # safe to re-run; this checkout already has all submodules
```

**What changed:** nothing. All eleven submodules are already checked out on this host.

### Step 1. Create the conda environment

```bash
conda create -n lyra2 python=3.10 pip cmake ninja libgl ffmpeg packaging \
  libgl-devel xorg-libx11 xorg-xorgproto -c conda-forge --override-channels -y
conda activate lyra2
CONDA_BACKUP_CXX="" conda install gcc=13.3.0 gxx=13.3.0 eigen=3.4.0 zlib -c conda-forge --override-channels -y
```

On this host the `lyra2` environment already exists (created as the baseline with
`python=3.10 pip`). Use the install form instead of `conda create`:

```bash
conda activate lyra2
conda install python=3.10 pip cmake ninja libgl ffmpeg packaging \
  libgl-devel xorg-libx11 xorg-xorgproto -c conda-forge --override-channels -y
CONDA_BACKUP_CXX="" conda install gcc=13.3.0 gxx=13.3.0 eigen=3.4.0 zlib -c conda-forge --override-channels -y
```

**What changed:**
- `eigen` is pinned to `3.4.0`. Unpinned, conda-forge now resolves Eigen 5.0.1 on this host.
  VIPE's own `setup.py` downloads Eigen 3.4.0 when `USE_SYSTEM_EIGEN` is unset, so 3.4.0 is the
  version it is written against. `eigen=3.4.0` exists for `linux-aarch64`.
- The compiler packages resolve to `gcc_impl_linux-aarch64-13.3.0`, so the binaries are named
  `aarch64-conda-linux-gnu-gcc` and `aarch64-conda-linux-gnu-g++` (used in Step 4). GCC 13 is
  within the host-compiler range supported by CUDA 13.0 nvcc.
- Everything else (`cmake`, `ninja`, `libgl`, `ffmpeg`, `packaging`, `zlib`) resolves for
  `linux-aarch64` unchanged (verified with `--dry-run`).
- `libgl-devel`, `xorg-libx11` and `xorg-xorgproto` are added. Step 5b installs `moderngl` (a
  dependency of MoGe's `utils3d_moge`), which depends on `glcontext`; `glcontext` has no aarch64
  wheel and compiles from source against `GL/gl.h` and `X11/Xlib.h`/`X11/X.h`. On this host that
  compile only succeeded because the GUI section had already installed those conda packages; in a
  clean environment (the Docker build on 2026-10-19) it failed with
  `fatal error: X11/X.h: No such file or directory`. `moge.model.v1` does not import moderngl at
  import time, but the package is kept to match the verified host environment.

### Step 2. Install the CUDA toolkit inside the environment

```bash
conda install cuda -c nvidia/label/cuda-13.0.2 -c conda-forge --override-channels -y
export CUDA_HOME=$CONDA_PREFIX
```

**What changed:** CUDA **13.0.2** replaces CUDA 12.8.0, and `-c conda-forge --override-channels`
is added to the command.
- `cuda-nvcc 13.0.88` for `linux-aarch64` depends on the `gcc_linux-aarch64` compiler-wrapper
  package. With only the NVIDIA label and `defaults` visible, the solver can offer that wrapper
  only in versions 10.2, 11.2, 14.3 or 15.2, none of which matches the `gcc=13.3.0` installed in
  Step 1, and the solve fails with `LibMambaUnsatisfiableError ... gcc =13.3.0 is not installable`.
  With conda-forge in the list the solver takes `gcc_linux-aarch64 13.3.0` and
  `gxx_linux-aarch64 13.3.0` from conda-forge and keeps the existing compilers (verified by a
  dry-run against the populated environment on 2026-10-19: nothing downgraded or removed).
- The wrapper package also exports `CC`/`CXX` pointing at `aarch64-conda-linux-gnu-gcc`/`g++` on
  `conda activate`, which matches the values set explicitly in Step 4.
- `sm_121` support first appeared in CUDA 12.9. CUDA 12.8's `nvcc`, `ptxas` and NVRTC do not
  accept `sm_121`, so anything that compiles at runtime fails on the GB10.
- The host driver (580.178.04) reports CUDA 13.0, and `/usr/local/cuda-13.0` is already installed
  system-wide. Using the same major version inside conda avoids `libcudart.so.12` vs `.so.13`
  mismatches between the toolkit, the PyTorch wheels (Step 3) and Transformer Engine (Step 5).
- The Transformer Engine release currently on PyPI (2.20.2) ships its PyTorch binding against
  `transformer_engine_cu13` only, so a CUDA 12 environment would not get a matching TE anyway.
- The `nvidia/label/cuda-13.0.2` label resolves `cuda-nvcc 13.0.88`, `cuda-cudart 13.0.96`,
  `cuda-nvrtc 13.0.88` for `linux-aarch64` (verified with `--dry-run`). CUDA 12.8.0 also exists
  for `linux-aarch64`, so the original command would have installed, but the result could not
  target this GPU.

Alternative: skip this step and use the system toolkit with `export CUDA_HOME=/usr/local/cuda-13.0`
(also nvcc 13.0.88). The conda toolkit keeps the environment self-contained, which is why it stays
the primary path here.

### Step 3. Install PyTorch

```bash
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
```

**What changed:** torch 2.7.1+cu128 / torchvision 0.22.1 become **torch 2.9.1+cu130 /
torchvision 0.24.1+cu130**, and `--extra-index-url` becomes `--index-url`.
- `torch-2.7.1+cu128-cp310-manylinux_2_28_aarch64.whl` does exist, but its bundled NVRTC 12.8 and
  Triton's ptxas reject `sm_121` (community reports of `nvrtc: invalid value for --gpu-architecture`
  and broken `torch.compile` on DGX Spark with cu128). The cu128 index also has no `+cu128` tagged
  aarch64 torchvision 0.22.1 wheel, only an untagged one.
- torch 2.9.1+cu130 is the oldest stable line with cu130 aarch64 wheels (2.9.0, 2.9.1, then 2.14.x).
  Its dependencies, read from the wheel metadata, are `nvidia-cuda-nvrtc==13.0.48`,
  `nvidia-cuda-runtime==13.0.48`, `nvidia-cudnn-cu13==9.13.0.50`, `nvidia-nccl-cu13==2.27.7`,
  `triton==3.5.1`, all with aarch64 wheels. The torch 2.9 `cpp_extension` module accepts `12.1`
  and `12.1a` as architecture strings.
- `--index-url` (not `--extra-index-url`) prevents pip from preferring the CPU-only `torch` on PyPI
  when versions tie, a known pitfall on aarch64.
- The GB10 runs the `sm_120` kernels in these wheels; torch prints a warning that `sm_121` is not
  in its compiled list. That warning is expected and harmless.

### Step 4. Set build environment variables

```bash
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export CUDA_HOME=$CONDA_PREFIX
export CPATH="$CUDA_HOME/include:$SITE/nvidia/cu13/include:$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include:$CPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cu13/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/nccl/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CC="$CONDA_PREFIX/bin/aarch64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/aarch64-conda-linux-gnu-g++"

# GB10-specific
export TORCH_CUDA_ARCH_LIST="12.1"          # native arch for torch cpp extensions (VIPE, gsplat); CUDA 13 nvcc knows sm_121
export FLASH_ATTN_CUDA_ARCHS="120"          # flash-attn only has gencode rules for 80/90/100/120; sm_120 runs on sm_121
export NVTE_CUDA_ARCHS="120"                # only consulted if Transformer Engine is ever built from source
export TRITON_PTXAS_PATH="$CONDA_PREFIX/bin/ptxas"   # Triton 3.5.1 bundles a ptxas that does not know sm_121
export MAX_JOBS=8                           # 20 cores, but RAM is shared with the GPU; nvcc jobs are memory hungry
```

Run these in a shell where `conda activate lyra2` has been executed, as in the original
instructions. This is not cosmetic: the environment's activation scripts (from `cuda-nvcc` and
`gcc`/`gxx_linux-aarch64`) export `CFLAGS`, `CXXFLAGS` and `CPPFLAGS` containing
`-I$CONDA_PREFIX/targets/sbsa-linux/include`, `LDFLAGS` with the matching `-L` paths, and
`NVCC_PREPEND_FLAGS=-ccbin=...`. The conda CUDA toolkit on aarch64 keeps its headers under
`targets/sbsa-linux/include`, not `$CONDA_PREFIX/include`, so without activation the host
compiler cannot find `crt/host_config.h` and the Transformer Engine build in Step 5c fails
(observed in the Docker build on 2026-10-19, which does not activate by default; the image now
wraps every build step in an activated shell). The activation script also sets
`TORCH_CUDA_ARCH_LIST` to a long list ending in `12.1+PTX`; the explicit export below overrides it.

**What changed:**
- `nvidia/cuda_runtime/{include,lib}` becomes `nvidia/cu13/{include,lib}`. CUDA 13 pip wheels
  (`nvidia-cuda-runtime`, `nvidia-cuda-nvrtc`, `nvidia-cublas`, ...) install into
  `site-packages/nvidia/cu13/`; verified by listing the wheels. cuDNN and NCCL keep their
  `nvidia/cudnn` and `nvidia/nccl` layout.
- `x86_64-conda-linux-gnu-*` becomes `aarch64-conda-linux-gnu-*`.
- New: `TORCH_CUDA_ARCH_LIST`, `FLASH_ATTN_CUDA_ARCHS`, `NVTE_CUDA_ARCHS`, `TRITON_PTXAS_PATH`,
  `MAX_JOBS`. Without `TORCH_CUDA_ARCH_LIST`, torch autodetects `12.1`, which works with CUDA 13 but
  is pinned here for reproducibility. `TRITON_PTXAS_PATH` matters only for `torch.compile` paths
  (`use_torch_compile`, `CompileTokenizer`), which Lyra exposes but does not enable by default.
- `MAX_JOBS` moves from `16` (original flash-attn step) to `8` because the 128 GB is unified memory.

### Step 5. Install Python dependencies

```bash
# 5a. Lyra requirements, with six substitutions applied on the fly:
#     four for aarch64 wheel availability, two for the Transformer Engine 2.20 dependency chain
sed -e 's/^OpenEXR==3.2.3$/OpenEXR==3.2.5/' \
    -e 's/^tensorstore==0.1.45$/tensorstore==0.1.78/' \
    -e 's/^decord==0.6.0$/# decord==0.6.0   # no aarch64 wheel; training-only, built from source in Step 9/' \
    -e 's/^numcodecs==0.13.1$/numcodecs==0.13.1\n--no-binary numcodecs/' \
    -e 's/^protobuf==4.25.3$/protobuf>=6.31.1   # onnx 1.23 (required and imported by transformer_engine_torch) needs >=6.31.1/' \
    -e 's/^typing-extensions==4.14.1$/typing-extensions>=4.16.0   # anyio 4.15 (httpx, fastapi) needs >=4.16/' \
    requirements.txt > requirements-dgx-spark.txt
pip install --no-deps -r requirements-dgx-spark.txt

# 5b. MoGe, without letting its dependency chain (gradio 6) replace pinned packages
pip install --no-deps "git+https://github.com/microsoft/MoGe.git"
pip install --no-deps \
  "utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37" \
  "pipeline @ git+https://github.com/EasternJournalist/pipeline.git@1c511390d90226c00c101f34b84df26a0f8789b4"
pip install moderngl click "trimesh>=4.11"

# 5c. Transformer Engine (pinned, CUDA 13 native)
pip install --no-build-isolation "transformer_engine[pytorch]==2.20.2"

# 5d. Transitive dependencies that `--no-deps` skipped and that Lyra imports at runtime
#     antlr4 MUST be 4.9.x: omegaconf 2.3.0 / hydra-core 1.3.2 ship a parser generated with ANTLR 4.9,
#     and the 4.13 runtime fails with "Could not deserialize ATN with version 3 (expected 4)"
pip install "antlr4-python3-runtime==4.9.3" wcwidth yacs wadler-lindig ml-dtypes asciitree \
            docstring-parser shtab typeguard matplotlib-inline exceptiongroup \
            starlette annotated-doc httpcore h11 anyio sniffio \
            absl-py grpcio markdown tensorboard-data-server werkzeug \
            iniconfig pluggy execnet pathspec trove-classifiers setuptools-scm

# 5e. Re-apply the pins (restores anything an earlier install replaced) and audit
pip install --no-deps -r requirements-dgx-spark.txt
pip check
```

After 5e, `pip check` should list only:

- the four `megatron-core` extras that Lyra does not use (`flask-restful`, `nltk`,
  `nvidia-modelopt`, `wandb`);
- `moge 3.0.0 requires gradio`, which is deliberately not installed;
- `nvidia-cusparselt-cu13 0.8.0 is not supported on this platform`. This is a tagging quirk, not
  a wrong wheel: NVIDIA publishes the aarch64 wheel with the non-standard `manylinux2014_sbsa` tag
  inside its WHEEL metadata, which pip cannot match against the host. The shipped
  `libcusparseLt.so.0` is a genuine `ARM aarch64` ELF, and Lyra never uses cuSPARSELt
  (semi-structured sparsity) anyway.

Any other line is a real problem.

**If you already ran the original `pip install "git+https://github.com/microsoft/MoGe.git"`**
(with dependency resolution), gradio and its helpers are in the environment and
`huggingface-hub` has been replaced. Nothing in Lyra imports gradio, so remove it, then run 5e:

```bash
pip uninstall -y gradio gradio-client hf-gradio safehttpx groovy pydub python-multipart \
                 semantic-version brotli httpx2 httpcore2 truststore
pip install --no-deps -r requirements-dgx-spark.txt
pip check
```

The listed packages were checked against the environment on 2026-10-19: none is required by any
other installed package once `huggingface-hub` is back at 0.35.3 (`httpx2`, `httpcore2` and
`truststore` were pulled in only by huggingface-hub 2.x). Lyra's FastAPI server does not use
multipart uploads, so `python-multipart` is not needed. `flex-gemm` (MoGe-3 only) and
`hf-xet` (a regular huggingface-hub dependency) can stay. `moge` keeps declaring `gradio` as a
requirement, so `pip check` will print one `moge 3.0.0 requires gradio` line; that is expected.

**What changed:**
- `OpenEXR==3.2.3` → `3.2.5`: 3.2.5 is the first release with an aarch64 wheel. It is the same
  3.2 API line and still ships the `Imath` module that VIPE's `utils/io.py` imports (verified by
  listing the wheel). 3.2.3 only has an sdist that needs the OpenEXR C++ libraries.
- `tensorstore==0.1.45` → `0.1.78`: 0.1.45 has x86-only wheels and its sdist needs Bazel. 0.1.78
  is the newest release with a `cp310` aarch64 wheel. Lyra never imports tensorstore, zarr or
  numcodecs directly; they are declared dependencies of `megatron-core`.
- `numcodecs==0.13.1` is built from source (`--no-binary`), since no aarch64 wheel exists for any
  numcodecs version that supports Python 3.10. It is a small Cython build.
- `decord==0.6.0` is removed from this step. PyPI has x86 wheels only and no sdist. It is imported
  only by the training data loader (`lyra_2/_src/datasets/radym.py`), so inference does not need
  it. See Step 9.
- `protobuf==4.25.3` → `protobuf>=6.31.1` and `typing-extensions==4.14.1` →
  `typing-extensions>=4.16.0`. Installing `transformer_engine[pytorch]==2.20.2` on 2026-10-19
  brought in `onnx 1.23.2` and `onnxscript 0.7.2` (hard requirements of `transformer_engine_torch`)
  and raised protobuf to 7.36.2; `transformer_engine.pytorch` imports `onnx` at import time, and
  onnx 1.23 requires `protobuf>=6.31.1`, so re-pinning protobuf to 4.25.3 breaks the TE import.
  Everything else that uses protobuf (`tensorboard`, `sentencepiece`, the cutlass-dsl packages)
  accepts 7.x; the sentencepiece model proto was verified to load under 7.36.2. `anyio 4.15.1`
  (via `httpx`, which `fastapi` and `huggingface-hub` use) and `typing-inspection 0.4.4` (via
  `pydantic`) require typing-extensions ≥ 4.16 / 4.15; nothing in the environment caps it below 5.
  Upstream never re-applies its pins after the TE install, so an x86 install ends up with the
  same newer versions; the relaxed pins only make that explicit.
- `megatron-core==0.12.1` has a `cp310` aarch64 wheel; all other pinned packages are pure Python or
  have aarch64 wheels (checked individually: opencv, scipy, scikit-image, av, imageio-ffmpeg,
  sentencepiece, tiktoken, tokenizers, safetensors, orjson, pydantic-core, psutil, matplotlib,
  pillow, pandas, regex, ninja, pyyaml, markupsafe, lru-dict, pyexr/OpenEXR).
- The `cuda_runtime → cudart` symlink is **removed**. It worked around an old Transformer Engine
  library lookup under `nvidia/cudart`. TE 2.20 searches `nvidia/cu13/lib`, `nvidia/<lib>/lib`
  and `CUDA_HOME` itself, so the symlink is unnecessary and `nvidia/cuda_runtime` does not exist
  in a CUDA 13 environment.
- Transformer Engine is pinned to `2.20.2`. The `pytorch` extra resolves to
  `transformer_engine_torch` (an sdist, compiled here against torch 2.9.1 and the CUDA 13 nvcc,
  which is why `--no-build-isolation` and the `CPATH` cuDNN include path matter) plus
  `transformer_engine_cu13` (a prebuilt `manylinux_2_28_aarch64` wheel; `cuobjdump` on the
  installed `libtransformer_engine.so` shows `sm_75 sm_80 sm_89 sm_90 sm_100 sm_103 sm_107 sm_120`,
  so the GB10 runs its `sm_120` kernels). TE 2.20 requires cuDNN ≥ 9.12 (torch brings 9.13) and accepts
  flash-attn versions from 2.7.3 (Blackwell minimum) up to but excluding 2.8.5, which fixes the
  flash-attn pin in Step 6. The install also pulls `nvidia-cudnn-frontend 1.31`, `onnx`,
  `onnxscript`, `nvdlfw-inspect`, the `nvidia-cutlass-dsl` packages (both cu12 and cu13 variants,
  harmless) and `cuda-bindings 13.4.4`; the latter is a CUDA 13.x driver-API binding and works
  with the 13.0 driver as long as no 13.4-only entry point is called, which TE's default paths
  do not. **Verified on this host on 2026-10-19**: `import transformer_engine.pytorch` succeeds
  and a `DotProductAttention` forward pass (the only TE module Lyra uses, in `wan2pt1.py`) runs on
  the GB10 in bfloat16.
- MoGe is installed with `--no-deps`. Running the original `pip install "git+.../MoGe.git"` on
  2026-10-19 pulled in `gradio 6.30`, which requires `huggingface-hub>=1.16` and therefore
  **replaced the pinned `huggingface-hub 0.35.3` with 2.2.0**, breaking `transformers 4.57.1` and
  `tokenizers 0.22.0` (both require `<1.0`). It also moved `typing-extensions` and `idna` off their
  pins. Lyra only uses `moge.model.v1`, which imports `utils3d_moge` and `huggingface_hub`; gradio
  is used only by MoGe's demo app, and `flex-gemm` only by the MoGe-3 sparse modules, so neither is
  installed. The two git dependencies are pinned to the commits MoGe's own `pyproject` names;
  `utils3d_moge` needs `moderngl`, `numpy`, `scipy`. This is not an aarch64 problem; the same
  thing happens on x86 with the current MoGe `main`. Because gradio is absent, `pip check` reports
  `moge 3.0.0 requires gradio, which is not installed`; this is the one expected line besides the
  megatron-core extras.
- Step 5d is new. `requirements.txt` is installed with `--no-deps`, as upstream does, and upstream
  then relies on the full-dependency installs of VIPE and DA3 to drag in the missing transitive
  packages. Step 7 installs those two with `--no-deps` on aarch64, so the gap has to be closed
  explicitly. Lyra's own code imports `hydra`/`omegaconf` (need `antlr4-python3-runtime==4.9.*`;
  an unpinned install picks 4.13, which omegaconf rejects at the first interpolation, verified on
  this host), `ftfy`
  (`wcwidth`), `fvcore` (`yacs`); DA3 and gsplat import `jaxtyping` (`wadler-lindig`); the pinned
  `tyro` needs `docstring-parser`, `shtab`, `typeguard`; `tensorstore` needs `ml-dtypes` and `zarr`
  needs `asciitree`; the rest serve the pytest, ipython, tensorboard and hatch tooling the
  requirements file pins. `starlette` and `annotated-doc` (needed by the pinned `fastapi`, i.e.
  the GUI server) and `exceptiongroup` (ipython, pytest on Python 3.10) were added after the
  Docker build showed them missing: on this host they had arrived as side effects of the
  since-removed gradio install. The same applies to `httpcore`, `h11`, `anyio` and `sniffio`,
  the transport stack behind the pinned `httpx` (installed with `--no-deps`): the GUI client
  (`gui/api/client.py`) failed with `No module named 'httpcore'` in the container on 2026-10-20. All are pure Python or have aarch64 wheels (`grpcio`,
  `ml-dtypes`, `tensorboard-data-server` checked).
- Step 5e is new: re-running the pinned requirements file with `--no-deps` is idempotent and puts
  back any pin a `pip install` with dependency resolution may have moved.

### Step 6. Install Flash Attention

```bash
FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS="120" MAX_JOBS=8 \
  pip install --no-build-isolation --no-binary :all: flash-attn==2.8.3.post2
```

**What changed:** `flash-attn==2.6.3` → **`2.8.3.post2`**, plus `FLASH_ATTN_CUDA_ARCHS=120`.
- 2.6.3's `setup.py` hard-codes `-gencode` for `sm_80` and `sm_90` only. Those kernels cannot run
  on `sm_121` ("no kernel image is available"). From 2.7.4.post1 onwards the build honours
  `FLASH_ATTN_CUDA_ARCHS` and emits `sm_120` when CUDA ≥ 12.8.
- The architecture list must say `120`, not `121`: the setup script only has rules for
  `80;90;100;120`, and `sm_120` SASS is binary compatible with `sm_121`. Building only `120` keeps
  the compile to one architecture (community reports 15 to 75 minutes on the GB10's Arm cores).
- `FLASH_ATTENTION_FORCE_BUILD=TRUE` skips the attempt to download a prebuilt wheel from GitHub;
  none exists for `linux_aarch64`, so without the flag it only adds a failed download before
  falling back to the same source build.
- 2.8.3.post2 is the newest release and is inside Transformer Engine 2.20's accepted range.
- Flash Attention 3 is Hopper-only. Lyra's `attention.py` already falls back to PyTorch SDPA on
  non-Hopper GPUs; on the GB10 it will use the SDPA flash/cuDNN backends. The `flash_attn` package
  is still required for `flash_attn.layers.rotary.apply_rotary_emb` in `wan2pt1.py` and as a
  Transformer Engine backend.

### Step 7. Build vendored CUDA extensions

```bash
# 7a. VIPE: install the package without resolving its metadata, then its real dependencies
USE_SYSTEM_EIGEN=1 pip install --no-build-isolation --no-deps -e 'lyra_2/_src/inference/vipe'
pip install --no-deps python-pycg            # hard-requires usd-core, which has no aarch64 wheel; pycg imports it lazily
pip install calmsize screeninfo pyquaternion plyfile trimesh \
            rerun-sdk timm kornia gdown viser click

# 7b. Depth Anything 3: same pattern, plus the gsplat CUDA extension
pip install --no-build-isolation --no-deps -e 'lyra_2/_src/inference/depth_anything_3'
pip install trimesh evo e3nn moviepy plyfile pillow_heif typer pre-commit uvicorn addict
pip install open3d==0.20.0                   # optional: only imported by DA3's benchmark code
TORCH_CUDA_ARCH_LIST="12.1" pip install --no-build-isolation \
  "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70"

# 7c. pycolmap has no aarch64 wheel. DA3 imports it at module load through utils/export; Lyra only
#     uses the "mini_npz" export, so make the COLMAP import lazy.
python - <<'EOF'
import pathlib, re
p = pathlib.Path("lyra_2/_src/inference/depth_anything_3/src/depth_anything_3/utils/export/__init__.py")
s = p.read_text()
s = s.replace("from .colmap import export_to_colmap\n", "")
s = s.replace('    elif export_format == "colmap":\n        export_to_colmap(',
              '    elif export_format == "colmap":\n        from .colmap import export_to_colmap\n        export_to_colmap(')
p.write_text(s)
print("patched", p)
EOF
```

Expected `pip check` additions after Step 7, on top of the Step 5e residue. Each comes from a
`--no-deps` install made deliberately, and each is harmless for Lyra's code paths:

- `python-pycg 1.0.3 requires usd-core, which is not installed` (no aarch64 wheel; pycg imports it
  only inside its USD export functions);
- `depth-anything-3 ... requires pycolmap, which is not installed` (no aarch64 wheel; the COLMAP
  import is made lazy in 7c);
- `depth-anything-3 ... requires numpy<2, but you have numpy 2.x` (Lyra, MoGe and rerun-sdk need
  `numpy>=2`; DA3's inference path works on numpy 2);
- `depth-anything-3 ... requires open3d` only if you skipped the optional `open3d` install.

**What changed:**
- VIPE is installed with `--no-deps`. Its `pyproject` requires `python-pycg`, which in every
  released version hard-requires `usd-core`; `usd-core` has never published an aarch64 wheel and
  has no sdist, so a normal `pip install -e vipe` fails to resolve on this host. `pycg` only
  imports the USD module inside its export functions, so installing it with `--no-deps` is safe.
  The remaining VIPE dependencies are installed explicitly; all have aarch64 wheels or are pure
  Python (`rerun-sdk` 0.39 ships `manylinux_2_28_aarch64`). `xformers` is optional in both VIPE and
  DA3 (`try/except` import) and is skipped: it has no aarch64 wheel.
- `USE_SYSTEM_EIGEN=1` is unchanged and now points at the Eigen 3.4.0 pinned in Step 1. VIPE's
  `setup.py` picks up `$CONDA_PREFIX/bin/nvcc` automatically.
- DA3 is installed with `--no-deps` for two reasons: its `pyproject` requires `pycolmap`
  (x86-only wheels, no sdist) and `numpy<2`, which would have silently downgraded the `numpy>=2`
  that Lyra, MoGe and rerun-sdk require. The `[gs]` extra is replaced by an explicit install of the
  same pinned gsplat commit; `TORCH_CUDA_ARCH_LIST=12.1` makes gsplat's `setup.py` compile for the
  GB10 instead of a default list.
- `addict` is added to the explicit DA3 dependency list. DA3's `model/da3.py` imports it at module
  level, but DA3's `pyproject` does not declare it; upstream gets it transitively from `open3d`.
  With open3d optional here, the first Step 8 import failed with `No module named 'addict'`
  (observed 2026-10-19). A scan of every third-party import in Lyra, VIPE, DA3 and MoGe's v1 path
  against the finished environment found no other unguarded gap: the remaining absent modules
  (`aioboto3`, `cloudpickle`, `dill`, `straggler`, `flash_attn_3`, `xformers`) sit behind
  `try/except`, `debugpy` and `imaginaire` behind CLI flags, `decord` in the training loader, and
  VIPE's `mmcv`, `pycocotools` and `vslam` in paths the default pipeline does not import.
- The COLMAP export import is made lazy with a small patch to the vendored submodule. Lyra's
  reconstruction scripts call DA3 with `export_format="mini_npz"` only. Alternative without
  patching: `conda install -c conda-forge --override-channels "pycolmap=4.1.1=cpu_py310*"`, which resolves on
  `linux-aarch64` but pulls a `colmap` build pinned to CUDA 12.9 libraries into the environment,
  which is why it is not the default here.
- `open3d` is listed by DA3 but only imported by its benchmark datasets. 0.20.0 has a
  `manylinux_2_35_aarch64` wheel (Ubuntu 24.04 has glibc 2.39), so it can be installed, but it is
  optional.

### Shell profile

Add to `~/.bashrc` (or equivalent) so the libraries are found in every session:

```bash
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cu13/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TRITON_PTXAS_PATH="$CONDA_PREFIX/bin/ptxas"
```

**What changed:** `nvidia/cuda_runtime/lib` → `nvidia/cu13/lib`; `nvidia/nccl/lib` and
`TRITON_PTXAS_PATH` added.

### Step 8. Verify installation

```bash
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cu13/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PYTHONPATH=. python -c "
import torch, flash_attn, transformer_engine.pytorch, vipe_ext, depth_anything_3.api, moge.model.v1
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0), '| capability:', torch.cuda.get_device_capability(0))
print('torch arch list:', torch.cuda.get_arch_list())
print('flash_attn:', flash_attn.__version__)
# GB10 kernel smoke test: a real forward pass through flash-attn and SDPA
from flash_attn import flash_attn_func
q = torch.randn(1, 128, 8, 64, device='cuda', dtype=torch.bfloat16)
print('flash_attn_func OK:', flash_attn_func(q, q, q).shape)
print('sdpa OK:', torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2), q.transpose(1,2), q.transpose(1,2)).shape)
from transformer_engine.pytorch.attention import DotProductAttention
te_attn = DotProductAttention(num_attention_heads=8, kv_channels=64, num_gqa_groups=8, attention_dropout=0, qkv_format='bshd').cuda()
print('TE DotProductAttention OK:', te_attn(q, q, q).shape)
print('all imports OK')
"
PYTHONPATH=. python -m lyra_2._src.inference.lyra2_zoomgs_inference --help
PYTHONPATH=. python -m lyra_2._src.inference.vipe_da3_gs_recon --help
```

Expected output (observed on this host, 2026-10-19):

```text
torch: 2.9.1+cu130 | cuda: True
device: NVIDIA GB10 | capability: (12, 1)
torch arch list: ['sm_80', 'sm_90', 'sm_100', 'sm_110', 'sm_120', 'compute_120']
flash_attn: 2.8.3.post2
flash_attn_func OK: torch.Size([1, 128, 8, 64])
sdpa OK: torch.Size([1, 8, 128, 64])
TE DotProductAttention OK: torch.Size([1, 128, 512])
all imports OK
```

The arch list contains `sm_120` and no `sm_121`, which is normal: the GB10 runs the `sm_120`
kernels. The SDPA shape is head-major because the test transposes before the call. Torch prints a
warning that capability 12.1 is above its supported maximum of 12.0; that is the expected
`sm_120`-on-`sm_121` situation and can be ignored.

**What changed:** path update as above, plus explicit checks that the kernels actually launch on
`sm_121`. A clean import proves little on this GPU; a "no kernel image is available" error only
appears at the first kernel launch.

### Step 9 (new, training only). Build decord from source

Skip this unless you intend to run [TRAINING.md](https://github.com/nv-tlabs/lyra/blob/main/Lyra-2/TRAINING.md). `decord` is imported by the Radym
dataset loader only.

```bash
# decord 0.6.0 still calls av_register_all(), which FFmpeg 5+ removed, so build against FFmpeg 4.4
conda install "ffmpeg=4.4" -c conda-forge --override-channels -y   # downgrades the conda ffmpeg CLI only; pip wheels bundle their own
git clone --recursive https://github.com/dmlc/decord.git /tmp/decord-src
cmake -S /tmp/decord-src -B /tmp/decord-src/build -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
cmake --build /tmp/decord-src/build -j8
pip install /tmp/decord-src/python
```

Alternative: conda-forge ships patched `decord 0.6.0` for `linux-aarch64`, but only for Python
3.11 and newer, so it is not usable in this Python 3.10 environment.

**What changed:** new step. Unverified on this host; the FFmpeg API constraint is read from the
decord sources.

## GUI ([gui/README.md](https://github.com/nv-tlabs/lyra/blob/main/Lyra-2/gui/README.md)) on the DGX Spark

The Spark runs a local desktop (Xorg and GNOME are active), so server and viewer can both run on it.

```bash
conda activate lyra2
pip install commentjson pybind11 pyexr     # instead of `pip install -r gui/requirements.txt`
# NOT `pip install -U huggingface_hub` (upstream gui/README): that upgrades huggingface-hub to 2.x and
# breaks transformers 4.57.1 / tokenizers 0.22.0 again. The `hf` CLI is already in the pinned 0.35.3.
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir checkpoints/qwen/Qwen3-VL-4B-Instruct

conda install -c conda-forge --override-channels -y glew libgl-devel xorg-libx11 xorg-libxrandr \
  xorg-libxinerama xorg-libxcursor xorg-libxi pkg-config
bash gui/scripts/setup_dependencies.sh

# Patch: the CMake script caps any detected architecture above 90 at 90 for CUDA >= 11.8.
# The GB10 autodetects as 121, would be rebuilt as sm_90, and could not run. Allow 121 on CUDA >= 12.9.
sed -i 's/^\(\s*\)set(LATEST_SUPPORTED_CUDA_ARCHITECTURE 90)$/\1set(LATEST_SUPPORTED_CUDA_ARCHITECTURE 90)\n\1if (CUDA_VERSION VERSION_GREATER_EQUAL 12.9)\n\1\tset(LATEST_SUPPORTED_CUDA_ARCHITECTURE 121)\n\1endif()/' gui/CMakeLists.txt
grep -n "LATEST_SUPPORTED_CUDA_ARCHITECTURE 121" gui/CMakeLists.txt   # must print one line

# Patch: the vendored ImGuizmo switches to ImGui::SetNextFrameWantCaptureMouse at ImGui version
# number 18700, but the pinned ImGui snapshot (Feb 2022, "1.88 WIP", 18709) predates that rename
# (May 2022; final 1.88 is 18800). Raise the guard so ImGuizmo keeps using CaptureMouseFromApp.
sed -i 's/IMGUI_VERSION_NUM >= 18700/IMGUI_VERSION_NUM >= 18800/g' gui/dependencies/imguizmo/ImGuizmo.cpp
grep -c "IMGUI_VERSION_NUM >= 18800" gui/dependencies/imguizmo/ImGuizmo.cpp   # must print 6

# Patch: the Python bindings use the ImGuiMod_* modifier names introduced in ImGui 1.89; the pinned
# ImGui snapshot only has the 1.88-era ImGuiKeyModFlags_* names (same bit values, same io.KeyMods field).
sed -i 's/\bImGuiMod_\(Alt\|Ctrl\|Shift\|Super\)\b/ImGuiKeyModFlags_\1/g' gui/src/python_api.cu
grep -c "ImGuiKeyModFlags_" gui/src/python_api.cu   # must print 4

export CMAKE_PREFIX_PATH="$CONDA_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export CUDACXX=$CONDA_PREFIX/bin/nvcc
cmake gui -B gui/build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DNGP_BUILD_WITH_PYTHON_BINDINGS=ON \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DPython_EXECUTABLE=$CONDA_PREFIX/bin/python    # the host also has a system Python 3.14; bind to the env's 3.10
cmake --build gui/build --config RelWithDebInfo -j8
```

**What changed:**
- `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` is added to the configure command. conda-forge now
  resolves CMake 4.4.4, and CMake 4 refuses projects that declare `cmake_minimum_required` below
  3.5. Five vendored GUI submodules do (`filesystem` 2.8, `tinylogger` 2.8.12, `OpenXR-SDK` 3.0,
  `args` 3.2, `pybind11` 3.4); the first one reached, OpenXR-SDK, aborted the configure on
  2026-10-19 with `Compatibility with CMake < 3.5 has been removed`. The flag tells CMake to treat
  those projects as if they required 3.5. Alternative: `conda install "cmake<4" -c conda-forge
  --override-channels` and drop the flag.
- The configure log also prints `Could NOT find Vulkan`. That is harmless: Vulkan is only used
  for DLSS, which is x86-only and stays disabled (`NGP_BUILD_WITH_VULKAN=OFF`).
- `gui/dependencies/imguizmo/ImGuizmo.cpp` is patched (six occurrences of one version guard).
  The build failed on 2026-10-19 with `'SetNextFrameWantCaptureMouse' is not a member of 'ImGui'`.
  ImGuizmo is vendored in-tree and uses that function when `IMGUI_VERSION_NUM >= 18700`; the
  `imgui` submodule is pinned to commit `fa2b318dd` (2022-02-22, `1.88 WIP`, version number
  18709), while the ImGui commit that introduced the function (`7bf07d252`) is dated 2022-05-23
  and is not an ancestor of the pinned snapshot. The vendored sources are therefore inconsistent
  with each other independent of platform. Raising the guard to 18800 (the final 1.88 release
  number) makes ImGuizmo fall back to `CaptureMouseFromApp`, which the pinned ImGui provides; the
  patched file passes a syntax-only compile against the pinned headers. Moving the submodule
  forward would also work but could change other ImGui APIs the GUI relies on.
- `gui/src/python_api.cu` is patched (four lines). After the ImGuizmo fix the build reached 98%
  and failed in the Python bindings with `identifier "ImGuiMod_Alt" is undefined` (and Ctrl,
  Shift, Super). Those names arrived in ImGui 1.89; the pinned 1.88-WIP snapshot spells them
  `ImGuiKeyModFlags_*`, with identical bit values and the same `io.KeyMods` field, so the rename is
  behavior-neutral. A scan of every `ImGui*_` identifier in `gui/src` and `gui/include` against the
  pinned `imgui.h` found no other mismatch, and this is the only file that uses the new names. Both
  patches point the same way: the repo's GUI sources were written against a newer ImGui than the
  submodule pin records. Upgrading the submodule to 1.89.x would be the upstream-style fix; the two
  local patches are the smaller change.
- `gui/CMakeLists.txt` is patched so the detected architecture (121) survives. Without the patch
  the build silently targets `sm_90`. `TCNN_CUDA_ARCHITECTURES=120` as an environment variable does
  **not** help, because the cap is applied after that variable is read.
- The DLSS submodule ships x86_64 libraries only. It is only compiled when
  `NGP_BUILD_WITH_VULKAN=ON`, which is off by default; leave it off.
- `pip install -r gui/requirements.txt` is replaced by installing only the three packages that are
  not already present. The GUI requirements file is unpinned and would otherwise let pip move
  pinned packages (`pyexr` requires `OpenEXR>=3.2.3`, satisfied by 3.2.5).
- All X11, GLEW and GL development packages resolve for `linux-aarch64` (verified with `--dry-run`).
- `-j` is capped at 8 for the same unified-memory reason as `MAX_JOBS`.

The build produces a single artifact, `gui/build/pyngp.cpython-310-aarch64-linux-gnu.so` (about
33 MB); `gui/api/client.py` finds it by globbing `gui/build*/**/*.so`, so no install step follows.

## Running the inference server and GUI client

Everything below was derived from `gui/api/server.py`, `server_lyra.py`, `lyra_persistent.py`,
`qwen_captioner.py` and `client.py` on 2026-10-19.

### R0. Repair the environment first

If `pip install -U huggingface_hub` was run (it was in the original GUI instructions), transformers
is broken right now: `import transformers` fails with huggingface-hub 2.2.0. Re-apply the pins and
confirm:

```bash
pip install --no-deps -r requirements-dgx-spark.txt
python -c "import transformers, huggingface_hub; print(transformers.__version__, huggingface_hub.__version__)"
# expect: 4.57.1 0.35.3
hf version   # the hf CLI is part of 0.35.3
```

### R1. Download the Lyra checkpoints (about 97 GB)

```bash
hf download nvidia/Lyra-2.0 --include "checkpoints/*" --local-dir .
```

This is the README command with the current `hf` CLI name; `huggingface-cli download` still works
as an alias. The repo's `checkpoints/` tree on Hugging Face and what each part is for:

| Path | Size | Used by |
|---|---:|---|
| `checkpoints/model/` (65 `.distcp` shards) | 68.3 GB | diffusion transformer; default `LYRA_GUI_CHECKPOINT_PATH` |
| `checkpoints/recon/model.pt` | 13.4 GB | DA3 reconstruction model, loaded by the GUI server for its 3D cache |
| `checkpoints/text_encoder/` (`encoder.pth`, `negative_prompt.pt`) | 11.4 GB | UMT5 text encoder; `negative_prompt.pt` is read by the server at startup |
| `checkpoints/image_encoder/model.pth` | 2.4 GB | image conditioning |
| `checkpoints/vae/` | 0.5 GB | video VAE and normalisation stats |
| `checkpoints/lora/` | 1.1 GB | includes `dmd_distillation.safetensors` for `--use_dmd` / `LYRA_GUI_USE_DMD=1` |
| `checkpoints/qwen/Qwen3-VL-4B-Instruct/` | 8.3 GB | GUI prompt rewriting; already present on this host |

The server resolves `checkpoints/model`, `checkpoints/recon/model.pt`,
`checkpoints/text_encoder/negative_prompt.pt` relative to the working directory and the Qwen folder
relative to the repository root, so run everything from the repository root. The NVMe has
about 1.8 TB free, so space is not a concern.

### R2. Session environment (every new terminal)

```bash
cd /path/to/Lyra-2
conda activate lyra2
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cu13/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TRITON_PTXAS_PATH="$CONDA_PREFIX/bin/ptxas"
export PYTHONPATH="$PWD:$PWD/gui/api"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

`PYTHONPATH` must contain both the repo root (for `lyra_2`) and `gui/api` (the server modules
import each other as top-level names). These are the same variables as the shell-profile block
plus the two the GUI README asks for.

### R3. Smoke-test server and client without the weights

The server has a protocol-compatible dummy mode that needs no checkpoints and no GPU model
loading. Use it to prove the `pyngp` build, the client window and the HTTP protocol before the
97 GB download finishes.

Terminal 1:

```bash
LYRA_GUI_DUMMY=1 python -m uvicorn server:app --app-dir gui/api --host 0.0.0.0 --port 8000
```

Terminal 2 (inside the desktop session, see R5):

```bash
python gui/api/client.py --host 127.0.0.1 --port 8000
```

The viewer opens with a **black viewport**; that is expected until a scene is seeded. The
"Video generation server" panel on the right should read `Connected to server 127.0.0.1:8000,
model name: Lyra2Dummy` and `This model requires seeding data` (verified on this host,
2026-10-19). Seed by typing `assets/samples/00.png` into the Seeding **Path** box and clicking
**Seed**, or by dragging the image onto the viewport. The dummy model downsizes the image to
96×64, returns it as a flat point cloud at constant depth 2.0, and marks the model seeded. After
that, author a short camera path and click **Generate video**: the dummy returns a tiny MP4 whose
frames are the seed image scrolled sideways, which exercises the full request, video decode and
cache-update path without any GPU model. Nothing more meaningful is possible in dummy mode.

### R4. Start the real inference server

```bash
python -m uvicorn server:app --app-dir gui/api --host 0.0.0.0 --port 8000
```

What happens at startup: Uvicorn's lifespan hook constructs `LyraModel`, which loads the
diffusion transformer from `checkpoints/model`, the UMT5 text encoder, the DA3 reconstruction
model from `checkpoints/recon/model.pt`, the negative prompt embedding, and the Qwen3-VL
captioner. That is roughly 95 GB read from NVMe, so the first start takes minutes; the server is
ready when Uvicorn logs `Application startup complete` and `Uvicorn running on http://0.0.0.0:8000`.

Memory on this machine: all of the above shares one 128 GB pool with the host. The defaults
(`LYRA_GUI_OFFLOAD=1`, `LYRA_GUI_OFFLOAD_VAE=0`) are the safe starting point. Note that on unified
memory "offloading to CPU" does not free physical memory the way it does on a discrete GPU; it only
moves tensors within the same pool at the cost of copies. Once the default configuration works, try
`LYRA_GUI_OFFLOAD=0` and compare; watch `free -g` and `nvidia-smi` during a generation. Start with
a `--inference-resolution` of `576 320` on the client (its default) before trying larger frames.

Useful overrides (all read from the environment before startup; the full table is in
[gui/README.md](https://github.com/nv-tlabs/lyra/blob/main/Lyra-2/gui/README.md)):

| Variable | Default | Note for this host |
|---|---:|---|
| `LYRA_GUI_USE_DMD` | `0` | `1` enables the 4-step DMD LoRA, roughly 15× faster per chunk; good for a first run |
| `LYRA_GUI_OFFLOAD` | `1` | see memory note above |
| `LYRA_GUI_CHECKPOINT_PATH` | `checkpoints/model` | only if the weights live elsewhere |
| `LYRA_GUI_OUTPUT_DIR` | `outputs/gui_sessions` | stitched `full_video.mp4` per session |
| `LYRA_GUI_SEED`, `LYRA_GUI_GUIDANCE`, `LYRA_GUI_SAMPLING_STEPS` | `1`, `5.0`, `35` | generation controls |

### R5. Start the GUI client

The client opens a GLFW/OpenGL window, so it must run inside the Spark's desktop session (Xorg
and GNOME are running on this host). From a terminal in that session:

```bash
python gui/api/client.py --host 127.0.0.1 --port 8000
```

From an SSH session instead, point it at the local display first: `export DISPLAY=:0`. The client
finds the freshly built `pyngp` module in `gui/build` automatically. Options worth knowing:
`--inference-resolution W H` (default `576 320`; the server advertises the resolutions it
supports and the client falls back to one of them), `--output-dir` for client-side renders, and
`--vr` (needs an OpenXR runtime; not set up here). Usage of the viewer itself (seeding, camera
path authoring, Generate video, Revert) is unchanged from [gui/README.md](https://github.com/nv-tlabs/lyra/blob/main/Lyra-2/gui/README.md).

Running the client on a different workstation is also possible: that machine needs its own GUI
build per the GUI README, and the server port is forwarded with
`ssh -N -L 8000:127.0.0.1:8000 <spark-hostname>`.

### R6. From a GUI session to a 3D scene

The stitched video of a session is written to `outputs/gui_sessions/<timestamp>_<seed>/full_video.mp4`.
Lift it to Gaussian splats exactly as in README Step 2:

```bash
PYTHONPATH=. python -m lyra_2._src.inference.vipe_da3_gs_recon --input_video_path outputs/gui_sessions/<session>/full_video.mp4
```

### R7. Command-line inference (no GUI)

The README's Step 1 commands work unchanged once R0 to R2 are done. For a first end-to-end check
on this machine, run the zoom preset with DMD, which cuts sampling to 4 steps:

```bash
PYTHONPATH=. python -m lyra_2._src.inference.lyra2_zoomgs_inference \
  --input_image_path assets/samples --sample_id 4 --experiment lyra2 \
  --checkpoint_dir checkpoints/model --prompt_dir assets/samples \
  --output_path outputs/zoomgs --num_frames_zoom_in 81 --num_frames_zoom_out 241 \
  --zoom_in_strength 0.5 --zoom_out_strength 1.5 --use_dmd
```

The README quotes about 9 minutes per 80 frames on an H100 (about 35 s with `--use_dmd`). Measured
on this host (2026-10-19, inside the Docker image, which runs the identical stack): the command
above took 29.5 minutes wall-clock, of which about 2.5 minutes were model loading (68 GB DCP
checkpoint, three LoRAs, DA3) and about 26.5 minutes were generation of the 322 frames with DMD,
so roughly 6.5 minutes per 80-frame chunk. Without `--use_dmd` (35 sampling steps instead of 4)
expect it to be many times slower. The native run and the container run produced byte-identical
`outputs/zoomgs/videos/04.mp4` (322 frames, 832×480), which also closes the torch 2.9.1
question under "Status and known risks" for the inference path.

## Docker image for the CLI inference task

This repository packages Steps 1 to 7 of this document into a self-contained image
(`lyra2-dgx-spark`, about 13 GB). The Dockerfile's first stage clones
`https://github.com/nv-tlabs/lyra` at a pinned commit (`LYRA_REF`, default the commit verified
here) with the VIPE and DA3 submodules, so the `docker/` directory is the only build input; no
local checkout or conda environment is involved. The image runs the R7 zoom-preset inference,
the custom-trajectory inference, the Gaussian-splat reconstruction, or the GUI server, with a host
assets directory (`./assets`, populated from the cloned source by `build.sh` and read by
inference), `outputs`, the checkpoints and the Hugging Face/torch caches bind-mounted. See
[README.md](README.md) for build and run instructions and the checkpoint-resolution
rules. Building it from a clean base is what exposed the three environment assumptions now
recorded in Steps 1, 4 and 5d (X11/GL headers for `glcontext`, the activated-shell compiler
flags, and the fastapi/ipython transitive packages).

## Training ([TRAINING.md](https://github.com/nv-tlabs/lyra/blob/main/Lyra-2/TRAINING.md)) notes

- `scripts.convert_wan_checkpoint` needs about 45 GB of CPU memory; the Spark's 128 GB is enough
  provided nothing is resident on the GPU at the same time (the memory is shared).
- Replace `torchrun --nproc_per_node=8` with `--nproc_per_node=1`. The documented runs were done
  on 8× H100 / 64× B200; a 14B diffusion transformer fine-tune on a single GB10 is far outside the
  tested configuration and will need its own memory tuning (or will not fit). The two-iteration
  smoke test is the realistic target on this machine.
- `decord` (Step 9) is required for the dataset loader; `OpenEXR` is required for the depth
  archives and is covered by Step 5.

## Status and known risks

**Verified on this host (2026-10-19), Steps 1 to 8:** the conda toolchain and CUDA 13.0.2
install, torch 2.9.1+cu130, the from-source builds of `transformer_engine_torch`, `flash-attn
2.8.3.post2`, `vipe_ext` and `gsplat` (commit `0b4dddf`) against CUDA 13.0 and GCC 13.3 on
aarch64, and the Step 8 import and kernel-launch checks (flash-attn, SDPA and Transformer Engine
attention all execute on the GB10). The GUI (`pyngp`) also built after the two ImGui patches.
Not yet done: the checkpoint download, a server/client session, an end-to-end inference run, and
the training-only Step 9 (decord).

Remaining risks:

- **torch 2.9.1 instead of 2.7.1**: `megatron-core 0.12.1`, `diffusers 0.35.2`, `transformers
  4.57.1` and `accelerate 1.8.1` are kept at the original pins. They import, but an end-to-end
  inference run (README Step 1 and Step 2) is the real test of the combination, which is two minor
  torch releases newer than upstream tested.
- **Transformer Engine attention backend on `sm_121`**: TE 2.20 has explicit handling for `(12, 0)`
  but not `(12, 1)`. It will choose between its cuDNN fused attention, flash-attn 2.8.3 and the
  unfused fallback at runtime. A fallback to unfused attention would be slow but not fatal.
- **`torch.compile`**: only used when `use_torch_compile` or the tokenizer compile option is
  enabled. If enabled, `TRITON_PTXAS_PATH` must point at the CUDA 13 `ptxas`; otherwise Triton
  fails with `sm_121a is not defined for option 'gpu-name'`.
- **Memory**: inference of the 14B model plus T5 and VAE fits comfortably in 128 GB unified memory,
  but GPU and host allocations compete. Keep `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` as
  in the README.
- **Alternative stack**: if CUDA 13 causes build failures in a vendored extension, the next-closest
  working option is CUDA 12.9 (`nvidia/label/cuda-12.9.0`, aarch64 available) with torch 2.8.0 or
  2.9.x `cu129` aarch64 wheels, which also know `sm_121`. CUDA 12.8 is not a viable fallback.

## Sources

- PyTorch wheel index for [cu128](https://download.pytorch.org/whl/cu128/torch/) and
  [cu130](https://download.pytorch.org/whl/cu130/torch/) (wheel filenames and metadata).
- [flash-attention `setup.py` v2.6.3](https://github.com/Dao-AILab/flash-attention/blob/v2.6.3/setup.py)
  and [v2.8.3](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/setup.py);
  [flash-attention issue #1969: sm_121 / GB10](https://github.com/Dao-AILab/flash-attention/issues/1969).
- [Transformer Engine `build_tools/utils.py`](https://github.com/NVIDIA/TransformerEngine/blob/main/build_tools/utils.py),
  [`transformer_engine/common/__init__.py`](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/__init__.py),
  [attention utils](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention/dot_product_attention/utils.py),
  [PyPI transformer-engine](https://pypi.org/project/transformer-engine/).
- DGX Spark community guides: [martimramos/dgx-spark-ml-guide](https://github.com/martimramos/dgx-spark-ml-guide),
  [natolambert/dgx-spark-setup](https://github.com/natolambert/dgx-spark-setup),
  [PyTorch forums: NVIDIA DGX Spark support](https://discuss.pytorch.org/t/nvidia-dgx-spark-support/223677),
  [ogulcanaydogan/dgx-spark-llm-stack](https://github.com/ogulcanaydogan/dgx-spark-llm-stack),
  [IryNeko patched flash_attn 2.8.3 for DGX Spark](https://github.com/IryNeko/patched-flash_attn-2.8.3-for-dgx-spark),
  [Fulton-Engineering-Services/dgx-spark-wheels](https://github.com/Fulton-Engineering-Services/dgx-spark-wheels).
- [PyTorch 2.7 release notes](https://pytorch.org/blog/pytorch-2-7/) (Blackwell / cu128 aarch64 wheels),
  [pytorch issue #159207](https://github.com/pytorch/pytorch/issues/159207).
- [gsplat `setup.py` at the pinned commit](https://github.com/nerfstudio-project/gsplat/blob/0b4dddf04cb687367602c01196913cde6a743d70/setup.py),
  [MoGe `pyproject.toml`](https://github.com/microsoft/MoGe/blob/main/pyproject.toml),
  [decord sources](https://github.com/dmlc/decord), [pycg](https://github.com/heiwang1997/pycg).
- PyPI JSON metadata for every pinned package (wheel platform tags), conda-forge and
  `nvidia/label/*` channel searches on this host.
