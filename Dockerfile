# syntax=docker/dockerfile:1.7
#
# Lyra 2.0 on NVIDIA DGX Spark (GB10, aarch64) -- CLI inference image.
#
# Self-contained: the image clones the Lyra repository (pinned commit + the two inference submodules)
# in its first stage, so the only build input is this repository. It replays
# INSTALL_DGX_SPARK.md Steps 1-7 as executed and verified on a DGX Spark on 2026-10-19. The GUI
# viewer and the training-only decord build are not included.
#
#   docker buildx build -t lyra2-dgx-spark:latest .            # or: docker compose build
#   docker buildx build --build-arg LYRA_REF=main -t lyra2-dgx-spark:main .
#
# No GPU is needed to build. flash-attn, Transformer Engine's torch binding, vipe_ext and gsplat
# are compiled for sm_120, which the GB10 (sm_121) executes. Expect 30-60 minutes.

ARG LYRA_REPO=https://github.com/nv-tlabs/lyra
ARG LYRA_REF=9fffc9adc37004091ecf26ef03abfb3abdf4d59a

# =============================================================================================
# Stage 1: source. Sparse, blob-filtered clone of the Lyra-2 project directory at LYRA_REF plus the
# VIPE and Depth-Anything-3 submodules. Git metadata and the GUI submodules are dropped.
# =============================================================================================
FROM ubuntu:24.04 AS source
ARG LYRA_REPO
ARG LYRA_REF
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates git \
 && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none --no-checkout "$LYRA_REPO" /src \
 && cd /src \
 && git sparse-checkout set Lyra-2 \
 && git checkout "$LYRA_REF" \
 && git submodule update --init --recursive -- \
      Lyra-2/lyra_2/_src/inference/vipe \
      Lyra-2/lyra_2/_src/inference/depth_anything_3 \
      Lyra-2/gui/dependencies/OpenXR-SDK Lyra-2/gui/dependencies/args Lyra-2/gui/dependencies/fmt \
      Lyra-2/gui/dependencies/glfw Lyra-2/gui/dependencies/imgui Lyra-2/gui/dependencies/pybind11 \
      Lyra-2/gui/dependencies/tinylogger Lyra-2/gui/dependencies/zlib \
 && echo "$LYRA_REPO@$(git rev-parse HEAD)" > /src/Lyra-2/SOURCE_REVISION \
 && rm -rf /src/.git \
 && find /src/Lyra-2 -name .git -prune -exec rm -rf {} + \
 && test -f /src/Lyra-2/lyra_2/_src/inference/vipe/setup.py \
 && test -f /src/Lyra-2/lyra_2/_src/inference/depth_anything_3/pyproject.toml \
 && test -f /src/Lyra-2/gui/dependencies/glfw/CMakeLists.txt

# =============================================================================================
# Stage 2: runtime image
# =============================================================================================
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
# Base tools, plus the in-container display stack for the GUI service (Design 1 of
# GUI_WEB_SERVING.md): an X server that loads the NVIDIA X driver injected by the container runtime
# (NVIDIA_DRIVER_CAPABILITIES=all), a window manager, x11vnc for capture + XTEST input, noVNC +
# websockify for the browser, xdotool/scrot for automated checks. The vendor-neutral libglvnd
# libraries dispatch GLX/EGL to the injected libGLX_nvidia / libEGL_nvidia.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git bzip2 file \
      xserver-xorg-core xserver-xorg-legacy x11-xserver-utils xauth xkb-data \
      libglvnd0 libgl1 libglx0 libegl1 libopengl0 \
      openbox x11vnc novnc websockify xdotool scrot procps \
 && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------------------------
# conda (Miniforge: conda-forge by default, no `defaults` channel involved)
# ---------------------------------------------------------------------------------------------
ENV CONDA_DIR=/opt/conda
RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh \
      -o /tmp/miniforge.sh \
 && bash /tmp/miniforge.sh -b -p "$CONDA_DIR" \
 && rm /tmp/miniforge.sh \
 && "$CONDA_DIR/bin/conda" clean -afy
ENV PATH=$CONDA_DIR/bin:$PATH

# ---------------------------------------------------------------------------------------------
# Step 1. conda environment, compiler toolchain, Eigen 3.4
# libgl-devel + xorg-libx11 + xorg-xorgproto: GL/X11 headers for glcontext (a moderngl dependency
# pulled in by MoGe's utils3d), which has no aarch64 wheel and is compiled from source in Step 5b.
# ---------------------------------------------------------------------------------------------
RUN conda create -y -n lyra2 python=3.10 pip cmake ninja libgl ffmpeg packaging \
      libgl-devel xorg-libx11 xorg-xorgproto \
      glew xorg-libxrandr xorg-libxinerama xorg-libxcursor xorg-libxi pkg-config \
      -c conda-forge --override-channels \
 && CONDA_BACKUP_CXX="" conda install -y -n lyra2 gcc=13.3.0 gxx=13.3.0 eigen=3.4.0 zlib \
      -c conda-forge --override-channels \
 && conda clean -afy

# ---------------------------------------------------------------------------------------------
# Step 2. CUDA 13.0.2 toolkit inside the environment (conda-forge supplies gcc_linux-aarch64 13.3.0)
# ---------------------------------------------------------------------------------------------
RUN conda install -y -n lyra2 cuda -c nvidia/label/cuda-13.0.2 -c conda-forge --override-channels \
 && conda clean -afy

ENV CONDA_PREFIX=/opt/conda/envs/lyra2
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CUDA_HOME=$CONDA_PREFIX

# Every later RUN (and the runtime entrypoint) executes inside `conda activate lyra2`. The env's
# activation scripts (cuda-nvcc, gcc/gxx_linux-aarch64) export CFLAGS/CXXFLAGS/CPPFLAGS with
# `-I$CONDA_PREFIX/targets/sbsa-linux/include`, LDFLAGS with the matching -L paths and NVCC's
# -ccbin. That is how the host build found crt/host_config.h: the conda CUDA toolkit keeps its
# headers under targets/sbsa-linux, not $CONDA_PREFIX/include. Without activation the Transformer
# Engine build fails with `fatal error: crt/host_config.h: No such file or directory`.
# The wrapper re-applies the Step 4 overrides after activation (the cuda-nvcc script would
# otherwise reset TORCH_CUDA_ARCH_LIST to a long list and CC/CXX to the -cc/-c++ wrappers).
RUN cat > /usr/local/bin/with-env <<'EOF'
#!/bin/bash
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate lyra2
export CC="$CONDA_PREFIX/bin/aarch64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/aarch64-conda-linux-gnu-g++"
export TORCH_CUDA_ARCH_LIST="${LYRA_TORCH_CUDA_ARCH_LIST:-12.1}"
exec "$@"
EOF
RUN chmod +x /usr/local/bin/with-env
SHELL ["/usr/local/bin/with-env", "/bin/bash", "-c"]

# ---------------------------------------------------------------------------------------------
# Step 3. PyTorch 2.9.1 + torchvision 0.24.1, CUDA 13.0 wheels (aarch64)
# ---------------------------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130

# ---------------------------------------------------------------------------------------------
# Step 4. Build environment (CUDA 13 pip wheels live under nvidia/cu13; cuDNN/NCCL keep their dirs)
# ---------------------------------------------------------------------------------------------
ENV SITE=$CONDA_PREFIX/lib/python3.10/site-packages
ENV CPATH=$CUDA_HOME/include:$SITE/nvidia/cu13/include:$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include
ENV LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cu13/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/nccl/lib:$CUDA_HOME/lib64
ENV CC=$CONDA_PREFIX/bin/aarch64-conda-linux-gnu-gcc
ENV CXX=$CONDA_PREFIX/bin/aarch64-conda-linux-gnu-g++
ENV TORCH_CUDA_ARCH_LIST=12.1
ENV FLASH_ATTN_CUDA_ARCHS=120
ENV NVTE_CUDA_ARCHS=120
ENV TRITON_PTXAS_PATH=$CONDA_PREFIX/bin/ptxas
ARG MAX_JOBS=8
ENV MAX_JOBS=$MAX_JOBS

# ---------------------------------------------------------------------------------------------
# Step 5. Python dependencies
# ---------------------------------------------------------------------------------------------
WORKDIR /workspace/Lyra-2
COPY --from=source /src/Lyra-2/requirements.txt ./requirements.txt

# 5a. pinned requirements with the DGX Spark substitutions (same sed as INSTALL_DGX_SPARK.md)
RUN sed -e 's/^OpenEXR==3.2.3$/OpenEXR==3.2.5/' \
        -e 's/^tensorstore==0.1.45$/tensorstore==0.1.78/' \
        -e 's/^decord==0.6.0$/# decord==0.6.0   # no aarch64 wheel; training-only/' \
        -e 's/^numcodecs==0.13.1$/numcodecs==0.13.1\n--no-binary numcodecs/' \
        -e 's/^protobuf==4.25.3$/protobuf>=6.31.1   # onnx 1.23 (transformer_engine_torch) needs >=6.31.1/' \
        -e 's/^typing-extensions==4.14.1$/typing-extensions>=4.16.0   # anyio 4.15 (httpx, fastapi) needs >=4.16/' \
        requirements.txt > requirements-dgx-spark.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps -r requirements-dgx-spark.txt

# 5b. MoGe without its gradio dependency chain
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps "git+https://github.com/microsoft/MoGe.git" \
 && pip install --no-deps \
      "utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37" \
      "pipeline @ git+https://github.com/EasternJournalist/pipeline.git@1c511390d90226c00c101f34b84df26a0f8789b4" \
 && pip install moderngl click "trimesh>=4.11"

# 5c. Transformer Engine 2.20.2 (CUDA 13 native; torch binding compiled here)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-build-isolation "transformer_engine[pytorch]==2.20.2"

# 5d. transitive dependencies skipped by --no-deps (antlr4 must stay 4.9.x for omegaconf/hydra)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "antlr4-python3-runtime==4.9.3" wcwidth yacs wadler-lindig ml-dtypes asciitree \
                docstring-parser shtab typeguard matplotlib-inline exceptiongroup \
                starlette annotated-doc httpcore h11 anyio sniffio \
                absl-py grpcio markdown tensorboard-data-server werkzeug \
                iniconfig pluggy execnet pathspec trove-classifiers setuptools-scm

# 5e. re-apply the pins
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps -r requirements-dgx-spark.txt

# ---------------------------------------------------------------------------------------------
# Step 6. Flash Attention 2.8.3.post2 for sm_120
# ---------------------------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    pip install --no-build-isolation --no-binary :all: flash-attn==2.8.3.post2

# ---------------------------------------------------------------------------------------------
# Step 7. Repository + vendored CUDA extensions
# ---------------------------------------------------------------------------------------------
COPY --from=source /src/Lyra-2/ /workspace/Lyra-2/
# DA3's hatch-vcs versioning needs git metadata, which the source stage stripped.
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0

# 7a. VIPE (python-pycg hard-requires usd-core, which has no aarch64 wheel; pycg imports it lazily)
RUN --mount=type=cache,target=/root/.cache/pip \
    USE_SYSTEM_EIGEN=1 pip install --no-build-isolation --no-deps -e lyra_2/_src/inference/vipe \
 && pip install --no-deps python-pycg \
 && pip install calmsize screeninfo pyquaternion plyfile trimesh \
                rerun-sdk timm kornia gdown viser click

# 7b. Depth Anything 3 (--no-deps: pycolmap has no aarch64 wheel, numpy<2 would downgrade numpy) + gsplat
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-build-isolation --no-deps -e lyra_2/_src/inference/depth_anything_3 \
 && pip install trimesh evo e3nn moviepy plyfile pillow_heif typer pre-commit uvicorn addict \
 && pip install --no-build-isolation \
      "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70"

# 7c. make DA3's COLMAP export import lazy (pycolmap has no aarch64 wheel; Lyra uses mini_npz export only)
RUN python - <<'EOF'
import pathlib
p = pathlib.Path("lyra_2/_src/inference/depth_anything_3/src/depth_anything_3/utils/export/__init__.py")
s = p.read_text()
if "from .colmap import export_to_colmap\n" in s and 'from .colmap import export_to_colmap\n        export_to_colmap(' not in s:
    s = s.replace("from .colmap import export_to_colmap\n", "")
    s = s.replace('    elif export_format == "colmap":\n        export_to_colmap(',
                  '    elif export_format == "colmap":\n        from .colmap import export_to_colmap\n        export_to_colmap(')
    p.write_text(s)
    print("patched", p)
else:
    print("already patched", p)
EOF

# ---------------------------------------------------------------------------------------------
# GUI viewer (pyngp) build, per the GUI section of INSTALL_DGX_SPARK.md. No GPU at build time:
# TCNN_CUDA_ARCHITECTURES pins sm_120 (runs on the GB10), after the CMake arch cap is raised.
# ---------------------------------------------------------------------------------------------
ARG WITH_GUI=1
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$WITH_GUI" = "1" ]; then \
      pip install commentjson pybind11 pyexr \
   && sed -i 's/^\(\s*\)set(LATEST_SUPPORTED_CUDA_ARCHITECTURE 90)$/\1set(LATEST_SUPPORTED_CUDA_ARCHITECTURE 90)\n\1if (CUDA_VERSION VERSION_GREATER_EQUAL 12.9)\n\1\tset(LATEST_SUPPORTED_CUDA_ARCHITECTURE 121)\n\1endif()/' gui/CMakeLists.txt \
   && grep -q "LATEST_SUPPORTED_CUDA_ARCHITECTURE 121" gui/CMakeLists.txt \
   && sed -i 's/IMGUI_VERSION_NUM >= 18700/IMGUI_VERSION_NUM >= 18800/g' gui/dependencies/imguizmo/ImGuizmo.cpp \
   && test "$(grep -c 'IMGUI_VERSION_NUM >= 18800' gui/dependencies/imguizmo/ImGuizmo.cpp)" = "6" \
   && sed -i 's/\bImGuiMod_\(Alt\|Ctrl\|Shift\|Super\)\b/ImGuiKeyModFlags_\1/g' gui/src/python_api.cu \
   && test "$(grep -c 'ImGuiKeyModFlags_' gui/src/python_api.cu)" = "4" \
   && export CMAKE_PREFIX_PATH="$CONDA_PREFIX" CUDACXX="$CONDA_PREFIX/bin/nvcc" TCNN_CUDA_ARCHITECTURES=120 \
   && cmake gui -B gui/build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DNGP_BUILD_WITH_PYTHON_BINDINGS=ON \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DPython_EXECUTABLE="$CONDA_PREFIX/bin/python" \
   && cmake --build gui/build --config RelWithDebInfo -j"$MAX_JOBS" \
   && ls gui/build/pyngp*.so \
   && find gui/build -name "*.o" -delete; \
    fi

# ---------------------------------------------------------------------------------------------
# Runtime layout
#   /workspace/Lyra-2/checkpoints -> /models/checkpoints   (bind-mounted, or resolved from the HF cache)
#   /workspace/Lyra-2/assets                                (bind-mounted host directory, read-write;
#                                                            `sync-assets` copies the cloned assets into it)
#   /opt/lyra2/assets-default                               (pristine copy of the cloned Lyra-2/assets)
#   /workspace/Lyra-2/outputs                               (bind-mounted)
#   /cache/huggingface, /cache/torch                        (bind-mounted caches: umt5 tokenizer, MoGe, VIPE priors)
# ---------------------------------------------------------------------------------------------
COPY entrypoint.sh /workspace/Lyra-2/docker/entrypoint.sh
COPY xorg.conf.template /opt/lyra2/xorg.conf.template
ENV HF_HOME=/cache/huggingface
ENV TORCH_HOME=/cache/torch
ENV HOME=/home/lyra
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV PYTHONPATH=/workspace/Lyra-2:/workspace/Lyra-2/gui/api
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN mkdir -p /models /cache/huggingface /cache/torch /home/lyra /opt/lyra2 outputs \
 && cp -a assets /opt/lyra2/assets-default \
 && chmod 1777 /models /cache /cache/huggingface /cache/torch /home/lyra outputs assets \
 && ln -sfn /models/checkpoints checkpoints \
 && chmod +x docker/entrypoint.sh

# Import-level self check (no GPU needed): every component that Step 8 verifies on the host.
RUN python -c "import torch, flash_attn, transformer_engine.pytorch, vipe_ext, depth_anything_3.api, moge.model.v1; print('image self-check OK: torch', torch.__version__, '| flash_attn', flash_attn.__version__, '| source', open('SOURCE_REVISION').read().strip())"

ENTRYPOINT ["/usr/local/bin/with-env", "/workspace/Lyra-2/docker/entrypoint.sh"]
CMD ["zoomgs"]
