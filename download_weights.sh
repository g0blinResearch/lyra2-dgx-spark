#!/usr/bin/env bash
# Download every model file the image needs into the directories compose mounts.
#
#   ./download_weights.sh                 # Lyra 2.0 checkpoints (~97 GB) + Qwen3-VL-4B (~8 GB, GUI server) + runtime cache (~1.2 GB)
#   ./download_weights.sh --skip-qwen     # CLI inference only (no GUI server)
#   ./download_weights.sh --recon-priors  # additionally prefetch the depth priors used by Step 2 reconstruction
#   ./download_weights.sh --dry-run       # print what would be downloaded and where
#
# Targets (overridable in .env or the environment, same variables as compose.yml):
#   LYRA_CHECKPOINTS_DIR  default ./checkpoints   nvidia/Lyra-2.0 "checkpoints/*" -> model/, recon/, text_encoder/, vae/,
#                                                  image_encoder/, lora/ ; Qwen -> qwen/Qwen3-VL-4B-Instruct/
#   HF_HOME               default ~/.cache/huggingface   Hugging Face hub cache, mounted into the containers; receives the
#                                                  files the inference code fetches at run time (google/umt5-xxl tokenizer,
#                                                  Ruicheng/moge-vitl weights) so the containers never need the network
#   HF_TOKEN              optional                 passed to the downloader if the hub requires authentication
#
# Downloads are resumable and idempotent (existing files are skipped). The `hf` CLI is used if it is
# on PATH; otherwise the lyra2-dgx-spark image (which contains it) is used via `docker run`, so no
# Python environment is required on the host beyond Docker.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ -f .env ]] && set -a && source .env && set +a

CKPT="${LYRA_CHECKPOINTS_DIR:-./checkpoints}"
HFH="${HF_HOME:-$HOME/.cache/huggingface}"
IMAGE="${LYRA_IMAGE:-lyra2-dgx-spark:latest}"
DO_LYRA=1 DO_QWEN=1 DO_CACHE=1 DO_RECON=0 DRY=0
for a in "$@"; do
    case "$a" in
        --skip-lyra) DO_LYRA=0 ;;
        --skip-qwen) DO_QWEN=0 ;;
        --skip-cache) DO_CACHE=0 ;;
        --recon-priors) DO_RECON=1 ;;
        --dry-run) DRY=1 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown option: $a" >&2; exit 2 ;;
    esac
done

mkdir -p "$CKPT" "$HFH"
CKPT_ABS="$(cd "$CKPT" && pwd)"; HFH_ABS="$(cd "$HFH" && pwd)"

# run_hf <args...>: `hf <args>` on the host if available, else inside the image with the two
# directories mounted at the same absolute paths, as the invoking user.
if command -v hf >/dev/null 2>&1; then
    HF_MODE="host hf CLI ($(hf version 2>/dev/null | head -1))"
    run_hf() { HF_HOME="$HFH_ABS" hf "$@"; }
else
    HF_MODE="hf CLI inside $IMAGE (docker run)"
    run_hf() {
        docker run --rm --user "$(id -u):$(id -g)" \
            -e HF_HOME="$HFH_ABS" ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} -e HF_HUB_ENABLE_HF_TRANSFER=0 \
            -v "$HFH_ABS:$HFH_ABS" -v "$CKPT_ABS:$CKPT_ABS" \
            --entrypoint /usr/local/bin/with-env "$IMAGE" hf "$@"
    }
    if [[ $DRY -eq 0 ]] && ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
        echo "neither the 'hf' CLI nor the $IMAGE image is available; run ./build.sh first or 'pip install huggingface_hub'" >&2
        exit 3
    fi
fi

say() { echo "== $*" >&2; }
run() { if [[ $DRY -eq 1 ]]; then echo "[dry-run] $*"; else "$@"; fi; }

say "downloader: $HF_MODE"
say "checkpoints -> $CKPT_ABS | HF cache -> $HFH_ABS"
avail_gb=$(df -BG --output=avail "$CKPT_ABS" | tail -1 | tr -dc '0-9')
need_gb=$(( DO_LYRA * 100 + DO_QWEN * 9 + DO_CACHE * 2 + DO_RECON * 8 ))
[[ "$avail_gb" -lt "$need_gb" ]] && say "warning: ${avail_gb} GB free, about ${need_gb} GB needed"

# 1. nvidia/Lyra-2.0 checkpoints/* -> $CKPT/{model,recon,text_encoder,vae,image_encoder,lora}
if [[ $DO_LYRA -eq 1 ]]; then
    # the DCP checkpoint lives one level down: checkpoints/model/model/{.metadata,__*.distcp}
    if [[ -f "$CKPT_ABS/model/model/.metadata" && -f "$CKPT_ABS/text_encoder/negative_prompt.pt" && -f "$CKPT_ABS/recon/model.pt" && $DRY -eq 0 ]]; then
        say "Lyra 2.0 checkpoints already present in $CKPT_ABS (model/model/.metadata found); skipping"
    else
        say "Lyra 2.0 checkpoints (~97 GB): nvidia/Lyra-2.0 checkpoints/*"
        # The repo nests everything under checkpoints/; download into a staging dir on the same
        # filesystem and move the subdirectories up so $CKPT/model etc. match what the code expects.
        STAGE="$CKPT_ABS/.download"
        if [[ $DRY -eq 0 ]]; then
            run_hf download nvidia/Lyra-2.0 --include "checkpoints/*" --local-dir "$STAGE" >/dev/null
            for d in "$STAGE"/checkpoints/*/; do
                name="$(basename "$d")"
                mkdir -p "$CKPT_ABS/$name"
                cp -rn --preserve=timestamps "$d". "$CKPT_ABS/$name/" 2>/dev/null || cp -rp "$d". "$CKPT_ABS/$name/"
            done
            rm -rf "$STAGE"
        else
            echo "[dry-run] hf download nvidia/Lyra-2.0 --include 'checkpoints/*' --local-dir $STAGE ; then move $STAGE/checkpoints/* -> $CKPT_ABS/"
        fi
    fi
fi

# 2. Qwen/Qwen3-VL-4B-Instruct -> $CKPT/qwen/Qwen3-VL-4B-Instruct (GUI server prompt rewriting)
if [[ $DO_QWEN -eq 1 ]]; then
    say "Qwen3-VL-4B-Instruct (~8 GB) -> $CKPT_ABS/qwen/Qwen3-VL-4B-Instruct"
    if [[ $DRY -eq 1 ]]; then echo "[dry-run] hf download Qwen/Qwen3-VL-4B-Instruct --local-dir $CKPT_ABS/qwen/Qwen3-VL-4B-Instruct";
    else run_hf download Qwen/Qwen3-VL-4B-Instruct --local-dir "$CKPT_ABS/qwen/Qwen3-VL-4B-Instruct" >/dev/null; fi
fi

# 3. Run-time downloads the inference code makes through the HF hub cache
if [[ $DO_CACHE -eq 1 ]]; then
    say "HF hub cache: google/umt5-xxl tokenizer (~20 MB) and Ruicheng/moge-vitl (~1.2 GB) -> $HFH_ABS/hub"
    if [[ $DRY -eq 1 ]]; then
        echo "[dry-run] hf download google/umt5-xxl tokenizer_config.json special_tokens_map.json spiece.model tokenizer.json"
        echo "[dry-run] hf download Ruicheng/moge-vitl"
    else
        # explicit filenames (positional) rather than --include: the tokenizer is all the code needs,
        # not the 11 GB of umt5-xxl weights
        run_hf download google/umt5-xxl tokenizer_config.json special_tokens_map.json spiece.model tokenizer.json >/dev/null
        run_hf download Ruicheng/moge-vitl >/dev/null
    fi
fi

# 4. Optional: depth priors VIPE/DA3 fetch on first use of the Step 2 reconstruction
if [[ $DO_RECON -eq 1 ]]; then
    say "reconstruction priors: depth-anything/DA3-GIANT and depth-anything/DA3METRIC-LARGE -> $HFH_ABS/hub"
    for r in depth-anything/DA3-GIANT depth-anything/DA3METRIC-LARGE; do
        if [[ $DRY -eq 1 ]]; then echo "[dry-run] hf download $r"; else run_hf download "$r" >/dev/null; fi
    done
    say "note: VIPE's torch.hub priors (Metric3D, Video-Depth-Anything) are fetched into TORCH_HOME on first use if the pipeline selects them"
fi

if [[ $DRY -eq 0 ]]; then
    say "done. layout:"
    for d in model recon text_encoder vae image_encoder lora qwen; do
        [[ -e "$CKPT_ABS/$d" ]] && printf '   %-14s %s\n' "$d/" "$(du -sh "$CKPT_ABS/$d" 2>/dev/null | cut -f1)" >&2 || printf '   %-14s (missing)\n' "$d/" >&2
    done
    printf '   %-14s %s\n' "hub cache" "$(ls "$HFH_ABS/hub" 2>/dev/null | grep -E "models--(google--umt5-xxl|Ruicheng--moge-vitl)" | tr '\n' ' ')" >&2
fi
