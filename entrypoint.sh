#!/usr/bin/env bash
# Entrypoint for the lyra2-dgx-spark image.
#
#   zoomgs [args]   README Step 1, Option 1 (preset zoom trajectory). Defaults = INSTALL_DGX_SPARK.md R7.
#   custom [args]   README Step 1, Option 2 (custom trajectory); pass the paths explicitly.
#   recon  [args]   README Step 2 (VIPE + DA3 Gaussian splat reconstruction).
#   server          GUI inference server on port 8000 (needs the Qwen weights under checkpoints/qwen).
#   gui             the native viewer on an in-container NVIDIA X screen, served to the browser through
#                   x11vnc + noVNC on port 6080 (GUI_WEB_SERVING.md, Design 1). Needs the container to run as
#                   root with NVIDIA_DRIVER_CAPABILITIES=all. Environment: LYRA_SERVER_HOST/PORT (inference
#                   server, default server:8000), LYRA_GUI_SCREEN (WxH, default 1920x1080),
#                   LYRA_VNC_PASSWORD (optional), LYRA_NOVNC_PORT (default 6080), extra client flags in
#                   LYRA_CLIENT_ARGS.
#   sync-assets     copy every file/directory of the cloned Lyra-2/assets (kept pristine in the image at
#                   /opt/lyra2/assets-default) that does not yet exist in the mounted host assets
#                   directory. Existing host files are never overwritten. Run by build.sh.
#   <anything else> executed as-is (e.g. `bash`, `python ...`).
#
# Checkpoint resolution, in order:
#   1. LYRA_CHECKPOINTS_DIR            explicit directory containing model/, recon/, text_encoder/, ...
#   2. /models/checkpoints             bind mount (the compose file maps the host checkpoints/ here)
#   3. Hugging Face cache ($HF_HOME)   snapshot of nvidia/Lyra-2.0 with checkpoints/* downloaded via
#                                      `hf download nvidia/Lyra-2.0 --include "checkpoints/*"` (no --local-dir).
#                                      Set LYRA_HF_DOWNLOAD=1 to let the container download it (~97 GB).
#
# Assets: /workspace/Lyra-2/assets is the bind-mounted host directory (./assets on the host by default) and
# is what inference reads (assets/samples, assets/custom_trajectory_examples, ...). It is populated
# by `sync-assets`, not automatically.
set -euo pipefail
cd /workspace/Lyra-2

sync_assets() {
    local src=/opt/lyra2/assets-default dst=/workspace/Lyra-2/assets
    if [[ ! -d "$dst" || ! -w "$dst" ]]; then
        echo "[entrypoint] $dst is not a writable mounted directory; mount the host assets directory there" >&2
        exit 2
    fi
    local before after
    before=$(find "$dst" -type f | wc -l)
    # --update=none: never overwrite existing files; -r: directories; -p: keep timestamps/modes
    cp -rp --update=none "$src"/. "$dst"/
    after=$(find "$dst" -type f | wc -l)
    echo "[entrypoint] sync-assets: $((after - before)) file(s) added to the host assets directory ($after total); existing files untouched" >&2
    if [[ ! -d "$dst/samples" ]]; then
        echo "[entrypoint] warning: $dst/samples still missing after sync" >&2
    fi
}

resolve_checkpoints() {
    local target=""
    if [[ -n "${LYRA_CHECKPOINTS_DIR:-}" ]]; then
        target="$LYRA_CHECKPOINTS_DIR"
    elif [[ -d /models/checkpoints/model ]]; then
        target=/models/checkpoints
    else
        echo "[entrypoint] no checkpoints mounted at /models/checkpoints; looking in the Hugging Face cache ($HF_HOME)" >&2
        target="$(python - <<'EOF'
import os, sys
from huggingface_hub import snapshot_download
try:
    snap = snapshot_download("nvidia/Lyra-2.0", allow_patterns=["checkpoints/*"],
                             local_files_only=os.environ.get("LYRA_HF_DOWNLOAD", "0") != "1")
except Exception as e:
    sys.stderr.write(f"[entrypoint] could not resolve nvidia/Lyra-2.0 from the HF cache: {e}\n"
                     "[entrypoint] either mount your checkpoints directory at /models/checkpoints, set LYRA_CHECKPOINTS_DIR,\n"
                     "[entrypoint] or run once with LYRA_HF_DOWNLOAD=1 to download them into the mounted HF cache.\n")
    sys.exit(2)
print(os.path.join(snap, "checkpoints"))
EOF
)"
    fi
    if [[ ! -d "$target/model" ]]; then
        echo "[entrypoint] $target does not look like a Lyra checkpoint root (no model/ subdirectory)" >&2
        exit 2
    fi
    # /workspace/Lyra-2/checkpoints is a symlink to /models/checkpoints baked into the image.
    # Repoint /models/checkpoints when the real location is elsewhere.
    if [[ "$target" != /models/checkpoints ]]; then
        ln -sfn "$target" /models/checkpoints
    fi
    echo "[entrypoint] checkpoints -> $(readlink -f checkpoints)" >&2
}

start_gui() {
    local screen="${LYRA_GUI_SCREEN:-1920x1080}" w h busid pci
    w="${screen%x*}"; h="${screen#*x}"
    # PCI bus id in X format: DDDDDDDD:BB:DD.F (hex) -> PCI:BB@DDDD:DD:F (decimal), e.g. 0000000F:01:00.0 -> PCI:1@15:0:0
    pci="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 | tr -d ' ')"
    busid="$(python3 - "$pci" <<'EOF'
import sys
dom, bus, devfn = sys.argv[1].split(":"); dev, fn = devfn.split(".")
print(f"PCI:{int(bus,16)}@{int(dom,16)}:{int(dev,16)}:{int(fn,16)}")
EOF
)"
    export DISPLAY=:1
    sed -e "s|@BUSID@|$busid|" -e "s|@W@|$w|g" -e "s|@H@|$h|g" /opt/lyra2/xorg.conf.template > /tmp/xorg-lyra.conf
    echo "[entrypoint] starting Xorg $DISPLAY on GPU $pci ($busid), ${w}x${h}" >&2
    rm -f /tmp/.X1-lock /tmp/.X11-unix/X1
    Xorg "$DISPLAY" -config /tmp/xorg-lyra.conf -noreset -novtswitch -sharevts -nolisten tcp \
         -dpi 96 +extension GLX +extension RANDR +extension RENDER +extension XTEST \
         -logfile /tmp/xorg-lyra.log -verbose 3 > /tmp/xorg-lyra.stdout 2>&1 &
    local i=0
    until [[ -S /tmp/.X11-unix/X1 ]] && xset -display "$DISPLAY" q >/dev/null 2>&1; do
        sleep 0.5; i=$((i+1))
        if [[ $i -gt 40 ]] || ! kill -0 %1 2>/dev/null; then
            echo "[entrypoint] Xorg failed to start; log follows" >&2; tail -n 40 /tmp/xorg-lyra.log /tmp/xorg-lyra.stdout >&2; exit 3
        fi
    done
    grep -E "NVIDIA\(0\): (NVIDIA GPU|Virtual screen size)|\(EE\)" /tmp/xorg-lyra.log | sed 's/^/[Xorg] /' >&2 || true
    openbox > /tmp/openbox.log 2>&1 &
    local vncargs=(-display "$DISPLAY" -forever -shared -rfbport 5900 -localhost -noxdamage -repeat -nopw -bg -o /tmp/x11vnc.log)
    if [[ -n "${LYRA_VNC_PASSWORD:-}" ]]; then
        x11vnc -storepasswd "$LYRA_VNC_PASSWORD" /tmp/vncpasswd >/dev/null 2>&1
        vncargs=("${vncargs[@]/-nopw/-rfbauth}"); vncargs+=(/tmp/vncpasswd)
    fi
    x11vnc "${vncargs[@]}"
    local port="${LYRA_NOVNC_PORT:-6080}"
    # Serve a private copy of noVNC whose vnc.html connects on load: opened without parameters it
    # redirects to itself with autoconnect=true (plus resize=scale so the 1920x1080 screen fits the
    # browser window, and reconnect=true so a dropped WebSocket comes back). "/" goes to vnc.html.
    rm -rf /tmp/novnc && cp -r /usr/share/novnc /tmp/novnc
    python3 - <<'EOF'
import pathlib
p = pathlib.Path("/tmp/novnc/vnc.html"); s = p.read_text()
inject = """<head>
    <script>
    // Lyra 2.0: auto-connect. Append the noVNC URL parameters unless the user supplied their own.
    (function () {
        var q = new URLSearchParams(window.location.search);
        if (!q.has("autoconnect")) {
            q.set("autoconnect", "true");
            if (!q.has("resize")) q.set("resize", "scale");
            if (!q.has("reconnect")) q.set("reconnect", "true");
            window.location.replace(window.location.pathname + "?" + q.toString() + window.location.hash);
        }
    })();
    </script>"""
assert "<head>" in s and "Lyra 2.0: auto-connect" not in s
p.write_text(s.replace("<head>", inject, 1))
pathlib.Path("/tmp/novnc/index.html").write_text(
    '<!doctype html><meta http-equiv="refresh" content="0; url=vnc.html"><title>Lyra 2.0 GUI</title>'
    '<p>Opening the <a href="vnc.html">Lyra 2.0 GUI</a>&hellip;</p>\n')
EOF
    websockify --web /tmp/novnc "$port" localhost:5900 > /tmp/websockify.log 2>&1 &
    echo "[entrypoint] GUI ready: open http://<this host>:$port/ (connects automatically; VNC password: ${LYRA_VNC_PASSWORD:+set}${LYRA_VNC_PASSWORD:-none})" >&2
    # Wait for the inference server's API. The real server loads ~95 GB of weights before uvicorn
    # starts accepting connections, so /metadata can take several minutes to answer; the viewer
    # would crash on its first request without this. LYRA_SERVER_WAIT_S bounds the wait.
    local host="${LYRA_SERVER_HOST:-server}" sport="${LYRA_SERVER_PORT:-8000}" waited=0 limit="${LYRA_SERVER_WAIT_S:-3600}"
    until curl -fsS -m 5 "http://$host:$sport/metadata" >/dev/null 2>&1; do
        if (( waited == 0 )); then echo "[entrypoint] waiting for the inference server at http://$host:$sport (model loading can take minutes)" >&2; fi
        if (( waited >= limit )); then echo "[entrypoint] server not reachable after ${limit}s; giving up" >&2; exit 4; fi
        sleep 5; waited=$((waited + 5))
        if (( waited % 60 == 0 )); then echo "[entrypoint] still waiting for the server (${waited}s)" >&2; fi
    done
    echo "[entrypoint] server is up after ${waited}s; starting the viewer" >&2
    # The viewer itself (foreground; the container ends when it exits).
    export __GLX_VENDOR_LIBRARY_NAME=nvidia
    # shellcheck disable=SC2086
    exec python gui/api/client.py --host "${LYRA_SERVER_HOST:-server}" --port "${LYRA_SERVER_PORT:-8000}" ${LYRA_CLIENT_ARGS:-}
}

case "${1:-zoomgs}" in
    sync-assets)
        sync_assets
        ;;
    gui)
        shift
        start_gui "$@"
        ;;
    zoomgs)
        shift
        resolve_checkpoints
        exec python -m lyra_2._src.inference.lyra2_zoomgs_inference \
            --input_image_path assets/samples --sample_id "${SAMPLE_ID:-4}" --experiment lyra2 \
            --checkpoint_dir checkpoints/model --prompt_dir assets/samples \
            --output_path outputs/zoomgs --num_frames_zoom_in 81 --num_frames_zoom_out 241 \
            --zoom_in_strength 0.5 --zoom_out_strength 1.5 --use_dmd \
            "$@"
        ;;
    custom)
        shift
        resolve_checkpoints
        exec python -m lyra_2._src.inference.lyra2_custom_traj_inference \
            --experiment lyra2 --checkpoint_dir checkpoints/model --output_path outputs/custom_traj \
            "$@"
        ;;
    recon)
        shift
        resolve_checkpoints
        exec python -m lyra_2._src.inference.vipe_da3_gs_recon "$@"
        ;;
    server)
        shift
        resolve_checkpoints
        exec python -m uvicorn server:app --app-dir gui/api --host 0.0.0.0 --port 8000 "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
