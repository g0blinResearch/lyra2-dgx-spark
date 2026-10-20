# Serving the Lyra 2.0 GUI through a web page: feasibility study

Research only (2026-10-20); no code changes. Question: does the framework the GUI is built on
allow the same GUI, with the same features, to be served via a browser, either natively or through
known methods?

## Short answer

The GUI framework itself (instant-ngp's "testbed": a C++/CUDA renderer with Dear ImGui widgets,
ImGuizmo gizmos, GLFW windowing and OpenGL presentation, driven from Python through `pyngp`) has
**no built-in web mode**. There are three known ways to put it in a browser, with very different
cost and fidelity:

| Route | Same features? | Effort | Verdict |
|---|---|---|---|
| **A. Stream the unchanged native viewer** (noVNC, or WebRTC/H.264 via Selkies; Sunshine/Moonlight already runs on this host but has no browser client) | Yes, pixel-exact, everything except VR and file drag-and-drop | hours | **the only route that keeps the same GUI**; recommended |
| **B. Stream the ImGui draw lists** (imgui-ws) with the CUDA viewport as a texture | UI yes, viewport as a compressed texture; gizmos work since they use ImGui input | days to weeks of C++ work | possible, not worth it over A |
| **C. Rebuild the client as a web app** on top of the existing headless `pyngp` renderer and the existing FastAPI server | Only what gets re-implemented; camera-path authoring is the big piece | weeks | the long-term "real web GUI"; verified feasible because `pyngp` renders without a display |

"Exponential" or "free" is not on offer here either: the trade is fidelity-for-free (A) versus a
real port (C).

## What the GUI actually is

From `gui/src` (3,904 lines in `testbed.cu`, 725 in `camera_path.cu`, 844 in `python_api.cu`)
and `gui/api/client.py` (1,462 lines):

- **Rendering** is CUDA: the viewport (reprojected point cloud from the seeded image and generated
  frames, `add_src_view` / `reproject`) is rendered by CUDA kernels into a render buffer, then
  blitted to the screen through an OpenGL texture. OpenGL is only used for presentation, the
  ImGui overlay and the gizmo lines.
- **UI** is Dear ImGui: ~90 widgets (buttons, sliders, inputs, combos, collapsing headers) in
  `testbed.cu` and `camera_path.cu`, plus two `ImGuizmo::Manipulate` gizmos (keyframe
  translate/rotate in the viewport). ImGui reads mouse and keyboard from GLFW through `ImGui::GetIO`.
- **Windowing** is GLFW 3.4 (X11 or Wayland, OpenGL core profile; optional Vulkan for DLSS, off
  here; optional OpenXR for VR). `init_window(w, h, hidden, second_window)` can create a *hidden*
  window, but a hidden GLFW window still needs a display server and a GL context.
- **Loop**: `client.py` calls `testbed.init_window()` then `while testbed.frame():`; `frame()`
  returns immediately when there is no window (`if (m_render_window) {...}` in `testbed.cu`), so
  the ImGui UI only exists inside a window.
- **Client/server split already exists**: all inference runs in the FastAPI server
  (`gui/api/server.py`, allowlisted JSON messages over HTTP); the viewer is a pure client. This is
  what makes routes A and C clean: the GPU-heavy part is already behind a network API.

**Verified on this host:** with `DISPLAY` unset, `pyngp.Testbed()` constructs and
`testbed.render(320, 180, 1, False)` returns a float32 RGBA frame in 33 ms. The renderer does not
need a window; only the ImGui/ImGuizmo UI does.

## Route A: stream the native viewer (same GUI, pixel-exact)

Run `client.py` unchanged on an X display backed by the NVIDIA GPU and stream that display to the
browser. Everything the GUI does (seeding, camera-path keyframes, gizmos, Generate/Revert, the video
player, keyboard shortcuts) works because it *is* the native program. Lost: OpenXR VR (needs a
local headset) and dragging files from the browser onto the viewport (use the **Path** box).

This host already has what is needed: an NVIDIA-accelerated Xorg session on `:0` (GDM, X11,
`Xorg` visible in `nvidia-smi`), and Sunshine is running (`/usr/bin/sunshine`), which already
streams this desktop to Moonlight clients with NVENC, but Moonlight has no official browser client.

Options, all packaged for Ubuntu 24.04 arm64 except where noted:

1. **noVNC + x11vnc (or TigerVNC `x0vncserver`) on the existing `:0`**: `apt install novnc
   websockify x11vnc`; `x11vnc -display :0 -localhost`; `websockify --web /usr/share/novnc 6080
   localhost:5900`; open `http://<spark>:6080/vnc.html`. Simplest; shares the physical desktop;
   VNC's tile encoding is fine for the UI but the 60 fps viewport and the video player will look
   choppy. Secure it with an SSH tunnel.
2. **A second, private NVIDIA X screen** so the viewer does not take over the desktop: an
   Xorg instance on `:1` with the NVIDIA driver and `--allow-empty-initial-configuration` (the
   usual headless-NVIDIA recipe; the VirtualGL "HeadlessNV" page documents it) or, where available,
   VirtualGL + TurboVNC (both ship arm64 `.deb`s on their own sites; neither is in Ubuntu's apt).
   Then run `DISPLAY=:1 python gui/api/client.py ...` and VNC that screen.
3. **Selkies (WebRTC / WebSocket, H.264)**: the modern GPU-desktop-in-browser stack. `selkies`
   2.0.0 is a pure-Python wheel (installs on aarch64), but the ready-made
   `ghcr.io/selkies-project/nvidia-glx-desktop` and `nvidia-egl-desktop` images are published for
   **amd64 only** (checked with `docker manifest inspect`), so on the Spark it would be a manual
   install; NVENC through GStreamer on SBSA arm64 is unverified, software x264 works. Smoothest
   result for the viewport if the encoder path works; the most setup.
4. **Sunshine → Moonlight** (already running): not a web page, but zero setup for a desktop or
   mobile Moonlight app, with NVENC and sub-frame latency.

Feature retention: complete. Latency and smoothness: 1 < 3 ≈ 4. Multi-user: one session per
display. Containers: the viewer (and option 2's Xorg) could live in a container with the NVIDIA
runtime, but the current Docker image deliberately excludes the GUI build (needs the GUI
submodules, GLEW, X11 libs and the three local patches); adding it is mechanical.

## Route B: stream ImGui draw lists (imgui-ws)

[imgui-ws](https://github.com/ggerganov/imgui-ws) (MIT) serialises Dear ImGui's draw data every
frame, sends frame deltas over WebSockets, renders them with WebGL in the browser, supports several
clients at once, and carries textures and browser input back to the app. Because the Lyra UI,
including ImGuizmo, drives everything through `ImGui::GetIO`, the widgets and gizmos would work in
the browser unchanged. The viewport, however, is a CUDA-rendered image: it would have to be
uploaded as a texture each frame (1534×831 RGBA ≈ 5 MB raw at 60 fps), so a JPEG/H.264 side channel
for the viewport is needed in practice. The change is in C++: replace the GLFW/OpenGL backend in
`testbed.cu` with an offscreen CUDA render plus imgui-ws, keep `frame()` running without
`m_render_window`. Days to weeks for someone comfortable in the testbed code, and the result is still
"the native UI drawn remotely", not a web-native interface. Route A gives the same user experience
without touching the code.

## Route C: a web client on the headless renderer (the real port)

What exists today makes this feasible:

- The server side needs nothing new: `gui/api/server.py` already is a network API.
- `pyngp` renders headlessly (verified above): `Testbed()`, `add_src_view` / `reproject`,
  `render()`, `camera_path` (keyframes, `get_pos`, `get_keyframe`, load/save), `fov`,
  `camera_matrix` are all exposed without a window. A Python web process can hold the testbed,
  push rendered frames to the browser (WebSocket + JPEG, or WebRTC) and accept camera moves and
  path edits from it.
- `client.py` holds the protocol logic (seeding requests, chunked inference requests, revert,
  video playback) in plain Python and can be reused.

What has to be rebuilt: the entire UI layer. The camera-path authoring window (record, keyframe
add/split/dup/delete/set, gizmo manipulation, FOV, play/preview) and the panel widgets would be
re-implemented in the browser (Three.js `TransformControls` for the gizmo; or
[viser](https://viser.studio/), which this environment already installs as a VIPE dependency and
provides a Python-driven 3D viewer with GUI panels, camera control and a WebXR mode). VR would
become WebXR rather than OpenXR. Estimated effort: weeks; result: a true web GUI, multi-user
capable, no X server anywhere.

## Recommendation

- **Now**: Route A, option 1 (noVNC on `:0`) for access from any browser in minutes, or
  Sunshine/Moonlight (already running) for the smoothest interactive experience. Option 2 if the
  viewer should not occupy the desktop. Nothing in the repository changes.
- **If a browser-native GUI is a product goal**: Route C, reusing `client.py`'s protocol code and
  `pyngp`'s headless renderer, with viser as the fastest path to a working UI.
- **Skip** Route B unless the goal is specifically "native UI in a browser with zero UI rewrite"
  and someone wants to own the C++ backend change.

## Sources

- Code inspected: `gui/src/testbed.cu`, `camera_path.cu`, `python_api.cu`, `main.cu`,
  `gui/CMakeLists.txt`, `gui/api/client.py`, `server.py`; GLFW 3.4 headers in `gui/dependencies`.
- Host checks: Xorg on `:0` (X11 session, NVIDIA GPU), Sunshine running, apt candidates for
  `novnc`, `websockify`, `x11vnc`, `tigervnc-standalone-server`, `xserver-xorg-video-dummy`
  (present), `virtualgl`/`turbovnc` (not in apt), `docker manifest inspect` of the Selkies NVIDIA
  images (amd64 only), PyPI `selkies` 2.0.0 (pure Python), headless `pyngp` render test.
- [imgui-ws](https://github.com/ggerganov/imgui-ws), [netImgui](https://github.com/sammyfreg/netImgui)
  (native remote-UI server, no browser), [Selkies](https://github.com/selkies-project/selkies) and
  [docs](https://docs.selkies.io/latest/), [docker-selkies-glx-desktop](https://github.com/selkies-project/docker-selkies-glx-desktop),
  [noVNC/websockify setup](https://dev.to/datawookie/websockify-novnc-behind-an-nginx-proxy-1gia),
  [VirtualGL + TurboVNC on NVIDIA](https://stfc.atlassian.net/wiki/spaces/CLOUDKB/pages/437256268/Setup+Nvidia+Accelerated+Graphics),
  [NVIDIA forum: headless remote server setup](https://forums.developer.nvidia.com/t/headless-remote-server-setup/52225),
  [Dear ImGui as a web application (discussion)](https://github.com/ocornut/imgui/discussions/5428),
  [instant-ngp build/GUI dependencies](https://deepwiki.com/NVlabs/instant-ngp).
