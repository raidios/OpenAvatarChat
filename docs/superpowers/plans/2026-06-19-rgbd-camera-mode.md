# RGB-D Camera Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the deployed robot client start from an Orbbec SDK RGB-D camera path so existing color consumers keep working and aligned depth is available for future object-depth lookup.

**Architecture:** Add an RGB-D-capable camera wrapper that keeps the existing `SharedCamera` color API while optionally storing an aligned depth frame from the same SDK frameset. Route CLI through an explicit `--camera-mode` so UVC tuning remains available only for color-only fallback, while service defaults use `orbbec-rgbd-sdk`.

**Tech Stack:** Python 3.11, OpenCV-compatible color frames, OrbbecSDK v1 ctypes shim when present, unittest with mocked SDK/cv2.

---

### Task 1: Add Explicit Camera Modes

**Files:**
- Modify: `client/main.py`
- Modify: `client/camera.py`
- Test: `tests/unittest/test_camera_controls.py`

- [ ] Add `--camera-mode color|rgbd-sdk` with default `rgbd-sdk`.
- [ ] Preserve `--camera` as the color-only device selector and map `rgbd-sdk` to the Orbbec SDK backend.
- [ ] Add a unit test that `SharedCamera(..., rgbd=True)` exposes a depth snapshot API and ignores V4L2 controls for non-V4L2 camera IDs.

### Task 2: Implement RGB-D SDK Wrapper

**Files:**
- Modify: `client/camera.py`
- Test: `tests/unittest/test_camera_rgbd.py`

- [ ] Extend `_OrbbecBackend` to open `streams=("depth", "color")` with `d2c="hw"` when RGB-D is requested.
- [ ] In RGB-D mode, read `read_rgbd_sdk()` and cache `aligned_depth_mm()` alongside the BGR color frame.
- [ ] Keep the existing `read()` return shape unchanged for current chat/marker consumers.
- [ ] Add tests with a fake `orbbec_gemini` module for color-only and RGB-D paths.

### Task 3: Update Deployment Defaults and Docs

**Files:**
- Modify: `scripts/openavatarchat-client.service.example`
- Modify: `docs/PROJECT_STATE.md`
- Modify: `docs/TODO.md`

- [ ] Replace UVC 60 FPS defaults with `--camera-mode rgbd-sdk --camera orbbec`.
- [ ] Document that UVC exposure parameters are color-only fallback knobs.
- [ ] Run local pure-Python tests and car-side import/unittest checks.
