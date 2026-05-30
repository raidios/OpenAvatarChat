"""OpenCV display helpers: local DISPLAY for SSH, X11 auth, and HighGUI probe."""

import glob
import os
import re
import subprocess
import sys

import cv2


def _is_local_numeric_display(display: str) -> bool:
    """True for :0, :1 — Pi HDMI / local Xorg, not SSH X forward (e.g. localhost:10.0)."""
    d = (display or "").strip()
    return bool(re.match(r"^:\d+$", d))


def _try_set_xauthority():
    """
    SSH often exports DISPLAY=:0 but XAUTHORITY points at an SSH/fake cookie that
    cannot talk to the real :0 → 'Authorization required'.

    For local :N displays we pick a readable cookie (~/.Xauthority, gdm, mutter),
    replacing a wrong XAUTHORITY.

    Set OPENCV_KEEP_XAUTH=1 to never override an existing XAUTHORITY.
    """
    disp = (os.environ.get("DISPLAY") or "").strip()
    local = _is_local_numeric_display(disp)
    if not (local or disp.startswith(":") or "localhost" in disp):
        return

    uid = os.getuid()
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, ".Xauthority"),
        f"/run/user/{uid}/gdm/Xauthority",
    ]
    candidates.extend(sorted(glob.glob(f"/run/user/{uid}/.mutter-Xwayland-auth.*")))

    def first_readable(paths):
        for path in paths:
            if path and os.path.isfile(path) and os.access(path, os.R_OK):
                return path
        return None

    chosen = first_readable(candidates)
    keep = os.environ.get("OPENCV_KEEP_XAUTH", "").strip()

    current = (os.environ.get("XAUTHORITY") or "").strip()
    if keep and current and os.path.isfile(current) and os.access(current, os.R_OK):
        return

    if current and (not os.path.isfile(current) or not os.access(current, os.R_OK)):
        print(f"[display] Ignoring unreadable XAUTHORITY={current!r}")
        del os.environ["XAUTHORITY"]
        current = ""

    if not local:
        if not current and chosen:
            os.environ["XAUTHORITY"] = chosen
            print(f"[display] XAUTHORITY was unset; using {chosen}")
        elif not current:
            print(
                "[display] No readable Xauthority cookie. "
                "You may need `xhost` on the Pi desktop (see help if windows fail)."
            )
        return

    if chosen:
        if current and os.path.normpath(current) != os.path.normpath(chosen):
            print(
                f"[display] For local display {disp}, switching XAUTHORITY to {chosen} "
                f"(was {current!r}; SSH cookies often break :0)"
            )
        elif not current:
            print(f"[display] Using XAUTHORITY={chosen}")
        os.environ["XAUTHORITY"] = chosen
        return

    print(
        "[display] No readable .Xauthority under this user. "
        "Use the graphical desktop once, or on the Pi monitor run: "
        f"xhost +SI:localuser:{os.environ.get('USER', 'USER')}"
    )


def ensure_local_display():
    """
    For Pi + SSH + local HDMI: normalize DISPLAY and set XAUTHORITY for :0.
    """
    display = os.environ.get("DISPLAY")
    if not display:
        os.environ["DISPLAY"] = ":0"
        print("[display] DISPLAY not set (SSH session?), using :0 (local screen)")
    else:
        os.environ["DISPLAY"] = display.strip()
        print(f"[display] Using DISPLAY={os.environ['DISPLAY']}")

    _try_set_xauthority()


def _opencv_build_gui_line() -> str | None:
    for line in cv2.getBuildInformation().splitlines():
        s = line.strip()
        if s.startswith("GUI:"):
            return s
    return None


def opencv_highgui_available() -> bool:
    """
    True if cv2.imshow works. Runs the probe in a subprocess so Qt does not
    SIGABRT the parent when X11 auth fails.
    """
    script = (
        "import cv2,numpy as np;"
        "img=np.zeros((2,2),dtype=np.uint8);"
        "cv2.imshow('__cv2_gui_probe__',img);cv2.waitKey(1);cv2.destroyAllWindows();"
        "print('OK')"
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", script],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=25,
        )
        return r.returncode == 0 and "OK" in (r.stdout or "")
    except (subprocess.TimeoutExpired, OSError):
        return False


def print_display_failure_help(context: str) -> None:
    """
    After ensure_local_display + failed imshow probe, or Qt stderr about
    Authorization / xcb / display :0.
    """
    print(f"[opencv] {context}")
    gui_line = _opencv_build_gui_line()
    if gui_line and re.search(r"GUI:\s*NONE\b", gui_line, re.I):
        print("  This OpenCV wheel has no GUI (HighGUI). Install a full build:")
        print("    uv pip uninstall opencv-python-headless opencv-python 2>/dev/null; true")
        print("    uv pip install opencv-contrib-python")
        return

    if gui_line is None:
        print("  (Could not read OpenCV GUI backend from build info; trying X11 hints below.)")

    print("  OpenCV has a GUI backend, but the window system rejected the connection.")
    print("  Typical fix on Raspberry Pi (SSH → show on local HDMI):")
    print("    1) Stay logged into the graphical desktop on the Pi (not only text console).")
    print("    2) On the Pi desktop terminal (local keyboard), allow your SSH user:")
    u = os.environ.get("USER", "ssh_username")
    print(f"         xhost +SI:localuser:{u}")
    print("       (broader:  xhost +local:  — weaker security)")
    print("    3) Or:  export XAUTHORITY=$HOME/.Xauthority   (same user as desktop)")
    print("    4) To keep a custom XAUTHORITY, run with:  OPENCV_KEEP_XAUTH=1")
    print("    5) Optional Qt/XCB packages if errors persist:")
    print("         sudo apt install libxcb-xinerama0 libxcb-cursor0 libxkbcommon-x11-0")
    if gui_line:
        print(f"  (OpenCV reports: {gui_line})")
