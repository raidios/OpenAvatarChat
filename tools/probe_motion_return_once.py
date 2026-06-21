from __future__ import annotations

import argparse
import json
import time
import urllib.request


def _json_get(base_url: str, path: str) -> dict:
    with urllib.request.urlopen(base_url + path, timeout=2.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post(base_url: str, path: str) -> str:
    req = urllib.request.Request(base_url + path, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=2.0) as resp:
        return resp.read().decode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip_id")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--samples", type=int, default=70)
    parser.add_argument("--interval", type=float, default=0.25)
    args = parser.parse_args()

    status = _json_get(args.base_url, "/api/status")
    print("pre", json.dumps(status.get("player", {}), ensure_ascii=False))
    print("play", _post(args.base_url, f"/api/clips/{args.clip_id}/play"))

    seen_active = False
    for idx in range(args.samples):
        status = _json_get(args.base_url, "/api/status")
        player = status.get("player", {})
        mcu = status.get("telemetry", {}).get("mcu", {})
        ctrl = status.get("telemetry", {}).get("ctrl", {})
        pose = player.get("pose") or {}
        print(
            "{:02d} phase={} progress={}/{} pose=({},{},{}) vel={} ctrl={} bat={}".format(
                idx,
                player.get("phase"),
                player.get("current_progress_ms"),
                player.get("current_total_ms"),
                pose.get("x_m"),
                pose.get("y_m"),
                pose.get("yaw_deg"),
                mcu.get("vel"),
                ctrl.get("source_name"),
                mcu.get("bat_voltage"),
            )
        )
        if player.get("phase") != "idle":
            seen_active = True
        elif seen_active and idx > 5:
            break
        time.sleep(args.interval)

    status = _json_get(args.base_url, "/api/status")
    print("post", json.dumps(status.get("player", {}), ensure_ascii=False))


if __name__ == "__main__":
    main()
