#!/usr/bin/env python3
"""Measure streaming first-token latency for the OpenAI-compatible LLM endpoint."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.7-plus"
DEFAULT_QUERIES = [
    "向左转一点。",
    "小黑，你怎么这么可爱啊。",
    "解释一下 VAD 是怎么工作的。",
    "你刚才为什么撞到桌腿了？",
    "砰！刚才外面好大一声。",
]
THINKING_MODES = ("disabled", "enabled", "omit")
CLIENTS = ("urllib", "openai-sdk")


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def read_prompt(path: Path | None, git_ref: str | None) -> str:
    if git_ref:
        return subprocess.check_output(
            ["git", "show", f"{git_ref}:config/system_prompt.txt"],
            cwd=REPO_ROOT,
        ).decode("utf-8").strip()
    prompt_path = path or (REPO_ROOT / "config" / "system_prompt.txt")
    return prompt_path.read_text(encoding="utf-8").strip()


def stream_once(
    *,
    api_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    query: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    no_proxy: bool,
    thinking_mode: str,
    client_kind: str,
) -> dict[str, float | int | str | None]:
    if client_kind == "openai-sdk":
        return stream_once_openai_sdk(
            api_url=api_url,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            query=query,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            thinking_mode=thinking_mode,
        )
    endpoint = api_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if thinking_mode != "omit":
        # OpenAI SDK's extra_body merges this into the JSON root. When using
        # urllib directly we must send the DashScope extension at the root too.
        payload["enable_thinking"] = thinking_mode == "enabled"
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    t0 = time.perf_counter()
    first = None
    pieces: list[str] = []
    opener = (
        urllib.request.build_opener(urllib.request.ProxyHandler({}))
        if no_proxy
        else urllib.request.build_opener()
    )
    with opener.open(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = (choices[0].get("delta") or {}).get("content")
            if delta:
                if first is None:
                    first = time.perf_counter()
                pieces.append(delta)
    t1 = time.perf_counter()
    return {
        "query": query,
        "first_ms": None if first is None else (first - t0) * 1000.0,
        "total_ms": (t1 - t0) * 1000.0,
        "chars": len("".join(pieces)),
    }


def stream_once_openai_sdk(
    *,
    api_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    query: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    thinking_mode: str,
) -> dict[str, float | int | str | None]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required for --client openai-sdk") from exc

    client = OpenAI(api_key=api_key, base_url=api_url, timeout=timeout)
    kwargs = {}
    if thinking_mode != "omit":
        kwargs["extra_body"] = {"enable_thinking": thinking_mode == "enabled"}
    t0 = time.perf_counter()
    first = None
    pieces: list[str] = []
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
        **kwargs,
    )
    for chunk in completion:
        if chunk and chunk.choices and chunk.choices[0] and chunk.choices[0].delta.content:
            delta = chunk.choices[0].delta.content
            if first is None:
                first = time.perf_counter()
            pieces.append(delta)
    t1 = time.perf_counter()
    return {
        "query": query,
        "first_ms": None if first is None else (first - t0) * 1000.0,
        "total_ms": (t1 - t0) * 1000.0,
        "chars": len("".join(pieces)),
    }


def summarize(rows: list[dict[str, float | int | str | None]]) -> dict[str, float | int]:
    first = [float(r["first_ms"]) for r in rows if r["first_ms"] is not None]
    total = [float(r["total_ms"]) for r in rows]
    return {
        "n": len(rows),
        "first_mean_ms": round(statistics.mean(first)),
        "first_p50_ms": round(statistics.median(first)),
        "first_min_ms": round(min(first)),
        "first_max_ms": round(max(first)),
        "total_mean_ms": round(statistics.mean(total)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", type=Path, default=None)
    parser.add_argument("--prompt-from-git", default=None, help="Read config/system_prompt.txt from a git ref, e.g. HEAD")
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--no-proxy", action="store_true", help="Bypass env/system proxies for this process")
    parser.add_argument(
        "--thinking-mode",
        choices=THINKING_MODES,
        default="disabled",
        help="DashScope thinking flag mode: disabled sends enable_thinking=false, enabled sends true, omit sends no flag",
    )
    parser.add_argument("--client", choices=CLIENTS, default="urllib")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("DASHSCOPE_API_KEY is not set", file=sys.stderr)
        return 2

    prompt = read_prompt(args.prompt, args.prompt_from_git)
    queries = args.query or DEFAULT_QUERIES
    rows = [
        stream_once(
            api_url=args.api_url,
            api_key=api_key,
            model=args.model,
            system_prompt=prompt,
            query=query,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            no_proxy=args.no_proxy,
            thinking_mode=args.thinking_mode,
            client_kind=args.client,
        )
        for query in queries
    ]
    summary = summarize(rows)
    if args.json:
        print(json.dumps({"model": args.model, "summary": summary, "rows": rows}, ensure_ascii=False, indent=2))
        return 0
    for row in rows:
        print(
            f"{args.model} first_ms={row['first_ms']:.0f} total_ms={row['total_ms']:.0f} "
            f"chars={row['chars']} query={row['query']}"
        )
    print("SUMMARY " + " ".join(f"{k}={v}" for k, v in summary.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
