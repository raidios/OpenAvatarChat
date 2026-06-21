#!/usr/bin/env python3
"""Evaluate action-tag selection for the robot dialog prompt.

The live stack asks the LLM to optionally emit one of:
``[happy]``, ``[shy]``, ``[apologize]``, ``[scared]``.
This tool runs fixed cases through an OpenAI-compatible chat endpoint and
scores the parsed tag against the expected label.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from handlers.common.action_tags import EMOTION_TAGS, extract_action_tags  # noqa: E402


DEFAULT_CASES = REPO_ROOT / "tests" / "fixtures" / "action_tag_eval_cases.jsonl"
DEFAULT_PROMPT = REPO_ROOT / "config" / "system_prompt.txt"
DEFAULT_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.7-plus"
THINKING_MODES = ("disabled", "enabled", "omit")


@dataclass(frozen=True)
class EvalCase:
    id: str
    user_text: str
    expected_tag: str | None
    category: str = ""
    visual_context: str = ""
    why: str = ""


@dataclass(frozen=True)
class EvalResult:
    id: str
    model: str
    expected_tag: str | None
    predicted_tag: str | None
    ok: bool
    raw_tags: list[str]
    multi_tag: bool
    invented_bracket: bool
    response_text: str
    clean_text: str
    error: str | None = None


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            try:
                cases.append(
                    EvalCase(
                        id=str(raw["id"]),
                        user_text=str(raw["user_text"]),
                        expected_tag=raw.get("expected_tag"),
                        category=str(raw.get("category", "")),
                        visual_context=str(raw.get("visual_context", "")),
                        why=str(raw.get("why", "")),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{lineno}: missing field {exc}") from exc
    return cases


def read_prompt(path: Path, git_ref: str | None) -> str:
    if git_ref:
        return subprocess.check_output(
            ["git", "show", f"{git_ref}:config/system_prompt.txt"],
            cwd=REPO_ROOT,
        ).decode("utf-8").strip()
    return path.read_text(encoding="utf-8")


def selected_cases(
    cases: Iterable[EvalCase],
    *,
    case_ids: set[str] | None,
    categories: set[str] | None,
    limit: int | None,
) -> list[EvalCase]:
    out: list[EvalCase] = []
    for case in cases:
        if case_ids and case.id not in case_ids:
            continue
        if categories and case.category not in categories:
            continue
        out.append(case)
        if limit is not None and len(out) >= limit:
            break
    return out


def build_user_message(case: EvalCase) -> str:
    if not case.visual_context:
        return case.user_text
    return f"{case.user_text}\n\n当前视觉上下文：{case.visual_context}"


def openai_compatible_chat(
    *,
    api_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    case: EvalCase,
    temperature: float,
    max_tokens: int,
    timeout: float,
    thinking_mode: str,
) -> str:
    endpoint = api_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_message(case)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if thinking_mode != "omit":
        payload["enable_thinking"] = thinking_mode == "enabled"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:500]}") from exc
    content = raw["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise RuntimeError(f"unexpected content type: {type(content)!r}")
    return content


def predict_tag(response_text: str) -> tuple[str | None, list[str], str, bool]:
    clean, anchored = extract_action_tags(response_text)
    tags = [tag.name for tag in anchored]
    return (tags[0] if tags else None), tags, clean, len(tags) > 1


def has_invented_bracket(response_text: str, raw_tags: list[str]) -> bool:
    if "[" not in response_text and "]" not in response_text:
        return False
    lowered = response_text.lower()
    allowed = {f"[{tag}]" for tag in EMOTION_TAGS}
    # After removing whitespace-tolerant known tags via the parser, any remaining
    # square bracket is suspicious because TTS would read it.
    for allowed_tag in allowed:
        lowered = lowered.replace(allowed_tag, "")
    return "[" in lowered or "]" in lowered


def score_response(case: EvalCase, model: str, response_text: str, error: str | None = None) -> EvalResult:
    predicted, raw_tags, clean, multi_tag = predict_tag(response_text)
    return EvalResult(
        id=case.id,
        model=model,
        expected_tag=case.expected_tag,
        predicted_tag=predicted,
        ok=(error is None and predicted == case.expected_tag and not multi_tag),
        raw_tags=raw_tags,
        multi_tag=multi_tag,
        invented_bracket=has_invented_bracket(response_text, raw_tags),
        response_text=response_text,
        clean_text=clean,
        error=error,
    )


def summarize(results: list[EvalResult]) -> dict[str, Any]:
    total = len(results)
    ok = sum(1 for r in results if r.ok)
    none_cases = [r for r in results if r.expected_tag is None]
    tagged_cases = [r for r in results if r.expected_tag is not None]
    false_positive = sum(1 for r in none_cases if r.predicted_tag is not None)
    missed = sum(1 for r in tagged_cases if r.predicted_tag is None)
    wrong_tag = sum(1 for r in tagged_cases if r.predicted_tag not in (None, r.expected_tag))
    multi = sum(1 for r in results if r.multi_tag)
    errors = sum(1 for r in results if r.error)
    return {
        "total": total,
        "ok": ok,
        "accuracy": (ok / total) if total else 0.0,
        "false_positive": false_positive,
        "missed": missed,
        "wrong_tag": wrong_tag,
        "multi_tag": multi,
        "errors": errors,
    }


def parse_csv(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        out.extend(v.strip() for v in value.split(",") if v.strip())
    return out


def write_jsonl(path: Path, rows: Iterable[EvalResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-from-git", default=None, help="Read config/system_prompt.txt from a git ref, e.g. HEAD")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--model", action="append", default=[], help="Model id; repeat or comma-separate")
    parser.add_argument("--case-id", action="append", default=[], help="Run only selected case id")
    parser.add_argument("--category", action="append", default=[], help="Run only selected category")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument(
        "--thinking-mode",
        choices=THINKING_MODES,
        default="disabled",
        help="DashScope thinking flag mode: disabled sends enable_thinking=false, enabled sends true, omit sends no flag",
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds between API calls")
    parser.add_argument("--out-jsonl", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate data and print selected cases without API calls")
    args = parser.parse_args(argv)

    load_dotenv(REPO_ROOT / ".env")
    models = parse_csv(args.model) or [DEFAULT_MODEL]
    cases = selected_cases(
        load_cases(args.cases),
        case_ids=set(parse_csv(args.case_id)) or None,
        categories=set(parse_csv(args.category)) or None,
        limit=args.limit,
    )
    system_prompt = read_prompt(args.prompt, args.prompt_from_git)

    print(f"cases={len(cases)} models={','.join(models)} prompt={args.prompt}")
    if args.dry_run:
        for case in cases:
            print(f"{case.id:15s} expected={case.expected_tag or '-':10s} {build_user_message(case)}")
        return 0

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("DASHSCOPE_API_KEY is not set; put it in .env or the environment", file=sys.stderr)
        return 2

    all_results: list[EvalResult] = []
    for model in models:
        results: list[EvalResult] = []
        for i, case in enumerate(cases, start=1):
            error = None
            response = ""
            try:
                response = openai_compatible_chat(
                    api_url=args.api_url,
                    api_key=api_key,
                    model=model,
                    system_prompt=system_prompt,
                    case=case,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    thinking_mode=args.thinking_mode,
                )
            except Exception as exc:  # pragma: no cover - network path
                error = str(exc)
            result = score_response(case, model, response, error)
            results.append(result)
            mark = "OK" if result.ok else "!!"
            predicted = result.predicted_tag or "-"
            expected = result.expected_tag or "-"
            print(f"{model:14s} {i:02d}/{len(cases):02d} {mark} {case.id:15s} expected={expected:10s} got={predicted:10s}")
            if error:
                print(f"  error: {error}")
            elif not result.ok:
                print(f"  response: {response}")
            if args.sleep > 0 and i < len(cases):
                time.sleep(args.sleep)
        summary = summarize(results)
        print(
            f"SUMMARY {model}: ok={summary['ok']}/{summary['total']} "
            f"acc={summary['accuracy']:.1%} fp={summary['false_positive']} "
            f"missed={summary['missed']} wrong={summary['wrong_tag']} "
            f"multi={summary['multi_tag']} errors={summary['errors']}"
        )
        all_results.extend(results)

    if args.out_jsonl:
        write_jsonl(args.out_jsonl, all_results)
        print(f"wrote {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
