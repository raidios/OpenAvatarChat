from __future__ import annotations

import unittest
from pathlib import Path

from tools.eval_action_tags import EvalCase, load_cases, score_response, summarize


REPO_ROOT = Path(__file__).resolve().parents[2]


class EvalActionTagsTest(unittest.TestCase):
    def test_load_cases(self) -> None:
        cases = load_cases(REPO_ROOT / "tests" / "fixtures" / "action_tag_eval_cases.jsonl")
        self.assertGreaterEqual(len(cases), 40)
        self.assertEqual(cases[0].id, "happy_001")

    def test_score_matching_tag(self) -> None:
        case = EvalCase(id="c1", user_text="hi", expected_tag="happy")
        result = score_response(case, "qwen3.7-plus", "[happy]我也很开心。")
        self.assertTrue(result.ok)
        self.assertEqual(result.predicted_tag, "happy")
        self.assertEqual(result.clean_text, "我也很开心。")

    def test_score_no_tag(self) -> None:
        case = EvalCase(id="c2", user_text="hi", expected_tag=None)
        result = score_response(case, "qwen3.7-plus", "现在电池电压需要从日志里读。")
        self.assertTrue(result.ok)
        self.assertIsNone(result.predicted_tag)

    def test_multi_tag_fails(self) -> None:
        case = EvalCase(id="c3", user_text="hi", expected_tag="apologize")
        result = score_response(case, "qwen3.7-plus", "[apologize]对不起。[happy]我会改。")
        self.assertFalse(result.ok)
        self.assertTrue(result.multi_tag)
        self.assertEqual(result.raw_tags, ["apologize", "happy"])

    def test_observed_scared_typo_is_tolerated(self) -> None:
        case = EvalCase(id="c4", user_text="boom", expected_tag="scared")
        result = score_response(case, "qwen3.7-plus", "[scaed]啊！吓我一跳！")
        self.assertTrue(result.ok)
        self.assertEqual(result.predicted_tag, "scared")
        self.assertEqual(result.clean_text, "啊！吓我一跳！")

    def test_summary_counts(self) -> None:
        cases = [
            EvalCase(id="ok", user_text="", expected_tag="happy"),
            EvalCase(id="fp", user_text="", expected_tag=None),
            EvalCase(id="miss", user_text="", expected_tag="scared"),
            EvalCase(id="wrong", user_text="", expected_tag="shy"),
        ]
        results = [
            score_response(cases[0], "m", "[happy]好呀。"),
            score_response(cases[1], "m", "[happy]好呀。"),
            score_response(cases[2], "m", "没事。"),
            score_response(cases[3], "m", "[scared]啊。"),
        ]
        summary = summarize(results)
        self.assertEqual(summary["ok"], 1)
        self.assertEqual(summary["false_positive"], 1)
        self.assertEqual(summary["missed"], 1)
        self.assertEqual(summary["wrong_tag"], 1)


if __name__ == "__main__":
    unittest.main()
