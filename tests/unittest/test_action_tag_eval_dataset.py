from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "tests" / "fixtures" / "action_tag_eval_cases.jsonl"
ALLOWED = {"happy", "shy", "apologize", "scared", None}


class ActionTagEvalDatasetTest(unittest.TestCase):
    def test_dataset_is_valid_and_balanced(self) -> None:
        rows = []
        with DATASET.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self.assertIn("id", row, lineno)
                self.assertIn("user_text", row, row.get("id", lineno))
                self.assertIn("expected_tag", row, row.get("id", lineno))
                self.assertIn(row["expected_tag"], ALLOWED, row["id"])
                self.assertIsInstance(row.get("visual_context", ""), str, row["id"])
                rows.append(row)

        ids = [row["id"] for row in rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(rows), 40)

        counts = Counter(row["expected_tag"] for row in rows)
        for tag in ("happy", "shy", "apologize", "scared"):
            self.assertGreaterEqual(counts[tag], 6, tag)
        self.assertGreaterEqual(counts[None], 10)


if __name__ == "__main__":
    unittest.main()
