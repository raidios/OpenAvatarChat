import unittest
from dataclasses import dataclass

from src.chat_engine.core.log_redaction import redact_sensitive_config_for_log


@dataclass
class NestedConfig:
    model_name: str = "qwen"
    api_key: str = "sk-nested-secret"
    options: dict = None

    def __post_init__(self):
        if self.options is None:
            self.options = {"refresh_token": "tok-refresh-secret", "safe": "visible"}


class HandlerManagerRedactionTest(unittest.TestCase):
    def test_redacts_sensitive_values_from_nested_config(self):
        config = {
            "api_key": "sk-top-secret",
            "password": "plain-password",
            "nested": NestedConfig(),
            "items": [
                {"access_token": "tok-list-secret"},
                {"normal": "keep-me"},
            ],
            "model_name": "qwen3.5-flash",
        }

        rendered = repr(redact_sensitive_config_for_log(config))

        self.assertIn("qwen3.5-flash", rendered)
        self.assertIn("keep-me", rendered)
        self.assertIn("<redacted>", rendered)
        self.assertNotIn("sk-top-secret", rendered)
        self.assertNotIn("plain-password", rendered)
        self.assertNotIn("sk-nested-secret", rendered)
        self.assertNotIn("tok-refresh-secret", rendered)
        self.assertNotIn("tok-list-secret", rendered)


if __name__ == "__main__":
    unittest.main()
