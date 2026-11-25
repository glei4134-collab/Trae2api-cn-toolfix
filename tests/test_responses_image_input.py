"""Regression coverage for Responses API image input preservation."""

from __future__ import annotations

import unittest

from src import images, responses_api


class ResponsesImageInputTests(unittest.TestCase):
    def test_input_image_becomes_image_url_part(self):
        messages: list[dict] = []
        responses_api._append_message_item(
            messages,
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what color is this?"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,AAAA",
                    },
                ],
            },
        )
        self.assertEqual(len(messages), 1)
        self.assertIsInstance(messages[0]["content"], list)
        self.assertTrue(images.messages_have_images(messages))
        image_parts = [
            part
            for part in messages[0]["content"]
            if isinstance(part, dict) and part.get("type") == "image_url"
        ]
        self.assertEqual(len(image_parts), 1)
        self.assertEqual(
            image_parts[0]["image_url"]["url"], "data:image/png;base64,AAAA"
        )

    def test_text_only_content_stays_a_string(self):
        messages: list[dict] = []
        responses_api._append_message_item(
            messages,
            {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
        )
        self.assertEqual(messages[0]["content"], "hello")


if __name__ == "__main__":
    unittest.main()
