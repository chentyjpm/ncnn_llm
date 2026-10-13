"""Discoverable synthetic regression for streamed official HF VQA reference."""
import unittest
from minicpm_streamed_reference import self_test


class StreamedReferenceTests(unittest.TestCase):
    def test_embedding_head_and_official_cached_vqa(self):
        # Constructs a tiny temporary BF16 checkpoint, never reads real weights.
        self_test()


if __name__ == '__main__':
    unittest.main()
