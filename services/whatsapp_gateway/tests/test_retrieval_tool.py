import os
import sys
import unittest
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.retrieval_tool import RetrievalTool


class RetrievalToolTests(unittest.TestCase):
    def setUp(self):
        self._previous_env = dict(os.environ)
        os.environ["RETRIEVAL_TOP_K_DEFAULT"] = "6"
        os.environ["RETRIEVAL_TOP_K_MIN"] = "2"
        os.environ["RETRIEVAL_TOP_K_MAX"] = "20"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._previous_env)

    def test_top_k_clamped_to_min(self):
        tool = RetrievalTool()
        self.assertEqual(tool._normalize_top_k(0), 2)

    def test_top_k_clamped_to_max(self):
        tool = RetrievalTool()
        self.assertEqual(tool._normalize_top_k(999), 20)

    def test_top_k_uses_default_on_bad_input(self):
        tool = RetrievalTool()
        self.assertEqual(tool._normalize_top_k("bad"), 6)


if __name__ == "__main__":
    unittest.main()
