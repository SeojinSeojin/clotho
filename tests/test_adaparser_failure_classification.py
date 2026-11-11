"""Only a demonstrated permanent input-context overflow permits protocol skipping."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from adaparser_failures import context_error


class ContextFailureTests(unittest.TestCase):
    def test_actual_overflow(self):
        msg='Error code: 400 maximum context length is 8192 tokens. Prompt contains at least 8193 input tokens'
        self.assertEqual(context_error(msg),{'configured_context':8192,'minimum_prompt_tokens':8193})

    def test_not_an_api_error_or_not_an_overflow(self):
        self.assertIsNone(context_error('maximum context length is 8192 tokens; at least 8193 input tokens'))
        self.assertIsNone(context_error('Error code: 400 maximum context length is 8192 tokens; at least 3 input tokens'))
        self.assertIsNone(context_error('Error code: 500 transient internal server error'))


if __name__=='__main__':unittest.main()
