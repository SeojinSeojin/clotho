"""Tests for the log_parsing task.  Run:  python -m unittest discover tests -v"""
import json
import unittest
from pathlib import Path

from clotho.dataset.log_parsing.output_parsers import (
    label_output, normalize, parse_result_messages_template as parse, score_output)
from clotho.dataset.log_parsing.templates import template_map

DATA_DIR = Path(__file__).resolve().parents[1] / "clotho" / "dataset" / "log_parsing"
T = "messages_template"


class TestNormalize(unittest.TestCase):
    def test_strip_and_collapse_whitespace(self):
        self.assertEqual(normalize("  Verification   succeeded\tfor <*> \n"), "Verification succeeded for <*>")

    def test_merge_adjacent_wildcards(self):
        self.assertEqual(normalize("<*> <*>"), "<*>")
        self.assertEqual(normalize("<*><*>"), "<*>")
        self.assertEqual(normalize("<*>  <*>\t<*><*>"), "<*>")
        self.assertEqual(normalize("allocateBlock: <*> <*>"), "allocateBlock: <*>")
        self.assertEqual(normalize("a <*> <*> b <*><*> c"), "a <*> b <*> c")

    def test_keeps_separated_wildcards_and_constants(self):
        self.assertEqual(normalize("acls to: <*>,<*>"), "acls to: <*>,<*>")
        self.assertEqual(normalize("<*>:<*>"), "<*>:<*>")
        self.assertEqual(normalize("BLOCK* NameSystem"), "BLOCK* NameSystem")  # lone '*' untouched

    def test_idempotent(self):
        for s in ["<*> <*> x", "  a  b ", "<*><*><*>", ""]:
            self.assertEqual(normalize(normalize(s)), normalize(s))


class TestParser(unittest.TestCase):
    def test_clean(self):
        out = "Reasoning: the block id and IPs vary.\nTemplate: Receiving block <*> src: <*> dest: <*>"
        self.assertEqual(parse(out), ("Receiving block <*> src: <*> dest: <*>", None))

    def test_missing_template_line(self):
        self.assertEqual(parse("Reasoning: the id varies.\nReceiving block <*>"), (None, "parse_failure"))
        self.assertEqual(parse(""), (None, "parse_failure"))

    def test_empty_template_value(self):
        self.assertEqual(parse("Reasoning: x\nTemplate:"), (None, "parse_failure"))
        self.assertEqual(parse("Reasoning: x\nTemplate:   \n\n"), (None, "parse_failure"))

    def test_extra_text_around(self):
        out = ("Sure! Here is the parsed log.\n\nReasoning: numbers vary.\nTemplate: Starting task <*>\n\n"
               "Let me know if you need anything else.")
        self.assertEqual(parse(out)[0], "Starting task <*>")

    def test_markdown_variants(self):
        cases = {
            "**Template:** Starting task <*>": "Starting task <*>",
            "**Template**: Starting task <*>": "Starting task <*>",
            "- **Template:** `Starting task <*>`": "Starting task <*>",
            "### Template: Starting task <*>": "Starting task <*>",
            "Template: ```Starting task <*>```": "Starting task <*>",
            "Template: **Starting task <*>**": "Starting task <*>",
            "template: Starting task <*>": "Starting task <*>",
            "Template : Starting task <*>": "Starting task <*>",
            "* Template: Starting task <*>": "Starting task <*>",
            "**Template:** **Starting task <*>**": "Starting task <*>",
            "> Template: Starting task <*>": "Starting task <*>",
        }
        for line, want in cases.items():
            with self.subTest(line=line):
                self.assertEqual(parse("Reasoning: x\n" + line), (want, None))

    def test_answer_on_next_line_in_code_fence(self):
        out = "Reasoning: x\nTemplate:\n```\nStarting task <*>\n```"
        self.assertEqual(parse(out), ("Starting task <*>", None))

    def test_last_template_line_wins(self):
        out = "Template: draft <*> <*>\nWait, let me fix that.\nTemplate: Starting task <*>"
        self.assertEqual(parse(out)[0], "Starting task <*>")

    def test_template_mentioned_mid_line_is_not_an_answer(self):
        self.assertEqual(parse("Reasoning: the Template: field holds ids."), (None, "parse_failure"))

    def test_literal_asterisk_runs_preserved(self):
        for t in ["******* GOODBYE <*> ********", "**** [X][start] -- Completed -- <*> ****"]:
            self.assertEqual(parse(f"Reasoning: x\nTemplate: {t}"), (t, None))
        self.assertEqual(parse("Template: **a <*> b**"), ("a <*> b", None))

    def test_inner_backticks_preserved(self):
        self.assertEqual(parse("Template: run `ls` on <*>")[0], "run `ls` on <*>")


class TestLabelOutput(unittest.TestCase):
    GT = ["BLOCK* NameSystem.allocateBlock: <*> <*>"]

    def test_normalized_pass_strict_fail(self):
        out = "Reasoning: path and block id.\nTemplate: BLOCK* NameSystem.allocateBlock: <*>"
        self.assertEqual(label_output(out, self.GT, T), (True, None))
        s = score_output(out, self.GT, T)
        self.assertEqual((s["strict_match"], s["normalized_match"], s["parse_failure"]), (False, True, False))

    def test_strict_pass(self):
        s = score_output("Template: BLOCK* NameSystem.allocateBlock: <*> <*>", self.GT, T)
        self.assertTrue(s["strict_match"] and s["normalized_match"])

    def test_mismatch(self):
        out = "Reasoning: x\nTemplate: BLOCK* NameSystem.allocateBlock: /mnt/<*> <*>"
        self.assertEqual(label_output(out, self.GT, T), (False, "mismatch"))
        self.assertEqual(label_output(out, self.GT, T, return_parsed_answer=True),
                         (False, "BLOCK* NameSystem.allocateBlock: /mnt/<*> <*>"))

    def test_parse_failure_is_fail(self):
        self.assertEqual(label_output("no answer", self.GT, T), (False, "parse_failure"))
        self.assertTrue(score_output("no answer", self.GT, T)["parse_failure"])


class TestDatasets(unittest.TestCase):
    """Every ground-truth template must pass when echoed back in the expected format."""

    def test_ground_truth_round_trip(self):
        files = sorted(DATA_DIR.glob("loghub_*.jsonl"))
        self.assertTrue(files)
        for f in files:
            recs = [json.loads(l) for l in f.read_text().splitlines()]
            self.assertEqual(len({r["id"] for r in recs}), len(recs), f"duplicate ids in {f.name}")
            self.assertEqual(len({r["content"] for r in recs}), len(recs), f"duplicate content in {f.name}")
            for r in recs:
                self.assertEqual(set(r), {"id", "system", "content", "template", "template_id"})
                out = f"Reasoning: x\nTemplate: {r['template']}"
                self.assertEqual(label_output(out, [r["template"]], T), (True, None), (f.name, r["id"]))

    def test_prompt_contains_content(self):
        msgs = template_map[T]("Verification succeeded for blk_1")
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        self.assertIn("Log message: Verification succeeded for blk_1\n", msgs[1]["content"])
        self.assertTrue(msgs[1]["content"].endswith("Template: <the log template>"))

    def test_every_template_renders_and_parses(self):
        from clotho.dataset.log_parsing.output_parsers import template2parser
        self.assertEqual(set(template_map), set(template2parser))
        for name, fn in template_map.items():
            user = fn("Verification succeeded for blk_1")[1]["content"]
            self.assertIn("Log message: Verification succeeded for blk_1\n", user, name)
            self.assertTrue(user.endswith("Template: <the log template>"), name)
            self.assertEqual(label_output("Reasoning: x\nTemplate: a <*>", ["a <*>"], name), (True, None))

    def test_registered_datasets_exist(self):
        from clotho.exp_config import supported_datasets
        for ds in supported_datasets["log_parsing"]:
            self.assertTrue((DATA_DIR / f"{ds}.jsonl").exists(), ds)


if __name__ == "__main__":
    unittest.main()
