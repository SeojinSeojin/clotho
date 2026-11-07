import re

# A line that introduces the answer, tolerating markdown decoration:
#   "Template: x", "**Template:** x", "- Template: x", "### Template: x", "Template : x"
#   group 1: "**" opening a bold label, group 2: "**" closing it before the colon
_TEMPLATE_LINE = re.compile(r"^\s*(?:[-*+>#]+\s+)?(\*\*)?\s*Template\s*(\*\*)?\s*:\s*(.*)$", re.IGNORECASE)
_FENCE = re.compile(r"^\s*```\w*\s*$")


def _strip_decoration(s):
    s = s.strip()
    # surrounding inline code / fenced code on one line: `x`, ```x```
    m = re.fullmatch(r"(`{1,3})(.*)\1", s, flags=re.S)
    if m:
        s = m.group(2).strip()
    # leftover bold markers around the whole answer: **x**. Not when the inner text
    # itself starts/ends with '*': literal asterisk runs are part of real templates
    # (Zookeeper "******* GOODBYE <*> ********", Mac "**** [...] ****").
    m = re.fullmatch(r"\*\*(.*)\*\*", s, flags=re.S)
    if m and not m.group(1).strip().startswith("*") and not m.group(1).strip().endswith("*"):
        s = m.group(1).strip()
    return s


def parse_result_messages_template(result):
    """Text after the LAST 'Template:' line. If that line is empty, the answer may
    be on the next non-empty line (possibly inside a ``` fence).
    Returns (template, None) or (None, 'parse_failure')."""
    lines = result.split("\n")
    found = None
    for i, line in enumerate(lines):
        m = _TEMPLATE_LINE.match(line)
        if m:
            value = m.group(3)
            if m.group(1) and not m.group(2) and value.startswith("**"):
                value = value[2:]  # "**Template:** x" -> the "**" closes the label
            found = (i, value)
    if found is None:
        return None, "parse_failure"

    i, value = found
    value = _strip_decoration(value)
    if not value:
        for nxt in lines[i + 1:]:
            if not nxt.strip() or _FENCE.match(nxt):
                continue
            value = _strip_decoration(nxt)
            break
    if not value:
        return None, "parse_failure"
    return value, None


template2parser = {
    'messages_template': parse_result_messages_template,
    'rules_template': parse_result_messages_template,  # same answer format
}


def normalize(template):
    """Applied identically to prediction and ground truth: strip, collapse
    whitespace, merge adjacent <*> tokens ("<*> <*>", "<*><*>" -> "<*>")."""
    t = re.sub(r"\s+", " ", template.strip())
    t = re.sub(r"<\*>(?:\s*<\*>)+", "<*>", t)
    return t


def score_output(actual_answer, expected_answer, template_name='messages_template'):
    """Full record for analysis: parsed template, parse_failure, strict_match, normalized_match."""
    parsed, error = template2parser[template_name](actual_answer)
    expected = expected_answer[0]
    if error:
        return {"parsed": None, "parse_failure": True, "strict_match": False, "normalized_match": False}
    return {"parsed": parsed, "parse_failure": False,
            "strict_match": parsed == expected,
            "normalized_match": normalize(parsed) == normalize(expected)}


def label_output(actual_answer, expected_answer, template_name, return_parsed_answer=False):
    # answer key order: template. Pass/fail label for Clotho = normalized match.
    s = score_output(actual_answer, expected_answer, template_name)
    if s["parse_failure"]:
        return False, "parse_failure"
    if s["normalized_match"]:
        return True, (s["parsed"] if return_parsed_answer else None)
    return False, (s["parsed"] if return_parsed_answer else "mismatch")
