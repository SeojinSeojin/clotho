"""AdaParser @232951a scoring-only extraction; MIT, see LICENSE.
Source: benchmark/logparser/AdaParser/utils.py. Function verbatim; no API client imported.
"""
import json
import re
import string
from pathlib import Path
from .adaparser_post import correct_single_template
regs_common = [re.compile(p) for p in json.loads((Path(__file__).parent / "adaparser_common.json").read_text())["COMMON"]["regex"]]
compiled_pattern1 = re.compile(r"\{\w+}")
def post_process_template(template):
    template = compiled_pattern1.sub("<*>", template)
    for reg in regs_common:
        template = reg.sub("<*>", template)
    template = correct_single_template(template)
    static_part = template.replace("<*>", "")
    punc = string.punctuation
    for s in static_part:
        if s != ' ' and s not in punc:
            template = template.replace("<*>", "{variables}")
            return template, True
    # print("Get a too general template. Error.")
    template = template.replace("<*>", "{variables}")
    return template, False
