"""The per-tier quality wording rule, shared by enhance_switch_test (README + workflow notes) and loader_dispatch_test
(every rendered quality tooltip). The user accepted that few-step models may give another variation of the seed in the
fast tiers on the condition that the text SAYS so; balance keeps the subject and scene."""
import re

FAST_OPTION = re.compile(r"(?<![\w-])`?(?:super_fast|fast)`?(?![\w-])")   # the option NAMES, not "faster" / "fast-moving"


def tier_problems(text):
    """[] when the text is right: a text that offers fast / super_fast carries the variation clause, and no sentence with
    the subject/scene claim (balance's) names fast / super_fast."""
    probs = []
    if FAST_OPTION.search(text) and "different variation of the same seed" not in text:
        probs.append("offers fast / super_fast without the variation clause")
    probs += [f"subject/scene claim on fast: {s[:70]!r}" for s in re.split(r"(?<=\.)\s+", text)
              if "subject and scene stay the same" in s and FAST_OPTION.search(s)]
    return probs
