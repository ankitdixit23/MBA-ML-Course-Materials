from __future__ import annotations

from curses import raw
import os
import re
import json
import random
from collections import Counter, defaultdict
import re
import pandas as pd
import time
from openai import OpenAI
from openai import RateLimitError, APIError, APITimeoutError, APIConnectionError


# -----------------------------
# User config
# -----------------------------
CLEAN_CSV_PATH = "C:\\Users\\chansen1\\Chicago Booth Dropbox\\Chris Hansen\\Public\\Resume_cleaned.csv" # update if needed

RANDOM_SEED = 7
EXAMPLES_PER_DEPT = 6 # For team labels
# EXAMPLES_PER_DEPT = 3 # For raw departments (more classes, so fewer examples each to fit in context)

# How much resume text to send (approx; tokenization differs by model)
MAX_WORDS = 150  

# If you prefer HTML, set USE_HTML=True. Otherwise uses plain text, with html fallback if plain missing.
USE_HTML = False

# Model choice (swap as you like)
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

# Hypothetical firm department structure
DEPT_TO_TEAM = {
    "ACCOUNTANT": "CORPORATE-FINANCE",
    "FINANCE": "CORPORATE-FINANCE",
    "BANKING": "CORPORATE-FINANCE",

    "CONSULTANT": "STRATEGY-ADVISORY",
    "BUSINESS-DEVELOPMENT": "STRATEGY-ADVISORY",

    "HR": "PEOPLE-OPS",
    "BPO": "PEOPLE-OPS",

    "ENGINEERING": "TECH-ENGINEERING",
    "INFORMATION-TECHNOLOGY": "TECH-ENGINEERING",

    "SALES": "COMMERCIAL-SALES",
    "PUBLIC-RELATIONS": "COMMERCIAL-SALES",

    "DIGITAL-MEDIA": "BRAND-CREATIVE",
    "DESIGNER": "BRAND-CREATIVE",
    "ARTS": "BRAND-CREATIVE",

    "HEALTHCARE": "HEALTH-SERVICES",
    "FITNESS": "HEALTH-SERVICES",

    "AUTOMOBILE": "INDUSTRIAL-OPS",
    "AVIATION": "INDUSTRIAL-OPS",
    "CONSTRUCTION": "INDUSTRIAL-OPS",
    "AGRICULTURE": "INDUSTRIAL-OPS",

    "CHEF": "CONSUMER-HOSPITALITY",
    "APPAREL": "CONSUMER-HOSPITALITY",

    "TEACHER": "EDUCATION-PROGRAMS",

    "ADVOCATE": "LEGAL-ADVOCACY",
}

useteam = True   # Switch to false to use raw departments instead of "team labels"

# -----------------------------
# Heuristics
# -----------------------------

# Match: leading whitespace + 2+ all-caps "tokens" (allow HR/CPA/VP, allow punctuation)
# then whitespace + a normal word (e.g., Summary, Objective, Profile, etc.)
LEADING_CAPS_TITLE_RE = re.compile(
    r"""^\s*
    (?P<title>
        [A-Z][A-Z0-9&/\-\.]{1,}          # token 1 (e.g., HR, CPA, VP, DIRECTOR, FINANCE)
        (?:\s+[\|\-–—•]*\s*[A-Z][A-Z0-9&/\-\.]{1,}){0,15}   # 1+ more caps tokens
    )
    (?P<gap>\s+)
    (?P<nextword>[A-Za-z])              # next char exists and starts a normal word
    """,
    re.VERBOSE,
)

def strip_leading_caps_title(text: str, max_strip_chars: int = 120) -> tuple[str, bool]:
    """
    Strip a leading ALL-CAPS job-title-like prefix even if line breaks are lost.

    Example:
      ' HR DIRECTOR       Summary ...'  -> strip 'HR DIRECTOR'
      'SENIOR ACCOUNTANT | CPA  Summary ...' -> strip 'SENIOR ACCOUNTANT | CPA'

    Only strips if the matched title isn't absurdly long (guardrail).
    """
    if not text:
        return text, False

    s = text.lstrip()
    m = LEADING_CAPS_TITLE_RE.match(s)
    if not m:
        return s, False

    title = m.group("title").strip()
    if len(title) > max_strip_chars:
        return s, False  # avoid nuking huge caps blocks

    # Cut everything up through the matched title + gap (but keep the nextword)
    cut = m.start("nextword")
    return s[cut:].lstrip(), True


def truncate_words(text: str, max_words: int) -> str:
    """
    Cheap truncation: keep first max_words whitespace-separated tokens.
    """
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip() + " ..."


def build_resume_snippet(row: pd.Series, max_words: int) -> tuple[str, dict]:
    """
      snippet: str
      meta: dict (e.g. {"header_removed": True, "used_html": False})
    """
    used_html = False
    raw = ""

    if USE_HTML:
        raw = (row.get("resume_text_html") or "").strip()
        used_html = True
        if not raw:
            raw = (row.get("resume_text") or "").strip()
            used_html = False
    else:
        raw = (row.get("resume_text") or "").strip()
        if not raw:
            raw = (row.get("resume_text_html") or "").strip()
            used_html = True

 #   cleaned, removed = strip_leading_caps_title(raw, max_strip_chars=120)  # Uncomment to strip out headers
    cleaned, removed = strip_leading_caps_title(raw, max_strip_chars=0)
    snippet = truncate_words(cleaned, max_words)

    meta = {"header_removed": removed, "used_html": used_html}
    return snippet, meta


# -----------------------------
# Prompting / API calls
# -----------------------------
def build_label_set(depts: list[str]) -> str:
    # shown in prompt; keep stable
    return ", ".join(sorted(set(depts)))


def make_zero_shot_prompt(resume_snippet: str, label_list: list[str]) -> tuple[str, str]:
    """
    Returns (instructions, input_text) for Responses API.
    Output format is a single line:
      DEPARTMENT=<LABEL>;CONF=<NUMBER>
    """
    labels = build_label_set(label_list)
    instructions = (
        "You are routing resumes to the correct department.\n"
        "Return EXACTLY ONE LINE in this format:\n"
        "DEPARTMENT=<LABEL>;CONF=<NUMBER>\n"
        "Where <LABEL> is exactly one of the allowed labels, and <NUMBER> is a decimal in [0,1].\n"
        "The line must begin with DEPARTMENT=.\n"
        "No extra text. No markdown. No code fences.\n"
        f"Allowed labels: [{labels}]"
    )

    input_text = (
        "Resume:\n"
        "-----\n"
        f"{resume_snippet}\n"
        "-----\n"
        "Classify this resume into the correct department."
    )
    return instructions, input_text


def make_few_shot_prompt(resume_snippet: str, label_list: list[str], examples: list[dict]) -> tuple[str, str]:
    """
    examples: list of dicts with keys {"resume_snippet", "department"}
    Output format is a single line:
      DEPARTMENT=<LABEL>;CONF=<NUMBER>
    """
    labels = build_label_set(label_list)
    instructions = (
        "You are routing resumes to the correct department.\n"
        "Use the examples to infer the routing standard.\n"
        "Return EXACTLY ONE LINE in this format:\n"
        "DEPARTMENT=<LABEL>;CONF=<NUMBER>\n"
        "Where <LABEL> is exactly one of the allowed labels, and <NUMBER> is a decimal in [0,1].\n"
        "The line must begin with DEPARTMENT=.\n"
        "No extra text. No markdown. No code fences.\n"
        f"Allowed labels: [{labels}]"
    )

    ex_blocks = []
    for k, ex in enumerate(examples, start=1):
        ex_blocks.append(
            f"Example {k} Resume:\n"
            "-----\n"
            f"{ex['resume_snippet']}\n"
            "-----\n"
            f"Correct department: {ex['department']}\n"
        )

    input_text = (
        "Here are labeled examples:\n\n"
        + "\n".join(ex_blocks)
        + "\nNow classify this new resume. Return exactly one line in the required format:\n"
          "-----\n"
        + f"{resume_snippet}\n"
          "-----\n"
    )
    return instructions, input_text

LINE_RE_STRICT = re.compile(
    r"DEPARTMENT\s*=\s*([^;]+)\s*;\s*CONF\s*=\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE
)

LINE_RE_LOOSE = re.compile(
    r"^\s*([^;]+)\s*;\s*CONF\s*=\s*([0-9]*\.?[0-9]+)\s*$",
    re.IGNORECASE
)

def parse_line_output(out: str) -> tuple[str, float]:
    s = (out or "").strip()

    # Strip markdown fences if present
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s).strip()
        s = re.sub(r"\s*```$", "", s).strip()

    # Strict: DEPARTMENT=...;CONF=...
    m = LINE_RE_STRICT.search(s)
    if m:
        dep = m.group(1).strip()
        conf = float(m.group(2))
        return dep, max(0.0, min(1.0, conf))

    # Loose: <LABEL>;CONF=...
    m = LINE_RE_LOOSE.search(s)
    if m:
        dep = m.group(1).strip()
        conf = float(m.group(2))
        return dep, max(0.0, min(1.0, conf))

    return "__PARSE_FAIL__", 0.0

_RETRY_AFTER_RE = re.compile(r"try again in ([0-9]*\.?[0-9]+)s", re.IGNORECASE)

def _sleep_seconds_from_rate_limit_message(msg: str, default: float = 5.0) -> float:
    """
    Extract 'Please try again in Xs' from the error message.
    Returns default if not found.
    """
    if not msg:
        return default
    m = _RETRY_AFTER_RE.search(msg)
    if not m:
        return default
    try:
        return float(m.group(1))
    except Exception:
        return default


def create_with_rate_limit_retry(client, *, max_retries: int = 8, **kwargs):
    """
    Calls client.responses.create with retries on rate limits/transient errors.
    - If RateLimitError includes 'try again in Xs', sleep that long (+ small jitter) then retry.
    - Otherwise use exponential backoff.
    """
    backoff = 1.0
    for attempt in range(max_retries + 1):
        try:
            return client.responses.create(**kwargs)

        except RateLimitError as e:
            # Prefer server-provided wait time if present
            wait = _sleep_seconds_from_rate_limit_message(str(e), default=backoff)
            # Add jitter so you don't sync up with yourself
            wait = wait + random.uniform(0.1, 0.6)
            if attempt == max_retries:
                raise
            print(f"[RateLimit] attempt {attempt+1}/{max_retries}: sleeping {wait:.2f}s")
            time.sleep(wait)
            # Increase fallback backoff
            backoff = min(backoff * 1.7, 30.0)

        except (APITimeoutError, APIConnectionError, APIError) as e:
            # transient-ish
            if attempt == max_retries:
                raise
            wait = backoff + random.uniform(0.1, 0.6)
            print(f"[TransientError] {type(e).__name__} attempt {attempt+1}/{max_retries}: sleeping {wait:.2f}s")
            time.sleep(wait)
            backoff = min(backoff * 2.0, 30.0)

def call_openai_classifier(resume_snippet: str, label_list: list[str], examples: list[dict] | None = None) -> tuple[str, float, str]:
    """
    Returns: (pred_department, confidence, raw_json_text)

    Requires: pip install openai
    and environment var OPENAI_API_KEY set.
    """

    client = OpenAI()

    if examples:
        instructions, input_text = make_few_shot_prompt(resume_snippet, label_list, examples)
    else:
        instructions, input_text = make_zero_shot_prompt(resume_snippet, label_list)

    resp = create_with_rate_limit_retry(
        client,
        model=MODEL,
        instructions=instructions,
        input=input_text,
        max_output_tokens=120,
    )

    out = resp.output_text.strip()
    dep, conf = parse_line_output(out)

    return dep, conf, out


# -----------------------------
# Evaluation / reporting
# -----------------------------
def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / max(1, len(y_true))


def top_confusions(y_true: list[str], y_pred: list[str], k: int = 15) -> list[tuple[tuple[str,str], int]]:
    c = Counter()
    for yt, yp in zip(y_true, y_pred):
        if yt != yp:
            c[(yt, yp)] += 1
    return c.most_common(k)


def main():
    random.seed(RANDOM_SEED)

    df = pd.read_csv(CLEAN_CSV_PATH)
    # Normalize column names just in case
    df.columns = [c.strip() for c in df.columns]

    # Basic sanity
    assert "department" in df.columns, "Expected column 'department'"
    assert "resume_number" in df.columns, "Expected column 'resume_number'"

    if useteam:
        df["department"] = df["department"].map(DEPT_TO_TEAM)

    # Build snippets + track how many headers removed
    snippets = []
    metas = []
    for _, row in df.iterrows():
        snippet, meta = build_resume_snippet(row, MAX_WORDS)
        snippets.append(snippet)
        metas.append(meta)

    df["snippet"] = snippets
    df["header_removed"] = [m["header_removed"] for m in metas]
    df["used_html"] = [m["used_html"] for m in metas]

    # Drop rows with empty snippet or missing department
    before = len(df)
    df = df[df["snippet"].astype(str).str.len() > 20].copy()
    df = df[df["department"].notna()].copy()
    after = len(df)
    print(f"Loaded {before} rows; keeping {after} with non-empty text + department.")
    print(f"Header line removed for {df['header_removed'].sum()} resumes.")
    print(f"Used HTML fallback for {df['used_html'].sum()} resumes.")

    # Build few-shot examples
    depts = sorted(df["department"].unique().tolist())
    print(f"Departments ({len(depts)}): {depts}")

    examples = []
    example_idx = set()

    for dept in depts:
        sub = df[df["department"] == dept]
        if len(sub) < EXAMPLES_PER_DEPT:
            raise ValueError(f"Not enough resumes in dept '{dept}' for {EXAMPLES_PER_DEPT} examples.")
        picked = sub.sample(n=EXAMPLES_PER_DEPT, random_state=RANDOM_SEED)
        for idx, row in picked.iterrows():
            examples.append({"resume_snippet": row["snippet"], "department": row["department"], "resume_number": row["resume_number"]})
            example_idx.add(idx)

    # Evaluation set = everything else
    eval_df = df.drop(index=list(example_idx)).copy().reset_index(drop=True)
    print(f"Few-shot examples: {len(examples)} (={EXAMPLES_PER_DEPT} per dept).")
    print(f"Evaluation resumes: {len(eval_df)}")

    # Pick one resume from eval set
    sample_row = eval_df.iloc[0]

    zs_instructions, zs_input_text = make_zero_shot_prompt(sample_row["snippet"], depts)

    print("\n================ ZERO-SHOT PROMPT (example) ================\n")
    print("INSTRUCTIONS:\n")
    print(zs_instructions)
    print("\nINPUT TEXT:\n")
    print(zs_input_text[:1200])   # truncate for readability
    print("\n===========================================================\n")

    # Show one few-shot example block
    ex = examples[0]

    example_block = (
        "Example Resume:\n"
        "-----\n"
        f"{ex['resume_snippet'][:800]}\n"
        "-----\n"
        f"Correct department: {ex['department']}\n"
    )

    print("\n================ FEW-SHOT EXAMPLE BLOCK =====================\n")
    print(example_block)
    print("\n============================================================\n")

    # --- Run evaluations ---
    # IMPORTANT: This will call the API once per eval resume (could be costly).
    # For quick iteration, start with a small eval subset:
    #eval_df = eval_df.sample(n=100, random_state=RANDOM_SEED).reset_index(drop=True)

    y_true = eval_df["department"].tolist()

    # Zero-shot
    zs_pred, zs_conf = [], []
    print("\nRunning zero-shot...")
    for i, row in eval_df.iterrows():
        pred, conf, _ = call_openai_classifier(row["snippet"], depts, examples=None)
        print(
            f"[ZERO-SHOT] Resume {row['resume_number']} | "
            f"True: {row['department']} | "
            f"Pred: {pred!r} | "
            f"Conf: {conf:.3f}"
        )
        zs_pred.append(pred)
        zs_conf.append(conf)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(eval_df)} done (zero-shot)")

    # Few-shot
    fs_pred, fs_conf = [], []
    print("\nRunning few-shot...")
    for i, row in eval_df.iterrows():
        pred, conf, _ = call_openai_classifier(row["snippet"], depts, examples=examples)
        print(
            f"[FEW-SHOT] Resume {row['resume_number']} | "
            f"True: {row['department']} | "
            f"Pred: {pred!r} | "
            f"Conf: {conf:.3f}"
        )
        fs_pred.append(pred)
        fs_conf.append(conf)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(eval_df)} done (few-shot)")

    # Report
    print("\n=== Results ===")
    print(f"Zero-shot accuracy: {accuracy(y_true, zs_pred):.3f}")
    print(f"Few-shot accuracy:  {accuracy(y_true, fs_pred):.3f}")

    print("\nTop zero-shot confusions (true -> pred):")
    for (t, p), n in top_confusions(y_true, zs_pred, k=15):
        print(f"  {t} -> {p}: {n}")

    print("\nTop few-shot confusions (true -> pred):")
    for (t, p), n in top_confusions(y_true, fs_pred, k=15):
        print(f"  {t} -> {p}: {n}")

    # Optional: save predictions
    out = eval_df[["resume_number", "department"]].copy()
    out["zs_pred"] = zs_pred
    out["zs_conf"] = zs_conf
    out["fs_pred"] = fs_pred
    out["fs_conf"] = fs_conf
    out_path = "C:\\Users\\chansen1\\Chicago Booth Dropbox\\Chris Hansen\\Public\\resume_routing_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote per-resume predictions to: {out_path}")


if __name__ == "__main__":
    main()
