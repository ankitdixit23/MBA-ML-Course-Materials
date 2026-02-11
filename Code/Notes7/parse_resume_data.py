from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import zipfile

with zipfile.ZipFile("C:\\Users\\chansen1\\Chicago Booth Dropbox\\Chris Hansen\\Public\\ResumeData.zip") as z:
    file_count = sum(1 for f in z.infolist() if not f.is_dir())

print(file_count)


IN_PATH = Path("C:\\Users\\chansen1\\Chicago Booth Dropbox\\Chris Hansen\\Public\\Resume.csv")
OUT_PATH = Path("C:\\Users\\chansen1\\Chicago Booth Dropbox\\Chris Hansen\\Public\\Resume_cleaned.csv")
REPORT_PATH = Path("C:\\Users\\chansen1\\Chicago Booth Dropbox\\Chris Hansen\\Public\\Resume_cleaning_report.txt")

# Record-start heuristic: line begins with an 8 digit integer resume id followed by a comma
START_RE = re.compile(r"^\s*(\d{7,8})\s*,")


@dataclass
class Stats:
    physical_lines: int = 0
    logical_records: int = 0
    written: int = 0
    dropped: int = 0
    repaired_len_lt4: int = 0
    repaired_len_gt4: int = 0
    stitched_continuations: int = 0
    ignored_leading_junk_lines: int = 0
    bad_examples: List[str] = None

    def __post_init__(self):
        if self.bad_examples is None:
            self.bad_examples = []


def stitch_lines(lines: List[str], stats: Stats) -> List[str]:
    """
    Stitch physical lines into logical records using START_RE.
    Continuation lines (not matching START_RE) are appended to current buffer.
    """
    records: List[str] = []
    buf: List[str] = []

    for line in lines:
        line = line.rstrip("\n\r")
        if not line.strip() and not buf:
            # ignore leading blanks
            stats.ignored_leading_junk_lines += 1
            continue

        if START_RE.match(line):
            # start a new record
            if buf:
                records.append("\n".join(buf))
            buf = [line]
        else:
            # continuation or junk
            if buf:
                buf.append(line)
                stats.stitched_continuations += 1
            else:
                # junk before first true record
                stats.ignored_leading_junk_lines += 1

    if buf:
        records.append("\n".join(buf))

    return records


def parse_first_row(stitched_record: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """
    Parse the stitched record as a single CSV row.
    We only take the first parsed row; stitching is intended to ensure it's one.
    """
    try:
        f = io.StringIO(stitched_record)
        reader = csv.reader(f)
        row = next(reader)
        return row, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def repair_to_4_cols(row: List[str], stats: Stats) -> Optional[List[str]]:
    """
    Force row into [resume_number, resume_text, resume_text_html, department].

    - len==4: ok
    - len<4: pad with empty strings
    - len>4: keep first as id, last as department, join middle into one blob
             and split into (text, html) at first HTML-ish marker.
    """
    if row is None:
        return None

    if len(row) == 4:
        return row

    if len(row) < 4:
        stats.repaired_len_lt4 += 1
        return row + [""] * (4 - len(row))

    stats.repaired_len_gt4 += 1
    rid = row[0]
    dept = row[-1]
    middle_blob = ",".join(row[1:-1])

    # Heuristic boundary for html:
    # look for common tags / encoded tags. Use earliest occurrence.
    markers = [
        "<html", "<div", "<p", "<span", "<table", "<!DOCTYPE", "<body",
        "&lt;html", "&lt;div", "&lt;p", "&lt;span", "&lt;table", "&lt;!DOCTYPE",
        "<"
    ]
    idx = None
    for m in markers:
        j = middle_blob.find(m)
        if j != -1:
            idx = j
            break

    if idx is None:
        resume_text = middle_blob
        resume_html = ""
    else:
        resume_text = middle_blob[:idx].rstrip()
        resume_html = middle_blob[idx:].lstrip()

    return [rid, resume_text, resume_html, dept]


def looks_like_header(row: List[str]) -> bool:
    """
    Drop header-like rows if present.
    """
    if not row:
        return True
    first = (row[0] or "").strip().lower()
    return first in {"resume_number", "resumenumber", "id"}  # cheap guard


def main() -> None:
    stats = Stats()

    raw = IN_PATH.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    stats.physical_lines = len(lines)

    records = stitch_lines(lines, stats)
    stats.logical_records = len(records)

    cleaned: List[List[str]] = []

    for rec in records:
        row, err = parse_first_row(rec)
        if row is None:
            stats.dropped += 1
            if len(stats.bad_examples) < 5:
                stats.bad_examples.append(f"PARSE_FAIL: {err}\n---\n{rec[:600]}\n---\n")
            continue

        if looks_like_header(row):
            # ignore possible header row
            continue

        row4 = repair_to_4_cols(row, stats)
        if row4 is None or len(row4) != 4:
            stats.dropped += 1
            if len(stats.bad_examples) < 5:
                stats.bad_examples.append(f"REPAIR_FAIL: len={len(row)}\n---\n{rec[:600]}\n---\n")
            continue

        # Clean fields
        rid = (row4[0] or "").strip()
        dept = (row4[3] or "").strip()
        row4[0] = rid
        row4[3] = dept

        # Enforce 8-digit id
        if not re.fullmatch(r"\d{7,8}", rid):
            stats.dropped += 1
            if len(stats.bad_examples) < 5:
                stats.bad_examples.append(f"BAD_ID: {rid!r}\n---\n{rec[:600]}\n---\n")
            continue

        cleaned.append(row4)

    # Write cleaned CSV
    with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["resume_number", "resume_text", "resume_text_html", "department"])
        w.writerows(cleaned)

    stats.written = len(cleaned)

    # Report
    report_lines = [
        f"Input:  {IN_PATH}",
        f"Output: {OUT_PATH}",
        "",
        "=== Summary ===",
        f"Physical lines read:                  {stats.physical_lines}",
        f"Logical records after stitching:      {stats.logical_records}",
        f"Continuation lines stitched:          {stats.stitched_continuations}",
        f"Ignored leading junk/blank lines:     {stats.ignored_leading_junk_lines}",
        f"Rows written:                         {stats.written}",
        f"Dropped records:                      {stats.dropped}",
        "",
        "=== Repairs Applied ===",
        f"Repaired rows (len < 4):              {stats.repaired_len_lt4}",
        f"Repaired rows (len > 4):              {stats.repaired_len_gt4}",
        "",
    ]
    if stats.bad_examples:
        report_lines.append("=== First bad examples (truncated) ===")
        report_lines.extend(stats.bad_examples)

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Wrote cleaned CSV to: {OUT_PATH}")
    print(f"Wrote report to:      {REPORT_PATH}")
    print(f"Rows written: {stats.written} (dropped {stats.dropped})")


if __name__ == "__main__":
    main()