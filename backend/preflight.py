# backend/preflight.py
#
# ══════════════════════════════════════════════════════════════════════
# PRE-FLIGHT VALIDATION  (S1.3 + S1.4)
# ══════════════════════════════════════════════════════════════════════
#
# Why this file exists
# --------------------
# Without these checks, several classes of bad input cause silent,
# confusing failures deep inside the pipeline:
#
#   • A .log file that is actually a binary dump, or a file with only
#     newlines, produces a Stage 1 run with 0 useful results and the
#     user gets no feedback on why.
#
#   • A 100 MB file with 15 million lines produces a Stage 2 manifest
#     with potentially 500k+ unique templates.  Stage 3's embedding
#     step will run for hours and produce meaningless clusters.
#
# Both cases are caught here before any stage runs, and a clear
# rejection reason is returned to the caller.
#
# Public API
# ----------
#   validate_log_content(path, cfg) -> (bool, str)
#       S1.3 — Checks content quality (parseable rate, binary noise).
#       Returns (True, "") on pass.
#       Returns (False, reason_string) on fail.
#
#   validate_log_size(path, cfg) -> (bool, str)
#       S1.4 — Checks line count and file size.
#       Returns (True, "") on pass.
#       Returns (False, reason_string) on fail.
#
# Usage in the API upload endpoint
# ---------------------------------
#   from preflight import validate_log_content, validate_log_size
#
#   ok, reason = validate_log_size(upload_path, cfg={})
#   if not ok:
#       raise HTTPException(status_code=422, detail=reason)
#
#   ok, reason = validate_log_content(upload_path, cfg={})
#   if not ok:
#       raise HTTPException(status_code=422, detail=reason)
#
# PLACEMENT: backend/preflight.py
# ══════════════════════════════════════════════════════════════════════

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple

# Allow imports from the project root (where config.py lives)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    PREFLIGHT_MIN_LINES,        # S1.3
    PREFLIGHT_MIN_PARSE_RATE,   # S1.3
    PREFLIGHT_MAX_BINARY_RATE,  # S1.3
    MAX_LOG_LINES,              # S1.4
    MAX_UPLOAD_SIZE_BYTES,      # S1.4 — reuse existing upload cap
)

# Stage 1's format probe and encoding detection are imported so the
# preflight check uses exactly the same patterns as the real parser.
# This guarantees that a file that passes preflight will also parse
# correctly in Stage 1 — no divergence between the two.
from stages.stage1 import _probe_format, _detect_encoding

logger = logging.getLogger("preflight")


# ══════════════════════════════════════════════════════════════════════
# S1.3 — CONTENT QUALITY VALIDATION
# ══════════════════════════════════════════════════════════════════════

def validate_log_content(
    path: Path,
    cfg:  dict,
) -> Tuple[bool, str]:
    """
    S1.3 — Validate that the file contains recognisable log content.

    Reads the first 500 lines and checks three things:

    1. Non-empty line count >= PREFLIGHT_MIN_LINES
       Catches empty files, files with only whitespace/newlines, and
       extremely short files that would produce no useful clusters.

    2. Parseable rate >= PREFLIGHT_MIN_PARSE_RATE
       Uses Stage 1's _probe_format() patterns to count how many of
       the first 500 lines match a known log format.  A file that is
       mostly unrecognisable (e.g. a binary dump decoded as text, or a
       CSV, or a config file) will have a very low parseable rate.

    3. Binary noise rate <= PREFLIGHT_MAX_BINARY_RATE
       Counts lines that contain non-printable characters outside the
       normal ASCII range.  A genuine log file should have very few
       such lines.

    Parameters
    ----------
    path : Path
        Absolute path to the file to validate.
    cfg  : dict
        Optional overrides for the thresholds.  Keys:
            preflight_min_lines       (default: PREFLIGHT_MIN_LINES)
            preflight_min_parse_rate  (default: PREFLIGHT_MIN_PARSE_RATE)
            preflight_max_binary_rate (default: PREFLIGHT_MAX_BINARY_RATE)

    Returns
    -------
    (True,  "")             — file passed all checks
    (False, reason_string)  — file failed; reason_string is human-readable
    """
    min_lines    = int(cfg.get("preflight_min_lines",       PREFLIGHT_MIN_LINES))
    min_parse    = float(cfg.get("preflight_min_parse_rate", PREFLIGHT_MIN_PARSE_RATE))
    max_binary   = float(cfg.get("preflight_max_binary_rate", PREFLIGHT_MAX_BINARY_RATE))

    # ── Detect encoding so we open the file the same way Stage 1 will ──
    encoding = _detect_encoding(path, fallback="utf-8")

    probe_lines  = 500
    non_empty    = 0
    binary_lines = 0

    try:
        with path.open("r", encoding=encoding, errors="replace") as fh:
            for i, raw in enumerate(fh):
                if i >= probe_lines:
                    break
                line = raw.rstrip("\n")
                if not line.strip():
                    continue
                non_empty += 1
                # Binary noise check: count non-printable, non-whitespace
                # characters outside the normal ASCII printable range.
                non_ascii = sum(
                    1 for c in line
                    if ord(c) > 127 or (ord(c) < 0x20 and c not in "\t\r")
                )
                if len(line) > 0 and (non_ascii / len(line)) > 0.30:
                    binary_lines += 1

    except Exception as exc:
        reason = f"Could not read file for pre-flight content check: {exc}"
        logger.warning("preflight content: %s — %s", path.name, reason)
        return False, reason

    # ── Check 1: minimum line count ────────────────────────────────────
    if non_empty < min_lines:
        reason = (
            f"File contains only {non_empty} non-empty line(s); "
            f"minimum required is {min_lines}.  "
            f"The file may be empty, contain only whitespace, or be too short "
            f"to produce meaningful analysis."
        )
        logger.info("preflight content FAIL (min_lines): %s — %s", path.name, reason)
        return False, reason

    # ── Check 2: binary noise rate ─────────────────────────────────────
    binary_rate = binary_lines / max(non_empty, 1)
    if binary_rate > max_binary:
        reason = (
            f"File appears to be binary or heavily encoded "
            f"({binary_rate * 100:.1f}% of sampled lines contain binary noise; "
            f"maximum allowed is {max_binary * 100:.0f}%).  "
            f"Please upload a plain-text log file."
        )
        logger.info("preflight content FAIL (binary_rate): %s — %s", path.name, reason)
        return False, reason

    # ── Check 3: parseable rate ─────────────────────────────────────────
    # _probe_format() reads the file independently with its own 200-line
    # sample and returns a dict of {format_name: fraction}.  We sum all
    # fractions to get the overall parseable rate.
    try:
        probe_result = _probe_format(
            path, encoding=encoding, errors="replace", probe_lines=probe_lines
        )
        parseable_rate = sum(probe_result.values()) if probe_result else 0.0
    except Exception as exc:
        logger.warning(
            "preflight content: _probe_format failed for %s — %s; "
            "skipping parseable-rate check",
            path.name, exc,
        )
        parseable_rate = 1.0  # assume parseable if probe fails — don't block the run

    if parseable_rate < min_parse:
        reason = (
            f"Only {parseable_rate * 100:.1f}% of sampled lines match a recognised "
            f"log format (minimum required: {min_parse * 100:.0f}%).  "
            f"The file may not be a log file, or may use an unsupported format."
        )
        logger.info("preflight content FAIL (parse_rate): %s — %s", path.name, reason)
        return False, reason

    logger.info(
        "preflight content PASS: %s  non_empty=%d  binary_rate=%.2f  parse_rate=%.2f",
        path.name, non_empty, binary_rate, parseable_rate,
    )
    return True, ""


# ══════════════════════════════════════════════════════════════════════
# S1.4 — SIZE / LINE-COUNT GATE
# ══════════════════════════════════════════════════════════════════════

def validate_log_size(
    path: Path,
    cfg:  dict,
) -> Tuple[bool, str]:
    """
    S1.4 — Validate that the file is within acceptable size limits.

    Checks two things:

    1. File size in bytes <= MAX_UPLOAD_SIZE_BYTES
       Reuses the same cap used by the API upload endpoint.

    2. Line count <= MAX_LOG_LINES
       Counted by reading the file in buffered chunks (never loaded
       fully into memory).  A file with millions of lines will produce
       a Stage 2 manifest with hundreds of thousands of unique
       templates and make Stage 3 embedding infeasible.

    Parameters
    ----------
    path : Path
        Absolute path to the file to validate.
    cfg  : dict
        Optional overrides.  Keys:
            max_log_lines         (default: MAX_LOG_LINES)
            max_upload_size_bytes (default: MAX_UPLOAD_SIZE_BYTES)

    Returns
    -------
    (True,  "")             — file is within limits
    (False, reason_string)  — file exceeds a limit
    """
    max_lines     = int(cfg.get("max_log_lines",         MAX_LOG_LINES))
    max_size      = int(cfg.get("max_upload_size_bytes", MAX_UPLOAD_SIZE_BYTES))

    # ── Check 1: file size ─────────────────────────────────────────────
    try:
        size_bytes = path.stat().st_size
    except Exception as exc:
        reason = f"Could not stat file for pre-flight size check: {exc}"
        logger.warning("preflight size: %s — %s", path.name, reason)
        return False, reason

    if size_bytes > max_size:
        size_mb     = size_bytes / (1024 * 1024)
        max_size_mb = max_size   / (1024 * 1024)
        reason = (
            f"File is {size_mb:.1f} MB; maximum accepted size is "
            f"{max_size_mb:.0f} MB.  Please reduce the file size or split "
            f"it into smaller chunks before uploading."
        )
        logger.info("preflight size FAIL (file_size): %s — %s", path.name, reason)
        return False, reason

    # ── Check 2: line count (buffered, never fully in memory) ──────────
    line_count = 0
    chunk_size = 1024 * 1024  # 1 MB read buffer — large enough for speed,
                               # small enough never to OOM on this check alone
    try:
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                line_count += chunk.count(b"\n")
                if line_count > max_lines:
                    # Fail early — no need to count the rest
                    reason = (
                        f"File exceeds the maximum line count of "
                        f"{max_lines:,} lines.  Files this large would cause "
                        f"Stage 3 embedding to run for an impractical length "
                        f"of time.  Please split the file or use a smaller "
                        f"time window."
                    )
                    logger.info(
                        "preflight size FAIL (line_count): %s — %s",
                        path.name, reason,
                    )
                    return False, reason

    except Exception as exc:
        reason = f"Could not count lines for pre-flight size check: {exc}"
        logger.warning("preflight size: %s — %s", path.name, reason)
        return False, reason

    logger.info(
        "preflight size PASS: %s  size_bytes=%d  line_count=%d",
        path.name, size_bytes, line_count,
    )
    return True, ""