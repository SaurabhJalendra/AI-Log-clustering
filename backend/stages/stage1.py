"""
stages/stage1.py
================
Stage 1 — Ingestion & Format Detection for SkyAI.

Extracted verbatim from the solution3.ipynb notebook.
Public API:
    run_stage1(input_path, **kwargs) -> (chunk_iterator, ParseStats)

Usage:
    from stages.stage1 import run_stage1
    chunk_iter, stats = run_stage1("/path/to/log.log")
    import pandas as pd
    df = pd.concat(list(chunk_iter), ignore_index=True)

MPCD alignment (§3.1 — Solid, No Changes Needed):
    Stage 1 requires no runtime LLM integration.  Regex parsing is faster,
    deterministic, and correct for this stage.

    Naming consistency (§3.1 note): the `service` field produced by this
    stage (e.g. "auth-service", "payment-svc") is the raw process/component
    name extracted from logs.  It is NOT the same as the Stage 2 domain name
    (e.g. "auth", "payment").  Downstream stages (Stage 2 domain_priority,
    Stage 5 STAGE5_CONFIG domain_priority) must use domain names, not service
    field values.  See _JAVA_PACKAGE_TO_SERVICE for annotated examples.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import pandas as pd
from zoneinfo import ZoneInfo

# ── Top-level regex anchors ───────────────────────────────────────────
RE_YEAR_START     = re.compile(r'^\d{4}[/\-T]')
RE_IP_START       = re.compile(r'^\d{1,3}\.\d')
RE_DATE_START     = re.compile(r'^\d{2}/')
RE_HAPROXY_STATUS = re.compile(r'\b(\d{3})\s+\d+\s+-\s+-\s+[-\w]+\s+\d+/\d+\b')

# ── Format-specific regexes ───────────────────────────────────────────
RE_SYSLOG_F1 = re.compile(
    r'^(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+'
    r'(?P<host>[\w\-.]+)\s+(?P<svc>[\w\-.]+)(?:\[(?P<pid>\d+)\])?:\s*'
    r'(?:<(?P<sev>[A-Za-z]+)>\s*)?(?P<msg>.*)$')
RE_ISO_SPACE_BRACKET_F2 = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+'
    r'(?P<sev>TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL)\s+\[(?P<svc>[^\]]+)\]\s*(?P<msg>.*)$',
    re.I)
RE_RFC3339Z_COLON_F3 = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2}T[\d:\.]+Z)\s+'
    r'(?P<sev>TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL)\s+(?P<svc>[\w\-.]+):\s*(?P<msg>.*)$',
    re.I)
RE_BRACKET_SHORT_F4 = re.compile(
    r'^\[(?P<time>\d{2}:\d{2}:\d{2})\]\s*(?P<sevabbr>[IWE DT])\s*/\s*(?P<svc>[\w\-.]+):\s*(?P<msg>.*)$')
RE_APACHE_PIPE_F6 = re.compile(
    r'^(?P<ts>\d{2}/[A-Z][a-z]{2}/\d{4}:\d{2}:\d{2}:\d{2}\s+[+-]\d{4})\s*'
    r'\|\s*(?P<sev>TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL)\s*\|\s*(?P<svc>[\w\-.]+)\s*\|\s*(?P<msg>.*)$',
    re.I)
RE_ISO_PIPE_F7 = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+'
    r'(?P<sev>TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL)\s*\|\s*(?P<svc>[\w\-.]+)\s*\|\s*(?P<msg>.*)$',
    re.I)
RE_NGINX_ERR_F8 = re.compile(
    r'^(?P<ts>\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})\s+\[(?P<sev>\w+)\]\s+'
    r'(?P<pid>\d+)#\d+:\s+\*\d+\s+(?P<msg>.+?)(?=,\s*(?:client|server|request|upstream|host):|$)'
    r'(?:.*,\s*client:\s*(?P<client>[\d.]+))?(?:.*,\s*server:\s*(?P<server>[^,]+))?'
    r'(?:.*,\s*request:\s*"(?P<request>[^"]*)")?(?:.*,\s*upstream:\s*"(?P<upstream>[^"]*)")?.*$',
    re.DOTALL)
RE_ACCESS_F9 = re.compile(
    r'^(?P<client>[\d.]+)\s+-\s+(?P<user>\S+)\s+\[(?P<ts>[^\]]+)\]\s+'
    r'"(?P<method>\w+)\s+(?P<path>\S+)\s+HTTP/[\d.]+"\s+(?P<status>\d{3})\s+(?P<bytes>\d+)'
    r'(?:\s+"(?P<referer>[^"]*)")?(?:\s+"(?P<ua>[^"]*)")?(?:\s+rt=(?P<rt>[\d.]+))?\s*.*$')
RE_POSTGRES_F10 = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\s+UTC\s+\[(?P<pid>\d+)\]\s+'
    r'user=(?P<user>[^,]+),db=(?P<db>[^,]+),app=(?P<svc>[^,]+),client=(?P<client>\S+)\s+'
    r'(?P<sev>ERROR|WARNING|FATAL|LOG|NOTICE|INFO|DEBUG|STATEMENT|HINT|DETAIL|CONTEXT):\s+(?P<msg>.+)$',
    re.I)
RE_JAVA_F11 = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}),\d{3}\s+'
    r'(?P<sev>TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL)\s+\[(?P<thread>[^\]]+)\]\s+'
    r'(?P<class>[\w.$]+)\s+-\s+(?P<msg>.+)$',
    re.I)
RE_CEF_F12 = re.compile(
    r'^CEF:(?P<cef_ver>\d+)\|(?P<vendor>[^|]*)\|(?P<product>[^|]*)\|(?P<prod_ver>[^|]*)\|'
    r'(?P<event_id>[^|]*)\|(?P<event_name>[^|]*)\|(?P<severity>\d+)\|(?P<ext>.*)$')
RE_AUDIT_F13 = re.compile(
    r'^type=(?P<audit_type>\w+)\s+msg=audit\((?P<epoch>[\d.]+):(?P<serial>\d+)\):\s+(?P<fields>.+)$')
RE_K8S_EVT_F14 = re.compile(
    r'^(?P<age>\d+[smh](?:\d+[smh])?)\s+(?P<etype>Normal|Warning)\s+(?P<reason>\S+)\s+'
    r'(?P<object>\S+/\S+)\s+(?P<msg>.+)$')
RE_WINDOWS_EVT_F15 = re.compile(
    r'^(?P<ts>\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}\s+(?:AM|PM))\s+'
    r'EventID=(?P<event_id>\d+)\s+Level=(?P<sev>\w+)\s+Provider=(?P<provider>.+?)\s+Message="(?P<msg>[^"]*)"',
    re.I)
RE_SYSTEMD_F16 = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2}T[\d:.]+[+-]\d{2}:\d{2})\s+host=(?P<host>\S+)\s+'
    r'unit=(?P<unit>\S+)\s+priority=(?P<priority>\d+)\s+msg="(?P<msg>[^"]*)"',
    re.I)

# ── F17: PM2 wrapper format ───────────────────────────────────────────
# PM2 v4 emits second-resolution timestamps: 2024-01-15T10:23:45:
# PM2 v5+ emits millisecond-resolution:      2024-01-15T10:23:45.123:
# The fractional part is optional so both variants are matched.
RE_PM2_OUTER = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?):[ \t]*(?P<inner>.*)$'
)
RE_PM2_EXPRESS_INNER = re.compile(
    r'^(?P<itime>\d{2}:\d{2}:\d{2} (?:AM|PM)) \[express\] '
    r'(?P<method>GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS) '
    r'(?P<path>\S+) (?P<status>\d{3})'
    r'(?:\u2026|\.\.\.)?'
    r'(?:\s+in\s+(?P<ms>\d+)ms)?'
    r'(?:\s*::\s*(?P<body>.*))?$',
    re.S,
)
RE_PM2_BRACKET_INNER = re.compile(
    r'^(?P<tags>(?:\[[^\]]+\])+)\s*(?P<msg>.*)$'
)
RE_PM2_SYMBOL_NOISE = re.compile(r'^\[Symbol\(')
RE_PM2_STACK_AT = re.compile(r'^\s*at\s+[\w.<>\[\]$]+[\s(]')
RE_PM2_STRUCTURAL_TOKEN = re.compile(r'^\s*(?:[{}\[\]],?|},|],)\s*$')
RE_PM2_OBJ_PROP = re.compile(r'^\s{0,6}[\w\[\].()\\s]+:\s')
RE_PM2_OBJ_PROP_UNINDENTED = re.compile(
    r'^[\w_]+:\s+(?:\'[^\']*\'|"[^"]*"|\d+|true|false|null|\[\]|\{\}|<\*>|undefined)\s*,?\s*$'
)
RE_PM2_PDF_STRUCTURAL = re.compile(
    r'^(?:'
    r'\d+\s+\d+\s+obj\b'
    r'|endobj\b|endstream\b|stream\b'
    r'|xref\b|startxref\b|trailer\b|%%EOF'
    r'|<<|>>|<</'
    r'|\d+\s+\d+\s+R\b'
    r'|%(?:PDF|%)'
    r'|\d{10}\s+\d{5}\s+[fn]\b'
    r'|\d+\s+\d+\s*$'
    r'|\d{5,}\b'
    r'|<\?xpacket\b'
    r'|</?(?:rdf|x|dc|xmp|xmpMM|pdf|adhocwf)[\s:>]'
    r'|xmlns:\w+='
    r'|rm/BBox\b'
    r')'
)
# ── Fix 2: HTTP status-code severity overrides ───────────────────────
# Applied as a post-processing step on f17a/b/c PM2 rows after the initial
# severity is set from the pm2 log prefix.
RE_HTTP_5XX = re.compile(
    r'(?:GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+\S+\s+5\d\d\b', re.I)
RE_HTTP_4XX = re.compile(
    r'(?:GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+\S+\s+4\d\d\b', re.I)
RE_HTTP_401 = re.compile(
    r'(?:GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+\S+\s+401\b', re.I)
RE_LOG_ARCHIVE_SUCCESS = re.compile(
    r'finished:\s*\d+\s+file\(s\)\s+archived,\s*0\s+error\(s\)', re.I)

_HTTP_SEV_OVERRIDE_FORMATS = frozenset({
    "f17a_pm2_express", "f17b_pm2_bracket", "f17c_pm2_plain"
})


def _apply_http_severity_override(msg: str, current_sev: str, fmt: str) -> str:
    """
    Post-processing severity override for PM2 express/bracket/plain lines.

    Rules (applied in order):
      1. Log-archive success line falsely logged at ERROR → INFO
      2. HTTP 5xx → ERROR
      3. HTTP 401 that is currently ERROR → downgrade to WARN (auth failure,
         not a server error)
      4. HTTP 4xx that is currently INFO → upgrade to WARN
    """
    if fmt not in _HTTP_SEV_OVERRIDE_FORMATS:
        return current_sev
    if not msg:
        return current_sev

    # Rule 1: log-archive service emits ERROR on successful completion
    if RE_LOG_ARCHIVE_SUCCESS.search(msg):
        return "INFO"

    # Rule 2: any HTTP 5xx → ERROR
    if RE_HTTP_5XX.search(msg):
        return "ERROR"

    # Rule 3: HTTP 401 downgrade (401 is auth failure, not a server error)
    if RE_HTTP_401.search(msg) and current_sev == "ERROR":
        return "WARN"

    # Rule 4: HTTP 4xx upgrade from INFO → WARN
    if RE_HTTP_4XX.search(msg) and current_sev == "INFO":
        return "WARN"

    return current_sev


RE_PM2_PDF_OP = re.compile(
    r'^(?:'
    r'EMC|/Tx\s+BMC'
    r'|(?:q|Q)\s*$'
    r'|(?:W\*?|BT|ET)\s*$'
    r'|n\s*$|f\s*$'
    r'|(?:0 g|1 w|1 g)\s*$'
    r'|re\s*$'
    r'|Tf\b|Td\b|Tj\b|TJ\b'
    r'|cm\b|Do\b|cs\b|gs\b'
    r'|/\w[\w.]* \d+(?:\.\d+)? Tf\b'
    r'|[-\d.]+ [-\d.]+ Td\b'
    r'|[-\d.]+ [-\d.]+ [-\d.]+ [-\d.]+ re\b'
    r'|[-\d. ]+ [-\d.]+ [-\d.]+ [-\d.]+ [-\d.]+ [-\d.]+ cm\b'
    r'|\((?:[^()\\]|\\.)*\)\s*(?:Tj\b)?'
    r'|\[[^\]]*\]\s*$'
    r'|(?:[-\d]+\s+){3,}[-\d]+\s*$'
    r')'
)

# ── Severity & service normalisation maps ─────────────────────────────
SEV_MAP = {
    "WARNING": "WARN",  "WARN": "WARN",    "INFO": "INFO",
    "ERROR":   "ERROR", "DEBUG": "DEBUG",   "TRACE": "TRACE",
    "FATAL":   "FATAL", "CRITICAL": "ERROR","NOTICE": "INFO",
    "SEVERE":  "ERROR", "LOG": "INFO",      "STATEMENT": "INFO",
    "HINT":    "INFO",  "DETAIL": "INFO",   "CONTEXT": "INFO",
    "INFORMATION": "INFO", "VERBOSE": "DEBUG",
    "AUDIT_SUCCESS": "INFO", "AUDIT_FAILURE": "WARN",
}
SEV_ABBR_MAP = {"I": "INFO", "W": "WARN", "E": "ERROR", "D": "DEBUG", "T": "TRACE", " ": None}
_SYSTEMD_PRIORITY_MAP = {
    "0": "ERROR", "1": "ERROR", "2": "ERROR", "3": "ERROR",
    "4": "WARN",  "5": "INFO",  "6": "INFO",  "7": "DEBUG",
}

# ── Fix 3: Service name blocklist ────────────────────────────────────
# Values that can never be valid service names — they are severity levels,
# generic log tokens, or null-ish placeholders.  If a parser extracts one
# of these as a service name it falls back to 'unknown'.
# Pattern validation: service names must match [a-z][a-z0-9-]* (lowercase
# alphanumeric with hyphens).  Anything not matching is also set to 'unknown'.
_SERVICE_NAME_BLOCKLIST: frozenset = frozenset({
    "warn", "warning", "error", "debug", "info", "trace",
    "critical", "fatal", "log", "stdout", "stderr",
    "null", "undefined", "none",
})
_RE_VALID_SERVICE_NAME = re.compile(r'^[a-z][a-z0-9-]*$')


def _sanitise_service(svc: Optional[str]) -> Optional[str]:
    """
    Return 'unknown' if svc is a blocklisted token or doesn't match the
    valid service-name pattern [a-z][a-z0-9-]*.  Used as a post-extraction
    guard in _pm2_bracket_service and _infer_pm2_plain_service.
    """
    if not svc:
        return svc
    lower = svc.strip().lower()
    if lower in _SERVICE_NAME_BLOCKLIST:
        return "unknown"
    if not _RE_VALID_SERVICE_NAME.match(lower):
        return "unknown"
    return lower


_WINDOWS_PROVIDER_NORM = {
    "microsoft-windows-security-auditing":   "windows-security",
    "microsoft-windows-kernel-general":      "windows-kernel",
    "microsoft-windows-kernel-power":        "windows-kernel",
    "microsoft-windows-kernel-boot":         "windows-kernel",
    "microsoft-windows-kernel-pnp":          "windows-kernel",
    "microsoft-windows-wlan-autoconfig":     "windows-wlan",
    "microsoft-windows-dns-client":          "windows-dns",
    "microsoft-windows-dhcp-client":         "windows-dhcp",
    "microsoft-windows-bits-client":         "windows-bits",
    "microsoft-windows-winlogon":            "windows-winlogon",
    "microsoft-windows-user-profiles-service": "windows-profiles",
    "microsoft-windows-windowsupdateclient": "windows-update",
    "service control manager":               "windows-scm",
    "application error":                     "windows-apperror",
}

# Base Java package-fragment → service name map.
# These are intentionally generic fragments that appear across many Java stacks.
# App-specific fragments (e.g. com.acme.billing) should be added via
# extend_java_package_map() rather than editing this dict directly.
#
# ── NAMING CONSISTENCY NOTE (MPCD §3.1) ──────────────────────────────
# The VALUES in this map are SERVICE field values (e.g. "auth-service").
# They are distinct from DOMAIN names used in Stage 2+ (e.g. "auth").
# These are two separate concepts:
#
#   service field  → what process/component emitted the log  → "auth-service"
#   domain         → what business/technical area it belongs to → "auth"
#
# Never use a service field value (e.g. "auth-service") as a domain name
# in domain_priority, domain_taxonomy, or keyword dicts.  The canonical
# domain name for authentication is "auth" (not "authentication", not
# "auth-service").  Stage 2 keyword matching, Stage 5 domain_priority,
# and all schema contracts use the domain name, not the service field.
# ─────────────────────────────────────────────────────────────────────
_JAVA_PACKAGE_TO_SERVICE: Dict[str, str] = {
    # Generic domain fragments (safe across most Java apps)
    "billing":   "payment-svc",
    "payment":   "payment-svc",
    "inventory": "inventory",
    "auth":      "auth-service",   # service field: "auth-service" | domain: "auth"
    "security":  "auth-service",   # service field: "auth-service" | domain: "security"
    "messaging": "email-service",
    "email":     "email-service",
    "storage":   "storage",
    "scheduler": "scheduler",
    "gateway":   "api-gateway",
    "api":       "api-gateway",
    "profile":   "profile-svc",
    "search":    "search-service",
    # Common Java framework package fragments that should NOT become service names
    # — these are explicitly excluded to prevent misleading service labels like
    # "runtime", "context", "servlet", "handler", "controller".
}

# Fragments whose last path component should NOT be used as a service name
# (they are framework internals, not application services).
_JAVA_GENERIC_CLASS_FRAGMENTS: frozenset = frozenset({
    "runtime", "context", "servlet", "handler", "controller",
    "filter", "interceptor", "dispatcher", "factory", "provider",
    "manager", "registry", "bootstrap", "launcher", "main",
    "application", "app", "server", "client", "core", "util", "utils",
    "helper", "base", "abstract", "impl", "service",
})

# ── Fix 6: Domain vs Service validation (S1-6) ───────────────────────
# Bare domain names used by Stage 2+ taxonomy and domain_priority configs.
# A service field value that exactly matches one of these is almost certainly
# a misclassification — the developer accidentally used a domain name where a
# service field value (e.g. "auth-service") was expected.
#
# Detection only: a DEBUG warning is emitted at Stage 1 output time.
# No blocking, no assertion error, no pipeline disruption.
# The Java package map already maps "auth" → "auth-service" correctly;
# this guard catches collisions from other format parsers (PM2 plain,
# JSON, relaxed) that infer service names from raw message content.
#
# To extend this set for your deployment, call extend_domain_name_set()
# before the first run_stage1() invocation.
_KNOWN_DOMAIN_NAMES: frozenset = frozenset({
    "auth", "authentication", "payment", "billing", "inventory",
    "storage", "scheduler", "api", "security", "messaging", "email",
    "database", "network", "search", "profile", "gateway", "metrics",
    "monitoring", "logging", "cache", "cdn", "dns", "compute",
    "notification", "reporting", "analytics", "deployment",
})

_S1_DOMAIN_COLLISION_LOGGER = logging.getLogger("stage1_parser.domain_collision")


def extend_domain_name_set(extra_domains) -> None:
    """
    Register additional bare domain names that should never appear as
    service field values.  Call before the first run_stage1() invocation.

    Example::

        from stages.stage1 import extend_domain_name_set
        extend_domain_name_set({"orders", "fulfillment", "returns"})
    """
    # frozensets are immutable; swap the module-level reference instead.
    global _KNOWN_DOMAIN_NAMES
    _KNOWN_DOMAIN_NAMES = _KNOWN_DOMAIN_NAMES | frozenset(extra_domains)


def _check_service_domain_collision(service_value: str, line_no: int) -> None:
    """
    Emit a DEBUG warning if a service field value exactly matches a known
    bare domain name.  This is a non-blocking diagnostic only.

    Collision scenario (from Fix 6 report):
        A downstream developer adds "auth" to domain_priority thinking it is
        a domain filter, but the Stage 1 service field for that line already
        contains "auth" — silent misclassification with no warning surfaced.

    The Java package map already prevents this for Java logs by mapping
    "auth" → "auth-service".  This guard covers PM2 plain, JSON, relaxed,
    and any other parser that may infer bare domain names directly.
    """
    if service_value and service_value.lower() in _KNOWN_DOMAIN_NAMES:
        _S1_DOMAIN_COLLISION_LOGGER.debug(
            "line %d: service field value %r exactly matches a known domain "
            "name.  Stage 1 service fields should be process/component names "
            "(e.g. 'auth-service'), not domain names (e.g. 'auth').  "
            "Downstream domain_priority configs use domain names — a bare "
            "domain name in the service field may cause silent "
            "misclassification in Stage 2+.  Consider updating "
            "_JAVA_PACKAGE_TO_SERVICE or _PM2_KEYWORD_SERVICE_MAP to produce "
            "a qualified service name (e.g. 'auth-service', 'auth-svc').",
            line_no,
            service_value,
        )


def extend_java_package_map(extra: Dict[str, str]) -> None:
    """
    Register additional package-fragment → service mappings at runtime.
    Call this from your deployment config before the first run_stage1() call.

    Example:
        from stages.stage1 import extend_java_package_map
        extend_java_package_map({"reporting": "report-svc", "orders": "order-svc"})
    """
    _JAVA_PACKAGE_TO_SERVICE.update(extra)

# ── _DATEABLE: patterns used by _infer_default_date_from_file ─────────
# Compiled once at module level rather than inside the function body to
# avoid repeated re.compile() calls (even with the LRU cache, this is
# cleaner and guaranteed zero overhead on repeated pipeline runs).
_DATEABLE_PATTERNS = [
    (re.compile(r'^(?P<ts>\d{4}-\d{2}-\d{2}T[\d:.]+Z)'),              "f3"),
    (re.compile(r'^(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})'),  "f2"),
    (re.compile(r'^(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})'), "f1"),
    (re.compile(r'^(?P<ts>\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})'),  "f8"),
    (re.compile(r'^(?P<ts>\d{4}-\d{2}-\d{2}T[\d:.]+[+-]\d{2}:\d{2})'), "f16"),
    (re.compile(r'^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}):'),   "f17"),
]

# ── Noise regex ───────────────────────────────────────────────────────
# Patterns merged from Stage 2's _NOISE_PATTERNS (Section 1C).
# NOTE: Raw-line noise detection is owned exclusively by Stage 1 (is_noise column).
RE_NOISE = re.compile(
    r'^\s*$|^\s*(null|none|n/a)\s*$|^\s*at\s+[\w.$<>]+\s*\(.*\)\s*$'
    r'|^\s*\.\.\.\s*\d*\s*more\s*$|\^\s*C\s*$|#{5,}|={3,}\s*(log|switch|rotat)'
    r'|==>\s*\S|\s*-{3,}\s*connection\s+reset|\bgoroutine\s+\d+\s+\['
    r'|^panic:|\s*main\.\w+\(0x[0-9a-f]|\s*/[\w./]+\.go:\d+|\?{3,}'
    r'|==>\s*.+\s*<==$|[^\x09\x0a\x0d\x20-\x7e]|^[\w/]+\.\(\*?[\w]+\)\.[\w]+\('
    # Merged from Stage 2 _NOISE_PATTERNS (Section 1C):
    r'|^\s*\d+\|\s*'                           # Pipe-prefixed line numbers (CI output)
    r'|^(PASS|FAIL|OK|SKIP|ERROR)\s*$'         # Bare single-word test result tokens
    r'|^\s+\.\.\.'                             # Continuation ellipsis lines
    r'|^\s*/[a-zA-Z0-9_\-/]+\.[a-zA-Z]{1,5}:\d+'  # File path with line number
    r'|^[-=─━═]{3,}\s*\w[\w\s]*\s*[-=─━═]{3,}$'   # Section divider banners
    r'|^\s+File ".*", line \d+'                # Python traceback file lines
    r'|^net/http\.\(',                         # Go net/http internal stack frames
    re.IGNORECASE)

# ── Cloud-storage safelist (Section 1D) ──────────────────────────────
# Lines matching any of these patterns are KEPT (is_noise = False) even
# if they would otherwise match RE_NOISE. Evaluated BEFORE RE_NOISE.
# Ordering requirement: safelist → binary check → PM2/PDF check → RE_NOISE.
_NOISE_SAFELIST = [
    re.compile(r'storage\.googleapis\.com', re.IGNORECASE),  # GCS
    re.compile(r's3\.amazonaws\.com',       re.IGNORECASE),  # AWS S3
    re.compile(r'blob\.core\.windows\.net', re.IGNORECASE),  # Azure Blob
    re.compile(r'\.minio\.',               re.IGNORECASE),   # MinIO
    re.compile(r'object.?storage',         re.IGNORECASE),   # Generic object storage (from Stage 2)
]

RE_TS_RANGE_END = re.compile(r'^(?P<start>.+?)(?:–|-)\s*(?P<end>\d{2})$')
RE_BURST        = re.compile(r'(?P<count>\d+)\s*\(\s*burst\s*\)', re.IGNORECASE)

# ── Severity inference helpers ────────────────────────────────────────
_SEV_QUOTED_VAL = re.compile(
    r'''(?:[:,]\s*['\"`])[^'\"`]*\b(?:error|fail|exception|critical|fatal|panic|crash|traceback|stderr|warn|warning|deprecated|timeout|retry|slow|disconnect|debug|trace|verbose)\b[^'\"`]*['\"`]''',
    re.I,
)
# Compiled once at module level — used in _infer_severity_safe to strip
# key-value pairs like `status: "error"` before keyword scanning, so that
# values in string literals don't pollute the severity signal.
_SEV_KV_STRIP = re.compile(r'''[\w\s]+:\s*['\"`][^'\"`]*['\"`],?''')
_SEV_KEYWORD_ERROR = re.compile(
    r'\b(error|fail|exception|critical|fatal|panic|crash|traceback|stderr)\b', re.I)
_SEV_KEYWORD_WARN  = re.compile(
    r'\b(warn|warning|deprecated|timeout|retry|slow|disconnect)\b', re.I)
_SEV_KEYWORD_DEBUG = re.compile(
    r'\b(debug|trace|verbose)\b', re.I)

# NOTE: _MASK_PATTERNS and _mask_message() removed (Section 1A).
# PII masking is now owned exclusively by Stage 2, which produces
# normalized_message. Stage 1 no longer generates a message_masked column.


# ══════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════

def _parse_pm2_ts(ts_str: str) -> Optional[datetime]:
    # PM2 v4 emits second-resolution:       2024-01-15T10:23:45
    # PM2 v5+ emits millisecond-resolution: 2024-01-15T10:23:45.123
    # RE_PM2_OUTER captures the optional fractional part; this parser
    # must handle both so the timestamp_parsed_ok flag is set correctly.
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(ts_str, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _pm2_bracket_service(tags_str: str) -> str:
    parts = re.findall(r'\[([^\]]+)\]', tags_str)
    svc = parts[-1] if parts else tags_str
    raw = svc.lower().strip().replace(" ", "-").replace("_", "-")
    # Fix 3: reject blocklisted / invalid tokens
    return _sanitise_service(raw) or "unknown"


# ── S1-1 FIX: PM2 plain-line service inference ────────────────────────
# Instead of hardcoding service='app' for every unbracketed PM2 line,
# we infer the service from the message content.
#
# Priority order:
#   1. Leading emoji label  (🧠 → daemon, 📄 → document-preview, etc.)
#   2. Path prefix          (/api/auth → auth, /api/documents → documents)
#   3. Keyword scan         (first matching keyword wins)
#   4. Fallback             → 'app'  (unchanged from before)
#
# To add project-specific entries call extend_pm2_service_map() before
# the first run_stage1() invocation.

_PM2_EMOJI_SERVICE_MAP: Dict[str, str] = {
    # Common emoji labels used in Node.js/PM2 apps
    "🧠": "daemon",
    "📄": "document-preview",
    "📁": "file-service",
    "🗄️": "database",
    "🔒": "auth",
    "🔑": "auth",
    "🌐": "network",
    "📧": "email",
    "📨": "email",
    "💾": "storage",
    "🚀": "server",
    "⚙️": "config",
    "🔧": "config",
    "📊": "metrics",
    "⚠️": "warn",
    "❌": "error",
    "✅": "status",
    "🔄": "scheduler",
    "⏱️": "scheduler",
    "🗑️": "cleanup",
}

# Keyword → service name.  Checked via substring scan of the lowercased
# message.  More-specific / longer keywords should come first so they
# shadow shorter generic ones.
_PM2_KEYWORD_SERVICE_MAP: list = [
    # Storage / GCS
    ("streaming document from gcs",         "gcs"),
    ("file not found in gcs",               "gcs"),
    ("error downloading from storage",      "gcs"),
    ("gcs",                                 "gcs"),
    ("google cloud storage",               "gcs"),
    ("storage emulator",                   "gcs"),
    # Embedder / ML
    ("dropping embedding table",           "embedder"),
    ("embedding",                          "embedder"),
    ("embed",                              "embedder"),
    # Daemon / background processing
    ("daemonclient",                       "daemon"),
    ("daemon",                             "daemon"),
    ("worker",                             "daemon"),
    ("job queue",                          "daemon"),
    ("background",                         "daemon"),
    # Auth / session
    ("localstrategy",                      "auth"),
    ("passport",                           "auth"),
    ("login",                              "auth"),
    ("logout",                             "auth"),
    ("session",                            "auth"),
    ("jwt",                                "auth"),
    ("oauth",                              "auth"),
    ("unauthorized",                       "auth"),
    # NOTE: The service name here is "auth" (the inferred service field value).
    # The Stage 2 domain name is also "auth".  This is a coincidence of naming
    # for PM2 services — do NOT interpret this as the service field equalling
    # the domain name.  For Java logs, the service field is "auth-service"
    # (see _JAVA_PACKAGE_TO_SERVICE) while the domain is still "auth".
    ("authentication",                     "auth"),
    # Database
    ("pg ",                                "database"),
    ("postgres",                           "database"),
    ("sequelize",                          "database"),
    ("mongoose",                           "database"),
    ("database",                           "database"),
    ("connection pool",                    "database"),
    ("db ",                                "database"),
    # Scheduler / cron
    ("cron",                               "scheduler"),
    ("scheduled",                          "scheduler"),
    ("heartbeat",                          "scheduler"),
    # Server / infrastructure
    ("server is running",                  "server"),
    ("listening on port",                  "server"),
    ("pm2",                                "server"),
    ("process manager",                    "server"),
    ("sigint",                             "server"),
    ("shutting down",                      "server"),
    # Document / content
    ("document preview",                   "document-preview"),
    ("serving docx",                       "document-preview"),
    ("document",                           "document-svc"),
    ("pdf",                                "document-svc"),
    # Version stack
    ("version-stack",                      "version-stack"),
    ("version stack",                      "version-stack"),
    ("outcome=stacked",                    "version-stack"),
    # Email / messaging
    ("email",                              "email"),
    ("smtp",                               "email"),
    ("notification",                       "email"),
    # Security
    ("rate limit",                         "security-filter"),
    ("blocked",                            "security-filter"),
    ("suspicious",                         "security-filter"),
    # Generic API — checked last because it's broad
    ("/api/",                              "api"),
]

# Path-prefix patterns: checked before keyword scan.
# Regex → service name.  First match wins.
_PM2_PATH_SERVICE_PATTERNS: list = [
    (re.compile(r'/api/auth\b',            re.I), "auth"),
    (re.compile(r'/api/user',              re.I), "auth"),
    (re.compile(r'/api/session',           re.I), "auth"),
    (re.compile(r'/api/document',          re.I), "document-svc"),
    (re.compile(r'/api/embed',             re.I), "embedder"),
    (re.compile(r'/api/storage',           re.I), "gcs"),
    (re.compile(r'/api/notification',      re.I), "email"),
    (re.compile(r'/api/scheduler',         re.I), "scheduler"),
]


def extend_pm2_service_map(
    emoji_map: Optional[Dict[str, str]] = None,
    keyword_pairs: Optional[list] = None,
) -> None:
    """
    Register additional PM2 plain-line service inference rules at runtime.
    Call this from your deployment config before the first run_stage1() call.

    Parameters
    ----------
    emoji_map     : dict mapping emoji prefix strings to service names.
    keyword_pairs : list of (keyword_substring, service_name) tuples.
                    Prepended so they take priority over the built-in list.

    Example::

        from stages.stage1 import extend_pm2_service_map
        extend_pm2_service_map(
            emoji_map={"🏭": "factory"},
            keyword_pairs=[("my-custom-module", "custom-svc")],
        )
    """
    if emoji_map:
        _PM2_EMOJI_SERVICE_MAP.update(emoji_map)
    if keyword_pairs:
        # Prepend so project-specific rules shadow built-ins
        _PM2_KEYWORD_SERVICE_MAP[:0] = keyword_pairs


def _infer_pm2_plain_service(inner_str: str) -> str:
    """
    Infer a service name for an unbracketed PM2 log line.

    Priority:
      1. Leading emoji label
      2. URL path prefix  (via _PM2_PATH_SERVICE_PATTERNS)
      3. Keyword substring scan  (via _PM2_KEYWORD_SERVICE_MAP)
      4. Fallback → 'app'
    """
    if not inner_str:
        return "app"

    # 1. Emoji prefix — check the first 1–3 characters
    for emoji, svc in _PM2_EMOJI_SERVICE_MAP.items():
        if inner_str.startswith(emoji):
            # Fix 3: emoji map entries like ⚠️→"warn" / ❌→"error" are invalid
            return _sanitise_service(svc) or "app"

    # 2. URL path prefix
    for pat, svc in _PM2_PATH_SERVICE_PATTERNS:
        if pat.search(inner_str):
            return _sanitise_service(svc) or "app"

    # 3. Keyword scan (case-insensitive via lowercased message)
    lower = inner_str.lower()
    for keyword, svc in _PM2_KEYWORD_SERVICE_MAP:
        if keyword in lower:
            # Fix 3: guard against blocklisted service names from keyword map
            return _sanitise_service(svc) or "app"

    # 4. Fallback
    return "app"


def _is_binary_noise(s: str) -> bool:
    if not s:
        return False
    if '\x00' in s:
        null_ratio = s.count('\x00') / len(s)
        if null_ratio > 0.3:
            return True
    if ord(s[0]) > 127:
        return True
    non_ascii = sum(1 for c in s if ord(c) > 127)
    if non_ascii / len(s) > 0.30:
        return True
    # Short-line guard: only reject if the line also contains a non-printable
    # character.  Pure-ASCII short lines (e.g. "OK", "0", "Yes", a 3-digit
    # status code) are valid log content and must not be discarded.
    if len(s) <= 40:
        for c in s:
            o = ord(c)
            if o > 127 or (o < 0x20 and o not in (0x09, 0x0a, 0x0d)):
                return True
    # Do NOT unconditionally reject 1–3 char lines — "OK", "0", "No"
    # are legitimate single-token log outputs.
    return False


def _pm2_inner_is_pdf_noise(inner: str) -> bool:
    if not inner:
        return False
    if RE_PM2_PDF_STRUCTURAL.match(inner):
        return True
    if RE_PM2_PDF_OP.match(inner):
        return True
    non_ascii = sum(1 for c in inner if ord(c) > 127)
    if non_ascii / len(inner) > 0.30:
        return True
    if non_ascii > 0:
        ctrl = sum(1 for c in inner if ord(c) < 0x20 and c not in '\t\n\r')
        if ctrl >= 2:
            return True
        if len(inner) <= 60 and ctrl >= 1:
            return True
    if len(inner) == 1 and inner.isalpha() and inner.isascii():
        return True
    if inner.rstrip().endswith('>>'):
        return True
    return False


def _infer_severity_from_text(text: str) -> str:
    if _SEV_KEYWORD_ERROR.search(text):  return "ERROR"
    if _SEV_KEYWORD_WARN.search(text):   return "WARN"
    if _SEV_KEYWORD_DEBUG.search(text):  return "DEBUG"
    return "INFO"


def _infer_severity_safe(text: str) -> str:
    cleaned = _SEV_QUOTED_VAL.sub('', text)
    cleaned = _SEV_KV_STRIP.sub('', cleaned)
    if _SEV_KEYWORD_ERROR.search(cleaned):  return "ERROR"
    if _SEV_KEYWORD_WARN.search(cleaned):   return "WARN"
    if _SEV_KEYWORD_DEBUG.search(cleaned):  return "DEBUG"
    return "INFO"


def _norm_severity(sev):
    if sev is None: return None
    s = sev.strip().upper()
    return SEV_MAP.get(s, s if s else None)


def _norm_service(svc):
    if svc is None: return None
    svc = svc.strip()
    return svc if svc else None


def _extract_burst_count(text):
    m = RE_BURST.search(text)
    if not m: return None, False
    try:    return int(m.group("count")), True
    except: return None, True


def _try_parse_json(line):
    t = line.lstrip()
    if not (t.startswith("{") and line.rstrip().endswith("}")): return None
    try:
        obj = json.loads(line)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _split_ts_range(raw):
    m = RE_TS_RANGE_END.match(raw)
    if not m: return raw, None, False
    return m.group("start").strip(), m.group("end").strip(), True


def _parse_iso_z(ts):
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _parse_naive_iso(ts, assumed_tz):
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            dt_naive = datetime.strptime(ts.strip(), fmt)
            return dt_naive.replace(tzinfo=assumed_tz).astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def _parse_syslog(ts, default_year, assumed_tz):
    try:
        dt = datetime.strptime(f"{default_year} {ts}", "%Y %b %d %H:%M:%S")
        return dt.replace(tzinfo=assumed_tz).astimezone(timezone.utc)
    except Exception:
        return None


def _parse_time_only(hms, default_date, assumed_tz):
    if default_date is None: return None, "time_only_no_default_date"
    try:
        t  = datetime.strptime(hms, "%H:%M:%S").time()
        return datetime.combine(default_date, t).replace(tzinfo=assumed_tz).astimezone(timezone.utc), None
    except Exception:
        return None, "time_only_parse_failed"


def _parse_apache(ts):
    try:    return datetime.strptime(ts, "%d/%b/%Y:%H:%M:%S %z").astimezone(timezone.utc)
    except: return None


def _parse_nginx_ts(ts, assumed_tz):
    try:
        dt = datetime.strptime(ts, "%Y/%m/%d %H:%M:%S")
        return dt.replace(tzinfo=assumed_tz).astimezone(timezone.utc)
    except Exception:
        return None


def _parse_iso_with_offset(ts):
    try:    return datetime.fromisoformat(ts).astimezone(timezone.utc)
    except: return None


def _parse_windows_evt_ts(ts, assumed_tz):
    try:
        dt = datetime.strptime(ts, "%m/%d/%Y %I:%M:%S %p")
        return dt.replace(tzinfo=assumed_tz).astimezone(timezone.utc)
    except Exception:
        return None


def _parse_postgres_ts(ts, assumed_tz):
    try:
        dt = datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _parse_java_ts(ts, assumed_tz):
    try:
        dt = datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=assumed_tz).astimezone(timezone.utc)
    except Exception:
        return None


def _parse_audit_epoch(epoch_str):
    try:    return datetime.fromtimestamp(float(epoch_str), tz=timezone.utc)
    except: return None


def _norm_windows_provider(provider: str) -> str:
    if not provider:
        return provider
    key = provider.strip().lower()
    if key in _WINDOWS_PROVIDER_NORM:
        return _WINDOWS_PROVIDER_NORM[key]
    if key.startswith("microsoft-windows-"):
        suffix = key[len("microsoft-windows-"):]
        short  = suffix.split("-")[0] if "-" in suffix else suffix
        return f"windows-{short}"
    if key.startswith("microsoft-"):
        suffix = key[len("microsoft-"):]
        return f"ms-{suffix.split('-')[0]}"
    return provider.strip()


def _extract_java_service(cls: str) -> Optional[str]:
    if not cls:
        return None
    parts = cls.split(".")
    for part in reversed(parts[:-1]):
        mapped = _JAVA_PACKAGE_TO_SERVICE.get(part.lower())
        if mapped:
            return mapped
    # Use the last package component as the service name, but skip it if it
    # is a generic framework class fragment that would produce a misleading
    # label like "runtime", "context", "handler", etc.
    last = parts[-1].lower() if parts else ""
    if last and last not in _JAVA_GENERIC_CLASS_FRAGMENTS:
        return last
    # If last part is generic, try the second-to-last package component
    if len(parts) >= 2:
        parent = parts[-2].lower()
        if parent not in _JAVA_GENERIC_CLASS_FRAGMENTS:
            return parent
    return parts[-1].lower() if parts else None


def _cef_severity(n):
    try:
        v = int(n)
        if v >= 8: return "ERROR"
        if v >= 5: return "WARN"
        return "INFO"
    except Exception:
        return "INFO"


def _parse_timestamp(ts_raw, fmt, *, assumed_tz, default_year, default_date):
    if ts_raw is None:
        return None, None, False, "missing_timestamp", False
    start_raw, end_suffix, is_range = _split_ts_range(ts_raw)
    start_dt = None; end_dt = None; err = None
    if fmt == "f1_syslog":
        start_dt = _parse_syslog(start_raw, default_year, assumed_tz)
        err = None if start_dt else "syslog_ts_parse_failed"
    elif fmt in {"f2_iso_space_bracket", "f7_iso_pipe"}:
        start_dt = _parse_naive_iso(start_raw, assumed_tz)
        err = None if start_dt else "iso_naive_ts_parse_failed"
    elif fmt in {"f3_rfc3339z", "f5_json"}:
        start_dt = _parse_iso_z(start_raw)
        err = None if start_dt else "iso_z_ts_parse_failed"
    elif fmt == "f4_bracket_short":
        start_dt, err = _parse_time_only(start_raw, default_date, assumed_tz)
    elif fmt == "f6_apache_pipe":
        start_dt = _parse_apache(start_raw)
        err = None if start_dt else "apache_ts_parse_failed"
    elif fmt == "f8_nginx_err":
        start_dt = _parse_nginx_ts(start_raw, assumed_tz)
        err = None if start_dt else "nginx_ts_parse_failed"
    elif fmt == "f9_access":
        start_dt = _parse_apache(start_raw)
        err = None if start_dt else "access_ts_parse_failed"
    elif fmt == "f10_postgres":
        start_dt = _parse_postgres_ts(start_raw, assumed_tz)
        err = None if start_dt else "postgres_ts_parse_failed"
    elif fmt == "f11_java":
        start_dt = _parse_java_ts(start_raw, assumed_tz)
        err = None if start_dt else "java_ts_parse_failed"
    elif fmt == "f16_systemd":
        start_dt = _parse_iso_with_offset(start_raw)
        err = None if start_dt else "systemd_ts_parse_failed"
    elif fmt == "f15_windows_evt":
        start_dt = _parse_windows_evt_ts(start_raw, assumed_tz)
        err = None if start_dt else "windows_evt_ts_parse_failed"
    elif fmt == "f13_audit":
        start_dt = _parse_audit_epoch(start_raw)
        err = None if start_dt else "audit_epoch_parse_failed"
    elif fmt in {"f12_cef", "f14_k8s_event"}:
        err = "no_timestamp_in_format"
    elif fmt in {"f17a_pm2_express", "f17b_pm2_bracket",
                 "f17c_pm2_plain", "f17d_pm2_continuation"}:
        start_dt = _parse_pm2_ts(start_raw)
        err = None if start_dt else "pm2_ts_parse_failed"
    else:
        ts = pd.to_datetime(start_raw, utc=True, errors="coerce")
        if pd.isna(ts): err = "unknown_ts_parse_failed"
        else:           start_dt = ts.to_pydatetime()
    if is_range and start_dt is not None and end_suffix is not None:
        try:
            end_sec = int(end_suffix)
            end_dt  = start_dt.replace(second=end_sec)
            if end_dt < start_dt:
                # Minute (or hour) rolled over — use timedelta so the carry
                # propagates correctly through the full hour boundary.
                # e.g. start=12:59:45, end_sec=12 → 13:00:12, not 12:00:12.
                end_dt = end_dt + timedelta(minutes=1)
        except Exception:
            err = err or "range_end_parse_failed"
    ok = start_dt is not None
    return start_dt, end_dt, ok, err, is_range


def _finalise_pm2(out: dict, assumed_tz, default_year, default_date) -> dict:
    start_dt, end_dt, ok, terr, is_range = _parse_timestamp(
        out["timestamp_raw"], fmt=out["format_type"],
        assumed_tz=assumed_tz, default_year=default_year, default_date=default_date,
    )
    out["timestamp_parsed"]      = start_dt
    out["timestamp_end_parsed"]  = end_dt
    out["timestamp_parsed_ok"]   = ok
    out["timestamp_error_reason"] = terr
    out["timestamp_is_range"]    = is_range
    if out.get("message") is not None:
        bc, bf = _extract_burst_count(out["message"])
        out["burst_count"] = bc
        out["burst_flag"]  = bf
    return out


# ══════════════════════════════════════════════════════════════════════
# ParseStats
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ParseStats:
    total_lines:      int = 0
    parsed_ok:        int = 0
    unknown:          int = 0
    noise:            int = 0
    json_ok:          int = 0
    ts_parsed_ok:     int = 0
    ts_failed:        int = 0
    burst_lines:      int = 0
    parse_failed:     int = 0
    quarantined:      int = 0
    detected_encoding: str = "utf-8"
    default_date_inferred: bool = True   # False means we fell back to today
    format_probe:     Dict[str, float] = field(default_factory=dict)
    format_counts:    Counter = field(default_factory=Counter)
    error_reasons:    Counter = field(default_factory=Counter)

    def as_dict(self):
        return {
            "total_lines":          self.total_lines,
            "parsed_ok":            self.parsed_ok,
            "unknown":              self.unknown,
            "noise":                self.noise,
            "json_ok":              self.json_ok,
            "ts_parsed_ok":         self.ts_parsed_ok,
            "ts_failed":            self.ts_failed,
            "burst_lines":          self.burst_lines,
            "lines_parsed_ok":      self.parsed_ok,
            "lines_noise_stripped": self.noise,
            "lines_parse_failed":   self.parse_failed,
            "lines_quarantined":    self.quarantined,
            "detected_encoding":    self.detected_encoding,
            "default_date_inferred": self.default_date_inferred,
            "format_probe":         self.format_probe,
            "format_counts":        dict(self.format_counts),
            "top_error_reasons":    dict(self.error_reasons.most_common(10)),
        }


# ══════════════════════════════════════════════════════════════════════
# BP-D: Encoding auto-detection
# ══════════════════════════════════════════════════════════════════════

def _detect_encoding(path: Path, fallback: str = "utf-8") -> str:
    raw_sample: Optional[bytes] = None
    try:
        with path.open("rb") as fh:
            raw_sample = fh.read(65536)
    except Exception:
        return fallback
    try:
        import chardet
        result = chardet.detect(raw_sample)
        if result and result.get("encoding") and result.get("confidence", 0) >= 0.7:
            return result["encoding"]
    except ImportError:
        pass
    try:
        from charset_normalizer import from_bytes
        result = from_bytes(raw_sample).best()
        if result is not None:
            return str(result.encoding)
    except ImportError:
        pass

    # S3.4 — BOM check before surrendering to the fallback.
    # Neither chardet nor charset_normalizer is installed (or both failed),
    # so check for the three most common byte-order marks directly.  This
    # catches UTF-16 and UTF-8-BOM files without any third-party library.
    # Cost: 3 byte-prefix comparisons — effectively free.
    if raw_sample[:2] == b'\xff\xfe':
        return "utf-16-le"
    if raw_sample[:2] == b'\xfe\xff':
        return "utf-16-be"
    if raw_sample[:3] == b'\xef\xbb\xbf':
        return "utf-8-sig"

    # Warn if we are falling back without confirmation — any non-ASCII
    # content in the file will be silently corrupted by errors="replace".
    import logging as _logging
    _logging.getLogger("stage1_parser").warning(
        "_detect_encoding: could not confirm encoding for %s — "
        "falling back to '%s'.  Non-ASCII characters may be corrupted.  "
        "Install 'chardet' or 'charset-normalizer' to fix this.",
        getattr(raw_sample, '__class__', ''), fallback,
    )
    return fallback


# ══════════════════════════════════════════════════════════════════════
# BP-E: Format probe
# ══════════════════════════════════════════════════════════════════════

_PROBE_PATTERNS = [
    ("json",            re.compile(r'^\s*\{')),
    ("pm2",             re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}:')),
    ("iso8601_rfc3339", re.compile(r'^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s+(?:TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL)', re.I)),
    ("iso_space",       re.compile(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}.*(?:TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL)', re.I)),
    ("syslog",          re.compile(r'^[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\S+\s+\S+(?:\[\d+\])?:')),
    ("nginx_error",     re.compile(r'^\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\s+\[\w+\]')),
    ("access_log",      re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\s+-\s+')),
    ("java",            re.compile(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}\s+(?:TRACE|DEBUG|INFO|WARN|ERROR|FATAL)', re.I)),
    ("postgres",        re.compile(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+\s+UTC')),
    ("cef",             re.compile(r'^CEF:\d+')),
    ("audit",           re.compile(r'^type=\w+\s+msg=audit\(')),
    ("windows_evt",     re.compile(r'^\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}\s+(?:AM|PM)\s+EventID=')),
    ("systemd",         re.compile(r'^\d{4}-\d{2}-\d{2}T[\d:.]+[+-]\d{2}:\d{2}\s+host=')),
    ("k8s_event",       re.compile(r'^\d+[smh].*(?:Normal|Warning)\s+\S+\s+\S+/\S+')),
]


def _probe_format(path: Path, *, encoding: str, errors: str,
                  probe_lines: int = 200) -> Dict[str, float]:
    counts: Counter = Counter()
    total = 0
    try:
        with path.open("r", encoding=encoding, errors=errors) as fh:
            for raw in fh:
                if total >= probe_lines:
                    break
                s = raw.rstrip("\n")
                if not s.strip() or RE_NOISE.match(s):
                    continue
                total += 1
                for name, pat in _PROBE_PATTERNS:
                    if pat.match(s):
                        counts[name] += 1
                        break
    except Exception:
        return {}
    if total == 0:
        return {}
    return {name: round(cnt / total, 3) for name, cnt in counts.most_common()}


def _infer_default_date_from_file(path, *, encoding, errors, assumed_tz, default_year, max_lines=500):
    try:
        with path.open("r", encoding=encoding, errors=errors) as fh:
            for i, raw in enumerate(fh, start=1):
                if i > max_lines:
                    break
                s = raw.rstrip("\n")
                if not s.strip() or RE_NOISE.match(s):
                    continue
                if s.startswith("[") and RE_BRACKET_SHORT_F4.match(s):
                    continue
                for pat, fmt in _DATEABLE_PATTERNS:
                    m = pat.match(s)
                    if not m:
                        continue
                    ts_raw = m.group("ts")
                    dt     = None
                    if fmt == "f3":   dt = _parse_iso_z(ts_raw)
                    elif fmt == "f2": dt = _parse_naive_iso(ts_raw, assumed_tz)
                    elif fmt == "f1": dt = _parse_syslog(ts_raw, default_year, assumed_tz)
                    elif fmt == "f8": dt = _parse_nginx_ts(ts_raw, assumed_tz)
                    elif fmt == "f16":dt = _parse_iso_with_offset(ts_raw)
                    elif fmt == "f17":dt = _parse_pm2_ts(ts_raw)
                    if dt is not None:
                        return dt.astimezone(assumed_tz).date()
                    break
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════
# parse_line — core per-line parser
# ══════════════════════════════════════════════════════════════════════

def parse_line(
    raw_line,
    line_no,
    *,
    assumed_tz,
    default_year,
    default_date,
    allow_relaxed=True,
    last_error_sev: str = "INFO",
    last_pm2_service: str = "app",
):
    s   = raw_line.rstrip("\n")

    # S3.3 — Max line length guard.
    # A single megabyte-long line (e.g. a serialised stack trace or base64
    # blob) would cause every compiled regex below to run against a massive
    # string, stalling the parser for seconds per line.  Truncate to 8000
    # chars — enough for any realistic log line — and record the reason.
    # The truncated line is still parsed normally; nothing is discarded.
    _line_was_truncated = False
    if len(s) > 8000:
        s = s[:8000]
        _line_was_truncated = True

    out: Dict[str, Any] = {
        "line_no":               int(line_no),
        "raw_line":              s,
        "format_type":           None,
        "parsed_ok":             False,
        "parse_confidence":      "low",
        "parse_error_reason":    None,
        "timestamp_raw":         None,
        "timestamp_parsed":      None,
        "timestamp_end_parsed":  None,
        "timestamp_parsed_ok":   False,
        "timestamp_error_reason":None,
        "timestamp_is_range":    False,
        "service":               None,
        "service_raw":           None,
        "severity":              None,
        "severity_raw":          None,
        "host":                  None,
        "pid":                   None,
        "message":               None,
        "is_noise_candidate":    False,
        "noise_reason":          None,
        "burst_count":           None,
        "burst_flag":            False,
        "cluster_id":            pd.NA,
        "format_tag":            None,
        "is_continuation":       False,
        "repeat_count":          1,
    }

    # S3.3 continued — record truncation in the output record now that
    # the out dict exists.
    if _line_was_truncated:
        out["parse_error_reason"] = "line_truncated"

    # S1-1D: Cloud-storage safelist — lines matching these are KEPT (is_noise = False)
    # even if they would otherwise match RE_NOISE. Must be checked first.
    _safelist_match = any(pat.search(s) for pat in _NOISE_SAFELIST)

    # Binary noise guard
    if not _safelist_match and _is_binary_noise(s):
        out.update({
            "format_type":        "noise",
            "format_tag":         "noise",
            "is_noise_candidate": True,
            "noise_reason":       "binary_noise",
            "parse_error_reason": "noise_candidate",
            "message":            s,
        })
        return out

    if not _safelist_match and RE_NOISE.match(s):
        out.update({
            "format_type":        "noise",
            "is_noise_candidate": True,
            "noise_reason":       "noise_regex_match",
            "parse_error_reason": "noise_candidate",
            "message":            s,
            "format_tag":         "noise",
        })
        return out

    # F17: PM2 outer envelope
    pm2_m = RE_PM2_OUTER.match(s)
    if pm2_m:
        pm2_ts    = pm2_m.group("ts")
        inner     = pm2_m.group("inner")
        inner_str = inner.strip() if inner else ""

        if not inner_str:
            out.update({
                "format_type":        "noise",
                "format_tag":         "noise",
                "is_noise_candidate": True,
                "noise_reason":       "pm2_empty_inner",
                "parse_error_reason": "noise_candidate",
                "message":            s,
            })
            return out

        if _pm2_inner_is_pdf_noise(inner_str):
            out.update({
                "format_type":        "noise",
                "format_tag":         "noise",
                "is_noise_candidate": True,
                "noise_reason":       "pm2_pdf_blob",
                "parse_error_reason": "noise_candidate",
                "message":            inner_str,
            })
            return out

        if RE_PM2_SYMBOL_NOISE.match(inner_str):
            out.update({
                "format_type":        "noise",
                "format_tag":         "noise",
                "is_noise_candidate": True,
                "noise_reason":       "pm2_symbol_noise",
                "parse_error_reason": "noise_candidate",
                "message":            inner_str,
            })
            return out

        # Sub-type A: Express HTTP
        em = RE_PM2_EXPRESS_INNER.match(inner_str)
        if em:
            status = em.group("status") or ""
            sev    = ("ERROR" if status.startswith("5")
                      else "WARN" if status.startswith("4")
                      else "INFO")
            method = em.group("method") or ""
            path   = em.group("path")   or ""
            ms     = em.group("ms")     or ""
            truncated = bool(re.search(r'[\u2026\.]{1,}$', inner_str) and not ms)
            msg    = f"{method} {path} {status}"
            if ms:
                msg += f" in {ms}ms"
            elif truncated:
                msg += " (response truncated)"
            # Fix 2: apply post-processing override (handles log-archive false ERROR, etc.)
            sev = _apply_http_severity_override(msg, sev, "f17a_pm2_express")
            out.update({
                "format_type":     "f17a_pm2_express",
                "format_tag":      "pm2_express",
                "parsed_ok":       True,
                "parse_confidence":"high",
                "timestamp_raw":   pm2_ts,
                "severity_raw":    status,
                "severity":        sev,
                "service_raw":     "express",
                "service":         "express",
                "message":         msg.strip(),
            })
            return _finalise_pm2(out, assumed_tz, default_year, default_date)

        # Sub-type B: Bracketed service
        bm = RE_PM2_BRACKET_INNER.match(inner_str)
        if bm:
            tags_str = bm.group("tags")
            msg_body = bm.group("msg").strip()
            svc      = _pm2_bracket_service(tags_str)
            sev = _infer_severity_safe(msg_body or inner_str)
            # Fix 2: apply post-processing HTTP severity override
            sev = _apply_http_severity_override(msg_body or inner_str, sev, "f17b_pm2_bracket")
            out.update({
                "format_type":     "f17b_pm2_bracket",
                "format_tag":      "pm2_bracket",
                "parsed_ok":       True,
                "parse_confidence":"high",
                "timestamp_raw":   pm2_ts,
                "severity_raw":    None,
                "severity":        sev,
                "service_raw":     tags_str,
                "service":         svc,
                "message":         msg_body if msg_body else inner_str,
            })
            return _finalise_pm2(out, assumed_tz, default_year, default_date)

        # Sub-type D: Continuation lines
        is_stack_at = bool(RE_PM2_STACK_AT.match(inner_str))
        is_structural = bool(RE_PM2_STRUCTURAL_TOKEN.match(inner_str))
        is_obj_prop = (
            (
                bool(RE_PM2_OBJ_PROP.match(inner_str))
                and inner_str[0] in (' ', '\t')
                and not re.match(r'^\s*\d{2}:\d{2}:\d{2}', inner_str)
            )
            or bool(RE_PM2_OBJ_PROP_UNINDENTED.match(inner_str))
        )
        if is_stack_at or is_structural or is_obj_prop:
            msg = inner_str.strip()
            out.update({
                "format_type":     "f17d_pm2_continuation",
                "format_tag":      "pm2_continuation",
                "parsed_ok":       True,
                "parse_confidence":"medium",
                "timestamp_raw":   pm2_ts,
                "severity_raw":    None,
                "severity":        last_error_sev,
                "service_raw":     last_pm2_service,
                "service":         last_pm2_service,
                "message":         msg,
                "is_continuation": True,
            })
            return _finalise_pm2(out, assumed_tz, default_year, default_date)

        # Sub-type C: Plain app log
        sev = _infer_severity_safe(inner_str)
        # Fix 2: apply post-processing HTTP severity override
        sev = _apply_http_severity_override(inner_str, sev, "f17c_pm2_plain")
        # S1-1 FIX: infer service from message content instead of hardcoding 'app'.
        inferred_svc = _infer_pm2_plain_service(inner_str)
        out.update({
            "format_type":     "f17c_pm2_plain",
            "format_tag":      "pm2_plain",
            "parsed_ok":       True,
            "parse_confidence":"high",
            "timestamp_raw":   pm2_ts,
            "severity_raw":    None,
            "severity":        sev,
            "service_raw":     inferred_svc,
            "service":         inferred_svc,
            "message":         inner_str,
        })
        return _finalise_pm2(out, assumed_tz, default_year, default_date)

    # JSON
    obj = _try_parse_json(s)
    if obj is not None:
        # Timestamp: try common field names across major JSON logging libraries.
        # Bunyan/Pino use "time", Logrus uses "time", Datadog uses "date",
        # Loki uses "t", ECS uses "@timestamp".
        ts_raw = (
            obj.get("ts")
            or obj.get("timestamp")
            or obj.get("time")
            or obj.get("@timestamp")
            or obj.get("date")
            or obj.get("t")
        )
        # Severity: "level" (Winston/Bunyan/Pino), "severity" (GCP/Stackdriver),
        # "status" (Datadog), "lvl" (zerolog)
        sev_raw = (
            obj.get("level")
            or obj.get("severity")
            or obj.get("status")
            or obj.get("lvl")
            or obj.get("log_level")
        )
        # Service: "svc", "service" (standard), "name" (Bunyan),
        # "logger" (Logback/log4j), "app" (some custom loggers)
        svc_raw = (
            obj.get("svc")
            or obj.get("service")
            or obj.get("name")
            or obj.get("logger")
            or obj.get("app")
        )
        # Message: "msg" (Bunyan/Pino/Logrus), "message" (Winston/standard),
        # "log" (some Docker/k8s shippers), "text" (some structured loggers)
        msg = (
            obj.get("msg")
            or obj.get("message")
            or obj.get("log")
            or obj.get("text")
            or s
        )
        out.update({
            "format_type":    "f5_json",
            "format_tag":     "json",
            "parsed_ok":      True,
            "parse_confidence":"high",
            "timestamp_raw":  str(ts_raw)  if ts_raw  is not None else None,
            "severity_raw":   str(sev_raw) if sev_raw is not None else None,
            "severity":       _norm_severity(str(sev_raw)) if sev_raw is not None else None,
            "service_raw":    str(svc_raw) if svc_raw is not None else None,
            "service":        _norm_service(str(svc_raw)) if svc_raw is not None else None,
            "message":        str(msg),
        })
    else:
        first1 = s[:1]
        if first1 == "[":
            m = RE_BRACKET_SHORT_F4.match(s)
            if m:
                sevabbr = m.group("sevabbr").strip().upper()
                out.update({
                    "format_type":     "f4_bracket_short",
                    "format_tag":      "bracket_short",
                    "parsed_ok":       True,
                    "parse_confidence":"high",
                    "timestamp_raw":   m.group("time"),
                    "severity_raw":    sevabbr,
                    "severity":        SEV_ABBR_MAP.get(sevabbr),
                    "service_raw":     m.group("svc"),
                    "service":         _norm_service(m.group("svc")),
                    "message":         m.group("msg"),
                })

        elif RE_YEAR_START.match(s):
            for fmt, rx, tag in (
                ("f3_rfc3339z",          RE_RFC3339Z_COLON_F3,    "rfc3339z"),
                ("f2_iso_space_bracket", RE_ISO_SPACE_BRACKET_F2, "iso_bracket"),
                ("f7_iso_pipe",          RE_ISO_PIPE_F7,          "iso_pipe"),
            ):
                m = rx.match(s)
                if m:
                    out.update({
                        "format_type":     fmt,
                        "format_tag":      tag,
                        "parsed_ok":       True,
                        "parse_confidence":"high",
                        "timestamp_raw":   m.group("ts"),
                        "severity_raw":    m.group("sev"),
                        "severity":        _norm_severity(m.group("sev")),
                        "service_raw":     m.group("svc"),
                        "service":         _norm_service(m.group("svc")),
                        "message":         m.group("msg"),
                    })
                    break

            if not out["parsed_ok"] and "/" in s:
                m = RE_NGINX_ERR_F8.match(s)
                if m:
                    msg_body = (m.group("msg") or "").strip()
                    request  = m.group("request") or ""
                    msg_full = f"{msg_body} [{request}]" if request else msg_body
                    out.update({
                        "format_type":     "f8_nginx_err",
                        "format_tag":      "nginx_error",
                        "parsed_ok":       True,
                        "parse_confidence":"high",
                        "timestamp_raw":   m.group("ts"),
                        "severity_raw":    m.group("sev"),
                        "severity":        _norm_severity(m.group("sev")),
                        "service_raw":     "nginx",
                        "service":         "nginx",
                        "host":            m.group("server"),
                        "pid":             m.group("pid"),
                        "message":         msg_full,
                    })

            if not out["parsed_ok"] and "UTC" in s:
                m = RE_POSTGRES_F10.match(s)
                if m:
                    out.update({
                        "format_type":     "f10_postgres",
                        "format_tag":      "postgres",
                        "parsed_ok":       True,
                        "parse_confidence":"high",
                        "timestamp_raw":   m.group("ts"),
                        "severity_raw":    m.group("sev"),
                        "severity":        _norm_severity(m.group("sev")),
                        "service_raw":     m.group("svc"),
                        "service":         _norm_service(m.group("svc")),
                        "pid":             m.group("pid"),
                        "message":         m.group("msg"),
                    })

            if not out["parsed_ok"] and "," in s[:24]:
                m = RE_JAVA_F11.match(s)
                if m:
                    cls      = m.group("class") or ""
                    svc_hint = _extract_java_service(cls)
                    out.update({
                        "format_type":     "f11_java",
                        "format_tag":      "java",
                        "parsed_ok":       True,
                        "parse_confidence":"high",
                        "timestamp_raw":   m.group("ts"),
                        "severity_raw":    m.group("sev"),
                        "severity":        _norm_severity(m.group("sev")),
                        "service_raw":     svc_hint,
                        "service":         _norm_service(svc_hint),
                        "message":         m.group("msg"),
                    })

            if not out["parsed_ok"] and "host=" in s:
                m = RE_SYSTEMD_F16.match(s)
                if m:
                    priority = m.group("priority") or "6"
                    sev      = _SYSTEMD_PRIORITY_MAP.get(priority, "INFO")
                    unit     = m.group("unit") or ""
                    svc      = unit.replace(".service", "").replace(".socket", "") if unit else None
                    out.update({
                        "format_type":     "f16_systemd",
                        "format_tag":      "heuristic_systemd",
                        "parsed_ok":       True,
                        "parse_confidence":"high",
                        "timestamp_raw":   m.group("ts"),
                        "severity_raw":    priority,
                        "severity":        sev,
                        "service_raw":     unit,
                        "service":         _norm_service(svc),
                        "host":            m.group("host"),
                        "message":         m.group("msg"),
                    })

        elif RE_IP_START.match(s):
            if " - " in s:
                m = RE_ACCESS_F9.match(s)
                if m:
                    status = m.group("status") or ""
                    sev    = ("ERROR" if status.startswith("5")
                              else "WARN" if status.startswith("4")
                              else "INFO")
                    method = m.group("method") or ""
                    path   = m.group("path")   or ""
                    bytes_ = m.group("bytes")  or ""
                    rt     = m.group("rt")
                    msg    = f"{method} {path} {status}"
                    if bytes_: msg += f" {bytes_}b"
                    if rt:     msg += f" rt={rt}s"
                    user = m.group("user") or "-"
                    svc  = _norm_service(user) if user != "-" else "http"
                    out.update({
                        "format_type":     "f9_access",
                        "format_tag":      "heuristic_access",
                        "parsed_ok":       True,
                        "parse_confidence":"high",
                        "timestamp_raw":   m.group("ts"),
                        "severity_raw":    status,
                        "severity":        sev,
                        "service_raw":     user,
                        "service":         svc,
                        "host":            m.group("client"),
                        "message":         msg.strip(),
                    })

        elif RE_DATE_START.match(s):
            if "EventID=" in s:
                m = RE_WINDOWS_EVT_F15.match(s)
                if m:
                    provider_raw = m.group("provider")
                    svc_norm     = _norm_windows_provider(provider_raw)
                    out.update({
                        "format_type":     "f15_windows_evt",
                        "format_tag":      "heuristic_windows_evt",
                        "parsed_ok":       True,
                        "parse_confidence":"high",
                        "timestamp_raw":   m.group("ts"),
                        "severity_raw":    m.group("sev"),
                        "severity":        _norm_severity(m.group("sev")),
                        "service_raw":     provider_raw,
                        "service":         svc_norm,
                        "message":         m.group("msg"),
                    })
            if not out["parsed_ok"] and "|" in s:
                m = RE_APACHE_PIPE_F6.match(s)
                if m:
                    out.update({
                        "format_type":     "f6_apache_pipe",
                        "format_tag":      "apache_pipe",
                        "parsed_ok":       True,
                        "parse_confidence":"high",
                        "timestamp_raw":   m.group("ts"),
                        "severity_raw":    m.group("sev"),
                        "severity":        _norm_severity(m.group("sev")),
                        "service_raw":     m.group("svc"),
                        "service":         _norm_service(m.group("svc")),
                        "message":         m.group("msg"),
                    })

        elif first1.isalpha():
            m = RE_SYSLOG_F1.match(s)
            if m:
                sev_raw = m.group("sev")
                sev     = _norm_severity(sev_raw)
                msg     = m.group("msg")
                if sev is None and m.group("svc") and "haproxy" in m.group("svc").lower():
                    sm = RE_HAPROXY_STATUS.search(msg or "")
                    if sm:
                        status  = sm.group(1)
                        sev_raw = status
                        sev     = ("ERROR" if status.startswith("5")
                                   else "WARN" if status.startswith("4")
                                   else "INFO")
                out.update({
                    "format_type":     "f1_syslog",
                    "format_tag":      "syslog",
                    "parsed_ok":       True,
                    "parse_confidence":"high",
                    "timestamp_raw":   m.group("ts"),
                    "severity_raw":    sev_raw,
                    "severity":        sev,
                    "service_raw":     m.group("svc"),
                    "service":         _norm_service(m.group("svc")),
                    "host":            m.group("host"),
                    "pid":             m.group("pid"),
                    "message":         msg,
                })

            if not out["parsed_ok"] and s.startswith("CEF:"):
                m = RE_CEF_F12.match(s)
                if m:
                    ext       = m.group("ext") or ""
                    msg_match = re.search(r'\bmsg=(.+?)(?:\s+\w+=|$)', ext)
                    msg       = msg_match.group(1).strip() if msg_match else m.group("event_name")
                    out.update({
                        "format_type":     "f12_cef",
                        "format_tag":      "heuristic_cef",
                        "parsed_ok":       True,
                        "parse_confidence":"high",
                        "timestamp_raw":   None,
                        "severity_raw":    m.group("severity"),
                        "severity":        _cef_severity(m.group("severity")),
                        "service_raw":     m.group("product"),
                        "service":         _norm_service(m.group("product")),
                        "message":         msg,
                    })

            if not out["parsed_ok"] and s.startswith("type="):
                m = RE_AUDIT_F13.match(s)
                if m:
                    fields     = m.group("fields") or ""
                    comm_match = re.search(r'\bcomm="?([^"\s]+)"?', fields)
                    svc        = comm_match.group(1) if comm_match else "audit"
                    msg        = f"audit {m.group('audit_type')}: {fields}"
                    out.update({
                        "format_type":     "f13_audit",
                        "format_tag":      "heuristic_audit",
                        "parsed_ok":       True,
                        "parse_confidence":"high",
                        "timestamp_raw":   m.group("epoch"),
                        "severity_raw":    "INFO",
                        "severity":        "INFO",
                        "service_raw":     svc,
                        "service":         _norm_service(svc),
                        "message":         msg,
                    })

        else:
            m = RE_K8S_EVT_F14.match(s)
            if m:
                etype = m.group("etype") or "Normal"
                sev   = "WARN" if etype == "Warning" else "INFO"
                obj_  = m.group("object") or ""
                svc   = obj_.split("/")[0] if "/" in obj_ else obj_
                out.update({
                    "format_type":     "f14_k8s_event",
                    "format_tag":      "heuristic_k8s",
                    "parsed_ok":       True,
                    "parse_confidence":"high",
                    "timestamp_raw":   None,
                    "severity_raw":    etype,
                    "severity":        sev,
                    "service_raw":     svc,
                    "service":         _norm_service(svc),
                    "message":         m.group("msg"),
                })

        if not out["parsed_ok"] and allow_relaxed:
            ts_match  = re.search(r'(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)', s)
            sev_match = re.search(r'\b(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL)\b', s, re.IGNORECASE)
            # Prefer hyphenated service names (e.g. db-primary, auth-service)
            svc_match = re.search(r'\b([a-z][a-z0-9]+-[a-z][a-z0-9-]+)\b', s)
            if not svc_match:
                # Then CamelCase class/service names
                svc_match = re.search(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', s)
            if not svc_match:
                # Original broad fallback last
                svc_match = re.search(r'\b([a-zA-Z][\w\-.]{2,})\b', s)
            ts_cand   = ts_match.group(1)  if ts_match  else None
            sev_cand  = sev_match.group(1) if sev_match else None
            svc_cand  = svc_match.group(1) if svc_match else None
            # Fix 3: sanitise relaxed-parser service candidates
            if svc_cand:
                svc_cand = _sanitise_service(svc_cand) or None
            if sum(x is not None for x in (ts_cand, sev_cand, svc_cand)) >= 2:
                out.update({
                    "format_type":        "relaxed_partial",
                    "format_tag":         "relaxed",
                    "parsed_ok":          True,
                    "parse_confidence":   "medium",
                    "timestamp_raw":      ts_cand,
                    "severity_raw":       sev_cand,
                    "severity":           _norm_severity(sev_cand),
                    "service_raw":        svc_cand,
                    "service":            _norm_service(svc_cand),
                    "message":            s,
                    "parse_error_reason": "relaxed_parse_used",
                })
            else:
                out.update({
                    "format_type":        "unknown",
                    "format_tag":         "unknown",
                    "parse_error_reason": "unmatched_known_formats",
                    "message":            s,
                })

    if out["message"] is not None:
        burst_count, burst_flag = _extract_burst_count(out["message"])
        out["burst_count"] = burst_count
        out["burst_flag"]  = burst_flag

    if out["parsed_ok"]:
        start_dt, end_dt, ok, terr, is_range = _parse_timestamp(
            out["timestamp_raw"], fmt=str(out["format_type"]),
            assumed_tz=assumed_tz, default_year=default_year, default_date=default_date,
        )
        out["timestamp_parsed"]     = start_dt
        out["timestamp_end_parsed"] = end_dt
        out["timestamp_parsed_ok"]  = ok
        out["timestamp_error_reason"] = terr
        out["timestamp_is_range"]   = is_range
        if not ok and out["parse_error_reason"] is None:
            if terr not in ("no_timestamp_in_format", "missing_timestamp"):
                out["parse_error_reason"] = "timestamp_parse_failed"

    # Fix 6: Domain vs Service collision check (non-blocking, DEBUG only).
    # Only fires for successfully parsed records with a non-None service value.
    # Noise lines, unknown lines, and continuation lines are intentionally
    # excluded — they carry no reliable service value.
    if out.get("parsed_ok") and out.get("service"):
        _check_service_domain_collision(out["service"], int(line_no))

    return out


# ══════════════════════════════════════════════════════════════════════
# _finalize_chunk
# ══════════════════════════════════════════════════════════════════════

def _finalize_chunk(df):
    string_cols = [
        "raw_line", "format_type", "format_tag", "parse_confidence",
        "parse_error_reason", "timestamp_raw", "timestamp_error_reason",
        "service", "service_raw", "severity", "severity_raw",
        "host", "pid", "message",
        "noise_reason", "cluster_id",
    ]
    for c in string_cols:
        if c in df.columns:
            df[c] = df[c].astype("string")
    bool_cols = ["parsed_ok", "timestamp_parsed_ok", "timestamp_is_range",
                 "is_noise_candidate", "burst_flag", "is_continuation"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)
    if "line_no" in df.columns:
        df["line_no"] = pd.to_numeric(df["line_no"], errors="coerce").fillna(-1).astype("int64")
    if "burst_count" in df.columns:
        df["burst_count"] = pd.to_numeric(df["burst_count"], errors="coerce").astype("Int64")
    if "repeat_count" in df.columns:
        df["repeat_count"] = pd.to_numeric(df["repeat_count"], errors="coerce").fillna(1).astype("Int64")
    for c in ["timestamp_parsed", "timestamp_end_parsed"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df


# ══════════════════════════════════════════════════════════════════════
# stage1_parse — main streaming iterator (from notebook)
# ══════════════════════════════════════════════════════════════════════

def stage1_parse(
    input_path,
    *,
    batch_size=10000,
    encoding="utf-8",
    errors="replace",
    default_tz="UTC",
    default_year=None,
    default_date=None,
    allow_relaxed=True,
    logger=None,
):
    path = Path(input_path)
    if logger is None:
        logger = logging.getLogger("stage1_parser")
    try:
        tz = ZoneInfo(default_tz)
    except Exception:
        logger.warning(
            "stage1: ZoneInfo(%r) failed — tzdata may not be installed. "
            "Falling back to UTC. Install the 'tzdata' package to use "
            "non-UTC timezones in production.",
            default_tz,
        )
        tz = timezone.utc
    if default_year is None:
        default_year = datetime.now(tz=tz).year

    detected_enc = _detect_encoding(path, fallback=encoding)
    active_encoding = detected_enc if encoding == "utf-8" else encoding

    stats = ParseStats()
    stats.detected_encoding = active_encoding

    if default_date is None:
        inferred = _infer_default_date_from_file(
            path, encoding=active_encoding, errors=errors,
            assumed_tz=tz, default_year=default_year,
        )
        if inferred is not None:
            default_date = inferred
        else:
            default_date = datetime.now(tz=tz).date()
            logger.warning(
                "stage1: could not infer log date from first 500 lines — "
                "using today (%s) as default_date. F4 bracket-short timestamps "
                "may be stamped with the wrong date. Install chardet or ensure "
                "the file contains at least one fully-dated line in the first 500 rows.",
                default_date,
            )
    else:
        # default_date was supplied by the caller — skip inference entirely.
        inferred = None

    stats.default_date_inferred = inferred is not None

    stats.format_probe = _probe_format(
        path, encoding=active_encoding, errors=errors, probe_lines=200
    )
    if stats.format_probe:
        dominant = max(stats.format_probe, key=stats.format_probe.get)
        logger.debug(
            "Format probe: dominant=%s (%.0f%%), full=%s",
            dominant, stats.format_probe[dominant] * 100, stats.format_probe,
        )

    def _iter():
        rows = []
        last_error_sev   = "INFO"
        last_pm2_service = "app"
        _dedup_key: Optional[tuple] = None
        _dedup_count: int = 0
        _dedup_rec: Optional[dict] = None

        def _flush_dedup():
            nonlocal _dedup_key, _dedup_count, _dedup_rec
            _dedup_key   = None
            _dedup_count = 0
            _dedup_rec   = None

        with path.open("r", encoding=active_encoding, errors=errors) as f:
            for i, raw in enumerate(f, start=1):
                stats.total_lines += 1
                rec = parse_line(
                    raw, i,
                    assumed_tz=tz, default_year=default_year,
                    default_date=default_date, allow_relaxed=allow_relaxed,
                    last_error_sev=last_error_sev,
                    last_pm2_service=last_pm2_service,
                )
                fmt = str(rec.get("format_type") or "unknown")
                stats.format_counts[fmt] += 1
                if rec.get("parsed_ok"):          stats.parsed_ok    += 1
                if fmt == "unknown":               stats.unknown      += 1
                if fmt == "noise":
                    stats.noise += 1
                    if rec.get("noise_reason") in ("binary_noise", "pm2_pdf_blob"):
                        stats.quarantined += 1
                if not rec.get("parsed_ok") and fmt != "noise":
                    stats.parse_failed += 1
                if fmt == "f5_json":               stats.json_ok      += 1
                if rec.get("timestamp_parsed_ok"): stats.ts_parsed_ok += 1
                elif rec.get("timestamp_raw") is not None: stats.ts_failed += 1
                if rec.get("burst_flag"):          stats.burst_lines  += 1
                if rec.get("parse_error_reason"):
                    stats.error_reasons[str(rec["parse_error_reason"])] += 1

                if fmt not in ("noise",) and not rec.get("is_continuation", False):
                    sev = rec.get("severity")
                    svc = rec.get("service")
                    if sev:
                        last_error_sev = sev
                    # S1-7 FIX: update last_pm2_service whenever we get a
                    # non-'app' service from any PM2 sub-type, so continuation
                    # lines (stack traces, object dumps) inherit the correct
                    # service even when the preceding line was a plain message
                    # whose service was inferred as something specific.
                    if svc and fmt.startswith("f17") and svc != "app":
                        last_pm2_service = svc
                    elif svc and fmt in ("f17b_pm2_bracket",):
                        # Bracket lines always carry a reliable service — update
                        # unconditionally so the next continuation inherits it.
                        last_pm2_service = svc

                if fmt == "noise" or rec.get("is_continuation", False):
                    _flush_dedup()
                    rows.append(rec)
                else:
                    msg = rec.get("message") or ""
                    svc = rec.get("service") or ""
                    key = (svc, msg)
                    if key == _dedup_key and msg:
                        _dedup_count += 1
                        if _dedup_rec is not None:
                            _dedup_rec["repeat_count"] = _dedup_count
                        rec["repeat_count"]   = _dedup_count
                        rec["noise_reason"]   = "consecutive_duplicate"
                        rec["is_noise_candidate"] = True
                        rows.append(rec)
                    else:
                        _flush_dedup()
                        _dedup_key   = key
                        _dedup_count = 1
                        _dedup_rec   = rec
                        rows.append(rec)

                # S2.5 — Memory safety valve.
                # If someone changes batch_size to a very large value in
                # config, the rows list can balloon to tens of MB before
                # a single yield.  Flush early if the estimated footprint
                # of the accumulated rows exceeds 50 MB regardless of
                # batch_size.  This is a safety valve only — normal runs
                # at batch_size=10_000 will never trigger it.
                # Heuristic: 3000 bytes per row (conservative estimate
                # accounting for dicts with ~25 fields and variable-length
                # message strings; the original 1000-byte figure was ~3x
                # too low for logs with long messages or stack traces).
                if len(rows) >= batch_size or len(rows) * 3000 > 50_000_000:
                    yield _finalize_chunk(pd.DataFrame.from_records(rows))
                    rows.clear()
                    # _dedup_rec pointed to a dict that has now been serialised
                    # into the emitted DataFrame — writing repeat_count to it
                    # further would be a no-op.  Null it out so we stop trying,
                    # but preserve _dedup_key and _dedup_count so that the next
                    # line is still correctly identified as a duplicate if it
                    # matches.  The first record of the run already has whatever
                    # repeat_count it had at emit time; that is acceptable given
                    # that cross-boundary dedup runs are rare at batch_size=10000.
                    _dedup_rec = None

            _flush_dedup()
        if rows:
            yield _finalize_chunk(pd.DataFrame.from_records(rows))
            rows.clear()

    return _iter(), stats


# ══════════════════════════════════════════════════════════════════════
# Public entry point used by pipeline.py
# ══════════════════════════════════════════════════════════════════════

def run_stage1(input_path, **kwargs):
    """
    Run Stage 1 ingestion and format detection.

    Returns:
        (chunk_iterator, ParseStats)

    Example:
        chunk_iter, stats = run_stage1("/tmp/app.log")
        df = pd.concat(list(chunk_iter), ignore_index=True)
        print(stats.as_dict())
    """
    return stage1_parse(input_path, **kwargs)