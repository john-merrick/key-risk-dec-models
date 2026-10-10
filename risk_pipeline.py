#!/usr/bin/env python3
"""Track how a company's disclosed risk factors evolve across SEC filings.

Pipeline:
  1. Ingest   - resolve a ticker to a CIK, list its 10-K / 10-Q filings from the
                SEC EDGAR Submissions API and isolate Item 1A ("Risk Factors").
  2. Baseline - Claude Opus 5.5 (OpenRouter chat completions, strict JSON schema)
                extracts a structured risk register from the oldest filing.
  3. Assess   - TypeSafe Jev (OpenRouter Decisions API) evaluates every baseline
                risk against each later filing with a typed `choice` (trajectory)
                and `score` (materiality) question.
  4. Output   - a long-format time series (one row per risk per filing) as CSV,
                plus a JSON file with the risk register and filing metadata.

Usage:
  python risk_pipeline.py NVDA --num-filings 6
  python risk_pipeline.py AAPL --start 2023-01-01 --end 2025-12-31 --forms 10-K

Environment (read from the process environment, then from ./.env):
  OPENROUTER_API_KEY  OpenRouter key, or a 1Password `op://` secret reference.
  SEC_USER_AGENT      Contact string SEC requires, e.g. "Jane Doe jane@example.com".
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

import requests

import report

log = logging.getLogger("risk_pipeline")

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

BASELINE_MODEL = "anthropic/claude-opus-5.5"
DECISION_MODEL = "typesafe/jev-1.13"

SEC_MIN_INTERVAL = 0.15  # seconds between SEC requests; SEC allows at most 10/s
CATEGORIES = ("operational", "financial", "regulatory", "macroeconomic")
TRAJECTORIES = ("increased", "decreased", "unchanged", "resolved")

# Jev's context window is 32,000 tokens for state plus questions, so each request
# carries only the passages relevant to one risk. At roughly 4 characters per
# token, two sides of this size leave headroom for the questions and JSON framing.
DEFAULT_EXCERPT_CHARS = 36_000
# A full Item 1A restates every risk factor. A 10-Q usually carries an update-only
# section instead: either a short "no material changes" note or just the factors
# that changed, introduced by wording like the patterns below. Update-only sections
# are never used as a comparison point, and silence in one is not a resolved risk.
FULL_SECTION_CHARS = 20_000
_UPDATE_ONLY_MARKER = re.compile(
    r"no material changes?|other than the risk factors?|except as (?:set forth|described)"
    r"|in addition to the risk factors|refer to part i",
    re.IGNORECASE,
)
CHUNK_CHARS = 1_200


class PipelineError(RuntimeError):
    """A failure the user can act on; reported without a traceback."""


class RetryableError(RuntimeError):
    """A transient HTTP failure (rate limit, server error, dropped connection)."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines from `path` without overriding existing variables."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.removeprefix("export ").strip(), value)


def get_secret(name: str) -> str:
    """Return env var `name`, resolving a 1Password `op://` reference via `op read`."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise PipelineError(f"{name} is not set. Add it to the environment or to .env.")
    if not value.startswith("op://"):
        return value
    try:
        result = subprocess.run(
            ["op", "read", value], capture_output=True, text=True, timeout=120, check=True
        )
    except FileNotFoundError as exc:
        raise PipelineError(
            f"{name} is a 1Password reference but the `op` CLI is not installed."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise PipelineError(f"Timed out waiting for 1Password to release {name}.") from exc
    except subprocess.CalledProcessError as exc:
        raise PipelineError(f"1Password could not resolve {name}: {exc.stderr.strip()}") from exc
    return result.stdout.strip()


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #


def _retry_after(response: requests.Response) -> float | None:
    try:
        seconds = float(response.headers.get("Retry-After", ""))
    except ValueError:
        return None
    return seconds if seconds > 0 else None


def with_retry(fn, attempts: int = 5):
    """Call `fn`, retrying `RetryableError` with exponential backoff and jitter."""
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt + 1 >= attempts:
                raise PipelineError(f"Gave up after {attempts} attempts: {exc}") from exc
            delay = exc.retry_after or min(30.0, 2.0**attempt) + random.random()
            log.warning("%s; retrying in %.1fs", exc, delay)
            time.sleep(delay)


def _is_in_flight_budget_error(body: str) -> bool:
    """True for the one transient 402: OpenRouter's in-flight spending budget."""
    try:
        metadata = json.loads(body)["error"]["metadata"]
        return metadata["limit_source"] == "openrouter_in_flight_budget"
    except (ValueError, KeyError, TypeError):
        return False


def openrouter_post(url: str, api_key: str, payload: dict, timeout: float) -> dict:
    """POST JSON to OpenRouter and return the parsed body, raising typed errors."""
    try:
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RetryableError(f"OpenRouter request failed: {exc}") from exc
    if not response.ok:
        message = f"OpenRouter {response.status_code}: {response.text[:500]}"
        transient = (
            response.status_code == 429
            or response.status_code >= 500
            or (response.status_code == 402 and _is_in_flight_budget_error(response.text))
        )
        if transient:
            raise RetryableError(message, _retry_after(response))
        raise PipelineError(message)
    try:
        body = response.json()
    except ValueError as exc:
        raise RetryableError(f"OpenRouter returned non-JSON: {response.text[:200]}") from exc
    # OpenRouter can report an upstream provider failure inside a 200 response.
    if isinstance(body, dict) and "error" in body and "choices" not in body and "answers" not in body:
        raise RetryableError(f"OpenRouter error body: {json.dumps(body['error'])[:500]}")
    return body


# --------------------------------------------------------------------------- #
# SEC EDGAR
# --------------------------------------------------------------------------- #


@dataclass
class Filing:
    """One 10-K or 10-Q, with its Item 1A text once fetched."""

    form: str
    filing_date: str
    report_date: str
    accession: str
    url: str
    risk_text: str = ""
    section_found: bool = False
    chunks: list[str] = field(default_factory=list, repr=False)

    @property
    def is_full(self) -> bool:
        if self.form.startswith("10-K"):
            return True
        if not self.section_found or len(self.risk_text) < FULL_SECTION_CHARS:
            return False
        return not _UPDATE_ONLY_MARKER.search(self.risk_text[:1500])


class EdgarClient:
    """Rate-limited SEC EDGAR client with an on-disk cache of fetched documents."""

    def __init__(self, user_agent: str, cache_dir: Path):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.cache_dir = cache_dir
        self._lock = threading.Lock()
        self._last_request = 0.0

    def _get(self, url: str) -> requests.Response:
        def attempt() -> requests.Response:
            with self._lock:
                wait = self._last_request + SEC_MIN_INTERVAL - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self._last_request = time.monotonic()
            try:
                response = self.session.get(url, timeout=60)
            except requests.RequestException as exc:
                raise RetryableError(f"SEC request failed: {exc}") from exc
            if response.status_code == 429 or response.status_code >= 500:
                raise RetryableError(f"SEC {response.status_code} for {url}", _retry_after(response))
            if response.status_code == 403:
                raise PipelineError(
                    f"SEC refused {url} (403). Set SEC_USER_AGENT to a descriptive "
                    "string with a contact email, as SEC's fair-access policy requires."
                )
            if not response.ok:
                raise PipelineError(f"SEC {response.status_code} for {url}")
            return response

        return with_retry(attempt)

    def resolve_cik(self, ticker: str) -> tuple[str, str]:
        """Return (10-digit CIK, company name) for `ticker`."""
        wanted = ticker.upper().replace(".", "-")
        for entry in self._get(TICKER_MAP_URL).json().values():
            if entry["ticker"].upper() == wanted:
                return f"{int(entry['cik_str']):010d}", entry["title"]
        raise PipelineError(f"Ticker {ticker!r} was not found in SEC's ticker map.")

    def list_filings(self, cik: str, forms: set[str], start: str | None, end: str | None) -> list[Filing]:
        """Return matching filings oldest-first, reading older pages only if needed."""
        submissions = self._get(SUBMISSIONS_URL.format(name=f"CIK{cik}.json")).json()
        filings = self._filings_from_table(cik, submissions["filings"]["recent"], forms)
        for page in submissions["filings"].get("files", []):
            # Pages are newest-first; without a start date the recent page (the
            # latest ~1,000 filings) is all a "last N filings" request needs.
            if start is None or page["filingTo"] < start:
                continue
            table = self._get(SUBMISSIONS_URL.format(name=page["name"])).json()
            filings.extend(self._filings_from_table(cik, table, forms))
        filings = [
            f
            for f in filings
            if (start is None or f.filing_date >= start) and (end is None or f.filing_date <= end)
        ]
        return sorted(filings, key=lambda f: (f.filing_date, f.accession))

    @staticmethod
    def _filings_from_table(cik: str, table: dict, forms: set[str]) -> list[Filing]:
        filings = []
        for form, filed, reported, accession, document in zip(
            table["form"],
            table["filingDate"],
            table["reportDate"],
            table["accessionNumber"],
            table["primaryDocument"],
        ):
            if form in forms and document:
                url = ARCHIVE_URL.format(
                    cik=int(cik), accession=accession.replace("-", ""), document=document
                )
                filings.append(Filing(form, filed, reported, accession, url))
        return filings

    def load_risk_section(self, filing: Filing) -> None:
        """Fetch the filing's primary document and populate its Item 1A text."""
        cache_path = self.cache_dir / f"{filing.accession}.html"
        if cache_path.is_file():
            html = cache_path.read_text(encoding="utf-8")
        else:
            html = self._get(filing.url).text
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(html, encoding="utf-8")
        text = html_to_text(html)
        section = extract_item_1a(text)
        filing.section_found = section is not None
        if section is None:
            log.warning(
                "%s %s: Item 1A heading not found; using the full document text.",
                filing.form,
                filing.filing_date,
            )
        filing.risk_text = section if section is not None else text
        filing.chunks = chunk_text(filing.risk_text)


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #

_BLOCK_TAGS = {
    "p", "div", "br", "tr", "li", "table", "ul", "ol", "hr", "section",
    "h1", "h2", "h3", "h4", "h5", "h6",
}  # fmt: skip
_SKIP_TAGS = {"script", "style", "head", "ix:header"}


class _TextExtractor(HTMLParser):
    """Flatten filing HTML to text, one line per block element."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip_depth:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    """Convert filing HTML to plain text with normalised whitespace."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    text = "".join(parser.parts).replace("\xa0", " ").replace("​", "")
    lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


_ITEM_1A_START = re.compile(r"^item\s*1a\b[\s.:\-–—]*risk\s+factors", re.IGNORECASE | re.MULTILINE)
_ITEM_1A_END = re.compile(r"^item\s*(?:1b|1c|2|3|4|5|6)\b", re.IGNORECASE | re.MULTILINE)


def extract_item_1a(text: str) -> str | None:
    """Return the Item 1A section of plain-text filing `text`, or None.

    The heading appears more than once (table of contents, page headers), so every
    occurrence is paired with the next item heading and the longest span wins.
    """
    best = ""
    for start in _ITEM_1A_START.finditer(text):
        end = _ITEM_1A_END.search(text, start.end())
        span = text[start.start() : end.start() if end else len(text)].strip()
        if len(span) > len(best):
            best = span
    # A lone heading with no body (e.g. only a table-of-contents entry) is a miss.
    # A bare table-of-contents entry is only ~25 characters; a "no material
    # changes" body in a 10-Q is longer, and still a real (update-only) section.
    return best if len(best) > 60 else None


def chunk_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Group consecutive lines into chunks of about `size` characters."""
    chunks, current, length = [], [], 0
    for line in text.split("\n"):
        if current and length + len(line) > size:
            chunks.append("\n".join(current))
            current, length = [], 0
        current.append(line)
        length += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


_STOPWORDS = frozenset(
    "the and for that with are our may could would have has from this which not its their "
    "any other such can will also been were was but these those than more into over under "
    "about including result results adversely affect affected business company risk risks".split()
)


def _tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z0-9'-]{2,}", text.lower()) if w not in _STOPWORDS]


def select_excerpts(chunks: list[str], query: str, budget: int) -> str:
    """Return the chunks most relevant to `query` (BM25), in document order.

    If the whole section fits within `budget` characters it is returned intact.
    """
    if sum(len(c) + 2 for c in chunks) <= budget:
        return "\n\n".join(chunks)
    tokenised = [_tokens(c) for c in chunks]
    doc_freq = Counter(term for doc in tokenised for term in set(doc))
    avg_len = sum(len(doc) for doc in tokenised) / len(tokenised) or 1.0
    query_terms = set(_tokens(query))
    scores = []
    for doc in tokenised:
        counts = Counter(doc)
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if tf:
                idf = math.log(1 + (len(chunks) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
                score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * len(doc) / avg_len))
        scores.append(score)
    chosen, used = [], 0
    for index in sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True):
        if scores[index] <= 0 or used + len(chunks[index]) + 2 > budget:
            continue
        chosen.append(index)
        used += len(chunks[index]) + 2
    return "\n\n".join(chunks[i] for i in sorted(chosen))


# --------------------------------------------------------------------------- #
# Step 1: baseline risk extraction (Claude Opus 5.5)
# --------------------------------------------------------------------------- #

RISK_SCHEMA = {
    "type": "object",
    "properties": {
        "risks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "risk_id": {"type": "string", "description": "Stable id such as R01."},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "title": {"type": "string", "description": "Short name of the risk."},
                    "description": {
                        "type": "string",
                        "description": "Two to four sentences on the exposure and its drivers.",
                    },
                },
                "required": ["risk_id", "category", "title", "description"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["risks"],
    "additionalProperties": False,
}

BASELINE_SYSTEM_PROMPT = """\
You are a senior equity analyst building a risk register from a company's SEC filing.

Read the Risk Factors text and extract the company's core risks. The register will be \
tracked across later filings, so each risk must be a distinct, durable exposure that a \
later filing could plausibly describe as better, worse, or gone.

- Cover the whole section: operational, financial, regulatory (including legal), and \
macroeconomic risks. Merge near-duplicate disclosures into one risk; split a paragraph \
that bundles unrelated exposures.
- Write each description in specific terms taken from the filing: the named products, \
customers, suppliers, geographies, regulations, and mechanisms of harm. Later steps \
match on this wording, so generic phrasing such as "competition may hurt results" is \
not useful.
- Ground every risk in the text provided. Leave out boilerplate that applies to any \
public company unless the filing gives it company-specific substance.
- Number the risks R01, R02, ... in the order they appear in the filing."""


def extract_baseline_risks(api_key: str, company: str, filing: Filing, model: str) -> list[dict]:
    """Ask Opus for a structured risk register from the baseline filing."""
    payload = {
        "model": model,
        "max_tokens": 16_000,
        "messages": [
            {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Company: {company}\nFiling: {filing.form} filed {filing.filing_date}\n\n"
                    f"<risk_factors>\n{filing.risk_text}\n</risk_factors>"
                ),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "risk_register", "strict": True, "schema": RISK_SCHEMA},
        },
    }
    body = with_retry(lambda: openrouter_post(CHAT_URL, api_key, payload, timeout=900))
    try:
        choice = body["choices"][0]
        risks = json.loads(choice["message"]["content"])["risks"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise PipelineError(f"Baseline model returned an unusable response: {exc}") from exc
    if choice.get("finish_reason") == "length":
        raise PipelineError("Baseline risk extraction was truncated by the output token limit.")
    return validate_risks(risks)


def validate_risks(risks: object) -> list[dict]:
    """Check the risk register against the schema and guarantee unique ids."""
    if not isinstance(risks, list) or not risks:
        raise PipelineError("Baseline model returned no risks.")
    for risk in risks:
        required = {"risk_id", "category", "title", "description"}
        if not isinstance(risk, dict) or not required <= set(risk) <= required | {"first_seen"}:
            raise PipelineError(f"Risk object does not match the schema: {risk!r}")
        if risk["category"] not in CATEGORIES:
            raise PipelineError(f"Unknown risk category {risk['category']!r}.")
    if len({risk["risk_id"] for risk in risks}) != len(risks):
        for number, risk in enumerate(risks, start=1):
            risk["risk_id"] = f"R{number:02d}"
    return risks


NEW_RISKS_SYSTEM_PROMPT = """\
You are a senior equity analyst maintaining a risk register for one company across \
successive annual reports.

You are given the current register and the Risk Factors text of a newer filing. Return \
only the risks in the filing that the register does NOT already cover.

- A risk is already covered if a register entry describes the same underlying exposure, \
even in different words or with updated detail. Do not re-list it.
- Return a new risk only when the filing describes a distinct exposure: a new product, \
market, regulation, supplier dependency, litigation, or macro condition.
- Use the same style as the register: specific, grounded in the filing's own terms, with \
a two-to-four sentence description. Use the same four categories.
- Return an empty list when nothing is new. Never pad the list."""


def extract_new_risks(
    api_key: str, company: str, filing: Filing, register: list[dict], model: str
) -> list[dict]:
    """Ask Opus which risks in `filing` are missing from `register`."""
    listing = "\n".join(f"- {r['title']}: {r['description']}" for r in register)
    payload = {
        "model": model,
        "max_tokens": 8_000,
        "messages": [
            {"role": "system", "content": NEW_RISKS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Company: {company}\nFiling: {filing.form} filed {filing.filing_date}\n\n"
                    f"<register>\n{listing}\n</register>\n\n"
                    f"<risk_factors>\n{filing.risk_text}\n</risk_factors>"
                ),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "new_risks", "strict": True, "schema": RISK_SCHEMA},
        },
    }
    body = with_retry(lambda: openrouter_post(CHAT_URL, api_key, payload, timeout=900))
    try:
        choice = body["choices"][0]
        new = json.loads(choice["message"]["content"])["risks"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise PipelineError(f"New-risk screening returned an unusable response: {exc}") from exc
    if choice.get("finish_reason") == "length":
        raise PipelineError(f"New-risk screening for {filing.filing_date} was truncated.")
    if not new:
        return []
    for number, risk in enumerate(validate_risks(new), start=len(register) + 1):
        risk["risk_id"] = f"R{number:02d}"
        risk["first_seen"] = filing.filing_date
    return new


# --------------------------------------------------------------------------- #
# Step 2: longitudinal assessment (TypeSafe Jev, Decisions API)
# --------------------------------------------------------------------------- #

MATERIALITY_QUESTION = {
    "type": "score",
    "instructions": (
        "How material is the risk described in `risk` to the company as of `current_filing`? "
        "Judge from `current_filing.excerpts`. When `current_filing.section_is_full` is false "
        "the filing only lists changes, so the discussion in `prior_filing.excerpts` remains "
        "in force and counts as current disclosure unless the current filing revises it."
    ),
    "criteria": [
        "Low / mentioned: the risk is absent from the disclosure in force, or appears only "
        "as generic or boilerplate language with no company-specific detail.",
        "Moderate / active concern: the filing describes specific exposure, drivers, or "
        "mitigation, without reporting that the risk has materialised.",
        "Severe / critical focus: the filing reports that the risk has materialised or is "
        "imminent, quantifies significant impact, or gives it prominent, urgent emphasis.",
    ],
}

TRAJECTORY_QUESTION = {
    "type": "choice",
    "instructions": (
        "Compare how `current_filing.excerpts` discusses the risk described in `risk` with "
        "how `prior_filing.excerpts` discussed it. How has the company's disclosed exposure "
        "to this risk changed between the two filings?"
    ),
    "criteria": {
        "increased": (
            "The current filing adds new adverse developments, stronger or more specific "
            "warnings, larger quantified exposure, or reports that the risk has materialised."
        ),
        "decreased": (
            "The current filing softens or narrows the language, removes adverse details, "
            "or reports mitigation that reduces the exposure, while still disclosing the risk."
        ),
        "unchanged": (
            "The current filing discusses the risk in substantially the same terms, or it "
            "is an update-only section (`current_filing.section_is_full` is false) that "
            "does not mention the risk."
        ),
        "resolved": (
            "The current filing states that the underlying matter has concluded, or its "
            "full risk section (`current_filing.section_is_full` is true) no longer "
            "discloses the risk at all."
        ),
    },
}


def _filing_state(filing: Filing, query: str, budget: int) -> dict:
    return {
        "form": filing.form,
        "filing_date": filing.filing_date,
        "section_is_full": filing.is_full,
        "excerpts": select_excerpts(filing.chunks, query, budget),
    }


def assess_risk(
    api_key: str, model: str, risk: dict, filing: Filing, prior: Filing | None, budget: int
) -> dict:
    """Run one Decisions request for `risk` against `filing` and return a flat row.

    `prior` is None for the baseline filing, which gets a materiality score only.
    """
    query = f"{risk['title']} {risk['description']}"
    state = {"risk": risk, "current_filing": _filing_state(filing, query, budget)}
    questions = {"materiality": MATERIALITY_QUESTION}
    if prior is not None:
        state["prior_filing"] = _filing_state(prior, query, budget)
        questions["trajectory"] = TRAJECTORY_QUESTION
    payload = {"model": model, "state": state, "questions": questions}
    body = with_retry(lambda: openrouter_post(DECISIONS_URL, api_key, payload, timeout=120))

    try:
        answers = body["answers"]
        materiality = answers["materiality"]
        trajectory = answers["trajectory"] if prior is not None else {}
        row = {
            "risk_id": risk["risk_id"],
            "category": risk["category"],
            "title": risk["title"],
            "filing_date": filing.filing_date,
            "report_date": filing.report_date,
            "form": filing.form,
            "accession": filing.accession,
            "compared_to": prior.filing_date if prior is not None else "",
            "trajectory": trajectory["choice"] if prior is not None else "baseline",
            "trajectory_confidence": trajectory.get("confidence"),
            **{
                f"p_{name}": trajectory.get("probabilities", {}).get(name)
                for name in TRAJECTORIES
            },
            "materiality_score": float(materiality["score"]),
            "materiality_confidence": materiality.get("confidence"),
            **{
                f"p_materiality_{level}": materiality.get("probabilities", {}).get(str(level))
                for level in range(len(MATERIALITY_QUESTION["criteria"]))
            },
            "current_excerpt_chars": len(state["current_filing"]["excerpts"]),
            "decision_model": body.get("model", model),
            "cost_usd": body.get("usage", {}).get("cost"),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineError(f"Unexpected Decisions response shape: {exc}: {body!r}"[:800]) from exc
    if prior is not None and row["trajectory"] not in TRAJECTORIES:
        raise PipelineError(f"Decisions API returned unknown trajectory {row['trajectory']!r}.")
    return row


def build_assessment_plan(filings: list[Filing]) -> list[tuple[Filing, Filing | None]]:
    """Pair each filing with the filing it is compared against.

    The comparison point is the most recent earlier filing with a full risk section,
    falling back to the baseline, so an update-only 10-Q is never the reference.
    """
    plan: list[tuple[Filing, Filing | None]] = [(filings[0], None)]
    reference = filings[0]
    for filing in filings[1:]:
        plan.append((filing, reference))
        if filing.is_full:
            reference = filing
    return plan


def assess_all(
    api_key: str, model: str, risks: list[dict], filings: list[Filing], workers: int, budget: int
) -> list[dict]:
    """Assess every risk against every filing concurrently; rows keep timeline order."""
    jobs = []
    for filing, prior in build_assessment_plan(filings):
        for risk in risks:
            first_seen = risk.get("first_seen", filings[0].filing_date)
            if filing.filing_date < first_seen:
                continue
            # The filing a risk first appears in is its baseline: no prior to compare.
            jobs.append((risk, filing, None if filing.filing_date == first_seen else prior))
    done = 0
    lock = threading.Lock()

    def run(job):
        nonlocal done
        risk, filing, prior = job
        row = assess_risk(api_key, model, risk, filing, prior, budget)
        with lock:
            done += 1
            if done % 25 == 0 or done == len(jobs):
                log.info("Assessed %d/%d risk-filing pairs", done, len(jobs))
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, jobs))


# --------------------------------------------------------------------------- #
# Output and CLI
# --------------------------------------------------------------------------- #


def write_outputs(out_dir: Path, ticker: str, company: str, cik: str, risks: list[dict],
                  filings: list[Filing], rows: list[dict], models: dict) -> tuple[Path, Path]:
    """Write the time series as CSV and the full run as JSON; return both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{ticker.upper()}_{filings[0].filing_date}_{filings[-1].filing_date}"
    csv_path = out_dir / f"{stem}_risk_timeseries.csv"
    json_path = out_dir / f"{stem}_risk_timeseries.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ticker", *rows[0].keys()])
        writer.writeheader()
        writer.writerows({"ticker": ticker.upper(), **row} for row in rows)

    document = {
        "ticker": ticker.upper(),
        "company": company,
        "cik": cik,
        "models": models,
        "filings": [
            {
                "form": f.form,
                "filing_date": f.filing_date,
                "report_date": f.report_date,
                "accession": f.accession,
                "url": f.url,
                "item_1a_found": f.section_found,
                "item_1a_chars": len(f.risk_text),
                "section_is_full": f.is_full,
            }
            for f in filings
        ],
        "risks": risks,
        "timeseries": rows,
    }
    json_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return csv_path, json_path


def _iso_date(value: str) -> str:
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a YYYY-MM-DD date") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a company's risk factors and track them across SEC filings."
    )
    parser.add_argument("ticker", help="Stock ticker, e.g. NVDA")
    parser.add_argument("--start", type=_iso_date, help="Earliest filing date (YYYY-MM-DD)")
    parser.add_argument("--end", type=_iso_date, help="Latest filing date (YYYY-MM-DD)")
    parser.add_argument(
        "--num-filings",
        type=int,
        help="Keep only the N most recent matching filings (default 8 when no --start is given)",
    )
    parser.add_argument("--forms", default="10-K,10-Q", help="Comma-separated form types")
    parser.add_argument("--baseline-model", default=BASELINE_MODEL)
    parser.add_argument(
        "--no-new-risks",
        action="store_true",
        help="Track only the baseline register; skip screening later 10-Ks for new risks",
    )
    parser.add_argument(
        "--risks-json",
        type=Path,
        help="Reuse the risk register from an earlier run's JSON output instead of extracting it",
    )
    parser.add_argument("--decision-model", default=DECISION_MODEL)
    parser.add_argument("--workers", type=int, default=8, help="Concurrent Decisions requests")
    parser.add_argument(
        "--excerpt-chars",
        type=int,
        default=DEFAULT_EXCERPT_CHARS,
        help="Character budget per filing in each Decisions request",
    )
    parser.add_argument(
        "--no-report", action="store_true", help="Write only the CSV and JSON, not the HTML report"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("output"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/edgar"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.num_filings is None and args.start is None:
        args.num_filings = 8
    if args.num_filings is not None and args.num_filings < 2:
        parser.error("--num-filings must be at least 2 (a baseline plus one later filing)")
    if args.start and args.end and args.start > args.end:
        parser.error("--start must not be after --end")
    return args


def run(args: argparse.Namespace) -> None:
    load_env_file(Path(".env"))
    user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if not user_agent:
        raise PipelineError(
            'SEC_USER_AGENT is not set. SEC requires a contact string such as '
            '"Jane Doe jane@example.com"; add it to the environment or to .env.'
        )
    api_key = get_secret("OPENROUTER_API_KEY")

    edgar = EdgarClient(user_agent, args.cache_dir)
    cik, company = edgar.resolve_cik(args.ticker)
    log.info("%s -> CIK %s (%s)", args.ticker.upper(), cik, company)

    forms = {form.strip().upper() for form in args.forms.split(",") if form.strip()}
    filings = edgar.list_filings(cik, forms, args.start, args.end)
    if args.num_filings is not None:
        filings = filings[-args.num_filings :]
    if len(filings) < 2:
        raise PipelineError(
            f"Found {len(filings)} matching filing(s); need a baseline plus at least one later filing."
        )
    for filing in filings:
        edgar.load_risk_section(filing)
        log.info(
            "%s filed %s: Item 1A %s chars%s",
            filing.form,
            filing.filing_date,
            f"{len(filing.risk_text):,}",
            "" if filing.is_full else " (update-only)",
        )

    # The register has to come from a filing that restates every risk factor.
    first_full = next((i for i, filing in enumerate(filings) if filing.is_full), None)
    if first_full is None:
        log.warning(
            "No filing in range has a full risk section, so the register built from the %s "
            "filed %s may be thin. Widen the range to include a 10-K.",
            filings[0].form,
            filings[0].filing_date,
        )
    elif first_full:
        log.warning(
            "Skipping %d update-only filing(s) before the first full risk section (%s filed %s).",
            first_full,
            filings[first_full].form,
            filings[first_full].filing_date,
        )
        filings = filings[first_full:]
        if len(filings) < 2:
            raise PipelineError(
                "Only the most recent filing in range has a full risk section. Widen the range "
                "so a 10-K is followed by at least one later filing."
            )
    baseline = filings[0]
    if args.risks_json:
        try:
            risks = validate_risks(json.loads(args.risks_json.read_text(encoding="utf-8"))["risks"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise PipelineError(f"Could not read a risk register from {args.risks_json}: {exc}") from exc
        log.info("Loaded risk register from %s", args.risks_json)
    else:
        log.info("Extracting baseline risks with %s", args.baseline_model)
        risks = extract_baseline_risks(api_key, company, baseline, args.baseline_model)
        for risk in risks:
            risk["first_seen"] = baseline.filing_date
    log.info("Baseline register: %d risks", len(risks))

    if not args.no_new_risks and not args.risks_json:
        for filing in filings[1:]:
            if not (filing.is_full and filing.form.startswith("10-K")):
                continue
            new = extract_new_risks(api_key, company, filing, risks, args.baseline_model)
            log.info("%s 10-K: %d new risk(s)", filing.filing_date, len(new))
            risks.extend(new)

    rows = assess_all(
        api_key, args.decision_model, risks, filings, args.workers, args.excerpt_chars
    )
    models = {"baseline": args.baseline_model, "decision": args.decision_model}
    csv_path, json_path = write_outputs(
        args.out_dir, args.ticker, company, cik, risks, filings, rows, models
    )
    cost = sum(row["cost_usd"] or 0 for row in rows)
    log.info("Wrote %d rows to %s and %s (Decisions cost $%.4f)", len(rows), csv_path, json_path, cost)
    if args.no_report:
        return
    try:
        log.info("Wrote report %s", report.build_report(json_path))
    except ValueError as exc:
        log.warning("Report skipped: %s. Widen the date range to cover two or more 10-Ks.", exc)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        run(args)
    except PipelineError as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
