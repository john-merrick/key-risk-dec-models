# key-risk-dec-models

Tracks how a company's disclosed risk factors evolve across its SEC filings, and
produces a one-page HTML report plus the underlying time series.

1. **Ingest**: resolves a ticker to a CIK, lists its 10-K / 10-Q filings from the SEC
   EDGAR Submissions API and isolates Item 1A ("Risk Factors") from each.
2. **Baseline**: Claude Opus 5.5 (`anthropic/claude-opus-5.5`, via OpenRouter with a
   strict JSON schema) builds a risk register from the oldest full filing, then screens
   each later 10-K for risks the register is missing.
3. **Assess**: TypeSafe Jev (`typesafe/jev-1.13`, via OpenRouter's Decisions API)
   rates every risk against each filing with a `choice` question (trajectory) and a
   `score` question (materiality).
4. **Report**: writes the CSV, the JSON, and an HTML report, all in `output/`.

## Setup

```bash
pip install -r requirements.txt
```

Create `.env` (gitignored):

```bash
OPENROUTER_API_KEY="op://Dev-Secrets/OpenRouter API Key - eval-key/credential"
SEC_USER_AGENT="Your Name you@example.com"
```

`OPENROUTER_API_KEY` may be a plain key or a 1Password `op://` reference, which is
resolved with `op read` at start-up (approve the 1Password prompt when it appears).
SEC rejects requests whose `User-Agent` has no contact email.

## Run

One command runs everything and writes the report at the end:

```bash
python risk_pipeline.py POOL --start 2016-10-10      # last 10 years
python risk_pipeline.py NVDA --num-filings 12        # last 12 filings
python risk_pipeline.py AAPL --start 2024-01-01 --end 2026-06-30
python risk_pipeline.py MSFT --start 2023-07-01 --forms 10-K
```

A 10-year run on POOL takes about 4 minutes and costs under $1 in Jev scoring, plus
roughly ten Opus calls (one baseline extraction and one new-risk screen per later 10-K).

### Output

Written to `output/` (override with `--out-dir`), named by ticker and date range:

| File | What it is |
| - | - |
| `<TICKER>_risk_report.html` | The report. Self-contained, open it in a browser. |
| `<TICKER>_<start>_<end>_risk_timeseries.csv` | One row per risk per filing, including quarterlies. |
| `<TICKER>_<start>_<end>_risk_timeseries.json` | Same rows plus the risk register and filing metadata. |

The report needs at least two full 10-Ks in the run. With fewer, the pipeline logs
`Report skipped` and still writes the CSV and JSON. Pass `--no-report` to skip it.

### Re-running cheaply

The report can be rebuilt from a finished run without any API calls:

```bash
python report.py output/POOL_2017-02-24_2026-07-29_risk_timeseries.json
```

To re-score with the same risk register (skips the Opus calls and keeps `risk_id`s
stable between runs):

```bash
python risk_pipeline.py POOL --start 2016-10-10 \
  --risks-json output/POOL_2017-02-24_2026-07-29_risk_timeseries.json
```

`python risk_pipeline.py --help` lists the other options (`--workers`, `--no-new-risks`,
`--excerpt-chars`, `--baseline-model`, `--decision-model`).

## The report

Annual, one column per 10-K. It contains key takeaways, a short guide to reading the
scores, average materiality and severe-risk counts by year, year-over-year direction,
a category breakdown, the biggest risers and fallers, today's top 10 risks, a full
risk × year heatmap, the risk register, and a method-and-limits section.

## Output columns (CSV)

| Column | Meaning |
| - | - |
| `risk_id`, `category`, `title` | The tracked risk. |
| `filing_date`, `report_date`, `form`, `accession` | The filing assessed. |
| `compared_to` | Filing date the trajectory is measured against. |
| `trajectory` | `increased`, `decreased`, `unchanged`, `resolved`, or `baseline` for the first filing a risk appears in. |
| `trajectory_confidence`, `p_increased` … `p_resolved` | Jev's confidence and per-option probabilities. |
| `materiality_score` | Probability-weighted position from 0 (low / mentioned) to 2 (severe / critical focus). |
| `materiality_confidence`, `p_materiality_0` … `_2` | Jev's confidence and per-level probabilities. |
| `cost_usd` | Cost of that Decisions request. |

## How it behaves

- **Update-only 10-Qs.** Most 10-Qs do not restate every risk factor; they say "no
  material changes" or list only what changed. Such a filing is flagged
  `section_is_full: false`, is never used as a comparison point, and its silence about
  a risk reads as `unchanged`, not `resolved`. Its materiality is judged together with
  the risk factors it leaves in force. The report uses 10-Ks only for this reason.
- **What trajectory is relative to.** Each filing is compared with the most recent
  earlier filing that has a full risk section (`compared_to`), usually the last 10-K.
- **Baseline and new risks.** The register comes from the oldest filing in range with a
  full risk section; update-only filings before it are skipped with a warning. Each
  later 10-K is screened for risks not yet in the register, and a new risk is tracked
  from the filing it first appears in (`first_seen`).
- **Context limit.** Jev accepts 32,000 tokens per request, so each request carries
  only the passages most relevant to one risk (BM25 over the section), up to
  `--excerpt-chars` per filing.
- **Fallback.** If no Item 1A heading is found the whole document text is used and
  `item_1a_found` is `false` in the JSON.
- **Scale.** Most risks score between 0.5 and 1.5, so rank and change are more
  informative than the absolute number.

## Tests

```bash
python -m unittest test_risk_pipeline test_report
```
