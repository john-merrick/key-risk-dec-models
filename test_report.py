"""Unit tests for report.py using a small synthetic run."""

import unittest

import report


def row(risk_id, date, score, trajectory, conf=0.8, form="10-K"):
    return {"risk_id": risk_id, "filing_date": date, "form": form, "materiality_score": score,
            "trajectory": trajectory, "trajectory_confidence": conf, "cost_usd": 0.001}


DOC = {
    "ticker": "TEST", "company": "Test Co", "models": {"baseline": "m1", "decision": "m2"},
    "filings": [
        {"form": "10-K", "filing_date": "2020-02-01", "section_is_full": True},
        {"form": "10-Q", "filing_date": "2020-05-01", "section_is_full": False},
        {"form": "10-K", "filing_date": "2021-02-01", "section_is_full": True},
        {"form": "10-K", "filing_date": "2022-02-01", "section_is_full": True},
    ],
    "risks": [
        {"risk_id": "R01", "category": "regulatory", "title": "Rules", "description": "d", "first_seen": "2020-02-01"},
        {"risk_id": "R02", "category": "operational", "title": "Supply", "description": "d", "first_seen": "2020-02-01"},
        {"risk_id": "R03", "category": "operational", "title": "Cyber", "description": "d", "first_seen": "2021-02-01"},
    ],
    "timeseries": [
        row("R01", "2020-02-01", 0.6, "baseline"), row("R01", "2021-02-01", 1.2, "increased"),
        row("R01", "2022-02-01", 1.8, "increased"),
        row("R02", "2020-02-01", 1.6, "baseline"), row("R02", "2020-05-01", 1.6, "unchanged", form="10-Q"),
        row("R02", "2021-02-01", 1.0, "decreased"), row("R02", "2022-02-01", 0.2, "resolved"),
        row("R03", "2021-02-01", 0.9, "baseline"), row("R03", "2022-02-01", 1.0, "unchanged"),
    ],
}


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.run = report.Run(DOC)

    def test_only_full_10k_filings_are_annual_columns(self):
        self.assertEqual(self.run.annual_dates, ["2020-02-01", "2021-02-01", "2022-02-01"])
        self.assertNotIn(("R02", "2020-05-01"), self.run.cell)

    def test_yearly_stats_count_bands_and_new_risks(self):
        stats = report.yearly_stats(self.run)
        self.assertEqual([s["tracked"] for s in stats], [2, 3, 3])
        self.assertEqual([s["severe"] for s in stats], [1, 0, 1])
        self.assertEqual([s["new"] for s in stats], [0, 1, 0])  # first year's register is not "new"
        self.assertEqual(stats[2]["trajectory"], {"increased": 1, "resolved": 1, "unchanged": 1})

    def test_changes_skip_risks_seen_once_and_compute_delta(self):
        changes = {c["risk"]["risk_id"]: c for c in report.risk_changes(self.run)}
        self.assertAlmostEqual(changes["R01"]["delta"], 1.2)
        self.assertAlmostEqual(changes["R02"]["delta"], -1.4)
        self.assertEqual(changes["R02"]["peak_date"], "2020-02-01")
        self.assertEqual(changes["R03"]["years"], 2)

    def test_render_produces_report_with_escaped_text(self):
        doc = dict(DOC, risks=[dict(DOC["risks"][0], title="A <b>&</b> B")] + DOC["risks"][1:])
        html = report.render(report.Run(doc))
        self.assertIn("A &lt;b&gt;&amp;&lt;/b&gt; B", html)
        self.assertIn("Key takeaways", html)
        self.assertIn("▲", html)

    def test_single_annual_filing_is_rejected(self):
        doc = dict(DOC, filings=DOC["filings"][:2])
        with self.assertRaises(ValueError):
            report.Run(doc)


if __name__ == "__main__":
    unittest.main()
