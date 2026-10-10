"""Unit tests for the offline parts of risk_pipeline (no network calls)."""

import os
import unittest
from unittest import mock

import risk_pipeline as rp

BODY = "\n".join(f"Risk paragraph {i} about supply chains and export controls." for i in range(40))
FILING_TEXT = (
    "Table of Contents\nItem 1A. Risk Factors 12\nItem 1B. Unresolved Staff Comments 30\n"
    "Item 1. Business\nWe make things. See Item 1A. Risk Factors for more.\n"
    f"ITEM 1A. RISK FACTORS\n{BODY}\nItem 1B. Unresolved Staff Comments\nNone.\n"
)


def filing(form, filed, text):
    return rp.Filing(form, filed, filed, f"acc-{filed}", "https://example.test", risk_text=text,
                     section_found=True, chunks=rp.chunk_text(text))


class TextExtractionTests(unittest.TestCase):
    def test_item_1a_takes_body_not_table_of_contents(self):
        section = rp.extract_item_1a(FILING_TEXT)
        self.assertTrue(section.startswith("ITEM 1A. RISK FACTORS"))
        self.assertIn("Risk paragraph 39", section)
        self.assertNotIn("Unresolved Staff Comments", section)

    def test_short_no_material_changes_body_is_a_section(self):
        text = ("Item 1A. Risk Factors\n28\nItem 2. Unregistered Sales\n28\n"
                "Item 1A. Risk Factors\nThere have been no material changes from the risk "
                "factors disclosed in our Annual Report on Form 10-K.\nItem 2. Unregistered Sales\n")
        self.assertIn("no material changes", rp.extract_item_1a(text))

    def test_item_1a_missing_returns_none(self):
        self.assertIsNone(rp.extract_item_1a("Item 1. Business\nNothing else here."))

    def test_html_to_text_drops_hidden_xbrl_and_splits_blocks(self):
        html = (
            "<html><head><title>x</title></head><body><ix:header>hidden facts</ix:header>"
            "<div>Item&nbsp;1A.</div><table><tr><td>Risk</td><td>Factors</td></tr></table>"
            "<p>Body   text</p><script>var a;</script></body></html>"
        )
        self.assertEqual(rp.html_to_text(html), "Item 1A.\nRisk Factors\nBody text")

    def test_select_excerpts_returns_whole_section_when_it_fits(self):
        chunks = ["alpha beta", "gamma delta"]
        self.assertEqual(rp.select_excerpts(chunks, "anything", 1000), "alpha beta\n\ngamma delta")

    def test_select_excerpts_ranks_by_relevance_within_budget_in_document_order(self):
        chunks = [
            "Tariffs and trade policy raise our costs.",
            "Our employees are important to our culture.",
            "Export controls restrict semiconductor sales to China.",
            "New export licensing rules for China may tighten further.",
        ]
        excerpt = rp.select_excerpts(chunks, "export controls China semiconductor", 120)
        self.assertEqual(excerpt, f"{chunks[2]}\n\n{chunks[3]}")


class FilingTests(unittest.TestCase):
    def test_update_only_10q_is_not_full_even_when_long(self):
        long_update = "Item 1A. Risk Factors\nOther than the risk factors listed below, there have been no material changes.\n" + "x" * 30_000
        self.assertFalse(filing("10-Q", "2026-05-01", long_update).is_full)
        self.assertTrue(filing("10-Q", "2026-05-01", "Item 1A. Risk Factors\n" + "x" * 30_000).is_full)
        self.assertTrue(filing("10-K", "2026-02-01", "Item 1A. Risk Factors\n" + "x" * 30_000).is_full)

    def test_short_10k_is_full_but_unfound_10q_is_not(self):
        short_10k = filing("10-K", "2017-02-01", "Item 1A. Risk Factors\n" + "x" * 5_000)
        short_10k.section_found = True
        self.assertTrue(short_10k.is_full)
        fallback_10q = filing("10-Q", "2017-05-01", "y" * 90_000)
        fallback_10q.section_found = False
        self.assertFalse(fallback_10q.is_full)

    def test_plan_compares_against_last_full_section(self):
        full = "Item 1A. Risk Factors\n" + "x" * 30_000
        k1, q1, q2, k2, q3 = (
            filing("10-K", "2025-02-01", full),
            filing("10-Q", "2025-05-01", "No material changes."),
            filing("10-Q", "2025-08-01", "No material changes."),
            filing("10-K", "2026-02-01", full),
            filing("10-Q", "2026-05-01", "No material changes."),
        )
        plan = rp.build_assessment_plan([k1, q1, q2, k2, q3])
        self.assertEqual([prior for _, prior in plan], [None, k1, k1, k1, k2])


class RiskRegisterTests(unittest.TestCase):
    RISK = {"risk_id": "R01", "category": "regulatory", "title": "Export controls",
            "description": "US export controls limit China sales."}

    def test_duplicate_ids_are_renumbered(self):
        risks = rp.validate_risks([dict(self.RISK), dict(self.RISK, title="Other")])
        self.assertEqual([r["risk_id"] for r in risks], ["R01", "R02"])

    def test_invalid_register_is_rejected(self):
        for bad in ([], [dict(self.RISK, category="strategic")], [{"risk_id": "R01"}]):
            with self.assertRaises(rp.PipelineError):
                rp.validate_risks(bad)

    def test_assess_risk_builds_request_and_flattens_answers(self):
        current = filing("10-Q", "2026-05-01", "Export controls tightened on China sales.")
        prior = filing("10-K", "2026-02-01", "Export controls may affect China sales.")
        response = {
            "model": "typesafe/jev-1.13-20260917",
            "usage": {"input_tokens": 500, "output_tokens": 70, "cost": 0.00002},
            "answers": {
                "trajectory": {"type": "choice", "choice": "increased", "confidence": 0.8,
                               "probabilities": {"increased": 0.9, "unchanged": 0.1,
                                                 "decreased": 0, "resolved": 0}},
                "materiality": {"type": "score", "score": 1.7, "confidence": 0.6,
                                "probabilities": {"0": 0, "1": 0.3, "2": 0.7}},
            },
        }
        with mock.patch.object(rp, "openrouter_post", return_value=response) as post:
            row = rp.assess_risk("key", rp.DECISION_MODEL, self.RISK, current, prior, 1000)
        payload = post.call_args.args[2]
        self.assertEqual(set(payload["questions"]), {"trajectory", "materiality"})
        self.assertEqual(payload["state"]["prior_filing"]["filing_date"], "2026-02-01")
        self.assertEqual((row["trajectory"], row["p_increased"], row["compared_to"]),
                         ("increased", 0.9, "2026-02-01"))
        self.assertEqual((row["materiality_score"], row["p_materiality_2"]), (1.7, 0.7))

        del response["answers"]["trajectory"]
        with mock.patch.object(rp, "openrouter_post", return_value=response) as post:
            row = rp.assess_risk("key", rp.DECISION_MODEL, self.RISK, current, None, 1000)
        self.assertEqual(set(post.call_args.args[2]["questions"]), {"materiality"})
        self.assertEqual((row["trajectory"], row["p_increased"]), ("baseline", None))


class SecretTests(unittest.TestCase):
    def test_op_reference_is_resolved_with_op_read(self):
        completed = mock.Mock(stdout="sk-or-secret\n")
        with mock.patch.dict(os.environ, {"K": "op://Vault/Item/credential"}), \
                mock.patch.object(rp.subprocess, "run", return_value=completed) as run:
            self.assertEqual(rp.get_secret("K"), "sk-or-secret")
        self.assertEqual(run.call_args.args[0], ["op", "read", "op://Vault/Item/credential"])

    def test_missing_secret_raises(self):
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(rp.PipelineError):
            rp.get_secret("K")


if __name__ == "__main__":
    unittest.main()
