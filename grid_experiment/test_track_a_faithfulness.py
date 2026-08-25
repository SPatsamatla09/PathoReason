import json
import random
import tempfile
import unittest
from pathlib import Path

from grid_experiment.run_track_a_faithfulness import (
    load_features,
    tissue_matched_random_controls,
)


class TrackAFaithfulnessTests(unittest.TestCase):
    def test_controls_are_disjoint_and_same_size(self):
        tissue={f"{r}{c}":100 for r in "ABCD" for c in range(1,5)}
        cited=["A1","B2","C3"]
        controls=tissue_matched_random_controls(cited,tissue,10,random.Random(1))
        self.assertEqual(len(controls),10)
        for cells,gap in controls:
            self.assertEqual(len(cells),len(cited))
            self.assertFalse(set(cells)&set(cited))
            self.assertEqual(gap,0)

    def test_no_control_when_cited_set_exceeds_half_grid(self):
        tissue={f"{r}{c}":100 for r in "ABCD" for c in range(1,5)}
        cited=list(tissue)[:9]
        self.assertEqual(tissue_matched_random_controls(cited,tissue,5,random.Random(1)),[])

    def write_jsonl(self, records):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        with tmp:
            for record in records:
                tmp.write(json.dumps(record) + "\n")
        self.addCleanup(Path(tmp.name).unlink)
        return Path(tmp.name)

    def test_loads_original_gemma_schema(self):
        path = self.write_jsonl([{
            "image": "MHIST_old.png", "replicate": 1, "model": "gemma",
            "prompt_id": "cte_p1", "parsed": {"evidence": [{
                "feature": "Serration", "grid_cells_valid": ["B2", "A1"]
            }]},
        }])
        jobs = load_features(path)
        self.assertEqual(jobs[0]["image"], "MHIST_old.png")
        self.assertEqual(jobs[0]["cells"], ["A1", "B2"])
        self.assertEqual(jobs[0]["citation_model"], "gemma")

    def test_loads_full_gpt4o_schema(self):
        path = self.write_jsonl([{
            "image_id": "MHIST_full.png", "model": "gpt-4o", "track": "B",
            "prompt_id": "cte_p1", "seed": 42, "evidence": [{
                "feature": "Crypt/Gland Architecture",
                "grid_cells": ["D3", "C2", "not-a-cell"],
            }],
        }])
        jobs = load_features(path)
        self.assertEqual(jobs, [{
            "image": "MHIST_full.png",
            "feature": "crypt/gland architecture",
            "cells": ["C2", "D3"],
            "citation_model": "gpt-4o",
            "citation_track": "B",
            "citation_prompt_id": "cte_p1",
            "citation_seed": 42,
        }])


if __name__ == "__main__":
    unittest.main()
