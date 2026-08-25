"""Analyze Track-A feature-specific comprehensiveness and sufficiency JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from stats_analysis import bh_adjust, bootstrap_mean_ci

ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / "runs" / "track_a_faithfulness.jsonl"
DEFAULT_OUT = ROOT / "runs" / "track_a_faithfulness"


def score_for(label, row, prefix=""):
    return float(row[f"{prefix}prob_{label.lower()}"])


def load(path):
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    table = []
    for r in rows:
        original_label = r["baseline"]["predicted_label"]
        p0 = float(r["baseline"][f"prob_{original_label.lower()}"])
        p1 = float(r[f"prob_{original_label.lower()}"])
        table.append({
            "image": r["image"], "feature": r["feature"], "feature_cells": "|".join(r["feature_cells"]),
            "variant": r["variant"], "occlusion": r["occlusion"],
            "control_index": r.get("control_index"), "true_label": r["true_label"],
            "ssa_votes_out_of_7": r["ssa_votes_out_of_7"],
            "majority_agreement": max(r["ssa_votes_out_of_7"], 7-r["ssa_votes_out_of_7"]),
            "agreement_band": "unanimous" if r["ssa_votes_out_of_7"] in (0,7) else "strong" if r["ssa_votes_out_of_7"] in (1,2,5,6) else "borderline",
            "original_label": original_label, "original_probability": p0,
            "new_label": r["predicted_label"], "new_probability_for_original": p1,
            "probability_drop": p0-p1, "label_flip": int(r["predicted_label"] != original_label),
            "control_tissue_gap_pp": r.get("control_tissue_gap_pp"), "model": r["model"],
        })
    return pd.DataFrame(table)


def image_feature_table(df):
    cited = df[df.variant == "cited_removed"].copy()
    cited = cited.rename(columns={"probability_drop":"comprehensiveness", "label_flip":"cited_flip"})
    suff = df[df.variant == "evidence_only"][["image","feature","occlusion","probability_drop","label_flip"]].rename(columns={"probability_drop":"sufficiency", "label_flip":"evidence_only_flip"})
    controls = df[df.variant == "random_control"].groupby(["image","feature","occlusion"]).agg(
        control_drop_mean=("probability_drop","mean"), control_drop_sd=("probability_drop","std"),
        control_drop_p95=("probability_drop",lambda x: float(np.quantile(x,.95))),
        control_flip_rate=("label_flip","mean"), n_controls=("control_index","count"),
    ).reset_index()
    out = cited.merge(suff,on=["image","feature","occlusion"],validate="one_to_one").merge(controls,on=["image","feature","occlusion"],validate="one_to_one")
    out["effect_over_control"] = out.comprehensiveness - out.control_drop_mean
    out["faithful_over_p95_control"] = out.comprehensiveness > out.control_drop_p95
    keep = ["image","feature","feature_cells","occlusion","true_label","ssa_votes_out_of_7","majority_agreement","agreement_band","original_label","original_probability","comprehensiveness","sufficiency","cited_flip","evidence_only_flip","control_drop_mean","control_drop_sd","control_drop_p95","control_flip_rate","n_controls","effect_over_control","faithful_over_p95_control","model"]
    return out[keep]


def summarize(table, groups):
    rows=[]
    grouper=groups[0] if len(groups)==1 else groups
    for keys,g in table.groupby(grouper,sort=True):
        keys=keys if isinstance(keys,tuple) else (keys,)
        effects=g.effect_over_control.to_numpy(float)
        lo,hi=bootstrap_mean_ci(effects)
        p=1.0 if np.allclose(effects,0) else float(wilcoxon(effects,zero_method="pratt").pvalue)
        rows.append({**dict(zip(groups,keys)),"n":len(g),"mean_comprehensiveness":g.comprehensiveness.mean(),"mean_sufficiency":g.sufficiency.mean(),"mean_effect_over_control":effects.mean(),"effect_ci_low":lo,"effect_ci_high":hi,"wilcoxon_p":p,"cited_flip_rate":g.cited_flip.mean(),"faithful_rate_over_p95_control":g.faithful_over_p95_control.mean()})
    result=pd.DataFrame(rows)
    result["wilcoxon_p_bh"]=bh_adjust(result.wilcoxon_p)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run",type=Path,default=DEFAULT_RUN)
    ap.add_argument("--out",type=Path,default=DEFAULT_OUT)
    args=ap.parse_args()
    raw=load(args.run)
    table=image_feature_table(raw)
    overall=summarize(table,["occlusion"])
    features=summarize(table,["feature","occlusion"])
    agreement=summarize(table,["agreement_band","occlusion"])
    labels=summarize(table,["original_label","occlusion"])
    args.out.mkdir(parents=True,exist_ok=True)
    raw.to_csv(args.out/"all_predictions.csv",index=False)
    table.to_csv(args.out/"image_feature_results.csv",index=False)
    overall.to_csv(args.out/"overall.csv",index=False)
    features.to_csv(args.out/"per_feature.csv",index=False)
    agreement.to_csv(args.out/"by_agreement.csv",index=False)
    labels.to_csv(args.out/"by_predicted_label.csv",index=False)
    print(overall.to_string(index=False))
    print(f"\noutputs -> {args.out}")


if __name__ == "__main__":
    main()
