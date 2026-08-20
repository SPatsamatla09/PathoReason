"""Materialise the nine pathoreason prompts from the YAML source.

The YAML is the single source of truth; this renders it so the exact strings
sent to the model are inspectable and diffable rather than assembled implicitly
at call time. Run from grid_experiment/:

    python3 render_prompts.py            # write prompts/rendered_pcam/*.txt
    python3 render_prompts.py --check    # validate only, no writes
"""

import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(ROOT, "prompts", "pathoreason_pcam.yaml")
OUT = os.path.join(ROOT, "prompts", "rendered_pcam")


def load():
    with open(SPEC) as fh:
        return yaml.safe_load(fh)


def render(spec, cond_key, para):
    """Assemble one full prompt string per the spec's `assembly.order`."""
    cond = spec["conditions"][cond_key]
    shared = spec["shared"]
    blocks = []
    for step in spec["assembly"]["order"]:
        if step == "instruction":
            blocks.append(para["instruction"])
        elif step == "output_format":
            blocks.append(cond["output_format"])
        elif step == "shared.evidence_item":
            if cond["include_evidence_item_block"]:
                blocks.append(shared["evidence_item"])
        else:
            blocks.append(shared[step.split(".", 1)[1]])
    return "\n\n".join(b.strip() for b in blocks if b and b.strip()) + "\n"


def check(spec, rendered):
    """Guard the invariants the experiment depends on."""
    errs = []
    vocab = spec["feature_vocabulary"]
    if len(vocab) != 8:
        errs.append(f"feature_vocabulary lists {len(vocab)} terms, expected 8")
    for term in vocab:
        if term not in spec["shared"]["vocabulary"]:
            errs.append(f"'{term}' is in feature_vocabulary but absent from the prose block")
    if len(spec["valid_grid_cells"]) != 16:
        errs.append(f"valid_grid_cells has {len(spec['valid_grid_cells'])} entries, expected 16")

    for cond_key, cond in spec["conditions"].items():
        # the declared key order must match the order keys appear in the literal
        pos = [(cond["output_format"].find(f'"{k}"'), k) for k in cond["key_order"]]
        if any(p < 0 for p, _ in pos):
            errs.append(f"{cond_key}: a declared key is missing from output_format")
        elif [k for _, k in sorted(pos)] != cond["key_order"]:
            errs.append(
                f"{cond_key}: output_format key order {[k for _, k in sorted(pos)]} "
                f"!= declared key_order {cond['key_order']}"
            )

    etc = spec["conditions"]["explain_then_classify"]
    if etc["key_order"][0] != "evidence":
        errs.append("explain_then_classify must emit `evidence` first or the condition is inert")
    for k in ("classify_only", "classify_then_explain"):
        if spec["conditions"][k]["key_order"][0] != "label":
            errs.append(f"{k} must emit `label` first")

    # paraphrase voices must be crossed with condition, not nested
    voices = {
        ck: [p["voice"] for p in c["paraphrases"]] for ck, c in spec["conditions"].items()
    }
    if len({tuple(v) for v in voices.values()}) != 1:
        errs.append(f"paraphrase voices are not crossed across conditions: {voices}")

    # Nothing may hint that blank image regions exist -- that would leak the
    # control. Phrases, not bare words: "empty list" is a legitimate reference
    # to the JSON field in classify_only and must not trip this.
    leaks = (
        "background", "off-slide", "white space", "whitespace", "no tissue",
        "empty region", "empty area", "empty cell", "blank region", "blank area",
        "blank cell", "contains tissue", "bare glass", "slide glass",
    )
    for name, text in rendered.items():
        low = text.lower()
        for term in leaks:
            if term in low:
                errs.append(f"{name}: leaks negative-control hint via '{term}'")

    for name, text in rendered.items():
        if "A1 is the top-left" not in text:
            errs.append(f"{name}: missing grid convention")
        for term in vocab:
            if term not in text:
                errs.append(f"{name}: missing vocabulary term '{term}'")

    return errs


def main():
    spec = load()
    rendered = {}
    for cond_key, cond in spec["conditions"].items():
        for para in cond["paraphrases"]:
            rendered[para["id"]] = render(spec, cond_key, para)

    errs = check(spec, rendered)
    for e in errs:
        print(f"FAIL {e}")
    if errs:
        sys.exit(1)
    print(f"OK  {len(rendered)} prompts, all invariants hold")

    # A frozen spec has runs recorded against it. Never overwrite; instead verify
    # the files on disk still match, so drift is an error rather than a silent
    # invalidation of every comparison in runs/.
    if spec.get("frozen"):
        drift = []
        for pid, text in rendered.items():
            path = os.path.join(OUT, f"{pid}.txt")
            if not os.path.exists(path):
                drift.append(f"{pid}: rendered file missing")
            elif open(path).read() != text:
                drift.append(f"{pid}: rendered file differs from the spec")
        for d in drift:
            print(f"FAIL {d}")
        if drift:
            print("spec is frozen: refusing to overwrite. Set frozen: false to re-render.")
            sys.exit(1)
        print(f"frozen: verified {len(rendered)} rendered prompts match the spec, no writes")
        return

    if "--check" in sys.argv:
        return

    os.makedirs(OUT, exist_ok=True)
    index = []
    for cond_key, cond in spec["conditions"].items():
        for para in cond["paraphrases"]:
            pid = para["id"]
            with open(os.path.join(OUT, f"{pid}.txt"), "w") as fh:
                fh.write(rendered[pid])
            index.append(
                {
                    "prompt_id": pid,
                    "condition": cond_key,
                    "voice": para["voice"],
                    "key_order": cond["key_order"],
                    "chars": len(rendered[pid]),
                    "file": f"prompts/rendered_pcam/{pid}.txt",
                }
            )
    with open(os.path.join(OUT, "index.json"), "w") as fh:
        json.dump(index, fh, indent=2)
    print(f"wrote {len(index)} prompts to prompts/rendered_pcam/")


if __name__ == "__main__":
    main()
