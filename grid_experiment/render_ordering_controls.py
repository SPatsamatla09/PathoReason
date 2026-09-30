"""Render the ordering-control prompt conditions.

Question (Kiran): the explain-then-classify prompt (etc) shifts gemma's label
toward SSA. On the 100 abl100 tiles (Cerebras) etc vs classify-then-explain was
41 HP->SSA / 0 SSA->HP, and etc vs classify-only was 28 / 3 (SSA rate 55% ->
80%). Does that need the model's own diagnostic explanation before the label,
or does ANY text before the label shift it?

Every condition below is built from prompts/rendered/co_p1.txt (classify-only)
by replacing exactly two blocks -- the instruction paragraph and the output
format -- so the task, grid, vocabulary, confidence and reply-format blocks are
byte-identical to the baseline. All new conditions put a `preamble` field
BEFORE `label` in the output, the same position the explanation occupies in
explain-then-classify, so position is held constant and only content varies:

  ins_filler      copy a non-diagnostic filler paragraph (content: irrelevant)
  ins_checklist   copy a generic, direction-symmetric polyp checklist: HP and SSA
                  patterns each get four parallel criteria; HP-first and
                  SSA-first orderings are assigned 50/50 across tiles
  ins_copied      copy the explain-then-classify explanation of a DIFFERENT tile
                  (diagnostic, wrong tile); donors balanced 50/50 between
                  HP-labelled and SSA-labelled source explanations, restricted to
                  texts that never name SSA/SSL/sessile, so an HP-direction text
                  does not carry the SSA label as a word
  ins_self        copy THIS tile's own explain-then-classify explanation
                  (diagnostic, this tile, but copied rather than generated):
                  with ins_copied it separates "about this tile" from "written by
                  the model"
  describe_first  model writes a plain-visual description of this tile with no
                  diagnostic terms (model-generated, tile-specific, non-diagnostic)

Inserted texts are single-line (explanation items joined with "; ") so a
verbatim copy stays valid inside a JSON string.

The three copy conditions share one wrapper sentence word for word; only the
text between the markers differs. Filler and checklist are length-matched to
the median explain-then-classify explanation (52 words).

Outputs prompts/rendered_ordering_controls/:
  ins_filler.txt, describe_first.txt
  ins_checklist_hp_first.txt, ins_checklist_ssa_first.txt
  ins_copied/<image>.txt          one per recipient tile
  ins_self/<image>.txt            one per tile
  index.json                      conditions, word counts, donor manifest, checks

    python3 render_ordering_controls.py          # render + check
    python3 render_ordering_controls.py --check  # check only
"""

import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(ROOT, "prompts", "rendered", "co_p1.txt")
OUT = os.path.join(ROOT, "prompts", "rendered_ordering_controls")
TILES = os.path.join(ROOT, "runs", ".abl100_tiles.json")
SEED = 20260929

BASE_INSTRUCTION = (
    "Classify this tile as HP or SSA. Do not explain your reasoning and do\n"
    "not describe any feature you see. Return the label and your confidence\n"
    "only, with `evidence` left as an empty list."
)
BASE_OUTPUT = (
    "{\n"
    '  "label": "HP" or "SSA",\n'
    '  "confidence": number between 0.0 and 1.0,\n'
    '  "evidence": []\n'
    "}"
)
NEW_OUTPUT = (
    "{\n"
    '  "preamble": "...",\n'
    '  "label": "HP" or "SSA",\n'
    '  "confidence": number between 0.0 and 1.0,\n'
    '  "evidence": []\n'
    "}"
)

COPY_WRAPPER = (
    "Before you classify, copy the text between the markers below into the\n"
    "`preamble` field, word for word. Then classify this tile as HP or SSA.\n"
    "Do not explain your reasoning and do not describe any feature you see.\n"
    "Return the preamble, the label and your confidence, with `evidence` left\n"
    "as an empty list.\n"
    "\n"
    "<<<BEGIN TEXT>>>\n"
    "{content}\n"
    "<<<END TEXT>>>"
)

DESCRIBE_INSTRUCTION = (
    "Before you classify, describe what is visible in the image in the\n"
    "`preamble` field, in two to four sentences of plain visual language:\n"
    "colours, shapes, textures and how things are arranged. Do not use\n"
    "histological or diagnostic terms and do not reason about the diagnosis.\n"
    "Then classify this tile as HP or SSA. Do not explain your reasoning.\n"
    "Return the preamble, the label and your confidence, with `evidence` left\n"
    "as an empty list."
)

FILLER = (
    "The lighthouse on the northern headland was first lit in 1874. For decades "
    "its keepers logged wind direction, visibility and passing ships in a leather "
    "book every four hours. When the lamp was electrified in 1931 the log "
    "continued, though the entries grew shorter. The tower is still repainted "
    "every seven years."
)

CHECK_HEAD = (
    "Colorectal polyp review checklist. 1. Confirm the specimen is adequate and "
    "well oriented. 2. Assess the overall crypt architecture."
)
CHECK_HP = ("Hyperplastic pattern: serration confined to the upper crypt; straight, "
            "narrow, orderly crypt bases.")
CHECK_SSA = ("Sessile serrated pattern: serration reaching the crypt base; basal "
             "dilation; horizontal or L-shaped crypts.")
CHECK_TAIL = "Evaluate nuclear features and stroma."
CHECKLISTS = {
    "hp_first": f"{CHECK_HEAD} 3. {CHECK_HP} 4. {CHECK_SSA} 5. {CHECK_TAIL}",
    "ssa_first": f"{CHECK_HEAD} 3. {CHECK_SSA} 4. {CHECK_HP} 5. {CHECK_TAIL}",
}
HP_SIDE_TERMS = ["upper crypt", "straight", "narrow", "orderly"]
SSA_SIDE_TERMS = ["reaching the crypt base", "basal dilation", "horizontal", "l-shaped"]
LEXICAL_SSA = re.compile(r"\bSSA\b|\bSSL\b|sessile", re.I)

# Terms that would make filler non-neutral, or a visual description diagnostic.
# histological or diagnostic vocabulary the describe condition forbids
DIAGNOSTIC_TERMS = [
    "hyperplast", "serrat", "sessile", "adenoma", "lesion", "polyp", "crypt",
    "gland", "dilat", "basal", "atypi", "mitos", "mitot", "dysplas", "stroma",
    "epitheli", "inflamm", "saw-tooth", "sawtooth", r"\bboot", "muscularis",
    "lamina propria", "goblet", "nucle", r"\bhp\b", r"\bssa\b", r"\bssl\b",
    "neoplas", "malignan", "benign", "diagnos", "precancer",
]
# plain-language paraphrases of the HP/SSA criteria: allowed words, but a
# description using them is carrying diagnostic content, so they are reported
# separately (the analyzer's sensitivity analysis drops either kind)
CRITERION_PARAPHRASES = [
    r"\btooth", r"\bteeth\b", "toothed", "jagged", "zig-?zag", r"\bwiden", r"\bflared?\b",
    "sideways", r"\bl-shaped", r"\bboot-shaped", "inverted t", r"\banchor", "bulbous",
    r"\bbranch",
]
LEAKS = (
    "background", "off-slide", "white space", "whitespace", "no tissue",
    "empty region", "empty area", "empty cell", "blank region", "blank area",
    "blank cell", "contains tissue", "bare glass", "slide glass",
)


def diagnostic_hits(text):
    low = text.lower()
    return [t for t in DIAGNOSTIC_TERMS if re.search(t, low)]


def load_etc_explanations():
    """Replicate-1 explain-then-classify responses on the abl100 tiles (Cerebras gemma)."""
    out = {}
    for f in ("runs/etc_p1.jsonl", "runs/etc_p1_abl100.jsonl"):
        for line in open(os.path.join(ROOT, f)):
            r = json.loads(line)
            if r.get("replicate", 1) == 1 and (r.get("parsed") or {}).get("label") in ("HP", "SSA"):
                out[r["image"]] = r
    return out


def explanation_text(rec):
    """One line, items joined with '; ' so a verbatim copy is JSON-safe."""
    items = []
    for e in rec["parsed"]["evidence"]:
        cells = " ".join(e.get("grid_cells_valid") or [])
        items.append(f"{e['feature']} ({cells}): {e['description'].strip().rstrip('.')}")
    return "; ".join(items) + "."


def tile_meta():
    meta = {}
    for line in open(os.path.join(ROOT, "runs", "cte_p1_full.jsonl")):
        r = json.loads(line)
        if r.get("replicate", 1) == 1:
            meta[r["image"]] = {"label_true": r["label_true"], "band": r["agreement_band"]}
    return meta


def assign_checklist_variants(tiles, meta):
    """HP-first vs SSA-first checklist, alternating within true-label x band strata."""
    rng = random.Random(SEED + 1)
    strata = defaultdict(list)
    for t in sorted(tiles):
        strata[(meta[t]["label_true"], meta[t]["band"])].append(t)
    out, flip = {}, 0
    for key in sorted(strata):
        members = strata[key][:]
        rng.shuffle(members)
        for t in members:
            out[t] = "hp_first" if flip % 2 == 0 else "ssa_first"
            flip += 1
    return out


def assign_donors(tiles, etc, meta):
    """Balanced donor assignment: 50 recipients get an HP-labelled source
    explanation, 50 an SSA-labelled one, alternating within each
    (true label x agreement band) stratum; donor != recipient; donor texts never
    name SSA/SSL/sessile; HP donors reused at most 6x, SSA donors at most once."""
    rng = random.Random(SEED)
    pools = {lab: sorted(t for t in tiles if etc[t]["parsed"]["label"] == lab
                         and not LEXICAL_SSA.search(explanation_text(etc[t])))
             for lab in ("HP", "SSA")}
    for p in pools.values():
        rng.shuffle(p)
    # only 9 HP-labelled explanations avoid naming SSA; each is reused up to 6x
    # (donor-level clustering is handled in the analysis)
    cap = {"HP": 6, "SSA": 1}
    used = Counter()
    strata = defaultdict(list)
    for t in sorted(tiles):
        strata[(meta[t]["label_true"], meta[t]["band"])].append(t)
    order = []
    flip = 0
    for key in sorted(strata):
        members = strata[key][:]
        rng.shuffle(members)
        for t in members:
            order.append((t, "HP" if flip % 2 == 0 else "SSA"))
            flip += 1
    donors = {}
    for t, want in order:
        choice = None
        for d in pools[want]:
            if d != t and used[d] < cap[want]:
                choice = d
                break
        if choice is None:
            raise RuntimeError(f"no donor available for {t} ({want})")
        used[choice] += 1
        # rotate the pool so reuse is spread
        pools[want].remove(choice)
        pools[want].append(choice)
        donors[t] = {
            "donor_image": choice,
            "donor_etc_label": want,
            "donor_true_label": meta[choice]["label_true"],
        }
    return donors


def render():
    base = open(BASE).read()
    assert base.count(BASE_INSTRUCTION) == 1, "instruction anchor missing in co_p1.txt"
    assert base.count(BASE_OUTPUT) == 1, "output anchor missing in co_p1.txt"

    def build(instruction):
        return base.replace(BASE_INSTRUCTION, instruction).replace(BASE_OUTPUT, NEW_OUTPUT)

    tiles = json.load(open(TILES))
    etc = load_etc_explanations()
    missing = [t for t in tiles if t not in etc]
    assert not missing, f"no etc explanation for {missing}"
    meta = tile_meta()
    donors = assign_donors(tiles, etc, meta)

    prompts = {
        "ins_filler": build(COPY_WRAPPER.replace("{content}", FILLER)),
        "ins_checklist_hp_first": build(COPY_WRAPPER.replace("{content}", CHECKLISTS["hp_first"])),
        "ins_checklist_ssa_first": build(COPY_WRAPPER.replace("{content}", CHECKLISTS["ssa_first"])),
        "describe_first": build(DESCRIBE_INSTRUCTION),
    }
    variants = assign_checklist_variants(tiles, meta)
    copied, selfs, self_meta = {}, {}, {}
    for t in tiles:
        text = explanation_text(etc[donors[t]["donor_image"]])
        donors[t]["inserted_text"] = text
        donors[t]["inserted_words"] = len(text.split())
        donors[t]["names_ssa_lexically"] = bool(LEXICAL_SSA.search(text))
        copied[t] = build(COPY_WRAPPER.replace("{content}", text))
        own = explanation_text(etc[t])
        selfs[t] = build(COPY_WRAPPER.replace("{content}", own))
        self_meta[t] = {"own_etc_label": etc[t]["parsed"]["label"], "inserted_words": len(own.split()),
                        "names_ssa_lexically": bool(LEXICAL_SSA.search(own))}
    return base, prompts, copied, donors, tiles, etc, variants, selfs, self_meta


def check(base, prompts, copied, donors, tiles, etc, variants, selfs, self_meta):
    errs = []
    shared_base = base.replace(BASE_INSTRUCTION, "").replace(BASE_OUTPUT, "")

    def shared_part(p, instruction):
        return p.replace(instruction, "").replace(NEW_OUTPUT, "")

    instr = {
        "ins_filler": COPY_WRAPPER.replace("{content}", FILLER),
        "ins_checklist_hp_first": COPY_WRAPPER.replace("{content}", CHECKLISTS["hp_first"]),
        "ins_checklist_ssa_first": COPY_WRAPPER.replace("{content}", CHECKLISTS["ssa_first"]),
        "describe_first": DESCRIBE_INSTRUCTION,
    }
    for t, p in selfs.items():
        own = explanation_text(etc[t])
        if shared_part(p, COPY_WRAPPER.replace("{content}", own)) != shared_base:
            errs.append(f"ins_self/{t}: shared blocks differ from co_p1")
    # every inserted text is single-line (JSON-safe when copied)
    for label, text in [("filler", FILLER)] + [(f"checklist_{k}", v) for k, v in CHECKLISTS.items()] + \
            [(f"copied/{t}", d["inserted_text"]) for t, d in donors.items()] + \
            [(f"self/{t}", explanation_text(etc[t])) for t in tiles]:
        if "\n" in text or '"' in text:
            errs.append(f"{label}: inserted text has a newline or double quote")
    # checklist symmetry: same criterion count per side, same length for both orders
    for k, v in CHECKLISTS.items():
        low = v.lower()
        nh, ns = sum(low.count(x) for x in HP_SIDE_TERMS), sum(low.count(x) for x in SSA_SIDE_TERMS)
        if nh != ns:
            errs.append(f"checklist {k}: HP-side terms {nh} != SSA-side terms {ns}")
        if not 49 <= len(v.split()) <= 55:
            errs.append(f"checklist {k}: {len(v.split())} words, outside 52 +/- 3")
    if Counter(variants.values()) != Counter({"hp_first": 50, "ssa_first": 50}):
        errs.append(f"checklist variant split {dict(Counter(variants.values()))}")
    if any(d["names_ssa_lexically"] for d in donors.values()):
        errs.append("a donor text names SSA/SSL/sessile")
    for name, p in prompts.items():
        if shared_part(p, instr[name]) != shared_base:
            errs.append(f"{name}: shared blocks differ from co_p1")
    for t, p in copied.items():
        ins = COPY_WRAPPER.replace("{content}", donors[t]["inserted_text"])
        if shared_part(p, ins) != shared_base:
            errs.append(f"ins_copied/{t}: shared blocks differ from co_p1")
        if donors[t]["donor_image"] == t:
            errs.append(f"ins_copied/{t}: donor is the recipient")
    # position: preamble must come before label in every output format
    for name, p in list(prompts.items()) + [(f"ins_copied/{t}", p) for t, p in copied.items()]:
        if p.find('"preamble"') > p.find('"label"'):
            errs.append(f"{name}: preamble not before label")
        low = p.lower()
        for term in LEAKS:
            if term in low:
                errs.append(f"{name}: negative-control leak '{term}'")
    if diagnostic_hits(FILLER):
        errs.append(f"filler contains diagnostic terms: {diagnostic_hits(FILLER)}")
    bal = Counter(d["donor_etc_label"] for d in donors.values())
    if abs(bal["HP"] - bal["SSA"]) > 1:
        errs.append(f"donor label imbalance {dict(bal)}")
    reuse = Counter(d["donor_image"] for d in donors.values())
    if max(reuse.values()) > 6:
        errs.append("a donor is reused more than 6x")
    return errs


def main():
    base, prompts, copied, donors, tiles, etc, variants, selfs, self_meta = render()
    errs = check(base, prompts, copied, donors, tiles, etc, variants, selfs, self_meta)
    for e in errs:
        print("FAIL", e)
    if errs:
        sys.exit(1)
    words = {
        "etc_median_reference": 52,
        "ins_filler": len(FILLER.split()),
        "ins_checklist": len(CHECKLISTS["hp_first"].split()),
        "ins_copied_median": sorted(d["inserted_words"] for d in donors.values())[len(donors) // 2],
        "ins_self_median": sorted(m["inserted_words"] for m in self_meta.values())[len(self_meta) // 2],
    }
    print("OK  all invariants hold;", "inserted-text words:", words)
    if "--check" in sys.argv:
        return

    import shutil
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)  # prompts are pure functions of this script; rebuild cleanly
    os.makedirs(os.path.join(OUT, "ins_copied"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "ins_self"), exist_ok=True)
    for name, p in prompts.items():
        open(os.path.join(OUT, f"{name}.txt"), "w").write(p)
    for t, p in copied.items():
        open(os.path.join(OUT, "ins_copied", f"{t.replace('.png', '')}.txt"), "w").write(p)
    for t, p in selfs.items():
        open(os.path.join(OUT, "ins_self", f"{t.replace('.png', '')}.txt"), "w").write(p)
    index = {
        "seed": SEED,
        "base": "prompts/rendered/co_p1.txt",
        "tiles": tiles,
        "inserted_words": words,
        "sha256": {n: hashlib.sha256(p.encode()).hexdigest()[:16] for n, p in prompts.items()},
        "donor_balance": dict(Counter(d["donor_etc_label"] for d in donors.values())),
        "donor_reuse_max": max(Counter(d["donor_image"] for d in donors.values()).values()),
        "n_distinct_donor_texts": {lab: len({d["donor_image"] for d in donors.values() if d["donor_etc_label"] == lab})
                                   for lab in ("HP", "SSA")},
        "checklist_variants": variants,
        "ins_self": self_meta,
        "donors": donors,
    }
    json.dump(index, open(os.path.join(OUT, "index.json"), "w"), indent=1)
    print(f"wrote {len(prompts)} shared prompts + {len(copied)} ins_copied + {len(selfs)} ins_self prompts -> {OUT}")


if __name__ == "__main__":
    main()
