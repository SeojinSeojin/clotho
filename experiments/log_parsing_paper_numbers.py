"""Assemble reports/paper_numbers.md, the single source of numbers for the paper.

  cd experiments && python log_parsing_paper_numbers.py [--push]

Every table is either computed here from the stored per-job JSON (main table) or copied verbatim from the
report its script generated; each section names that script. The short "Reading" lines are written by hand and
quote only numbers that appear in the tables of the same section. Regenerate after rerunning any script.
"""
import argparse
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "sharded"))
import log_parsing_phase3_final as F  # noqa: E402
import log_parsing_followups as FU  # noqa: E402

REP = EXP / "reports"
SETTING = "fair_leakfree"
MAIN = ["clotho", "MDSA", "GMMb", "SLL", "tok_prob", "tok_ent", "sem_ent", "loh_var", "out_entropy",
        "lookup_drain", "lookup_drain_default", "lookup_oracle", "hybrid", "random"]

READ = {
    "main": ("Reading: on the standard split Clotho (0.749 / 0.794) beats every other pre-generation method and every "
             "lookup; leakage-free output entropy ties or beats it (0.792 / 0.794), and the rank average of the two is "
             "the best score (0.826 / 0.845). On the template-disjoint split every lookup is at chance, Clotho and MDSA "
             "tie (0.645-0.657), and output entropy keeps 0.796."),
    "heldout": ("Reading: dropping the three tuning systems lowers Clotho by about 0.01-0.02 (0.737 / 0.777 standard) "
                "but changes no ranking or sign: Clotho still beats MDSA on every seed on the standard split "
                "(+0.042, +0.057) and Lookup (Drain) on every seed on both splits; the disjoint Clotho vs MDSA "
                "difference stays within seed noise."),
    "fairness": ("Reading: a random reference set helps the baselines only slightly, mostly at @100 on the standard "
                 "split (MDSA +0.013, Lookup (Drain) +0.017, both within about one seed std), and slightly hurts "
                 "them on the disjoint split. Clotho stays ahead of MDSA and Lookup (Drain) on every seed on the "
                 "standard split under both reference sets (smallest per-seed gap +0.014, vs MDSA with random "
                 "references @100); on the disjoint split Clotho vs MDSA remains a tie and Lookup remains at chance."),
    "cost": ("Reading: output entropy needs 4 samples to match Clotho at reference size 100 on the standard split "
             "(5 samples only tie it at 200; on the template-disjoint split 2 samples already tie it and 3 beat it by "
             "about +0.08). Generating 4 samples costs "
             "2.4x Clotho's batched HF forward pass, 8.3x a vLLM prefill (the same engine), and 36x online (batch 1). "
             "Clotho + output entropy (k=5) is the best score at about 4x Clotho's batched cost."),
    "cross": ("Reading: across systems Clotho (0.624 / 0.641, mean over pairs) and MDSA (0.634 / 0.618) are both "
              "well above chance but far below within-system (e.g. Spark -> HDFS 0.692 vs 0.873 within HDFS @200); "
              "the Clotho vs MDSA sign flips between @100 and @200. Both lookups are at chance (the pairs share at "
              "most 8 templates), so Clotho beats Lookup (Drain) by +0.12 / +0.14. Output entropy, which needs no "
              "reference set, gives 0.832."),
    "leakage": ("Reading: in the original protocol labels and post-generation metrics share one set of 10 generations. "
                "In our setting this inflates output entropy by +0.125 / +0.132 AUC (net), LOHSvar by +0.067 / +0.070 "
                "and Sem-Ent by +0.040 / +0.038; Tok-Prob/Tok-Ent move by at most +0.011 net."),
    "noise": ("Reading: near-chance within-template AUC is mostly label noise on top of a small real signal. 95-99% of "
              "the reliable variance of an input's pass rate lies between templates on 11 of 13 systems (Mac and "
              "Thunderbird trivially, with one input per template; HPC 0.80, OpenSSH 0.86), and even "
              "the 40-sample pass rate has a median within-template reliability of only 0.56, so a 5-sample "
              "within-template label is mostly sampling noise. Against the clean 40-sample label, within-template "
              "AUC rises on the same systems and test sets (Clotho 0.555 -> 0.609 @100, output entropy 0.529 -> "
              "0.584; MDSA unchanged at 0.557), against a ceiling of 0.698 for perfect knowledge of the original "
              "10-sample pass rate. So Clotho does carry some within-template signal, but it is small next to the "
              "between-template signal (overall AUC 0.832 on the same systems)."),
    "gemma": ("Reading: every Llama finding about the pre-generation methods and lookups holds for Gemma-2-9B on 12 "
              "systems. On the standard split Clotho (0.780 / 0.832) beats MDSA (+0.063 / +0.095) and Lookup (Drain) "
              "(+0.074 / +0.058) on every seed, also on the 9 held-out systems and with random reference sets for "
              "the baselines; on the template-disjoint split the lookups fall to chance and Clotho ties MDSA (+0.004 / "
              "+0.010). Leakage-free output entropy (0.821 / 0.818) beats Clotho at @100 (-0.040) but Clotho is "
              "ahead at @200 on every seed (+0.014, smallest +0.006); output entropy again needs 4 samples to match "
              "Clotho at @100, and Clotho + output entropy is the best score (0.869 / 0.890). Across systems Clotho "
              "and MDSA are again close (Clotho minus MDSA -0.025 / +0.003). On the natural samples the Drain lookup "
              "beats Clotho on every seed (-0.037 ± 0.018 @100, -0.057 ± 0.009 @200), a larger gap than for Llama, "
              "and Clotho beats output entropy by about 0.1 (+0.108 / +0.093). The other post-generation baselines stay well below Clotho on the standard split (LOHSvar 0.677 / "
              "0.672, Tok-Ent 0.578 / 0.575, Tok-Prob 0.486 / 0.480; Sem-Ent not run for Gemma)."),
    "natural": ("Reading: sampled at their natural frequency, 500 lines cover 5-120 templates per system (the suites: "
                "29-500), and at @100 the Drain group of 59-100% of test inputs (100% on 7 of 13 systems) already occurs "
                "in the reference set. Every reference-based score rises: Clotho 0.749 -> 0.831 @100 and 0.794 -> 0.845 "
                "@200, MDSA +0.10, Lookup (Drain) most (0.680 -> 0.838, 0.733 -> 0.861). Clotho's lead over Lookup "
                "(Drain) on the suites (+0.069 / +0.061) vanishes: -0.006 ± 0.014 @100 and -0.016 ± 0.014 @200 (Clotho "
                "ahead on 6 and 4 of 13 systems). Its lead over MDSA shrinks to +0.016 / +0.007. Output entropy ignores "
                "the reference set and stays at 0.789 / 0.784, so Clotho now beats it on every seed (+0.042 / +0.061; "
                "smallest per-seed gap +0.023), the reverse of the suites at @100. Clotho + output entropy is the best "
                "score at @100 (0.854, +0.016 ± 0.011 over Lookup (Drain), positive on every seed) and ties the lookup "
                "at @200 (-0.005 ± 0.007). Stage 8's 4-system result (`reports/log_parsing_natural.md`) holds on all 13 "
                "systems: on the natural distribution, a Drain lookup over labelled reference lines matches Clotho, "
                "and Clotho's value over cheap lookups lies in templates the reference set has not seen (the "
                "template-capped suites and the disjoint split)."),
}


def section(name, heading, until="## ", keep_heading=False):
    """Lines of report `name` from the line starting with `heading` up to the next line starting with `until`."""
    lines = (REP / name).read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(heading))
    stop = (lambda l: l.strip() == "") if until == "" else (lambda l: l.startswith(until))
    j = next((k for k in range(i + 1, len(lines)) if stop(lines[k])), len(lines))
    return [("#" + l) if l.startswith("## ") else l for l in lines[i if keep_heading else i + 1:j]]


def main_table():
    R, RU = F.load_all(), FU.load()
    cols = [(sp, r) for sp in F.SPLITS for r in (100, 200)]
    out = ["| method | " + " | ".join(f"{sp} @{r}" for sp, r in cols) + " |", "|---" * (1 + len(cols)) + "|"]
    names = {**F.NAMES, "out_entropy": "Output entropy*", "clotho+out_entropy": "Clotho + output entropy*"}
    for m in MAIN + ["clotho+out_entropy"]:
        cells = []
        for sp, r in cols:
            if m == "clotho+out_entropy":
                xs = FU.macro_seed(RU, sp, m, r, "auc", F.SYSTEMS)
            else:
                xs, _ = F.macro(R, SETTING, sp, m, r, "auc")
            cells.append(F.fmt(xs))
        out.append(f"| {names[m]} | " + " | ".join(cells) + " |")
    return out


def build():
    L_ = ["# Paper numbers (single source)\n",
          "Generated by `experiments/log_parsing_paper_numbers.py`. Llama-3.1-8B-Instruct, log parsing on Loghub-2.0, "
          "rules_template, strict match, 13 systems (Proxifier dropped). Main setting **fair_leakfree**: Clotho with the "
          "paper's balanced sampling, layer 16 for every hidden-state method, label from samples 5-9 (PASS if >= 3 of 5 "
          "exact matches), post-generation scores from samples 0-4 only. Unless a section says otherwise, every number "
          "is ROC-AUC (PASS = positive class), **mean ± std over 5 seeds of the per-seed macro mean over systems**. "
          "Reference size counts the 10 initial tests. Splits: standard = Clotho's protocol; disjoint = no Loghub "
          "template shared between the reference pool and the test set. * = needs the generations (post-generation). "
          "Sections 1-8 are Llama; section 9 repeats the core comparison and the natural check for Gemma-2-9B-it.\n",
          "## 1. Main table (13 systems)\n",
          "Script: `log_parsing_phase3_final.py run/post` (jobs in RESULTS_REPO `results_phase3/final/fair_leakfree/`); "
          "Clotho + output entropy from `log_parsing_followups.py`. Report: `reports/log_parsing_phase3_final.md`.\n"]
    L_ += main_table() + ["", READ["main"], ""]
    L_ += ["## 2. Held-out headline (10 systems not used for layer tuning; HDFS, Spark, Linux excluded)\n",
           "Script: `log_parsing_heldout.py` -> `reports/log_parsing_heldout.md`. Cells: held-out 10 / all 13.\n"]
    L_ += section("log_parsing_heldout.md", "| method |", until="", keep_heading=True) + [""]
    L_ += ["Paired per-seed differences on the held-out systems:\n"] + section("log_parsing_heldout.md", "## Paired")[1:]
    L_ += [READ["heldout"], ""]
    L_ += ["## 3. Baseline fairness: Clotho's balanced reference set vs a random one of the same size\n",
           "Script: `log_parsing_fairness.py` -> `reports/log_parsing_fairness.md`. All cells in a column share one test "
           "set per job: Clotho's test set minus the random reference inputs.\n"]
    L_ += section("log_parsing_fairness.md", "## AUC on T'") + section("log_parsing_fairness.md", "## Paired")
    L_ += [READ["fairness"], ""]
    L_ += ["## 4. Cost vs quality\n",
           "Scripts: `log_parsing_timing.py hf|vllm` (timings, `results_phase3/timing/`), `log_parsing_cost_quality.py` "
           "-> `reports/log_parsing_cost_quality.{md,png}`.\n"]
    L_ += section("log_parsing_cost_quality.md", "| method |", until="![", keep_heading=True) + [READ["cost"], ""]
    L_ += ["## 5. Cross-system transfer (fair_leakfree; reference set from A, test = all of B)\n",
           "Script: `log_parsing_cross_fair.py` -> `reports/log_parsing_cross_fair.md`. Mean ± std over 5 seeds per pair; "
           "the last row of each table is the mean over the 5 pairs.\n"]
    L_ += section("log_parsing_cross_fair.md", "## Reference size 100", keep_heading=True)
    L_ += section("log_parsing_cross_fair.md", "## Reference size 200", keep_heading=True, until="#####")
    L_ += [READ["cross"], ""]
    L_ += ["## 6. Leakage\n",
           "Original protocol (code reading; `reports/log_parsing_leakage_original.md`, upstream commit e8aed3d):\n"]
    L_ += section("log_parsing_leakage_original.md", "**Answer", until="Source:", keep_heading=True)
    L_ += ["Measured effect in our setting (script `log_parsing_phase3_final.py report`; leaky = label and scores share "
           "all 10 samples; `net` subtracts the shift of methods that cannot see the generations):\n"]
    L_ += section("log_parsing_phase3_final.md", "## Leakage effect")[1:] + [READ["leakage"], ""]
    L_ += ["## 7. Label noise vs within-template signal (40 extra samples per input)\n",
           "Scripts: `log_parsing_extra_samples.py` (vLLM, engine seed 1), `log_parsing_label_noise.py` -> "
           "`reports/log_parsing_label_noise.md`.\n"]
    L_ += [l for l in section("log_parsing_label_noise.md", "Median within-template", until="## ", keep_heading=True) if l]
    L_ += [""] + section("log_parsing_label_noise.md", "## 2.")[1:] + ["Same systems, noisy vs clean label:\n"]
    L_ += section("log_parsing_label_noise.md", "## 3.")[3:] + [READ["noise"], ""]
    nat = REP / "log_parsing_natural13.md"
    if nat.exists():
        L_ += ["## 8. Natural-distribution check (13 systems, 500 uniform raw lines each)\n",
               "Script: `log_parsing_natural.py` -> `reports/log_parsing_natural13.md` (per-system tables there). "
               "The 4-system version of 2026-11-08 Stage 8 is kept in `reports/log_parsing_natural.md`.\n"]
        L_ += section("log_parsing_natural13.md", "## Samples")
        L_ += [l for l in nat.read_text().splitlines() if l.startswith("Skipped")]
        L_ += section("log_parsing_natural13.md", "## ROC-AUC @100", keep_heading=True, until="Per system")
        L_ += section("log_parsing_natural13.md", "## ROC-AUC @200", keep_heading=True, until="Per system")
        L_ += [READ["natural"], ""]
    L_ += gemma_section()
    L_ += ["## Other reports\n",
           "- Qualitative examples (HDFS, Hadoop): `reports/log_parsing_qualitative.md` (`log_parsing_qualitative.py`).",
           "- Per-system tables, Drain grouping accuracy, hybrid coverage: `reports/log_parsing_phase3_final.md`.",
           "- Combinations and per-system disjoint Clotho vs MDSA: `reports/log_parsing_followups.md`.", ""]
    return L_


def paired_lines():
    """Paired per-seed differences, Clotho minus baseline, macro over all systems of the current model."""
    import statistics
    R = F.load_all()
    cols = [(sp, r) for sp in F.SPLITS for r in (100, 200)]
    out = ["| comparison | " + " | ".join(f"{sp} @{r}" for sp, r in cols) + " |", "|---" * (1 + len(cols)) + "|"]
    for b in ("MDSA", "lookup_drain", "out_entropy", "hybrid"):
        cells = []
        for sp, r in cols:
            d = [statistics.mean(F.val(R[(SETTING, sp, ds)][s], "clotho", r, "auc") - F.val(R[(SETTING, sp, ds)][s], b, r, "auc")
                                 for ds in F.SYSTEMS if "rows" in R[(SETTING, sp, ds)][s]) for s in F.SEEDS]
            cells.append(f"{statistics.mean(d):+.3f} ± {statistics.stdev(d):.3f} (min {min(d):+.3f}, max {max(d):+.3f})")
        out.append(f"| Clotho minus {F.NAMES[b]} | " + " | ".join(cells) + " |")
    return out


def gemma_block():
    import os
    import subprocess
    env = {**os.environ, "CLOTHO_MODEL": "gemma"}
    r = subprocess.run([sys.executable, __file__, "--tables_only"], env=env, capture_output=True, text=True, cwd=EXP)
    if r.returncode != 0:
        return ["(Gemma tables failed: " + r.stderr.strip().splitlines()[-1] + ")"]
    return r.stdout.rstrip("\n").splitlines()


def gemma_section():
    if not (REP / "log_parsing_phase3_final_gemma.md").exists():
        return []
    L_ = ["## 9. Second model: Gemma-2-9B-it (same protocol)\n",
          "Same rules_template prompt, strict scorer, label rule and layer-tuning procedure as for Llama; nothing was "
          "re-tuned. Generation with vLLM (prompt/token parity with `clotho/models/gemma.py` checked before every run). "
          "PASS rates and drops: `reports/log_parsing_pass_rates_gemma.md` (Apache dropped at 94.4% PASS under the "
          "> 5 of 10 rule; Proxifier not generated, as for Llama), so **12 systems**. Layer tuning on HDFS, Spark, Linux "
          "(jobs in RESULTS_REPO `results_phase3_gemma/tuning/`; mean AUC under balanced: layer 14 0.758, **21 0.773**, 28 0.765, "
          "35 0.769) chose **layer 21** of 42 (half depth, as Llama's 16 of 32). Reports: `*_gemma.md`.\n",
          "### 9.1 Main table (fair_leakfree, 12 systems)\n"]
    blk = gemma_block()
    k = blk.index("@@PAIRED@@") if "@@PAIRED@@" in blk else len(blk)
    L_ += blk[:k] + ["", "Paired per-seed differences (12 systems):\n"] + blk[k + 1:] + [""]
    L_ += ["### 9.2 Held-out (9 systems not used for layer tuning)\n"]
    L_ += section("log_parsing_heldout_gemma.md", "## Paired")[1:] + [""]
    L_ += ["### 9.3 Baseline fairness (random reference sets)\n"] + section("log_parsing_fairness_gemma.md", "## Paired")[1:]
    L_ += ["### 9.4 Quality vs number of samples (no Gemma timings)\n"]
    L_ += section("log_parsing_cost_quality_gemma.md", "| method |", until="", keep_heading=True) + [""]
    L_ += [l for l in (REP / "log_parsing_cost_quality_gemma.md").read_text().splitlines() if l.startswith("Smallest")] + [""]
    L_ += ["### 9.5 Cross-system transfer\n"]
    L_ += [l for l in (REP / "log_parsing_cross_fair_gemma.md").read_text().splitlines()
           if l.startswith("| **mean over") or l.startswith("Paired per-seed, mean")]
    L_ += ["(rows: @100, then @200; columns as in section 5)", ""]
    if (REP / "log_parsing_natural13_gemma.md").exists():
        L_ += ["### 9.6 Natural-distribution check (12 systems)\n"]
        L_ += section("log_parsing_natural13_gemma.md", "## ROC-AUC @100", keep_heading=True, until="Per system")
        L_ += section("log_parsing_natural13_gemma.md", "## ROC-AUC @200", keep_heading=True, until="Per system")
    L_ += [READ.get("gemma", ""), ""]
    return L_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--tables_only", action="store_true", help="print main table + paired lines for CLOTHO_MODEL")
    a = ap.parse_args()
    if a.tables_only:
        print("\n".join(main_table() + ["@@PAIRED@@"] + paired_lines()))
        return
    out = REP / "paper_numbers.md"
    out.write_text("\n".join(build()) + "\n")
    print(out.read_text())
    if a.push:
        F._push([out])


if __name__ == "__main__":
    main()
