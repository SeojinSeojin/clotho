# Results outline

ROC-AUC, fair_leakfree, mean over 5 seeds of the macro mean; "every seed" = paired difference of one sign on all 5 seeds. Pairs are @100 / @200. Llama-3.1-8B: 13 systems; Gemma-2-9B: 12.

**RQ1: Does Clotho work?** On the template-capped suites (Table main, standard split) Clotho gets 0.749 / 0.794 (Llama) and 0.780 / 0.832 (Gemma), beating MDSA (+0.034 / +0.057 Llama, +0.063 / +0.095 Gemma) and Lookup (Drain) (+0.069 / +0.061, +0.074 / +0.058) on every seed. Output entropy (5 samples) is higher at @100 (0.792 Llama, 0.821 Gemma). Clotho + output entropy is best (0.826 / 0.845, 0.869 / 0.890). Output entropy needs 4 samples to match Clotho at @100, costing 2.4x (Llama) and 1.5x (Gemma) Clotho's batched forward pass (Figure cost_auc).

**RQ2: Template confound.** On the template-disjoint split (Table disjoint) Clotho falls to 0.645 / 0.657 (Llama) and 0.659 / 0.671 (Gemma), level with MDSA (0.656 / 0.657, 0.655 / 0.661); Lookup (Drain) is at chance (0.505 to 0.512). Within a template (Figure within_template, Llama) Clotho scores 0.555 with the noisy label and 0.609 with a clean 40-sample label at @100 (ceiling 0.698).

**RQ3: Natural distribution.** On uniform raw-line samples (Table main) Clotho reaches 0.831 / 0.845 (Llama) and 0.880 / 0.855 (Gemma), but Lookup (Drain) rises more (0.838 / 0.861, 0.916 / 0.912). Clotho minus Lookup is -0.006 / -0.016 (Llama) and -0.037 / -0.057 (Gemma, every seed). Per system, 59 to 100% of natural test inputs have their Drain group in the reference set at @100 (Figure clotho_vs_lookup). Clotho beats output entropy (+0.042 / +0.061 Llama, +0.108 / +0.093 Gemma, every seed).

**RQ4: Leakage and post-generation baselines.** Sharing 10 generations between label and score (Table leakage) lifts output entropy from 0.792 to 0.940 (Llama) and from 0.821 to 0.901 (Gemma) at @100 (net of the pre-generation label-noise shift: +0.125 / +0.132 and +0.064 / +0.066). Leak-free, LOHSvar (0.647 / 0.644, 0.677 / 0.672), Sem-Ent (0.543 / 0.535, 0.617 / 0.612) and Tok-Prob (0.518 / 0.516, 0.486 / 0.480) stay below Clotho; on natural samples Clotho leads LOHSvar by +0.146 / +0.149 (Llama) and +0.210 / +0.193 (Gemma).

**RQ5: Cross-system.** With references from another system (Table cross), Clotho averages 0.624 / 0.641 (Llama) and 0.580 / 0.606 (Gemma) (5 pairs), MDSA 0.634 / 0.618 and 0.605 / 0.602, Lookup (Drain) 0.500 to 0.502. Within-system Clotho on the targets: 0.809 and 0.823 at @200.
