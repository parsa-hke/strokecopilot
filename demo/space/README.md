---
title: Stroke Decision Copilot
emoji: 🧠
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
license: cc-by-nc-sa-4.0
short_description: Research prototype, not for clinical use. LLM stroke decisions, every claim checked.
pinned: false
---

# Stroke Decision Copilot

**Research prototype — not for clinical use.** No output of this Space is medical advice.

A university group project (MAAI 901/902, UOWD, 2026). Two vision models read stroke CT; a language model recommends
thrombolysis and thrombectomy against a rulebook written in our own words from the 2026 AHA/ASA guideline, DAWN and
DEFUSE-3; a verifier checks every claim against the imaging numbers, the triage note and the rulebook.

The page shows the pipeline, 175 precomputed cases answered by six models, the results, and a scenario builder that runs
Claude Sonnet 5.5 and the verifier live on structured inputs only, falling back to the closest precomputed case.

**Data and credit.** ISLES'24 (Zenodo 17652035, CC BY-NC-SA 4.0): Riedel et al., arXiv:2408.11142; de la Rosa et al.,
arXiv:2408.10966. Cases appear by alias only. RSNA Intracranial Haemorrhage (Kaggle): aggregate numbers only;
Flanders et al., Radiol Artif Intell 2020;2:3. This Space is shared under CC BY-NC-SA 4.0.
