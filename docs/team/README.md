# Team guide (Phase 3)

Welcome to the Stroke Decision Copilot. This page covers what every teammate needs: access, rules and how work gets back into the project. Your own brief is next to this file.

| Person | Role | Brief |
|---|---|---|
| Parsa | Lead. Runs the long GPU runs and the paid model runs, and makes the project decisions | — |
| Lorna | Vision A: perfusion (infarct segmentation) | [lorna_vision_a.md](lorna_vision_a.md) |
| Mohamed | Vision B: haemorrhage detection | [mohamed_vision_b.md](mohamed_vision_b.md) |
| Hana | NLP A: reasoner | [hana_nlp_a.md](hana_nlp_a.md) |
| Benny | NLP B: verifier and evaluation | [benny_nlp_b.md](benny_nlp_b.md) |

Phase 3 runs over weeks 3–5. Its goal (plan §7) is to make each component good, replace hand-written test inputs with real vision outputs, and build the table that compares hallucination rates across language models. That table is the paper's main result.

## 1. Get access (once)

1. **GitHub:** accept Parsa's invitation to `parsa-hke/strokecopilot`. Then create a fine-grained token: GitHub → Settings → Developer settings → Fine-grained tokens. Give it access to that repository only, with *Contents: read and write*. In Colab, store it in Secrets as `GITHUB_TOKEN`. Never paste it into a cell.
2. **Google Drive:** in *Shared with me*, right-click `StrokeCopilot` → Organize → Add shortcut → My Drive. Notebooks expect `MyDrive/StrokeCopilot`.
3. **Notion:** you're already invited. Your tasks are on the Tasks board, and every run gets a row in Experiments.
4. **Kaggle (Mohamed only, or anyone who touches RSNA):** accept the RSNA Intracranial Hemorrhage Detection competition rules on your Kaggle account first. Then Parsa shares the private `rsna-slim-v1-*` datasets.
5. **First run:** open `notebooks/00_template.ipynb` in Colab and run it top to bottom. It should take under 5 minutes. If it fails, send Parsa the error.

## 2. Read first

- `STROKE_COPILOT_PLAN.md` §0 (summary), §2 (fixed decisions), §7 Phase 3, and the §8 section for your role.
- `docs/DATA.md` (what the data looks like) and `docs/DATA_USE.md` (what the licences allow).
- The latest entry in `CHANGELOG.md`.

## 3. Rules (these are not optional)

- **Never open, print, `cat` or attach `splits/folds.json`**, and never ask an AI tool to read it. It holds the 20 held-out cases that stay unseen until Phase 4. Code checks it by checksum and prints only group sizes. The loaders refuse held-out cases, so use them rather than reading files directly.
- **Development cases only** (129 ISLES'24 cases). Print counts and metrics. Case IDs may appear in per-case tables on Drive when your task needs them, never in chat with AI tools.
- **RSNA data stays on Kaggle.** No RSNA images, shards, indexes, or slice, patient or study IDs on Drive, in the repo, or in any output that leaves Kaggle. Model weights stay private to the team (`docs/DATA_USE.md`).
- **Secrets** (tokens, API keys) live in Colab or Kaggle Secrets, never in code, notebooks, Notion or chat.
- **No paid API calls.** All paid model calls and rented-GPU runs go through Parsa. Build and test with the fake models in the tests, and Parsa runs the paid version.
- **Guideline and trial papers** (AHA/ASA guideline, DAWN, DEFUSE-3) stay on Drive. Never paste their text into AI tools. The rulebook is written in our own words.
- **Fixed decisions** (plan §2) and the two schemas (`schemas/vision_output.json`, `schemas/recommendation.json`) don't change without Parsa.

## 4. How work gets back into the project

- **Code lives in `src/strokecopilot/`, with tests in `tests/`.** Notebooks stay thin: settings, install, call the package, print results. `pytest` must pass.
- **Branches and pull requests.** Work on a branch named `<your-name>/<topic>` (for example `lorna/error-analysis`). Open a pull request to `main`, and Parsa reviews and merges it. Don't push to `main`.
- **Every run gets a record.**
  1. Before running, add a row in Notion **Experiments** with Status *Running*. Its ID (EXP-n) is your experiment number.
  2. Save outputs on Drive under `runs/EXP-<n>_<short_name>/`, with at least `config.json` (written before the run) and `metrics.json`.
  3. Afterwards, fill in the row: value, notes, and the commit the notebook installed.
- **Report honestly.** Say what failed, what is unmeasured, and what is an estimate.
- **Questions:** comment on your Notion task, or ask Parsa directly. If a rule above blocks you, ask; don't work around it.
